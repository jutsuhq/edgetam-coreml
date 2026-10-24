import { spawn, type ChildProcess } from "node:child_process";
import { createInterface } from "node:readline";
import { childEnv, killGroup } from "./util.ts";

export type JobKind = "convert" | "track";
export type JobPhase = "starting" | "extract" | "convert" | "track" | "done" | "error" | "cancelled";

export interface JobSnapshot {
  id: number;
  kind: JobKind;
  phase: JobPhase;
  label: string;
  progress: number | null;
  startedAt: number;
  endedAt: number | null;
  framesDone: number;
  framesTotal: number | null;
  error: string | null;
  /** The last lines of output when a job fails, shown inline so the cause is visible without opening the log. */
  detail: string | null;
  summary: string | null;
  outDir: string | null;
  running: boolean;
  logFrom: number;
  log: string[];
}

export interface ChildOutcome {
  code: number | null;
  signal: NodeJS.Signals | null;
  error: Error | null;
}

const MAX_LOG_LINES = 3000;
const ANSI = /\u001b\[[0-9;]*[A-Za-z]/g;

/** One run of `convert.sh` or `track.py`: its progress, its log, and how to kill it. */
export class Job {
  readonly id: number;
  readonly kind: JobKind;
  readonly startedAt = Date.now();
  phase: JobPhase = "starting";
  label = "Preparing";
  progress: number | null = null;
  endedAt: number | null = null;
  framesDone = 0;
  framesTotal: number | null = null;
  error: string | null = null;
  detail: string | null = null;
  summary: string | null = null;
  outDir: string | null = null;
  cancelled = false;

  /** Absolute line index of `log[0]`, so clients can ask for only what they have not seen. */
  private logOffset = 0;
  private readonly lines: string[] = [];
  private cancelHandler: (() => void) | null = null;
  private readonly children = new Set<ChildProcess>();

  constructor(id: number, kind: JobKind) {
    this.id = id;
    this.kind = kind;
  }

  get running(): boolean {
    return this.endedAt === null;
  }

  append(line: string): void {
    const clean = line.replace(ANSI, "").replace(/\r$/, "");
    for (const part of clean.split("\n")) {
      this.lines.push(part);
      if (this.lines.length > MAX_LOG_LINES) {
        this.lines.shift();
        this.logOffset += 1;
      }
    }
  }

  setPhase(phase: JobPhase, label: string): void {
    this.phase = phase;
    this.label = label;
  }

  setProgress(progress: number | null): void {
    this.progress = progress === null ? null : Math.max(0, Math.min(1, progress));
  }

  setFrames(done: number, total: number | null): void {
    this.framesDone = done;
    this.framesTotal = total;
  }

  onCancel(handler: () => void): void {
    this.cancelHandler = handler;
  }

  track(child: ChildProcess): void {
    this.children.add(child);
    child.once("close", () => this.children.delete(child));
  }

  cancel(): void {
    if (!this.running) return;
    this.cancelled = true;
    this.label = "Stopping";
    this.cancelHandler?.();
    const children = [...this.children];
    for (const child of children) {
      if (child.pid) killGroup(child.pid, "SIGTERM");
    }
    // Anything that ignores SIGTERM (a stuck Core ML call) is killed, so the UI can never be left locked.
    setTimeout(() => {
      for (const child of children) {
        if (child.exitCode === null && child.signalCode === null && child.pid) killGroup(child.pid, "SIGKILL");
      }
    }, 4000);
  }

  finish(phase: JobPhase, error: string | null = null, summary: string | null = null, detail: string | null = null): void {
    if (!this.running) return;
    this.phase = phase;
    this.error = error;
    this.summary = summary;
    this.detail = detail;
    this.endedAt = Date.now();
    if (phase === "done") {
      this.progress = 1;
      this.label = "Finished";
    }
    if (phase === "cancelled") this.label = "Stopped";
    if (phase === "error") this.label = "Failed";
  }

  /** The lines that explain a failure: the last traceback block if there is one, else the last few lines. */
  tail(count = 14): string {
    const lines = this.lines.filter((line) => line.trim().length > 0);
    let start = Math.max(0, lines.length - count);
    for (let index = lines.length - 1; index >= 0; index -= 1) {
      if (/^Traceback \(most recent call last\):/.test(lines[index] ?? "")) {
        start = index;
        break;
      }
    }
    return lines.slice(start).join("\n");
  }

  snapshot(from = -1): JobSnapshot {
    // Never hand back lines the buffer has already dropped: answers start at the oldest line still held.
    const start = Math.max(this.logOffset, Math.min(from, this.logOffset + this.lines.length));
    const skip = start - this.logOffset;
    return {
      id: this.id,
      kind: this.kind,
      phase: this.phase,
      label: this.label,
      progress: this.progress,
      startedAt: this.startedAt,
      endedAt: this.endedAt,
      framesDone: this.framesDone,
      framesTotal: this.framesTotal,
      error: this.error,
      detail: this.detail,
      summary: this.summary,
      outDir: this.outDir,
      running: this.running,
      logFrom: start,
      log: this.lines.slice(skip),
    };
  }
}

export class JobManager {
  private current: Job | null = null;
  private nextId = 1;

  running(): Job | null {
    return this.current?.running ? this.current : null;
  }

  /** Starts a job, refusing a second concurrent one: both jobs would fight over the same models and folder. */
  start(kind: JobKind): Job {
    if (this.running()) throw new Error("Another job is already running");
    const job = new Job(this.nextId++, kind);
    this.current = job;
    return job;
  }

  snapshot(from = -1): JobSnapshot | null {
    return this.current?.snapshot(from) ?? null;
  }

  cancel(): boolean {
    const job = this.running();
    if (!job) return false;
    job.cancel();
    return true;
  }
}

/**
 * Spawns a child in its own process group, streams both pipes into the job log line by line, and resolves only
 * once the output is drained. `quiet` can drop high-frequency progress lines (ffmpeg's `-progress`) from the log;
 * `onLine` still sees every line.
 */
export function startChild(
  job: Job,
  command: string,
  args: string[],
  options: { cwd?: string; quiet?: RegExp; onLine?: (line: string, stream: "stdout" | "stderr") => void } = {},
): { child: ChildProcess; done: Promise<ChildOutcome> } {
  job.append(`$ ${[command, ...args].map(quoteArg).join(" ")}`);
  let child: ChildProcess;
  try {
    child = spawn(command, args, {
      cwd: options.cwd,
      env: childEnv({ PYTHONUNBUFFERED: "1", PYTHONIOENCODING: "utf-8" }),
      detached: true,
      stdio: ["ignore", "pipe", "pipe"],
    });
  } catch (error) {
    job.append(String(error));
    return {
      // A throw here is not a signal failure; hand back a dead handle so callers share one code path.
      child: { pid: undefined, exitCode: null, signalCode: null } as unknown as ChildProcess,
      done: Promise.resolve({ code: null, signal: null, error: error as Error }),
    };
  }

  child.on("error", (error) => {
    job.append(`Could not start ${command}: ${error.message}`);
  });
  if (child.pid) job.track(child);

  const read = (stream: NodeJS.ReadableStream | null, name: "stdout" | "stderr") => {
    if (!stream) return Promise.resolve();
    return new Promise<void>((resolve) => {
      const reader = createInterface({ input: stream, crlfDelay: Infinity });
      reader.on("line", (line) => {
        options.onLine?.(line, name);
        if (!options.quiet?.test(line)) job.append(line);
      });
      reader.on("close", () => resolve());
    });
  };

  const done = new Promise<ChildOutcome>((resolve) => {
    let spawnError: Error | null = null;
    child.once("error", (error) => {
      spawnError = error;
    });
    // Start reading at spawn time: a stream that has already ended will not replay its buffer to a late reader.
    const stdoutDone = read(child.stdout, "stdout");
    const stderrDone = read(child.stderr, "stderr");
    child.once("close", (code, signal) => {
      void Promise.all([stdoutDone, stderrDone]).then(() => resolve({ code, signal, error: spawnError }));
    });
  });

  return { child, done };
}

function quoteArg(arg: string): string {
  return /^[A-Za-z0-9_@%+=:,./-]+$/.test(arg) ? arg : `'${arg.replace(/'/g, `'\\''`)}'`;
}
