import { spawn, spawnSync } from "node:child_process";
import { existsSync, statSync, statfsSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";

/** The four packages `examples/track.py` expects under `<work>/out`. */
export const MODEL_NAMES = [
  "EdgeTAMVideoImageEncoder",
  "EdgeTAMVideoInitializer",
  "EdgeTAMVideoMemoryEncoder",
  "EdgeTAMVideoPropagator",
] as const;

/** Directories a GUI-launched process rarely has on PATH, checked after the inherited one. */
const EXTRA_BIN_DIRS = ["/opt/homebrew/bin", "/usr/local/bin", path.join(homedir(), ".local", "bin")];

export function childEnv(extra: Record<string, string> = {}): NodeJS.ProcessEnv {
  const inherited = (process.env.PATH ?? "").split(path.delimiter).filter(Boolean);
  const dirs = [...new Set([...inherited, ...EXTRA_BIN_DIRS])];
  return { ...process.env, PATH: dirs.join(path.delimiter), ...extra };
}

/** Resolves an executable on PATH (plus Homebrew's usual locations) without a shell. */
export function which(name: string): string | null {
  const dirs = [ ...(process.env.PATH ?? "").split(path.delimiter).filter(Boolean), ...EXTRA_BIN_DIRS];
  for (const dir of dirs) {
    const candidate = path.join(dir, name);
    try {
      if (statSync(candidate).isFile()) return candidate;
    } catch {
      // Not here; keep looking.
    }
  }
  return null;
}

export function isDirectory(target: string): boolean {
  try {
    return statSync(target).isDirectory();
  } catch {
    return false;
  }
}

export function isFile(target: string): boolean {
  try {
    return statSync(target).isFile();
  } catch {
    return false;
  }
}

/** Free bytes on the volume holding `target` (walking up to the nearest existing parent). */
export function freeBytes(target: string): number | null {
  let dir = path.resolve(target);
  for (;;) {
    try {
      const fs = statfsSync(dir);
      return Number(fs.bavail) * Number(fs.bsize);
    } catch {
      const parent = path.dirname(dir);
      if (parent === dir) return null;
      dir = parent;
    }
  }
}

export function expandHome(target: string): string {
  if (target === "~") return homedir();
  if (target.startsWith("~/")) return path.join(homedir(), target.slice(2));
  return target;
}

/** An absolute, symlink-free-enough path for a user-typed string, or null when it is not absolute. */
export function normalizePath(target: string): string | null {
  const expanded = expandHome(target.trim());
  if (!expanded || !path.isAbsolute(expanded)) return null;
  return path.resolve(expanded);
}

/** True when `child` is `parent` or inside it (used to keep static serving inside dist/). */
export function isInside(parent: string, child: string): boolean {
  const rel = path.relative(path.resolve(parent), path.resolve(child));
  return rel === "" || (!rel.startsWith("..") && !path.isAbsolute(rel));
}

export interface RunResult {
  status: number | null;
  stdout: string;
  stderr: string;
}

/** Runs a short command and buffers its output; only for probes, never for jobs. */
export function run(command: string, args: string[], timeoutMs = 15_000): RunResult {
  const result = spawnSync(command, args, {
    encoding: "utf8",
    timeout: timeoutMs,
    env: childEnv(),
    maxBuffer: 16 * 1024 * 1024,
  });
  return {
    status: result.status,
    stdout: result.stdout ?? "",
    stderr: result.stderr ?? "",
  };
}

/** Kills a detached child's whole process group, so ffmpeg's children do not survive a cancel. */
export function killGroup(pid: number, signal: NodeJS.Signals = "SIGTERM"): void {
  try {
    process.kill(-pid, signal);
  } catch {
    try {
      process.kill(pid, signal);
    } catch {
      // Already gone.
    }
  }
}

/** The one-line description of a spawn failure the UI can show as-is. */
export function spawnError(command: string, error: unknown): string {
  if (error instanceof Error) {
    if ((error as NodeJS.ErrnoException).code === "ENOENT") return `${command} not found`;
    return error.message;
  }
  return String(error);
}

export function exists(target: string): boolean {
  return existsSync(target);
}

export function spawnDetached(command: string, args: string[]): void {
  const child = spawn(command, args, { detached: true, stdio: "ignore", env: childEnv() });
  child.on("error", () => {
    // Nothing useful to do: the folder is still on disk and the path is shown in the UI.
  });
  child.unref();
}
