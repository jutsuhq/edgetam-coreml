import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { homedir } from "node:os";
import path from "node:path";
import { isDirectory, normalizePath, expandHome } from "./util.ts";

export const UNITS = ["ane", "gpu", "all", "cpu"] as const;
export type Units = (typeof UNITS)[number];

export const DIRECTIONS = ["forward", "backward", "both"] as const;
export type Direction = (typeof DIRECTIONS)[number];

export interface Settings {
  /** The folder passed to `convert.sh`; its `out/` and `venv/` are what tracking uses. */
  workDir: string;
  /** The video to track, or "" until one is opened. */
  videoPath: string;
  /** Where `track.py` writes the masks and overlays. */
  outDir: string;
  /** `--units` for `track.py`. */
  units: Units;
  /** 0-based frame the prompt points are placed on (`--start`). */
  startFrame: number;
  /** `--direction`: forward, backward, or both from the start frame. */
  direction: Direction;
  /** Write `overlay-*.jpg` next to each mask. Off by default: the app previews masks over the video instead. */
  writeOverlays: boolean;
  /** Skip `convert.sh`'s parity check (much faster, but unverified packages). */
  noParity: boolean;
  /** Reveal the output folder in Finder when an export finishes. */
  openOnFinish: boolean;
}

export interface SettingsPatch {
  workDir?: unknown;
  videoPath?: unknown;
  outDir?: unknown;
  units?: unknown;
  startFrame?: unknown;
  direction?: unknown;
  writeOverlays?: unknown;
  noParity?: unknown;
  openOnFinish?: unknown;
}

export const DEFAULT_UNITS: Units = "ane";
export const DEFAULT_DIRECTION: Direction = "forward";

/** The work folder the README's examples produce, preferring the one this machine already has. */
function defaultWorkDir(): string {
  const candidates = [path.join(homedir(), "work", "edgetam"), path.join(homedir(), "EdgeTAM")];
  return candidates.find(isDirectory) ?? path.join(homedir(), "EdgeTAM");
}

export function defaultSettings(repoRoot: string): Settings {
  return {
    workDir: defaultWorkDir(),
    videoPath: "",
    outDir: path.join(repoRoot, "masks"),
    units: DEFAULT_UNITS,
    startFrame: 0,
    direction: DEFAULT_DIRECTION,
    writeOverlays: false,
    noParity: false,
    openOnFinish: true,
  };
}

function isUnits(value: unknown): value is Units {
  return typeof value === "string" && (UNITS as readonly string[]).includes(value);
}

function isDirection(value: unknown): value is Direction {
  return typeof value === "string" && (DIRECTIONS as readonly string[]).includes(value);
}

/**
 * Coerces whatever is in the settings file into a valid Settings. Unknown keys are dropped, bad values fall back to
 * defaults, and empty paths stay empty so the UI can keep showing a placeholder.
 */
export function normalizeSettings(input: unknown, repoRoot: string): Settings {
  const defaults = defaultSettings(repoRoot);
  if (typeof input !== "object" || input === null) return defaults;
  const raw = input as Record<string, unknown>;

  const workDir = typeof raw.workDir === "string" && raw.workDir.trim() ? expandHome(raw.workDir.trim()) : defaults.workDir;
  const videoPath = typeof raw.videoPath === "string" ? expandHome(raw.videoPath.trim()) : "";
  const outDir = typeof raw.outDir === "string" && raw.outDir.trim() ? expandHome(raw.outDir.trim()) : defaults.outDir;

  return {
    workDir,
    videoPath,
    outDir,
    units: isUnits(raw.units) ? raw.units : defaults.units,
    startFrame:
      typeof raw.startFrame === "number" && Number.isInteger(raw.startFrame) && raw.startFrame >= 0
        ? raw.startFrame
        : defaults.startFrame,
    direction: isDirection(raw.direction) ? raw.direction : defaults.direction,
    writeOverlays: typeof raw.writeOverlays === "boolean" ? raw.writeOverlays : defaults.writeOverlays,
    noParity: typeof raw.noParity === "boolean" ? raw.noParity : defaults.noParity,
    openOnFinish: typeof raw.openOnFinish === "boolean" ? raw.openOnFinish : defaults.openOnFinish,
  };
}

/** Reads and validates the patch a client sent; throws with a readable message when a path is unusable. */
export function applyPatch(current: Settings, patch: SettingsPatch): Settings {
  const next: Settings = { ...current };

  if (patch.workDir !== undefined) {
    const dir = normalizePath(String(patch.workDir));
    if (!dir) throw new Error("The work folder must be an absolute path");
    next.workDir = dir;
  }
  if (patch.videoPath !== undefined) {
    const value = String(patch.videoPath).trim();
    if (!value) next.videoPath = "";
    else {
      const file = normalizePath(value);
      if (!file) throw new Error("The video must be an absolute path");
      next.videoPath = file;
    }
  }
  if (patch.outDir !== undefined) {
    const dir = normalizePath(String(patch.outDir));
    if (!dir) throw new Error("The export folder must be an absolute path");
    next.outDir = dir;
  }
  if (patch.units !== undefined) {
    if (!isUnits(patch.units)) throw new Error("Invalid compute unit");
    next.units = patch.units;
  }
  if (patch.startFrame !== undefined) {
    const frame = Number(patch.startFrame);
    if (!Number.isInteger(frame) || frame < 0) throw new Error("Start frame must be a non-negative integer");
    next.startFrame = frame;
  }
  if (patch.direction !== undefined) {
    if (!isDirection(patch.direction)) throw new Error("Invalid direction");
    next.direction = patch.direction;
  }
  if (patch.writeOverlays !== undefined) {
    if (typeof patch.writeOverlays !== "boolean") throw new Error("Invalid overlay setting");
    next.writeOverlays = patch.writeOverlays;
  }
  if (patch.noParity !== undefined) {
    if (typeof patch.noParity !== "boolean") throw new Error("Invalid parity setting");
    next.noParity = patch.noParity;
  }
  if (patch.openOnFinish !== undefined) {
    if (typeof patch.openOnFinish !== "boolean") throw new Error("Invalid on-finish setting");
    next.openOnFinish = patch.openOnFinish;
  }
  return next;
}

export class SettingsStore {
  readonly file: string;
  private readonly repoRoot: string;
  private current: Settings;

  constructor(repoRoot: string) {
    this.repoRoot = repoRoot;
    this.file = path.join(repoRoot, ".local", "ui-settings.json");
    this.current = this.load();
  }

  get(): Settings {
    return { ...this.current };
  }

  update(patch: SettingsPatch): Settings {
    const next = applyPatch(this.current, patch);
    this.save(next);
    this.current = next;
    return this.get();
  }

  private load(): Settings {
    try {
      return normalizeSettings(JSON.parse(readFileSync(this.file, "utf8")), this.repoRoot);
    } catch {
      return defaultSettings(this.repoRoot);
    }
  }

  private save(settings: Settings): void {
    mkdirSync(path.dirname(this.file), { recursive: true });
    writeFileSync(this.file, `${JSON.stringify(settings, null, 2)}\n`, "utf8");
  }
}
