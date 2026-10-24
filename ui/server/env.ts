import path from "node:path";
import type { Settings } from "./settings.ts";
import { MODEL_NAMES, exists, freeBytes, isDirectory, isFile, which } from "./util.ts";

export interface ModelState {
  name: string;
  path: string;
  present: boolean;
}

export interface EnvironmentInfo {
  repoRoot: string;
  workDir: string;
  modelsDir: string;
  models: ModelState[];
  modelsReady: boolean;
  python: string;
  pythonReady: boolean;
  convertScript: string;
  convertReady: boolean;
  uv: string | null;
  git: string | null;
  ffmpeg: string | null;
  ffprobe: string | null;
  mediaReady: boolean;
  freeBytes: number | null;
  /** `convert.sh` refuses to start with under 2 GB free where the work folder goes. */
  enoughSpace: boolean;
  ready: boolean;
}

/** Everything the UI needs to explain, up front, why a button is disabled. */
export function probeEnvironment(repoRoot: string, settings: Settings): EnvironmentInfo {
  const modelsDir = path.join(settings.workDir, "out");
  const python = path.join(settings.workDir, "venv", "bin", "python");
  const convertScript = path.join(repoRoot, "convert.sh");
  const models = MODEL_NAMES.map((name) => ({
    name,
    path: path.join(modelsDir, `${name}.mlpackage`),
    present: isDirectory(path.join(modelsDir, `${name}.mlpackage`)),
  }));
  const modelsReady = models.every((model) => model.present);
  const pythonReady = isFile(python);
  const ffmpeg = which("ffmpeg");
  const ffprobe = which("ffprobe");
  const free = isDirectory(settings.workDir) ? freeBytes(settings.workDir) : freeBytes(path.dirname(settings.workDir));

  return {
    repoRoot,
    workDir: settings.workDir,
    modelsDir,
    models,
    modelsReady,
    python,
    pythonReady,
    convertScript,
    convertReady: isFile(convertScript) && exists(convertScript),
    uv: which("uv"),
    git: which("git"),
    ffmpeg,
    ffprobe,
    mediaReady: Boolean(process.platform === "darwin" && ffmpeg && ffprobe),
    freeBytes: free,
    enoughSpace: free === null ? true : free > 2 * 1024 * 1024 * 1024,
    ready: modelsReady && pythonReady && Boolean(ffmpeg && ffprobe),
  };
}

/** Problems that make tracking impossible, in the order the user should fix them. */
export function trackingBlockers(env: EnvironmentInfo, settings: Settings): string[] {
  const blockers: string[] = [];
  if (!isDirectory(settings.workDir)) blockers.push(`Work folder not found: ${settings.workDir}`);
  if (!env.modelsReady) blockers.push("The four Core ML packages are missing (run Initialize)");
  if (!env.pythonReady) blockers.push(`Python environment not found: ${env.python}`);
  if (!env.mediaReady) blockers.push("ffmpeg / ffprobe not found");
  if (settings.videoPath && !isFile(settings.videoPath)) blockers.push(`Video not found: ${settings.videoPath}`);
  return blockers;
}
