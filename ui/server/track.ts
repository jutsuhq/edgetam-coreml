import { mkdirSync, mkdtempSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import path from "node:path";
import type { EnvironmentInfo } from "./env.ts";
import { MAX_PROMPT_POINTS, type PromptPoint } from "../shared/prompt.ts";
import type { Job } from "./jobs.ts";
import { startChild } from "./jobs.ts";
import { probeVideo } from "./media.ts";
import type { Settings } from "./settings.ts";
import { spawnDetached } from "./util.ts";

export type { PromptPoint };

export interface TrackContext {
  repoRoot: string;
  settings: Settings;
  env: EnvironmentInfo;
  points: PromptPoint[];
}

/** Guards against absurd coordinates that would only make SAM 2 return an empty mask. */
export function normalizePoints(input: unknown): PromptPoint[] {
  if (!Array.isArray(input)) throw new Error("Invalid point");
  const points: PromptPoint[] = [];
  for (const item of input) {
    if (typeof item !== "object" || item === null) throw new Error("Invalid point");
    const raw = item as Record<string, unknown>;
    const x = Number(raw.x);
    const y = Number(raw.y);
    const label = Number(raw.label);
    if (!Number.isFinite(x) || !Number.isFinite(y)) throw new Error("Invalid point coordinates");
    if (label !== 0 && label !== 1) throw new Error("Invalid point label");
    points.push({ x, y, label });
  }
  if (points.length === 0) throw new Error("Add at least one point");
  // The initializer's point_coords is [1, N, 2] with N in 1..16; more points fail inside Core ML.
  if (points.length > MAX_PROMPT_POINTS) {
    throw new Error(`At most ${MAX_PROMPT_POINTS} points (object + background)`);
  }
  if (!points.some((point) => point.label === 1)) throw new Error("Add at least one object point");
  return points;
}

/**
 * Runs the same loop as `examples/track.py`: extract the video to JPEGs, then hand the folder to the script.
 * Pre-extracting is what makes extraction progress visible; `track.py` would otherwise do it silently.
 */
export function startTracking(job: Job, context: TrackContext): void {
  const { repoRoot, settings, env, points } = context;
  void (async () => {
    let tempRoot: string | null = null;
    try {
      const meta = probeVideo(settings.videoPath);
      mkdirSync(settings.outDir, { recursive: true });
      const existing = countMasks(settings.outDir);
      if (existing > 0) job.append(`Note: ${settings.outDir} already has ${existing} masks (same indices will be overwritten)`);

      tempRoot = mkdtempSync(path.join(tmpdir(), "edgetam-ui-"));
      const framesDir = path.join(tempRoot, "frames");
      mkdirSync(framesDir, { recursive: true });

      job.outDir = settings.outDir;
      job.setPhase("extract", "Extracting frames");
      job.setProgress(0.01);
      job.setFrames(0, meta.frames);
      const total = meta.frames;

      const extract = startChild(
        job,
        env.ffmpeg ?? "ffmpeg",
        [
          "-hide_banner", "-nostats", "-loglevel", "error",
          "-i", settings.videoPath,
          "-q:v", "2",
          "-progress", "pipe:1",
          path.join(framesDir, "%05d.jpg"),
        ],
        {
          quiet: /^(frame|fps|stream_\d+_\d+_q|bitrate|total_size|out_time(_us|_ms|)|dup_frames|drop_frames|speed|progress)=/,
          onLine: (line) => {
            const match = /^frame=(\d+)$/.exec(line.trim());
            if (!match) return;
            const done = Number(match[1]);
            job.setFrames(done, total);
            if (total) job.setProgress(0.01 + 0.09 * Math.min(1, done / total));
          },
        },
      );
      const extracted = await extract.done;
      if (job.cancelled) return job.finish("cancelled");
      if (extracted.error) throw new Error(`Could not start ffmpeg: ${extracted.error.message}`);
      if (extracted.code !== 0) throw new Error(`ffmpeg failed (exit ${extracted.code ?? extracted.signal})`);

      const frameCount = readdirSync(framesDir).filter((name) => name.endsWith(".jpg")).length;
      if (frameCount === 0) throw new Error("Could not extract frames from the video");
      const start = Math.min(Math.max(settings.startFrame, 0), frameCount - 1);
      // Each direction writes the start frame once; "both" covers every frame exactly once.
      const expected =
        settings.direction === "forward" ? frameCount - start
          : settings.direction === "backward" ? start + 1
            : frameCount;
      job.setFrames(0, expected);
      job.append(`Extracted ${frameCount} frames; ${settings.direction} from frame ${start}`);

      const args = [
        path.join(repoRoot, "examples", "track.py"),
        "--models", env.modelsDir,
        "--video", framesDir,
        "--out", settings.outDir,
        "--units", settings.units,
        "--start", String(start),
        "--direction", settings.direction,
      ];
      if (!settings.writeOverlays) args.push("--no-overlay");
      for (const point of points.filter((point) => point.label === 1)) {
        args.push("--point", `${Math.round(point.x)},${Math.round(point.y)}`);
      }
      for (const point of points.filter((point) => point.label === 0)) {
        args.push("--negative", `${Math.round(point.x)},${Math.round(point.y)}`);
      }

      job.setPhase("track", "Loading models");
      job.setProgress(0.1);
      let started = false;
      let done = 0;
      const track = startChild(job, env.python, args, {
        cwd: repoRoot,
        onLine: (line) => {
          // One line per written frame, whatever index it has; the count is what drives the bar.
          if (!/^frame \d+:/.test(line)) return;
          if (!started) {
            started = true;
            job.label = "Tracking the object";
          }
          done += 1;
          job.setFrames(done, expected);
          job.setProgress(0.1 + 0.9 * Math.min(1, done / expected));
        },
      });
      const tracked = await track.done;
      if (job.cancelled) return job.finish("cancelled");
      if (tracked.error) throw new Error(`Could not start Python: ${tracked.error.message}`);
      if (tracked.code !== 0) throw new Error(`track.py failed (exit ${tracked.code ?? tracked.signal})`);

      job.setFrames(expected, expected);
      job.finish("done", null, `Wrote ${expected} frames (${settings.direction} from frame ${start})`);
      job.append(`Export finished: ${settings.outDir}`);
      if (settings.openOnFinish) spawnDetached("open", [settings.outDir]);
    } catch (error) {
      job.finish("error", error instanceof Error ? error.message : String(error), null, job.tail());
    } finally {
      if (tempRoot) rmSync(tempRoot, { recursive: true, force: true });
    }
  })();
}

function countMasks(dir: string): number {
  try {
    return readdirSync(dir).filter((name) => /^\d{5}\.png$/.test(name)).length;
  } catch {
    return 0;
  }
}
