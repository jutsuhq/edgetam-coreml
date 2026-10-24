import { createReadStream, statSync } from "node:fs";
import { createHash } from "node:crypto";
import type { IncomingMessage, ServerResponse } from "node:http";
import path from "node:path";
import { isFile, run } from "./util.ts";

export interface VideoMeta {
  path: string;
  width: number;
  height: number;
  duration: number;
  fps: number;
  frames: number | null;
  codec: string | null;
  browserPlayable: boolean;
  sizeBytes: number;
}

const MIME: Record<string, string> = {
  ".mp4": "video/mp4",
  ".m4v": "video/mp4",
  ".mov": "video/quicktime",
  ".webm": "video/webm",
  ".ogv": "video/ogg",
  ".mkv": "video/x-matroska",
  ".avi": "video/x-msvideo",
};

const BROWSER_PLAYABLE = new Set([".mp4", ".m4v", ".mov", ".webm", ".ogv"]);

const cache = new Map<string, { mtimeMs: number; meta: VideoMeta }>();

function parseRate(rate: string | undefined): number {
  if (!rate) return 0;
  const [num, den] = rate.split("/").map(Number);
  if (!num || !den) return 0;
  return num / den;
}

/** Reads the first video stream's geometry with ffprobe; results are cached per path and mtime. */
export function probeVideo(file: string): VideoMeta {
  if (!isFile(file)) throw new Error(`Video not found: ${file}`);
  const stat = statSync(file);
  const cached = cache.get(file);
  if (cached && cached.mtimeMs === stat.mtimeMs) return cached.meta;

  const result = run("ffprobe", [
    "-v", "error",
    "-select_streams", "v:0",
    "-show_entries", "stream=width,height,duration,nb_frames,avg_frame_rate,r_frame_rate,codec_name",
    "-of", "json",
    file,
  ]);
  if (result.status !== 0) {
    throw new Error(`ffprobe failed: ${result.stderr.trim() || `exit ${result.status}`}`);
  }

  const parsed = JSON.parse(result.stdout) as {
    streams?: Array<{
      width?: number; height?: number; duration?: string; nb_frames?: string;
      avg_frame_rate?: string; r_frame_rate?: string; codec_name?: string;
    }>;
  };
  const stream = parsed.streams?.[0];
  if (!stream?.width || !stream.height) throw new Error("Could not read a video stream from the file");

  const fps = parseRate(stream.avg_frame_rate) || parseRate(stream.r_frame_rate) || 30;
  const duration = Number(stream.duration) || 0;
  const declaredFrames = Number(stream.nb_frames);
  const frames = Number.isFinite(declaredFrames) && declaredFrames > 0
    ? declaredFrames
    : duration > 0 ? Math.max(1, Math.round(duration * fps)) : null;

  const ext = path.extname(file).toLowerCase();
  const meta: VideoMeta = {
    path: file,
    width: stream.width,
    height: stream.height,
    duration,
    fps,
    frames,
    codec: stream.codec_name ?? null,
    browserPlayable: BROWSER_PLAYABLE.has(ext),
    sizeBytes: stat.size,
  };
  cache.set(file, { mtimeMs: stat.mtimeMs, meta });
  return meta;
}

export function videoMime(file: string): string {
  return MIME[path.extname(file).toLowerCase()] ?? "application/octet-stream";
}

/**
 * Serves a video file with Range support so the browser can seek. This is the only way to preview a local file:
 * the page is served over http, so a `file://` source is blocked.
 */
export function streamVideo(req: IncomingMessage, res: ServerResponse, file: string): void {
  if (!isFile(file)) {
    res.writeHead(404, { "content-type": "application/json; charset=utf-8" });
    res.end(JSON.stringify({ error: `Video not found: ${file}` }));
    return;
  }
  const size = statSync(file).size;
  const type = videoMime(file);
  const range = req.headers.range;

  res.setHeader("accept-ranges", "bytes");
  res.setHeader("content-type", type);
  res.setHeader("cache-control", "no-store");

  if (!range) {
    res.writeHead(200, { "content-length": size });
    if (req.method === "HEAD") return void res.end();
    createReadStream(file).pipe(res);
    return;
  }

  const match = /^bytes=(\d*)-(\d*)$/.exec(range.trim());
  if (!match) {
    res.writeHead(416, { "content-range": `bytes */${size}` });
    res.end();
    return;
  }
  const rawStart = match[1] ?? "";
  const rawEnd = match[2] ?? "";
  let start = rawStart ? Number(rawStart) : 0;
  let end = rawEnd ? Number(rawEnd) : size - 1;
  if (!rawStart && rawEnd) {
    // A suffix range: the last N bytes.
    start = Math.max(0, size - Number(rawEnd));
    end = size - 1;
  }
  if (!Number.isFinite(start) || !Number.isFinite(end) || start > end || start >= size) {
    res.writeHead(416, { "content-range": `bytes */${size}` });
    res.end();
    return;
  }
  end = Math.min(end, size - 1);

  res.writeHead(206, {
    "content-range": `bytes ${start}-${end}/${size}`,
    "content-length": end - start + 1,
  });
  if (req.method === "HEAD") return void res.end();
  createReadStream(file, { start, end }).pipe(res);
}

/** Extracts the first frame as a JPEG, used as a poster while the video streams in. */
export function posterFile(file: string, cacheDir: string): Promise<string> {
  return new Promise((resolve, reject) => {
    const key = createHash("sha1").update(file).digest("hex").slice(0, 16);
    const out = path.join(cacheDir, `${key}-${statSync(file).mtimeMs}.jpg`);
    if (isFile(out)) return resolve(out);
    const result = run("ffmpeg", ["-v", "error", "-y", "-i", file, "-frames:v", "1", "-q:v", "3", out], 30_000);
    if (result.status === 0 && isFile(out)) resolve(out);
    else reject(new Error(result.stderr.trim() || "Could not extract the first frame"));
  });
}
