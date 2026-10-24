import { createReadStream, mkdirSync, readdirSync, statSync } from "node:fs";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import path from "node:path";
import { startConvert } from "./convert.ts";
import { chooseDialog, type DialogKind } from "./dialog.ts";
import { probeEnvironment, trackingBlockers } from "./env.ts";
import { JobManager } from "./jobs.ts";
import { probeVideo, posterFile, streamVideo } from "./media.ts";
import { SettingsStore, type SettingsPatch } from "./settings.ts";
import { normalizePoints, startTracking } from "./track.ts";
import { exists, isDirectory, isFile, isInside, normalizePath, spawnDetached } from "./util.ts";

const uiRoot = path.resolve(import.meta.dirname, "..");
const repoRoot = path.resolve(uiRoot, "..");
const distDir = path.join(uiRoot, "dist");
const cacheDir = path.join(repoRoot, ".local", "posters");
const port = Number(process.env.ETAM_PORT ?? 8765);

const settings = new SettingsStore(repoRoot);
const jobs = new JobManager();

const STATIC_TYPES: Record<string, string> = {
  ".html": "text/html; charset=utf-8",
  ".js": "text/javascript; charset=utf-8",
  ".css": "text/css; charset=utf-8",
  ".svg": "image/svg+xml",
  ".png": "image/png",
  ".jpg": "image/jpeg",
  ".webp": "image/webp",
  ".ico": "image/x-icon",
  ".json": "application/json; charset=utf-8",
  ".woff2": "font/woff2",
};

function state(logFrom: number) {
  const current = settings.get();
  return {
    settings: current,
    env: probeEnvironment(repoRoot, current),
    job: jobs.snapshot(logFrom),
  };
}

function json(res: ServerResponse, status: number, body: unknown): void {
  const payload = JSON.stringify(body);
  res.writeHead(status, {
    "content-type": "application/json; charset=utf-8",
    "content-length": Buffer.byteLength(payload),
    "cache-control": "no-store",
  });
  res.end(payload);
}

function fail(res: ServerResponse, status: number, message: string): void {
  json(res, status, { error: message });
}

async function readJson(req: IncomingMessage): Promise<Record<string, unknown>> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = chunk as Buffer;
    size += buffer.length;
    if (size > 1_000_000) throw new Error("Request body too large");
    chunks.push(buffer);
  }
  if (chunks.length === 0) return {};
  try {
    const parsed: unknown = JSON.parse(Buffer.concat(chunks).toString("utf8"));
    if (typeof parsed !== "object" || parsed === null) return {};
    return parsed as Record<string, unknown>;
  } catch {
    throw new Error("Could not parse JSON");
  }
}

/**
 * Only the local UI may drive this server: it spawns scripts, opens Finder and serves arbitrary video paths.
 * Browsers attach Origin to cross-origin writes, so refusing foreign origins stops a drive-by page from using it.
 */
function sameOrigin(req: IncomingMessage): boolean {
  const origin = req.headers.origin;
  if (!origin) return true;
  try {
    const host = new URL(origin).hostname;
    return host === "127.0.0.1" || host === "localhost" || host === "[::1]";
  } catch {
    return false;
  }
}

function requirePath(value: unknown, label: string): string {
  const resolved = normalizePath(String(value ?? ""));
  if (!resolved) throw new Error(`${label} path is required`);
  return resolved;
}

async function handleApi(req: IncomingMessage, res: ServerResponse, url: URL): Promise<void> {
  const route = `${req.method ?? "GET"} ${url.pathname}`;

  switch (route) {
    case "GET /api/state": {
      const logFrom = Number(url.searchParams.get("logFrom") ?? 0);
      return json(res, 200, state(Number.isFinite(logFrom) ? logFrom : 0));
    }

    case "GET /api/job": {
      const logFrom = Number(url.searchParams.get("logFrom") ?? 0);
      return json(res, 200, { job: jobs.snapshot(Number.isFinite(logFrom) ? logFrom : 0) });
    }

    case "PUT /api/settings": {
      const patch = await readJson(req);
      const saved = settings.update(patch as SettingsPatch);
      return json(res, 200, { settings: saved, env: probeEnvironment(repoRoot, saved) });
    }

    case "POST /api/dialog": {
      const body = await readJson(req);
      const kind: DialogKind = body.kind === "file" ? "file" : "folder";
      const prompt = typeof body.prompt === "string" ? body.prompt : "Choose a file";
      const startPath = typeof body.startPath === "string" ? body.startPath : "";
      return json(res, 200, chooseDialog(kind, prompt, startPath));
    }

    case "GET /api/video/meta": {
      const file = requirePath(url.searchParams.get("path"), "Video");
      return json(res, 200, probeVideo(file));
    }

    case "GET /api/video/poster": {
      const file = requirePath(url.searchParams.get("path"), "Video");
      mkdirSync(cacheDir, { recursive: true });
      const poster = await posterFile(file, cacheDir);
      const stat = statSync(poster);
      res.writeHead(200, {
        "content-type": "image/jpeg",
        "content-length": stat.size,
        "cache-control": "private, max-age=60",
      });
      createReadStream(poster).pipe(res);
      return;
    }

    case "GET /api/masks": {
      const dir = requirePath(url.searchParams.get("path"), "Mask folder");
      const ranges: Array<[number, number]> = [];
      let count = 0;
      if (isDirectory(dir)) {
        const frames = readdirSync(dir)
          .filter((name) => /^\d{5}\.png$/.test(name))
          .map((name) => Number(name.slice(0, 5)))
          .sort((a, b) => a - b);
        count = frames.length;
        for (const frame of frames) {
          const last = ranges[ranges.length - 1];
          if (last && frame === last[1] + 1) last[1] = frame;
          else ranges.push([frame, frame]);
        }
      }
      return json(res, 200, { count, ranges });
    }

    case "GET /api/mask": {
      const dir = requirePath(url.searchParams.get("path"), "Mask folder");
      const frame = Number(url.searchParams.get("frame"));
      if (!Number.isInteger(frame) || frame < 0 || frame > 99999) return fail(res, 400, "Invalid frame number");
      const file = path.join(dir, `${String(frame).padStart(5, "0")}.png`);
      if (!isFile(file)) return fail(res, 404, `No mask for frame ${frame}`);
      const stat = statSync(file);
      res.writeHead(200, {
        "content-type": "image/png",
        "content-length": stat.size,
        "cache-control": "private, max-age=3600",
      });
      createReadStream(file).pipe(res);
      return;
    }

    case "POST /api/convert/start": {
      const current = settings.get();
      const env = probeEnvironment(repoRoot, current);
      if (!env.convertReady) return fail(res, 400, "convert.sh not found");
      if (!env.uv) return fail(res, 400, "uv not found (https://github.com/astral-sh/uv)");
      if (!env.git) return fail(res, 400, "git not found");
      if (!env.enoughSpace) return fail(res, 400, `Less than 2 GB free in ${current.workDir}`);
      const job = jobs.start("convert");
      startConvert(job, repoRoot, current);
      return json(res, 200, { job: job.snapshot() });
    }

    case "POST /api/track/start": {
      const current = settings.get();
      const env = probeEnvironment(repoRoot, current);
      const blockers = trackingBlockers(env, current);
      if (!current.videoPath) blockers.unshift("Open a video");
      if (blockers.length > 0) return fail(res, 400, blockers.join(" / "));

      const body = await readJson(req);
      const points = normalizePoints(body.points);
      const meta = probeVideo(current.videoPath);
      if (meta.frames !== null && current.startFrame >= meta.frames) {
        return fail(res, 400, `Start frame ${current.startFrame} is outside the video (0–${meta.frames - 1})`);
      }
      if (points.some((point) => point.x < 0 || point.y < 0 || point.x > meta.width || point.y > meta.height)) {
        return fail(res, 400, "A point is outside the video frame");
      }

      const job = jobs.start("track");
      startTracking(job, { repoRoot, settings: current, env, points });
      return json(res, 200, { job: job.snapshot() });
    }

    case "POST /api/job/cancel": {
      if (!jobs.cancel()) return fail(res, 409, "No job is running");
      return json(res, 200, { job: jobs.snapshot() });
    }

    case "POST /api/reveal": {
      const body = await readJson(req);
      const target = requirePath(body.path, "Folder");
      if (!exists(target)) return fail(res, 404, `Folder not found: ${target}`);
      spawnDetached("open", [target]);
      return json(res, 200, { ok: true });
    }

    case "GET /api/video/stream": {
      const file = requirePath(url.searchParams.get("path"), "Video");
      streamVideo(req, res, file);
      return;
    }

    default:
      return fail(res, 404, `Unknown API route: ${route}`);
  }
}

function serveStatic(req: IncomingMessage, res: ServerResponse, url: URL): void {
  if (req.method !== "GET" && req.method !== "HEAD") return fail(res, 405, "Method Not Allowed");

  if (!isFile(path.join(distDir, "index.html"))) {
    res.writeHead(503, { "content-type": "text/html; charset=utf-8" });
    res.end(
      "<!doctype html><meta charset=utf-8><body style=\"font:16px/1.6 system-ui;background:#0d0f13;color:#e7e9ee;padding:40px\">" +
        "<h1>UI is not built</h1><p>Run <code>npm run build</code> in <code>ui/</code>, or <code>npm run dev</code> for HMR.</p></body>",
    );
    return;
  }

  let requested: string;
  try {
    requested = decodeURIComponent(url.pathname);
  } catch {
    requested = "/";
  }
  const candidate = path.join(distDir, requested === "/" ? "index.html" : requested);  const file = isInside(distDir, candidate) && isFile(candidate) ? candidate : path.join(distDir, "index.html");
  const ext = path.extname(file).toLowerCase();
  const stat = statSync(file);
  res.writeHead(200, {
    "content-type": STATIC_TYPES[ext] ?? "application/octet-stream",
    "content-length": stat.size,
    "cache-control": file.includes(`${path.sep}assets${path.sep}`) ? "public, max-age=31536000, immutable" : "no-cache",
  });
  if (req.method === "HEAD") return void res.end();
  createReadStream(file).pipe(res);
}

const server = createServer((req, res) => {
  const url = new URL(req.url ?? "/", `http://${req.headers.host ?? "127.0.0.1"}`);

  if (url.pathname.startsWith("/api/")) {
    if (!sameOrigin(req)) return fail(res, 403, "This API only accepts requests from the local UI");
    handleApi(req, res, url).catch((error: unknown) => {
      const message = error instanceof Error ? error.message : String(error);
      if (!res.headersSent) fail(res, 400, message);
      else res.end();
    });
    return;
  }

  if (req.method === "GET" && req.url?.startsWith("/favicon")) {
    res.writeHead(204);
    return res.end();
  }

  serveStatic(req, res, url);
});

server.listen(port, "127.0.0.1", () => {
  const address = `http://127.0.0.1:${port}`;
  console.log(`EdgeTAM UI API: ${address}`);
  if (process.env.ETAM_OPEN === "1") spawnDetached("open", [address]);
});

function shutdown(): void {
  jobs.cancel();
  server.close(() => process.exit(0));
  setTimeout(() => process.exit(0), 2000);
}

process.on("SIGINT", shutdown);
process.on("SIGTERM", shutdown);
