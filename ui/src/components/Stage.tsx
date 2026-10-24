import type { JSX } from "@solidjs/web";
import { For, Show, createEffect, createSignal, onCleanup } from "solid-js";
import type { PromptPoint, VideoMeta } from "../types.ts";
import { Icon } from "./parts.tsx";

interface StageProps {
  meta: VideoMeta;
  streamUrl: string;
  posterUrl: string;
  points: PromptPoint[];
  /** 0-based frame the points belong to; the player parks there when points are added. */
  promptFrame: number;
  locked: boolean;
  /** Show the written masks composited over the video instead of the point editor. */
  preview: boolean;
  /** A tracking job is running, so frames that are missing now may appear shortly. */
  tracking: boolean;
  /** Frames that already have a mask, as [first, last] ranges. */
  coverage: Array<[number, number]>;
  /** Identifies the mask folder; changing it drops the cached frames. */
  maskKey: string;
  maskUrl: (frame: number) => string;
  onPreview: (value: boolean) => void;
  onAdd: (x: number, y: number) => void;
  onTime: (seconds: number) => void;
  onRemove: (index: number) => void;
}

function formatTime(seconds: number): string {
  if (!Number.isFinite(seconds)) return "0:00.00";
  const minutes = Math.floor(seconds / 60);
  const rest = seconds - minutes * 60;
  return `${minutes}:${rest.toFixed(2).padStart(5, "0")}`;
}

/** Masks are previewed at half size; the tint is 50% opaque, so the pixels saved are invisible. */
const PREVIEW_MAX_WIDTH = 960;
const TINT = [255, 64, 64] as const;
const TINT_ALPHA = 128;
const CACHE_LIMIT = 48;

/** The video preview, the prompt editor, and the mask preview: points live in the prompt frame's pixels. */
export function Stage(props: StageProps): JSX.Element {
  let video: HTMLVideoElement | undefined;
  let canvas: HTMLCanvasElement | undefined;
  const [playing, setPlaying] = createSignal(false);
  const [time, setTime] = createSignal(0);
  const [failed, setFailed] = createSignal(false);
  const [noMasks, setNoMasks] = createSignal(false);

  const radius = () => Math.max(7, props.meta.width * 0.0105);
  const aspect = () => props.meta.width / props.meta.height;
  const fps = () => props.meta.fps || 30;
  const promptTime = () => props.promptFrame / fps();
  /** The last frame an export can have written: ffmpeg emits one past `nb_frames`, at the duration mark. */
  const lastFrame = () =>
    Math.max(0, Math.round(props.meta.duration * fps()), (props.meta.frames ?? 1) - 1);
  const currentFrame = () =>
    Math.max(0, Math.min(lastFrame(), Math.round((video?.currentTime ?? 0) * fps())));
  const shownFrame = () => Math.max(0, Math.min(lastFrame(), Math.round(time() * fps())));

  const seek = (seconds: number) => {
    if (!video) return;
    video.currentTime = Math.max(0, Math.min(seconds, props.meta.duration || seconds));
    setTime(video.currentTime);
  };

  const toggle = () => {
    if (!video) return;
    if (video.paused) void video.play().catch(() => setFailed(true));
    else video.pause();
  };

  const addPoint = (event: MouseEvent & { currentTarget: SVGSVGElement }) => {
    if (props.locked || props.preview) return;
    const rect = event.currentTarget.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    const x = ((event.clientX - rect.left) / rect.width) * props.meta.width;
    const y = ((event.clientY - rect.top) / rect.height) * props.meta.height;
    if (x < 0 || y < 0 || x > props.meta.width || y > props.meta.height) return;
    props.onAdd(Math.round(x), Math.round(y));
    // Points describe the prompt frame, so show the frame they describe.
    video?.pause();
    seek(promptTime());
  };

  // Follow the prompt frame when it is changed elsewhere (the number field or Use playhead). The time and the
  // tolerance are computed in the tracked phase; the effect itself only touches the player.
  createEffect(
    () => ({ seconds: promptTime(), step: 0.5 / fps() }),
    ({ seconds, step }) => {
      if (video && Math.abs(video.currentTime - seconds) > step) seek(seconds);
    },
  );

  // ── mask preview ────────────────────────────────────────────────────────
  // Decoded at preview size: a full 1080p frame would be 8 MB per entry.
  const cache = new Map<number, ImageBitmap>();
  const pending = new Set<number>();
  const failedAt = new Map<number, number>();
  const RETRY_MS = 1200;
  let lastDrawn = -1;
  let raf = 0;

  const previewSize = () => {
    const width = Math.min(PREVIEW_MAX_WIDTH, props.meta.width);
    return { width, height: Math.max(1, Math.round((width * props.meta.height) / props.meta.width)) };
  };

  const clearCanvas = () => {
    lastDrawn = -1;
    if (!canvas) return;
    canvas.getContext("2d")?.clearRect(0, 0, canvas.width, canvas.height);
  };

  /** Turns the white-on-black mask into a translucent tint by using its luminance as alpha. */
  const drawMask = (image: ImageBitmap, frame: number) => {
    if (!canvas || !props.preview) return;
    const { width, height } = previewSize();
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const context = canvas.getContext("2d", { willReadFrequently: true });
    if (!context) return;
    context.clearRect(0, 0, width, height);
    context.drawImage(image, 0, 0, width, height);
    const pixels = context.getImageData(0, 0, width, height);
    const data = pixels.data;
    for (let i = 0; i < data.length; i += 4) {
      if ((data[i] ?? 0) > 127) {
        data[i] = TINT[0];
        data[i + 1] = TINT[1];
        data[i + 2] = TINT[2];
        data[i + 3] = TINT_ALPHA;
      } else {
        data[i + 3] = 0;
      }
    }
    context.putImageData(pixels, 0, 0);
    lastDrawn = frame;
  };

  const trimCache = () => {
    while (cache.size > CACHE_LIMIT) {
      const oldest = cache.keys().next().value;
      if (oldest === undefined) break;
      cache.get(oldest)?.close();
      cache.delete(oldest);
    }
  };

  const clearCache = () => {
    for (const bitmap of cache.values()) bitmap.close();
    cache.clear();
    pending.clear();
  };

  const requestMask = (frame: number) => {
    if (frame < 0 || frame > lastFrame() || cache.has(frame) || pending.has(frame)) return;
    const failed = failedAt.get(frame);
    if (failed !== undefined) {
      // While an export runs the frame may simply not be written yet; retry, then stop asking.
      if (!props.tracking || performance.now() - failed < RETRY_MS) return;
      failedAt.delete(frame);
    }
    pending.add(frame);
    const image = new Image();
    image.decoding = "async";
    image.onload = () => {
      const { width, height } = previewSize();
      void createImageBitmap(image, { resizeWidth: width, resizeHeight: height, resizeQuality: "medium" })
        .then((bitmap) => {
          pending.delete(frame);
          if (!props.preview) {
            bitmap.close();
            return;
          }
          failedAt.delete(frame);
          cache.set(frame, bitmap);
          trimCache();
          setNoMasks(false);
          if (currentFrame() === frame) drawMask(bitmap, frame);
        })
        .catch(() => {
          pending.delete(frame);
        });
    };
    image.onerror = () => {
      pending.delete(frame);
      failedAt.set(frame, performance.now());
      // A missing mask is normal before the first export, or for a frame the export did not cover.
      if (frame === 0) setNoMasks(true);
    };
    image.src = props.maskUrl(frame);
  };

  const renderPreview = () => {
    if (!props.preview || !canvas) return;
    const frame = currentFrame();
    const image = cache.get(frame);
    if (image) {
      if (frame !== lastDrawn) drawMask(image, frame);
    } else {
      if (frame !== lastDrawn) clearCanvas();
      requestMask(frame);
      // Prefetch the frames the player is about to reach so playback stays smooth.
      const step = playing() ? 1 : 0;
      for (let ahead = 1; ahead <= 3 + step; ahead += 1) requestMask(frame + ahead);
    }
  };

  const loop = () => {
    if (!props.preview) {
      raf = 0;
      return;
    }
    renderPreview();
    raf = requestAnimationFrame(loop);
  };

  createEffect(
    () => props.preview,
    (on) => {
      if (on) {
        if (!raf) raf = requestAnimationFrame(loop);
      } else {
        cancelAnimationFrame(raf);
        raf = 0;
        clearCanvas();
      }
    },
  );

  // A new mask folder invalidates every decoded frame.
  createEffect(
    () => props.maskKey,
    () => {
      clearCache();
      failedAt.clear();
      setNoMasks(false);
      clearCanvas();
    },
  );

  // When the export ends, give the frame that was being written one last chance.
  createEffect(
    () => props.tracking,
    (tracking) => {
      if (!tracking) failedAt.clear();
    },
  );

  onCleanup(() => {
    cancelAnimationFrame(raf);
    clearCache();
    failedAt.clear();
  });

  const covered = () => props.coverage.reduce((total, [first, last]) => total + last - first + 1, 0);

  return (
    <div class="stage">
      <div
        class="stage-inner"
        style={`aspect-ratio:${props.meta.width}/${props.meta.height}; width:min(100%, calc(58vh * ${aspect()}))`}
      >
        <video
          ref={video}
          class="stage-video"
          src={props.streamUrl}
          poster={props.posterUrl}
          preload="metadata"
          playsinline
          onLoadedMetadata={() => seek(promptTime())}
          onTimeUpdate={(event) => {
            setTime(event.currentTarget.currentTime);
            props.onTime(event.currentTarget.currentTime);
          }}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => setPlaying(false)}
          onError={() => setFailed(true)}
        />
        <canvas ref={canvas} class="stage-mask" aria-hidden="true" />
        <svg
          class={["stage-overlay", { locked: props.locked, hidden: props.preview }]}
          viewBox={`0 0 ${props.meta.width} ${props.meta.height}`}
          preserveAspectRatio="none"
          onClick={addPoint}
        >
          <For each={props.points}>
            {(point, index) => (
              <g
                class={["marker", { negative: point.label === 0 }]}
                onClick={(event) => {
                  event.stopPropagation();
                  if (!props.locked) props.onRemove(index());
                }}
              >
                <title>{point.label === 1 ? "Object" : "Background"} — click to remove</title>
                <circle cx={point.x} cy={point.y} r={radius()} />
                <text x={point.x} y={point.y} font-size={String(radius() * 1.05)}>
                  {index() + 1}
                </text>
              </g>
            )}
          </For>
        </svg>

        <div class="stage-modes" role="group" aria-label="Stage mode">
          <button type="button" class={{ active: !props.preview }} onClick={() => props.onPreview(false)}>
            Edit
          </button>
          <button type="button" class={{ active: props.preview }} onClick={() => props.onPreview(true)}>
            Preview
          </button>
        </div>

        <Show
          when={props.preview}
          fallback={<div class="stage-frame mono">Start frame {props.promptFrame}</div>}
        >
          <div class="stage-frame mono">Result preview · frame {shownFrame()}</div>
        </Show>

        <Show when={props.preview && noMasks()}>
          <div class="stage-hint">No masks in the export folder for this frame yet</div>
        </Show>
        <Show when={!props.preview && props.points.length === 0 && !props.locked}>
          <div class="stage-hint">Click the object to add a point</div>
        </Show>
        <Show when={failed()}>
          <div class="stage-error">
            <Icon name="alert" />
            This video cannot play in the browser (tracking and export still work)
          </div>
        </Show>
      </div>

      <div class="transport">
        <button class="btn icon" type="button" onClick={toggle} title={playing() ? "Pause" : "Play"} aria-label={playing() ? "Pause" : "Play"}>
          <Icon name={playing() ? "pause" : "play"} size={18} />
        </button>
        <button class="btn icon" type="button" onClick={() => seek(promptTime())} title="Jump to the start frame" aria-label="Jump to the start frame">
          <Icon name="rewind" size={16} />
        </button>
        <span class="time mono">
          {formatTime(time())} <span class="dim">/ {formatTime(props.meta.duration)}</span>
        </span>
        <div class="scrub-wrap" title={`${covered()} frames tracked`}>
          <div class="scrub-track" aria-hidden="true">
            <For each={props.coverage}>
              {([first, last]) => (
                <span
                  class="scrub-covered"
                  style={`left:${(first / (lastFrame() + 1)) * 100}%;width:${((last - first + 1) / (lastFrame() + 1)) * 100}%`}
                />
              )}
            </For>
          </div>
          <input
            class="scrub"
            type="range"
            min="0"
            max={Math.max(props.meta.duration, 0.01)}
            step="0.01"
            value={time()}
            aria-label="Playback position"
            disabled={!props.meta.browserPlayable}
            onInput={(event) => {
              video?.pause();
              seek(Number(event.currentTarget.value));
            }}
          />
        </div>
        <span class="meta mono dim">
          {props.meta.width}×{props.meta.height} · {props.meta.fps.toFixed(props.meta.fps % 1 === 0 ? 0 : 2)} fps
          <Show when={props.meta.frames}>
            {(frames) => <span> · {frames()} frames</span>}
          </Show>
        </span>
      </div>
    </div>
  );
}
