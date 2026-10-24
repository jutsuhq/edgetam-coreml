import type { JSX } from "@solidjs/web";
import { Show } from "solid-js";
import type { Settings, VideoMeta } from "../types.ts";
import { Note, PathField, Section, formatBytes } from "./parts.tsx";

function formatDuration(seconds: number): string {
  const whole = Math.max(0, seconds);
  const minutes = Math.floor(whole / 60);
  const rest = whole - minutes * 60;
  return `${minutes}:${rest.toFixed(2).padStart(5, "0")}`;
}

function baseName(path: string): string {
  const parts = path.split("/");
  return parts[parts.length - 1] ?? path;
}

export function VideoPanel(props: {
  settings: Settings;
  meta: VideoMeta | null;
  error: string | null;
  busy: boolean;
  onOpen: () => void;
  onVideoPath: (value: string) => void;
}): JSX.Element {
  return (
    <Section
      step="02"
      title="Video"
      aside={
        <Show when={props.meta}>
          {(meta) => <span class="badge good">{meta().browserPlayable ? "Preview available" : "Preview unavailable"}</span>}
        </Show>
      }
    >
      <PathField
        value={props.settings.videoPath}
        placeholder=""
        disabled={props.busy}
        onBrowse={props.onOpen}
        onCommit={props.onVideoPath}
      />

      <Show when={props.error}>
        {(message) => <Note tone="error">{message()}</Note>}
      </Show>

      <Show when={props.meta}>
        {(meta) => (
          <>
            <div class="video-name mono" title={meta().path}>{baseName(meta().path)}</div>
            <dl class="facts">
              <div>
                <dt>Duration</dt>
                <dd class="mono">{formatDuration(meta().duration)}</dd>
              </div>
              <div>
                <dt>Frames</dt>
                <dd class="mono">{meta().frames ?? "—"}</dd>
              </div>
              <div>
                <dt>fps</dt>
                <dd class="mono">{meta().fps.toFixed(meta().fps % 1 === 0 ? 0 : 2)}</dd>
              </div>
              <div>
                <dt>Size</dt>
                <dd class="mono">{formatBytes(meta().sizeBytes)}</dd>
              </div>
            </dl>
          </>
        )}
      </Show>

      <p class="hint">Clicking the preview adds a point in first-frame coordinates.</p>
    </Section>
  );
}
