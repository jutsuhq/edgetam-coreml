import type { JSX } from "@solidjs/web";
import { For, Show } from "solid-js";
import type { Direction, EnvironmentInfo, JobSnapshot, PromptPoint, Settings, Units, VideoMeta } from "../types.ts";
import { JobStatus } from "./JobStatus.tsx";
import { Icon, LogPanel, PathField, Section, Switch } from "./parts.tsx";

const UNIT_LABELS: Record<Units, string> = {
  ane: "ANE (Neural Engine)",
  gpu: "GPU",
  all: "All",
  cpu: "CPU only",
};

const DIRECTION_LABELS: Record<Direction, string> = {
  forward: "Forward",
  backward: "Backward",
  both: "Both",
};

export function ExportPanel(props: {
  settings: Settings;
  env: EnvironmentInfo;
  job: JobSnapshot | null;
  now: number;
  logs: string[];
  points: PromptPoint[];
  meta: VideoMeta | null;
  onPickOutDir: () => void;
  onOutDir: (value: string) => void;
  onUnits: (value: Units) => void;
  onDirection: (value: Direction) => void;
  onWriteOverlays: (value: boolean) => void;
  onOpenOnFinish: (value: boolean) => void;
  onExport: () => void;
  onCancel: () => void;
  onReveal: (path: string) => void;
}): JSX.Element {
  const trackRunning = () => (props.job?.kind === "track" ? props.job.running : false);
  const anyRunning = () => props.job?.running ?? false;
  const positives = () => props.points.filter((point) => point.label === 1).length;

  const blockers = () => {
    const found: string[] = [];
    if (!props.meta) found.push("Open a video");
    if (!props.env.modelsReady) found.push("Models are not converted yet (run Initialize)");
    if (!props.env.pythonReady) found.push("The Python environment is missing");
    if (!props.env.mediaReady) found.push("ffmpeg / ffprobe not found");
    if (positives() === 0) found.push("Add at least one object point");
    return found;
  };

  const ready = () => blockers().length === 0 && !anyRunning();

  const lastFrame = () => Math.max(0, (props.meta?.frames ?? 1) - 1);
  const startFrame = () => Math.min(props.settings.startFrame, lastFrame());
  const written = () =>
    props.settings.direction === "forward" ? Math.max(0, lastFrame() + 1 - startFrame())
      : props.settings.direction === "backward" ? startFrame() + 1
        : lastFrame() + 1;

  const summary = () => {
    if (!props.meta) return "Open a video to see the range.";
    if (props.settings.direction === "both") {
      return `Frames 0–${lastFrame()}, both ways from ${startFrame()} · ≈${written()} frames`;
    }
    if (props.settings.direction === "forward") {
      return `Frames ${startFrame()}–${lastFrame()}, forward from ${startFrame()} · ≈${written()} frames`;
    }
    return `Frames 0–${startFrame()}, backward from ${startFrame()} · ≈${written()} frames`;
  };

  return (
    <Section
      step="04"
      title="Export"
      aside={
        <Show when={props.meta}>
          {(meta) => <span class="badge">{meta().frames ? `${meta().frames} frames` : "Video"}</span>}
        </Show>
      }
    >
      <PathField
        value={props.settings.outDir}
        placeholder="/path/to/masks"
        disabled={trackRunning()}
        onBrowse={props.onPickOutDir}
        onCommit={props.onOutDir}
      />

      <div class="field-row">
        <label for="units">Compute unit</label>
        <select
          id="units"
          value={props.settings.units}
          disabled={trackRunning()}
          onChange={(event) => props.onUnits(event.currentTarget.value as Units)}
        >
          <For each={Object.keys(UNIT_LABELS) as Units[]}>
            {(unit) => <option value={unit}>{UNIT_LABELS[unit]}</option>}
          </For>
        </select>
      </div>

      <div class="field-row">
        <span class="field-label">Direction</span>
        <div class="segmented" role="group" aria-label="Tracking direction">
          <For each={Object.keys(DIRECTION_LABELS) as Direction[]}>
            {(direction) => (
              <button
                type="button"
                class={{ active: props.settings.direction === direction }}
                disabled={trackRunning()}
                onClick={() => props.onDirection(direction)}
              >
                {DIRECTION_LABELS[direction]}
              </button>
            )}
          </For>
        </div>
      </div>
      <p class="hint mono">{summary()}</p>

      <Switch
        checked={props.settings.writeOverlays}
        label="Write overlay JPEGs"
        disabled={trackRunning()}
        onChange={props.onWriteOverlays}
      />
      <Switch
        checked={props.settings.openOnFinish}
        label="Open the folder when done"
        disabled={trackRunning()}
        onChange={props.onOpenOnFinish}
      />
      <p class="hint">
        Overlays are only for viewing outside the app. Leave them off and press <strong>Preview</strong> on the
        video to see the masks composited over the source; the folder keeps the masks alone.
      </p>

      <Show when={blockers().length > 0 && !anyRunning()}>
        <ul class="blockers">
          <For each={blockers()}>{(blocker) => <li>{blocker}</li>}</For>
        </ul>
      </Show>

      <button class="btn primary wide" type="button" disabled={!ready()} onClick={props.onExport}>
        <Icon name="export" />
        Track and export
      </button>

      <JobStatus job={props.job} kind="track" now={props.now} onCancel={props.onCancel} onReveal={props.onReveal} />

      <Show when={props.logs.length > 0 || trackRunning()}>
        <LogPanel
          lines={props.logs}
          running={trackRunning()}
          tall
          empty="track.py output appears here"
        />
      </Show>
    </Section>
  );
}
