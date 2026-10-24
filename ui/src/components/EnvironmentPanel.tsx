import type { JSX } from "@solidjs/web";
import { Show } from "solid-js";
import type { EnvironmentInfo, JobSnapshot, Settings } from "../types.ts";
import { JobStatus } from "./JobStatus.tsx";
import { Icon, LogPanel, Note, PathField, Section, StatusRow, Switch, formatBytes } from "./parts.tsx";

export function EnvironmentPanel(props: {
  settings: Settings;
  env: EnvironmentInfo;
  job: JobSnapshot | null;
  now: number;
  logs: string[];
  busy: boolean;
  onPickWorkDir: () => void;
  onWorkDir: (value: string) => void;
  onNoParity: (value: boolean) => void;
  onConvert: () => void;
  onCancel: () => void;
  onReveal: (path: string) => void;
}): JSX.Element {
  const convertRunning = () => (props.job?.kind === "convert" ? props.job.running : false);
  const present = () => props.env.models.filter((model) => model.present).length;
  const missing = () => props.env.models.filter((model) => !model.present).map((model) => model.name);

  return (
    <Section
      step="01"
      title="Environment"
      aside={
        <span class={["badge", { good: props.env.ready }]}>
          {props.env.ready ? "Ready" : "Not ready"}
        </span>
      }
    >
      <PathField
        value={props.settings.workDir}
        placeholder=""
        disabled={convertRunning() || props.busy}
        onBrowse={props.onPickWorkDir}
        onCommit={props.onWorkDir}
      />
      <p class="hint">
        Runs the same setup as <code>convert.sh</code>. Packages are written to <code>{props.env.modelsDir}</code>.
      </p>

      <div class="statuses">
        <StatusRow
          ok={props.env.modelsReady}
          label="Models"
          detail={props.env.modelsReady ? "4 / 4" : `${present()} / 4`}
        />
        <StatusRow ok={props.env.pythonReady} label="Python" detail={props.env.pythonReady ? "venv/bin/python" : "Not found"} />
        <StatusRow
          ok={props.env.mediaReady}
          label="ffmpeg"
          detail={props.env.mediaReady ? "Available" : "Not found"}
        />
        <StatusRow
          ok={props.env.enoughSpace}
          label="Free space"
          detail={props.env.freeBytes === null ? "Unknown" : `${formatBytes(props.env.freeBytes)} free`}
        />
      </div>

      <Show when={!props.env.modelsReady && missing().length > 0 && missing().length < 4}>
        <p class="missing mono">{missing().join(" · ")}</p>
      </Show>
      <Show when={!props.env.enoughSpace}>
        <Note tone="warn">Less than 2 GB free in the work folder. Free some space before initializing.</Note>
      </Show>
      <Show when={!props.env.uv || !props.env.git}>
        <Note tone="error">
          {!props.env.uv ? "uv" : "git"} not found. Both are required to initialize.
        </Note>
      </Show>

      <div class="row between">
        <Switch
          checked={!props.settings.noParity}
          label="Run the parity check"
          disabled={convertRunning() || props.busy}
          onChange={(value) => props.onNoParity(!value)}
        />
        <button
          class="btn primary"
          type="button"
          onClick={props.onConvert}
          disabled={convertRunning() || props.busy}
        >
          <Icon name="spark" />
          {props.env.ready ? "Re-initialize" : "Initialize"}
        </button>
      </div>

      <JobStatus job={props.job} kind="convert" now={props.now} onCancel={props.onCancel} onReveal={props.onReveal} />

      <Show when={props.logs.length > 0 || convertRunning()}>
        <LogPanel
          lines={props.logs}
          running={convertRunning()}
          empty="convert.sh output appears here"
        />
      </Show>
    </Section>
  );
}
