import type { JSX } from "@solidjs/web";
import { Show } from "solid-js";
import type { JobKind, JobSnapshot } from "../types.ts";
import { Icon, Note, ProgressBar, formatDuration } from "./parts.tsx";

/** Shared progress/result block for both job kinds: phase label, bar, frame counter, elapsed time, cancel/reveal. */
export function JobStatus(props: {
  job: JobSnapshot | null;
  kind: JobKind;
  now: number;
  onCancel: () => void;
  onReveal: (path: string) => void;
}): JSX.Element {
  const job = () => (props.job && props.job.kind === props.kind ? props.job : null);
  const elapsed = (active: JobSnapshot) => formatDuration((active.endedAt ?? props.now) - active.startedAt);

  return (
    <Show when={job()}>
      {(active) => (
        <div class={["job", { running: active().running, failed: active().phase === "error" }]}>
          <Show when={active().running}>
            <div class="job-line">
              <span class="spinner" aria-hidden="true" />
              <span class="job-label">{active().label}</span>
              <span class="job-elapsed mono dim">{elapsed(active())}</span>
            </div>
            <ProgressBar progress={active().progress} active />
            <div class="job-meta mono dim">
              <Show when={active().progress !== null}>
                <span>{Math.round((active().progress ?? 0) * 100)}%</span>
              </Show>
              <Show when={active().framesTotal}>
                <span>
                  {active().framesDone} / {active().framesTotal} frames
                </span>
              </Show>
            </div>
            <button class="btn ghost small" type="button" onClick={props.onCancel}>
              <Icon name="stop" size={13} />
              Stop
            </button>
          </Show>

          <Show when={!active().running && active().phase === "error"}>
            <Note tone="error">{active().error ?? "The job failed"}</Note>
            <Show when={active().detail}>
              {(detail) => <pre class="job-detail mono">{detail()}</pre>}
            </Show>
            <p class="job-hint">The full output is in the log below (copyable).</p>
          </Show>
          <Show when={!active().running && active().phase === "cancelled"}>
            <Note tone="warn">Stopped</Note>
          </Show>
          <Show when={!active().running && active().phase === "done"}>
            <Note>{active().summary ?? "Finished"}</Note>
            <div class="job-meta mono dim">{elapsed(active())}</div>
            <Show when={active().outDir}>
              {(dir) => (
                <button class="btn ghost small" type="button" onClick={() => props.onReveal(dir())}>
                  <Icon name="folder" size={14} />
                  Open folder
                </button>
              )}
            </Show>
          </Show>
        </div>
      )}
    </Show>
  );
}
