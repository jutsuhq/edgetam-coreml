import { createEffect, createSignal, For, Show } from "solid-js";
import type { JSX } from "@solidjs/web";

export type IconName =
  | "folder"
  | "video"
  | "play"
  | "pause"
  | "rewind"
  | "trash"
  | "spark"
  | "stop"
  | "check"
  | "alert"
  | "target"
  | "export"
  | "chevron";

const PATHS: Record<IconName, JSX.Element> = {
  folder: <path d="M3 7.5A2.5 2.5 0 0 1 5.5 5h3.2c.5 0 1 .2 1.4.6l1.1 1.1c.3.3.7.5 1.2.5h5.1A2.5 2.5 0 0 1 20 9.7v6.8a2.5 2.5 0 0 1-2.5 2.5h-12A2.5 2.5 0 0 1 3 16.5z" />,
  video: (
    <>
      <rect x="3" y="5" width="13" height="14" rx="2.5" />
      <path d="M16 10.2l4.2-2.4a.8.8 0 0 1 1.2.7v7a.8.8 0 0 1-1.2.7L16 13.8z" />
    </>
  ),
  play: <path d="M8 5.5v13l10.5-6.5z" />,
  pause: (
    <>
      <rect x="7" y="5" width="3.5" height="14" rx="1.2" />
      <rect x="13.5" y="5" width="3.5" height="14" rx="1.2" />
    </>
  ),
  rewind: (
    <>
      <path d="M11 6.5v11L3.5 12z" />
      <path d="M20.5 6.5v11L13 12z" />
    </>
  ),
  trash: (
    <>
      <path d="M4 7h16" />
      <path d="M9.5 7V5.5A1.5 1.5 0 0 1 11 4h2a1.5 1.5 0 0 1 1.5 1.5V7" />
      <path d="M6.5 7l.8 11.2A2 2 0 0 0 9.3 20h5.4a2 2 0 0 0 2-1.8L17.5 7" />
      <path d="M10.5 11v5M13.5 11v5" />
    </>
  ),
  spark: <path d="M12 3l1.9 5.6L19.5 10l-5.6 1.9L12 17.5l-1.9-5.6L4.5 10l5.6-1.4z" />,
  stop: <rect x="6" y="6" width="12" height="12" rx="2" />,
  check: <path d="M5 12.5l4.5 4.5L19 7" />,
  alert: (
    <>
      <path d="M12 4.5l8.5 15h-17z" />
      <path d="M12 10v4.5M12 17.2v.3" />
    </>
  ),
  target: (
    <>
      <circle cx="12" cy="12" r="7.5" />
      <circle cx="12" cy="12" r="2.5" />
    </>
  ),
  export: (
    <>
      <path d="M12 4v11" />
      <path d="M8 8.5L12 4.5l4 4" />
      <path d="M4.5 15v3A2 2 0 0 0 6.5 20h11a2 2 0 0 0 2-2v-3" />
    </>
  ),
  chevron: <path d="M8.5 10l3.5 3.5 3.5-3.5" />,
};

export function Icon(props: { name: IconName; size?: number }): JSX.Element {
  return (
    <svg
      class="icon"
      width={props.size ?? 16}
      height={props.size ?? 16}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      stroke-width="1.7"
      stroke-linecap="round"
      stroke-linejoin="round"
      aria-hidden="true"
    >
      {PATHS[props.name]}
    </svg>
  );
}

export function Section(props: {
  step: string;
  title: string;
  aside?: JSX.Element;
  children: JSX.Element;
}): JSX.Element {
  return (
    <section class="panel">
      <header class="panel-head">
        <span class="step">{props.step}</span>
        <h2>{props.title}</h2>
        <div class="panel-aside">{props.aside}</div>
      </header>
      <div class="panel-body">{props.children}</div>
    </section>
  );
}

/** A read-only path with a native picker; the browser cannot type a real path for us, so edits commit on blur/Enter. */
export function PathField(props: {
  value: string;
  placeholder: string;
  disabled?: boolean;
  onBrowse: () => void;
  onCommit: (value: string) => void;
}): JSX.Element {
  return (
    <div class="path-field">
      <input
        class="mono"
        value={props.value}
        placeholder={props.placeholder}
        disabled={props.disabled}
        spellcheck={false}
        title={props.value}
        onChange={(event) => {
          const next = event.currentTarget.value.trim();
          if (next !== props.value) props.onCommit(next);
        }}
        onKeyDown={(event) => {
          if (event.key === "Enter") event.currentTarget.blur();
        }}
      />
      <button type="button" class="btn ghost" onClick={props.onBrowse} disabled={props.disabled}>
        <Icon name="folder" />
        Browse
      </button>
    </div>
  );
}

export function StatusRow(props: { ok: boolean; label: string; detail: string }): JSX.Element {
  return (
    <div class={["status-row", { ok: props.ok }]}>
      <span class="status-dot" />
      <span class="status-label">{props.label}</span>
      <span class="status-detail mono" title={props.detail}>
        {props.detail}
      </span>
    </div>
  );
}

export function ProgressBar(props: { progress: number | null; active: boolean }): JSX.Element {
  const percent = () => (props.progress === null ? 0 : Math.round(props.progress * 100));
  return (
    <div class={["progress", { active: props.active, determinate: props.progress !== null }]}>
      <div class="progress-fill" style={props.progress === null ? undefined : `width:${percent()}%`} />
    </div>
  );
}

/**
 * A log tail. It follows the bottom only while the user is already there, so scrolling back to read an error is not
 * yanked away by the next progress line.
 */
export function LogConsole(props: { lines: string[]; tall?: boolean; empty?: string }): JSX.Element {
  let pre: HTMLPreElement | undefined;
  let follow = true;

  const atBottom = (element: HTMLPreElement) =>
    element.scrollHeight - element.scrollTop - element.clientHeight < 24;

  createEffect(
    () => props.lines.length,
    () => {
      if (follow && pre) pre.scrollTop = pre.scrollHeight;
    },
  );

  return (
    <div class={["log-wrap", { tall: props.tall }]}>
      <pre
        ref={pre}
        class="log mono"
        onScroll={(event) => {
          follow = atBottom(event.currentTarget);
        }}
      >
        <Show when={props.lines.length > 0} fallback={<span class="log-empty">{props.empty ?? "No output yet"}</span>}>
          <For each={props.lines}>{(line) => <span class={{ "log-err": /error|failed|not found|traceback/i.test(line) }}>{line}{"\n"}</span>}</For>
        </Show>
      </pre>
    </div>
  );
}

/**
 * A collapsible log. Its open state belongs to the user: a reactive `open` attribute on `<details>` gets re-applied by
 * every job poll, which snapped the panel shut a moment after it was clicked. It opens once when a job starts and is
 * left alone afterwards, so an error stays readable.
 */
export function LogPanel(props: {
  lines: string[];
  running: boolean;
  empty: string;
  tall?: boolean;
}): JSX.Element {
  const [open, setOpen] = createSignal(false);
  const [copied, setCopied] = createSignal(false);

  createEffect(
    () => props.running,
    (running) => {
      if (running) setOpen(true);
    },
  );

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(props.lines.join("\n"));
      setCopied(true);
      setTimeout(() => setCopied(false), 1600);
    } catch {
      // Clipboard permission can be denied; the text is still selectable in the console.
    }
  };

  return (
    <div class={["logs", { open: open() }]}>
      <div class="logs-head">
        <button class="logs-toggle" type="button" aria-expanded={open() ? "true" : "false"} onClick={() => setOpen(!open())}>
          <span class="chevron" aria-hidden="true">
            <Icon name="chevron" size={12} />
          </span>
          Log
          <span class="dim mono">{props.lines.length} lines</span>
        </button>
        <Show when={open()}>
          <button class="btn ghost small" type="button" onClick={copy} disabled={props.lines.length === 0}>
            {copied() ? "Copied" : "Copy"}
          </button>
        </Show>
      </div>
      <Show when={open()}>
        <LogConsole lines={props.lines} tall={props.tall} empty={props.empty} />
      </Show>
    </div>
  );
}

export function Switch(props: { checked: boolean; label: string; disabled?: boolean; onChange: (value: boolean) => void }): JSX.Element {
  return (
    <label class={["switch", { disabled: props.disabled }]}>
      <input
        type="checkbox"
        checked={props.checked}
        disabled={props.disabled}
        onChange={(event) => props.onChange(event.currentTarget.checked)}
      />
      <span class="switch-track" aria-hidden="true">
        <span class="switch-thumb" />
      </span>
      <span class="switch-label">{props.label}</span>
    </label>
  );
}

export function Note(props: { tone?: "warn" | "error" | "info"; children: JSX.Element }): JSX.Element {
  return (
    <p class={["note", `tone-${props.tone ?? "info"}`]}>
      <Icon name={props.tone === "error" ? "alert" : props.tone === "warn" ? "alert" : "check"} size={14} />
      <span>{props.children}</span>
    </p>
  );
}

export function formatBytes(bytes: number | null): string {
  if (bytes === null) return "Unknown";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes;
  let unit = 0;
  while (value >= 1024 && unit < units.length - 1) {
    value /= 1024;
    unit += 1;
  }
  return `${value.toFixed(value < 10 && unit > 0 ? 1 : 0)} ${units[unit]}`;
}

export function formatDuration(ms: number): string {
  const total = Math.max(0, Math.floor(ms / 1000));
  const minutes = Math.floor(total / 60);
  const seconds = total % 60;
  return `${minutes}:${String(seconds).padStart(2, "0")}`;
}
