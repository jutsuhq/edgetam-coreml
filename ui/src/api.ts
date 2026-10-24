import type { EnvironmentInfo, JobSnapshot, PromptPoint, Settings, SettingsPatch, VideoMeta } from "./types.ts";

export interface StateResponse {
  settings: Settings;
  env: EnvironmentInfo;
  job: JobSnapshot | null;
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { "content-type": "application/json", ...(init.headers ?? {}) },
  });
  const text = await response.text();
  let body: unknown = null;
  try {
    body = text ? JSON.parse(text) : null;
  } catch {
    // Keep the raw text for the error message below.
  }
  if (!response.ok) {
    const message = body && typeof body === "object" && "error" in body ? String((body as { error: unknown }).error) : text;
    throw new Error(message || `HTTP ${response.status}`);
  }
  return body as T;
}

const query = (value: string) => encodeURIComponent(value);

export const api = {
  state: (logFrom: number) => request<StateResponse>(`/api/state?logFrom=${logFrom}`),

  job: (logFrom: number) => request<{ job: JobSnapshot | null }>(`/api/job?logFrom=${logFrom}`),

  updateSettings: (patch: SettingsPatch) =>
    request<{ settings: Settings; env: EnvironmentInfo }>("/api/settings", {
      method: "PUT",
      body: JSON.stringify(patch),
    }),

  dialog: (kind: "folder" | "file", prompt: string, startPath: string) =>
    request<{ path: string | null; canceled: boolean }>("/api/dialog", {
      method: "POST",
      body: JSON.stringify({ kind, prompt, startPath }),
    }),

  videoMeta: (path: string) => request<VideoMeta>(`/api/video/meta?path=${query(path)}`),

  posterUrl: (path: string) => `/api/video/poster?path=${query(path)}`,

  maskUrl: (outDir: string, frame: number) => `/api/mask?path=${query(outDir)}&frame=${frame}`,

  masks: (outDir: string) =>
    request<{ count: number; ranges: Array<[number, number]> }>(`/api/masks?path=${query(outDir)}`),

  streamUrl: (path: string) => `/api/video/stream?path=${query(path)}`,

  startConvert: () => request<{ job: JobSnapshot }>("/api/convert/start", { method: "POST", body: "{}" }),

  startTrack: (points: PromptPoint[]) =>
    request<{ job: JobSnapshot }>("/api/track/start", {
      method: "POST",
      body: JSON.stringify({ points }),
    }),

  cancelJob: () => request<{ job: JobSnapshot }>("/api/job/cancel", { method: "POST", body: "{}" }),

  reveal: (path: string) => request<{ ok: true }>("/api/reveal", { method: "POST", body: JSON.stringify({ path }) }),
};

export function errorMessage(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}
