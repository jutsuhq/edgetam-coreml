// Shared with the server so the wire format cannot drift. Type-only, so none of the server's Node imports reach the
// browser bundle.
export type { Settings, Units, SettingsPatch, Direction } from "../server/settings.ts";
export type { EnvironmentInfo, ModelState } from "../server/env.ts";
export type { JobPhase, JobSnapshot, JobKind } from "../server/jobs.ts";
export type { VideoMeta } from "../server/media.ts";
export type { PromptPoint } from "../shared/prompt.ts";
export { MAX_PROMPT_POINTS } from "../shared/prompt.ts";
