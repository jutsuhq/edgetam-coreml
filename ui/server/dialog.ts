import { isDirectory, isFile, run } from "./util.ts";

export type DialogKind = "folder" | "file";

interface DialogResult {
  path: string | null;
  canceled: boolean;
}

const FOLDER_SCRIPT = `
on run argv
  set promptText to item 1 of argv
  set startPath to item 2 of argv
  if startPath is "" then
    set chosen to choose folder with prompt promptText
  else
    set chosen to choose folder with prompt promptText default location (POSIX file startPath)
  end if
  return POSIX path of chosen
end run
`;

const FILE_SCRIPT = `
on run argv
  set promptText to item 1 of argv
  set startPath to item 2 of argv
  set typeList to {item 3 of argv, item 4 of argv}
  if startPath is "" then
    set chosen to choose file with prompt promptText of type typeList
  else
    set chosen to choose file with prompt promptText of type typeList default location (POSIX file startPath)
  end if
  return POSIX path of chosen
end run
`;

/**
 * Asks macOS for a real filesystem path. The browser's file input hands back a sandboxed copy with no path, which is
 * useless for spawning ffmpeg and Python, so the dialog runs in the server process.
 */
export function chooseDialog(kind: DialogKind, prompt: string, startPath: string): DialogResult {
  const script = kind === "folder" ? FOLDER_SCRIPT : FILE_SCRIPT;
  const usableStart = kind === "folder" ? isDirectory(startPath) : isFile(startPath);
  const args = ["-e", script, "--", prompt, usableStart ? startPath : ""];
  // Two UTIs, because `choose file` only filters reliably with a list: any movie, then MPEG-4 specifically.
  if (kind === "file") args.push("public.movie", "public.mpeg-4");

  const result = run("/usr/bin/osascript", args, 10 * 60_000);
  if (result.status === 0) {
    const chosen = result.stdout.trim();
    return { path: chosen || null, canceled: !chosen };
  }
  // AppleScript reports a canceled dialog as error -128; anything else is a real failure.
  if (/-128/.test(result.stderr) || /User canceled/i.test(result.stderr)) return { path: null, canceled: true };
  throw new Error(result.stderr.trim() || "Could not open the file dialog");
}
