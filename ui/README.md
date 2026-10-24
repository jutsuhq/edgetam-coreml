# EdgeTAM Track — GUI

A local web UI for the tracking loop in [`examples/track.py`](../examples/track.py): pick a work folder, initialize it
exactly as [`convert.sh`](../convert.sh) does, open a video, click points on the first frame, and export masks and
overlays while a progress bar follows the frames.

Solid 2.0 (RC) + Vite for the page, and a small Node HTTP server for everything a browser cannot do.

## Run

```bash
npm install
npm run dev        # Vite on http://localhost:5173, API on 127.0.0.1:8765
```

```bash
npm run build      # writes dist/
npm start          # serves the built page and the API from http://127.0.0.1:8765
```

Requirements: macOS, Node 22.12+, `ffmpeg`/`ffprobe` on `PATH`, and the work folder `convert.sh` produces (or press
**Initialize** and let it build one).

## What each button does

| UI | Command |
|---|---|
| Initialize | `./convert.sh <work>` (plus `--no-parity` when the parity check is off) |
| Track and export | `examples/track.py --models <work>/out --video <video> --start <frame> --direction <dir> --point/--negative … --out <out> --units <unit>` (plus `--no-overlay` unless overlays are on) |
| Open folder | `open <out>` (also on completion, unless "Open the folder when done" is off) |
| Browse | `osascript` `choose folder` / `choose file` |

Points are pixel coordinates on the **start frame**. Scrub the player, press **Use playhead** (or type a frame
number) to move the prompt frame; the preview parks there, clicking the stage adds a point, and changing the frame
clears the points because they describe the old one. Frame numbers are 0-based, matching the mask filenames.

Click a marker to delete it, or press `Backspace` to remove the last one. Background points become `--negative`. The
initializer accepts **1–16 points in total** (object + background), so the editor stops at 16 and says why.

**Direction** chooses what gets exported from the start frame: `Forward` to the last frame, `Backward` down to frame 0,
or `Both` (two passes from the same seed, each with its own memory bank, as SAM 2 does). The line under the control
shows the frames that will be written.

**Preview** on the video composites the written masks over the source, at 50% red, and follows the player. It turns
on as soon as an export starts, so the frames can be checked while they are written; a frame that is not on disk yet
retries until it appears. The scrubber itself shows which frames already have a mask: tracked frames are a mint fill
on the dark track, growing in whichever direction the export runs. The thumb is the only playback indicator, so
coverage reads the same on either side of it.

The masks are enough for the preview, so **Write overlay JPEGs** is off by default: the folder keeps ~10 KB per frame
instead of ~370 KB, and the preview has no JPEG artifacts. Turn overlays on when you need a ready-made image for
QuickTime, Finder or an editor.

## How it works

- `server/` — no npm dependencies. Routes live in `server/index.ts`; settings in `.local/ui-settings.json`, jobs in
  `server/jobs.ts`, tracking in `server/track.ts`, conversion in `server/convert.ts`.
- The video is streamed from `/api/video/stream` with HTTP Range support, because a page served over http cannot play
  a `file://` source.
- Tracking does **not** reimplement `track.py`: it extracts frames with `ffmpeg -progress` (phase 1 of the bar), then
  runs `examples/track.py` on that folder (phase 2, parsed from its `frame N:` lines). The temp frames are removed
  when the job ends.
- One job runs at a time. Stopping kills the child's process group and cleans up.
- The API binds to `127.0.0.1` and rejects cross-origin `Origin` headers, so a random page cannot drive it.
- The page polls `/api/job?logFrom=N`; the log is capped at 3000 lines server-side and the client only fetches lines
  it has not seen.

## Layout

```
src/App.tsx                 state, polling, settings writes
src/components/Stage.tsx    video, transport, point overlay
src/components/*Panel.tsx   environment, video, points, export
src/components/parts.tsx    shared controls, icons, log console
src/styles.css              the whole look
server/                     the local API and job runner
scripts/dev.ts              runs Vite and the API together
```

## Notes and limits

- Points are not saved between page loads; a new video or start frame clears them.
- Existing masks in the export folder are overwritten by index; extra frames from a longer earlier run stay. The job
  log says so when it finds them.
- The preview reads `NNNNN.png` from the export folder through `/api/mask`; it is a half-resolution canvas tint, not a
  rendered file.
- The first tracked frame waits for Python and Core ML to load the four packages (tens of seconds); the progress label
  says Loading models until frame 0 arrives.
- macOS only: `osascript`, `open`, and the Neural Engine.
