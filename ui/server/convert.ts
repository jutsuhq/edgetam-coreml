import path from "node:path";
import type { Job } from "./jobs.ts";
import { startChild } from "./jobs.ts";
import type { Settings } from "./settings.ts";
import { isDirectory, isFile } from "./util.ts";

const MODELS = [
  "EdgeTAMVideoImageEncoder",
  "EdgeTAMVideoInitializer",
  "EdgeTAMVideoMemoryEncoder",
  "EdgeTAMVideoPropagator",
];

/**
 * Runs `./convert.sh <work folder>` and turns the contents of the work folder into a coarse phase label, because the
 * script is deliberately quiet and otherwise leaves the UI on a spinner for ten minutes.
 */
export function startConvert(job: Job, repoRoot: string, settings: Settings): void {
  const script = path.join(repoRoot, "convert.sh");
  const args = settings.noParity ? [script, settings.workDir, "--no-parity"] : [script, settings.workDir];

  job.setPhase("convert", "Preparing EdgeTAM");
  job.setProgress(null);
  job.outDir = path.join(settings.workDir, "out");

  const watch = setInterval(() => {
    job.label = describePhase(settings.workDir, settings.noParity);
  }, 1500);

  const { done } = startChild(job, "/bin/zsh", args, { cwd: repoRoot, quiet: /^\s*$/ });
  void done.then((outcome) => {
    clearInterval(watch);
    if (job.cancelled) return job.finish("cancelled");
    if (outcome.error) {
      return job.finish("error", `Could not start convert.sh: ${outcome.error.message}`, null, job.tail());
    }
    if (outcome.code === 75) {
      return job.finish("error", "Less than 2 GB free. Free space on the work folder's disk.", null, job.tail());
    }
    if (outcome.code !== 0) {
      return job.finish("error", `convert.sh failed (exit ${outcome.code ?? outcome.signal})`, null, job.tail());
    }
    job.finish("done", null, "Core ML packages updated");
    job.append(`packages: ${path.join(settings.workDir, "out")}`);
  });
}

function describePhase(workDir: string, noParity: boolean): string {
  const present = MODELS.filter((name) => isDirectory(path.join(workDir, "out", `${name}.mlpackage`))).length;
  if (present === 0) return "Preparing EdgeTAM and the Python environment";
  if (present < MODELS.length) return `Converting models (${present}/${MODELS.length})`;
  if (!noParity && !isFile(path.join(workDir, "parity.json"))) return "Checking the conversion against PyTorch";
  if (!noParity && !isFile(path.join(workDir, "parity-fp16.json"))) return "Verifying the fp16 packages";
  return "Finishing up";
}
