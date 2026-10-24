import { spawn, type ChildProcess } from "node:child_process";
import path from "node:path";

// Runs the API server and Vite together so `npm run dev` is one command. Each child keeps its own colours; when one
// exits (Ctrl-C, crash, port clash) the other is stopped too.

const children: ChildProcess[] = [];
let stopping = false;

function run(command: string, args: string[], label: string): ChildProcess {
  const child = spawn(command, args, {
    stdio: "inherit",
    env: { ...process.env, FORCE_COLOR: "1" },
    cwd: import.meta.dirname + "/..",
  });
  child.on("error", (error) => {
    console.error(`[${label}] ${error.message}`);
    stop(1);
  });
  child.on("exit", (code, signal) => {
    if (!stopping) {
      console.error(`[${label}] exited (${signal ?? code})`);
      stop(code ?? 1);
    }
  });
  children.push(child);
  return child;
}

function stop(code: number): void {
  if (stopping) return;
  stopping = true;
  for (const child of children) {
    if (child.exitCode === null && child.signalCode === null) child.kill("SIGTERM");
  }
  setTimeout(() => process.exit(code), 300);
}

process.on("SIGINT", () => stop(0));
process.on("SIGTERM", () => stop(0));

run(process.execPath, ["--watch", path.join("server", "index.ts")], "api");
run(process.execPath, [path.join("node_modules", "vite", "bin", "vite.js")], "web");
