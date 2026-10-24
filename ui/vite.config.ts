import { defineConfig } from "vite";
import solid from "@solidjs/vite-plugin";

// The browser app is a plain Solid SPA; everything that needs a process (ffmpeg, Python, Finder) is proxied to
// the small Node server in server/. `npm run dev` starts both.
export default defineConfig({
  plugins: [solid()],
  server: {
    port: 5173,
    strictPort: true,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8765",
        // Model initialization and tracking run for minutes, and the native folder picker blocks until the user
        // answers, so the proxy must not time out before the backend does.
        timeout: 0,
        proxyTimeout: 0,
      },
    },
  },
  build: {
    target: "esnext",
    outDir: "dist",
    emptyOutDir: true,
  },
});
