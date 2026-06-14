import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const PORTLESS_MARKER = "CROSSWALK_REVIEW_VIA_PORTLESS";
const backendTarget = process.env.CROSSWALK_REVIEW_API_TARGET ?? "http://127.0.0.1:18787";

export default defineConfig(({ command }) => {
  if (command === "serve" && process.env[PORTLESS_MARKER] !== "1") {
    throw new Error("Crosswalk Detector frontend must be started through Portless. Use `bun run dev:client` or `bun run dev`.");
  }

  return {
    plugins: [react(), tailwindcss()],
    server: {
      port: 5173,
      allowedHosts: ["crosswalk-review.localhost", "os-machina.tail5bb1d7.ts.net"],
      proxy: {
        "/api": {
          target: backendTarget,
          changeOrigin: true,
          ws: true,
        },
        "/assets/raw": {
          target: backendTarget,
          changeOrigin: true,
        },
        "/assets/processed": {
          target: backendTarget,
          changeOrigin: true,
        },
        "/healthz": {
          target: backendTarget,
          changeOrigin: true,
        },
      },
    },
  };
});
