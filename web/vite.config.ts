import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: {
    port: 5173,
    allowedHosts: ["crosswalk-review.localhost", "os-machina.tail5bb1d7.ts.net"],
    proxy: {
      "/api": {
        target: "http://localhost:8787",
        changeOrigin: true,
        ws: true,
      },
      "/assets/raw": {
        target: "http://localhost:8787",
        changeOrigin: true,
      },
      "/assets/processed": {
        target: "http://localhost:8787",
        changeOrigin: true,
      },
      "/healthz": {
        target: "http://localhost:8787",
        changeOrigin: true,
      },
    },
  },
});
