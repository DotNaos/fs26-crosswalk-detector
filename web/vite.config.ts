import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    allowedHosts: ["crosswalk-review.localhost", "os-machina.tail5bb1d7.ts.net"],
  },
});
