import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// В dev-режиме WebSocket проксируется на python -m brain.server.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
    },
  },
});
