import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// В dev-режиме WebSocket и /api проксируются на python -m brain.server.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/ws": { target: "ws://127.0.0.1:8000", ws: true },
      "/api": { target: "http://127.0.0.1:8000" },
    },
  },
});
