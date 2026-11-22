import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// `npm run dev` proxies API calls to a local `effeval server serve` on port 8600.
export default defineConfig({
  plugins: [react()],
  server: { proxy: { "/api": "http://127.0.0.1:8600" } },
});
