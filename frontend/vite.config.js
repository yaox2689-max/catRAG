import { defineConfig } from "vite";
import vue from "@vitejs/plugin-vue";
import { fileURLToPath, URL } from "node:url";

const apiTarget = process.env.VITE_DEV_API_PROXY || "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [vue()],
  resolve: {
    alias: {
      "@": fileURLToPath(new URL("./src", import.meta.url)),
    },
  },
  server: {
    port: 5173,
    proxy: {
      "/auth": { target: apiTarget, changeOrigin: true },
      "/sessions": { target: apiTarget, changeOrigin: true },
      "/chat": { target: apiTarget, changeOrigin: true },
      "/documents": { target: apiTarget, changeOrigin: true },
    },
  },
  build: {
    outDir: "dist",
    emptyOutDir: true,
  },
});
