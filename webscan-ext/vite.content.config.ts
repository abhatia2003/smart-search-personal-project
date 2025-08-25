import { defineConfig } from "vite"

export default defineConfig({
  build: {
    lib: {
      entry: "src/content/index.ts",
      name: "content",
      formats: ["iife"],
      fileName: () => "content.js",
    },
    outDir: "dist",
    emptyOutDir: false,
  },
})