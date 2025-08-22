import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
// For multi-page (popup, options) and scripts (content, background)
export default defineConfig({
    plugins: [react()],
    build: {
        rollupOptions: {
            input: {
                background: "src/background/index.ts",
                content: "src/content/index.ts",
                popup: "src/popup/index.html",
                options: "src/options/index.html",
            },
            output: {
                entryFileNames: (chunk) => {
                    // keep clean dist folder
                    if (chunk.name === "background")
                        return "background.js";
                    if (chunk.name === "content")
                        return "content.js";
                    return "assets/[name].js";
                },
            },
        },
        outDir: "dist",
        emptyOutDir: true,
    },
});
