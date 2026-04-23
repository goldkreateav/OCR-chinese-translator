import { resolve } from "node:path";
import { defineConfig } from "vite";

export default defineConfig({
  build: {
    outDir: "src/ocr_chinese/web/static/dist",
    emptyOutDir: true,
    cssCodeSplit: false,
    rollupOptions: {
      input: resolve(__dirname, "src/ocr_chinese/web/static/app.js"),
      output: {
        entryFileNames: "app.js",
        assetFileNames: "styles.css",
      },
    },
  },
});
