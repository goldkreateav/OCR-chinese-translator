/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./src/ocr_chinese/web/static/index.html",
    "./src/ocr_chinese/web/static/app.js",
  ],
  theme: {
    extend: {
      colors: {
        sollers: {
          graphite: "#292523",
          white: "#FFFFFF",
          gray: "#606164",
          graySoft: "#E4E5E6",
          grayBorder: "#CACCCE",
          orange: "#F47C30",
          green: "#014637",
          blue: "#193C84",
          yellow: "#FFF200",
          gold: "#FFC805",
          red: "#C9252C",
          redDark: "#7C1C21",
        },
      },
    },
  },
  plugins: [],
};
