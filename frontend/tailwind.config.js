/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,jsx}"],
  theme: {
    extend: {
      colors: {
        clinical: {
          ink: "#12263a",
          muted: "#5b6b7c",
          line: "#d7e0ea",
          soft: "#f4f7fa",
          panel: "#ffffff",
          teal: "#0e7490",
          tealDark: "#155e75",
          alert: "#b45309",
          danger: "#b91c1c",
        },
      },
      fontFamily: {
        display: ['"Source Serif 4"', "Georgia", "serif"],
        sans: ['"IBM Plex Sans"', "system-ui", "sans-serif"],
      },
      boxShadow: {
        panel: "0 1px 2px rgba(18, 38, 58, 0.04), 0 8px 24px rgba(18, 38, 58, 0.06)",
      },
    },
  },
  plugins: [],
};
