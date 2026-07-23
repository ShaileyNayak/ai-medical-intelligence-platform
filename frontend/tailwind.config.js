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
        conf: {
          high: "#15803d",
          mid: "#d97706",
          low: "#dc2626",
        },
      },
      fontFamily: {
        display: ['"Source Serif 4"', "Georgia", "serif"],
        sans: ['"IBM Plex Sans"', "system-ui", "sans-serif"],
      },
      boxShadow: {
        panel: "0 1px 2px rgba(18, 38, 58, 0.04), 0 8px 24px rgba(18, 38, 58, 0.06)",
        panelHover: "0 2px 4px rgba(18, 38, 58, 0.06), 0 12px 28px rgba(18, 38, 58, 0.1)",
      },
    },
  },
  plugins: [],
};
