/** @type {import('tailwindcss').Config} */
module.exports = {
  content: [
    "./src/**/*.{js,jsx,ts,tsx}",
  ],
  theme: {
    extend: {
      fontFamily: {
        body: ["Segoe UI", "sans-serif"],
      },
      colors: {
        primary: "#0078D4", // Microsoft Blue
        secondary: "#0078D4", // Uniform color
        accent: "#50E6FF", // Light Blue accent
        "dark": {
          "50": "#2a2a2a",
          "100": "#222222", // Card Background (Dark Gray)
          "200": "#2d2d2d", // Lighter Card Hover
          "300": "#363636",
          "400": "#525252",
          "500": "#050505", // Main Background (Almost Black)
          "600": "#000000", // Gradient End (Pure Black)
          "700": "#000000",
          "800": "#000000",
          "900": "#000000"
        }
      },
    },
  },
  plugins: [],
}