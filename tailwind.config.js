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
        primary: "#2563eb", // Vibrant corporate blue for light theme
        secondary: "#3b82f6", // Secondary blue
        accent: "#60a5fa", // Light blue accent
      },
    },
  },
  plugins: [],
}