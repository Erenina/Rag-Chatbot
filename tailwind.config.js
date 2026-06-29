/** Tailwind build config — only used at dev-time to generate static/styles.css */
module.exports = {
  content: ["./static/index.html"],
  theme: {
    extend: {
      colors: {
        primary: '#1E293B',
        secondary: '#334155',
        accent: '#2563EB',
        'accent-hover': '#1D4ED8',
        background: '#F8FAFC',
        foreground: '#0F172A',
        muted: '#F1F5F9',
        'muted-fg': '#64748B',
        border: '#E2E8F0',
        destructive: '#DC2626',
      },
      fontFamily: {
        sans: ['"Plus Jakarta Sans"', 'system-ui', 'sans-serif'],
        mono: ['"JetBrains Mono"', 'monospace'],
      },
    },
  },
  plugins: [],
};
