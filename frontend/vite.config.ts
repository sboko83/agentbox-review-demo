import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// The browser talks to the dev server only; /api is proxied to the backend,
// so the frontend never needs to know the backend port or deal with CORS.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://127.0.0.1:8080',
        changeOrigin: true,
      },
    },
  },
})
