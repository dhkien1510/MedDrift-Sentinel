import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    host: '0.0.0.0',  // Bind to all interfaces inside Docker
    proxy: {
      '/api': 'http://backend:5000',    // ✅ tên service "backend", port 5000
      '/health': 'http://backend:5000'
    },
    watch: {
      usePolling: true
    }
  }
})