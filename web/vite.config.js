import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

const apiOrigin = `http://localhost:${process.env.PAPER_AGENT_API_PORT || '8001'}`

export default defineConfig({
  plugins: [vue()],
  server: {
    proxy: {
      '/api': {
        target: apiOrigin,
        changeOrigin: true
      },
      '/knowledge': {
        target: apiOrigin,
        changeOrigin: true
      },
      '/send_input': {
        target: apiOrigin,
        changeOrigin: true
      }
    }
  }
})