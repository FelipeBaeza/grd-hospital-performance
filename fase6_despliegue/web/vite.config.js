import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// En desarrollo, /api se reenvía a la API local; en producción lo hace nginx (ver nginx.conf).
export default defineConfig({
  plugins: [vue()],
  server: { port: 5173, proxy: { '/api': { target: 'http://localhost:8000', rewrite: (p) => p.replace(/^\/api/, '') } } },
})
