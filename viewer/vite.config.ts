import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'
import { dirname, resolve } from 'node:path'
import { fileURLToPath } from 'node:url'

const viewerRoot = dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  plugins: [react()],
  build: {
    // Three.js is isolated behind a lazy boundary; its 157 kB gzip chunk is intentional.
    chunkSizeWarningLimit: 650,
    rollupOptions: {
      input: {
        viewer: resolve(viewerRoot, 'index.html'),
        annotate: resolve(viewerRoot, 'annotate.html'),
      },
    },
  },
  server: {
    host: '127.0.0.1',
    port: 4173,
    strictPort: true,
    proxy: {
      '/api/annotation': 'http://127.0.0.1:4174',
    },
  },
  preview: {
    host: '127.0.0.1',
    port: 4173,
    strictPort: true,
  },
})
