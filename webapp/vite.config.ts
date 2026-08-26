import react from '@vitejs/plugin-react'
import { defineConfig } from 'vite'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    // wllama needs cross-origin isolation to use SharedArrayBuffer, and
    // without it llama.cpp falls back to a single thread. `credentialless`
    // rather than `require-corp` so that the model download from Hugging
    // Face, which sends no CORP header, is not blocked.
    headers: {
      'Cross-Origin-Opener-Policy': 'same-origin',
      'Cross-Origin-Embedder-Policy': 'credentialless',
    },
  },
  optimizeDeps: {
    // Pre-bundling rewrites the worker and wasm URLs wllama resolves at
    // runtime, so it has to load as-is.
    exclude: ['@wllama/wllama'],
  },
})
