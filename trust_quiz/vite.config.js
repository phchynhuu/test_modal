import { defineConfig } from 'vite';
import react from '@vitejs/plugin-react';

export default defineConfig({
  plugins: [react()],
  // MediaPipe is loaded via <script> CDN tags in index.html — no npm packages to configure.
});
