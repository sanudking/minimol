import { defineConfig } from 'vite';

export default defineConfig({
  // Relative asset URLs let the same build work on static hosts and subpaths.
  base: './',
  build: {
    target: 'es2022',
    sourcemap: false,
  },
});
