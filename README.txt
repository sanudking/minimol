MiniMoL QR pairing fix

This is a small source patch for the existing ~/minimol project.
It contains only the changed files; it does not include model weights.

Install on macOS:
  cd ~
  unzip -o ~/Downloads/minimol-qr-fix.zip -d ~/minimol
  cd ~/minimol
  npm run build

Then redeploy the new dist/ folder to the same HTTPS host and reload both phones.
If the app is installed as a PWA, close/reopen it after deployment so the updated
service worker (cache v7) takes effect.

Changes:
- QR becomes visible with a preparing state while ICE candidates are gathered.
- ICE gathering wait is race-safe and has a 12-second fallback.
- If gathering is slow, QR is still rendered using candidates gathered so far,
  with a clear warning in the status line.
- QR contains the compressed MiniMoL signaling payload only, not the full page URL,
  reducing QR data size. The scanner continues to accept both raw payloads and links.
- QR rendering errors are surfaced explicitly.
