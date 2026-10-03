# ZeroDelay — Frontend

Marketing landing page + hands-free voice app shell for ZeroDelay, an
offline, voice-guided repair/maintenance copilot. Built with Next.js
(App Router), TypeScript, Tailwind CSS, and Framer Motion, with an Electron
wrapper for a real installable desktop app.

The voice app is **wired to the ZeroDelay backend** ([`../backend`](../backend)) over
HTTP: it records the mic, streams each turn to `/converse/stream`, and plays
Piper audio chunks. The backend uses faster-whisper ASR, a persisted procedure
state engine for routine commands, and Gemma for open-ended questions and vision.
The checklist comes from the backend's active procedure.

## Structure

```
app/
  page.tsx            Landing page
  login/page.tsx       Company + access code gate before download
  app/page.tsx          Voice app shell (sidebar + voice UI)
  layout.tsx, globals.css
components/
  landing/              Landing page sections (Hero, Header, DownloadCta, ...)
  app-shell/            Sidebar, VoiceVisual, StepOverlay, SensorsPanel, LoginForm
lib/
  api.ts                 Thin HTTP client for the FastAPI backend (+ diagram URLs)
  useVoiceLoop.ts        Hands-free loop: record mic -> /converse/stream -> play audio
  audio.ts               PCM -> WAV encode + base64-WAV decode helpers
  sessions.ts            On-device (localStorage) persistence for past discussions
  mock-data.ts           Legacy demo data; no longer drives the step overlay
  types.ts               Shared types, incl. the backend decision/sensor contract
electron/
  main.js                Electron entry point — serves the static export
                         locally and opens straight to the app (no landing page)
build/
  icon.png               Source icon for the packaged app (used by electron-builder)
public/
  logo.png, diagrams/    Static assets
  downloads/             Where a built .dmg/.exe would be copied for the
                         website's download button (gitignored — see below)
```

## Requirements

- Node.js 18+
- npm

## Run the website locally

```bash
npm install
npm run dev
```

Open http://localhost:3000. Routes:
- `/` — landing page
- `/login?os=mac|pc` — company/access-code gate, auto-triggers the download on success
- `/app` — the voice app shell (also what the desktop app opens directly into)

The landing page works on its own, but `/app` needs the backend running on
`http://127.0.0.1:8000` (see [`../backend/README.md`](../backend/README.md)). Point it
at a different host with `NEXT_PUBLIC_ZD_API`.
The desktop package also needs that Python service running separately; it does
not bundle the backend or model weights. Check `/ready` before the voice demo.

## Build the desktop app (macOS)

```bash
npm run dist:mac
```

This runs `next build` (static export, see `next.config.mjs`) and then
`electron-builder`, producing an unsigned `.dmg`/`.app` in `dist/`. Since
it's unsigned, the first launch requires right-click → Open to bypass
Gatekeeper.

To make the website's "Download for macOS" button actually serve that
build, copy it into `public/downloads/`:

```bash
cp dist/ZeroDelay-*.dmg public/downloads/ZeroDelay-mac.dmg
```

**Note:** packaged binaries are gitignored (`public/downloads/*.dmg` etc.)
because they're much larger than GitHub's 100MB per-file limit. Distribute
real builds via GitHub Releases (or another host) and point the hrefs in
`components/landing/DownloadCta.tsx` and `app/login/page.tsx` at that URL
instead of the local static file.

A Windows build isn't packaged — cross-building an `.exe` from macOS needs
Wine or a Windows/CI machine. The `build.win` config in `package.json` can
be added once that's needed.

## Notes on the demo state

- Voice, retrieval, reasoning, diagrams, and speech use the local backend. The
  procedure engine handles routine commands and Gemma handles open-ended turns.
  The telemetry panel and fault buttons use `/sensors`.
- Turn-taking is voice-activity detection (speak, then pause) rather than a wake word —
  a short pause ends your turn and sends the clip.
- The step checklist uses the backend's authoritative state. Older saved
  transcripts remain visible, but their local step pointer is discarded when
  resumed; the procedure must be selected and confirmed again.
- Transcripts and backend session IDs are saved in `localStorage`; authoritative
  procedure state is in `backend/artifacts/sessions.sqlite`. Both are needed to
  resume a current session with its transcript.
