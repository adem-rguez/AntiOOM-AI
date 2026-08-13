# AntiOOM AI

**AntiOOM AI** is a local-first, daemon-based inference workspace designed to keep local AI within its memory budget. Its Rust daemon manages local model processes and an OpenAI-compatible API; the Electron/React desktop app provides model management, studios, and local chat history.

Suggested GitHub repository name: `anti-oom-ai`.

## Repository layout

- `crates/` — Rust daemon core, backend plugins, protocol, and SDK crates
- `apps/aiatm-desktop/` — Electron + React desktop application
- `models/` — local model storage; intentionally excluded from Git
- `examples/` — API and SDK examples
- `local-inference-daemon-plan.md` — private working plan; intentionally excluded from Git

## Local development

Build the daemon:

```powershell
cargo build --bin daemon-core
```

Run the desktop app:

```powershell
cd apps/aiatm-desktop
npm install
npm run start
```

Place supported local model files in `models/`. Model weights are never committed to this repository.
