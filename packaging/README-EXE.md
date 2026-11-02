# Portable exe — build & distribution guide

Goal: a **zip you can send to anyone**: they unzip it, double-click `GBVMonitor.exe`,
and the full tool (web UI + local AI engine + browser) runs with **zero installs**
— no Python, no LM Studio, no cloud, nothing to configure.

```
GBVMonitor/                      ← the distribution folder (zip this)
├── GBVMonitor.exe               ← launcher: starts engines, then the UI
├── README.txt
├── _internal/                   ← Python runtime + app + libs (do not touch)
│   └── app.py …                 ← app source read by Streamlit at runtime
├── runtime/
│   ├── vulkan/llama-server…     ← GPU engine (any NVIDIA/AMD/Intel GPU with Vulkan)
│   └── cpu/llama-server…        ← CPU engine (works everywhere)
├── models/
│   ├── Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf                 ← classifier, text AND images (~4.7 GB)
│   ├── mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf             ← vision tower (~1.35 GB)
│   └── nomic-embed-text-v1.5.Q8_0.gguf                    ← OSINT embeddings (~85 MB)
├── ms-playwright/               ← bundled Chromium (page collector)
├── output/                      ← reports, screenshots, case dataset (empty at first)
└── logs/                        ← engine logs for diagnostics
```

## What happens at runtime

`launcher.py` (compiled into the exe):
1. `chdir` to the exe folder (portable `output/`, SQLite DB, logs);
2. sets `GBV_BUNDLED=1`, `PLAYWRIGHT_BROWSERS_PATH`, `GBV_LLM_URL`, `GBV_EMBED_URL`;
3. starts the chat engine on a free `127.0.0.1:76xx` port (Vulkan first, CPU fallback;
   force one with `GBV_BACKEND=cpu`), waits for `/health`,
4. starts the embed engine (`nomic-embed`) if its GGUF is present;
5. serves Streamlit on a free port and opens the browser tab;
6. kills both engines when the console window closes (verified with `atexit`).

The app then talks to the engine through `local_llm.LocalLLMClient`, which exposes
exactly the `LMStudioClient` interface — `analyzer`, `osint`, `agent` and `monitor`
work unchanged. Classic LM Studio setups keep working too (`create_client()` picks
by environment).

## Build it

### Option A — one command on a Windows machine
```powershell
powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1
# options: -SkipVulkan (smaller), -SkipZip, -ChatPrefix qwen2.5-3b-instruct-q4_k_m
```
Needs Python 3.10–3.13 64-bit + internet + ~6 GB disk. Output:
`dist/GBVMonitor/` and `dist/GBVMonitor-win64.zip`.
For a friendlier (single-loader) delivery there's also
`packaging/installer.iss` (Inno Setup) for an optional setup.exe.

### Option B — GitHub Actions (no Windows needed)
Run **Actions → build-windows-exe → Run workflow** on your GitHub repo:
* `chat_prefix` — which GGUF to bake (default 7B q4_k_m; use
  `qwen2.5-3b-instruct-q4_k_m` for lighter bundles),
* `backends` — `cpu,vulkan` (default) or `cpu` only,
* `publish_release` — draft GitHub release with the complete zip split in parts.

Artifacts (14-day retention): `GBVMonitor-app-no-model.zip` (app, no model) and
`GBVMonitor-models.zip.001…N` (model parts). Reassemble on Windows:
```
copy /b GBVMonitor-models.zip.001+GBVMonitor-models.zip.002+… GBVMonitor-models.zip
```
then unzip both into one folder.

## Changing the model later (field upgrades)

Drop another `.gguf` into `models/` and remove the old files, or keep both —
the launcher picks the first single-file / first shard (`*-00001-of-*.gguf`)
found there. Recommendations for Arabic/Darija + JSON reliability:
Qwen2.5-VL-7B-Instruct Q4_K_M (+ `mmproj-*.gguf`, shipped — does text AND
image screening), Qwen2.5-14B on ≥16 GB RAM machines.

To keep image screening working after a swap, copy an `mmproj-*.gguf` that
MATCHES the new model family into `models/` as well; without it the bundle
still classifies text but screenshots/NCII screening stays disabled. A
text-only model (no mmproj anywhere) is fine for caption-only workflows.
Embedding model must stay `*nomic-embed-text-v1.5.Q8_0.gguf` for OSINT.

## Engine tuning (env vars)

| Variable | Meaning |
|---|---|
| `GBV_BACKEND` | force `cpu` / `vulkan` (default: vulkan → cpu fallback) |
| `GBV_CTX` | model context window (default 8192; `--ctx N` at launch) |
| `GBV_ARGS` | extra llama-server flags (e.g. `--no-webui -t 8 -ngl 99`) |

## End-user entry points

| Command | Purpose |
|---|---|
| `GBVMonitor.exe` | start the app (console shows progress/errors, browser opens) |
| `GBVMonitor.exe --no-browser --port 8502` | fixed port for kiosks/testing |
| `GBVMonitor.exe monitor run-due --enrich` | process due watchlists once |
| `GBVMonitor.exe monitor watch --interval 900` | continuous scheduler |
| `GBVMonitor.exe monitor add-watch <url> --label X` | register a page |

## Troubleshooting (foggy machines)

* **"No model found …"** — the `models/` folder was emptied; re-copy the zip.
* **Engine doesn't become healthy** — open `logs/engine-chat.log`; the most common
  cause is RAM (7B Q4_K_M ≈ 5 GB of RAM). Swap in a 3B model or close apps.
* **0 posts collected after long waits** — probably Facebook DOM drift (the same
  as the Docker version); see README "Notes".
* **port/corp proxies** — everything is `127.0.0.1`, no callback home.
* Antivirus: first run may be slow (Chromium first scan) — sign or whitelist as needed.
