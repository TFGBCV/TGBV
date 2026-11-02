# Facebook GBV & Hate-Speech Monitor (portable exe — or local AI via LM Studio)

Streamlit tool: Facebook cookies + page link → scrape posts (and comments) → classify gender-based
cyberviolence / hate speech against women with a **local AI model** → CSV / JSON / Markdown / PDF
documentation. No scraped content leaves your machine.

## Zero-install distribution (portable exe)

**Recommended for NGOs/activists without a technical support person.** Download or build the
Windows bundle, unzip, double-click `GBVMonitor.exe` — no Python, no LM Studio, no internet
beyond Facebook itself. The classifier (Qwen2.5-VL-7B Q4_K_M — one model that reads both text
and images) and the embedding model ship inside the folder; a bundled llama.cpp engine runs
them locally.

- Build guide: [`packaging/README-EXE.md`](packaging/README-EXE.md)
- One-shot local build: `powershell -ExecutionPolicy Bypass -File packaging\build_windows.ps1`
- CI build: run the **build-windows-exe** GitHub workflow (produces a downloadable bundle)
- Scheduler without a window: `GBVMonitor.exe monitor run-due --enrich` (watchlists keep working)

## Setup (local, no Docker) — LM Studio mode
```bash
pip install -r requirements.txt
playwright install chromium
# In LM Studio: Developer tab → Start Server (default http://localhost:1234)
streamlit run app.py
```
Optional env vars: `LMSTUDIO_URL` (default `http://localhost:1234`), `LM_API_TOKEN` (if you enabled auth in LM Studio).
If Streamlit runs in Docker/WSL and LM Studio on the host, point the URL at the host (e.g. `http://host.docker.internal:1234`) and enable "serve on local network" in LM Studio.

## Run it as a lab (Docker) — for distribution to activists & NGOs
The container only ships the web app + a scraping browser; **the AI model runs on each user's own
computer in LM Studio** — no cloud, no accounts, nothing uploaded. The app lists whichever models
the user has downloaded as soon as LM Studio's server is running.

1. On the user's machine: install [LM Studio](https://lmstudio.ai), download a model (see *Choosing a model* below), then **Developer tab → Start Server** (port 1234).
2. Then, from the project folder:
   ```bash
   docker compose up --build
   # open http://localhost:8501
   ```
   (or without compose: `docker run --rm -p 8501:8501 --add-host=host.docker.internal:host-gateway -v ./output:/app/output --env LMSTUDIO_URL=http://host.docker.internal:1234 <image>`)
3. The sidebar asks for LM Studio and shows the **list of available models automatically** — press *🔄 Connect / refresh models*. 🟢 = already loaded.

To publish the image:
```bash
docker build -t <yourdockerhubuser>/fb-gbv-monitor:latest .
docker push <yourdockerhubuser>/fb-gbv-monitor:latest
# a user then runs:
docker run --rm -p 8501:8501 --add-host=host.docker.internal:host-gateway \
  -v ./output:/app/output --env LMSTUDIO_URL=http://host.docker.internal:1234 \
  <yourdockerhubuser>/fb-gbv-monitor:latest
```
Notes: outputs (reports, screenshots) are written to a mounted `./output/` volume on the user's
machine, and the container never sees LM Studio's data beyond the model list/classification calls.

## The case dataset & the autonomous features (v3)

Since the roadmap rollout the tool has a persistent **case dataset** (`output/gbv.db`, SQLite):
runs, items, verdicts, human review decisions, watchlists, background jobs, agent case files,
per-item entities, proposals and an audit log all land there. Highlights:

- **Review & triage** (`app.py` → tabs): confirm/reject every AI flag with a note — this is the
  product; everything else generates leads. Pending items carry provenance badges
  (♟️ inherited verdict, ⚖️ verifier note, 🖼 media).
- **Structured output + self-consistency** (`analyzer.py`): the batch call enforces a bound JSON
  schema through the engine's `response_format` (auto-degrades to plain prompting for engines
  without schema support), and optional **N-pass voting** (sidebar *Self-consistency passes*)
  re-asks each item at alternating temperature and majority-votes the verdicts — confidence
  becomes inter-pass agreement instead of the model's self-report.
- **Verdict memory** (`memory.py`): any new item whose text is near-identical (cosine ≥ 0.97 over
  the local embedding model) to an item a human already reviewed **inherits that decision**
  (confirmed verdicts and rejected "clean" rulings both seed; cross-page twins included).
  Provenance is stamped as `verdict-memory` with an `inherited_from` anchor — fewer model
  calls, consistent rulings across recurring posts, human decision preserved in the chain of custody.
- **Lexicon safety net** (`lexicon.py`): a curated Arabic/Darija/Arabizi slur-and-threat lexicon
  (51 conservative terms, leet/normalisation aware) prioritises the worst content for the model,
  flags every hit, marks LLM-vs-lexicon disagreements (`lexicon_attention`) for the verifier,
  and exposes the term network as verifier bait.
- **Verifier pass**: a second adversarial classification of flagged items, lexicon-attention
  items, plus a sample of the others; disagreements appear as extra `verifier` verdict rows.
- **OSINT enrichment** (`osint.py`, all local): semantic theme clusters, near-duplicate AND
  verbatim template fingerprinting (word-shingle Jaccard), named-entity extraction persisted
  per-item (`items/entities "who mentioned CEDAW"` queries), posting-cadence buckets, the
  commenter→author + mention amplification graph with PageRank/betweenness leaders (GraphML
  export for Gephi/yEd), **burst detection** (z-score of flagged counts vs the page's own
  history) and **cross-page duplicate matching** (identical text on two or more watched pages).
- **Watchlists + scheduler** (`monitor.py`): register pages with a cadence; delta scraping fetches
  only new content on re-runs. In the UI, processing runs as a **background job** (survives
  page reloads; cookies pass to the worker via memory only). CLI:
  ```bash
  python monitor.py add-watch https://www.facebook.com/pageX --label "Wake" --cadence-hours 12
  python monitor.py run-due --enrich --verify          # one pass over what's due
  python monitor.py watch --interval 900               # continuous (cron/Task Scheduler-friendly)
  python monitor.py eval                               # score the model against your reviews
  python monitor.py selftest                           # environment sanity check
  ```
- **Autonomous agent** (`agent.py`, UI tab or `python monitor.py agent <page_id> "<goal>"`):
  a plan→act→reflect loop with read-only tools (`search_corpus` FTS5 search, `get_item`,
  `reclassify` under a lens, `cluster_corpus`, `entity_graph`, `expand_comments`) plus the gated
  `propose_target` / `queue_for_review`. A reflected step is forced every 3 tool calls and is
  written into the structured case file. Hard budgets (steps, seconds, expansions) are enforced
  in code — the model cannot exceed them, and it can never scrape, post or contact anything on
  its own.
- **Quality & eval tab**: Cohen's κ between your reviews and the AI, per-category human
  precision calibration, and the eval harness — the confirm/reject history becomes a labelled
  regression set (`eval.py` exports gold JSONL, scores precision/recall/F1/severity agreement,
  and snapshots `evals/latest.json`) so prompt/model changes are measured, not vibes.
- **Multimodal screening** (`vision.py`): if a vision-capable model is loaded — the packaged
  exe ships one by default (Qwen2.5-VL + mmproj; nothing to configure), or set `GBV_VISION_MODEL`
  with LM Studio — flagged posts' collected images (the scraper records scontent/fbcdn URLs)
  are screened locally for NCII/sextortion-type visual content — conservative triage leads only.
- **Encryption at rest** (`encryption.py`, optional): the case dataset can be opened through
  SQLCipher (`pip install pysqlcipher3`) with a key from `GBV_DB_KEY` (or the OS keyring via
  `keyring`); generate and store one with `python -m encryption --set-key`. The tool warns
  loudly if a key is configured but the driver is missing — it never silently keeps plaintext.
  A plaintext database must be handled per your org's data-protection rules.
- **Alerts** (`alerts.py`): optional webhook (`GBV_WEBHOOK_URL`) and email (`GBV_ALERT_EMAIL`,
  `SMTP_HOST`...) digests when new high/critical items appear on a scheduled run — now including
  coordinated-burst warnings.
- **Reports**: the PDF/Markdown include the OSINT dossier (themes, template reproduction with
  distinct-author counts, entities, cadence, network leaders), the longitudinal section, verdict
  memory counts, and the human–AI agreement / per-category calibration tables. PDF export also
  carries a burst warning banner when one fired.
- **DOM drift detection** (`tests/dom_smoke.py` + `tools/record_fixture.py`): record a rendered
  Facebook page snapshot once, then run the extractor against saved HTML fixtures offline —
  selector drift is caught with "0 posts found" and a baseline diff, in CI, without touching
  Facebook. Offline unit tests: `python tests/test_units.py` (18 tests, no network, no LM Studio).

## LM Studio endpoints used (`lmstudio.py`)
| Action | Endpoint |
|---|---|
| List models / see what is loaded | `GET /api/v1/models` |
| Load a model with a chosen context length | `POST /api/v1/models/load` |
| Download a model (catalog ID or Hugging Face link + quantization) | `POST /api/v1/models/download` |
| Download progress | `GET /api/v1/models/download/status/:job_id` |
| Classification (stateless, `store: false`) | `POST /api/v1/chat` |

All of these are in the sidebar: pick a model (🟢 = loaded), load it, or download a new one with a progress bar.

## Choosing a model
Needs solid Arabic / French and reliable JSON output. Prefer an instruction-tuned model of 7B+ (e.g. Qwen, Gemma, Llama
families). Very small models miss Darija and sarcasm. Use *Reasoning: auto* (off) for speed; use 4–6 items per request.

## Getting cookies
Log in to a **dedicated** Facebook account in Chrome → install the *Cookie-Editor* extension →
open facebook.com → Export → JSON → paste into the app. `c_user` and `xs` are required.

## Files
- `scraper.py`  Playwright scraper (cookie parsing, scrolling, "See more" expansion, comments, media capture)
- `lmstudio.py` LM Studio REST client + `create_client()` engine selector
- `local_llm.py` bundled llama.cpp engine client (drop-in, same interface; portable exe mode)
- `launcher.py` exe entry point: starts the bundled engine + UI, `monitor` CLI passthrough
- `packaging/`  exe build: PyInstaller spec, asset fetcher, Windows build script, docs
- `analyzer.py` Taxonomy + prompt + batched classification, structured-output schema, self-consistency voting, lexicon triage
- `lexicon.py`  Conservative Arabic/Darija/Arabizi slur & threat lexicon (safety net + verifier bait)
- `memory.py`   Verdict memory: near-identical items inherit human-confirmed/rejected decisions
- `osint.py`    Embeddings, theme/template clusters, entity graph, cadence, bursts, amplification network
- `agent.py`    Bounded plan→act→reflect agent + adversarial verifier pass
- `jobs.py`     Background job engine (background watchlist processing, in-memory cookie handoff)
- `eval.py`     Eval harness: gold export from your reviews + precision/recall/F1 scoring
- `vision.py`   Optional local vision-screening pass over flagged posts' images
- `encryption.py` Optional SQLCipher/keyring encryption at rest for the case dataset
- `storage.py`  SQLite schema v2 (FTS5 search, entities, jobs, case files, κ stats)
- `monitor.py`  CLI: run-due / watch / agent / eval / selftest / add-watch
- `report.py`   Markdown documentation report with hashes, timestamps, coordination + quality sections
- `pdf_report.py` PDF report (cover, summary, charts, detailed records, OSINT dossier, memory & review quality, sign-off, visual record)
- `app.py`      Streamlit UI. Outputs are saved in `output/`
- `tests/`      Offline unit tests + DOM-fixture smoke harness; `tools/record_fixture.py` records fixtures

## Notes
- Facebook changes its DOM often. If 0 posts come back, untick *Headless* and inspect; selectors live in `JS_EXTRACT` in `scraper.py`. Record a snapshot with `tools/record_fixture.py` and add it to CI (`tests/dom_smoke.py`) so drift is caught before users hit it.
- Scraping violates Facebook's ToS (account may be restricted). Cookies are kept in memory only — including for background jobs.
- Authors are pseudonymised by default. Local models are less accurate than large cloud models: treat flags as leads and
  review them by hand before using them as evidence. The *Quality & eval* tab tells you how much to trust the current
  setup (κ, per-category calibration) on **your** review history.
- Verdict memory is deliberately strict (similarity ≥ 0.97 and identical structure): rephrasings are re-classified,
  identical reposts inherit the human decision — and if in doubt, the seed is re-confirmable in the review inbox like any flag.
- Taxonomy (edit `CATEGORIES` in `analyzer.py`) is inspired by UN Women / Council of Europe definitions of online violence against women; adapt it to local law.
- **Screenshots**: with *Screenshot each post* ticked (default), the scraper saves a PNG of every post element
  under `output/shots/<run timestamp>/`, stores the path in each post's `screenshot` field, and the PDF embeds it in the
  flagged-item card plus a *Visual record* appendix (Appendix C) covering every captured item. Screenshots are visual
  context for human review — the SHA-256 integrity anchor remains the text itself (hashes computed on the raw text,
  invisible bidi control characters and *See more*/*See less* artifacts are stripped only before classification).
- **Text cleaning before analysis** (`analyzer.clean_for_analysis`): removes invisible directional marks (U+200B–U+200F,
  isolates), *See more*/*See less* leftovers and collapses excess whitespace. It never touches word order — scraped
  Arabic is stored in natural reading order and verified; never "fix" it by reversing words.
- **PDF report** (`pdf_report.py`): picks a Unicode TTF font automatically (Arial / Segoe UI / DejaVu) so Arabic/Darija
  excerpts render correctly; if your system has none, or you prefer another font, set `GBV_PDF_FONT` (plus optional
  `GBV_PDF_FONT_BOLD` / `GBV_PDF_FONT_ITALIC`) to a `.ttf` path. Requires `fpdf2` + `uharfbuzz` (already in
  `requirements.txt`); without `uharfbuzz` Arabic shaping falls back to unshaped latin-1 text.
