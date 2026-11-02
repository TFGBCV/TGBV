# ROADMAP — from single-page classifier to autonomous OSINT agent

Where the tool is today: cookie-scrape one page → classify with one local LLM → export static
reports. The plan below turns it into a longitudinal, multi-source, human-supervised monitoring
agent. Phases are ordered so each one is useful on its own; estimates assume the current
codebase (scraper / analyzer / lmstudio / report / pdf_report + Streamlit UI).

> **Status (Nov 2026) — second implementation pass shipped (v3):**
> ✅ Phase 0: data spine (`storage.py` v2: FTS5 search, entities, jobs, case files, κ stats) + review/triage UI
>   + **job engine** (`jobs.py`, background watchlist runs, in-memory cookie handoff) + **eval harness**
>   (`eval.py`: gold export → precision/recall/F1/severity scoring → `evals/latest.json` snapshots)
> ✅ Phase 1: watchlists + delta-aware scheduling (`monitor.py`, UI tab → background jobs) — *new source types & discovery queue still open*
> ✅ Phase 2: OSINT enrichment (`osint.py`) — embeddings, theme clusters, near-dup detection, **template fingerprints
>   (shingle-Jaccard, distinct-author counts)**, entity extraction **persisted per-item**, cadence, **burst
>   detection (σ vs page history)**, **cross-page duplicates**, amplification graph with **PageRank/betweenness
>   leaders** (GraphML)
> ✅ Phase 3: bounded agent loop (`agent.py`) with read-only tools (**reclassify lens, cluster_corpus,
>   entity_graph, queue_for_review**), structured case file (**observations/hypotheses/gaps** with
>   forced reflection), verifier pass (**lexicon-aware**), human-gated proposals — plus **verdict memory**
>   (`memory.py`: near-identical items inherit human-confirmed/rejected decisions, cross-page) and the
>   **lexicon safety net** (`lexicon.py`) with `lexicon_attention` verifier bait
> ✅ Phase 4: longitudinal + OSINT + **verdict-memory + human–AI agreement + per-category calibration**
>   sections in PDF/MD; κ/eval dashboard in the UI; burst-aware webhook/email alert digests
>   — *translations still open*
> ✅ Phase 5 (core): audit log, secrets-in-env, **optional SQLCipher/keyring encryption at rest**, **DOM
>   fixture smoke tests** (`tests/dom_smoke.py` + `tools/record_fixture.py`) + offline unit suite
>   (`tests/test_units.py`, 18/18). Still open: publication kit (signed exports), first-run wizard, error l10n.


## Guiding constraints (non-negotiable)

1. **Read-only.** The agent never likes, follows, comments, messages or otherwise acts on
   Facebook. Collection only.
2. **Nothing leaves the machine.** All models (LLM, embeddings) stay local via LM Studio.
3. **Human-in-the-loop.** The agent proposes; a human approves every impact step
   (new watch targets, case-file writes, external publication).
4. **Hard budgets.** Every run carries max_items, max_time, max_requests ceilings the agent
   cannot exceed by itself.
5. **Provenance first.** Every exported claim traces back to a SHA-256-anchored item.
   (Already partly true — keep it true.)

---

## Phase 0 — Reliable data spine (prerequisite for everything) · ~1 week

- **SQLite dataset layer** (`storage.py`): cases → runs → items → verdicts → screenshots.
  Replaces loose `output/*.json` files; enables history, deltas and review state.
  - Tables: `pages`, `posts`, `comments`, `verdicts(1..N per item)`, `human_review`,
    `jobs`, `audit_log`.
- **Job engine** (`jobs.py`): scrape / enrich / analyze / report as queued tasks with
  retries and progress, so long autonomous runs survive Streamlit refreshes (background
  worker thread or `streamlit run` + `subprocess`; no new infra needed).
- **Review UI v1**: confirm/reject each flag with a short note + reviewed-by/time.
  This is the floor for professional OSINT outputs — without it, sophistication is theatre.
- **Test harness**: ~30 hand-labelled items (from real runs) → precision/recall dashboard;
  every prompt change is measured, not vibes. `evals/label_XX.jsonl`.

## Phase 1 — Autonomous collection · ~2 weeks

- **More sources, same DOM strategy loosened from "page" to "view"**:
  - profiles, groups (read-only), hashtag/keyword search result feeds, comment deep threads.
- **Delta scraping**: snapshot pages (permalink + canonical URL + hash); re-runs collect only
  what is new/changed. Recurring monitoring needs ~10× less scraping.
- **Watchlists**: UI where a team registers `target ↔ cadence (daily/weekly) ↔ budget`.
 .scheduler inside the job engine wakes up, checks each target, records deltas.
- **Link discovery**: post permalinks, shared groups, author handles collected as *candidate*
  next targets → land in a **review queue** (Phase 0 review UI), never auto-scraped.
- Politeness layer: jittered delays, exponential backoff on 429, per-run request ceilings.

## Phase 2 — OSINT enrichment (all local/computed) · ~2–3 weeks

- **Embeddings, already available**: LM Studio exposes `text-embedding-nomic-embed-text-v1.5`.
  Use it for:
  - near-duplicate & template detection (MinHash/embedding similarity) → "same text, N
    accounts" = coordination signal,
  - clustering the corpus into themes/campaigns automatically,
  - "show me posts similar to this one" in the review UI.
- **Entity extraction** (structured-output LLM pass): persons, orgs, laws/treaties (CEDAW,
  مجلة الأحوال الشخصية), slurs/lexicon per locale, URLs. Stored as a label graph.
- **Network graph**: author ↔ page ↔ mention ↔ comment-reply edges; page/page amplification
  clusters. Render with networkx + matplotlib into the PDF report.
- **Forensics**: posting-cadence timelines, burst detection (e.g., coordinated wave after a
  CEDAW debate event), language distribution, engagement ratios.
- **Corpus cross-referencing**: claims found in one item that match/deny claims in other
  items → contradiction/confirmation notes for the reviewer.

## Phase 3 — The agentic loop · ~3–4 weeks

A bounded plan→act→reflect loop whose *only* capabilities are the local tools:

```
Goal ("investigate page X for GBV, weekly")
  └─ plan     → choose targets, budgets, lenses
  └─ act      → scrape deltas / expand comments on suspicious items / embed & cluster
  └─ reflect  → the LLM reviews its own flags (a second "verifier" prompt per batch)
              → re-reads items with widened context where confidence is low
  └─ report   → draft findings with evidence chains + open questions
  └─ stop/replan (budget check) ──┘
```

- **Agent tools exposed to the model (JSON-schema function calling):**
  `search_corpus(query)`, `get_item(id)`, `expand_comments(item_id)`, `reclassify(item_id,
  lens)`, `cluster_corpus()`, `entity_graph(entity)`, `queue_for_review(id, why)`,
  `propose_target(why)` — each implemented against the Phase 0 dataset layer.
- **Verifier pass**: every flagged item and a random sample of non-flagged items get a
  second classification with a "devil's advocate" prompt; disagreements → review queue.
- **Case-file memory**: the agent maintains a short structured case file (hypotheses,
  evidence, gaps) that it updates as it works; the reviewer sees it live.
- **Approval gates** (Hard rule 3): `propose_target` and anything writing beyond the case
  file must be clicked through by a human. The agent never stacks new pages to scrape on its
  own initiative.

## Phase 4 — Reporting & triage sophistication · ~2 weeks

- **Case-file PDFs**: longitudinal report across runs (what changed since last week:
  new items, new clusters, new actors), timeline + network sections added beside the
  current per-run report.
- **Severity triage inbox**: dashboard ordered by (severity × confidence × burst signal);
  bulk confirm/reject; keyboard shortcuts — designed for reviewer endurance.
- **Alerts**: on new high/critical verdicts or coordinated-burst detection →
  webhook/Matrix/email digest (still only *notification*, not automatic action).
- **Translations**: LLM draft translation of Darija evidence alongside the verbatim Arabic
  (clearly marked as machine translation).

## Phase 5 — Hardening for distribution · ~1 week

- **Security**: encrypted local dataset (SQLCipher or Fernet), cookies only in memory
  (already true) + optional OS keyring storage, audit log of every run/action.
- **Ops UX**: one-click LM Studio model check, first-run wizard, error l10n (FR/AR).
- **Resilience**: DOM-selector smoke tests against saved HTML snapshots (recorded fixtures)
  so Facebook layout changes are detected in CI without ever touching the live site.
- **Publication kit**: signed PDF exports, per-run reproducibility (config hash in report).

---

## Suggested milestone order

| Milestone | Phases | Outcome |
|---|---|---|
| M1 — "the tool grows memory" | 0 | cases, reviews, evals — professional baseline |
| M2 — "watches pages on its own" | 1 (+0 deltas) | recurring hands-off collection |
| M3 — "understands the corpus" | 2 | dupes, clusters, graphs, cadence |
| M4 — "reasons about the case" | 3 | verifier + agent loop with approval gates |
| M5 — "lab-ready distribution" | 4–5 | dashboards, alerts, encrypted case files |

## Risk register

- **Facebook ToS / account bans** — scraping automation is prohibited; the delta + budget
  design keeps run volume low and the watchlists explicit. Dedicated accounts only.
- **LLM drift & hallucination in agentic mode** — verifier pass, evaluation harness,
  confidence floors; nothing enters the case file without an anchored item.
- **Reviewer burnout on higher throughput** — triage inbox + blur-in-preview options;
  team rotation guidance already in the app's first-run notes.
- **Over-automation legitimacy risk for NGO evidence** — approval gates and audit logs keep
  the human chain-of-custody intact; this is a feature, not a limitation.
