========================================================
 Facebook GBV & Hate-Speech Monitor — portable edition
========================================================

WHAT THIS IS
  A survey/documentation tool for civil-society teams: collect posts from a
  Facebook page, have the AI engine bundled with this tool classify gender-based
  cyberviolence & hate speech against women (fully local — nothing is sent to a
  cloud service), and export CSV / JSON / Markdown / PDF documentation.

HOW TO RUN
  1) Extract the WHOLE folder (all .dll files stay next to the .exe).
  2) Double-click GBVMonitor.exe.
  3) Wait 15–90 s: the console shows the local model loading — the interface
     then opens itself in your browser.
  Nothing else to install: no Python, no LM Studio, no drivers, no account.

WHAT RUNS IN THE BACKGROUND
  * a local model engine (llama.cpp) serving the chat+vision model included in
    models/ (chat GGUF + mmproj vision tower — text AND image screening);
  * a small local embedding engine used by the OSINT enrichment;
  * a headless Chromium for page collection.
  Everything stays on 127.0.0.1 and dies with the app (closing the console
  window also stops the engines).

SCHEDULER MODE (optional, without the window)
  GBVMonitor.exe monitor run-due --enrich --verify   one pass over watchlists
  GBVMonitor.exe monitor watch --interval 900        continuous loop
  GBVMonitor.exe monitor add-watch <page-url> --label "Label" --cadence-hours 24
  (Use Task Scheduler to run these regularly.)

FILES & FOLDERS
  output\       reports (CSV/JSON/MD/PDF), screenshots, case dataset (SQLite)
  models\       the AI model (chat + mmproj vision tower; swap .gguf files to
                change engines — see README-EXE.md)
  runtime\      the engine binaries (cpu = works everywhere, vulkan = GPU)
  ms-playwright\  bundled Chromium for browsing Facebook pages
  logs\         engine logs (diagnostics if something fails)

REQUIREMENTS
  Windows 10/11 x64 · ~8 GB free RAM for the shipped model (or swap in a
  smaller .gguf) · 10 GB disk.

LEGAL, ETHICS & SAFETY NOTES
  * Facebook's Terms prohibit automated collection — use a DEDICATED account,
    not your personal one; your account may be restricted.
  * Your session cookies (c_user/xs) give full access to that Facebook
    account: only in memory for this session, never written by this tool.
    Never share them.
  * AI output is a first-pass flag, NOT proof: every flagged item must be
    reviewed by a human before being used as evidence.
  * Authors are pseudonymised by default. Content may contain third-party
    personal data — protect your export folder and follow your organisation's
    data-protection rules.
  * Reviewing this material can be distressing: plan for support.
