import json
import os
import re
import time
from datetime import datetime

import pandas as pd
import streamlit as st

from analyzer import CATEGORIES, SEVERITIES, PROMPT_SHA, analyze_items, flatten
from local_llm import is_bundled
from lmstudio import LMStudioError, create_client, llm_models, pick_reasoning
from pdf_report import build_pdf_report
from report import build_report
from scraper import ScrapeError, scrape_page

import agent as agent_mod
import alerts
import eval as eval_mod
import jobs as jobs_mod
import memory as memory_mod
import osint as osint_mod
import storage
import vision as vision_mod

OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)
jobs_mod.ensure_worker()  # daemon worker for background watchlist jobs

IN_DOCKER = os.environ.get("GBV_IN_DOCKER", "") == "1"

st.set_page_config(page_title="GBV Cyberviolence Monitor", page_icon="🛡️", layout="wide")
st.title("🛡️ Facebook GBV & Hate-Speech Monitor")
st.caption("Collect posts from a Facebook page, classify gender-based cyberviolence against women with a "
           + ("local AI model bundled inside this app" if is_bundled()
              else "local LM Studio model (nothing is sent to a cloud AI)")
           + ", and export documentation.")

with st.expander("⚠️ Read first: legal, safety and ethics notes", expanded=False):
    st.markdown(
        """
- **Facebook's Terms prohibit automated collection.** Scraping can get the account restricted or banned. Use a **dedicated account**, not your personal one, and collect only what you need.
- **Your session cookies (`c_user`, `xs`) give full access to that account.** They are only held in memory for this session and never written to disk by this tool. Never share them or commit them to git.
- Collected content may contain personal data of third parties. Authors are **pseudonymised by default** in outputs. Store files securely and follow your organisation's data-protection and consent rules (e.g. GDPR / Tunisian Law 2004-63).
- AI output is a **first-pass flag, not proof**. A human must review flagged items before they are used in reports, publications or complaints.
- Reviewing this material can be distressing. Plan for breaks and support for your team.
        """
    )

# ------------------------------------------------------------------ sidebar
BUNDLED = is_bundled()

with st.sidebar:
    st.header("AI engine" + (" (bundled)" if BUNDLED else " (LM Studio)"))
    if BUNDLED:
        st.caption(
            "The AI model ships **inside this app** and runs on this computer — no LM "
            "Studio, no cloud, no internet needed. If it appears stuck here, close the "
            "app and start **GBVMonitor.exe** again (its console window shows the details)."
        )
        base_url = os.environ.get("GBV_LLM_URL", "")
        token = ""
        lm = create_client("", "")
    else:
        if IN_DOCKER:
            st.caption("The model runs on **your computer**, not in this container. "
                       "1) Open **LM Studio** on the machine running this lab · 2) download a model · "
                       "3) start its server (**Developer tab → Start Server**, port 1234) · "
                       "4) press **Connect** below — your models appear here automatically.")
        default_url = os.environ.get("LMSTUDIO_URL", "") or (
            "http://host.docker.internal:1234" if IN_DOCKER else "http://localhost:1234")
        base_url = st.text_input("Server URL", value=default_url,
                                 help="LM Studio keeps its server on port 1234 by default.")
        token = st.text_input("API token (only if auth is enabled)", type="password", value=os.environ.get("LM_API_TOKEN", ""))
        lm = create_client(base_url, token)

    def refresh_models():
        try:
            st.session_state["lm_models"] = lm.list_models()
            st.session_state["lm_error"] = ""
        except LMStudioError as e:
            st.session_state["lm_models"] = []
            st.session_state["lm_error"] = str(e)

    if st.button("🔄 Connect / refresh models") or "lm_models" not in st.session_state:
        refresh_models()

    llms = llm_models(st.session_state.get("lm_models", []))
    model, model_info = None, None
    if st.session_state.get("lm_error"):
        st.error(st.session_state["lm_error"])
    elif not llms:
        st.info("Connected, but no language models are downloaded yet. Use the download box below.")
    else:
        keys = [m["key"] for m in llms]
        by_key = {m["key"]: m for m in llms}
        model = st.selectbox(
            "Model", keys,
            format_func=lambda k: f"{'🟢 ' if by_key[k].get('loaded_instances') else ''}{by_key[k].get('display_name', k)} ({k})",
        )
        model_info = by_key[model]
        q = (model_info.get("quantization") or {}).get("name") or "?"
        st.caption(f"{model_info.get('params_string') or '?'} · {q} · max context {model_info.get('max_context_length') or '?'}"
                   + (" · 👁 vision" if (model_info.get("capabilities") or {}).get("vision") else ""))

    ctx_len, reasoning_pref = 8192, "auto"
    if BUNDLED:
        st.caption("The bundled model is already loaded — it started with the app "
                   "(default context window 8192 tokens; change with `GBVMonitor.exe --ctx N`).")
    else:
        ctx_len = st.number_input("Context length (tokens)", 2048, 131072, 8192, step=1024,
                                  help="Used when the model is loaded. Larger = more RAM. 8192 fits ~4 items per call.")
        reasoning_pref = st.selectbox("Reasoning", ["auto", "off", "low", "medium", "high", "on"],
                                      help="'auto' turns reasoning off when the model supports it (faster, shorter output). Ignored if unsupported.")
    batch_size = st.slider("Items per request", 1, 12, 4, help="Smaller batches are more reliable with small local models.")
    passes = st.slider("Self-consistency passes", 1, 5, 1,
                       help="Classify every item N times (first pass at temperature 0, the rest at 0.7) and majority-vote. "
                            "Confidence then measures agreement between passes instead of the model's self-report. "
                            "Costs N× the requests — use 2–3 for sensitive decisions.")
    pseudonymize = st.checkbox("Pseudonymise author names in outputs", value=True)

    if not BUNDLED:
        if model and model_info is not None and not model_info.get("loaded_instances"):
            st.warning("Model is not loaded yet.")
            if st.button("⚡ Load model now"):
                with st.spinner("Loading model…"):
                    try:
                        res = lm.load_model(model, ctx_len)
                        st.success(f"Loaded in {res.get('load_time_seconds', '?')}s")
                        refresh_models()
                        st.rerun()
                    except LMStudioError as e:
                        st.error(str(e))

        with st.expander("⬇️ Download a model"):
            dl_model = st.text_input("Catalog ID or Hugging Face link", placeholder="e.g. ibm/granite-4-micro")
            dl_quant = st.text_input("Quantization (Hugging Face links only)", placeholder="Q4_K_M")
            if st.button("Download", disabled=not dl_model):
                try:
                    job = lm.download_model(dl_model.strip(), dl_quant.strip() or None)
                    if job.get("status") == "already_downloaded":
                        st.info("Already downloaded.")
                    else:
                        dbar = st.progress(0.0, text="Starting download…")

                        def on_status(s):
                            total = s.get("total_size_bytes") or 0
                            done = s.get("downloaded_bytes") or 0
                            frac = (done / total) if total else 0.0
                            speed = s.get("bytes_per_second")
                            txt = f"{s.get('status')} · {done / 1e9:.2f}/{total / 1e9:.2f} GB"
                            if speed:
                                txt += f" · {speed / 1e6:.1f} MB/s"
                            dbar.progress(min(max(frac, 0.0), 1.0), text=txt)

                        lm.wait_for_download(job["job_id"], progress=on_status)
                        dbar.empty()
                        st.success("Download complete.")
                        refresh_models()
                        st.rerun()
                except LMStudioError as e:
                    st.error(str(e))

    st.header("Browser")
    headless = st.checkbox("Headless browser", value=True, help="Untick to watch the browser (local runs only) when debugging.")

# ------------------------------------------------------------------ step 1: scrape
st.subheader("1. Collect posts")
c1, c2 = st.columns([2, 3])
with c1:
    page_url = st.text_input("Facebook page link", placeholder="https://www.facebook.com/somepage")
    max_posts = st.number_input("Max posts", 5, 1000, 50, step=5)
    include_comments = st.checkbox("Include comments (where most abuse happens)", value=True)
    deep_comments = st.checkbox("Open each post to load more comments (slower)", value=False, disabled=not include_comments)
    capture_shots = st.checkbox("Screenshot each post (embedded in the PDF export)", value=True)
with c2:
    cookies_raw = st.text_area(
        "Facebook cookies",
        height=190,
        placeholder="Paste cookies exported as JSON (Cookie-Editor → Export → JSON), Netscape cookies.txt, or a header string like: c_user=...; xs=...; datr=...",
    )

if st.button("🔎 Scrape page", type="primary", disabled=not (page_url and cookies_raw)):
    bar = st.progress(0.0, text="Starting…")
    try:
        data = scrape_page(
            page_url, cookies_raw, max_posts=int(max_posts), include_comments=include_comments,
            deep_comments=deep_comments, headless=headless, capture_screenshots=capture_shots,
            progress=lambda msg, f: bar.progress(min(max(f, 0.0), 1.0), text=msg),
        )
        slug = re.sub(r"[^a-zA-Z0-9]+", "_", page_url.split("facebook.com/")[-1]).strip("_")[:40] or "page"
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(OUT_DIR, f"{slug}_{stamp}_posts.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        st.session_state["data"] = data
        st.session_state["slug_stamp"] = f"{slug}_{stamp}"
        st.session_state.pop("rows", None)
        bar.empty()
        if data["meta"]["post_count"] == 0:
            st.warning("No posts extracted. Facebook's layout may differ from what the extractor expects. Diagnostics (counts only, no post text) are below; a screenshot of what the browser saw is saved too.")
            dbg = data["meta"].get("debug", {})
            if dbg.get("screenshot") and os.path.exists(dbg["screenshot"]):
                st.image(dbg["screenshot"], caption="What the browser saw", use_container_width=True)
            st.json(dbg.get("diagnostics", dbg))
        else:
            n_shots = sum(1 for p in data['posts'] if p.get('screenshot'))
            extra = f" · {n_shots} screenshots saved under `output/shots/`" if n_shots else ""
            st.success(f"Collected {data['meta']['post_count']} posts and {data['meta']['comment_count']} comments → saved to `{path}`{extra}")
    except ScrapeError as e:
        bar.empty()
        st.error(str(e))
    except Exception as e:
        bar.empty()
        st.error(f"Scraping failed: {type(e).__name__}: {e}")

data = st.session_state.get("data")
if data:
    with st.expander(f"Collected data ({data['meta']['post_count']} posts, {data['meta']['comment_count']} comments)"):
        st.dataframe(
            pd.DataFrame([{"author": p["author"], "text": p["text"][:200], "comments": len(p["comments"]), "link": p["permalink"]} for p in data["posts"]]),
            use_container_width=True,
        )
    st.download_button("⬇️ Raw posts (JSON)", json.dumps(data, ensure_ascii=False, indent=2), file_name=f"{st.session_state['slug_stamp']}_posts.json", mime="application/json")

    # -------------------------------------------------------------- step 2: analyse
    st.subheader("2. Analyse with a local model")
    v1, v2, v3 = st.columns(3)
    with v1:
        use_verifier = st.checkbox("Adversarial verifier pass (recommended)", value=True,
                                   help="A second 'devil's advocate' classification of every flagged item + a sample of the others. Disagreements land in the review inbox.")
    with v2:
        use_enrich = st.checkbox("OSINT enrichment after analysis", value=True,
                                 help="Embeddings, theme clusters, near-duplicate & template detection, entity extraction, amplification graph. Runs fully locally.")
    with v3:
        use_vision = st.checkbox("Multimodal image screening", value=False,
                                 help="Flagged posts' images are screened for NCII/sextortion-type "
                                      "visual content by the bundled vision-capable model (Qwen2.5-VL — "
                                      "no setup needed in the packaged exe) or, with LM Studio, by the "
                                      "model set in GBV_VISION_MODEL. Conservative triage leads only — "
                                      "a human must confirm.")
    if st.button("🤖 Detect gender-based violence & hate speech", type="primary", disabled=not model):
        items = flatten(data, pseudonymize=pseudonymize)
        if not items:
            st.warning("Nothing to analyse.")
        else:
            bar = st.progress(0.0, text="Preparing model…")
            try:
                # ---- persist to the case dataset (SQLite)
                page_id = storage.get_or_create_page(data["meta"].get("source_url", ""), "")
                run_id = storage.start_run(page_id, notes="UI run")
                storage.upsert_items(page_id, run_id, items)
                st.session_state["page_id"] = page_id
                st.session_state["run_id"] = run_id

                if not BUNDLED and not (model_info or {}).get("loaded_instances"):
                    lm.load_model(model, ctx_len)  # applies the chosen context length
                    refresh_models()

                # ---- verdict memory first: items whose text matches something a
                # human already reviewed inherit that decision (provenance-stamped
                # 'verdict-memory' verdicts; skipped by the model pass).
                seeded = memory_mod.seed_from_confirmed(
                    page_id, run_id, items, lm,
                    progress=lambda m, f: bar.progress(min(max(f, 0.0), 1.0), text=f"Verdict memory: {m}"))
                st.session_state["seeded_n"] = len(seeded)
                if seeded:
                    st.info(f"♟️ Verdict memory: **{len(seeded)}** item(s) inherit a human review "
                            "decision from an earlier identical/near-identical item — the model is skipped for them.")
                pending_items = [it for it in items if f"{page_id}:{it['sha256']}" not in set(seeded)]
                seeded_names = set(seeded)

                if pending_items:
                    rows = analyze_items(
                        pending_items, lm, model, batch_size, reasoning=pick_reasoning(model_info, reasoning_pref),
                        passes=passes,
                        progress=lambda m, f: bar.progress(min(max(f, 0.0), 1.0), text=m))
                else:
                    rows = []
                    st.success("Everything already covered by verdict memory — no model calls needed.")
                # merge seeded verdicts into the display rows (read back from storage)
                seeded_rows: list[dict] = []
                if seeded:
                    for d in storage.current_verdicts(page_id, run_id=run_id):
                        if d["item_key"] not in seeded_names:
                            continue
                        seeded_rows.append({
                            "id": d["item_key"], "kind": d["kind"], "parent_id": "", "author": d["author"],
                            "text": d["text"], "sha256": d["sha256"], "permalink": d["permalink"],
                            "time_text": d.get("time_text", ""), "collected_at": d.get("collected_at", ""),
                            "screenshot": d.get("screenshot", "") or "", "media": d.get("media", []),
                            "is_violation": d["is_violation"], "categories": d["categories"],
                            "severity": d["severity"], "target": d.get("target", "unclear"),
                            "language": d.get("language", ""), "evidence": d.get("evidence", ""),
                            "explanation": d.get("explanation", ""), "confidence": d.get("confidence", 0),
                            "analysis_error": "", "inherited_from": d.get("inherited_from", ""),
                        })
                rows = rows + seeded_rows
                st.session_state["rows"] = rows
                st.session_state["pseudo"] = pseudonymize
                if BUNDLED:
                    model_used = f"{os.environ.get('GBV_MODEL_FILE', model)} (local, bundled engine)"
                else:
                    model_used = f"{model} (local, LM Studio)"
                st.session_state["model_used"] = model_used
                if rows:
                    storage.add_verdicts(run_id, [r for r in rows if r["id"] not in seeded_names],
                                         key_of=lambda r: f"{page_id}:{r['sha256']}",
                                         model=st.session_state["model_used"], prompt_sha=PROMPT_SHA)

                if use_verifier and rows:
                    bar.progress(0.0, text="Verifier pass…")
                    reviewed = agent_mod.verifier(
                        lm, [r for r in rows if r["id"] not in seeded_names], model, run_id=run_id,
                        key_of=lambda r: f"{page_id}:{r['sha256']}",
                        progress=lambda m, f: bar.progress(min(max(f, 0.0) * 1.0, 1.0), text=f"Verifier: {m}"))
                    by_id = {r["id"]: r for r in reviewed}
                    rows = [by_id.get(r["id"], r) for r in rows]
                    st.session_state["rows"] = rows

                if use_vision and rows:
                    bar.progress(0.0, text="Vision screening…")
                    try:
                        vis = vision_mod.vision_screen(
                            rows, cookies_raw, lm,
                            progress=lambda m, f: bar.progress(min(max(f, 0.0), 1.0), text=f"Vision: {m}"))
                        if vis:
                            st.session_state["vision"] = vis
                            st.info(f"👁 Vision screening complete: {len(vis)} image(s) screened "
                                    "(results below; leads only — confirm them yourself).")
                        else:
                            st.session_state.pop("vision", None)
                            st.caption("Vision screening: nothing to screen (no flagged post carried "
                                       "downloadable media, or no vision-capable model is available — "
                                       "with LM Studio, set GBV_VISION_MODEL).")
                    except Exception as e:
                        st.warning(f"Vision screening skipped: {type(e).__name__}: {e}")

                if use_enrich:
                    bar.progress(0.0, text="OSINT enrichment…")
                    st.session_state["osint"] = osint_mod.run_osint(run_id, page_id, lm, llm_model=model)
                    st.session_state.pop("case_loaded", None)
                bar.empty()
            except LMStudioError as e:
                bar.empty()
                st.error(str(e))
            except Exception as e:
                bar.empty()
                st.error(f"Analysis failed: {type(e).__name__}: {e}")
    if not model:
        st.info("The bundled model is missing or could not be contacted — restart "
                "GBVMonitor.exe and check its console window."
                if BUNDLED else
                "Start the LM Studio server and pick a model in the sidebar to enable analysis.")

# ------------------------------------------------------------------ step 3: results
rows = st.session_state.get("rows")
if rows:
    st.subheader("3. Results")
    df = pd.DataFrame(rows)
    flagged = df[df["is_violation"]]
    errs = int((df["analysis_error"] != "").sum())
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Items analysed", len(df))
    m2.metric("Flagged", len(flagged))
    m3.metric("High / critical", int(flagged["severity"].isin(["high", "critical"]).sum()))
    m4.metric("Analysis errors", errs)
    m5.metric("♟️ Memory-inherited", int(st.session_state.get("seeded_n", 0)),
              help="Items whose verdict came from a human review of an identical item instead of the model.")

    if len(flagged):
        left, right = st.columns(2)
        with left:
            st.markdown("**By category**")
            st.bar_chart(flagged.explode("categories")["categories"].value_counts())
        with right:
            st.markdown("**By severity**")
            st.bar_chart(flagged["severity"].value_counts().reindex(SEVERITIES[1:]).fillna(0))

    f1, f2, f3 = st.columns(3)
    only_flagged = f1.checkbox("Only flagged items", value=True)
    sev_filter = f2.multiselect("Severity", SEVERITIES[1:], default=SEVERITIES[1:])
    cat_filter = f3.multiselect("Category", list(CATEGORIES))
    view = df[df["is_violation"]] if only_flagged else df
    if only_flagged:
        view = view[view["severity"].isin(sev_filter)]
    if cat_filter:
        view = view[view["categories"].apply(lambda cs: any(c in cs for c in cat_filter))]
    show = view[["id", "kind", "severity", "categories", "target", "language", "confidence", "text", "evidence", "explanation", "permalink"]].copy()
    show["categories"] = show["categories"].apply(", ".join)

    def _badges(r) -> str:
        marks = []
        if r.get("inherited_from"):
            marks.append("♟️ inherited")
        if r.get("lexicon_attention"):
            marks.append("🔍 lexicon-attention")
        elif r.get("lexicon"):
            marks.append("🔍 lexicon-hit")
        if str(r.get("verdict_model", "")).endswith("[verifier]"):
            marks.append("⚖️ verifier")
        if str(r.get("verdict_model", "")).endswith("[lens]"):
            marks.append("🔭 lens")
        if r.get("media"):
            marks.append(f"🖼 {len(r['media'])}")
        if st.session_state.get("vision", {}).get(r["id"]):
            marks.append(f"👁 risk={st.session_state['vision'][r['id']].get('risk', '?')}")
        return " · ".join(marks)

    show["provenance"] = view.apply(_badges, axis=1)
    st.dataframe(show, use_container_width=True, height=420)

    if st.session_state.get("vision"):
        with st.expander("👁 Multimodal screening results (flagged posts' images)"):
            vis = st.session_state["vision"]
            st.dataframe(pd.DataFrame([{"item": k, **v} for k, v in vis.items()]),
                         use_container_width=True)
            st.caption("Vision output is a conservative triage lead produced by a local model. "
                       "It cannot identify people or determine consent — a human decides.")

    meta = data["meta"]
    pseudo = st.session_state.get("pseudo", True)
    model_used = st.session_state.get("model_used", "unknown")

    extra: dict = {}
    page_id = st.session_state.get("page_id")
    run_id = st.session_state.get("run_id")
    if page_id is not None and run_id is not None:
        extra["longitudinal"] = storage.longitudinal(page_id, run_id)
        extra["quality"] = storage.human_ai_agreement(page_id)
        cat_quality = storage.category_review_stats(page_id, min_n=3)
        if cat_quality:
            extra["category_quality"] = cat_quality
    if st.session_state.get("seeded_n"):
        extra["memory"] = {"seeded": st.session_state["seeded_n"]}
    if st.session_state.get("vision"):
        extra["vision"] = st.session_state["vision"]
    if st.session_state.get("osint"):
        extra["osint"] = st.session_state["osint"]

    md = build_report(meta, rows, model_used, pseudo, extra)
    csv_df = df.copy()
    csv_df["categories"] = csv_df["categories"].apply(", ".join)
    base = st.session_state["slug_stamp"]

    pdf_path = os.path.join(OUT_DIR, f"{base}_report.pdf")
    pdf_error = ""
    try:
        pdf_bytes = build_pdf_report(meta, rows, model_used, pseudo, extra)
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
    except Exception as e:
        pdf_error = f"{type(e).__name__}: {e}"

    for name, content in [
        (f"{base}_analysis.csv", csv_df.to_csv(index=False)),
        (f"{base}_analysis.json", json.dumps(rows, ensure_ascii=False, indent=2)),
        (f"{base}_report.md", md),
    ]:
        with open(os.path.join(OUT_DIR, name), "w", encoding="utf-8-sig" if name.endswith(".csv") else "utf-8") as f:
            f.write(content)
    d1, d2, d3, d4 = st.columns(4)
    d1.download_button("⬇️ Report (PDF)", pdf_bytes, file_name=f"{base}_report.pdf", mime="application/pdf",
                       disabled=not pdf_bytes, help="Structured PDF: cover, summary, charts, detailed records, sign-off.")
    d2.download_button("⬇️ Report (Markdown)", md, file_name=f"{base}_report.md", mime="text/markdown")
    d3.download_button("⬇️ Analysis (CSV)", csv_df.to_csv(index=False).encode("utf-8-sig"), file_name=f"{base}_analysis.csv", mime="text/csv")
    d4.download_button("⬇️ Analysis (JSON)", json.dumps(rows, ensure_ascii=False, indent=2), file_name=f"{base}_analysis.json", mime="application/json")
    if pdf_error:
        st.warning(f"PDF report could not be generated: {pdf_error}")
    st.caption(f"Files also saved in `{OUT_DIR}/`.")

# ------------------------------------------------------------------ step 4+: ops tabs
page_id = st.session_state.get("page_id")

tabs = st.tabs(["🗂 Review & triage", "📈 Quality & eval", "⏱ Watchlists", "🧠 Autonomous agent", "🗃 Case dataset"])
with tabs[0]:
    if page_id is None:
        st.info("Run an analysis first — the review inbox fills from the case dataset.")
    else:
        pending = storage.pending_review(page_id)
        stats = storage.review_stats(page_id)
        agr = storage.human_ai_agreement(page_id)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Awaiting your review", len(pending))
        c2.metric("Confirmed", stats["confirmed"])
        c3.metric("Rejected (false positives)", stats["rejected"])
        c4.metric("Human–AI κ", agr["kappa"] if agr["kappa"] is not None else "—",
                  help=f"Cohen's κ between the AI flag and human decisions over {agr['n']} reviewed items. "
                       "Higher = the machine agrees with you beyond chance; below ~0.4 treat the machine as a strong assistant only.")
        if not pending:
            st.success("Nothing waiting for review. ✅")
        for row in pending[:25]:
            badge_bits = []
            if row.get("inherited_from"):
                badge_bits.append("♟️ inherited human verdict")
            if row.get("verifier_summary"):
                badge_bits.append("⚖️ verifier")
            if row.get("media"):
                badge_bits.append(f"🖼 {len(row['media'])} image(s)")
            badge = f" · *{' · '.join(badge_bits)}*" if badge_bits else ""
            with st.expander(f"[{row['severity']}] {' · '.join(row['categories'] or [])} — {row['author']} "
                             f"({row['kind']}, conf {row['confidence']:.2f}){badge}"):
                st.write(row["text"][:1500])
                if row["evidence"]:
                    st.markdown(f"**Evidence:** “{row['evidence']}”")
                st.markdown(f"**Machine verdict:** {row['explanation']}")
                if str(row.get("verdict_model", "")).endswith("[verifier]"):
                    st.markdown(f"**Verifier note:** {row.get('verifier_summary', '')}")
                elif row.get("inherited_from"):
                    st.markdown(f"**Inherited from:** `{row['inherited_from']}` "
                                "(decision comes from your earlier review of an identical item)")
                n1, n2, n3 = st.columns([1, 1, 3])
                note = n3.text_input("Reviewer note", key=f"note_{row['item_key']}")
                if n1.button("✅ Confirm violation", key=f"ok_{row['item_key']}"):
                    storage.set_review(row["item_key"], "confirm", note=note)
                    storage.audit("human", "review_confirm", row["item_key"])
                    st.rerun()
                if n2.button("⛔ Reject (not a violation)", key=f"rej_{row['item_key']}"):
                    storage.set_review(row["item_key"], "reject", note=note)
                    storage.audit("human", "review_reject", row["item_key"])
                    st.rerun()
        if pending:
            st.caption("Confirming locks the item as reviewed GBV evidence; rejecting creates a "
                       "calibration signal for future prompt tuning.")

with tabs[1]:
    if page_id is None:
        st.info("Quality metrics and prompt regression scoring unlock after you review a few items.")
    else:
        agr = storage.human_ai_agreement(page_id)
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("Human–AI κ", agr["kappa"] if agr["kappa"] is not None else "—",
                  help="Cohen's κ between the latest AI verdict and your review decisions.")
        q2.metric("Reviewed items", agr["n"])
        q3.metric("AI misses (false neg.)", agr.get("false_neg", "—"),
                  help="Clean per the machine, confirmed by you — the most dangerous quadrant.")
        q4.metric("AI fines (false pos.)", agr.get("false_pos", "—"),
                  help="Flagged by the machine, rejected by you.")
        if agr.get("kappa") is not None and agr["kappa"] < 0.4:
            st.warning("κ below 0.4 — treat flags as hints only and re-read the model's "
                       "explanations on thresholds before reporting them.")
        cat_q = storage.category_review_stats(page_id, min_n=2)
        if cat_q:
            st.markdown("**Per-category calibration** (lower precision = the model over-flags that label)")
            st.dataframe(pd.DataFrame(cat_q), use_container_width=True, hide_index=True)
        with st.expander("🧪 Prompt regression scoring (eval harness)", expanded=False):
            st.caption("Your confirm/reject decisions are the labelled dataset. Scoring runs the current "
                       "model over them (locally) and stores a result snapshot — so prompt/model changes are measured, not vibes.")
            if st.button("🧪 Score current model against reviews", disabled=not model):
                gold_path = os.path.join("evals", "gold.jsonl")
                try:
                    eval_mod.export_gold(page_id)
                except Exception as e:
                    st.error(f"Gold export failed: {e}")
                barq = st.progress(0.0, text="Scoring…")
                try:
                    res = eval_mod.run_eval(lm, model, page_id=page_id,
                                            gold_path=gold_path,
                                            batch_size=batch_size,
                                            progress=lambda m, f: barq.progress(min(max(f, 0.0), 1.0), text=m))
                    st.session_state["eval_result"] = res
                except Exception as e:
                    st.error(f"Eval failed: {type(e).__name__}: {e}")
                barq.empty()
            res = st.session_state.get("eval_result")
            if res and not res.get("error"):
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Precision", res.get("precision", "—"))
                k2.metric("Recall", res.get("recall", "—"))
                k3.metric("F1", res.get("f1", "—"))
                k4.metric("Severity exact", res.get("severity_exact_rate", "—"))
                st.caption(f"over {res.get('n_gold')} reviewed items · prompt_sha `{res.get('prompt_sha')}` · "
                           f"results: evals/latest.json")
            hist = eval_mod.history()
            if hist:
                st.markdown("**Past eval runs**")
                st.dataframe(pd.DataFrame(hist), use_container_width=True, hide_index=True)

with tabs[2]:
    if "wl_form" not in st.session_state:
        st.session_state["wl_form"] = {}
    with st.form("add_watchlist"):
        st.markdown("Register pages to re-scrape automatically. The scheduler runs delta "
                    "collection — only content that is new since the last snapshot.")
        c1, c2 = st.columns(2)
        wurl = c1.text_input("Page URL", key="wl_url")
        wlabel = c2.text_input("Label (optional)", key="wl_label")
        c1, c2, c3 = st.columns(3)
        wcad = c1.selectbox("Cadence", ["6 h", "12 h", "24 h (daily)", "3 days"], index=2)
        wmax = c2.number_input("Max posts per visit", 5, 300, 50)
        wdeep = c3.checkbox("Deep comments", value=False)
        if st.form_submit_button("Add to watchlist"):
            if not wurl.startswith("http"):
                st.error("Enter a valid URL.")
            else:
                hours = {"6 h": 6, "12 h": 12, "24 h (daily)": 24, "3 days": 72}[wcad]
                wid = storage.add_watchlist(wurl, wlabel, hours, int(wmax), True, wdeep)
                storage.audit("human", "watchlist_add", wurl)
                st.success(f"Watchlist #{wid} registered.")

    watches = storage.list_watchlists()
    if watches:
        df = pd.DataFrame([{"id": w["id"], "active": "🟢" if w["active"] else "⚪", "label": w["label"],
                            "url": w["url"], "cadence_h": w["cadence_hours"],
                            "next due (UTC)": w["next_due"]} for w in watches])
        st.dataframe(df, use_container_width=True, hide_index=True)
        tw, td = st.columns(2)
        with tw.form("toggle_wl"):
            tgl_id = st.number_input("Watchlist # to toggle active/inactive", 0, 9999, 0)
            tgl_btn = st.form_submit_button("Toggle")
        if tgl_btn and tgl_id:
            cur = next((w for w in watches if w["id"] == int(tgl_id)), None)
            if cur:
                storage.toggle_watchlist(int(tgl_id), not cur["active"])
                st.rerun()
        with td.form("del_wl"):
            del_id = st.number_input("Watchlist # to delete", 0, 9999, 0)
            del_btn = st.form_submit_button("Delete")
        if del_btn and del_id:
            storage.delete_watchlist(int(del_id))
            st.rerun()

        with st.form("process_watchlists"):
            cookies_ok = bool(cookies_raw or os.environ.get("GBV_FB_COOKIES"))
            st.caption("Processing needs cookies (paste them in step 1 above, or set GBV_FB_COOKIES for env use). "
                       "Cookies travel to the worker through memory only — never to the database or disk.")
            enrich_and_verify = st.checkbox("Enrichment + verifier on scheduled runs", value=True)
            go = st.form_submit_button("▶️ Process all due watchlists now")
        if go:
            if not cookies_ok:
                st.error("No cookies available. Paste them above first.")
            else:
                jobs_mod.ensure_worker()
                jid = jobs_mod.enqueue_run_due(
                    {"lm_url": base_url, "model": model or "", "batch_size": 2,
                     "enrich": enrich_and_verify, "verify": enrich_and_verify},
                    {"cookies": cookies_raw or os.environ.get("GBV_FB_COOKIES", "")})
                st.success(f"Queued background job #{jid}. It keeps running across "
                           "page reloads — poll its status below.")
        job_rows = storage.list_jobs()
        if job_rows:
            jot = pd.DataFrame([{"job": j["id"], "kind": j["kind"], "status": j["status"],
                                 "progress": f"{int((j.get('progress') or 0) * 100)}%",
                                 "message": (j.get("message") or "")[:120],
                                 "updated": (j.get("updated_at") or "")[:19]}
                                for j in job_rows])
            st.markdown("**Background jobs** (queued → running → done/failed)")
            st.dataframe(jot, use_container_width=True, hide_index=True)
            acb, acs = st.columns(2)
            if acs.button("🔄 Refresh job status"):
                st.rerun()
            if acb.checkbox("Auto-refresh every 5 s while jobs run", value=False):
                if any(j["status"] in ("queued", "running") for j in job_rows):
                    time.sleep(5)
                    st.rerun()
            if any(j["status"] == "failed" for j in job_rows):
                fails = [j for j in job_rows if j["status"] == "failed"]
                with st.expander(f"❌ {len(fails)} failed job(s) — details"):
                    for j in fails:
                        st.code((j.get("result") or "")[:2000])

with tabs[3]:
    if page_id is None:
        st.info("The agent investigates pages that are already in the case dataset (run a scrape first).")
    else:
        n_runs = len(storage.list_runs(page_id))
        st.markdown(f"**Target:** case of page #{page_id} · {n_runs} run(s) recorded")
        goal = st.text_input(
            "Investigation goal",
            placeholder="e.g. map who amplifies anti-CEDAW templates and flag emerging coordination")
        c1, c2, c3 = st.columns(3)
        max_steps = c1.slider("Max agent steps", 4, 30, 10,
                              help="Hard ceiling on tool calls — enforced in code, not by the model.")
        max_expand = c2.slider("Comment expansions", 0, 5, 1)
        max_seconds = c3.selectbox("Wall-clock budget", ["2 min", "5 min", "10 min"], index=1)
        if st.button("🧠 Run autonomous investigation", type="primary", disabled=not model or not goal):
            budget = agent_mod.AgentBudget(
                max_steps, {"2 min": 120, "5 min": 300, "10 min": 600}[max_seconds], max_expand)
            with st.spinner("Agent working (read-only, budgeted)…"):
                log, budget, stop = agent_mod.agent_run(
                    page_id, goal, lm, model, budget, batch_size=batch_size,
                    progress=None)
            st.session_state["agent_log"] = log
            st.session_state["agent_stop"] = stop
            st.rerun()
        if "agent_log" in st.session_state:
            st.write(f"**Stopped:** {st.session_state.get('agent_stop')}")
            for i, entry in enumerate(st.session_state["agent_log"], 1):
                call = entry.get("call", {})
                with st.expander(f"step {i} — {call.get('tool')}"):
                    st.json(call)
        with st.expander("Case file (agent memory)", expanded="agent_log" in st.session_state):
            cf = storage.case_file_get(page_id)
            try:
                open_n = sum(1 for h in cf.get("hypotheses", []) if h.get("status") == "open")
                sup_n = sum(1 for h in cf.get("hypotheses", []) if h.get("status") == "supported")
                ref_n = sum(1 for h in cf.get("hypotheses", []) if h.get("status") == "refuted")
                b1, b2, b3, b4 = st.columns(4)
                b1.metric("Observations", len(cf.get("observations", [])))
                b2.metric("Open hypotheses", open_n)
                b3.metric("Supported", sup_n)
                b4.metric("Refuted", ref_n)
            except Exception:
                pass
            for h in cf.get("hypotheses", []):
                st.markdown(f"- **[{h.get('status','open')}]** {h.get('text','')}"
                            + (f" — `{h.get('anchor','')}`" if h.get("anchor") else ""))
            with st.expander("Observations", expanded=bool(cf.get("observations")) and not cf.get("hypotheses")):
                for o in (cf.get("observations") or [])[-40:]:
                    body = o.get("text", "") if isinstance(o, dict) else str(o)
                    anchor = o.get("anchor", "") if isinstance(o, dict) else ""
                    st.markdown(f"- {body}" + (f" — `{anchor}`" if anchor else ""))
            if cf.get("gaps"):
                st.markdown("**Open gaps**")
                for g in cf["gaps"][-10:]:
                    st.markdown(f"- {g.get('text', g) if isinstance(g, dict) else g}")
            if cf.get("legacy"):
                st.markdown("**Legacy free-text case file**")
                st.markdown(cf["legacy"])

        st.divider()
        st.markdown("### 🔔 Pending proposals (require your decision)")
        props = storage.list_proposals("pending")
        if not props:
            st.caption("No proposals waiting.")
        for p in props:
            c1, c2, c3 = st.columns([4, 1, 1])
            c1.write(f"**{p['type']}** — {p['value']}\n\n> {p['why']}")
            if c2.button("Approve", key=f"pa_{p['id']}"):
                storage.decide_proposal(p["id"], "approved", "ui")
                st.rerun()
            if c3.button("Dismiss", key=f"pd_{p['id']}"):
                storage.decide_proposal(p["id"], "dismissed", "ui")
                st.rerun()
        st.caption("Approved targets still require you to add them to a watchlist manually — "
                   "approval only unlocks the recommendation, it never triggers scraping.")

with tabs[4]:
    if page_id is None:
        st.info("Case dataset statistics appear once a run has been analysed.")
    else:
        runs = storage.list_runs(page_id)
        df = pd.DataFrame([{"run": r["id"], "started": r["started_at"][:19], "status": r["status"],
                            "posts": r["post_count"], "comments": r["comment_count"],
                            "model": r["model"] or ""} for r in runs])
        st.markdown(f"**Runs recorded for page #{page_id}**")
        st.dataframe(df, use_container_width=True, hide_index=True)
        # OSINT / coordination summary of the most recent enrichment
        if run_id := st.session_state.get("run_id"):
            o = osint_mod.get_osint(run_id) or {}
            if o:
                oa, ob, oc = st.columns(3)
                oa.metric("Template clusters", len(o.get("templates") or []),
                          help="Copy-paste template groups; multi-author ones point at coordination.")
                ob.metric("Multi-author templates", o.get("template_groups_multi_author", 0))
                oc.metric("Cross-page duplicates", len(o.get("cross_page") or []),
                          help="Byte-identical posts also seen on other watched pages.")
                expl = o.get("amplifiers") or {}
                if expl.get("pagerank_leaders"):
                    st.markdown("**Top amplification nodes (PageRank / betweenness):")
                    st.dataframe(pd.DataFrame([
                        {"node": n["node"], "pagerank": n["pagerank"]}
                        for n in expl["pagerank_leaders"]]), use_container_width=True, hide_index=True)
                if o.get("burst"):
                    st.warning(f"⚡ Burst detected: {o['burst']['latest']} flagged items in the newest run "
                               f"vs baseline {o['burst']['baseline_mean']} (+{o['burst']['sigma']}σ "
                               f"over {o['burst']['baseline_runs']} prior runs).")
                if o.get("graphml_path"):
                    st.caption(f"Amplification graph (GraphML for Gephi/Cytoscape): `{o['graphml_path']}`")
        st.caption(f"SQLite dataset: `{os.path.abspath(storage.DB_PATH)}` — all decides on this file. "
                   "Back it up with your case documentation.")
