"""Builds a Markdown documentation report from analysed items."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone

from analyzer import CATEGORIES, SEVERITIES


def build_report(meta: dict, rows: list[dict], model: str, pseudonymized: bool,
                 extra: dict | None = None) -> str:
    total = len(rows)
    flagged = [r for r in rows if r["is_violation"]]
    errors = [r for r in rows if r.get("analysis_error")]
    sev_rank = {s: i for i, s in enumerate(SEVERITIES)}
    flagged.sort(key=lambda r: (-sev_rank[r["severity"]], -r["confidence"]))

    cat_counts = Counter(c for r in flagged for c in r["categories"])
    sev_counts = Counter(r["severity"] for r in flagged)
    tgt_counts = Counter(r["target"] for r in flagged)

    L = []
    L.append("# Report: Gender-Based Cyberviolence and Hate Speech Against Women\n")
    L.append(f"- **Source:** {meta.get('source_url', '')}")
    L.append(f"- **Collected (UTC):** {meta.get('collected_at_utc', '')}")
    L.append(f"- **Report generated (UTC):** {datetime.now(timezone.utc).isoformat(timespec='seconds')}")
    L.append(f"- **Items analysed:** {total} ({meta.get('post_count', 0)} posts, {meta.get('comment_count', 0)} comments)")
    L.append(f"- **Items flagged:** {len(flagged)} ({(len(flagged) / total * 100 if total else 0):.1f}%)")
    L.append(f"- **Classification model:** {model}")
    L.append(f"- **Author names pseudonymised:** {'yes' if pseudonymized else 'no'}\n")

    L.append("## Methodology and limits\n")
    L.append("Public content was collected from the page above and classified by an AI model against a taxonomy "
             "of gender-based online violence. Classifications are **automated first-pass flags and must be "
             "reviewed by a human** before being used as evidence, in publications or in complaints. Each item "
             "carries a SHA-256 hash of its text and a collection timestamp to support integrity checks. "
             "Dialect, sarcasm and missing context can cause false positives and false negatives.\n")
    if errors:
        L.append(f"> {len(errors)} item(s) could not be analysed (API/parse errors) and are not counted as clean.\n")

    L.append("## Summary\n")
    L.append("### By severity\n")
    L.append("| Severity | Count |\n|---|---|")
    for s in reversed(SEVERITIES[1:]):
        L.append(f"| {s} | {sev_counts.get(s, 0)} |")
    L.append("\n### By category\n")
    L.append("| Category | Count | Definition |\n|---|---|---|")
    for c, n in cat_counts.most_common():
        L.append(f"| {c} | {n} | {CATEGORIES[c]} |")
    L.append("\n### By target\n")
    L.append("| Target | Count |\n|---|---|")
    for t, n in tgt_counts.most_common():
        L.append(f"| {t} | {n} |")

    L.append("\n## Flagged items (most severe first)\n")
    if not flagged:
        L.append("_No items were flagged._")
    for r in flagged:
        L.append(f"### {r['id']} — {r['severity'].upper()} ({r['kind']})\n")
        L.append(f"- **Categories:** {', '.join(r['categories'])}")
        L.append(f"- **Target:** {r['target']} | **Language:** {r['language']} | **Confidence:** {r['confidence']:.2f}")
        L.append(f"- **Author:** {r['author']} | **Time (as displayed):** {r['time_text'] or 'n/a'}")
        if r["permalink"]:
            L.append(f"- **Link:** {r['permalink']}")
        if r.get("screenshot"):
            L.append(f"- **Screenshot:** {r['screenshot']}")
        L.append(f"- **Evidence:** > {r['evidence']}")
        L.append(f"- **Explanation:** {r['explanation']}")
        if r.get("inherited_from"):
            L.append(f"- **Verdict source:** inherited from human-reviewed item `{r['inherited_from']}` "
                     "(verdict memory — not a direct model call)")
        if r.get("lexicon"):
            L.append(f"- **Lexicon hits:** {', '.join(h.get('term', '?') for h in r['lexicon'][:6])}")
        vis = (extra or {}).get("vision", {}).get(r["id"]) if extra else None
        if vis:
            L.append(f"- **Image screening:** risk={vis.get('risk', '?')} — {vis.get('content', '')} "
                     f"({', '.join(vis.get('indicators') or [])})")
        if r.get("media"):
            L.append(f"- **Media:** {len(r['media'])} image/poster URL(s) collected")
        L.append(f"- **Collected:** {r['collected_at']} | **SHA-256:** `{r['sha256']}`\n")

    if extra:
        o = extra.get("osint") or {}
        if o:
            L.append("\n## OSINT dossier (automated, local)\n")
            if o.get("clusters"):
                L.append("### Theme clusters (semantic)\n")
                L.append("| Cluster (approx. label) | Items |\n|---|---|")
                for c in o["clusters"][:10]:
                    L.append(f"| {c['label'][:80]} | {c['size']} |")
            if o.get("near_dups"):
                L.append("\n### Possible template reproduction (near-duplicates)\n")
                for d in o["near_dups"][:10]:
                    L.append(f"- `{d['a']}` ↔ `{d['b']}` — similarity {d['sim']:.3f} "
                             "(same-narrative reposting / coordinated template)")
            if o.get("entities"):
                L.append("\n### Most frequent entities\n")
                L.append("| Type | Name | Mentions |\n|---|---|---|")
                for e in o["entities"][:15]:
                    L.append(f"| {e['type']} | {e['name']} | {e['count']} |")
            if o.get("cadence"):
                L.append("\n### Posting cadence (as displayed on page)\n")
                L.append("| Bucket | Items |\n|---|---|")
                for b, n in o["cadence"].items():
                    L.append(f"| {b} | {n} |")
            amp = o.get("amplifiers") or {}
            if amp.get("most_active_commenters"):
                L.append("\n### Amplification network\n")
                if o.get("graphml_path"):
                    L.append(f"Full graph export: `{o['graphml_path']}` (GraphML — open in Gephi/yEd)\n")
                L.append("| Commenter | Comments given | | Most commented | Comments received |")
                L.append("\n|---|---|---|---|---|")
                left = amp.get("most_active_commenters", [])
                right = amp.get("most_commented_authors", [])
                for i in range(max(len(left), len(right)) or 0):
                    l = f"{left[i]['node']} | {left[i]['comments_given']}" if i < len(left) else " | "
                    r = f"{right[i]['node']} | {right[i]['comments_received']}" if i < len(right) else " | "
                    L.append(f"| {l} | | {r} |")
                if amp.get("pagerank_leaders"):
                    L.append("\nInfluence leaders beyond raw volume (PageRank / betweenness, ◉ = entity node):")
                    L.append("| Node | PageRank | Betweenness |\n|---|---|---|")
                    pr = {p["node"]: p["pagerank"] for p in amp.get("pagerank_leaders", [])}
                    bw = {b["node"]: b["betweenness"] for b in amp.get("betweenness_leaders", [])}
                    for node in list(dict.fromkeys(list(pr) + list(bw)))[:8]:
                        L.append(f"| {node} | {pr.get(node, '—')} | {bw.get(node, '—')} |")
            if o.get("templates"):
                L.append("\n### Template fingerprint clusters (word-shingle Jaccard)\n")
                L.append("| Template (opening) | Copies | Distinct authors | Members |")
                L.append("\n|---|---|---|---|")
                for t in o["templates"][:10]:
                    L.append(f"| {t.get('label','')[:80]} | {t.get('size')} | {t.get('accounts', '—')} | "
                             f"{'` ' + '` `'.join(t.get('members', [])[:8]) + '`'} |")
                L.append("\nMulti-author template groups are the strongest simple coordination signal "
                         "available locally; identical text across *different* watched pages is stronger still.")
            if o.get("cross_page"):
                L.append("\n### Identical posts across watched pages (cross-page)\n")
                L.append("| sha256 | Pages | Items | Page labels |")
                L.append("\n|---|---|---|---|")
                for x in o["cross_page"][:10]:
                    L.append(f"| `{x['sha256'][:16]}…` | {x['pages']} | {x['n_items']} | "
                             f"{' · '.join(x.get('page_labels', []))} |")
            if o.get("burst"):
                b = o["burst"]
                L.append(f"\n### Posting burst detected\n\n{b['latest']} flagged items in the newest run vs "
                         f"baseline {b['baseline_mean']} over {b['baseline_runs']} prior runs — "
                         f"**+{b['sigma']}σ**. Sudden coordinated posting waves deserve a manual look.\n")
            L.append("\n_These findings are analytical leads for human investigation, not conclusions._\n")
        mem = extra.get("memory") or {}
        if mem.get("seeded"):
            L.append("\n## Verdict memory\n")
            L.append(f"- **{mem['seeded']}** item(s) inherited a human review decision from an identical "
                     "or near-identical earlier item (cosine ≥ 0.97 over local embeddings). Verdict provenance "
                     "is stamped `verdict-memory` with `inherited_from` anchors in the case dataset.\n")
        q = extra.get("quality") or {}
        if q.get("n"):
            L.append("\n## Human–AI agreement (review quality)\n")
            L.append("| Metric | Value |")
            L.append("\n|---|---|")
            L.append(f"| Reviewed items | {q.get('n')} |")
            L.append(f"| Cohen's κ | {q.get('kappa', '—')} |")
            L.append(f"| True positives (flagged, confirmed) | {q.get('true_pos', 0)} |")
            L.append(f"| False positives (flagged, rejected) | {q.get('false_pos', 0)} |")
            L.append(f"| False negatives (clean, confirmed) | {q.get('false_neg', 0)} |")
            L.append(f"| True negatives (clean, rejected) | {q.get('true_neg', 0)} |")
            if q.get("kappa") is not None and q["kappa"] < 0.4:
                L.append("\n> κ below 0.4: treat machine flags as hints, not decisions, and re-check "
                         "thresholds before using them externally.\n")
            cq = extra.get("category_quality") or []
            if cq:
                L.append("\nPer-category calibration (of items carrying this category label, share you confirmed):")
                L.append("\n| Category | Confirmed | Rejected | Human precision |")
                L.append("\n|---|---|---|---|")
                for c in cq[:12]:
                    L.append(f"| {c['category']} | {c['confirmed']} | {c['rejected']} | "
                             f"{c['human_precision']:.2f} |")
        lon = extra.get("longitudinal") or {}
        if lon and not lon.get("first_run"):
            L.append("\n## Longitudinal comparison vs previous run\n")
            L.append(f"- Items first seen in this run: **{lon['new_items']}**")
            L.append(f"- Flags added since previous run: **{lon['new_flags']}**")
            L.append(f"- Flags that disappeared (dropped/changed): **{lon['dropped_flags']}**")
            L.append(f"- Severity changes on continuing items: **{lon['severity_flips']}**\n")
    return "\n".join(L)
