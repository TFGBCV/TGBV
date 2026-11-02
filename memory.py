"""Verdict memory — reuse human review decisions for near-identical items.

Before the LLM pass runs, every unclassified item is compared (via the existing
nomic-embed embeddings) against items a human has already REVIEWED in the triage
stage. A new item nearly identical (cosine ≥ MIN_SEED_SIM, same kind) to a
reviewed item inherits that decision — clearly provenance-stamped:

  - a CONFIRMED item seeds its verdict (flag + categories + severity);
  - a REJECTED item seeds a clean verdict (the human ruled this text out) —
    keeps recurring benign posts from being re-flagged every run;

  - verdict `model` column: 'verdict-memory',
  - `inherited_from`: the confirmed item_key it was seeded from,
  - `explanation`: prefixed "[seeded: same text as confirmed item X (sim 0.98)] …".

Benefits: consistent rulings across recurring watchlists (same rant → same
outcome), fewer LLM calls, and every inherited verdict still carries a human
decision in its chain of custody. This is pure local logic on the SQLite
dataset; nothing new leaves the machine.
"""
from __future__ import annotations

import numpy as np

import storage
from lmstudio import LMStudioClient

DEFAULT_EMBED_MODEL = "text-embedding-nomic-embed-text-v1.5"
MIN_SEED_SIM = 0.97  # deliberately strict: seeding the WRONG verdict is worse than re-classifying


def _unit(vec: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(vec)) or 1.0
    return vec / n


def seed_from_confirmed(page_id: int, run_id: int, items: list[dict],
                        lm: LMStudioClient, embed_model: str = DEFAULT_EMBED_MODEL,
                        progress=None) -> list[str]:
    """Seed verdicts for items whose text is near-identical to human-confirmed
    items of the same page. `items` = analyzer-style dicts (id/kind/text/...).

    Returns the list of item_keys that were seeded (they must NOT be classified
    again by the main pass). Anything not seeded is returned untouched.
    """
    keys = [f"{page_id}:{it['sha256']}" for it in items if it.get("sha256")]
    if not keys:
        return []

    # ---- reviewed items across ALL pages (cross-page twins are exactly the
    # interesting case); same-page twins take priority when both exist.
    with storage.db() as c:
        confirmed = c.execute("""
            SELECT i.item_key, i.page_id, i.kind, i.text, r.action AS review_action,
                   v.is_violation, v.categories,
                   v.severity, v.target, v.language, v.evidence, v.explanation, v.confidence
            FROM items i
            JOIN reviews r ON r.item_key=i.item_key AND r.action IN ('confirm','reject')
            LEFT JOIN verdicts v ON v.item_key=i.item_key AND v.verifier_kind=''
              AND v.id IN (SELECT MAX(id) FROM verdicts WHERE verifier_kind='' GROUP BY item_key)
            """).fetchall()
        already = {row["item_key"] for row in c.execute(
            "SELECT DISTINCT item_key FROM verdicts WHERE verifier_kind=''")}
        emb_rows = c.execute("SELECT item_key, model, vec FROM embeddings").fetchall()
    emb_by_key: dict[str, np.ndarray] = {}
    for r in emb_rows:
        if r["model"] == embed_model:
            emb_by_key[r["item_key"]] = _unit(np.frombuffer(r["vec"], dtype=np.float32))

    if not confirmed:
        return []
    # confirmed sources KEEP their verdict; rejected sources seed a clean verdict.
    # `already` protects TARGET items that already carry a primary verdict.
    usable: list[dict] = []
    for row in confirmed:
        d = dict(row)
        if d["review_action"] == "reject":
            # human said clean — seed a clean verdict even when the model verdict
            # row is missing or stale.
            d.update({"is_violation": False, "severity": "none",
                      "target": d.get("target") or "none", "language": d.get("language") or "",
                      "evidence": "", "categories": "[]",
                      "explanation": "human rejected this text as not a violation"})
            usable.append(d)
        elif d["is_violation"]:
            usable.append(d)  # confirmed flag with a concrete verdict to inherit
    if not usable:
        return []
    confirmed = usable

    confirmed_by_kind: dict[str, list[dict]] = {}
    for c_it in confirmed:
        vec = emb_by_key.get(c_it["item_key"])
        if vec is not None:
            confirmed_by_kind.setdefault(c_it["kind"], []).append({**c_it, "_vec": vec})

    # ---- embed the new items (batched; cached in the embeddings table)
    from osint import embed_items
    new_keys = sorted({k for k in keys if k not in emb_by_key})
    texts_of = {f"{page_id}:{it['sha256']}": it.get("text", "") for it in items}
    if new_keys:
        vecs = embed_items(lm, new_keys, [texts_of[k] for k in new_keys], embed_model)
        emb_by_key.update({k: _unit(v) for k, v in vecs.items()})

    seeded: list[str] = []
    out_rows: list[dict] = []
    for it in items:
        key = f"{page_id}:{it['sha256']}"
        vec = emb_by_key.get(key)
        if vec is None or key in seeded or key in already:
            continue
        cands = []
        for cand in confirmed_by_kind.get(it.get("kind", ""), []):
            if cand["item_key"] == key:
                continue
            try:
                sim = float(np.dot(vec, cand["_vec"]))
            except ValueError:
                continue  # dimension mismatch (model swapped) — skip silently
            if sim >= MIN_SEED_SIM:
                cands.append((cand.get("page_id") == page_id, sim, cand))
        if not cands:
            continue
        cands.sort(key=lambda t: (not t[0], -t[1]))  # same-page first, then highest sim
        _, best_sim, best = cands[0]
        cross_note = "" if best["page_id"] == page_id else f" on page {best['page_id']}"
        src_label = ("confirmed" if best.get("review_action", "confirm") == "confirm"
                     else "human-rejected")
        seeded.append(key)
        out_rows.append({
            "id": key,
            "is_violation": bool(best["is_violation"]),
            "categories": _loads(best["categories"]),
            "severity": best["severity"],
            "target": best["target"],
            "language": best["language"],
            "evidence": best["evidence"],
            "explanation": f"[seeded: same text as {src_label} {best['item_key']}"
                           f"{cross_note} (sim {best_sim:.3f})] "
                           f"{best['explanation'] or 'human review decision'}"[:600],
            "confidence": round(min(0.99, best_sim), 3),
            "analysis_error": "",
            "inherited_from": best["item_key"],
        })
        if progress:
            progress(f"seeded {key} ← {best['item_key']}", len(seeded) / len(keys))

    if out_rows:
        storage.add_verdicts(run_id, out_rows, key_of=lambda r: r["id"],
                             model="verdict-memory", prompt_sha="")
        storage.audit("memory", "verdict_seeded",
                      f"page {page_id}: {len(out_rows)} item(s) inherited from confirmed verdicts")
    return seeded


def _loads(s):
    import json
    try:
        return json.loads(s or "[]")
    except (ValueError, TypeError):
        return []
