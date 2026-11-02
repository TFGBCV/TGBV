"""Evaluation harness — measure the classifier against the human review dataset.

The tool's reviews (confirm/reject in the triage UI) are a free, growing labelled
dataset. This module turns them into a regression harness:

  export_gold()  → evals/gold.jsonl   (every reviewed item; label = human decision)
  score()        → run the model over the gold set and compute
                   - overall precision / recall / F1 for the binary flag
                   - per-category precision / recall (does the right LABEL land?)
                   - severity agreement on true positives
                   - relation to today's PROMPT_SHA so results are comparable
  run_eval()     → both, writing evals/results_<ts>.json + evals/latest.json

Prompt/model changes are then MEASURED, not vibes. All computed locally.
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from datetime import datetime, timezone

import storage
from analyzer import PROMPT_SHA, SEVERITIES

SEV_RANK = {s: i for i, s in enumerate(SEVERITIES)}
OUT_DIR = "evals"


def export_gold(page_id: int | None = None, out_path: str = "") -> str:
    """Dump every reviewed item as a labelled example. Returns the file path."""
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = out_path or os.path.join(OUT_DIR, "gold.jsonl")
    with storage.db() as c:
        q = """
        SELECT i.item_key, i.page_id, i.text, r.action AS human, r.reviewer,
               v.is_violation, v.categories, v.severity
        FROM reviews r
        JOIN items i ON i.item_key=r.item_key
        LEFT JOIN verdicts v ON v.item_key=i.item_key AND v.verifier_kind=''
          AND v.id IN (SELECT MAX(id) FROM verdicts WHERE verifier_kind='' GROUP BY item_key)
        WHERE r.action IN ('confirm','reject')"""
        args: list = []
        if page_id is not None:
            q += " AND i.page_id=?"
            args.append(page_id)
        n = 0
        with open(out_path, "w", encoding="utf-8") as f:
            for r in c.execute(q, args):
                label = 1 if r["human"] == "confirm" else 0
                try:
                    cats = json.loads(r["categories"] or "[]")
                except json.JSONDecodeError:
                    cats = []
                f.write(json.dumps({
                    "item_key": r["item_key"], "page_id": r["page_id"],
                    "text": r["text"], "label": label, "gold_categories": cats,
                    "gold_severity": r["severity"], "human": r["human"], "reviewer": r["reviewer"],
                }, ensure_ascii=False) + "\n")
                n += 1
    storage.audit("eval", "gold_exported", f"{n} items → {out_path}")
    return out_path


def _load_gold(path: str, limit: int = 400) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


def score(client, model: str, gold_path: str, batch_size: int = 2,
          limit: int = 400, progress=None) -> dict:
    """Run the model over the gold set and score it. `progress(msg, frac)` optional."""
    gold = _load_gold(gold_path, limit)
    if not gold:
        return {"error": "no gold items (export_gold first, or review some items)"}
    items = [{"id": g["item_key"], "kind": "post", "text": g["text"],
              "sha256": "", "permalink": "", "author": "", "time_text": "",
              "collected_at": "", "screenshot": "", "media": []} for g in gold]
    preds = {r["id"]: r for r in
             client_analyze(items, client, model, batch_size, progress)}
    tp = fp = fn_ = tn = 0
    cats: dict[str, Counter] = defaultdict(Counter)   # cat -> {support, predicted, correct}
    sev_exact = sev_close = n_tp = 0
    conf_bins = Counter()
    for g in gold:
        p = preds.get(g["item_key"])
        if not p:
            fp += 1 if g["label"] else 0
            continue
        pred, gold_label = bool(p.get("is_violation")), g["label"] == 1
        if pred and gold_label:
            tp += 1
            n_tp += 1
            s_gold, s_pred = g.get("gold_severity"), p.get("severity", "none")
            if s_gold == s_pred:
                sev_exact += 1
            if s_gold in SEV_RANK and s_pred in SEV_RANK and abs(SEV_RANK[s_gold] - SEV_RANK[s_pred]) <= 1:
                sev_close += 1
            for cat in set(g["gold_categories"]) | set(p.get("categories") or []):
                has_gold = cat in g["gold_categories"]
                has_pred = cat in (p.get("categories") or [])
                if has_gold:
                    cats[cat]["support"] += 1
                if has_pred:
                    cats[cat]["predicted"] += 1
                if has_gold and has_pred:
                    cats[cat]["correct"] += 1
        elif pred and not gold_label:
            fp += 1
            for cat in p.get("categories") or []:
                cats[cat]["predicted"] += 1
        elif not pred and gold_label:
            fn_ += 1
            for cat in g["gold_categories"]:
                cats[cat]["support"] += 1
        else:
            tn += 1
        # confidence calibration: sharper confidence on correct calls?
        correct = pred == gold_label
        conf_bins["correct" if correct else "wrong"] += 1
    prec = tp / (tp + fp) if tp + fp else None
    rec = tp / (tp + fn_) if tp + fn_ else None
    f1 = 2 * prec * rec / (prec + rec) if prec is not None and rec is not None and (prec + rec) else None
    out = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model": model, "prompt_sha": PROMPT_SHA, "n_gold": len(gold),
        "tp": tp, "fp": fp, "fn": fn_, "tn": tn,
        "precision": round(prec, 3) if prec is not None else None,
        "recall": round(rec, 3) if rec is not None else None,
        "f1": round(f1, 3) if f1 is not None else None,
        "severity_exact_rate": round(sev_exact / n_tp, 3) if n_tp else None,
        "severity_within1_rate": round(sev_close / n_tp, 3) if n_tp else None,
        "mean_confidence_when_correct": None,
        "per_category": {cat: {k: v for k, v in st.items()}
                         for cat, st in sorted(cats.items())},
    }
    conf_when_correct = [p for g in gold
                         for p in [preds.get(g["item_key"], {}).get("confidence")]
                         if p is not None and bool(preds[g["item_key"]].get("is_violation")) == (g["label"] == 1)]
    if conf_when_correct:
        out["mean_confidence_when_correct"] = round(sum(conf_when_correct) / len(conf_when_correct), 3)
    return out


def client_analyze(items, client, model, batch_size, progress=None):
    """Thin wrapper for scoring-time classification with no lexicon badges
    (gold scoring should reflect the *policy prompt*, not the lexicon floor)."""
    from analyzer import analyze_items
    return analyze_items(items, client, model, batch_size=batch_size,
                         use_lexicon=False, progress=progress)


def run_eval(client, model: str, page_id: int | None = None, gold_path: str = "",
             batch_size: int = 2, limit: int = 400, progress=None) -> dict:
    """Export gold (if needed) + score + persist evals/latest.json. Returns summary."""
    gold_path = gold_path or os.path.join(OUT_DIR, "gold.jsonl")
    if not os.path.exists(gold_path):
        export_gold(page_id, gold_path)
    result = score(client, model, gold_path, batch_size=batch_size, limit=limit, progress=progress)
    os.makedirs(OUT_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    with open(os.path.join(OUT_DIR, f"results_{stamp}.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT_DIR, "latest.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1)
    storage.audit("eval", "run", f"model={model} f1={result.get('f1')} n={result.get('n_gold')}")
    return result


def history(limit: int = 10) -> list[dict]:
    """Compact list of past eval results (for the UI trend)."""
    out = []
    if not os.path.isdir(OUT_DIR):
        return out
    import re as _re
    names = [n for n in os.listdir(OUT_DIR) if _re.fullmatch(r"results_\d{8}_\d{6}\.json", n)]
    for n in sorted(names, reverse=True)[:limit]:
        try:
            with open(os.path.join(OUT_DIR, n), encoding="utf-8") as f:
                d = json.load(f)
            out.append({"file": n, "model": d.get("model"),
                        "prompt_sha": d.get("prompt_sha"), "n_gold": d.get("n_gold"),
                        "precision": d.get("precision"), "recall": d.get("recall"),
                        "f1": d.get("f1")})
        except Exception:
            continue
    return out
