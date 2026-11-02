#!/usr/bin/env python3
"""Offline unit tests for the v3 feature set. No network, no LM Studio:
a stub engine replays deterministic replies for the classifier / verifier /
entity prompts, and embeddings are deterministic per-text vectors.

Run:  python tests/test_units.py        (or: pytest tests/test_units.py -q)
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

_TMP = tempfile.mkdtemp(prefix="gbv_unit_")
os.environ["GBV_DB"] = os.path.join(_TMP, "test_gbv.db")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hashlib  # noqa: E402

import numpy as np  # noqa: E402

import agent  # noqa: E402
import alerts  # noqa: E402
import analyzer  # noqa: E402
import eval as eval_mod  # noqa: E402
import jobs  # noqa: E402
import lexicon  # noqa: E402
import memory  # noqa: E402
import osint  # noqa: E402
import storage  # noqa: E402
from analyzer import _extract_json_array, smart_truncate  # noqa: E402

EMBED_MODEL = osint.EMBED_MODEL


# ------------------------------------------------------------------ stub engine
class StubEngine:
    """Deterministic local-engine replacement.

    - classify: texts containing 'whore' become violations (misogynistic_insults);
      everything else is clean. Honors `response_format` (schema-capable).
    - verifier: always agrees.
    - entities: finds CEDAW as law_or_treaty in any text passed.
    - embed: 16-dim deterministic unit vector per text (identical text = identical).
    """

    schema_calls = 0
    plain_calls = 0

    def chat(self, model, user_input, system_prompt="", temperature=0.0,
             max_output_tokens=None, reasoning=None, response_format=None, **kw):
        if response_format:
            self.__class__.schema_calls += 1
        else:
            self.__class__.plain_calls += 1
        if "For EACH item decide" in system_prompt:
            return self._classify(user_input)
        if "adversarial reviewer" in system_prompt:
            prev = json.loads(user_input)
            return json.dumps({"id": prev["id"], "agree": True, "action": "keep",
                               "reason": "checked"})
        if "Extract named entities" in system_prompt:
            return self._entities(user_input)
        return json.dumps({"tool": "finish", "summary": "stub"})

    def _classify(self, user_input):
        out = []
        for it in json.loads(user_input):
            text = it.get("text") or ""
            hit = "whore" in text or "سيداو" in text
            if hit:
                out.append({"id": it["id"], "is_violation": True,
                            "categories": ["misogynistic_insults"], "severity": "medium",
                            "target": "individual_woman", "language": "en",
                            "evidence": "whore", "explanation": "gendered slur",
                            "confidence": 0.9})
            else:
                out.append({"id": it["id"], "is_violation": False, "categories": [],
                            "severity": "none", "target": "none", "language": "en",
                            "evidence": "", "explanation": "not gendered",
                            "confidence": 0.8})
        return json.dumps(out, ensure_ascii=False)

    def _entities(self, user_input):
        # payload may be ONE json object or several concatenated by "\\n\\n"
        objs, dec, idx = [], json.JSONDecoder(), 0
        s = user_input.strip()
        try:
            while idx < len(s):
                while idx < len(s) and s[idx] in " \n\r\t":
                    idx += 1
                if idx >= len(s):
                    break
                obj, idx = dec.raw_decode(s, idx)
                objs.append(obj)
        except json.JSONDecodeError:
            return json.dumps([], ensure_ascii=False)
        out = []
        for m in objs:
            ents = []
            if "CEDAW" in (m.get("text") or ""):
                ents.append({"type": "law_or_treaty", "name": "CEDAW"})
            if "TestOrg" in (m.get("text") or ""):
                ents.append({"type": "org", "name": "TestOrg"})
            out.append({"id": m["id"], "entities": ents})
        return json.dumps(out, ensure_ascii=False)

    def embed(self, texts, model):
        vecs = []
        for t in texts:
            b = hashlib.sha256(t.encode("utf-8")).digest() * 4
            v = np.frombuffer(b, dtype=np.uint8).astype(np.float32)[:16]
            v = v / (float(np.linalg.norm(v)) or 1.0)
            vecs.append(list(v))
        return vecs


LM = StubEngine()
MODEL = "stub-model"


def _mkitem(i, text, kind="post", sha=None, author="user_x", media=None):
    return {"id": i, "kind": kind, "parent_id": "" if kind == "post" else "p1",
            "author": author, "text": text, "permalink": "", "time_text": "",
            "sha256": sha or analyzer._sha(text), "collected_at": "",
            "screenshot": "", "media": media or []}


# ------------------------------------------------------------------ analyzer bits
def test_smart_truncate_keeps_head_and_tail():
    long_text = "HEAD-" + ("abcdefghij" * 400) + "-TAIL"
    out = smart_truncate(long_text, 500)
    assert len(out) <= 520
    assert out.startswith("HEAD-")
    assert out.endswith("-TAIL")
    assert "…truncated…" in out
    assert smart_truncate("short", 500) == "short"


def test_extract_json_array_variants():
    arr = _extract_json_array('[{"id": "p1", "confidence": 0.5}]')
    assert arr[0]["id"] == "p1"
    fenced = _extract_json_array('```json\n[{"id": "x"}]\n```')
    assert fenced[0]["id"] == "x"
    thinky = _extract_json_array('<think>hm</think>[{"id": "y"}]')
    assert thinky and thinky[0]["id"] == "y"
    wrapped = _extract_json_array('{"results": [{"id": "z"}]}')
    assert wrapped[0]["id"] == "z"


def test_lexicon_scan_arabic_and_arabizi():
    hits = lexicon.scan("هذه شرموطة تستحق الموت")
    kinds = {h["kind"] for h in hits}
    assert "slur" in kinds and "threat" in kinds
    hits2 = lexicon.scan("this 9ahba deserves to die")
    assert {h["kind"] for h in hits2} >= {"slur", "threat"}
    # neutral woman word must NOT trip the lexicon
    assert not lexicon.scan("المرأة تطالب بحقها في العمل")
    # non-gendered insult must not trip it
    assert not lexicon.scan("you are an idiot")


def test_vote_majority_and_confidence():
    r_viol = {"is_violation": True, "severity": "high", "categories": ["threats_of_violence"],
              "confidence": 0.9, "evidence": "x", "explanation": "e", "target": "t",
              "language": "ar", "analysis_error": ""}
    r_clean = {"is_violation": False, "severity": "none", "categories": [], "confidence": 0.8,
               "evidence": "", "explanation": "e", "target": "t", "language": "ar",
               "analysis_error": ""}
    voted = analyzer._vote([dict(r_viol), dict(r_viol), dict(r_clean)],
                           hits=[{"kind": "threat", "term": "kill her", "gloss": ""}])
    assert voted["is_violation"] is True
    assert abs(voted["confidence"] - 2 / 3) < 1e-3  # rounded to 3 decimals
    assert voted["severity"] == "high"
    assert voted["lexicon_attention"] is False
    clean_voted = analyzer._vote([dict(r_clean), dict(r_clean)], hits=None)
    assert clean_voted["is_violation"] is False


def test_analyze_items_passes_and_lexicon_attention():
    items = [
        _mkitem("p1", "what a nice day for rights work"),
        _mkitem("p2", "she is a whore and deserves it"),
        _mkitem("p3", "بيت و مطبخ و سيداو للمطبخ"),  # stub flags, lexicon lacks 'سيداو'
    ]
    rows = analyzer.analyze_items(items, LM, MODEL, batch_size=2, passes=3)
    by_id = {r["id"]: r for r in rows}
    # stub is deterministic → 3/3 agreement → confidence 1.0
    assert abs(by_id["p1"]["confidence"] - 1.0) < 1e-6
    assert by_id["p2"]["is_violation"] is True
    # lexicon attention: model clean + lexicon hit
    items2 = [_mkitem("p4", "this 9ahba text where the model is bribed to say clean")]
    LM2 = BribeEngine()
    rows2 = analyzer.analyze_items(items2, LM2, MODEL, batch_size=1)
    assert rows2[0]["lexicon_attention"] is True
    assert rows2[0]["is_violation"] is False


class BribeEngine(StubEngine):
    def _classify(self, user_input):
        out = []
        for it in json.loads(user_input):
            out.append({"id": it["id"], "is_violation": False, "categories": [],
                        "severity": "none", "target": "none", "language": "en",
                        "evidence": "", "explanation": "clean", "confidence": 0.9})
        return json.dumps(out, ensure_ascii=False)


# ------------------------------------------------------------------ storage
def _prime_page(page_url="https://facebook.com/UnitPage", label="UnitPage"):
    pid = storage.get_or_create_page(page_url, label)
    return pid


def test_storage_roundtrip_reviews_and_quality():
    pid = _prime_page()
    run1 = storage.start_run(pid, notes="unit")
    items = [_mkitem("p1", "she is a whore and should be shamed"),
             _mkitem("p2", "regular weather talk")]
    storage.upsert_items(pid, run1, items)
    rows = analyzer.analyze_items(items, LM, MODEL, batch_size=2)
    key_of = lambda r, p=pid: f"{p}:{r['sha256']}"
    storage.add_verdicts(run1, rows, key_of=key_of, model=MODEL,
                         prompt_sha=analyzer.PROMPT_SHA)
    cur = storage.current_verdicts(pid)
    flags = {c["item_key"]: c["is_violation"] for c in cur}
    assert flags[f"{pid}:{rows[0]['sha256']}"] is True
    assert flags[f"{pid}:{rows[1]['sha256']}"] is False
    # review both
    k1 = f"{pid}:{rows[0]['sha256']}"
    k2 = f"{pid}:{rows[1]['sha256']}"
    storage.set_review(k1, "confirm", reviewer="alice")
    storage.set_review(k2, "reject", reviewer="alice")
    stats = storage.review_stats(pid)
    assert stats == {"confirmed": 1, "rejected": 1}
    pend = storage.pending_review(pid)
    assert all(not r.get("review_action") for r in pend)
    # human-AI agreement: flag confirmed (TP) + clean rejected (TN) → perfect
    agr = storage.human_ai_agreement(pid)
    assert agr["n"] == 2 and agr["kappa"] == 1.0
    cats = storage.category_review_stats(pid, min_n=1)
    assert any(c["category"] == "misogynistic_insults" and c["human_precision"] == 1.0
               for c in cats)
    # verdict prompt_sha recorded
    with storage.db() as c:
        v = c.execute("SELECT prompt_sha FROM verdicts WHERE item_key=?", (k1,)).fetchone()
    assert v and v["prompt_sha"] == analyzer.PROMPT_SHA
    # media column survives upsert
    run2 = storage.start_run(pid, notes="unit media")
    media_items = [_mkitem("p1", "she is a whore and should be shamed", media=["https://scontent.xx/1.jpg"])]
    storage.upsert_items(pid, run2, media_items)
    with storage.db() as c:
        m = c.execute("SELECT media FROM items WHERE item_key=?", (k1,)).fetchone()
    assert "scontent" in m["media"]


def test_case_file_structured_and_legacy():
    pid = _prime_page("https://facebook.com/CaseFilePage")
    storage.write_case_file(pid, "- [2026-01-01] old markdown observation")
    cf = storage.case_file_get(pid)
    assert cf["legacy"].startswith("- [")
    assert cf["hypotheses"] == []

    def add_hyp(cf):
        cf.setdefault("hypotheses", []).append(
            {"text": "page amplifies anti-CEDAW templates", "status": "open",
             "anchor": "", "opened": "2026-01-02", "updated": "2026-01-02"})

    storage.case_file_apply(pid, add_hyp)
    cf = storage.case_file_get(pid)
    assert len(cf["hypotheses"]) == 1 and cf["legacy"]  # legacy preserved alongside
    # hypothesis tool updates status by matching text
    ctx = {"page_id": pid, "lm": LM, "model": MODEL, "batch_size": 1, "ts": "2026-01-03"}
    agent._tool_hypothesis({"text": "page amplifies anti-CEDAW templates",
                            "status": "supported", "anchor": f"{pid}:abc"}, ctx)
    cf = storage.case_file_get(pid)
    assert cf["hypotheses"][0]["status"] == "supported"
    assert cf["hypotheses"][0]["anchor"] == f"{pid}:abc"
    # case_add appends observations + gaps
    agent._tool_case_add({"observations": [f"two templated posts ({pid}:k1)"],
                          "gaps": ["who wrote the template?"]}, ctx)
    cf = storage.case_file_get(pid)
    assert any(f"{pid}:k1" in o.get("anchor", "") or f"{pid}:k1" in o.get("text", "")
               for o in cf["observations"])
    assert cf["gaps"] and "template" in cf["gaps"][-1]["text"]
    assert "REFLECTION" not in cf["observations"][0]["text"]


# ------------------------------------------------------------------ search + graph
def test_fts_and_search_corpus():
    pid = _prime_page("https://facebook.com/SearchPage")
    run = storage.start_run(pid)
    items = [_mkitem("p1", "سيداو معاهدة الخراب لا تساوي شيئا"),
             _mkitem("p2", "weather talk with no slur at all")]
    storage.upsert_items(pid, run, items)
    rows_ids = storage.search_corpus("سيداو", page_id=pid)
    keys = {r["item_key"] for r in rows_ids}
    assert f"{pid}:{items[0]['sha256']}" in keys
    if rows_ids and "excerpt" in rows_ids[0]:
        assert all(items[0]["sha256"] in k for k in keys)  # only the matching one


def test_entities_persist_and_entity_graph():
    pid = _prime_page("https://facebook.com/EntPage")
    run = storage.start_run(pid)
    items = [_mkitem("p1", "CEDAW destroys the nation says TestOrg")]
    storage.upsert_items(pid, run, items)
    counts, ent_rows = osint.extract_entities(
        LM, MODEL, [{"item_key": f"{pid}:{items[0]['sha256']}",
                     "text": items[0]["text"]}])
    assert counts[("law_or_treaty", "CEDAW")] == 1
    storage.replace_item_entities(run, ent_rows)
    g = storage.entity_graph("CEDAW", page_id=pid)
    assert g["matches"] and g["matches"][0]["count"] == 1
    assert g["matches"][0]["item_keys"] == [f"{pid}:{items[0]['sha256']}"]
    edges = storage.author_entity_edges(pid)
    assert edges and edges[0]["target"] in ("CEDAW", "TestOrg")


def test_templates_and_cross_page():
    sha_shared = analyzer._sha("RECTE_PLATFORM_CAMERA_TIMEOUT_8f3a")
    template_text = "النساء لا يفهمن السياسة، مكانهن المطبخ، هذا تكرار"
    pidA = _prime_page("https://facebook.com/TmplA")
    pidB = _prime_page("https://facebook.com/TmplB")
    for pid in (pidA, pidB):
        run = storage.start_run(pid)
        storage.upsert_items(pid, run, [
            _mkitem("p1", template_text, sha=sha_shared),
            _mkitem("p2", "unique text about rain and seasons nobody repeats"),
        ])
    texts = {f"{pidA}:{sha_shared}": template_text, f"{pidA}:x2": "unique text about rain and seasons nobody repeats"}
    cl = osint.template_clusters(texts)
    assert cl == []  # single twin here; cluster needs both copies → below
    texts2 = {f"{pidA}:{sha_shared}": template_text, f"{pidB}:{sha_shared}": template_text,
              f"{pidA}:x2": "unique text about rain and seasons nobody repeats"}
    cl2 = osint.template_clusters(texts2)
    assert len(cl2) == 1 and cl2[0]["size"] == 2
    cross = storage.cross_page_matches()
    assert any(c["sha256"] == sha_shared and c["pages"] == 2 for c in cross)


def test_burst_detection():
    pid = _prime_page("https://facebook.com/BurstPage")
    sev_texts = [f"burst filler text number {i}" for i in range(8)]
    texts_iter = iter(sev_texts)
    # three quiet runs, one each with a single flag
    for _ in range(3):
        run = storage.start_run(pid)
        it = [_mkitem("p1", next(texts_iter))]
        storage.upsert_items(pid, run, it)
        r = analyzer.analyze_items(it, LM, MODEL, batch_size=1)
        r[0]["is_violation"], r[0]["severity"] = True, "low"
        r[0]["categories"] = ["other_gbv"]
        storage.add_verdicts(run, r, key_of=lambda x: f"{pid}:{x['sha256']}", model=MODEL)
        storage.finish_run(run, "done")
    assert osint.burst_sigma(pid) is None  # flat baseline, no jump yet
    # wave run: 4 flags
    run = storage.start_run(pid)
    wave = [_mkitem("p1", next(texts_iter)) for _ in range(4)]
    storage.upsert_items(pid, run, wave)
    r = analyzer.analyze_items(wave, LM, MODEL, batch_size=2)
    for x in r:
        x["is_violation"], x["severity"], x["categories"] = True, "low", ["other_gbv"]
    storage.add_verdicts(run, r, key_of=lambda x: f"{pid}:{x['sha256']}", model=MODEL)
    storage.finish_run(run, "done")
    burst = osint.burst_sigma(pid)
    assert burst and burst["latest"] == 4 and burst["sigma"] >= 2.0


# ------------------------------------------------------------------ run_osint & memory
def test_run_osint_packages_extras():
    pid = _prime_page("https://facebook.com/OsintPage")
    run = storage.start_run(pid)
    items = [_mkitem("p1", "CEDAW talk by TestOrg volunteers"),
             _mkitem("p2", "CEDAW destroys families claims TestOrg again")]
    storage.upsert_items(pid, run, items)
    o = osint.run_osint(run, pid, LM, llm_model=MODEL)
    assert "templates" in o and "cross_page" in o and "burst" in o
    assert any("CEDAW" == e["name"] for e in o["entities"])
    back = osint.get_osint(run)
    assert "templates" in back and "cross_page" in back
    assert back.get("burst") is None  # single run so far — burst needs history


def test_memory_seeds_from_confirmed():
    # Note: within ONE page, a byte-identical post is the same dedup-key item, so
    # seeding matters (a) across DIFFERENT watched pages sharing identical text
    # and (b) for near-identical (embed-sim ≥ 0.97) rephrasings. Test (a) here,
    # for BOTH review outcomes: confirmed violations AND rejected (clean) texts.
    pidA = _prime_page("https://facebook.com/MemPageA")
    pidB = _prime_page("https://facebook.com/MemPageB")
    text_flag = "she is a whore and belongs in the kitchen said the confident stranger"
    text_clean = "score updates and weather talk for the whole week"
    run1 = storage.start_run(pidA)
    it1 = _mkitem("p1", text_flag)
    it2 = _mkitem("p2", text_clean)
    storage.upsert_items(pidA, run1, [it1, it2])
    rows = analyzer.analyze_items([it1, it2], LM, MODEL, batch_size=2)
    storage.add_verdicts(run1, rows, key_of=lambda r, p=pidA: f"{p}:{r['sha256']}", model=MODEL)
    storage.finish_run(run1, "done")
    key1 = f"{pidA}:{it1['sha256']}"
    key2 = f"{pidA}:{it2['sha256']}"
    storage.set_review(key1, "confirm", reviewer="r")
    storage.set_review(key2, "reject", reviewer="r")
    # embed BOTH reviewed sources (memory reuses the embeddings table)
    osint.embed_items(LM, [key1, key2], [text_flag, text_clean], EMBED_MODEL)

    # page B collects the same posts verbatim → must inherit the human decisions
    run2 = storage.start_run(pidB)
    itB1 = _mkitem("q1", text_flag)
    itB2 = _mkitem("q2", text_clean)
    itB3 = _mkitem("q3", "totally different content about football and rain")
    storage.upsert_items(pidB, run2, [itB1, itB2, itB3])
    seeded = memory.seed_from_confirmed(pidB, run2, [itB1, itB2, itB3], LM, EMBED_MODEL)
    assert seeded == [f"{pidB}:{itB1['sha256']}", f"{pidB}:{itB2['sha256']}"], seeded
    with storage.db() as c:
        v = c.execute("""SELECT model, inherited_from, explanation, is_violation, severity
                         FROM verdicts WHERE item_key=?""", (seeded[0],)).fetchone()
        v2 = c.execute("""SELECT model, inherited_from, explanation, is_violation, severity
                          FROM verdicts WHERE item_key=?""", (seeded[1],)).fetchone()
    assert v["model"] == "verdict-memory"
    assert v["inherited_from"] == key1
    assert "same text as confirmed" in v["explanation"]
    assert v["is_violation"] == 1 and v["severity"] == "medium"
    # rejected twin inherits a CLEAN verdict
    assert v2["inherited_from"] == key2
    assert v2["is_violation"] == 0 and v2["severity"] == "none"
    assert "same text as human-rejected" in v2["explanation"]


# ------------------------------------------------------------------ agent tools
def test_agent_tools_offline():
    pid = _prime_page()
    ctx = {"page_id": pid, "lm": LM, "model": MODEL, "batch_size": 1, "ts": "2026-01-05"}
    out = json.loads(agent._tool_search_corpus({"query": "whore", "flagged_only": True}, ctx))
    assert "matches" in out
    out = json.loads(agent._tool_propose_target(
        {"url": "https://www.facebook.com/SomePage", "why": "template author may be central"}, ctx))
    assert out["status"].startswith("pending")
    out = json.loads(agent._tool_propose_target({"url": "https://evil.com/x", "why": ""}, ctx))
    assert "error" in out
    # queue_for_review queues an existing item
    with storage.db() as c:
        any_key = c.execute("SELECT item_key FROM items LIMIT 1").fetchone()
    if any_key:
        out = json.loads(agent._tool_queue_for_review({"item_key": any_key[0], "why": "w"}, ctx))
        assert out.get("proposal_id"), out
        storage.decide_proposal(out["proposal_id"], "dismissed", "tests")
    # cluster_corpus without embeddings → graceful error
    pid2 = _prime_page("https://facebook.com/BareAgentPage")
    ctx2 = {"page_id": pid2, "lm": LM, "model": MODEL, "batch_size": 1, "ts": ""}
    out = json.loads(agent._tool_cluster_corpus({}, ctx2))
    assert "error" in out


def test_agent_reflect_and_tools_registered():
    for t in ("reclassify", "cluster_corpus", "entity_graph", "queue_for_review",
              "hypothesis", "case_add"):
        assert t in agent.TOOLS, f"missing agent tool {t}"


# ------------------------------------------------------------------ alerts + eval
def test_alerts_digest_and_burst():
    pid = _prime_page("https://facebook.com/AlertPage")
    run = storage.start_run(pid)
    items = [_mkitem("p1", "she is a whore and worse"), _mkitem("p2", "maybe seconds long text")]
    storage.upsert_items(pid, run, items)
    rows = analyzer.analyze_items(items, LM, MODEL, batch_size=2)
    storage.add_verdicts(run, rows, key_of=lambda r: f"{pid}:{r['sha256']}", model=MODEL)
    storage.finish_run(run, "done")
    d = alerts.check_run(pid, run)
    # stub flags at severity medium → below high/critical → empty text
    assert d["text"] == "" and d["new_flagged"] == 0
    assert d["sent"].get("note")


def test_eval_score_and_history():
    gold_path = os.path.join(os.path.dirname(_TMP), "evals_gold.jsonl")
    gold_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_gold_test.jsonl")
    rows = [
        {"item_key": "g1", "page_id": 1, "text": "she is a whore even today",
         "label": 1, "gold_categories": ["misogynistic_insults"], "gold_severity": "medium",
         "human": "confirm", "reviewer": "r"},
        {"item_key": "g2", "page_id": 1, "text": "another clean day text",
         "label": 0, "gold_categories": [], "gold_severity": "none",
         "human": "reject", "reviewer": "r"},
    ]
    with open(gold_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    res = eval_mod.score(LM, MODEL, gold_path, batch_size=2)
    assert res["tp"] == 1 and res["tn"] == 1 and res["fp"] == 0 and res["fn"] == 0
    assert res["precision"] == 1.0 and res["recall"] == 1.0
    assert res["severity_exact_rate"] == 1.0
    pc = res["per_category"]["misogynistic_insults"]
    assert pc["correct"] == 1 and pc["support"] == 1
    hist = eval_mod.history()
    assert isinstance(hist, list)
    os.remove(gold_path)


# ------------------------------------------------------------------ jobs
def test_jobs_enqueue_claim_finish_payload():
    jid = jobs.enqueue_run_due({"lm_url": "http://x", "model": "m"}, {"cookies": "c_user=1"})
    claimed = storage.claim_job()
    assert claimed and claimed["id"] == jid and claimed["status"] == "running"
    payload = storage.pop_payload(str(jid))
    assert payload.get("cookies") == "c_user=1"
    assert storage.pop_payload(str(jid)) == {}  # popped once
    storage.finish_job(jid, "done", "unit")
    lst = storage.list_jobs()
    assert any(j["id"] == jid and j["status"] == "done" for j in lst)


# ------------------------------------------------------------------ runner
def main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failed = 0
    for fn in fns:
        try:
            fn()
            print(f"  ✅ {fn.__name__}")
        except Exception as e:
            failed += 1
            import traceback
            print(f"  ❌ {fn.__name__}: {type(e).__name__}: {e}")
            traceback.print_exc()
    print(f"{len(fns) - failed}/{len(fns)} passed (DB: {os.environ['GBV_DB']})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
