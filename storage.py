"""SQLite dataset layer — cases → runs → items → verdicts → reviews → watchlists.

Every collection/classification/review action lands here, so the tool gains memory:
delta scraping, longitudinal reports, triage and the agent all read/write this one
file (output/gbv.db by default; override with GBV_DB).

Only the stdlib is used (sqlite3 + WAL). One short-lived connection per operation
keeps it safe from Streamlit's threading model.

Schema v2 additions over v1 (applied by `_migrate` on open, old DBs upgrade in place):
  - `item_entities`      per-item named entities (the OSINT rows are now persisted)
  - `jobs`               background job queue (kind, params, status, progress)
  - `verdicts.prompt_sha`/`inherited_from` — prompt provenance + verdict-memory links
  - `items.media`        JSON list of image/video URLs found in the post
  - `osint_runs.extra`   coordination extras (templates, cross-page, burst)
  - FTS5 full-text index `items_fts` with triggers keeping it in sync
Optional encryption at rest: see encryption.py (SQLCipher if available + GBV_DB_KEY).
"""
from __future__ import annotations

import json
import os
import sqlite3
import time
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone

DB_PATH = os.environ.get("GBV_DB", os.path.join("output", "gbv.db"))

SCHEMA_VERSION = 2

SCHEMA = """
CREATE TABLE IF NOT EXISTS pages(
  id INTEGER PRIMARY KEY, url TEXT UNIQUE, label TEXT, created_at TEXT);
CREATE TABLE IF NOT EXISTS runs(
  id INTEGER PRIMARY KEY, page_id INT, started_at TEXT, finished_at TEXT,
  status TEXT, post_count INT DEFAULT 0, comment_count INT DEFAULT 0,
  model TEXT, notes TEXT);
CREATE TABLE IF NOT EXISTS items(
  item_key TEXT PRIMARY KEY,
  page_id INT, kind TEXT, parent_key TEXT, author TEXT, text TEXT,
  sha256 TEXT, permalink TEXT, time_text TEXT, collected_at TEXT, screenshot TEXT,
  media TEXT DEFAULT '',
  first_seen_run INT, last_seen_run INT, times_seen INT DEFAULT 1);
CREATE INDEX IF NOT EXISTS idx_items_page ON items(page_id);
CREATE INDEX IF NOT EXISTS idx_items_sha ON items(sha256);
CREATE TABLE IF NOT EXISTS verdicts(
  id INTEGER PRIMARY KEY, item_key TEXT, run_id INT, model TEXT,
  is_violation INT, categories TEXT, severity TEXT, target TEXT,
  language TEXT, evidence TEXT, explanation TEXT, confidence REAL,
  analysis_error TEXT, verifier_kind TEXT DEFAULT '', verifier_summary TEXT,
  prompt_sha TEXT DEFAULT '', inherited_from TEXT DEFAULT '',
  created_at TEXT);
CREATE INDEX IF NOT EXISTS idx_verdicts_item ON verdicts(item_key);
CREATE TABLE IF NOT EXISTS reviews(
  item_key TEXT PRIMARY KEY, action TEXT, reviewer TEXT, note TEXT,
  reviewed_at TEXT, verdict_id INT);
CREATE TABLE IF NOT EXISTS item_entities(
  id INTEGER PRIMARY KEY, item_key TEXT, run_id INT, type TEXT, name TEXT);
CREATE INDEX IF NOT EXISTS idx_ent_item ON item_entities(item_key);
CREATE INDEX IF NOT EXISTS idx_ent_name ON item_entities(name);
CREATE TABLE IF NOT EXISTS watchlists(
  id INTEGER PRIMARY KEY, url TEXT UNIQUE, label TEXT, cadence_hours REAL DEFAULT 24,
  max_posts INT DEFAULT 50, include_comments INT DEFAULT 1, deep_comments INT DEFAULT 0,
  capture_shots INT DEFAULT 1, active INT DEFAULT 1,
  last_run_id INT, next_due TEXT);
CREATE TABLE IF NOT EXISTS proposals(
  id INTEGER PRIMARY KEY, created_at TEXT, type TEXT, value TEXT, why TEXT,
  status TEXT DEFAULT 'pending', decided_by TEXT, decided_at TEXT);
CREATE TABLE IF NOT EXISTS case_files(
  page_id INT PRIMARY KEY, name TEXT, body TEXT DEFAULT '', updated_at TEXT);
CREATE TABLE IF NOT EXISTS audit(
  id INTEGER PRIMARY KEY, ts TEXT, actor TEXT, action TEXT, detail TEXT);
CREATE TABLE IF NOT EXISTS embeddings(
  item_key TEXT PRIMARY KEY, model TEXT, dim INT, vec BLOB);
CREATE TABLE IF NOT EXISTS osint_runs(
  run_id INT PRIMARY KEY, clusters TEXT, near_dups TEXT, entities TEXT,
  cadence TEXT, amplifiers TEXT, graphml_path TEXT, extra TEXT DEFAULT '',
  done_at TEXT);
CREATE TABLE IF NOT EXISTS jobs(
  id INTEGER PRIMARY KEY, kind TEXT, params TEXT DEFAULT '{}',
  status TEXT DEFAULT 'queued', progress REAL DEFAULT 0, message TEXT DEFAULT '',
  result TEXT DEFAULT '', created_at TEXT, updated_at TEXT);
"""

_CASE_KEYS = ("observations", "hypotheses", "gaps", "legacy")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _connect(path: str) -> sqlite3.Connection:
    """Plain sqlite3 connect, or transparent SQLCipher via encryption.open_db()
    when GBV_DB_KEY is set and a SQLCipher driver is installed."""
    try:
        import encryption
        return encryption.open_db(path)
    except ImportError:
        return sqlite3.connect(path, timeout=30)


@contextmanager
def db(path: str = None):
    conn = _connect(path or DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.executescript(SCHEMA)
        _migrate(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def _add_col(conn: sqlite3.Connection, table: str, col: str, decl: str) -> None:
    cols = {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}
    if col not in cols:
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} {decl}")


def _migrate(conn: sqlite3.Connection) -> None:
    ver = conn.execute("PRAGMA user_version").fetchone()[0]
    if ver >= SCHEMA_VERSION:
        return
    _add_col(conn, "verdicts", "prompt_sha", "TEXT DEFAULT ''")
    _add_col(conn, "verdicts", "inherited_from", "TEXT DEFAULT ''")
    _add_col(conn, "items", "media", "TEXT DEFAULT ''")
    _add_col(conn, "osint_runs", "extra", "TEXT DEFAULT ''")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_items_sha ON items(sha256)")
    # FTS5 is optional in exotic sqlite builds — search falls back to LIKE without it.
    try:
        conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS items_fts "
                     "USING fts5(text, content='items', content_rowid='rowid')")
        conn.executescript("""
        CREATE TRIGGER IF NOT EXISTS items_fts_ins AFTER INSERT ON items BEGIN
          INSERT INTO items_fts(rowid, text) VALUES (new.rowid, new.text);
        END;
        CREATE TRIGGER IF NOT EXISTS items_fts_del AFTER DELETE ON items BEGIN
          INSERT INTO items_fts(items_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
        END;
        CREATE TRIGGER IF NOT EXISTS items_fts_upd AFTER UPDATE OF text ON items BEGIN
          INSERT INTO items_fts(items_fts, rowid, text) VALUES ('delete', old.rowid, old.text);
          INSERT INTO items_fts(rowid, text) VALUES (new.rowid, new.text);
        END;
        """)
        n_items = conn.execute("SELECT COUNT(*) FROM items").fetchone()[0]
        n_fts = conn.execute("SELECT COUNT(*) FROM items_fts").fetchone()[0]
        if n_items and not n_fts:
            conn.execute("INSERT INTO items_fts(rowid, text) SELECT rowid, text FROM items")
    except sqlite3.Error:
        pass  # no FTS5 in this sqlite build
    conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")


def audit(actor: str, action: str, detail: str = "", path: str = None) -> None:
    with db(path) as c:
        c.execute("INSERT INTO audit(ts, actor, action, detail) VALUES (?,?,?,?)",
                  (_now(), actor, action, detail))


# ---------------------------------------------------------------------- pages/runs
def get_or_create_page(url: str, label: str = "", path: str = None) -> int:
    url = url.strip()
    with db(path) as c:
        row = c.execute("SELECT id FROM pages WHERE url=?", (url,)).fetchone()
        if row:
            return row["id"]
        cur = c.execute("INSERT INTO pages(url, label, created_at) VALUES (?,?,?)",
                        (url, label or url, _now()))
        return cur.lastrowid


def start_run(page_id: int, page_url: str = "", notes: str = "") -> int:
    with db() as c:
        cur = c.execute(
            "INSERT INTO runs(page_id, started_at, status, notes) VALUES (?,?, 'running', ?)",
            (page_id, _now(), notes))
        return cur.lastrowid


def finish_run(run_id: int, status: str = "done", post_count: int = 0,
               comment_count: int = 0, model: str = "") -> None:
    with db() as c:
        c.execute("UPDATE runs SET finished_at=?, status=?, post_count=?, comment_count=?, model=? "
                  "WHERE id=?", (_now(), status, post_count, comment_count, model, run_id))


def list_runs(page_id: int, limit: int = 50) -> list[dict]:
    with db() as c:
        rows = c.execute("SELECT * FROM runs WHERE page_id=? ORDER BY id DESC LIMIT ?",
                         (page_id, limit)).fetchall()
        return [dict(r) for r in rows]


def get_run(run_id: int) -> dict | None:
    with db() as c:
        r = c.execute("SELECT * FROM runs WHERE id=?", (run_id,)).fetchone()
        return dict(r) if r else None


def latest_run(page_id: int) -> dict | None:
    runs = list_runs(page_id, limit=1)
    return runs[0] if runs else None


# ---------------------------------------------------------------------- items
def upsert_items(page_id: int, run_id: int, items: list[dict], path: str = None) -> dict:
    """items = analyzer.flatten(...) output (adding 'id','kind','parent_id',...).

    Dedupe key = page_id + sha256 of raw text (scraper-integrity anchored).
    Returns {"added": int, "updated": int}.
    """
    stats = {"added": 0, "updated": 0}
    with db(path) as c:
        # first pass: capture post ids → keys so comments can link to their parent post
        post_key_by_id: dict[str, str] = {}
        for it in items:
            if it["kind"] == "post":
                post_key_by_id[it["id"]] = f"{page_id}:{it['sha256'] or it['id']}"
        for it in items:
            key = f"{page_id}:{it['sha256'] or it['id']}"
            parent_key = ""
            if it["kind"] == "comment":
                post_id = (it.get("parent_id") or "").split("c")[0]
                parent_key = post_key_by_id.get(post_id, "")
            row = c.execute("SELECT * FROM items WHERE item_key=?", (key,)).fetchone()
            if row:
                new = max(row["text"] or "", it["text"] or "")
                c.execute("""UPDATE items SET text=?, parent_key=?,
                             screenshot=CASE WHEN ?<>'' THEN ? ELSE screenshot END,
                             media=?,
                             last_seen_run=?, times_seen=times_seen+1 WHERE item_key=?""",
                          (new, parent_key, it.get("screenshot", ""), it.get("screenshot", ""),
                           json.dumps(it.get("media") or [], ensure_ascii=False),
                           run_id, key))
                stats["updated"] += 1
            else:
                c.execute("""INSERT INTO items(item_key, page_id, kind, parent_key, author, text,
                             sha256, permalink, time_text, collected_at, screenshot, media,
                             first_seen_run, last_seen_run, times_seen)
                             VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                          (key, page_id, it["kind"], parent_key, it["author"], it["text"], it["sha256"],
                           it["permalink"], it.get("time_text", ""), it.get("collected_at", ""),
                           it.get("screenshot", ""), json.dumps(it.get("media") or [], ensure_ascii=False),
                           run_id, run_id, 1))
                stats["added"] += 1
    return stats


def add_verdicts(run_id: int, rows: list[dict], key_of, model: str,
                 verifier_kind: str = "", prompt_sha: str = "",
                 inherited_from: str = "") -> None:
    """rows: analyzer verdicts (must carry id/text refs); key_of(row) -> item_key."""
    with db() as c:
        for r in rows:
            c.execute("""INSERT INTO verdicts(item_key, run_id, model, is_violation, categories,
                         severity, target, language, evidence, explanation, confidence,
                         analysis_error, verifier_kind, verifier_summary,
                         prompt_sha, inherited_from, created_at)
                         VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                      (key_of(r), run_id, model + (f" [{verifier_kind}]" if verifier_kind else ""),
                       1 if r.get("is_violation") else 0,
                       json.dumps(r.get("categories", [])), r.get("severity", "none"),
                       r.get("target", "other"), r.get("language", "other"),
                       r.get("evidence", ""), r.get("explanation", ""),
                       float(r.get("confidence", 0) or 0), r.get("analysis_error", ""),
                       verifier_kind, r.get("verifier_summary", ""),
                       prompt_sha or r.get("prompt_sha", ""),
                       inherited_from or r.get("inherited_from", ""), _now()))


def current_verdicts(page_id: int, run_id: int | None = None,
                     flagged_only: bool = False, limit: int = 500) -> list[dict]:
    """Join latest items with their latest (non-verifier) verdict for triage."""
    with db() as c:
        q = """
        SELECT i.*, v.run_id AS v_run, v.is_violation AS is_violation, v.categories AS categories,
               v.severity, v.target, v.language, v.evidence, v.explanation, v.confidence,
               v.analysis_error, v.model AS verdict_model, v.verifier_summary AS verifier_summary,
               v.prompt_sha AS prompt_sha, v.inherited_from AS inherited_from,
               r.action AS review_action,
               r.reviewer, r.note
        FROM items i
        LEFT JOIN verdicts v ON v.item_key=i.item_key AND v.verifier_kind=''
          AND v.id IN (SELECT MAX(id) FROM verdicts WHERE verifier_kind='' GROUP BY item_key)
        LEFT JOIN reviews r ON r.item_key=i.item_key
        WHERE i.page_id=? """
        args: list = [page_id]
        if run_id:
            q += " AND i.last_seen_run=? "
            args.append(run_id)
        if flagged_only:
            q += " AND v.is_violation=1 "
        q += " ORDER BY v.is_violation DESC, v.confidence DESC LIMIT ?"
        args.append(limit)
        out = []
        for row in c.execute(q, args):
            d = dict(row)
            try:
                d["categories"] = json.loads(d["categories"] or "[]")
            except json.JSONDecodeError:
                d["categories"] = []
            try:
                d["media"] = json.loads(d.get("media") or "[]")
            except (json.JSONDecodeError, KeyError):
                d["media"] = []
            d["is_violation"] = bool(d["is_violation"])
            out.append(d)
        return out


def known_sha_keys(page_id: int) -> set[str]:
    with db() as c:
        return {r["item_key"] for r in c.execute(
            "SELECT item_key FROM items WHERE page_id=?", (page_id,))}


def set_review(item_key: str, action: str, reviewer: str = "", note: str = "",
               verdict_id: int | None = None) -> None:
    if action == "unset":
        with db() as c:
            c.execute("DELETE FROM reviews WHERE item_key=?", (item_key,))
        return
    with db() as c:
        c.execute("""INSERT INTO reviews(item_key, action, reviewer, note, reviewed_at, verdict_id)
                     VALUES (?,?,?,?,?,?)
                     ON CONFLICT(item_key) DO UPDATE SET action=excluded.action,
                       reviewer=excluded.reviewer, note=excluded.note,
                       reviewed_at=excluded.reviewed_at, verdict_id=excluded.verdict_id""",
                  (item_key, action, reviewer, note, _now(), verdict_id))


def pending_review(page_id: int, run_id: int | None = None) -> list[dict]:
    rows = [r for r in current_verdicts(page_id, run_id)
            if r.get("is_violation") and not r.get("review_action")]
    return rows


def review_stats(page_id: int) -> dict:
    with db() as c:
        n_conf = c.execute("""SELECT COUNT(*) n FROM reviews r JOIN items i
                              ON i.item_key=r.item_key WHERE i.page_id=? AND r.action='confirm'""",
                           (page_id,)).fetchone()["n"]
        n_rej = c.execute("""SELECT COUNT(*) n FROM reviews r JOIN items i
                              ON i.item_key=r.item_key WHERE i.page_id=? AND r.action='reject'""",
                           (page_id,)).fetchone()["n"]
        return {"confirmed": n_conf, "rejected": n_rej}


# ------------------------------------------------------- review quality / calibration
def category_review_stats(page_id: int, min_n: int = 3) -> list[dict]:
    """Per-category human precision: of the items a human reviewed that carried this
    category on the AI verdict, how many were confirmed vs rejected."""
    with db() as c:
        q = """
        SELECT v.categories, r.action FROM reviews r
        JOIN items i ON i.item_key=r.item_key AND i.page_id=?
        LEFT JOIN verdicts v ON v.item_key=i.item_key AND v.verifier_kind=''
          AND v.id IN (SELECT MAX(id) FROM verdicts WHERE verifier_kind='' GROUP BY item_key)
        WHERE r.action IN ('confirm','reject')"""
        per: dict[str, dict] = {}
        for r in c.execute(q, (page_id,)):
            try:
                cats = json.loads(r["categories"] or "[]")
            except (json.JSONDecodeError, TypeError):
                cats = []
            rec = per.setdefault(r["action"], {})
            for cat in cats:
                rec.setdefault(cat, 0)
                rec[cat] += 1
    out = []
    conf, rej = per.get("confirm", {}), per.get("reject", {})
    for cat in sorted(set(conf) | set(rej)):
        a, b = conf.get(cat, 0), rej.get(cat, 0)
        n = a + b
        if n >= min_n:
            out.append({"category": cat, "confirmed": a, "rejected": b,
                        "human_precision": round(a / n, 3)})
    out.sort(key=lambda d: d["human_precision"])
    return out


def human_ai_agreement(page_id: int) -> dict:
    """Cohen's κ between the (latest) AI flag and the human review decision, over all
    reviewed items of the page — the headline number for "can we trust the machine?". """
    with db() as c:
        q = """
        SELECT v.is_violation AS ai_flag, r.action AS human FROM reviews r
        JOIN items i ON i.item_key=r.item_key AND i.page_id=?
        LEFT JOIN verdicts v ON v.item_key=i.item_key AND v.verifier_kind=''
          AND v.id IN (SELECT MAX(id) FROM verdicts WHERE verifier_kind='' GROUP BY item_key)
        WHERE r.action IN ('confirm','reject')"""
        cells = {True: {"confirm": 0, "reject": 0}, False: {"confirm": 0, "reject": 0}}
        for r in c.execute(q, (page_id,)):
            cells[bool(r["ai_flag"])][r["human"]] += 1
    tp, fp, fn_, tn = (cells[True]["confirm"], cells[True]["reject"],
                       cells[False]["confirm"], cells[False]["reject"])
    n = tp + fp + fn_ + tn
    if not n:
        return {"n": 0, "kappa": None}
    po = (tp + tn) / n
    ai_pos, hum_pos = (tp + fp) / n, (tp + fn_) / n
    pe = ai_pos * hum_pos + (1 - ai_pos) * (1 - hum_pos)
    kappa = (po - pe) / (1 - pe) if pe < 1 else 1.0
    return {"n": n, "kappa": round(kappa, 3), "true_pos": tp, "false_pos": fp,
            "false_neg": fn_, "true_neg": tn}


# ---------------------------------------------------------------------- entities
def replace_item_entities(run_id: int, rows: list[dict]) -> int:
    """Persist per-item entity rows: [{"item_key","type","name"}]. Recompute-safe."""
    with db() as c:
        c.execute("DELETE FROM item_entities WHERE run_id=?", (run_id,))
        for r in rows:
            c.execute("INSERT INTO item_entities(item_key, run_id, type, name) VALUES (?,?,?,?)",
                      (r["item_key"], run_id, r["type"], r["name"]))
    return len(rows)


def entity_graph(entity: str, page_id: int | None = None, limit: int = 30) -> dict:
    """Everything linked to one entity name (LIKE match): counts + item keys."""
    with db() as c:
        q = """SELECT e.type, e.name, COUNT(*) AS n, GROUP_CONCAT(e.item_key) AS keys
               FROM item_entities e """
        args: list = []
        if page_id:
            q += "JOIN items i ON i.item_key=e.item_key AND i.page_id=? "
            args.append(page_id)
        q += "WHERE e.name LIKE ? GROUP BY e.type, e.name ORDER BY n DESC LIMIT ?"
        args += [f"%{entity}%", limit]
        matches = [{"type": r["type"], "name": r["name"], "count": r["n"],
                    "item_keys": (r["keys"] or "").split(",")[:50]}
                   for r in c.execute(q, args)]
    return {"entity": entity, "matches": matches}


def author_entity_edges(page_id: int, run_id: int | None = None, limit: int = 300) -> list[dict]:
    """Author → entity mention edges (person/org) for the amplification graph."""
    with db() as c:
        q = """SELECT i.author AS author, e.name AS name, COUNT(*) AS n
               FROM item_entities e JOIN items i ON i.item_key=e.item_key
               WHERE i.page_id=? AND e.type IN ('person','org') AND i.author IS NOT NULL"""
        args: list = [page_id]
        if run_id:
            q += " AND i.last_seen_run=?"
            args.append(run_id)
        q += " GROUP BY i.author, e.name ORDER BY n DESC LIMIT ?"
        args.append(limit)
        return [{"source": r["author"], "target": r["name"], "weight": r["n"]}
                for r in c.execute(q, args)]


# ---------------------------------------------------------------------- coordination
def cross_page_matches(limit: int = 20) -> list[dict]:
    """Same text (identical sha256) collected on 2+ different watched pages — the
    strongest local coordination signal. Global across the whole dataset."""
    with db() as c:
        rows = c.execute("""SELECT sha256, COUNT(DISTINCT page_id) AS pages, COUNT(*) AS n,
                                   MIN(item_key) AS key, GROUP_CONCAT(DISTINCT page_id) AS page_ids
                            FROM items WHERE kind='post' AND sha256 IS NOT NULL
                            GROUP BY sha256 HAVING pages>=2 ORDER BY n DESC LIMIT ?""",
                         (limit,)).fetchall()
        out = []
        for r in rows:
            pages = c.execute("SELECT id, label FROM pages WHERE id IN (%s)"
                              % ",".join("?" * len(r["page_ids"].split(","))),
                              tuple(int(p) for p in r["page_ids"].split(","))).fetchall()
            out.append({"sha256": r["sha256"], "n_items": r["n"], "pages": r["pages"],
                        "page_labels": [p["label"] or f"page {p['id']}" for p in pages],
                        "sample_item_key": r["key"]})
    return out


def flagged_per_run(page_id: int) -> list[dict]:
    """Flagged-item count per run, oldest first — the basis of burst detection."""
    with db() as c:
        rows = c.execute("""SELECT v.run_id AS run_id, COUNT(*) AS n
                            FROM verdicts v JOIN items i ON i.item_key=v.item_key
                            WHERE i.page_id=? AND v.verifier_kind='' AND v.is_violation=1
                            GROUP BY v.run_id ORDER BY v.run_id""", (page_id,)).fetchall()
        return [{"run_id": r["run_id"], "n": r["n"]} for r in rows]


# ---------------------------------------------------------------------- FTS search
def _fts_ok(c) -> bool:
    try:
        c.execute("SELECT 1 FROM items_fts LIMIT 1")
        return True
    except sqlite3.Error:
        return False


def search_corpus(query: str, page_id: int | None = None, flagged_only: bool = False,
                  kind: str | None = None, limit: int = 10) -> list[dict]:
    """Full-text (FTS5) search over items joined with their latest verdict, with a
    LIKE fallback when FTS is unavailable or the query is not a valid match expr."""
    with db() as c:
        base_cols = """i.item_key, i.page_id, i.kind, i.author, substr(i.text,1,300) AS excerpt,
               i.permalink, v.is_violation, v.severity, v.categories, v.explanation, v.run_id AS v_run"""
        join = """
        FROM items i
        LEFT JOIN verdicts v ON v.item_key=i.item_key AND v.verifier_kind=''
          AND v.id IN (SELECT MAX(id) FROM verdicts WHERE verifier_kind='' GROUP BY item_key)"""
        wheres, args = [], []
        if page_id:
            wheres.append("i.page_id=?")
            args.append(page_id)
        if flagged_only:
            wheres.append("v.is_violation=1")
        if kind in ("post", "comment"):
            wheres.append("i.kind=?")
            args.append(kind)
        rows, used_fts = [], False
        if query:
            if _fts_ok(c):
                try:
                    q = f"""SELECT {base_cols}, rank FROM items_fts
                        JOIN items i ON i.rowid = items_fts.rowid
                        LEFT JOIN verdicts v ON v.item_key=i.item_key AND v.verifier_kind=''
                          AND v.id IN (SELECT MAX(id) FROM verdicts WHERE verifier_kind='' GROUP BY item_key)
                        WHERE items_fts MATCH ? """
                    fargs = [query]
                    if wheres:
                        q += " AND " + " AND ".join(wheres)
                        fargs += args
                    q += " ORDER BY rank LIMIT ?"
                    fargs.append(limit)
                    rows = [dict(r) for r in c.execute(q, fargs)]
                    used_fts = True
                except sqlite3.Error:
                    rows, used_fts = [], False
            if not used_fts:  # LIKE fallback (also covers FTS5-adjacent syntax errors)
                q = f"SELECT {base_cols} {join} WHERE i.text LIKE ? "
                largs: list = [f"%{query}%"]
                if wheres:
                    q += " AND " + " AND ".join(wheres)
                    largs += args
                q += " ORDER BY i.times_seen DESC LIMIT ?"
                largs.append(limit)
                rows = [dict(r) for r in c.execute(q, largs)]
        else:
            q = f"SELECT {base_cols} {join} "
            if wheres:
                q += " WHERE " + " AND ".join(wheres)
            q += " ORDER BY i.times_seen DESC LIMIT ?"
            args.append(limit)
            rows = [dict(r) for r in c.execute(q, args)]
    for d in rows:
        try:
            d["categories"] = json.loads(d["categories"] or "[]")
        except (json.JSONDecodeError, TypeError):
            d["categories"] = []
        d.pop("rank", None)
    return rows


def item_full(item_key: str) -> dict | None:
    with db() as c:
        r = c.execute("SELECT * FROM items WHERE item_key=?", (item_key,)).fetchone()
        if not r:
            return None
        d = dict(r)
        try:
            d["media"] = json.loads(d.get("media") or "[]")
        except (json.JSONDecodeError, KeyError):
            d["media"] = []
        v = c.execute("""SELECT * FROM verdicts WHERE item_key=? AND verifier_kind=''
                         ORDER BY id DESC LIMIT 1""", (item_key,)).fetchone()
        d["verdict"] = dict(v) if v else None
        return d


# ---------------------------------------------------------------------- case files
def case_file_get(page_id: int) -> dict:
    """Structured agent memory: {"observations": [...], "hypotheses": [...], "gaps": [...],
    "legacy": "pre-JSON free text"}. Old markdown case files are preserved as legacy."""
    body = get_case_file(page_id)
    cf = {"observations": [], "hypotheses": [], "gaps": [], "legacy": body}
    if body and body.lstrip().startswith("{"):
        try:
            obj = json.loads(body)
            if isinstance(obj, dict):
                cf = {k: (obj.get(k) if isinstance(obj.get(k), (list, str)) else
                          ([obj[k]] if obj.get(k) else []))
                      for k in _CASE_KEYS}
                cf.setdefault("legacy", "")
        except json.JSONDecodeError:
            pass
    cf.setdefault("legacy", "")
    return cf


def case_file_apply(page_id: int, fn, name: str = "Autonomous case file") -> dict:
    """Read-modify-write the structured case file with fn(cf) mutating the dict."""
    cf = case_file_get(page_id)
    fn(cf)
    write_case_file(page_id, json.dumps(cf, ensure_ascii=False, indent=1), name)
    return cf


def case_file_render(page_id: int) -> str:
    """Human/markdown view of the structured case file (agent prompt + reports)."""
    cf = case_file_get(page_id)
    lines = []
    for h in cf.get("hypotheses") or []:
        lines.append(f"- [{h.get('status', 'open')}] {h.get('text', '')}")
    for o in cf.get("observations") or []:
        body = o if isinstance(o, str) else o.get("text", "")
        anchor = "" if not isinstance(o, dict) else (f" ({o.get('anchor')})" if o.get("anchor") else "")
        lines.append(f"- OBS {body}{anchor}")
    for g in cf.get("gaps") or []:
        lines.append(f"- GAP {g}")
    if cf.get("legacy"):
        lines.append(cf["legacy"])
    return "\n".join(lines).strip()


# ---------------------------------------------------------------------- watchlists
def add_watchlist(url: str, label: str = "", cadence_hours: float = 24,
                  max_posts: int = 50, include_comments: bool = True,
                  deep_comments: bool = False, capture_shots: bool = True,
                  active: bool = True) -> int:
    with db() as c:
        try:
            cur = c.execute("""INSERT INTO watchlists(url,label,cadence_hours,max_posts,
                               include_comments,deep_comments,capture_shots,active,next_due)
                               VALUES (?,?,?,?,?,?,?,?,?)""",
                            (url.strip(), label or url.strip(),
                             float(cadence_hours), int(max_posts), int(include_comments),
                             int(deep_comments), int(capture_shots), int(active), _now()))
            return cur.lastrowid
        except sqlite3.IntegrityError:
            c.execute("""UPDATE watchlists SET label=?, cadence_hours=?, max_posts=?, active=1
                         WHERE url=?""", (label or url, float(cadence_hours), int(max_posts), url.strip()))
            return c.execute("SELECT id FROM watchlists WHERE url=?", (url.strip(),)).fetchone()["id"]


def list_watchlists(active_only: bool = False) -> list[dict]:
    with db() as c:
        q = "SELECT * FROM watchlists" + (" WHERE active=1" if active_only else "") + " ORDER BY id"
        return [dict(r) for r in c.execute(q)]


def due_watchlists(now: datetime | None = None) -> list[dict]:
    now = now or datetime.now(timezone.utc)
    out = []
    for w in list_watchlists(active_only=True):
        if not w["next_due"]:
            out.append(w)
            continue
        try:
            due = datetime.fromisoformat(w["next_due"])
            if due.tzinfo is None:
                due = due.replace(tzinfo=timezone.utc)
            if due <= now:
                out.append(w)
        except ValueError:
            out.append(w)
    return out


def mark_watchlist_run(watch_id: int, run_id: int, cadence_hours: float) -> None:
    nxt = (datetime.now(timezone.utc) + timedelta(hours=cadence_hours)).isoformat()
    with db() as c:
        c.execute("UPDATE watchlists SET last_run_id=?, next_due=? WHERE id=?",
                  (run_id, nxt, watch_id))


def toggle_watchlist(watch_id: int, active: bool) -> None:
    with db() as c:
        c.execute("UPDATE watchlists SET active=? WHERE id=?", (int(active), watch_id))


def delete_watchlist(watch_id: int) -> None:
    with db() as c:
        c.execute("DELETE FROM watchlists WHERE id=?", (watch_id,))


# ---------------------------------------------------------------------- jobs queue
def enqueue_job(kind: str, params: dict | None = None, payload: dict | None = None) -> int:
    """Queue a background job. `payload` (e.g. cookies) is held in MEMORY ONLY —
    never written to the database — and can be fetched once with pop_payload."""
    with db() as c:
        cur = c.execute("INSERT INTO jobs(kind, params, status, created_at, updated_at) "
                        "VALUES (?,?, 'queued', ?, ?)",
                        (kind, json.dumps(params or {}, ensure_ascii=False), _now(), _now()))
        jid = cur.lastrowid
    if payload is not None:
        _put_payload(str(jid), payload)
    audit("jobs", "enqueue", f"{kind} #{jid}")
    return jid


def claim_job() -> dict | None:
    """Atomically claim the oldest queued job (single worker at a time)."""
    with db() as c:
        row = c.execute("SELECT * FROM jobs WHERE status='queued' ORDER BY id LIMIT 1").fetchone()
        if not row:
            return None
        cur = c.execute("UPDATE jobs SET status='running', updated_at=? WHERE id=? AND status='queued'",
                        (_now(), row["id"]))
        if cur.rowcount != 1:  # someone else claimed it between SELECT and UPDATE
            return None
        fresh = c.execute("SELECT * FROM jobs WHERE id=?", (row["id"],)).fetchone()
        d = dict(fresh)
    try:
        d["params"] = json.loads(d.get("params") or "{}")
    except json.JSONDecodeError:
        d["params"] = {}
    return d


def set_job_progress(job_id: int, progress: float, message: str = "") -> None:
    with db() as c:
        c.execute("UPDATE jobs SET progress=?, message=?, updated_at=? WHERE id=?",
                  (min(max(float(progress), 0.0), 1.0), message[:500], _now(), job_id))


def finish_job(job_id: int, status: str, result: str = "") -> None:
    with db() as c:
        c.execute("UPDATE jobs SET status=?, result=?, progress=1.0, updated_at=? WHERE id=?",
                  (status, result[:2000], _now(), job_id))


def list_jobs(limit: int = 20) -> list[dict]:
    with db() as c:
        rows = c.execute("SELECT * FROM jobs ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        out = []
        for r in rows:
            d = dict(r)
            try:
                d["params"] = json.loads(d.get("params") or "{}")
            except json.JSONDecodeError:
                d["params"] = {}
            out.append(d)
        return out


# ---------------------------------------------------------------- job payload secrets
# Ephemeral, in-memory payload registry (e.g. Facebook cookies for a queued scrape).
# Contents never touch the database or disk; entries expire after 24 h of idleness.
_PAYLOADS: dict[str, tuple[dict, float]] = {}
_PAYLOAD_TTL = 24 * 3600.0


def _put_payload(key: str, payload: dict) -> None:
    _PAYLOADS[key] = (payload, time.time())


def pop_payload(key: str) -> dict:
    item = _PAYLOADS.pop(key, None)
    if not item:
        return {}
    payload, _ts = item
    return payload if time.time() - _ts <= _PAYLOAD_TTL else {}


# ---------------------------------------------------------------------- agent
def add_proposal(p_type: str, value: str, why: str) -> int:
    with db() as c:
        dup = c.execute("SELECT id FROM proposals WHERE type=? AND value=? AND status='pending'",
                        (p_type, value)).fetchone()
        if dup:
            return dup["id"]
        cur = c.execute("INSERT INTO proposals(created_at, type, value, why) VALUES (?,?,?,?)",
                        (_now(), p_type, value, why))
        return cur.lastrowid


def list_proposals(status: str | None = "pending") -> list[dict]:
    with db() as c:
        q = "SELECT * FROM proposals"
        args: list = []
        if status:
            q += " WHERE status=?"
            args.append(status)
        q += " ORDER BY id DESC LIMIT 100"
        return [dict(r) for r in c.execute(q, args)]


def decide_proposal(prop_id: int, action: str, by: str) -> None:
    if action not in ("approved", "dismissed"):
        raise ValueError("action must be approved/dismissed")
    with db() as c:
        c.execute("UPDATE proposals SET status=?, decided_by=?, decided_at=? WHERE id=?",
                  (action, by, _now(), prop_id))
        row = c.execute("SELECT type, value FROM proposals WHERE id=?", (prop_id,)).fetchone()
    audit(by, f"proposal_{action}", f"{row['type']}: {row['value']}")


def get_case_file(page_id: int, name: str = "") -> str:
    with db() as c:
        r = c.execute("SELECT body FROM case_files WHERE page_id=?", (page_id,)).fetchone()
        return r["body"] if r else ""


def write_case_file(page_id: int, body: str, name: str = "Autonomous case file") -> None:
    with db() as c:
        c.execute("""INSERT INTO case_files(page_id, name, body, updated_at) VALUES (?,?,?,?)
                     ON CONFLICT(page_id) DO UPDATE SET body=excluded.body,
                       updated_at=excluded.updated_at""", (page_id, name, body, _now()))
    audit("agent", "case_file_update", f"page {page_id}, {len(body)} chars")


# ---------------------------------------------------------------------- agent helpers
def longitudinal(page_id: int, run_id: int) -> dict:
    """What changed vs the previous run: new items, new flags, severity flips."""
    with db() as c:
        prev = c.execute("""SELECT id FROM runs WHERE page_id=? AND id<? AND status='done'
                            ORDER BY id DESC LIMIT 1""", (page_id, run_id)).fetchone()
        if not prev:
            return {"first_run": True}
        p = prev["id"]
        new_items = c.execute("SELECT COUNT(*) n FROM items WHERE page_id=? AND first_seen_run=?",
                              (page_id, run_id)).fetchone()["n"]
        q = """SELECT i.item_key, v.is_violation, v.severity FROM verdicts v JOIN items i
               ON i.item_key=v.item_key WHERE v.run_id=? AND v.verifier_kind=''"""
        cur = {r["item_key"]: dict(is_violation=r["is_violation"], severity=r["severity"])
               for r in c.execute(q, (run_id,))}
        old = {r["item_key"]: dict(is_violation=r["is_violation"], severity=r["severity"])
               for r in c.execute(q, (p,))}
        new_flags = [k for k, v in cur.items() if v["is_violation"] and not old.get(k, {}).get("is_violation")]
        dropped = [k for k, v in old.items() if v["is_violation"] and not cur.get(k, {}).get("is_violation")]
        flips = [k for k in set(cur) & set(old) if cur[k]["is_violation"] and old[k]["is_violation"]
                 and cur[k]["severity"] != old[k]["severity"]]
        return {"first_run": False, "prev_run": p, "new_items": new_items,
                "new_flags": len(new_flags), "dropped_flags": len(dropped),
                "severity_flips": len(flips)}
