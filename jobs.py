"""Background job engine — long work that must survive Streamlit refreshes.

Jobs live in the `jobs` SQLite table (kind, params, status running→done/failed,
progress 0..1, message, result). A single daemon worker thread claims queued jobs
atomically so it is safe with any number of Streamlit reruns.

Kinds:
  run-due    — monitor.run_due over due watchlists. Facebook cookies are passed
               through put_payload()/pop_payload() (MEMORY ONLY — never the DB,
               never disk) keyed by job id.

Notifications/alerts and OSINT happen inside run-due itself, exactly as the CLI
does — the queue is about scheduling/robustness, not about changing behavior.
"""
from __future__ import annotations

import json
import threading
import time
import traceback

import storage

_worker_started = threading.Event()


def enqueue_run_due(params: dict | None = None, payload: dict | None = None) -> int:
    """params: {'lm_url','model','batch_size','enrich','verify'} — cookies go in payload."""
    return storage.enqueue_job("run-due", params, payload)


def claim_next(wait_seconds: float = 0.0) -> dict | None:
    deadline = time.time() + wait_seconds
    while True:
        job = storage.claim_job()
        if job is not None:
            return job
        if time.time() >= deadline:
            return None
        time.sleep(0.5)


def _apply_progress(job_id: int):
    def progress(msg: str, frac: float) -> None:
        try:
            storage.set_job_progress(job_id, frac, msg)
        except Exception:
            pass
    return progress


def _run_job(job: dict) -> str:
    job_id = job["id"]
    params = job.get("params") or {}
    if isinstance(params, str):
        try:
            params = json.loads(params or "{}")
        except json.JSONDecodeError:
            params = {}
    if job["kind"] == "run-due":
        import argparse
        import monitor

        ns = argparse.Namespace(
            lm_url=params.get("lm_url") or "",
            model=params.get("model") or "",
            batch_size=int(params.get("batch_size") or 2),
            cookies="", enrich=bool(params.get("enrich", True)),
            verify=bool(params.get("verify", True)),
        )
        payload = storage.pop_payload(str(job_id))
        if payload.get("cookies"):
            import os
            os.environ["GBV_FB_COOKIES"] = payload["cookies"]
        summary = monitor.run_due(ns, progress=_apply_progress(job_id))
        return f"watchlist pass finished (exit {summary})"
    raise ValueError(f"unknown job kind: {job['kind']}")


def _worker_loop(poll: float = 2.0) -> None:
    while True:
        job = None
        try:
            job = claim_next()
        except Exception:
            time.sleep(poll)
            continue
        if job is None:
            time.sleep(poll)
            continue
        try:
            result = _run_job(job)
            storage.finish_job(job["id"], "done", result)
        except Exception as e:
            try:
                storage.finish_job(job["id"], "failed",
                                   f"{type(e).__name__}: {e}\n{traceback.format_exc()[-1500:]}")
                storage.audit("jobs", "failed",
                              f"#{job['id']} {job['kind']}: {type(e).__name__}: {e}")
            except Exception:
                pass


def ensure_worker() -> None:
    """Start the daemon worker once per process."""
    if _worker_started.is_set():
        return
    t = threading.Thread(target=_worker_loop, name="gbv-jobs", daemon=True)
    t.start()
    _worker_started.set()
