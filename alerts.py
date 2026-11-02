"""Alerts: notify humans when new high/critical content appears (webhook / email),
plus coordinated-burst warnings (osint.burst_sigma).

Configured via env vars only — no secrets ever stored in the database:
  GBV_WEBHOOK_URL   → JSON POST {"text": "..."} (Slack/Discord-compatible webhook)
  GBV_ALERT_EMAIL   → recipient list, comma separated
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS  (STARTTLS)
Both are optional; with neither configured nothing is sent and the audit log notes it.
"""
from __future__ import annotations

import json
import os
from email.message import EmailMessage
from smtplib import SMTP, SMTPException

import requests

import storage


def _env_rcpt() -> list[str]:
    return [a.strip() for a in os.environ.get("GBV_ALERT_EMAIL", "").split(",") if a.strip()]


def send(text: str) -> dict:
    done = {}
    url = os.environ.get("GBV_WEBHOOK_URL", "").strip()
    if url:
        try:
            r = requests.post(url, data=json.dumps({"text": text}),
                              headers={"Content-Type": "application/json"}, timeout=30)
            done["webhook"] = r.status_code
        except Exception as e:
            done["webhook"] = f"error: {e}"
    rcpt = _env_rcpt()
    if rcpt and os.environ.get("SMTP_HOST"):
        try:
            msg = EmailMessage()
            msg["To"] = ", ".join(rcpt)
            msg["Subject"] = "fb_gbv_monitor alert"
            msg.set_content(text)
            with SMTP(os.environ["SMTP_HOST"], int(os.environ.get("SMTP_PORT", "587")), timeout=60) as s:
                s.starttls()
                u, p = os.environ.get("SMTP_USER", ""), os.environ.get("SMTP_PASS", "")
                if u:
                    s.login(u, p)
                s.send_message(msg)
            done["email"] = "sent"
        except (SMTPException, OSError, ValueError) as e:
            done["email"] = f"error: {e}"
    if not done:
        done["note"] = "no alerts configured (set GBV_WEBHOOK_URL and/or GBV_ALERT_EMAIL+SMTP_*)"
    storage.audit("monitor", "alert", text[:300])
    return done


def check_run(page_id: int, run_id: int, min_confidence: float = 0.7,
              min_severities: tuple[str, ...] = ("high", "critical"),
              max_examples: int = 10) -> dict:
    """One digest for a finished run: new high/critical verdicts + burst warning.
    Returns {"text", "new_flagged": n, "burst": {...}|None, "sent": {...}} — empty
    text means nothing worth paging out."""
    from osint import get_osint

    with storage.db() as c:
        rows = c.execute("""SELECT v.item_key, v.severity, v.evidence, v.confidence, i.author
                            FROM verdicts v JOIN items i ON i.item_key=v.item_key
                            WHERE v.run_id=? AND v.verifier_kind='' AND v.is_violation=1
                              AND v.severity IN (%s)
                              AND (v.confidence>=? OR v.confidence=0)
                              AND i.page_id=?
                            ORDER BY v.confidence DESC""" % ",".join("?" * len(min_severities)),
                         (run_id, *min_severities, min_confidence, page_id)).fetchall()
    sections = []
    if rows:
        lines = [f"- [{r['severity']}|{r['confidence']:.2f}] {r['author']}: {(r['evidence'] or '(no evidence excerpt)')[:120]}"
                 for r in rows[:max_examples]]
        if len(rows) > max_examples:
            lines.append(f"- … and {len(rows) - max_examples} more")
        sections.append(f"{len(rows)} new high/critical item(s):\n" + "\n".join(lines))
    burst = None
    try:
        o = get_osint(run_id) or {}
        burst = o.get("burst")
    except Exception:
        burst = None
    if not burst:  # osint may not have run — compute directly
        try:
            import osint
            burst = osint.burst_sigma(page_id)
        except Exception:
            burst = None
    if burst:
        sections.append(f"⚠️ possible coordinated burst: {burst['latest']} flagged in the newest run "
                        f"vs baseline {burst['baseline_mean']} (+{burst['sigma']}σ "
                        f"over {burst['baseline_runs']} runs)")
    text = f"\n\n".join(sections)
    storage.audit("monitor", "alert_digest",
                  f"run {run_id}: {len(rows)} high/critical, burst={'yes' if burst else 'no'}")
    if not text:
        return {"text": "", "new_flagged": 0, "burst": burst,
                "sent": {"note": "nothing high/critical on this run"}}
    return {"text": text, "new_flagged": len(rows), "burst": burst, "sent": send(text)}
