"""Multimodal screening — local vision-model pass over images found in flagged posts.

GBV evidence is often the IMAGE (NCII, sextortion screenshots), not the caption.
The scraper already collects per-post media URLs (scraper.py → items.media).
When a vision model is available this module:

  1. downloads the first images of FLAGGED posts (cookies from GBV_FB_COOKIES,
     held in RAM only, never stored),
  2. asks the local vision model to classify the image in conservative,
     evidence-oriented terms: {risk, content, note, indicators},
  3. returns {item_key: {...}} for the UI + reports, and writes
     output/vision_<stamp>.json for the case record.

It never claims to identify people or verify consent; it produces a triage lead
that a human must confirm. The packaged exe ships its own vision-capable model
(Qwen2.5-VL: llama-server is started with the model's mmproj tower) so no extra
setup is needed; with LM Studio, set GBV_VISION_MODEL to a downloaded VLM.
Without a vision model / cookies, it is a no-op.
"""
from __future__ import annotations

import base64
import json
import os
from datetime import datetime, timezone

import requests

import storage

VISION_SYSTEM_PROMPT = """You are a content-safety screening assistant for a human-rights team that
documents gender-based cyberviolence. You will receive ONE image from a social media post.
Classify what the image depicts, in conservative evidence-oriented terms. You CANNOT identify
people and cannot determine consent or age with certainty — do not speculate beyond visible content.

Reply ONLY with JSON:
{"content": "brief neutral description of what is visible (1-2 sentences)",
 "indicators": ["e.g. screenshot-of-chat", "suggestive-imagery", "document/id-card", "no concerning visual"],
 "risk": "none|low|medium|high|critical",
 "note": "why this may matter for GBV documentation (English, short)"}
"""  # noqa: E501

MAX_IMAGES_PER_ITEM = 3
MAX_ITEMS = 15
MAX_BYTES = 8 * 1024 * 1024


def pick_vision_model(lm, model_hint: str = "") -> str:
    """First model with vision capability (env GBV_VISION_MODEL overrides)."""
    env = os.environ.get("GBV_VISION_MODEL", "").strip()
    if env:
        return env
    try:
        for m in lm.list_models():
            caps = (m.get("capabilities") or {})
            if m.get("type") == "vlm" or (caps.get("vision") if isinstance(caps, dict) else False):
                return m["key"]
    except Exception:
        pass
    return ""


def _session(cookies_raw: str) -> requests.Session:
    from scraper import parse_cookies
    s = requests.Session()
    try:
        for c in parse_cookies(cookies_raw):
            s.cookies.set(c["name"], c["value"], domain=c.get("domain") or ".facebook.com")
    except Exception:
        pass
    s.headers["User-Agent"] = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                               "(KHTML, like Gecko) Chrome/124.0 Safari/537.36")
    return s


def _download_image(s: requests.Session, url: str) -> str | None:
    try:
        r = s.get(url, timeout=30)
        if r.status_code != 200 or len(r.content) > MAX_BYTES:
            return None
        ctype = r.headers.get("content-type", "")
        if not ctype.startswith("image/"):
            return None
        return f"data:{ctype};base64," + base64.b64encode(r.content).decode()
    except Exception:
        return None


def _engine_base(lm) -> str:
    """The base URL for OpenAI-shaped calls for whichever client is in use."""
    base = getattr(lm, "base", "")
    if base:
        return base.rstrip("/")
    try:
        import local_llm
        u = local_llm.chat_url()
        if u:
            return u.rstrip("/")
    except Exception:
        pass
    return os.environ.get("GBV_LLM_URL", "").rstrip("/")


def _ask_vision(lm, model: str, data_uri: str) -> dict | None:
    """OpenAI-shaped /v1/chat/completions call with an image_url content part."""
    base = _engine_base(lm)
    if not base:
        return None
    headers = {}
    try:
        headers = lm._headers()  # both client types provide this
    except Exception:
        pass
    headers.setdefault("Content-Type", "application/json")
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": VISION_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": data_uri}},
                {"type": "text", "text": "Screen this image."}]},
        ],
        "temperature": 0.0, "max_tokens": 400, "stream": False,
    }
    try:
        r = requests.post(base + "/v1/chat/completions", headers=headers,
                          data=json.dumps(body, ensure_ascii=False), timeout=180)
    except Exception:
        return None
    if r.status_code >= 400:
        return None
    try:
        text = r.json()["choices"][0]["message"]["content"]
        m = json.loads(text[text.find("{"):text.rfind("}") + 1])
        return m if isinstance(m, dict) else None
    except Exception:
        return None


def vision_screen(rows: list[dict], cookies_raw: str, lm, model: str | None = None,
                  max_items: int = MAX_ITEMS, progress=None) -> dict:
    """rows: analyzer verdict rows (need item-level media). Uses `media` lists.

    Returns {item_key: {"content","risk","indicators","note"}} for screened items.
    """
    model = model or pick_vision_model(lm)
    if not model or not cookies_raw:
        return {}
    import re as _re
    targets = [r for r in rows if r.get("is_violation") and (r.get("media") or [])]
    if not targets:
        return {}
    s = _session(cookies_raw)
    out: dict[str, dict] = {}
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    for r in targets[:max_items]:
        key = r.get("id") or r.get("item_key")
        merged = []
        for url in r["media"]:
            if isinstance(url, str) and _re.search(r"scontent|fbcdn|lookaside", url):
                merged.append(url)
        screened = None
        for url in merged[:MAX_IMAGES_PER_ITEM]:
            data_uri = _download_image(s, url)
            if not data_uri:
                continue
            screened = _ask_vision(lm, model, data_uri)
            if screened:
                out[str(key)] = screened
                break
        if progress:
            progress(f"vision {key}", len(out) / max(1, len(targets[:max_items])))
    if out:
        os.makedirs("output", exist_ok=True)
        with open(os.path.join("output", f"vision_{stamp}.json"), "w", encoding="utf-8") as f:
            json.dump({"generated": stamp, "model": model, "results": out},
                      f, ensure_ascii=False, indent=1)
        storage.audit("vision", "screen", f"{len(out)} image(s) screened with {model} (local)")
    return out
