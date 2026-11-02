"""Client for the BUNDLED local engine — a drop-in replacement for LMStudioClient.

When the app is distributed as the portable exe, launcher.py starts llama.cpp's
`llama-server` (bundled next to the executable) instead of requiring LM Studio:

  - chat server :  pre-baked GGUF chat+vision model (Qwen2.5-VL-7B-Instruct
                   Q4_K_M + mmproj tower; serves text AND image screening);
                   OpenAI-compatible `/v1/chat/completions`
  - embed server:  nomic-embed-text-v1.5 GGUF on OpenAI-compatible `/v1/embeddings`

launcher.py sets the environment variables
    GBV_BUNDLED=1
    GBV_LLM_URL=http://127.0.0.1:<chat port>
    GBV_EMBED_URL=http://127.0.0.1:<embed port>   (may be absent)
    GBV_MMPROJ=<mmproj gguf filename>   (present when the bundle ships the
                                        vision tower → capabilities.vision=True)
and this module is a thin HTTP adapter on top of those so analyzer.py, osint.py,
agent.py and monitor.py keep working unchanged.
"""
from __future__ import annotations

import os
import re
import time
from typing import Callable, Optional

import requests

from lmstudio import LMStudioConnectionError, LMStudioError

SHARD_RE = re.compile(r"-\d{5}-of-\d{5}$")
QUANT_RE = re.compile(r"(?:^|[-_.])(i?q\d+[a-z0-9_]*)$", re.I)


def _pretty_name(stem: str) -> str:
    return stem.replace("-", " ").replace("_", " ").title()


def _mmeta(model_id: str) -> tuple[str, str, str]:
    """Best-effort (pretty name, quant, stem) from an engine model id/path."""
    stem = os.path.basename(str(model_id).replace("\\", "/"))
    if stem.lower().endswith(".gguf"):
        stem = stem[:-5]
    stem = SHARD_RE.sub("", stem)
    quant = ((QUANT_RE.search(stem) or [None, "?"])[1]).upper()
    return _pretty_name(stem), quant, stem


def is_bundled() -> bool:
    """True when the app runs from the portable bundle (launcher sets GBV_BUNDLED=1)."""
    return os.environ.get("GBV_BUNDLED", "") == "1"


def chat_url() -> str:
    url = os.environ.get("GBV_LLM_URL", "").strip()
    if not url:
        # Should not happen inside the bundle; keep a clear message anyway.
        raise LMStudioConnectionError(
            "The bundled AI engine is not configured. Close the app and start "
            "GBVMonitor.exe again (it starts the engine on launch)."
        )
    return url.rstrip("/")


def embed_url() -> str:
    return os.environ.get("GBV_EMBED_URL", "").strip().rstrip("/")


def _pretty_name(stem: str) -> str:
    return stem.replace("-", " ").replace("_", " ").title()


class LocalLLMClient:
    """Same interface as LMStudioClient, pointed at the bundled llama.cpp servers."""

    def __init__(self, timeout: int = 900):
        self.timeout = timeout

    # ------------------------------------------------------------------ http
    @property
    def base(self) -> str:  # informational, for error/debug messages
        return chat_url()

    def _headers(self) -> dict:
        return {"Content-Type": "application/json"}

    def _request(self, method: str, url: str, body: Optional[dict] = None,
                 timeout: Optional[int] = None) -> dict:
        try:
            r = requests.request(method, url, headers=self._headers(), json=body,
                                 timeout=timeout or self.timeout)
        except requests.ConnectionError as e:
            raise LMStudioConnectionError(
                "Lost contact with the bundled AI engine. Close this app and start "
                "GBVMonitor.exe again. (Details: the engine no longer answers on "
                f"{url} — see logs/engine-*.log next to the executable.)"
            ) from e
        except requests.Timeout as e:
            raise LMStudioError(f"Request timed out: {method} {url}") from e
        if r.status_code >= 400:
            try:
                detail = r.json()
            except ValueError:
                detail = r.text[:300]
            raise LMStudioError(f"HTTP {r.status_code} on {method} {url}: {detail}")
        try:
            return r.json()
        except ValueError as e:
            raise LMStudioError(f"Non-JSON response from {url}") from e

    # ------------------------------------------------------------------ models
    def list_models(self) -> list[dict]:
        data = self._request("GET", chat_url() + "/v1/models", timeout=30)
        # The bundled engine may serve the mmproj vision tower next to the chat
        # GGUF (launcher starts it with --mmproj) → this model does text AND
        # vision; advertise that so vision.pick_vision_model() finds it.
        caps = {"vision": True} if os.environ.get("GBV_MMPROJ") else {}
        out = []
        for m in (data.get("data") or []):
            display, quant, stem = _mmeta(str(m.get("id", "model")))
            gauge = os.environ.get("GBV_CTX", "8192")
            out.append({
                "type": "llm",
                "key": stem,
                "display_name": display,
                "loaded_instances": 1,       # llama-server loads the model at startup
                "max_context_length": int(gauge) if str(gauge).isdigit() else 8192,
                "quantization": {"name": quant},
                "params_string": "",
                "capabilities": caps,
                "info": dict(m),
            })
        return out

    def load_model(self, model: str, context_length: Optional[int] = None,
                   flash_attention: Optional[bool] = None) -> dict:
        """The engine loads the model at startup with the chosen context; no-op here."""
        return {"load_time_seconds": 0.0, "success": True,
                "note": "The bundled engine already loaded this model with "
                        f"context {os.environ.get('GBV_CTX', '8192')} at startup."}

    def download_model(self, model: str, quantization: Optional[str] = None) -> dict:
        raise LMStudioError(
            "This app ships with its own AI model included — nothing to download. "
            "For an external-model workflow, use LM Studio instead of this bundle."
        )

    def download_status(self, job_id: str) -> dict:
        raise LMStudioError("Downloads are not used by the bundled version.")

    def wait_for_download(self, job_id: str, progress: Optional[Callable[[dict], None]] = None,
                          poll: float = 2.0, max_wait: float = 6 * 3600) -> dict:
        raise LMStudioError("Downloads are not used by the bundled version.")

    # ------------------------------------------------------------------ chat
    def chat(self, model: str, user_input: str, system_prompt: str = "", temperature: float = 0.0,
             max_output_tokens: Optional[int] = None, reasoning: Optional[str] = None,
             response_format: Optional[dict] = None) -> str:
        """Stateless chat call via OpenAI-compatible /v1/chat/completions."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_input})
        body: dict = {"messages": messages, "temperature": temperature, "stream": False}
        if max_output_tokens:
            body["max_tokens"] = int(max_output_tokens)
        if response_format:
            body["response_format"] = response_format
        data = self._request("POST", chat_url() + "/v1/chat/completions", body)
        try:
            text = data["choices"][0]["message"]["content"] or ""
        except (KeyError, IndexError, TypeError):
            reason = str(data.get("error") or "")[:300]
            raise LMStudioError(
                f"Bundled engine returned no message. {reason}".strip()
            ) from None
        text = text.strip()
        if not text:
            raise LMStudioError(
                "Model returned no message (try reducing items per request or ask "
                "the bundle maintainer for a larger model)."
            )
        return text

    # ------------------------------------------------------------------ embeddings
    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        """OpenAI-compatible /v1/embeddings from the bundled nomic-embed server."""
        url = embed_url()
        if not url:
            raise LMStudioError(
                "The bundled embedding model is missing from this distribution — "
                "OSINT enrichment needs it. (models/*embed*.gguf was not found next to the executable.)"
            )
        payload = {"model": model, "input": texts}
        data = self._request("POST", url + "/v1/embeddings", payload, timeout=300)
        try:
            items = sorted(data["data"], key=lambda d: d.get("index", 0))
        except TypeError:
            items = data.get("data", [])
        return [list(d.get("embedding", [])) for d in items]
