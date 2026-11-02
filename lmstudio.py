"""Client for local LLM engines.

Two engines are supported behind the same interface:

  * LM Studio (classic setup): the user runs LM Studio's local server and this
    client uses its v1 REST API:
      GET  /api/v1/models
      POST /api/v1/models/load
      POST /api/v1/models/download
      GET  /api/v1/models/download/status/:job_id
      POST /api/v1/chat
    Docs: https://lmstudio.ai/docs/developer/rest

  * Bundled engine (portable exe distribution): `local_llm.LocalLLMClient`
    speaks to the llama.cpp `llama-server` shipped next to the executable.
    `create_client()` picks the right one based on the GBV_BUNDLED env var set
    by launcher.py.
"""
from __future__ import annotations

import os
import time
from typing import Callable, Optional

import requests

DEFAULT_URL = "http://localhost:1234"


def create_client(base_url: str = "", token: str = "") -> "LMStudioClient":
    """Return the client for the active engine.

    Bundled exe runs set GBV_BUNDLED=1 (the launcher starts a llama.cpp server
   ); everything else keeps the classic LM Studio client.
    """
    if os.environ.get("GBV_BUNDLED", "") == "1":
        from local_llm import LocalLLMClient
        return LocalLLMClient()  # type: ignore[return-value]
    return LMStudioClient(base_url or os.environ.get("LMSTUDIO_URL", DEFAULT_URL), token)


class LMStudioError(Exception):
    pass


class LMStudioConnectionError(LMStudioError):
    pass


class LMStudioClient:
    def __init__(self, base_url: str = DEFAULT_URL, token: str = "", timeout: int = 900):
        self.base = (base_url or DEFAULT_URL).strip().rstrip("/")
        self.token = (token or "").strip()
        self.timeout = timeout

    # ------------------------------------------------------------------ http
    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.token:
            h["Authorization"] = f"Bearer {self.token}"
        return h

    def _request(self, method: str, path: str, body: Optional[dict] = None, timeout: Optional[int] = None) -> dict:
        try:
            r = requests.request(method, self.base + path, headers=self._headers(), json=body,
                                 timeout=timeout or self.timeout)
        except requests.ConnectionError as e:
            raise LMStudioConnectionError(
                f"Cannot reach the model server at {self.base}. "
                "Open LM Studio and start the local server (Developer tab → Start Server, or "
                "`lms server start`), then check the URL."
            ) from e
        except requests.Timeout as e:
            raise LMStudioError(f"Request timed out: {method} {path}") from e
        if r.status_code >= 400:
            try:
                detail = r.json()
            except ValueError:
                detail = r.text[:300]
            raise LMStudioError(f"HTTP {r.status_code} on {method} {path}: {detail}")
        try:
            return r.json()
        except ValueError as e:
            raise LMStudioError(f"Non-JSON response from {path}") from e

    # ------------------------------------------------------------------ models
    def list_models(self) -> list[dict]:
        return self._request("GET", "/api/v1/models", timeout=30).get("models", [])

    def load_model(self, model: str, context_length: Optional[int] = None,
                   flash_attention: Optional[bool] = None) -> dict:
        body: dict = {"model": model, "echo_load_config": True}
        if context_length:
            body["context_length"] = int(context_length)
        if flash_attention is not None:
            body["flash_attention"] = flash_attention
        return self._request("POST", "/api/v1/models/load", body)

    def download_model(self, model: str, quantization: Optional[str] = None) -> dict:
        body: dict = {"model": model}
        if quantization:
            body["quantization"] = quantization
        return self._request("POST", "/api/v1/models/download", body, timeout=60)

    def download_status(self, job_id: str) -> dict:
        return self._request("GET", f"/api/v1/models/download/status/{job_id}", timeout=30)

    def wait_for_download(self, job_id: str, progress: Optional[Callable[[dict], None]] = None,
                          poll: float = 2.0, max_wait: float = 6 * 3600) -> dict:
        start = time.time()
        while True:
            st = self.download_status(job_id)
            if progress:
                progress(st)
            status = st.get("status")
            if status == "completed":
                return st
            if status == "failed":
                raise LMStudioError("Model download failed.")
            if time.time() - start > max_wait:
                raise LMStudioError("Gave up waiting for the download.")
            time.sleep(poll)

    # ------------------------------------------------------------------ chat
    def chat(self, model: str, user_input: str, system_prompt: str = "", temperature: float = 0.0,
             max_output_tokens: Optional[int] = None, reasoning: Optional[str] = None,
             response_format: Optional[dict] = None) -> str:
        """Stateless chat call (store=false, so LM Studio keeps no history). Returns the message text."""
        body: dict = {"model": model, "input": user_input, "temperature": temperature,
                      "store": False, "stream": False}
        if system_prompt:
            body["system_prompt"] = system_prompt
        if max_output_tokens:
            body["max_output_tokens"] = int(max_output_tokens)
        if reasoning:
            body["reasoning"] = reasoning
        if response_format:
            body["response_format"] = response_format
        data = self._request("POST", "/api/v1/chat", body)
        text = "\n".join(
            str(it.get("content", "")) for it in data.get("output", []) if it.get("type") == "message"
        ).strip()
        if not text:
            raise LMStudioError("Model returned no message (try turning reasoning off or raising max tokens).")
        return text


    # ------------------------------------------------------------------ embeddings
    def embed(self, texts: list[str], model: str) -> list[list[float]]:
        """Embed via LM Studio's OpenAI-compatible /v1/embeddings endpoint."""
        payload = {"model": model, "input": texts}
        data = self._request("POST", "/v1/embeddings", payload, timeout=300)
        try:
            items = sorted(data["data"], key=lambda d: d.get("index", 0))
        except TypeError:
            items = data.get("data", [])
        return [list(d.get("embedding", [])) for d in items]


# ---------------------------------------------------------------------- helpers
def llm_models(models: list[dict]) -> list[dict]:
    return [m for m in models if m.get("type") == "llm"]


def pick_reasoning(model_info: Optional[dict], pref: str = "auto") -> Optional[str]:
    """Return a value for the `reasoning` param, or None to omit it (the API errors if unsupported)."""
    opts = (((model_info or {}).get("capabilities") or {}).get("reasoning") or {}).get("allowed_options") or []
    if pref == "auto":
        return "off" if "off" in opts else None
    return pref if pref in opts else None
