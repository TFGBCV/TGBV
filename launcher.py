#!/usr/bin/env python3
"""Portable launcher / engine supervisor for the GBV Monitor bundle.

This is the entry point compiled into GBVMonitor.exe. With one double click it

  1. makes the folder next to the executable the working directory
     (so `output/`, the SQLite case dataset and `logs/` are portable),
  2. points Playwright at the bundled Chromium (`ms-playwright/`),
  3. starts the bundled llama.cpp engine:
       * the chat model server  (GGUF shipped in `models/`, plus the `mmproj`
         vision tower when bundled — one server serves text AND images)
       * the nomic-embed server (for OSINT embeddings) when present
       * GPU (Vulkan) binary first, CPU binary as fallback unless overridden,
  4. serves the Streamlit app on a free localhost port and opens the browser,

or, in CLI/scheduler mode, runs monitor.py passthrough:

    GBVMonitor.exe monitor run-due --enrich --verify
    GBVMonitor.exe monitor watch --interval 900
    GBVMonitor.exe monitor add-watch https://www.facebook.com/... --label "X"

Environment knobs (all optional):
    GBV_BACKEND     force the engine backend ("vulkan", "cpu", ... as found in runtime/)
    GBV_CTX         context length for the chat model (default 8192)
    GBV_ARGS        extra llama-server args, e.g. "--no-webui -t 8"
    GBV_ENGINE_CMD  dev/test only: full command template to launch instead of
                    the bundled binary, e.g. 'python3 mock_llama_server.py --port {port}'
"""
from __future__ import annotations

import atexit
import multiprocessing
import os
import re
import shlex
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

DEFAULT_CTX = 8192
CHAT_PORT_FROM = 7641
UI_PORT_FROM = 8501
SERVER_WAIT_CHAT = 900     # a 7B model on a slow disk / weak CPU may take a while
SERVER_WAIT_EMBED = 120
BACKEND_RANK = {"vulkan": 0, "cuda": 1, "sycl": 2, "rocm": 3, "openvino": 4, "cpu": 90}


def _is_frozen() -> bool:
    return getattr(sys, "frozen", False)


def app_dir() -> Path:
    """Folder that holds the executable (repo root in dev mode)."""
    return Path(sys.executable).resolve().parent if _is_frozen() else Path(__file__).resolve().parent


def _resource_root() -> Path:
    """App source location: PyInstaller's _MEIPASS when frozen, repo root otherwise."""
    if _is_frozen():
        return Path(getattr(sys, "_MEIPASS", app_dir()))
    return app_dir()


# --------------------------------------------------------------------------- utils
def log(msg: str) -> None:
    print(f"[GBV] {msg}", flush=True)


def fail_pause(message: str) -> None:
    print(f"\n[GBV] ERROR: {message}\n", flush=True)
    if _is_frozen():
        try:
            input("Press Enter to close…")
        except EOFError:
            pass


def free_port(start: int = CHAT_PORT_FROM, span: int = 100) -> int:
    for p in range(start, start + span):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", p))
                return p
            except OSError:
                continue
    raise RuntimeError(f"No free port in {start}–{start + span - 1}.")


def wait_healthy(url: str, proc: Optional[subprocess.Popen], timeout: int = SERVER_WAIT_CHAT) -> bool:
    import urllib.request
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if proc is not None and proc.poll() is not None:
            return False  # engine died while loading (OOM, missing DLL, bad GGUF ...)
        try:
            r = urllib.request.urlopen(url + "/health", timeout=5)
            if getattr(r, "status", 0) == 200:
                return True
        except Exception:
            pass
        time.sleep(1.0)
    return False


# --------------------------------------------------------------------------- assets
def find_engines() -> dict[str, Path]:
    """Discover bundled llama-server binaries, keyed by backend (top folder under runtime/)."""
    out: dict[str, Path] = {}
    base = app_dir() / "runtime"
    if base.is_dir():
        for exe in sorted(base.rglob("llama-server*")):
            if not (exe.is_file() and exe.suffix.lower() in ("", ".exe", ".com")):
                continue
            rel = exe.relative_to(base)
            backend = rel.parts[0] if len(rel.parts) > 1 else "bundled"
            out.setdefault(backend, exe)
    if not out:  # dev fallback: whatever is on PATH (Linux/macOS development)
        which = shutil.which("llama-server")
        if which:
            out["cpu"] = Path(which)
    engines_order = sorted(out, key=lambda k: (BACKEND_RANK.get(k, 10), k))
    return {k: out[k] for k in engines_order}


def find_models() -> tuple[Optional[Path], Optional[Path], Optional[Path]]:
    """Return (chat_gguf_first_part, embed_gguf, mmproj_gguf) next to the exe.

    `mmproj-*.gguf` is the multimodal vision tower that ships with the default
    Qwen2.5-VL bundle; it is never a chat candidate. Bundles built from a
    text-only model simply have no mmproj file and vision stays disabled.
    """
    models_dir = app_dir() / "models"
    if not models_dir.is_dir():
        return None, None, None
    ggufs = [m for m in models_dir.glob("*.gguf") if m.is_file()]
    embed = next((m for m in ggufs if re.search(r"embed|bge-|gte", m.name, re.I)), None)
    mmproj = next((m for m in ggufs if m is not embed
                   and m.name.lower().startswith("mmproj")), None)
    chat_cands = [m for m in ggufs if m is not embed and m is not mmproj]
    shards = [m for m in chat_cands if re.search(r"-(\d{5})-of-(\d{5})\.gguf$", m.name)]
    if shards:
        chat = min(shards, key=lambda m: m.name)  # llama.cpp reads part 00001 + siblings
    elif chat_cands:
        chat = sorted(chat_cands)[0]
    else:
        chat = None
    return chat, embed, mmproj


# --------------------------------------------------------------------------- engine
class Engine:
    """Spawns the bundled llama.cpp servers and owns their lifetime."""

    def __init__(self) -> None:
        self.procs: list[subprocess.Popen] = []
        self.llm_port = 0

    def stop(self) -> None:
        for p in self.procs:
            try:
                if p.poll() is None:
                    p.terminate()
            except Exception:
                pass
        for p in self.procs:
            try:
                p.wait(timeout=15)
            except Exception:
                pass
        self.procs.clear()

    def _spawn(self, cmd: list[str], log_path: Path) -> subprocess.Popen:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        logf = open(log_path, "ab", buffering=0)
        logf.write(f"\n=== {time.strftime('%Y-%m-%d %H:%M:%S')} — launcher spawn ===\n".encode())
        logf.write((" ".join(shlex.quote(c) for c in cmd) + "\n\n").encode())
        kwargs: dict = {"cwd": str(app_dir())}
        if os.name == "nt":
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW  # no extra console popups
            kwargs["startupinfo"] = si
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
        return subprocess.Popen(cmd, stdout=logf, stderr=subprocess.STDOUT, **kwargs)

    def _cmd(self, engine: Path, model: Path, port: int, extra: list[str]) -> list[str]:
        ctx = os.environ.get("GBV_CTX", str(DEFAULT_CTX))
        cmd = [str(engine), "--host", "127.0.0.1", "--port", str(port),
               "--model", str(model), "--ctx-size", str(ctx)]
        cmd += shlex.split(os.environ.get("GBV_ARGS", ""), posix=os.name != "nt")
        return cmd + extra

    def _start_on(self, engines: dict[str, Path], model: Path, port: int, wait: int,
                  extra: list[str], log_name: str) -> tuple[str, subprocess.Popen] | None:
        logs = app_dir() / "logs" / f"engine-{log_name}.log"
        for backend, exe in engines.items():
            try:
                proc = self._spawn(self._cmd(exe, model, port, extra), logs)
            except OSError as e:
                log(f"  backend '{backend}' failed to spawn: {e}")
                continue
            if wait_healthy(f"http://127.0.0.1:{port}", proc, wait):
                log(f"  engine backend in use: {backend}")
                return backend, proc
            log(f"  backend '{backend}' did not become healthy "
                f"(see logs/engine-{log_name}.log)")
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                pass
        return None

    def start(self) -> None:
        chat_model: Optional[Path] = None
        embed_model: Optional[Path] = None
        # Dev/test hook: replace the engine binary with an external command template.
        template = os.environ.get("GBV_ENGINE_CMD", "").strip()
        if template:
            for port_key, pstart in (("chat", CHAT_PORT_FROM), ("embed", CHAT_PORT_FROM + 10)):
                if "{port}" not in template:
                    break
                port = free_port(pstart)
                cmd = [c.replace("{port}", str(port)).replace("{model}", str(chat_model or "mock.gguf"))
                       for c in shlex.split(template, posix=os.name != "nt")]
                proc = self._spawn(cmd, app_dir() / "logs" / f"engine-{port_key}.log")
                self.procs.append(proc)
                key = "GBV_LLM_URL" if port_key == "chat" else "GBV_EMBED_URL"
                if wait_healthy(f"http://127.0.0.1:{port}", proc, 30):
                    os.environ[key] = f"http://127.0.0.1:{port}"
                    log(f"{port_key} server (dev override) ready on 127.0.0.1:{port}")
            return

        engines = find_engines()
        chat_model, embed_model, mmproj_model = find_models()
        if not engines:
            raise RuntimeError(
                "The bundled AI engine is missing: no runtime/*/llama-server was found "
                "next to this executable. The distribution folder is incomplete — "
                "re-copy it from the original zip."
            )
        if chat_model is None:
            raise RuntimeError(
                "No model found next to this executable (models/*.gguf). The distribution "
                "folder is incomplete, or the model files were moved/renamed."
            )
        log(f"Chat model: {chat_model.name}"
            + (" (+ other parts)" if re.search(r"-0000\d-of-", chat_model.name) else ""))
        if embed_model:
            log(f"Embedding model: {embed_model.name}")
        if mmproj_model:
            log(f"Vision tower: {mmproj_model.name} (image screening enabled)")

        # 1) chat server — try the fastest backend first, CPU is the safety net.
        #    When the bundle ships the mmproj vision tower, load it alongside the
        #    chat GGUF so ONE server handles text and image requests.
        self.llm_port = free_port(CHAT_PORT_FROM)
        extra = ["--jinja"] + (["--mmproj", str(mmproj_model)] if mmproj_model else [])
        started = self._start_on(engines, chat_model, self.llm_port, SERVER_WAIT_CHAT, extra, "chat")
        if not started and mmproj_model:
            # Very old engine builds without multimodal support: fall back to
            # text-only classification rather than refusing to start.
            log("  ! engine did not come up with --mmproj — retrying text-only "
                "(image screening disabled; see logs/engine-chat.log)")
            started = self._start_on(engines, chat_model, self.llm_port,
                                     SERVER_WAIT_CHAT, ["--jinja"], "chat")
        if not started:
            raise RuntimeError(
                "The bundled chat model could not be loaded. See logs/engine-chat.log next "
                "to the executable. Most common cause: not enough free RAM — the shipped "
                "model needs roughly 8 GB; free memory bandwidth by closing other apps."
            )
        backend, proc = started
        if mmproj_model and "--mmproj" in extra:
            # Advertise the multimodal capability to the app (local_llm.list_models).
            os.environ["GBV_MMPROJ"] = mmproj_model.name
        elif not mmproj_model:
            os.environ.pop("GBV_MMPROJ", None)
        self.procs.append(proc)
        os.environ["GBV_LLM_URL"] = f"http://127.0.0.1:{self.llm_port}"
        os.environ["GBV_BACKEND_USED"] = backend
        os.environ["GBV_MODEL_FILE"] = chat_model.name
        log(f"Chat engine ready on 127.0.0.1:{self.llm_port} (ctx {os.environ.get('GBV_CTX', DEFAULT_CTX)})")

        # 2) embedding server (small model; CPU is always fast enough for it).
        if embed_model:
            port = free_port(CHAT_PORT_FROM + 10)
            started_e = self._start_on(engines, embed_model, port, SERVER_WAIT_EMBED,
                                       ["--embeddings", "-b", "2048", "-ub", "2048"], "embed")
            if started_e:
                _, embed_proc = started_e
                self.procs.append(embed_proc)
                os.environ["GBV_EMBED_URL"] = f"http://127.0.0.1:{port}"
                log(f"Embedding engine ready on 127.0.0.1:{port}")
            else:
                log("  ! Embedding server did not start — OSINT theme clustering will be "
                    "unavailable (classification still works). See logs/engine-embed.log.")


# --------------------------------------------------------------------------- env & modes
def setup_environment() -> None:
    app = app_dir()
    try:
        os.chdir(app)
    except OSError:
        pass
    os.environ.pop("GBV_IN_DOCKER", None)
    os.environ["GBV_BUNDLED"] = "1"
    os.environ.setdefault("STREAMLIT_BROWSER_GATHER_USAGE_STATS", "false")
    pw = app / "ms-playwright"
    if pw.is_dir():
        os.environ["PLAYWRIGHT_BROWSERS_PATH"] = str(pw)
    (app / "output" / "shots").mkdir(parents=True, exist_ok=True)
    (app / "logs").mkdir(parents=True, exist_ok=True)


def app_script() -> Path:
    root = _resource_root()
    cand = root / "app.py"
    return cand if cand.is_file() else app_dir() / "app.py"


def run_ui(args) -> int:
    url = f"http://127.0.0.1:{args.port}"
    log(f"Starting the monitor interface on {url} …")

    from streamlit.web import cli as stcli
    sys.argv = [
        "streamlit", "run", str(app_script()),
        "--server.address", "127.0.0.1",
        "--server.port", str(args.port),
        "--server.headless", "true",
        "--server.fileWatcherType", "none",
        "--browser.gatherUsageStats", "false",
        "--global.developmentMode", "false",
    ]

    if not args.no_browser:
        import threading
        import urllib.request
        import webbrowser

        def browse_when_ready() -> None:
            try:
                deadline = time.monotonic() + 180
                while time.monotonic() < deadline:
                    try:
                        r = urllib.request.urlopen(url + "/_stcore/health", timeout=2)
                        if getattr(r, "status", 0) == 200:
                            webbrowser.open_new_tab(url)
                            return
                    except Exception:
                        time.sleep(1.0)
                log("Could not open a browser automatically — see the URL above.")
            except Exception:
                pass

        threading.Thread(target=browse_when_ready, daemon=True).start()

    stcli.main()
    return 0


def run_monitor(rest: list[str]) -> int:
    """CLI passthrough: same subcommands as `python monitor.py …`."""
    import monitor
    return monitor.main(rest)


def _ui_main(argv: list[str]) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="GBVMonitor", add_help=True,
        description="Portable Facebook GBV monitor — bundles its own local AI model "
                    "(no LM Studio, no Python, no cloud).")
    ap.add_argument("--port", type=int, default=free_port(UI_PORT_FROM),
                    help="port for the web interface (default: first free above 8501)")
    ap.add_argument("--no-browser", action="store_true", help="don't auto-open the browser")
    ap.add_argument("--ctx", type=int, default=int(os.environ.get("GBV_CTX", DEFAULT_CTX)),
                    help="context length for the bundled model, applies at startup")
    ap.add_argument("--skip-engine", action="store_true",
                    help="(dev/test) do not start the bundled model engine")
    ap.add_argument("--backend", default="", help="force engine backend (vulkan/cpu)")
    args = ap.parse_args(argv)
    if args.backend:
        os.environ["GBV_BACKEND"] = args.backend
    os.environ["GBV_CTX"] = str(int(args.ctx))

    setup_environment()
    engine = Engine()
    atexit.register(engine.stop)
    note = " with context " + os.environ["GBV_CTX"]
    skip = args.skip_engine or os.environ.get("GBV_SKIP_ENGINE", "") == "1"
    if not skip:
        log("Starting the bundled local AI engine (15–90 s on first launch)…")
        try:
            engine.start()
        except RuntimeError as e:
            fail_pause(str(e))
            return 2
        log(f"AI engine ready{note}.")
    else:
        log("Engine startup skipped (--skip-engine).")
    try:
        return run_ui(args)
    except KeyboardInterrupt:
        log("Bye.")
        return 0


def _monitor_entry(rest: list[str]) -> int:
    setup_environment()
    engine = Engine()
    atexit.register(engine.stop)
    needs_engine = bool(rest) and rest[0] not in ("add-watch", "-h", "--help")
    if needs_engine:
        log("Starting the bundled local AI engine…")
        try:
            engine.start()
        except RuntimeError as e:
            fail_pause(str(e))
            return 2
    try:
        return run_monitor(rest)
    except KeyboardInterrupt:
        return 0


def main() -> int:
    multiprocessing.freeze_support()  # required first call on frozen Windows builds
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    argv = sys.argv[1:]
    if argv and argv[0] == "monitor":
        return _monitor_entry(argv[1:])
    if argv and argv[0] == "ui":
        argv = argv[1:]
    return _ui_main(argv)


if __name__ == "__main__":
    sys.exit(main())
