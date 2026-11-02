#!/usr/bin/env python3
"""Fetch build-time assets for the portable GBV Monitor bundle.

Downloads (into --DEST, default: the repo root):
    runtime/<backend>/   llama.cpp llama-server Windows binaries
    models/*.gguf         chat+vision model (default Qwen2.5-VL-7B-Instruct
                          Q4_K_M — handles TEXT and IMAGES — plus its mmproj
                          vision tower) + nomic-embed-text-v1.5 Q8_0

Usage:
    python packaging/fetch_assets.py --dest . --backends cpu --chat-prefix Qwen2.5-VL-7B-Instruct-Q4_K_M
    python packaging/fetch_assets.py --dest . --backends cpu,vulkan
    python packaging/fetch_assets.py --dest . --no-vision   (skip mmproj → ~1.3 GB smaller)

Notes:
  * llama.cpp binaries are resolved from the latest released tag; the asset
    name pattern must match 'llama-<tag>-bin-win-<backend>-x64.zip'.
  * The mmproj vision tower is fetched automatically from the chat repo when
    it exists (prefer F16/Q8_0 quants). Pass --no-vision to bundle a
    text-only build (image screening disabled).
  * Hugging Face downloads honour HF_TOKEN (for private mirrors) and can be
    pointed to a mirror with HF_ENDPOINT.
  * Safe to re-run: existing files are skipped.
"""
from __future__ import annotations

import argparse
import fnmatch
import os
import re
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

import requests
from huggingface_hub import HfApi, hf_hub_download

GH_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases"
ASSET_PATTERNS = {
    "cpu":        "llama-*-bin-win-cpu-x64.zip",
    "vulkan":     "llama-*-bin-win-vulkan-x64.zip",
    "cuda-12.4":  "llama-*-bin-win-cuda-12.4-x64.zip",
    "cuda-13.4":  "llama-*-bin-win-cuda-13.4-x64.zip",
}

# Default = one model that does BOTH text and vision: Qwen2.5-VL-7B-Instruct
# from the llama.cpp team's repo (ggml-org) — guaranteed llama.cpp-compatible
# conversion of the official Qwen/Qwen2.5-VL-7B-Instruct (same family as the
# previous text-only default, so classification quality is the same class).
DEFAULT_CHAT_REPO = "ggml-org/Qwen2.5-VL-7B-Instruct-GGUF"
DEFAULT_CHAT_PREFIX = "Qwen2.5-VL-7B-Instruct-Q4_K_M"
DEFAULT_EMBED_REPO = "nomic-ai/nomic-embed-text-v1.5-GGUF"
DEFAULT_EMBED_FILE = "nomic-embed-text-v1.5.Q8_0.gguf"


def gh_session() -> requests.Session:
    s = requests.Session()
    s.headers["User-Agent"] = "fb-gbv-monitor-packaging"
    if os.environ.get("GH_TOKEN"):
        s.headers["Authorization"] = f"Bearer {os.environ['GH_TOKEN']}"
    return s


def releases_pages(session: requests.Session, pages: int = 4):
    for p in range(1, pages + 1):
        r = session.get(GH_API, params={"per_page": 50, "page": p}, timeout=60)
        r.raise_for_status()
        data = r.json()
        if not data:
            return
        for rel in data:
            if rel.get("draft"):
                continue
            yield rel


def resolve_asset(session: requests.Session, backend: str) -> tuple[str, str]:
    """Return (release_tag, download_url) of the matching llama.cpp windows asset."""
    pat = ASSET_PATTERNS.get(backend)
    if not pat:
        raise SystemExit(f"Unknown backend '{backend}'. Choose from: {sorted(ASSET_PATTERNS)}")
    rx = re.compile(fnmatch.translate(pat))
    for rel in releases_pages(session):
        for asset in rel.get("assets", []):
            name = asset.get("name", "")
            if rx.match(name):
                tag = rel.get("tag_name", "")
                print(f"[assets] {backend}: {tag} -> {name}")
                return tag, asset["browser_download_url"]
    raise SystemExit(
        f"No llama.cpp release asset matched '{pat}'. Check {GH_API} "
        "(the repo now may use a different packaging; update ASSET_PATTERNS)."
    )


def fetch_engine(dest: Path, backends: list[str]) -> None:
    session = gh_session()
    for backend in backends:
        target = dest / "runtime" / backend
        if any(target.glob("**/llama-server*")):
            print(f"[assets] {backend} engine already present -> skip")
            continue
        _, url = resolve_asset(session, backend)
        tmpdir = Path(tempfile.mkdtemp(prefix="llamacpp_"))
        try:
            print(f"[assets] downloading {url} ...")
            zip_path = tmpdir / "engine.zip"
            with session.get(url, stream=True, timeout=600) as r:
                r.raise_for_status()
                with open(zip_path, "wb") as f:
                    shutil.copyfileobj(r.raw, f)
            target.mkdir(parents=True, exist_ok=True)
            print(f"[assets] extracting -> {target}")
            with zipfile.ZipFile(zip_path) as z:
                z.extractall(target)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)
            if not any(target.glob("**/llama-server*")):  # extractall failure guard
                shutil.rmtree(target, ignore_errors=True)


def fetch_models(dest: Path, chat_repo: str, chat_prefix: str,
                 embed_repo: str, embed_file: str, with_embed: bool = True,
                 with_vision: bool = True) -> None:
    api = HfApi()
    models_dir = dest / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # ---- chat model (may be sharded: every *.gguf matching the prefix gets pulled)
    files = [f for f in api.list_repo_files(chat_repo) if f.endswith(".gguf")]
    wanted = sorted(f for f in files if f.startswith(chat_prefix))
    if not wanted:
        raise SystemExit(
            f"No GGUF file in {chat_repo} matches prefix '{chat_prefix}'.\n"
            f"Available: {sorted(files)}\n"
            "Pick a --chat-prefix that exists (e.g. check the repo's file list)."
        )
    print(f"[assets] chat model {chat_repo}: {', '.join(wanted)}")
    for name in wanted:
        download_one(models_dir, chat_repo, name)

    # ---- mmproj vision tower (when the repo ships one → bundled vision)
    if with_vision:
        mm_files = [f for f in files if f.lower().startswith("mmproj")
                    and f.endswith(".gguf")]
        if mm_files:
            target = next((f for f in mm_files if "f16" in f.lower()), None) \
                or next((f for f in mm_files if "q8_0" in f.lower()), None) \
                or sorted(mm_files)[0]
            download_one(models_dir, chat_repo, target)
        else:
            print("[assets] NOTE: no mmproj file in this chat repo — bundling a "
                  "TEXT-ONLY model (image screening disabled).")

    # ---- embedding model for OSINT enrichment (small)
    if with_embed:
        embed_files = [f for f in api.list_repo_files(embed_repo) if f.endswith(".gguf")]
        target = embed_file if embed_file in embed_files else next(
            (f for f in embed_files if fnmatch.fnmatch(f, embed_file)), None)
        if not target:
            raise SystemExit(f"No GGUF in {embed_repo} matches '{embed_file}'. "
                             f"Available: {sorted(embed_files)}")
        download_one(models_dir, embed_repo, target)


def download_one(models_dir: Path, repo: str, filename: str) -> None:
    if (models_dir / filename).is_file():
        print(f"[assets] {filename} already present -> skip")
        return
    print(f"[assets] downloading  {repo}/{filename} ...")
    hf_hub_download(repo_id=repo, repo_type="model", filename=filename, local_dir=str(models_dir))
    if not (models_dir / filename).is_file():
        raise SystemExit(f"Download finished but {filename} is missing in {models_dir}")


def main() -> int:
    try:  # Windows runners default to cp1252 — keep all console I/O decodable
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dest", default=".", help="where to create runtime/ and models/")
    ap.add_argument("--backends", default="cpu",
                    help="comma list: cpu[,vulkan][,cuda-12.4][,cuda-13.4]")
    ap.add_argument("--chat-repo", default=DEFAULT_CHAT_REPO)
    ap.add_argument("--chat-prefix", default=DEFAULT_CHAT_PREFIX,
                    help="prefix of the chat GGUF filenames to bundle, e.g. "
                         "Qwen2.5-VL-7B-Instruct-Q4_K_M (default) or "
                         "Qwen2.5-7B-Instruct-Q4_K_M from Qwen/Qwen2.5-7B-Instruct-GGUF")
    ap.add_argument("--embed-repo", default=DEFAULT_EMBED_REPO)
    ap.add_argument("--embed-file", default=DEFAULT_EMBED_FILE)
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--no-embed", action="store_true",
                    help="omit the OSINT embedding model (smaller zip, depleted enrichment)")
    ap.add_argument("--no-vision", action="store_true",
                    help="skip the mmproj vision tower (smaller build, no image screening)")
    args = ap.parse_args()

    dest = Path(args.dest).resolve()
    dest.mkdir(parents=True, exist_ok=True)

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    if "win" not in sys.platform.lower():
        print("[assets] NOTE: llama-server Windows binaries are downloaded for the "
              "portable build -- they only *run* on Windows;")
    fetch_engine(dest, backends)

    if not args.skip_models:
        fetch_models(dest, args.chat_repo, args.chat_prefix,
                     args.embed_repo, args.embed_file,
                     with_embed=not args.no_embed, with_vision=not args.no_vision)
    # final listing
    for base in ("runtime", "models"):
        d = dest / base
        if d.is_dir():
            for p in sorted(d.rglob("*")):
                if p.is_file():
                    size = p.stat().st_size
                    hsize = (f"{size / 1e9:.2f} GB" if size > 5e8
                             else f"{size / 1e6:.1f} MB")
                    print(f"[assets] {p.relative_to(dest)}  ({hsize})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
