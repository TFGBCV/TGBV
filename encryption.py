"""Encryption at rest for the case dataset (optional).

Design:
  - The key comes from `GBV_DB_KEY` (env) or, fallback, the OS keyring
    (keyring.get_password("fb_gbv_monitor", "db") — set once with
    `python -m encryption --set-key`).
  - If a SQLCipher python driver is installed (`pysqlcipher3` or `sqlcipher3`),
    storage opens the dataset through `open_db()` which applies `PRAGMA key`.
    Without a driver the DB stays plaintext and a loud warning is issued once
    (never silently — evidence tooling must not pretend to be encrypted).
  - Migrate an existing plaintext dataset:
      python tools/encrypt_db.py output/gbv.db --output output/gbv_enc.db
    (see tools/encrypt_db.py; it re-attaches+rebuilds under the new key)
  - Nothing is sent anywhere; the key stays in your environment / keyring.
"""
from __future__ import annotations

import os

_WARNED = False


def get_key() -> str:
    """Resolve the encryption key: env GBV_DB_KEY, else the OS keyring."""
    key = os.environ.get("GBV_DB_KEY", "").strip()
    if key:
        return key
    try:
        import keyring
        stored = keyring.get_password("fb_gbv_monitor", "db")
        return (stored or "").strip()
    except Exception:
        return ""


def engine():
    """Return the SQLCipher dbapi2 module if one is importable, else None."""
    for name in ("pysqlcipher3", "sqlcipher3"):
        try:
            mod = __import__(name)
            return mod
        except ImportError:
            continue
    return None


def available() -> bool:
    return engine() is not None


def _warn_once() -> None:
    global _WARNED
    if not _WARNED:
        _WARNED = True
        import warnings
        warnings.warn(
            "GBV_DB_KEY is set, but no SQLCipher python driver (pysqlcipher3/sqlcipher3) "
            "is installed — the case dataset will be written UNENCRYPTED. "
            "Install pysqlcipher3 (pip install pysqlcipher3) to enable encryption at rest.",
            stacklevel=1)


def open_db(path: str):
    """Open `path` honoring the encryption config; always returns a sqlite-like conn."""
    return _plain_or_cipher(path)


def _plain_or_cipher(path: str):
    key = get_key()
    mod = engine()
    if key and not mod:
        _warn_once()
    if not key or not mod:
        import sqlite3
        return sqlite3.connect(path, timeout=30)
    try:
        conn = mod.connect(path, timeout=30)
        conn.execute("PRAGMA key = ?", (key,))  # sqlcipher ≥4 supports bound keys
        # verify the key actually decrypts (fails on wrong password before any table read)
        conn.execute("SELECT count(*) FROM sqlite_master")
        conn.execute("PRAGMA journal_mode=WAL")
        return conn
    except Exception:
        try:
            conn.close()
        except Exception:
            pass
        raise RuntimeError(
            "Opened the case dataset with the configured GBV_DB_KEY but it does not "
            "decrypt (wrong key, or the file is not a SQLCipher database yet — "
            "migrate plaintext files with tools/encrypt_db.py first).")


if __name__ == "__main__":
    # python -m encryption --set-key   → store a generated key in the OS keyring
    import argparse
    import secrets
    ap = argparse.ArgumentParser(description="fb_gbv_monitor encryption helpers")
    ap.add_argument("--set-key", action="store_true", help="generate + store DB key in OS keyring")
    args = ap.parse_args()
    if args.set_key:
        try:
            import keyring
        except ImportError:
            raise SystemExit("keyring is not installed: pip install keyring")
        key = keyring.get_password("fb_gbv_monitor", "db") or secrets.token_urlsafe(32)
        keyring.set_password("fb_gbv_monitor", "db", key)
        print("DB key stored in the OS keyring (fb_gbv_monitor/db). "
              "Keep a safe offline copy of it — lose it and you lose the case data.")
    else:
        print(f"SQLCipher driver: {'yes' if available() else 'NO (pip install pysqlcipher3)'}")
        print(f"Key source: {'env/GBV_DB_KEY or keyring (configured)' if get_key() else 'none configured'}")
