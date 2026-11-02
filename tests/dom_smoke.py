#!/usr/bin/env python3
"""DOM-extractor smoke tests, run OFFLINE against recorded HTML fixtures.

Facebook changes its layout constantly — the classic failure mode is "0 posts
extracted" discovered by a user in the field. These tests replay saved page
snapshots through the real extractor (`scraper.JS_EXTRACT`) in a headless
browser and FAIL when the extractor stops finding what it used to find, so
selector drift is caught in CI without ever touching the live site.

Usage:
  python tests/dom_smoke.py                 # all fixtures in dom_fixtures/
  python tests/dom_smoke.py <file.html>     # one fixture

Fixtures are recorded with tools/record_fixture.py (from a live page, locally).
If no fixtures or playwright exist yet, the script explains itself and exits 0
(so CI can include it guarded by fixture presence).
"""
from __future__ import annotations

import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraper import JS_EXTRACT  # noqa: E402

FIXTURE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           "dom_fixtures")
BASELINE_FILE = os.path.join(FIXTURE_DIR, "baseline.json")


def run_fixture(page, path: str) -> dict:
    with open(path, encoding="utf-8") as f:
        html = f.read()
    page.set_content(html, wait_until="domcontentloaded")
    page.wait_for_timeout(1500)  # let the DOM settle like a real visit
    res = page.evaluate(JS_EXTRACT, {"diag": True, "marks": False})
    items = res["items"]
    posts = [it for it in items if (it.get("text") or "").strip()]
    return {"fixture": os.path.basename(path), "items": len(items),
            "posts_with_text": len(posts),
            "with_permalink": sum(1 for p in items if p.get("permalink")),
            "with_media": sum(1 for p in items if (p.get("media") or [])),
            "diag": {k: res["diag"].get(k) for k in
                     ("articles", "posinset", "messages", "containersKept", "byStrategy")}}


def main(argv: list[str]) -> int:
    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        print("playwright not installed — cannot run DOM smoke tests here "
              "(pip install playwright && playwright install chromium)")
        return 0

    targets = argv[1:] or sorted(glob.glob(os.path.join(FIXTURE_DIR, "*.html")))
    if not targets:
        print("no fixtures found in dom_fixtures/. Record one locally first:")
        print("  GBV_FB_COOKIES=... python tools/record_fixture.py <page-url> --name <name>")
        return 0

    baseline: dict = {}
    if os.path.exists(BASELINE_FILE):
        try:
            baseline = json.load(open(BASELINE_FILE, encoding="utf-8"))
        except Exception:
            baseline = {}

    failures = 0
    drift = 0
    results = {}
    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        page = browser.new_page()
        for path in targets:
            try:
                r = run_fixture(page, path)
            except Exception as e:
                print(f"❌ {os.path.basename(path)}: failed to evaluate extractor: "
                      f"{type(e).__name__}: {e}")
                failures += 1
                continue
            results[r["fixture"]] = r["posts_with_text"]
            status = "✅"
            if r["posts_with_text"] == 0:
                status = "❌ extractor found 0 posts"
                failures += 1
            elif r["fixture"] in baseline and baseline[r["fixture"]] != r["posts_with_text"]:
                status = (f"⚠️ drift: baseline {baseline[r['fixture']]} → "
                          f"{r['posts_with_text']} posts (selectors may have changed)")
                drift += 1
            print(f"{status} {r['fixture']}: {r['posts_with_text']} posts "
                  f"({r['with_permalink']} with permalink, {r['with_media']} with media)")
        browser.close()

    if fixtures_changed(results, baseline):
        with open(BASELINE_FILE, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=1)
        print(f"(baseline updated → {BASELINE_FILE})")
    print("dom_smoke:", "FAIL" if failures else "PASS",
          f"({len(targets)} fixture(s), {drift} drift warning(s))")
    return 1 if failures else 0


def fixtures_changed(results: dict, baseline: dict) -> bool:
    return any(baseline.get(k) != v for k, v in results.items())


if __name__ == "__main__":
    sys.exit(main(sys.argv))
