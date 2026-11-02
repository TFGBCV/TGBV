#!/usr/bin/env python3
"""Record a DOM fixture for the offline smoke tests.

Saves the rendered HTML (after "See more"/banner dismissal and scrolling) of a
Facebook page into dom_fixtures/<name>.html so `python tests/dom_smoke.py` can
verify the DOM extractor (`scraper.JS_EXTRACT`) still finds posts — WITHOUT ever
touching Facebook in CI.

Usage (locally, with a logged-in dedicated account):
  GBV_FB_COOKIES="$(cat cookies.json)" python tools/record_fixture.py \
      https://www.facebook.com/somepage --name somepage

Fixtures contain the HTML of public posts of that page — treat them like any
other case data: sha-stored locally, never committed anywhere public.
"""
from __future__ import annotations

import argparse
import os
import random
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scraper import DISMISS, JS_CLICK_LABELS, SEE_MORE, parse_cookies  # noqa: E402

FIXTURE_DIR = "dom_fixtures"


def main() -> int:
    ap = argparse.ArgumentParser(description="Record a DOM fixture for offline smoke tests")
    ap.add_argument("url")
    ap.add_argument("--name", required=True, help="fixture name (e.g. 'somepage')")
    ap.add_argument("--scrolls", type=int, default=4)
    ap.add_argument("--cookies", default="")
    args = ap.parse_args()

    cookies_raw = args.cookies or os.environ.get("GBV_FB_COOKIES", "")
    if not cookies_raw or not os.path.isfile(cookies_raw):
        cookies_raw = os.environ.get("GBV_FB_COOKIES", "")
    if not cookies_raw:
        print("Set GBV_FB_COOKIES (or --cookies file) — the fixture needs a logged-in render.")
        return 2

    from playwright.sync_api import sync_playwright

    cookies = parse_cookies(cookies_raw)
    os.makedirs(FIXTURE_DIR, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    out = os.path.join(FIXTURE_DIR, f"{args.name}_{stamp}.html")

    with sync_playwright() as pw:
        browser = pw.chromium.launch(headless=True)
        ctx = browser.new_context(viewport={"width": 1280, "height": 900}, locale="en-US")
        ctx.add_cookies(cookies)
        page = ctx.new_page()
        page.goto(args.url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(3000)
        try:
            page.evaluate(JS_CLICK_LABELS, DISMISS)
            page.wait_for_selector('[role="article"], [aria-posinset]', timeout=15000)
        except Exception:
            pass
        for _ in range(max(1, args.scrolls)):
            page.evaluate(JS_CLICK_LABELS, SEE_MORE)
            page.wait_for_timeout(400)
            page.mouse.wheel(0, 2600)
            page.evaluate("window.scrollBy(0, 1800)")
            time.sleep(random.uniform(2.0, 3.5))
        html = page.evaluate("document.documentElement.outerHTML")
        browser.close()

    with open(out, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"fixture saved: {out} ({len(html) // 1024} KB)")
    print(f"run: python tests/dom_smoke.py   (extractor is tested against it offline)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
