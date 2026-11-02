"""Cookie-based Facebook page scraper (Playwright).

Collects posts (and optionally comments) from a Facebook page you can view with
the supplied session cookies.

Facebook's DOM is obfuscated and changes often, so extraction does NOT rely on a
single selector. `JS_EXTRACT` finds post containers four independent ways and
merges them:
  article   - top-level div[role="article"]
  posinset  - elements with aria-posinset (feed units)
  message   - elements marked as the post message (data-ad-*-message), climbing
              to the largest ancestor that holds exactly one message
  permalink - post permalink anchors, climbing to the largest ancestor that
              holds exactly one distinct permalink
If nothing is found, a diagnostics bundle (counts, DOM outline without text,
screenshot) is returned so the selectors can be adapted.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import re
import time
from datetime import datetime, timezone
from typing import Callable, Optional
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode

ProgressFn = Optional[Callable[[str, float], None]]

SEE_MORE = ["See more", "Voir plus", "Afficher plus", "عرض المزيد", "عرض المزيد..."]
MORE_COMMENTS = [
    "View more comments", "View more replies", "View previous comments",
    "Afficher plus de commentaires", "Voir plus de commentaires", "Voir les commentaires précédents",
    "عرض المزيد من التعليقات", "عرض تعليقات أخرى", "عرض التعليقات السابقة",
]
DISMISS = [
    "Decline optional cookies", "Refuser les cookies optionnels", "رفض ملفات تعريف الارتباط الاختيارية",
    "Not now", "Pas maintenant", "ليس الآن",
]
NOISE_LINES = {
    "like", "reply", "share", "edited", "author", "top fan", "follow",
    "j'aime", "répondre", "partager", "modifié", "auteur",
    "أعجبني", "رد", "مشاركة", "تم التعديل", "صاحب المحتوى",
}
_UI_LINE = re.compile(
    r"^(like|comment|comments|share|send|reply|follow|see more|see less|write a comment…?|most relevant|all comments|"
    r"j'aime|commenter|partager|envoyer|répondre|suivre|voir plus|écrivez un commentaire…?|"
    r"أعجبني|تعليق|مشاركة|إرسال|رد|متابعة|عرض المزيد|اكتب تعليقًا…?|·|public|shared with public)$", re.I)
_COUNT_LINE = re.compile(
    r"^(all reactions:?|toutes les réactions\s?:?|جميع التفاعلات:?|just now|à l'instant)$|^\d+([.,]\d+)?\s?[KkMm]?$|"
    r"^\d+\s?(s|m|h|d|w|y|min|mins|hrs?|j|sem|ans?)$|"
    r"^\d+([.,]\d+)?\s?[KkMm]?\s*(comments?|shares?|commentaires?|partages?|تعليقات?|مشاركات?)$", re.I)


class ScrapeError(Exception):
    pass


# --------------------------------------------------------------------------- cookies
_SAMESITE = {"no_restriction": "None", "none": "None", "lax": "Lax", "strict": "Strict"}


def _norm_cookie(c: dict) -> dict:
    out = {
        "name": str(c["name"]),
        "value": str(c["value"]),
        "domain": c.get("domain") or ".facebook.com",
        "path": c.get("path") or "/",
        "secure": bool(c.get("secure", True)),
        "httpOnly": bool(c.get("httpOnly", False)),
    }
    ss = _SAMESITE.get(str(c.get("sameSite", "")).lower())
    if ss:
        out["sameSite"] = ss
    exp = c.get("expirationDate", c.get("expires"))
    try:
        if exp and float(exp) > 0:
            out["expires"] = float(exp)
    except (TypeError, ValueError):
        pass
    return out


def parse_cookies(raw: str) -> list[dict]:
    """Accepts Cookie-Editor JSON, Netscape cookies.txt, or a 'a=b; c=d' header string."""
    raw = (raw or "").strip()
    if not raw:
        raise ScrapeError("No cookies provided.")
    cookies: list[dict] = []
    if raw[0] in "[{":
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as e:
            raise ScrapeError(f"Cookies look like JSON but could not be parsed: {e}")
        if isinstance(data, dict):
            data = data.get("cookies", [data])
        cookies = [_norm_cookie(c) for c in data if "name" in c and "value" in c]
    elif "\t" in raw:
        for line in raw.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split("\t")
            if len(parts) >= 7:
                cookies.append(_norm_cookie({
                    "domain": parts[0], "path": parts[2],
                    "secure": parts[3].upper() == "TRUE",
                    "expires": parts[4], "name": parts[5], "value": parts[6],
                }))
    else:
        for chunk in raw.split(";"):
            if "=" in chunk:
                k, v = chunk.split("=", 1)
                cookies.append(_norm_cookie({"name": k.strip(), "value": v.strip()}))
    names = {c["name"] for c in cookies}
    if not {"c_user", "xs"} <= names:
        raise ScrapeError("Cookies must include at least 'c_user' and 'xs' (Facebook session cookies).")
    return cookies


# --------------------------------------------------------------------------- helpers
def normalize_url(url: str) -> str:
    url = url.strip()
    if not url.startswith("http"):
        url = "https://" + url
    p = urlparse(url)
    if "facebook.com" not in p.netloc and "fb.com" not in p.netloc:
        raise ScrapeError("The link must be a facebook.com page URL.")
    return urlunparse(("https", "www.facebook.com", p.path, "", p.query, ""))


def canon_permalink(href: str) -> str:
    if not href:
        return ""
    p = urlparse(href)
    keep = [(k, v) for k, v in parse_qsl(p.query) if k in {"story_fbid", "id", "fbid", "v"}]
    return urlunparse(("https", "www.facebook.com", p.path, "", urlencode(keep), ""))


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _clean_post_text(text: str, author: str = "") -> str:
    """Strip UI chrome (Like/Share/counts) from fallback-extracted post text."""
    lines = [l.strip() for l in text.split("\n")]
    keep = [l for l in lines if l and not _UI_LINE.match(l) and not _COUNT_LINE.match(l)]
    if keep and author and keep[0] == author:
        keep = keep[1:]
    return "\n".join(keep).strip()


def _clean_comment(raw_text: str) -> tuple[str, str]:
    lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
    if not lines:
        return "", ""
    author, body = lines[0], []
    for l in lines[1:]:
        low = l.lower()
        if low in NOISE_LINES:
            continue
        if re.fullmatch(r"\d+\s*(s|m|h|d|w|y|min|j|sem|ans?|[a-zA-Zéû]{1,4})", low):
            continue
        body.append(l)
    return author, "\n".join(body).strip()


JS_EXTRACT = r"""
(opts) => {
  const wantDiag = !!opts.diag, wantMarks = !!opts.marks;
  const STRONG = /\/(posts|videos|reel|reels|permalink)\/[^\/?#]+|story_fbid=|pfbid|\/watch\/?\?v=|\/permalink\.php/;
  const WEAK = /\/photos?\/[^\/?#]+|[?&]fbid=|\/photo\/?\?/;
  const STOP = '[role="main"], [role="feed"], [data-pagelet="ProfileTimeline"], [role="complementary"], [role="banner"]';
  const MSG = '[data-ad-rendering-role="story_message"], [data-ad-comet-preview="message"], [data-ad-preview="message"]';
  const COMMENT = '[role="article"], [aria-label^="Comment by"], [aria-label^="Commentaire de"], [aria-label^="تعليق من"], [aria-label^="Reply by"], [aria-label^="Réponse de"]';

  const canon = (h) => {
    try {
      const u = new URL(h, location.href); const p = new URLSearchParams();
      ['story_fbid', 'id', 'fbid', 'v'].forEach(k => { if (u.searchParams.has(k)) p.set(k, u.searchParams.get(k)); });
      const q = p.toString();
      return 'https://www.facebook.com' + u.pathname + (q ? '?' + q : '');
    } catch (e) { return h; }
  };
  const anchors = Array.from(document.querySelectorAll('a[href]'));
  const strongA = anchors.filter(a => STRONG.test(a.href));
  const weakA = anchors.filter(a => WEAK.test(a.href));
  const permaA = strongA.length ? strongA : weakA;
  const permaRe = strongA.length ? STRONG : WEAK;
  const keysIn = (el) => {
    const s = new Set();
    el.querySelectorAll('a[href]').forEach(a => { if (permaRe.test(a.href)) s.add(canon(a.href)); });
    return s;
  };
  const stopAt = (el) => !el.parentElement || el.parentElement === document.body || el.parentElement.matches(STOP);

  const cont = [];
  const add = (el, how) => { if (el && !cont.some(c => c.el === el)) cont.push({el, how}); };

  // A: top-level articles
  document.querySelectorAll('[role="article"]').forEach(a => {
    if (!(a.parentElement && a.parentElement.closest('[role="article"]'))) add(a, 'article');
  });
  // C: feed units
  document.querySelectorAll('[aria-posinset]').forEach(a => add(a, 'posinset'));
  // B: message containers, climb to the largest ancestor holding one message
  document.querySelectorAll(MSG).forEach(m => {
    const art = m.closest('[role="article"]');
    if (art && art.parentElement && art.parentElement.closest('[role="article"]')) return; // inside a comment
    let el = m, depth = 0;
    while (!stopAt(el) && depth < 30) {
      const p = el.parentElement;
      if (p.querySelectorAll(MSG).length > 1) break;
      el = p; depth++;
    }
    add(el, 'message');
  });
  // E: permalink anchors, climb to the largest ancestor holding one distinct permalink
  const seen = new Set();
  permaA.forEach(a => {
    const k = canon(a.href); if (seen.has(k)) return; seen.add(k);
    let el = a, depth = 0;
    while (!stopAt(el) && depth < 30) {
      const p = el.parentElement;
      if (keysIn(p).size > 1) break;
      el = p; depth++;
    }
    if ((el.innerText || '').length > 30) add(el, 'permalink');
  });

  // Drop wrappers (contain 2+ disjoint other containers), then keep outermost per post.
  const els = cont.map(c => c.el);
  const valid = cont.filter(c => {
    const inner = els.filter(o => o !== c.el && c.el.contains(o));
    const disjoint = inner.filter(i => !inner.some(j => j !== i && j.contains(i)));
    return disjoint.length < 2;
  });
  const vEls = valid.map(c => c.el);
  const kept = valid.filter(c => !vEls.some(o => o !== c.el && o.contains(c.el)));

  // Tag containers so Python can screenshot each individually (marks map to the item index)
  if (wantMarks) {
    kept.forEach((c, i) => { try { c.el.setAttribute('data-gbv-shot', 'm' + i); } catch (e) {} });
  }

  const items = kept.map(({el, how}, i) => {
    const nested = Array.from(el.querySelectorAll(COMMENT)).filter(c => c !== el);
    const top = nested.filter(c => !nested.some(o => o !== c && o.contains(c)));
    let full = el.innerText || '';
    const comments = top.map(c => ({label: c.getAttribute('aria-label') || '', text: (c.innerText || '').trim()}))
                        .filter(c => c.text);
    comments.forEach(c => { full = full.replace(c.text, ''); });
    const msgEl = el.querySelector(MSG);
    const text = (msgEl ? msgEl.innerText : full).trim();
    const h = el.querySelector('[data-ad-rendering-role="profile_name"], h2, h3, h4');
    const author = h ? (h.innerText || '').trim().split('\n')[0] : '';
    let permalink = '', timeText = '';
    for (const a of el.querySelectorAll('a[href]')) {
      if (permaRe.test(a.href)) {
        permalink = a.href;
        timeText = (a.getAttribute('aria-label') || a.innerText || '').trim();
        break;
      }
    }
    // Media record for the multimodal pass: CDN image/poster URLs on this post only.
    const media = [];
    el.querySelectorAll('img').forEach(im => {
      const src = im.currentSrc || im.src || '';
      if (src && /scontent|fbcdn|lookaside/.test(src) && !media.includes(src)) media.push(src);
    });
    el.querySelectorAll('video').forEach(vd => {
      const src = vd.poster || vd.currentSrc || '';
      if (src && /scontent|fbcdn|lookaside/.test(src) && !media.includes(src)) media.push(src);
    });
    return {author, text, permalink, timeText, comments, how, fromMessage: !!msgEl,
            mark: wantMarks ? ('m' + i) : '', media: media.slice(0, 8)};
  });

  const byHow = {};
  cont.forEach(c => { byHow[c.how] = (byHow[c.how] || 0) + 1; });
  const diag = {
    url: location.href, title: document.title,
    bodyTextLength: (document.body.innerText || '').length,
    loginForm: !!document.querySelector('input[name="pass"]'),
    articles: document.querySelectorAll('[role="article"]').length,
    posinset: document.querySelectorAll('[aria-posinset]').length,
    messages: document.querySelectorAll(MSG).length,
    strongAnchors: strongA.length, weakAnchors: weakA.length,
    containersFound: cont.length, containersKept: kept.length, byStrategy: byHow,
  };
  if (wantDiag) {
    const roles = {}; document.querySelectorAll('[role]').forEach(e => { const r = e.getAttribute('role'); roles[r] = (roles[r] || 0) + 1; });
    diag.roles = roles;
    diag.pagelets = Array.from(new Set(Array.from(document.querySelectorAll('[data-pagelet]')).map(e => e.getAttribute('data-pagelet')))).slice(0, 40);
    const dataAd = new Set();
    document.querySelectorAll('*').forEach(e => { for (const a of e.attributes) if (a.name.startsWith('data-ad')) dataAd.add(a.name + '=' + a.value.slice(0, 40)); });
    diag.dataAdAttributes = Array.from(dataAd).slice(0, 40);
    const hs = new Set();
    anchors.forEach(a => { try { const u = new URL(a.href); if (/post|perma|story|reel|video|photo/i.test(u.pathname + u.search)) hs.add(u.pathname.slice(0, 60) + (u.search ? '?' + u.search.slice(1).split('&').map(x => x.split('=')[0]).join('&') : '')); } catch (e) {} });
    diag.sampleLinkShapes = Array.from(hs).slice(0, 15);
    diag.sampleFirstElementsText = Array.from(document.querySelectorAll('[role="main"] *')).filter(e => e.children.length === 0 && (e.innerText || '').length > 25).length;
  }
  return {items, diag};
}
"""

JS_CLICK_LABELS = """
(labels) => {
  let n = 0;
  document.querySelectorAll('div[role="button"], span[role="button"]').forEach(b => {
    const t = (b.innerText || '').trim();
    if (labels.some(l => t === l || t.startsWith(l))) { try { b.click(); n++; } catch (e) {} }
  });
  return n;
}
"""

READY_SELECTOR = '[role="article"], [data-ad-rendering-role="story_message"], [aria-posinset], [data-ad-comet-preview="message"]'


def _parse_comments(raw_comments: list[dict]) -> list[dict]:
    out = []
    for c in raw_comments:
        author, body = _clean_comment(c.get("text", ""))
        if body:
            out.append({"author": author, "text": body, "sha256": sha256(body)})
    return out


def _post_key(perma: str, author: str, text: str) -> str:
    if perma:
        return perma
    return "t:" + sha256((author + "|" + re.sub(r"\s+", " ", text)[:80]).lower())


def _element_screenshot(page, mark: str, out_dir: str, fname: str) -> str:
    """Screenshot the DOM element tagged `mark`, return a forward-slash relative path."""
    if not mark:
        return ""
    try:
        loc = page.locator(f'[data-gbv-shot="{mark}"]')
        loc.scroll_into_view_if_needed(timeout=8000)
        page.wait_for_timeout(350)  # let late images/lazy content settle
        os.makedirs(out_dir, exist_ok=True)
        path = os.path.join(out_dir, fname)
        loc.screenshot(path=path, timeout=15000)
        return path.replace(os.sep, "/")
    except Exception:
        return ""


# --------------------------------------------------------------------------- main
def scrape_page(
    url: str,
    cookies_raw: str,
    max_posts: int = 50,
    include_comments: bool = False,
    deep_comments: bool = False,
    max_comment_expansions: int = 5,
    headless: bool = True,
    capture_screenshots: bool = True,
    shots_dir: str = "output/shots",
    progress: ProgressFn = None,
    debug_dir: str = "output/debug",
) -> dict:
    """Returns {"meta": {...}, "posts": [...]} (cookies are never included).

    With capture_screenshots, each post container is also saved as a PNG under
    shots_dir/<run stamp>/ and referenced from the post dict ("screenshot").
    """
    from playwright.sync_api import sync_playwright  # lazy import

    url = normalize_url(url)
    cookies = parse_cookies(cookies_raw)
    report = progress or (lambda msg, frac: None)
    posts: dict[str, dict] = {}
    started = datetime.now(timezone.utc)
    debug: dict = {}
    run_shots_dir = os.path.join(shots_dir, started.strftime("%Y%m%d_%H%M%S"))
    shot_no = 0

    with sync_playwright() as pw:
        launch_args = ["--no-sandbox"] if os.environ.get("GBV_IN_DOCKER") == "1" else []
        browser = pw.chromium.launch(headless=headless, args=launch_args)
        ctx = browser.new_context(
            viewport={"width": 1280, "height": 900},
            user_agent=("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
            locale="en-US",
        )
        ctx.add_cookies(cookies)
        page = ctx.new_page()
        report("Opening page…", 0.0)
        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(3000)

        if any(x in page.url for x in ("login", "checkpoint")) or page.query_selector('input[name="pass"]'):
            browser.close()
            raise ScrapeError("Not logged in (or checkpoint triggered). Cookies are expired/invalid.")

        try:
            page.evaluate(JS_CLICK_LABELS, DISMISS)  # cookie / "not now" banners
            page.wait_for_selector(READY_SELECTOR, timeout=15000)
        except Exception:
            pass

        last_diag: dict = {}
        stagnant = 0
        while len(posts) < max_posts and stagnant < 6:
            page.evaluate(JS_CLICK_LABELS, SEE_MORE)
            page.wait_for_timeout(400)
            res = page.evaluate(JS_EXTRACT, {"diag": False, "marks": capture_screenshots})
            last_diag = res["diag"]
            new = 0
            for it in res["items"]:
                text = (it["text"] or "").strip()
                if not it["fromMessage"]:
                    text = _clean_post_text(text, it["author"])
                if len(text) < 2:
                    continue
                perma = canon_permalink(it["permalink"])
                key = _post_key(perma, it["author"], text)
                comments = _parse_comments(it["comments"]) if include_comments else []
                if key in posts:
                    old = posts[key]
                    if len(text) > len(old["text"]):  # "See more" expanded since last pass
                        old["text"], old["sha256"] = text, sha256(text)
                    if len(comments) > len(old["comments"]):
                        old["comments"] = comments
                    continue
                # keep the mark alive until we can shoot this new post
                post = {
                    "author": it["author"],
                    "text": text,
                    "permalink": perma,
                    "time_text": it["timeText"],
                    "comments": comments,
                    "media": it.get("media") or [],
                    "sha256": sha256(text),
                    "collected_at": datetime.now(timezone.utc).isoformat(),
                    "source": it["how"],
                    "screenshot": "",
                }
                posts[key] = post
                new += 1
                if capture_screenshots:
                    shot_no += 1
                    post["screenshot"] = _element_screenshot(
                        page, it.get("mark"), run_shots_dir, f"post_{shot_no:03d}.png")
                if len(posts) >= max_posts:
                    break
                if len(posts) >= max_posts:
                    break
            stagnant = 0 if new else stagnant + 1
            report(f"Collected {len(posts)}/{max_posts} posts", min(len(posts) / max_posts, 1.0) * 0.6)
            page.mouse.move(640, 450)
            page.mouse.wheel(0, 2600)
            page.evaluate("window.scrollBy(0, 1800)")
            time.sleep(random.uniform(2.0, 4.0))

        if not posts:
            try:
                os.makedirs(debug_dir, exist_ok=True)
                stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                shot = os.path.join(debug_dir, f"no_posts_{stamp}.png")
                page.screenshot(path=shot)
                debug = {"diagnostics": page.evaluate(JS_EXTRACT, {"diag": True, "marks": False})["diag"],
                         "screenshot": shot}
            except Exception as e:
                debug = {"diagnostics": last_diag, "error": f"{type(e).__name__}: {e}"}

        if include_comments and deep_comments and posts:
            plist = [p for p in posts.values() if p["permalink"]]
            for i, p in enumerate(plist, 1):
                report(f"Loading comments {i}/{len(plist)}", 0.6 + 0.4 * i / max(len(plist), 1))
                try:
                    cp = ctx.new_page()
                    cp.goto(p["permalink"], wait_until="domcontentloaded", timeout=45000)
                    cp.wait_for_timeout(3000)
                    for _ in range(max_comment_expansions):
                        if not cp.evaluate(JS_CLICK_LABELS, MORE_COMMENTS):
                            break
                        cp.wait_for_timeout(1800)
                    found = cp.evaluate(JS_EXTRACT, {"diag": False, "marks": capture_screenshots})["items"]
                    if found:
                        best = max(found, key=lambda x: len(x["comments"]))
                        deep = _parse_comments(best["comments"])
                        if len(deep) > len(p["comments"]):
                            p["comments"] = deep
                        if capture_screenshots and not p.get("screenshot") and best.get("mark"):
                            shot_no += 1
                            p["screenshot"] = _element_screenshot(
                                cp, best["mark"], run_shots_dir, f"post_{shot_no:03d}.png")
                    cp.close()
                except Exception:
                    pass
                time.sleep(random.uniform(2.0, 4.0))

        browser.close()

    result = list(posts.values())[:max_posts]
    meta = {
        "source_url": url,
        "collected_at_utc": started.isoformat(),
        "post_count": len(result),
        "comment_count": sum(len(p["comments"]) for p in result),
        "tool": "fb_gbv_monitor",
    }
    if debug:
        meta["debug"] = debug
    return {"meta": meta, "posts": result}
