"""Local lexicon safety net for GBV classification (no network, stdlib only).

The LLM pass misses items on dialect drift, sarcasm and rare slurs. This module
is a fast recall floor: a curated Darija/French/English lexicon of threat,
slur, sexual-harassment and anti-rights framings (incl. Arabizi transliteration
variants and leet-digit spellings) used to:

  1. prioritise batches (lexicon-positive items are classified first),
  2. flag *discrepancies*: lexicon-positive but LLM-negative items are pushed to
     the adversarial verifier, and
  3. badge items in the review UI so reviewers can see the independent signal.

It is deliberately conservative: prompt-injected matching avoids neutral words
(«مرا» = woman in Darija is NOT in the lexicon), and everything here is a lead
for review, never a verdict on its own.
"""
from __future__ import annotations

import re

# (kind, term, english gloss). kinds: threat | slur | sexual | anti_rights
TERMS: list[tuple[str, str, str]] = [
    # ---- threats of violence
    ("threat", "نقتلها", "kill her (1pl.)"),
    ("threat", "نقتلوها", "kill her/them (variant)"),
    ("threat", "يقتلوها", "they kill her"),
    ("threat", "نحرقها", "burn her"),
    ("threat", "نحرقوها", "burn her (variant)"),
    ("threat", "تستحق الموت", "deserves death"),
    ("threat", "deserve to die", "(EN)"),
    ("threat", "deserves to die", "(EN variant)"),
    ("threat", "تستحق الغرق", "deserves to drown"),
    ("threat", "نغرقوها", "drown her"),
    ("threat", "نضربوها", "beat her (1pl.)"),
    ("threat", "نكسرها", "break her"),
    ("threat", "نطعنوها", "stab her"),
    ("threat", "لازم تموت", "must die"),
    ("threat", "mérite de mourir", "deserves to die (FR)"),
    ("threat", "doit mourir", "must die (FR)"),
    ("threat", "on la tue", "we kill her (FR)"),
    ("threat", "deserve to die", "(EN)"),
    ("threat", "kill her", "(EN)"),
    ("threat", "stone her", "(EN)"),
    ("threat", "hang her", "(EN)"),
    ("threat", "acid attack", "(EN)"),
    # ---- gendered slurs / insults
    ("slur", "شرموط", "Darija slur"),
    ("slur", "شراميط", "Darija slur (pl.)"),
    ("slur", "قحب", "Darija slur"),
    ("slur", "كحب", "Darija slur (variant)"),
    ("slur", "عاهرة", "MSA slur"),
    ("slur", "ساقطة", "Darija slur"),
    ("slur", "ناقصة عقل", "'incomplete mind' — sexist insult"),
    ("slur", "sharmuta", "transliterated slur"),
    ("slur", "sharmouta", "transliterated slur"),
    ("slur", "charmouta", "transliterated slur"),
    ("slur", "9ahba", "leet-transliterated slur"),
    ("slur", "qahba", "transliterated slur"),
    ("slur", "ka7ba", "leet-transliterated slur"),
    ("slur", "kahba", "transliterated slur"),
    ("slur", "salope", "FR slur"),
    ("slur", "pute", "FR slur"),
    ("slur", "chaudasse", "FR slur"),
    ("slur", "gouine", "FR homophobic+misogynistic slur"),
    ("slur", "whore", "EN slur"),
    ("slur", "slut", "EN slur"),
    # ---- sexual harassment
    ("sexual", "عريانة", "naked/nude (as threat)"),
    ("sexual", "بورنو", "porn"),
    ("sexual", "nhawes", "transliterated harass"),
    # ---- anti-rights / demonising framings
    ("anti_rights", "معاهدة الخراب", "loaded name for CEDAW"),
    ("anti_rights", "تدمير الأسرة", "destroy-the-family narrative"),
    ("anti_rights", "تدمير العائلة", "destroy-the-family narrative"),
    ("anti_rights", "القضاء على الأسرة", "end-the-family narrative"),
    ("anti_rights", "حرب على الأسرة", "war-on-the-family narrative"),
    ("anti_rights", "lavage de cerveau", "FR brainwashing framing"),
]
TERMS = [t for t in TERMS if t[2].strip()]  # drop empty-gloss placeholders

# leet/Arabizi digit map (Tunisian chat convention) applied to the *latin* text:
_LEET = {str(d): c for d, c in [(3, "ع"), (7, "ح"), (9, "ق"), (5, "خ"), (2, "ء"), (8, "ع")]}

_HARAKAT_RE = re.compile(r"[\u064b-\u065f\u0670\u0640]")


def normalize(text: str) -> str:
    """NFC-ish folding used ONLY for lexicon matching (original text untouched):
    strip harakat/tatweel, fold alef forms and ة→ه (common in Darija spelling),
    map leet digits to Arabic letters, fold latin accents, lowercase."""
    t = str(text or "")
    t = _HARAKAT_RE.sub("", t)
    t = (t.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا").replace("ٱ", "ا")
           .replace("ة", "ه").replace("ى", "ي").replace("گ", "ك"))
    for d, c in _LEET.items():
        t = t.replace(d, c)
    t = (t.replace("é", "e").replace("è", "e").replace("ê", "e").replace("à", "a")
           .replace("ç", "c").replace("î", "i").replace("ô", "o").replace("û", "u"))
    return t.lower()


class _Rule:
    __slots__ = ("kind", "norm", "latin", "term", "gloss")

    def __init__(self, kind: str, term: str, gloss: str):
        self.kind, self.term, self.gloss = kind, term, gloss
        self.norm = normalize(term)
        self.latin = term.isascii()  # match with word boundaries

    def hits(self, hay: str) -> bool:
        if self.latin:
            return re.search(rf"\b{re.escape(self.norm)}\b", hay) is not None
        return self.norm in hay


_RULES = [_Rule(*t) for t in TERMS]


def scan(text: str) -> list[dict]:
    """Return [{"kind","term","gloss"}] for every lexicon term present in `text`."""
    hay = normalize(text)
    if not hay.strip():
        return []
    out, seen = [], set()
    for r in _RULES:
        if r.hits(hay):
            k = (r.kind, r.term)
            if k not in seen:
                seen.add(k)
                out.append({"kind": r.kind, "term": r.term, "gloss": r.gloss})
    return out


KINDS = ("threat", "slur", "sexual", "anti_rights")


def worst_kind(hits: list[dict]) -> str:
    """Highest-priority kind present (threat > slur > sexual > anti_rights)."""
    for k in KINDS:
        if any(h["kind"] == k for h in hits):
            return k
    return ""


def describe(hits: list[dict]) -> str:
    """Compact human badge text for the UI / review inbox."""
    if not hits:
        return ""
    return "; ".join(f"{h['kind']}:{h['term']}" for h in hits[:4]) + (" …" if len(hits) > 4 else "")
