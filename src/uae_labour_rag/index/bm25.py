"""Small BM25Okapi implementation with Arabic-aware tokenisation.

Arabic is morphologically rich: "العامل", "للعامل", "والعامل" are the same word with
clitics. We apply light prefix stripping (و ف ب ل ال لل) after search
normalisation – a cheap ISRI-style stem that gives most of the recall benefit.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from ..text.arabic import normalize_for_search

_TOKEN = re.compile(r"[ء-ي]+|[a-z0-9]+")
_AR_PREFIXES = ("وال", "بال", "كال", "فال", "لل", "ال", "و", "ف", "ب", "ل")
_AR_SUFFIXES = ("ات", "ون", "ين", "ها", "هم", "ه", "ي")
_EN_STOP = set("""a an the of to in on for and or is are be by with at as from that this
these those it its can may shall will what which who how when does do did my i me we our
you your after before under about per into than""".split())
_AR_STOP = set("في من على الى عن او ان هذا هذه التي الذي ما لا كل".split())


def _stem_ar(tok: str) -> str:
    for p in _AR_PREFIXES:
        if tok.startswith(p) and len(tok) - len(p) >= 3:
            tok = tok[len(p):]
            break
    for s in _AR_SUFFIXES:
        if tok.endswith(s) and len(tok) - len(s) >= 3:
            tok = tok[: -len(s)]
            break
    return tok


def _stem_en(tok: str) -> str:
    for s in ("ings", "ing", "ies", "ed", "es", "s"):
        if tok.endswith(s) and len(tok) - len(s) >= 3:
            return tok[: -len(s)] + ("y" if s == "ies" else "")
    return tok


def tokenize(text: str) -> list[str]:
    out = []
    for tok in _TOKEN.findall(normalize_for_search(text)):
        if tok[0].isascii():
            if tok not in _EN_STOP:
                out.append(_stem_en(tok))
        elif tok not in _AR_STOP:
            out.append(_stem_ar(tok))
    return out


class BM25:
    def __init__(self, corpus: list[str], k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = [Counter(tokenize(d)) for d in corpus]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avgdl = (sum(self.lens) / len(self.lens)) if self.lens else 0.0
        df: Counter = Counter()
        for d in self.docs:
            df.update(d.keys())
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def scores(self, query: str) -> list[float]:
        q = tokenize(query)
        res = []
        for d, dl in zip(self.docs, self.lens):
            s = 0.0
            for t in q:
                tf = d.get(t)
                if not tf:
                    continue
                s += self.idf[t] * tf * (self.k1 + 1) / (tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1)))
            res.append(s)
        return res
