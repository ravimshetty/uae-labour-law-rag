"""Bilingual code-switching handler.

Real UAE users rarely type clean English. Typical queries look like:

    "how many days مكافأة نهاية الخدمة after 3 years?"
    "is my kafeel allowed to keep my passport during fatrat al tajruba"
    "notice period ٣٠ يوم for DIFC?"

This module turns such a query into a structured `NormalizedQuery`:

1. **Script segmentation** – split into Latin / Arabic / numeric spans.
2. **Digit normalisation** – Arabic-Indic digits -> Western so "٣٠" == "30".
3. **Arabizi detection** – Latin-script Arabic ("3amal", "ijaza") is mapped via
   the glossary to a canonical concept.
4. **Concept linking** – EN, AR and Arabizi surface forms -> concept IDs
   (e.g. `end_of_service_gratuity`).
5. **Query expansion** – builds an English and an Arabic retrieval query, so
   BM25 hits both language versions of each article, and the dense retriever gets
   a clean monolingual string.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .arabic import (
    arabic_ratio,
    is_arabic_token,
    normalize_for_search,
    to_western_digits,
)

_TOKEN_RE = re.compile(r"[؀-ۿݐ-ݿ]+|[A-Za-z0-9'\-]+|[0-9٠-٩۰-۹]+")
# Arabizi uses digits as letters: 2=ء 3=ع 5=خ 6=ط 7=ح 8=ق 9=ص
_ARABIZI_DIGIT_RE = re.compile(r"[a-z]+[235679][a-z]+|[235679][a-z]{2,}", re.I)


@dataclass
class Segment:
    text: str
    script: str  # "latin" | "arabic" | "numeric"


@dataclass
class NormalizedQuery:
    original: str
    segments: list[Segment]
    concepts: list[str]
    query_en: str
    query_ar: str
    is_code_switched: bool
    arabizi_tokens: list[str] = field(default_factory=list)
    arabic_ratio: float = 0.0

    def debug(self) -> dict:
        return {
            "concepts": self.concepts,
            "query_en": self.query_en,
            "query_ar": self.query_ar,
            "code_switched": self.is_code_switched,
            "arabizi": self.arabizi_tokens,
            "arabic_ratio": round(self.arabic_ratio, 2),
        }


class Glossary:
    def __init__(self, concepts: dict[str, dict[str, list[str]]]):
        self.concepts = concepts
        # surface-form -> concept, longest-first matching
        self._en: list[tuple[str, str]] = []
        self._ar: list[tuple[str, str]] = []
        self._arabizi: list[tuple[str, str]] = []
        for cid, forms in concepts.items():
            for f in forms.get("en", []) or []:
                self._en.append((f.lower(), cid))
            for f in forms.get("ar", []) or []:
                self._ar.append((normalize_for_search(f), cid))
            for f in forms.get("arabizi", []) or []:
                self._arabizi.append((f.lower(), cid))
        for lst in (self._en, self._ar, self._arabizi):
            lst.sort(key=lambda x: -len(x[0]))

    @classmethod
    def load(cls, path: str | Path) -> "Glossary":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        return cls(data["concepts"])

    def preferred_ar(self, cid: str) -> str:
        return (self.concepts[cid].get("ar") or [cid])[0]

    def preferred_en(self, cid: str) -> str:
        return (self.concepts[cid].get("en") or [cid])[0]

    @staticmethod
    def _find(text: str, table: list[tuple[str, str]], word_boundary: bool) -> list[tuple[str, str]]:
        hits, consumed = [], text
        for form, cid in table:
            pattern = rf"(?<![\w]){re.escape(form)}(?![\w])" if word_boundary else re.escape(form)
            if re.search(pattern, consumed):
                hits.append((form, cid))
                consumed = re.sub(pattern, " ", consumed)  # avoid double counting sub-phrases
        return hits

    def link(self, latin_text: str, arabic_text: str) -> tuple[list[str], list[str]]:
        """Return (concept_ids, arabizi_forms_found)."""
        latin_low = latin_text.lower()
        concepts: list[str] = []
        arabizi_found: list[str] = []
        for _, cid in self._find(latin_low, self._en, True):
            concepts.append(cid)
        for form, cid in self._find(latin_low, self._arabizi, True):
            concepts.append(cid)
            arabizi_found.append(form)
        for _, cid in self._find(normalize_for_search(arabic_text), self._ar, False):
            concepts.append(cid)
        # de-duplicate, preserve order
        seen: set[str] = set()
        return [c for c in concepts if not (c in seen or seen.add(c))], arabizi_found


def segment(text: str) -> list[Segment]:
    segs: list[Segment] = []
    for tok in _TOKEN_RE.findall(text):
        script = "arabic" if is_arabic_token(tok) else (
            "numeric" if to_western_digits(tok).isdigit() else "latin")
        if segs and segs[-1].script == script:
            segs[-1].text += " " + tok
        else:
            segs.append(Segment(tok, script))
    return segs


_PLACEHOLDER_RE = re.compile(r"<[A-Z_]+_\d+>")


def normalize_query(text: str, glossary: Glossary) -> NormalizedQuery:
    text = to_western_digits(text.strip())
    text = _PLACEHOLDER_RE.sub(" ", text)  # PII placeholders carry no retrieval signal
    segs = segment(text)
    latin = " ".join(s.text for s in segs if s.script in {"latin", "numeric"})
    arabic = " ".join(s.text for s in segs if s.script == "arabic")
    concepts, arabizi = glossary.link(latin, arabic)

    # Arabizi tokens not in glossary (digit-letters like "3amal") are flagged for debugging
    for tok in latin.split():
        if _ARABIZI_DIGIT_RE.fullmatch(tok) and tok.lower() not in arabizi:
            arabizi.append(tok.lower())

    en_terms = [glossary.preferred_en(c) for c in concepts]
    ar_terms = [glossary.preferred_ar(c) for c in concepts]

    # English retrieval query: Latin part (minus arabizi noise) + EN names of concepts found in AR
    latin_clean = " ".join(t for t in latin.split() if t.lower() not in set(arabizi))
    query_en = " ".join(dict.fromkeys([latin_clean, *en_terms])).strip()
    # Arabic retrieval query: Arabic part + AR terms for concepts found anywhere.
    # (numbers are left out: "7 years" would otherwise match any article containing 7)
    query_ar = " ".join(dict.fromkeys([arabic, *ar_terms])).strip()

    has_ar = any(s.script == "arabic" for s in segs)
    has_la = any(s.script == "latin" for s in segs)
    return NormalizedQuery(
        original=text,
        segments=segs,
        concepts=concepts,
        query_en=query_en or text,
        query_ar=query_ar,
        is_code_switched=(has_ar and has_la) or bool(arabizi),
        arabizi_tokens=arabizi,
        arabic_ratio=arabic_ratio(text),
    )
