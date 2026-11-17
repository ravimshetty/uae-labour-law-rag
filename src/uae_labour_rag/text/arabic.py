"""Arabic text normalisation.

Two levels:
  * `normalize_for_display` – light: unify digits/whitespace, drop tatweel. Keeps the
    official legal text readable.
  * `normalize_for_search` – aggressive: also strips diacritics (tashkeel) and folds
    alef / ya / ta-marbuta / hamza variants so BM25 matches spelling variants
    (e.g. "مكافأة" == "مكافاة", "الإجازة" == "الاجازة").
"""
from __future__ import annotations

import re
import unicodedata

ARABIC_INDIC = "٠١٢٣٤٥٦٧٨٩"
EXT_ARABIC_INDIC = "۰۱۲۳۴۵۶۷۸۹"  # Persian/Urdu digits sometimes appear in PDFs
WESTERN = "0123456789"
_TO_WESTERN = str.maketrans(ARABIC_INDIC + EXT_ARABIC_INDIC, WESTERN * 2)
_TO_ARABIC_INDIC = str.maketrans(WESTERN, ARABIC_INDIC)

TATWEEL = "ـ"
DIACRITICS_RE = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
ARABIC_LETTER_RE = re.compile(r"[ء-يٱ-ۓ]")
LATIN_LETTER_RE = re.compile(r"[A-Za-z]")
# Presentation forms (U+FB50–U+FDFF, U+FE70–U+FEFF) appear in text extracted from PDFs.
PRESENTATION_RE = re.compile(r"[ﭐ-﷿ﹰ-﻿]")

_FOLD = str.maketrans({
    "أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا",
    "ى": "ي", "ئ": "ي",
    "ؤ": "و",
    "ة": "ه",
})


def to_western_digits(text: str) -> str:
    return text.translate(_TO_WESTERN)


def to_arabic_indic_digits(text: str) -> str:
    return text.translate(_TO_ARABIC_INDIC)


def fix_presentation_forms(text: str) -> str:
    """NFKC maps Arabic presentation forms (common in PDF extraction) back to base letters."""
    if PRESENTATION_RE.search(text):
        return unicodedata.normalize("NFKC", text)
    return text


def normalize_for_display(text: str) -> str:
    text = fix_presentation_forms(text)
    text = text.replace(TATWEEL, "")
    text = to_western_digits(text)
    text = text.replace("‏", "").replace("‎", "")  # RLM / LRM marks
    return re.sub(r"[ \t]+", " ", text).strip()


def normalize_for_search(text: str) -> str:
    text = normalize_for_display(text)
    text = DIACRITICS_RE.sub("", text)
    text = text.translate(_FOLD)
    return text.lower()


def arabic_ratio(text: str) -> float:
    """Share of Arabic letters among all letters (Arabic + Latin)."""
    ar = len(ARABIC_LETTER_RE.findall(text))
    la = len(LATIN_LETTER_RE.findall(text))
    total = ar + la
    return ar / total if total else 0.0


def is_arabic_token(token: str) -> bool:
    return bool(ARABIC_LETTER_RE.search(token))
