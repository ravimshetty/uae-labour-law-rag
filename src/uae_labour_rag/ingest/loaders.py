"""Document loaders: PDF (pypdf / pdfplumber), TXT/MD, HTML.

Arabic PDFs are the hard part: extracted text can come out in visual (reversed)
order or with presentation-form glyphs. We (1) NFKC-normalise presentation forms,
(2) try pdfplumber first for Arabic (better at RTL), (3) detect reversed lines and
flip them. For scanned PDFs, run OCR upstream (e.g. Azure Document Intelligence
`ar` model or Tesseract `ara`) and feed the resulting .txt here.
"""
from __future__ import annotations

import re
from pathlib import Path

from ..text.arabic import ARABIC_LETTER_RE, fix_presentation_forms

# Common Arabic words – if their *reversed* forms dominate a line it was extracted in visual order.
_COMMON_AR = ["المادة", "العامل", "العمل", "صاحب", "الأجر", "التي", "على", "هذا"]
_COMMON_AR_REV = [w[::-1] for w in _COMMON_AR]


def _fix_reversed_line(line: str) -> str:
    if not ARABIC_LETTER_RE.search(line):
        return line
    fwd = sum(line.count(w) for w in _COMMON_AR)
    rev = sum(line.count(w) for w in _COMMON_AR_REV)
    return line[::-1] if rev > fwd else line


def _clean_page(text: str, lang: str) -> str:
    text = fix_presentation_forms(text)
    lines = []
    for line in text.splitlines():
        if lang == "ar":
            line = _fix_reversed_line(line)
        # drop page numbers / running headers that are only digits
        if re.fullmatch(r"\s*[-–]?\s*\d{1,3}\s*[-–]?\s*", line):
            continue
        lines.append(line.rstrip())
    return "\n".join(lines)


def load_pdf(path: Path, lang: str) -> str:
    pages: list[str] = []
    if lang == "ar":
        try:
            import pdfplumber

            with pdfplumber.open(path) as pdf:
                pages = [p.extract_text() or "" for p in pdf.pages]
        except Exception:  # pragma: no cover - fall back to pypdf
            pages = []
    if not pages:
        from pypdf import PdfReader

        pages = [p.extract_text() or "" for p in PdfReader(str(path)).pages]
    return "\n".join(_clean_page(p, lang) for p in pages)


def load_html(path: Path) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    return soup.get_text("\n")


def load_document(path: str | Path, lang: str) -> str:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return load_pdf(path, lang)
    if suffix in {".html", ".htm"}:
        return _clean_page(load_html(path), lang)
    return _clean_page(path.read_text(encoding="utf-8"), lang)
