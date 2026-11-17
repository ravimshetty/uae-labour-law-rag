"""Article-aware chunking.

Legal text must never be split mid-article without keeping the article identity,
otherwise citations break. Strategy (parent / child):

* **Parent = one Article** (e.g. Article 51 – End of Service Gratuity). Parents are
  what the LLM sees and what we cite.
* **Child = sub-chunk of a parent** (clause-aligned, ~1,200 chars, 150 overlap).
  Children are what we embed and BM25-index – small chunks retrieve better.

Table-of-contents duplicates ("Article (21) .... 22") are removed by keeping the
longest body per article number.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from ..models import Chunk, Jurisdiction, Lang
from ..text.arabic import normalize_for_display, to_western_digits

MAX_CHARS = 1200
OVERLAP = 150


@dataclass
class Article:
    number: str
    title: str
    body: str


def split_articles(text: str, article_pattern: str, stop_pattern: str | None = None,
                   skip_until_pattern: str | None = None) -> list[Article]:
    lines = text.splitlines()
    if skip_until_pattern:
        rx = re.compile(skip_until_pattern)
        for i, line in enumerate(lines):
            if rx.search(line):
                lines = lines[i:]
                break
    head_rx = re.compile(article_pattern)
    stop_rx = re.compile(stop_pattern) if stop_pattern else None

    found: dict[str, Article] = {}
    cur_num, cur_title, buf = None, "", []

    def flush():
        if cur_num is None:
            return
        body = normalize_for_display("\n".join(buf)).strip()
        art = Article(cur_num, cur_title.strip(), body)
        prev = found.get(cur_num)
        if prev is None or len(art.body) > len(prev.body):  # drop TOC duplicates
            found[cur_num] = art

    for line in lines:
        if stop_rx and stop_rx.search(line):
            break
        m = head_rx.match(line)
        if m:
            flush()
            cur_num = to_western_digits(m.group(1))
            # title: inline group 2 (DIFC style) or remainder of the heading line
            inline = m.group(2) if m.lastindex and m.lastindex >= 2 else line[m.end():]
            cur_title = re.sub(r"^[\s:\-–.]+", "", inline or "")
            buf = []
            continue
        if cur_num is not None:
            if not cur_title and line.strip() and len(line.strip()) < 90:
                cur_title = line.strip()  # heading on the line after "Article (n)"
                continue
            buf.append(line)
    flush()
    # Ignore near-empty bodies (pure TOC entries that had no body anywhere)
    return [a for a in sorted(found.values(), key=lambda a: int(a.number)) if len(a.body) > 40]


def _split_body(body: str) -> list[str]:
    if len(body) <= MAX_CHARS:
        return [body]
    # Prefer clause boundaries: "1.", "(a)", "أ-", "1-" at line start
    parts = re.split(r"\n(?=\s*(?:\d+[.\-)]|\([a-z0-9]+\)|[أ-ي][\-)]))", body)
    chunks, cur = [], ""
    for p in parts:
        if len(cur) + len(p) > MAX_CHARS and cur:
            chunks.append(cur.strip())
            cur = cur[-OVERLAP:] + "\n" + p
        else:
            cur = f"{cur}\n{p}" if cur else p
    if cur.strip():
        chunks.append(cur.strip())
    # Hard-split anything still too long
    out = []
    for c in chunks:
        while len(c) > MAX_CHARS * 1.5:
            out.append(c[:MAX_CHARS])
            c = c[MAX_CHARS - OVERLAP:]
        out.append(c)
    return out


def articles_to_chunks(articles: list[Article], *, doc: dict) -> tuple[list[Chunk], list[Chunk]]:
    """Return (parents, children)."""
    jur = Jurisdiction(doc["jurisdiction"])
    lang = Lang(doc["lang"])
    parents, children = [], []
    for a in articles:
        base = dict(
            doc_id=doc["doc_id"], jurisdiction=jur, lang=lang, article=a.number,
            title=a.title, law_name=doc["law_name"], source_url=doc.get("url", ""),
            authoritative=bool(doc.get("authoritative")),
            pair_id=f"{doc.get('pair', doc['doc_id'])}:art{a.number}",
            meta={"law_name_ar": doc.get("law_name_ar", doc["law_name"] if lang == Lang.AR else ""),
                  "law_name_en": doc.get("law_name_en", doc["law_name"] if lang == Lang.EN else "")},
        )
        pid = f"{doc['doc_id']}:art{a.number}"
        parents.append(Chunk(chunk_id=pid, text=a.body, **base))
        for i, piece in enumerate(_split_body(a.body)):
            heading = f"{a.title}\n" if a.title else ""
            children.append(Chunk(chunk_id=f"{pid}:{i}", text=heading + piece,
                                  **{**base, "meta": {**base["meta"], "parent_id": pid}}))
    return parents, children
