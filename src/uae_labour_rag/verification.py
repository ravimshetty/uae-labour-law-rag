"""Verification report for reviewers who do not read Arabic.

Idea: never ask a reviewer to trust Arabic they can't read. Instead, check every
Arabic sentence against things an English reader CAN check:

  1. Citation  – which article(s) the sentence relies on, shown in English
                 (official EN translation for Federal law, original for DIFC).
  2. Numbers   – language-independent anchors. Every number in the Arabic sentence
                 (days, %, years) must appear in the cited article's English text.
  3. Concepts  – legal terms found in the Arabic sentence (via the bilingual glossary)
                 must match the topics of the cited English article.
  4. Back-translation – an independent LLM call translates the Arabic sentence
                 literally back into English (only with a real LLM provider).
  5. Judge     – an LLM compares back-translation vs. English source and labels
                 SUPPORTED / PARTIAL / UNSUPPORTED (only with a real LLM provider).

Each sentence gets a verdict: ✅ verified · ⚠️ review · ❌ unsupported.
"""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field

from .generation.llm import LLM, MockLLM
from .text.arabic import to_western_digits
from .text.codeswitch import Glossary

_CITE = re.compile(r"\[(S\d+)\]")
_NUM = re.compile(r"(?<![\w<])\d+(?:\.\d+)?(?![\w>])")

BACKTRANSLATE_PROMPT = """You are a certified Arabic->English legal translator.
Translate each Arabic sentence LITERALLY into English. Do not improve, summarise or
add anything. Keep [S1]-style citation markers. Return JSON: {"translations": ["...", ...]}"""

JUDGE_PROMPT = """You verify legal answers. For each item you get an English back-translation
of an answer sentence and the English text of the article it cites. Label it:
SUPPORTED (fully stated by the source), PARTIAL (partly supported or missing a condition),
UNSUPPORTED (not in the source or contradicts it). Give a short reason.
Return JSON: {"results": [{"label": "...", "reason": "..."}, ...]}"""


@dataclass
class SentenceCheck:
    arabic: str
    citations: list[str]
    cited_articles: list[str]
    source_en: str
    numbers: dict[str, bool]                 # number -> found in English source?
    concepts: list[str]                      # topics detected in the Arabic sentence
    concepts_match: bool
    back_translation: str | None = None
    judge: str | None = None
    judge_reason: str | None = None
    verdict: str = "review"                  # verified | review | unsupported
    notes: list[str] = field(default_factory=list)


def _sentences(answer_ar: str) -> list[str]:
    body = answer_ar.split("\n\nتنبيه:")[0]           # drop disclaimer
    parts = re.split(r"(?<=[.؟!:])\s+(?=\S)|\n+", body)
    out, buf = [], ""
    for p in parts:
        buf = f"{buf} {p}".strip() if buf else p.strip()
        if _CITE.search(buf):                       # a claim ends where its citation is
            out.append(buf)
            buf = ""
    if buf and out:
        out[-1] += " " + buf
    return [s for s in out if s]


def _json(raw: str) -> dict:
    m = re.search(r"\{.*\}", raw, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except json.JSONDecodeError:
        return {}


def build_report(answer_ar: str, sources: dict[str, dict], glossary: Glossary, llm: LLM) -> dict:
    checks: list[SentenceCheck] = []
    for s in _sentences(to_western_digits(answer_ar)):
        refs = list(dict.fromkeys(_CITE.findall(s)))
        srcs = [sources[r] for r in refs if r in sources]
        src_en = " ".join(x.get("text_en") or "" for x in srcs).strip()
        src_any = " ".join(f"{x.get('text_en', '')} {x.get('text_ar', '')}" for x in srcs)
        claim = _CITE.sub("", s)
        claim = re.sub(r"\((\d+)\)", " ", claim)    # article numbers like "(31)" are not claims
        nums = {n: (n in _NUM.findall(src_en or src_any)) for n in _NUM.findall(claim)}
        ar_concepts, _ = glossary.link("", claim)
        en_concepts, _ = glossary.link(src_en, "") if src_en else glossary.link("", src_any)
        topical = [c for c in ar_concepts if c not in {"employee", "employer", "difc", "mohre"}]
        c = SentenceCheck(
            arabic=s, citations=refs,
            cited_articles=[f"{x['law_en']} – Article {x['article']} ({x['title']})" for x in srcs],
            source_en=src_en[:600] or "(no English text for this source)",
            numbers=nums, concepts=[glossary.preferred_en(x) for x in ar_concepts],
            concepts_match=not topical or bool(set(topical) & set(en_concepts)),
        )
        if not refs:
            c.notes.append("no citation")
        if not src_en:
            c.notes.append("cited source has no English text – rely on back-translation")
        checks.append(c)

    # LLM-based steps (skipped in offline mock mode)
    if checks and not isinstance(llm, MockLLM):
        bt = _json(llm.complete(BACKTRANSLATE_PROMPT, json.dumps([c.arabic for c in checks], ensure_ascii=False)))
        for c, t in zip(checks, bt.get("translations", [])):
            c.back_translation = t
        items = [{"back_translation": c.back_translation, "source_en": c.source_en} for c in checks]
        jd = _json(llm.complete(JUDGE_PROMPT, json.dumps(items, ensure_ascii=False)))
        for c, r in zip(checks, jd.get("results", [])):
            c.judge, c.judge_reason = r.get("label"), r.get("reason")

    for c in checks:
        nums_ok = all(c.numbers.values())
        if c.judge == "UNSUPPORTED" or (c.numbers and not any(c.numbers.values())) or not c.citations:
            c.verdict = "unsupported"
        elif nums_ok and c.concepts_match and c.judge in (None, "SUPPORTED"):
            c.verdict = "verified"
        else:
            c.verdict = "review"
        if not nums_ok:
            c.notes.append("number(s) not found in English source: " +
                           ", ".join(n for n, ok in c.numbers.items() if not ok))
        if not c.concepts_match:
            c.notes.append("topic of sentence does not match cited article")

    summary = {v: sum(c.verdict == v for c in checks) for v in ("verified", "review", "unsupported")}
    return {
        "mode": "llm" if checks and checks[0].back_translation is not None else "deterministic",
        "summary": summary,
        "sentences": [asdict(c) for c in checks],
    }
