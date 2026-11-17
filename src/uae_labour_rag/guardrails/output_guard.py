"""Output guardrails – run AFTER generation, BEFORE returning to the user.

| check               | failure action                                   |
|---------------------|--------------------------------------------------|
| citations present   | regenerate once, then fallback "no_context"      |
| citations valid     | strip refs that point to non-existent sources    |
| numeric grounding   | regenerate once; then warn flag                  |
| Arabic ratio        | regenerate once; then warn flag                  |
| PII leak            | re-redact the answer (never return raw PII)      |
| no invented placeholders | strip + warn flag                           |
| disclaimer          | always appended                                  |
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from ..models import GuardrailFlag
from ..privacy.pii import PIIVault, contains_pii, redact
from ..text.arabic import arabic_ratio, to_western_digits

CITATION_RE = re.compile(r"\[(S\d+)\]")
NUM_RE = re.compile(r"(?<![\w<])\d+(?:[.,]\d+)?(?![\w>])")
DISCLAIMER_AR = ("تنبيه: هذه المعلومات لأغراض توعوية عامة ولا تُعد استشارة قانونية. "
                 "للحالات الخاصة يُرجى مراجعة محامٍ مرخّص أو وزارة الموارد البشرية والتوطين "
                 "أو محاكم مركز دبي المالي العالمي.")
HIGH_STAKES_AR = "نظراً لحساسية موضوع سؤالك، ننصح بشدة بالحصول على استشارة قانونية متخصصة قبل اتخاذ أي إجراء."


@dataclass
class OutputCheck:
    text: str
    flags: list[GuardrailFlag] = field(default_factory=list)
    needs_regen: bool = False
    citations_used: list[str] = field(default_factory=list)


def _numbers(text: str) -> set[str]:
    return {n.replace(",", "") for n in NUM_RE.findall(to_western_digits(text))}


def check_output(answer: str, *, source_texts: dict[str, str], query: str, vault: PIIVault,
                 min_arabic_ratio: float = 0.6, require_citations: bool = True,
                 numeric_grounding: bool = True) -> OutputCheck:
    flags: list[GuardrailFlag] = []
    regen = False
    text = to_western_digits(answer).strip()

    # 1. citation validity
    refs = CITATION_RE.findall(text)
    invalid = [r for r in refs if r not in source_texts]
    for r in set(invalid):
        text = text.replace(f"[{r}]", "")
    if invalid:
        flags.append(GuardrailFlag("invalid_citation", "warn", ",".join(sorted(set(invalid)))))
    used = sorted({r for r in refs if r in source_texts}, key=lambda s: int(s[1:]))
    if require_citations and not used:
        flags.append(GuardrailFlag("missing_citations", "warn"))
        regen = True

    # 2. numeric grounding: every number must exist in sources or the user's query
    if numeric_grounding:
        allowed = _numbers(" ".join(source_texts.values())) | _numbers(query)
        body = CITATION_RE.sub("", text)
        ungrounded = sorted(n for n in _numbers(body) if n not in allowed and len(n) < 7)
        if ungrounded:
            flags.append(GuardrailFlag("ungrounded_numbers", "warn", ",".join(ungrounded)))
            regen = True

    # 3. language
    ratio = arabic_ratio(re.sub(r"<[A-Z_]+_\d+>", "", CITATION_RE.sub("", text)))
    if ratio < min_arabic_ratio:
        flags.append(GuardrailFlag("not_arabic", "warn", f"arabic_ratio={ratio:.2f}"))
        regen = True

    # 4. PII leak – the model must never emit raw PII (only placeholders)
    leaked = contains_pii(text)
    if leaked:
        text, _ = redact(text, vault)
        flags.append(GuardrailFlag("pii_leak_redacted", "warn", ",".join(leaked)))

    # 5. placeholder integrity – the model must not invent placeholders
    invented = [ph for ph in re.findall(r"<[A-Z_]+_\d+>", text) if ph not in vault.mapping]
    if invented:
        for ph in invented:
            text = text.replace(ph, "")
        flags.append(GuardrailFlag("invented_placeholder", "warn", ",".join(invented)))

    return OutputCheck(text, flags, regen, used)


def finalize(text: str, high_stakes: bool) -> str:
    parts = [text.strip()]
    if high_stakes:
        parts.append(HIGH_STAKES_AR)
    parts.append(DISCLAIMER_AR)
    return "\n\n".join(parts)
