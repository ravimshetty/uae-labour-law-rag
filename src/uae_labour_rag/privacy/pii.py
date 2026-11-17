"""PII detection + reversible pseudonymisation (UAE-specific patterns).

Users paste real details into legal questions ("my Emirates ID 784-1990-1234567-1,
salary 12,000, call me on 050 123 4567"). None of that is needed to answer a
question about the law, so before anything leaves the process (LLM, logs, traces)
we replace PII with typed placeholders:

    784-1990-1234567-1  ->  <EMIRATES_ID_1>

The mapping lives only in a per-request `PIIVault` (memory, never persisted). The
LLM is instructed to keep placeholders verbatim; the answer can optionally be
re-hydrated for the end user (`privacy.rehydrate_user_pii_in_answer`).

For production, layer Microsoft Presidio (with an Arabic NER model such as
CAMeLBERT-NER) on top of these regexes for names/addresses – see docs/privacy.md.
"""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field

from ..text.arabic import to_western_digits


def _luhn_ok(num: str) -> bool:
    digits = [int(d) for d in re.sub(r"\D", "", num)][::-1]
    total = sum(d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2) for i, d in enumerate(digits))
    return len(digits) >= 13 and total % 10 == 0


# Order matters: more specific patterns first.
PATTERNS: list[tuple[str, re.Pattern, callable | None]] = [
    ("EMAIL", re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}"), None),
    ("IBAN", re.compile(r"\bAE\d{2}(?:\s?\d){19}\b", re.I), None),
    ("EMIRATES_ID", re.compile(r"\b784[-\s]?\d{4}[-\s]?\d{7}[-\s]?\d\b"), None),
    ("CARD", re.compile(r"\b(?:\d[ -]?){13,19}\b"), _luhn_ok),
    ("PHONE", re.compile(r"(?:(?<=\s)|^|(?<=[(:]))(?:\+971|00971|0)[\s-]?(?:5[0-9]|[2-9])[\s-]?\d{3}[\s-]?\d{4}\b"), None),
    ("PASSPORT", re.compile(r"(?i)(?<=passport)(?:\s*(?:no\.?|number|#|:))?\s*([A-Z]{1,2}\d{6,8})\b|"
                            r"(?<=جواز السفر)\s*(?:رقم)?\s*([A-Z]{1,2}\d{6,8})"), None),
    ("PERSON", re.compile(r"(?i)(?:my name is|i am called|this is)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2})|"
                          r"(?:اسمي)\s+([ء-ي]+(?:\s+[ء-ي]+){0,2})"), None),
]


@dataclass
class PIIVault:
    mapping: dict[str, str] = field(default_factory=dict)   # placeholder -> original
    counts: Counter = field(default_factory=Counter)

    def add(self, kind: str, value: str) -> str:
        for ph, orig in self.mapping.items():
            if orig == value:
                return ph
        self.counts[kind] += 1
        ph = f"<{kind}_{self.counts[kind]}>"
        self.mapping[ph] = value
        return ph

    def rehydrate(self, text: str) -> str:
        for ph, orig in self.mapping.items():
            text = text.replace(ph, orig)
        return text

    def summary(self) -> dict[str, int]:
        return dict(self.counts)


def redact(text: str, vault: PIIVault | None = None) -> tuple[str, PIIVault]:
    vault = vault or PIIVault()
    text = to_western_digits(text)
    for kind, rx, validator in PATTERNS:
        def _sub(m: re.Match) -> str:
            # use the capturing group if the pattern has one that matched (PASSPORT, PERSON)
            groups = [g for g in m.groups() if g] if m.groups() else []
            value = groups[0] if groups else m.group(0)
            if validator and not validator(value):
                return m.group(0)
            ph = vault.add(kind, value.strip())
            return m.group(0).replace(value.strip(), ph)
        text = rx.sub(_sub, text)
    return text, vault


def contains_pii(text: str) -> list[str]:
    """Kinds of raw PII present (used by the output guardrail to catch leaks)."""
    found = []
    for kind, rx, validator in PATTERNS:
        if kind == "PERSON":
            continue  # too fuzzy for output scanning
        for m in rx.finditer(to_western_digits(text)):
            value = next((g for g in m.groups() if g), m.group(0)) if m.groups() else m.group(0)
            if validator is None or validator(value):
                found.append(kind)
                break
    return found
