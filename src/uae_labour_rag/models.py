"""Core data models shared across the pipeline."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class Jurisdiction(str, Enum):
    FEDERAL = "federal"  # Onshore UAE – Federal Decree-Law No. 33 of 2021 (MOHRE)
    DIFC = "difc"        # DIFC Employment Law No. 2 of 2019
    BOTH = "both"        # Ambiguous – retrieve from both and compare


class Lang(str, Enum):
    EN = "en"
    AR = "ar"


@dataclass
class Chunk:
    """A retrievable unit. Each chunk belongs to exactly one article of one law."""

    chunk_id: str               # e.g. "federal:art51:en:0"
    doc_id: str                 # e.g. "fdl-33-2021"
    jurisdiction: Jurisdiction
    lang: Lang
    article: str                # normalised article number, e.g. "51"
    title: str                  # article heading in the chunk language
    text: str                   # chunk text (cleaned)
    law_name: str               # human readable law name in the chunk language
    source_url: str = ""
    authoritative: bool = False # True if this language version prevails legally
    pair_id: str = ""           # links EN<->AR versions of the same article
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def article_key(self) -> str:
        return f"{self.jurisdiction.value}:art{self.article}"

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["jurisdiction"] = self.jurisdiction.value
        d["lang"] = self.lang.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Chunk":
        d = dict(d)
        d["jurisdiction"] = Jurisdiction(d["jurisdiction"])
        d["lang"] = Lang(d["lang"])
        return cls(**d)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    sources: dict[str, float] = field(default_factory=dict)  # per-retriever scores/ranks


@dataclass
class Citation:
    ref: str            # "S1"
    law_name_ar: str
    law_name_en: str
    article: str
    title: str
    snippet: str
    lang: str
    source_url: str
    authoritative: bool


@dataclass
class GuardrailFlag:
    name: str
    severity: str       # "info" | "warn" | "block"
    detail: str = ""


@dataclass
class PipelineResult:
    request_id: str
    answer_ar: str
    english_gist: str
    jurisdiction: str
    citations: list[Citation]
    flags: list[GuardrailFlag]
    blocked: bool = False
    redactions: dict[str, int] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)
    verification: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "answer_ar": self.answer_ar,
            "english_gist": self.english_gist,
            "jurisdiction": self.jurisdiction,
            "citations": [asdict(c) for c in self.citations],
            "flags": [asdict(f) for f in self.flags],
            "blocked": self.blocked,
            "redactions": self.redactions,
            "debug": self.debug,
            "verification": self.verification,
        }
