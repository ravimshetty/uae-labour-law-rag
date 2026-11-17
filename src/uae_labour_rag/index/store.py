"""Knowledge base: parents + children + BM25 + dense vectors, persisted to disk.

Default backend is an in-process NumPy matrix (a few thousand chunks – both laws
together are < 2k children – so brute-force cosine is < 5 ms). For multi-tenant
production swap `VectorIndex` for Qdrant / pgvector / Azure AI Search; the
`search()` contract is the only thing the retriever depends on.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..models import Chunk, Jurisdiction
from .bm25 import BM25
from .embeddings import Embedder


class VectorIndex:
    def __init__(self, vectors: np.ndarray):
        self.vectors = vectors.astype(np.float32)

    def search(self, qvec: np.ndarray, mask: np.ndarray, k: int) -> list[tuple[int, float]]:
        if not len(self.vectors):
            return []
        sims = self.vectors @ qvec
        sims = np.where(mask, sims, -np.inf)
        idx = np.argsort(-sims)[:k]
        return [(int(i), float(sims[i])) for i in idx if np.isfinite(sims[i])]


class KnowledgeBase:
    def __init__(self, parents: list[Chunk], children: list[Chunk], vectors: np.ndarray):
        self.parents = {p.chunk_id: p for p in parents}
        self.children = children
        self.vectors = VectorIndex(vectors)
        self.bm25 = BM25([c.text for c in children])
        self._jur = np.array([c.jurisdiction.value for c in children])
        # pair index: pair_id -> {lang: parent}
        self.pairs: dict[str, dict[str, Chunk]] = {}
        for p in parents:
            self.pairs.setdefault(p.pair_id, {})[p.lang.value] = p

    # ------------------------------------------------------------------ build / io
    @classmethod
    def build(cls, parents: list[Chunk], children: list[Chunk], embedder: Embedder) -> "KnowledgeBase":
        vecs = embedder.embed([c.text for c in children])
        return cls(parents, children, vecs)

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / "parents.jsonl").write_text(
            "\n".join(json.dumps(p.to_dict(), ensure_ascii=False) for p in self.parents.values()), "utf-8")
        (d / "children.jsonl").write_text(
            "\n".join(json.dumps(c.to_dict(), ensure_ascii=False) for c in self.children), "utf-8")
        np.save(d / "vectors.npy", self.vectors.vectors)

    @classmethod
    def load(cls, directory: str | Path) -> "KnowledgeBase":
        d = Path(directory)

        def read(name: str) -> list[Chunk]:
            text = (d / name).read_text("utf-8").strip()
            return [Chunk.from_dict(json.loads(line)) for line in text.splitlines() if line]

        return cls(read("parents.jsonl"), read("children.jsonl"), np.load(d / "vectors.npy"))

    # ------------------------------------------------------------------ helpers
    def mask_for(self, jurisdiction: Jurisdiction) -> np.ndarray:
        if jurisdiction == Jurisdiction.BOTH:
            return np.ones(len(self.children), dtype=bool)
        return self._jur == jurisdiction.value

    def parent_of(self, child: Chunk) -> Chunk:
        return self.parents[child.meta["parent_id"]]

    def counterpart(self, parent: Chunk, lang: str) -> Chunk | None:
        return self.pairs.get(parent.pair_id, {}).get(lang)
