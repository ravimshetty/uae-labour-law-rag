"""Knowledge base: parents + children + BM25 + dense vectors, persisted to disk.

NumPy is an *optional accelerator*. Without it (e.g. Windows Smart App Control
blocks its DLL) vectors are plain Python lists and search is a pure-Python dot
product – a few hundred ms for the full corpus, fine for local use.

Vectors are stored as raw float32 (`vectors.f32`, stdlib `array`) + `meta.json`,
readable with or without NumPy. For multi-tenant production swap `VectorIndex`
for Qdrant / pgvector / Azure AI Search; `search()` is the only contract.
"""
from __future__ import annotations

import heapq
import json
from array import array
from pathlib import Path

from ..models import Chunk, Jurisdiction
from .bm25 import BM25
from .embeddings import Embedder

try:  # optional speed-up
    import numpy as _np
except ImportError:  # includes "DLL load failed ... Application Control policy"
    _np = None

Vector = list[float]


class VectorIndex:
    def __init__(self, vectors: list[Vector]):
        self.vectors = vectors
        self._mat = _np.asarray(vectors, dtype=_np.float32) if (_np is not None and vectors) else None

    def __len__(self) -> int:
        return len(self.vectors)

    def search(self, qvec: Vector, mask: list[bool], k: int) -> list[tuple[int, float]]:
        if not self.vectors:
            return []
        if self._mat is not None:
            sims = self._mat @ _np.asarray(qvec, dtype=_np.float32)
            cand = [(float(sims[i]), i) for i in range(len(mask)) if mask[i]]
        else:
            cand = [(sum(a * b for a, b in zip(v, qvec)), i)
                    for i, v in enumerate(self.vectors) if mask[i]]
        return [(i, s) for s, i in heapq.nlargest(k, cand)]


class KnowledgeBase:
    def __init__(self, parents: list[Chunk], children: list[Chunk], vectors: list[Vector]):
        self.parents = {p.chunk_id: p for p in parents}
        self.children = children
        self.vectors = VectorIndex(vectors)
        self.bm25 = BM25([c.text for c in children])
        self._jur = [c.jurisdiction.value for c in children]
        self.pairs: dict[str, dict[str, Chunk]] = {}
        for p in parents:
            self.pairs.setdefault(p.pair_id, {})[p.lang.value] = p

    # ------------------------------------------------------------------ build / io
    @classmethod
    def build(cls, parents: list[Chunk], children: list[Chunk], embedder: Embedder) -> "KnowledgeBase":
        return cls(parents, children, embedder.embed([c.text for c in children]))

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        (d / "parents.jsonl").write_text(
            "\n".join(json.dumps(p.to_dict(), ensure_ascii=False) for p in self.parents.values()), "utf-8")
        (d / "children.jsonl").write_text(
            "\n".join(json.dumps(c.to_dict(), ensure_ascii=False) for c in self.children), "utf-8")
        vecs = self.vectors.vectors
        dim = len(vecs[0]) if vecs else 0
        buf = array("f")
        for v in vecs:
            buf.extend(v)
        (d / "vectors.f32").write_bytes(buf.tobytes())
        (d / "meta.json").write_text(json.dumps({"count": len(vecs), "dim": dim}), "utf-8")

    @classmethod
    def load(cls, directory: str | Path) -> "KnowledgeBase":
        d = Path(directory)
        if not (d / "meta.json").exists():
            raise FileNotFoundError(f"No index in {d}. Run: python scripts/ingest.py --sources data/sample/sources.yaml")

        def read(name: str) -> list[Chunk]:
            text = (d / name).read_text("utf-8").strip()
            return [Chunk.from_dict(json.loads(line)) for line in text.splitlines() if line]

        meta = json.loads((d / "meta.json").read_text("utf-8"))
        flat = array("f")
        flat.frombytes((d / "vectors.f32").read_bytes())
        dim = meta["dim"]
        vecs = [flat[i * dim:(i + 1) * dim].tolist() for i in range(meta["count"])]
        return cls(read("parents.jsonl"), read("children.jsonl"), vecs)

    # ------------------------------------------------------------------ helpers
    def mask_for(self, jurisdiction: Jurisdiction) -> list[bool]:
        if jurisdiction == Jurisdiction.BOTH:
            return [True] * len(self.children)
        return [j == jurisdiction.value for j in self._jur]

    def parent_of(self, child: Chunk) -> Chunk:
        return self.parents[child.meta["parent_id"]]

    def counterpart(self, parent: Chunk, lang: str) -> Chunk | None:
        return self.pairs.get(parent.pair_id, {}).get(lang)
