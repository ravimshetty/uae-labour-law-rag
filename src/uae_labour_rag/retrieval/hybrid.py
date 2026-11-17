"""Hybrid cross-lingual retrieval.

    query_en ──► BM25 ─┐
    query_ar ──► BM25 ─┤
    query_en ──► dense ┼──► Reciprocal Rank Fusion ──► (optional cross-encoder) ──► parents
    query_ar ──► dense ┘

* Four ranked lists (2 languages x 2 retrievers) are fused with RRF – no score
  calibration needed between BM25 and cosine.
* Children are collapsed to their parent article; each article is then *paired*:
  for Federal law we attach the authoritative Arabic text next to the English one,
  so the model writes Arabic from the official Arabic wording instead of
  translating (the #1 source of legal mistranslation).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

from ..index.embeddings import Embedder
from ..index.store import KnowledgeBase
from ..models import Chunk, Jurisdiction, RetrievedChunk


@dataclass
class ArticleHit:
    key: str                      # pair_id, e.g. "fdl-33-2021:art51"
    score: float
    en: Chunk | None
    ar: Chunk | None
    matched_children: list[str]

    @property
    def primary(self) -> Chunk:
        """Authoritative version if available, else whichever exists."""
        for c in (self.ar, self.en):
            if c is not None and c.authoritative:
                return c
        return self.ar or self.en  # type: ignore[return-value]


class HybridRetriever:
    def __init__(self, kb: KnowledgeBase, embedder: Embedder, *, top_k_bm25: int = 20,
                 top_k_dense: int = 20, rrf_k: int = 60, reranker=None):
        self.kb, self.embedder = kb, embedder
        # Hashing embeddings are monolingual -> only compare same-language text.
        # Multilingual embeddings (OpenAI / bge-m3) may match EN query <-> AR article.
        self.cross_lingual_dense = getattr(embedder, "cross_lingual", True)
        self._lang = [c.lang.value for c in kb.children]
        self.top_k_bm25, self.top_k_dense, self.rrf_k = top_k_bm25, top_k_dense, rrf_k
        self.reranker = reranker

    def _bm25(self, query: str, mask) -> list[int]:
        if not query.strip():
            return []
        scores = self.kb.bm25.scores(query)
        ranked = sorted((i for i in range(len(scores)) if mask[i] and scores[i] > 0),
                        key=lambda i: -scores[i])
        return ranked[: self.top_k_bm25]

    def _dense(self, query: str, mask) -> list[int]:
        if not query.strip():
            return []
        qv = self.embedder.embed([query])[0]
        return [i for i, _ in self.kb.vectors.search(qv, mask, self.top_k_dense)]

    def retrieve_children(self, query_en: str, query_ar: str, jurisdiction: Jurisdiction) -> list[RetrievedChunk]:
        mask = self.kb.mask_for(jurisdiction)
        en = [m and lang == "en" for m, lang in zip(mask, self._lang)]
        ar = [m and lang == "ar" for m, lang in zip(mask, self._lang)]
        lists = {
            # lexical: each query only against its own language
            "bm25_en": self._bm25(query_en, en),
            "bm25_ar": self._bm25(query_ar, ar),
            "dense_en": self._dense(query_en, mask if self.cross_lingual_dense else en),
            "dense_ar": self._dense(query_ar, mask if self.cross_lingual_dense else ar),
        }
        fused: dict[int, float] = defaultdict(float)
        per: dict[int, dict[str, float]] = defaultdict(dict)
        for name, ranked in lists.items():
            for rank, idx in enumerate(ranked):
                fused[idx] += 1.0 / (self.rrf_k + rank + 1)
                per[idx][name] = rank + 1
        out = [RetrievedChunk(self.kb.children[i], s, per[i]) for i, s in fused.items()]
        out.sort(key=lambda r: -r.score)
        if self.reranker and out:
            out = self.reranker.rerank(query_en or query_ar, out)
        return out

    def retrieve(self, query_en: str, query_ar: str, jurisdiction: Jurisdiction, k: int = 6) -> list[ArticleHit]:
        children = self.retrieve_children(query_en, query_ar, jurisdiction)
        by_article: dict[str, ArticleHit] = {}
        for rc in children:
            parent = self.kb.parent_of(rc.chunk)
            key = parent.pair_id
            hit = by_article.get(key)
            if hit is None:
                hit = ArticleHit(key, 0.0, self.kb.counterpart(parent, "en"),
                                 self.kb.counterpart(parent, "ar"), [])
                by_article[key] = hit
            hit.score += rc.score  # evidence from EN and AR versions accumulates
            hit.matched_children.append(rc.chunk.chunk_id)
        hits = sorted(by_article.values(), key=lambda h: -h.score)
        if jurisdiction == Jurisdiction.BOTH:
            hits = _balance(hits, k)
        return hits[:k]


def _balance(hits: list[ArticleHit], k: int) -> list[ArticleHit]:
    """When comparing regimes make sure both are represented in the context."""
    fed = [h for h in hits if h.primary.jurisdiction == Jurisdiction.FEDERAL]
    difc = [h for h in hits if h.primary.jurisdiction == Jurisdiction.DIFC]
    if not fed or not difc:
        return hits
    out, i = [], 0
    while len(out) < k and (i < len(fed) or i < len(difc)):
        for lst in (fed, difc):
            if i < len(lst) and len(out) < k:
                out.append(lst[i])
        i += 1
    return out


class CrossEncoderReranker:
    """Optional multilingual reranker (BAAI/bge-reranker-v2-m3 handles Arabic)."""

    def __init__(self, model: str, top_n: int = 30):
        from sentence_transformers import CrossEncoder  # optional dependency

        self.model = CrossEncoder(model)
        self.top_n = top_n

    def rerank(self, query: str, items: list[RetrievedChunk]) -> list[RetrievedChunk]:
        head, tail = items[: self.top_n], items[self.top_n:]
        scores = self.model.predict([(query, r.chunk.text) for r in head])
        for r, s in zip(head, scores):
            r.sources["rerank"] = float(s)
            r.score = float(s)
        head.sort(key=lambda r: -r.score)
        return head + tail
