"""Embedding providers (pure Python – NumPy is optional, see index/store.py).

* `HashingEmbedder` – offline, deterministic, zero-dependency (char n-gram feature
  hashing on search-normalised text). Good for tests, CI and air-gapped demos.
  It is NOT cross-lingual – cross-lingual recall in offline mode comes from the
  glossary-based query expansion (EN query + AR query).
* `OpenAIEmbedder` – `text-embedding-3-large` (multilingual, strong on Arabic).
  Works with OpenAI or Azure OpenAI (UAE North region for data residency).

Swap-in alternatives: BAAI/bge-m3 or intfloat/multilingual-e5-large via
sentence-transformers (self-hosted, no data leaves your VPC).
"""
from __future__ import annotations

import hashlib
import math
from typing import Protocol

from ..text.arabic import normalize_for_search

Vector = list[float]


def l2_normalize(v: Vector) -> Vector:
    n = math.sqrt(sum(x * x for x in v))
    return [x / n for x in v] if n else v


class Embedder(Protocol):
    dim: int
    cross_lingual: bool

    def embed(self, texts: list[str]) -> list[Vector]: ...


class HashingEmbedder:
    cross_lingual = False

    def __init__(self, dim: int = 1024, ngram: tuple[int, int] = (3, 5)):
        self.dim = dim
        self.ngram = ngram

    def _vec(self, text: str) -> Vector:
        v = [0.0] * self.dim
        t = f" {normalize_for_search(text)} "
        for n in range(self.ngram[0], self.ngram[1] + 1):
            for i in range(len(t) - n + 1):
                h = hashlib.blake2b(t[i:i + n].encode(), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                v[idx] += 1.0 if h[4] & 1 else -1.0
        return l2_normalize(v)

    def embed(self, texts: list[str]) -> list[Vector]:
        return [self._vec(t) for t in texts]


class OpenAIEmbedder:
    cross_lingual = True

    def __init__(self, model: str = "text-embedding-3-large", azure: bool = False, batch: int = 64):
        from ..openai_http import OpenAIHTTP

        self.client = OpenAIHTTP(azure=azure)
        self.model = model
        self.batch = batch
        self.dim = 3072 if model.endswith("large") else 1536

    def embed(self, texts: list[str]) -> list[Vector]:
        out: list[Vector] = []
        for i in range(0, len(texts), self.batch):
            out.extend(l2_normalize(v) for v in self.client.embed(self.model, texts[i:i + self.batch]))
        return out


def build_embedder(provider: str, model: str, dim: int = 1024) -> Embedder:
    if provider == "hashing":
        return HashingEmbedder(dim)
    if provider in {"openai", "azure_openai"}:
        return OpenAIEmbedder(model, azure=provider == "azure_openai")
    raise ValueError(f"unknown embeddings provider: {provider}")
