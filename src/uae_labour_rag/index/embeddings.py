"""Embedding providers.

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
import os
from typing import Protocol

import numpy as np

from ..text.arabic import normalize_for_search


class Embedder(Protocol):
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashingEmbedder:
    cross_lingual = False

    def __init__(self, dim: int = 1024, ngram: tuple[int, int] = (3, 5)):
        self.dim = dim
        self.ngram = ngram

    def _vec(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        t = f" {normalize_for_search(text)} "
        for n in range(self.ngram[0], self.ngram[1] + 1):
            for i in range(len(t) - n + 1):
                h = hashlib.blake2b(t[i:i + n].encode(), digest_size=8).digest()
                idx = int.from_bytes(h[:4], "little") % self.dim
                sign = 1.0 if h[4] & 1 else -1.0
                v[idx] += sign
        norm = np.linalg.norm(v)
        return v / norm if norm else v

    def embed(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._vec(t) for t in texts]) if texts else np.zeros((0, self.dim), np.float32)


class OpenAIEmbedder:
    cross_lingual = True

    def __init__(self, model: str = "text-embedding-3-large", azure: bool = False, batch: int = 64):
        from openai import AzureOpenAI, OpenAI  # lazy import

        self.client = (
            AzureOpenAI(api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"))
            if azure else OpenAI()
        )
        self.model = model
        self.batch = batch
        self.dim = 3072 if model.endswith("large") else 1536

    def embed(self, texts: list[str]) -> np.ndarray:
        out = []
        for i in range(0, len(texts), self.batch):
            resp = self.client.embeddings.create(model=self.model, input=texts[i:i + self.batch])
            out.extend(d.embedding for d in resp.data)
        arr = np.asarray(out, dtype=np.float32)
        return arr / np.linalg.norm(arr, axis=1, keepdims=True).clip(min=1e-9)


def build_embedder(provider: str, model: str, dim: int = 1024) -> Embedder:
    if provider == "hashing":
        return HashingEmbedder(dim)
    if provider in {"openai", "azure_openai"}:
        return OpenAIEmbedder(model, azure=provider == "azure_openai")
    raise ValueError(f"unknown embeddings provider: {provider}")
