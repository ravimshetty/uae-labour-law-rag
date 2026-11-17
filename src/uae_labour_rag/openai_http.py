"""Minimal OpenAI / Azure OpenAI REST client using only the Python standard library.

Why not the `openai` SDK? It pulls in pydantic-core (a compiled Rust extension).
On Windows machines with Smart App Control / WDAC, unsigned compiled extensions
are blocked ("An Application Control policy has blocked this file"). Plain HTTPS
via urllib works everywhere Python works.

Env vars:
  OpenAI:  OPENAI_API_KEY, optional OPENAI_BASE_URL (default https://api.openai.com/v1)
  Azure:   AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY, AZURE_OPENAI_API_VERSION
"""
from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any


class OpenAIHTTPError(RuntimeError):
    pass


class OpenAIHTTP:
    def __init__(self, azure: bool = False, timeout_s: int = 60, retries: int = 3):
        self.azure, self.timeout, self.retries = azure, timeout_s, retries
        if azure:
            self.endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
            self.key = os.environ.get("AZURE_OPENAI_API_KEY", "")
            self.api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2024-10-21")
            if not (self.endpoint and self.key):
                raise OpenAIHTTPError("Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY")
        else:
            self.base = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
            self.key = os.environ.get("OPENAI_API_KEY", "")
            if not self.key:
                raise OpenAIHTTPError("Set OPENAI_API_KEY (or use RAG_LLM__PROVIDER=mock)")

    def _url(self, op: str, model: str) -> str:
        if self.azure:  # `model` is the Azure deployment name
            return f"{self.endpoint}/openai/deployments/{model}/{op}?api-version={self.api_version}"
        return f"{self.base}/{op}"

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.azure:
            h["api-key"] = self.key
        else:
            h["Authorization"] = f"Bearer {self.key}"
        return h

    def post(self, op: str, model: str, payload: dict[str, Any]) -> dict[str, Any]:
        if not self.azure:
            payload = {"model": model, **payload}
        data = json.dumps(payload).encode("utf-8")
        last: Exception | None = None
        for attempt in range(self.retries):
            req = urllib.request.Request(self._url(op, model), data=data, headers=self._headers(), method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                body = e.read().decode("utf-8", "ignore")[:500]
                last = OpenAIHTTPError(f"HTTP {e.code}: {body}")
                if e.code not in (408, 429, 500, 502, 503, 504):
                    raise last from e
            except urllib.error.URLError as e:
                last = OpenAIHTTPError(f"network error: {e.reason}")
            time.sleep(2 ** attempt)
        raise last or OpenAIHTTPError("request failed")

    def chat(self, model: str, messages: list[dict], temperature: float = 0.0, max_tokens: int = 900) -> str:
        r = self.post("chat/completions", model,
                      {"messages": messages, "temperature": temperature, "max_tokens": max_tokens})
        return r["choices"][0]["message"].get("content") or ""

    def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        r = self.post("embeddings", model, {"input": texts})
        return [d["embedding"] for d in sorted(r["data"], key=lambda d: d["index"])]
