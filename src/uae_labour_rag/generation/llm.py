"""LLM providers: OpenAI, Azure OpenAI (UAE North for data residency) and an
offline deterministic `MockLLM` used for tests / demos without API keys."""
from __future__ import annotations

import os
import re
from typing import Protocol


class LLM(Protocol):
    name: str

    def complete(self, system: str, user: str) -> str: ...


class OpenAIChat:
    def __init__(self, model: str, temperature: float = 0.0, max_tokens: int = 900,
                 azure: bool = False, timeout_s: int = 60):
        from openai import AzureOpenAI, OpenAI

        self.client = (
            AzureOpenAI(api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-10-21"), timeout=timeout_s)
            if azure else OpenAI(timeout=timeout_s)
        )
        self.model, self.temperature, self.max_tokens = model, temperature, max_tokens
        self.name = f"{'azure' if azure else 'openai'}:{model}"

    def complete(self, system: str, user: str) -> str:
        resp = self.client.chat.completions.create(
            model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        )
        return resp.choices[0].message.content or ""


class MockLLM:
    """Extractive, deterministic 'LLM'. Builds an Arabic answer from the Arabic text of
    the top sources (or a templated sentence when only English exists). It exists so
    the whole pipeline – guardrails, citations, UI – runs offline and in CI."""

    name = "mock:extractive"

    def complete(self, system: str, user: str) -> str:
        jur_m = re.search(r"JURISDICTION: (\w+)", user)
        src = user.split("SOURCES:\n", 1)[-1].split("\n\nQUESTION", 1)[0]
        blocks = []
        for block in re.split(r"\n\n(?=\[S\d+\])", src):
            m = re.match(r"\[(S\d+)\] (.+?) \| Article (\S+) – (.*?) \| jurisdiction=(\w+)[^\n]*\n?(.*)", block, re.S)
            if m:
                blocks.append(m.groups())
        if not blocks or not jur_m:
            return "NO_ANSWER"
        jur = jur_m.group(1)
        lines, gist = [], []
        for ref, law, art, title, j, body in blocks[:2]:
            ar = re.search(r"AR: (.*?)(?:\nEN: |$)", body, flags=re.S)
            if ar:
                clauses = [re.sub(r"^\s*\d+[.\-)]\s*", "", c) for c in ar.group(1).strip().splitlines()]
                first = " ".join(c for c in clauses if c)[:500]
                sentence = f"وفقاً للمادة ({art}) {('من القانون الاتحادي' if j == 'federal' else 'من قانون مركز دبي المالي العالمي')}: {first} [{ref}]"
            else:
                sentence = (f"تتناول المادة ({art}) من قانون التوظيف في مركز دبي المالي العالمي "
                            f"هذا الموضوع، ويُرجى الرجوع إلى نصها الرسمي باللغة الإنجليزية (ترجمة غير رسمية غير متاحة في الوضع التجريبي) [{ref}]")
            if jur == "both":
                sentence = ("أولاً: القانون الاتحادي (البر الرئيسي)\n" if j == "federal"
                            else "ثانياً: مركز دبي المالي العالمي\n") + sentence
            lines.append(sentence)
            gist.append(f"Article {art} ({title.strip()})")
        return "\n\n".join(lines) + f"\nGIST_EN: Based on {', '.join(gist)}."


def build_llm(provider: str, model: str, **kw) -> LLM:
    if provider == "mock":
        return MockLLM()
    if provider in {"openai", "azure_openai"}:
        return OpenAIChat(model, azure=provider == "azure_openai", **kw)
    raise ValueError(f"unknown llm provider: {provider}")
