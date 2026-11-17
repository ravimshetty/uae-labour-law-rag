"""Settings loaded from config/settings.yaml, overridable by environment variables.

Env var convention: RAG_<SECTION>__<KEY>, e.g. RAG_LLM__PROVIDER=openai
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "config" / "settings.yaml"


def _coerce(value: str) -> Any:
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


@dataclass
class Settings:
    raw: dict[str, Any] = field(default_factory=dict)

    def get(self, dotted: str, default: Any = None) -> Any:
        node: Any = self.raw
        for part in dotted.split("."):
            if not isinstance(node, dict) or part not in node:
                return default
            node = node[part]
        return node

    def path(self, dotted: str) -> Path:
        p = Path(self.get(dotted))
        return p if p.is_absolute() else ROOT / p


def load_settings(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Settings:
    cfg_path = Path(path or os.getenv("RAG_CONFIG", DEFAULT_CONFIG))
    raw: dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}

    # Environment overrides: RAG_LLM__PROVIDER=openai -> raw["llm"]["provider"]
    for key, value in os.environ.items():
        if not key.startswith("RAG_") or "__" not in key:
            continue
        section, _, sub = key[4:].lower().partition("__")
        raw.setdefault(section, {})[sub] = _coerce(value)

    for dotted, value in (overrides or {}).items():
        node = raw
        *parents, leaf = dotted.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = value
    return Settings(raw)
