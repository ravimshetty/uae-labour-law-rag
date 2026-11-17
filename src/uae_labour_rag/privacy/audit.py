"""Privacy-preserving audit trail (JSON Lines).

What is stored: request id, timestamp, SHA-256 of the raw query (for dedup /
abuse investigation without keeping content), the *redacted* query, routing
decision, retrieved article ids, guardrail flags, latency, model name.

What is never stored: raw query text, PII values, the PII vault, the answer
re-hydrated with PII. Retention is enforced by `purge_older_than()` (run daily
via cron / a scheduled job) to satisfy data-minimisation under UAE PDPL
(Federal Decree-Law 45/2021) and DIFC Data Protection Law No. 5 of 2020.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path
from typing import Any

_LOCK = threading.Lock()


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class AuditLogger:
    def __init__(self, path: str | Path, store_raw: bool = False):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.store_raw = store_raw

    def log(self, record: dict[str, Any]) -> None:
        record = {"ts": time.time(), **record}
        if not self.store_raw:
            record.pop("raw_query", None)
        line = json.dumps(record, ensure_ascii=False)
        with _LOCK, self.path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def purge_older_than(self, days: int) -> int:
        if not self.path.exists():
            return 0
        cutoff = time.time() - days * 86400
        keep, dropped = [], 0
        with _LOCK:
            for line in self.path.read_text("utf-8").splitlines():
                try:
                    if json.loads(line).get("ts", 0) >= cutoff:
                        keep.append(line)
                    else:
                        dropped += 1
                except json.JSONDecodeError:
                    dropped += 1
            self.path.write_text("\n".join(keep) + ("\n" if keep else ""), "utf-8")
        return dropped
