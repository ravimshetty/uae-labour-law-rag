"""Offline evaluation: retrieval recall@k / MRR, routing accuracy, guardrail accuracy,
Arabic-ness and citation coverage of answers.

    python scripts/eval.py [--k 6] [--out evals/report.json]
"""
import argparse
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uae_labour_rag.pipeline import get_pipeline  # noqa: E402
from uae_labour_rag.retrieval.router import route  # noqa: E402
from uae_labour_rag.text.arabic import arabic_ratio  # noqa: E402
from uae_labour_rag.text.codeswitch import normalize_query  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=6)
    ap.add_argument("--out", default=str(ROOT / "evals/report.json"))
    a = ap.parse_args()
    p = get_pipeline()
    cases = yaml.safe_load((ROOT / "evals/golden.yaml").read_text("utf-8"))["cases"]
    by_type = defaultdict(list)
    rows = []
    for c in cases:
        res = p.answer(c["q"])
        row = {"q": c["q"], "type": c["type"], "blocked": res.blocked}
        if c["type"] == "guardrail":
            row["pass"] = res.blocked == c["blocked"]
        else:
            r = route(c["q"])
            nq = normalize_query(c["q"], p.glossary)
            hits = [h.key for h in p.retriever.retrieve(nq.query_en, nq.query_ar, r.jurisdiction, k=a.k)]
            found = [e for e in c["expected"] if e in hits]
            row.update({
                "route_ok": r.jurisdiction.value == c["jur"],
                "recall": len(found) / len(c["expected"]),
                "rr": max((1 / (hits.index(e) + 1) for e in found), default=0.0),
                "arabic_ratio": round(arabic_ratio(res.answer_ar), 3),
                "cited": bool(res.citations),
                "pass": len(found) == len(c["expected"]) and not res.blocked,
            })
        rows.append(row)
        by_type[c["type"]].append(row)

    qa = [r for r in rows if r["type"] != "guardrail"]
    summary = {
        f"recall@{a.k}": round(statistics.mean(r["recall"] for r in qa), 3),
        "mrr": round(statistics.mean(r["rr"] for r in qa), 3),
        "routing_accuracy": round(statistics.mean(r["route_ok"] for r in qa), 3),
        "citation_rate": round(statistics.mean(r["cited"] for r in qa), 3),
        "mean_arabic_ratio": round(statistics.mean(r["arabic_ratio"] for r in qa), 3),
        "guardrail_accuracy": round(statistics.mean(r["pass"] for r in by_type["guardrail"]), 3),
        "pass_rate_by_type": {t: round(statistics.mean(r["pass"] for r in rs), 3) for t, rs in by_type.items()},
    }
    Path(a.out).write_text(json.dumps({"summary": summary, "cases": rows}, ensure_ascii=False, indent=2), "utf-8")
    print(json.dumps(summary, indent=2))
    for r in rows:
        if not r["pass"]:
            print("FAIL:", r["q"])


if __name__ == "__main__":
    main()
