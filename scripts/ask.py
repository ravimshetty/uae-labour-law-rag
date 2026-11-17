"""CLI:  python scripts/ask.py "how much gratuity after 7 years?" [--jurisdiction difc] [--debug]"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uae_labour_rag.pipeline import get_pipeline  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("question")
    ap.add_argument("--jurisdiction", choices=["federal", "difc", "both"])
    ap.add_argument("--debug", action="store_true")
    a = ap.parse_args()
    res = get_pipeline().answer(a.question, a.jurisdiction, debug=a.debug)
    print(json.dumps(res.to_dict(), ensure_ascii=False, indent=2))
