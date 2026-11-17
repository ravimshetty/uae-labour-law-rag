"""Build the index.  python scripts/ingest.py [--sources data/sample/sources.yaml]"""
import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uae_labour_rag.config import load_settings  # noqa: E402
from uae_labour_rag.ingest.build import build_knowledge_base  # noqa: E402

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default=None, help="sources yaml (default: paths.sources)")
    ap.add_argument("--out", default=None, help="index dir (default: paths.index_dir)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    s = load_settings()
    kb = build_knowledge_base(s, args.sources)
    out = Path(args.out) if args.out else s.path("paths.index_dir")
    kb.save(out)
    print(f"Indexed {len(kb.parents)} articles / {len(kb.children)} chunks -> {out}")
