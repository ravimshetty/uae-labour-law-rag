"""Download the official source PDFs listed in data/sources.yaml into data/raw/.

Some government portals block scripted downloads or change URLs; if a download
fails, open the URL in a browser and save the PDF to the `file` path shown.
"""
import sys
from pathlib import Path

import requests
import yaml

ROOT = Path(__file__).resolve().parents[1]

def main() -> None:
    docs = yaml.safe_load((ROOT / "data/sources.yaml").read_text("utf-8"))["documents"]
    for d in docs:
        dest = ROOT / d["file"]
        if dest.exists():
            print(f"✓ {dest.name} (exists)")
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            r = requests.get(d["url"], timeout=60, headers={"User-Agent": "Mozilla/5.0"})
            r.raise_for_status()
            if b"%PDF" not in r.content[:1024]:
                raise ValueError("response is not a PDF (portal page?)")
            dest.write_bytes(r.content)
            print(f"↓ {dest.name} ({len(r.content)//1024} KB)")
        except Exception as e:  # noqa: BLE001
            print(f"✗ {d['doc_id']}: {e}\n   -> save manually from {d['url']} to {dest}", file=sys.stderr)

if __name__ == "__main__":
    main()
