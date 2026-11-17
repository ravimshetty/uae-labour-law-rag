"""Simulate Windows Smart App Control / WDAC: block every compiled third-party
extension (numpy, pydantic-core, libyaml, pypdfium2, Pillow, ...) and check that the
core pipeline still runs.  Used in CI so the pure-Python path never regresses.

    python scripts/simulate_app_control.py
"""
import importlib.abc
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCKED = ("numpy", "pydantic", "pydantic_core", "fastapi", "openai", "yaml._yaml", "_yaml",
           "pdfplumber", "pypdfium2", "PIL", "charset_normalizer.md", "sentence_transformers", "torch")


class Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if any(name == b or name.startswith(b + ".") for b in BLOCKED):
            raise ImportError(f"DLL load failed while importing {name}: "
                              "An Application Control policy has blocked this file. (simulated)")
        return None


sys.meta_path.insert(0, Blocker())

if __name__ == "__main__":
    import pytest

    sys.argv = ["ingest", "--sources", str(ROOT / "data/sample/sources.yaml")]
    runpy.run_path(str(ROOT / "scripts/ingest.py"), run_name="__main__")
    code = pytest.main(["-q", str(ROOT / "tests")])
    sys.argv = ["smoke"]
    try:
        runpy.run_path(str(ROOT / "scripts/smoke_test.py"), run_name="__main__")
    except SystemExit as e:
        code = code or e.code
    leaked = [m for m in sys.modules if any(m == b or m.startswith(b + ".") for b in BLOCKED)]
    assert not leaked, f"blocked modules were imported: {leaked}"
    print("\nOK: pipeline runs with all compiled extensions blocked" if not code else "\nFAILED")
    sys.exit(code)
