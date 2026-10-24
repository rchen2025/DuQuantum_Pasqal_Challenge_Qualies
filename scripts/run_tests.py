"""Minimal pytest stand-in for environments without pytest (no network here).

Runs every ``test_*`` function in ``tests/test_*.py`` (optionally only files
matching the argv patterns).  Supports ``pytest.raises``-free tests only;
real pytest is used on the Windows venv.  Exit code 1 on any failure.
"""
from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(patterns: list[str]) -> int:
    files = sorted((ROOT / "tests").glob("test_*.py"))
    if patterns:
        files = [f for f in files if any(p in f.name for p in patterns)]
    passed = failed = skipped = 0
    for f in files:
        spec = importlib.util.spec_from_file_location(f.stem, f)
        mod = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(mod)  # type: ignore[union-attr]
        except Exception:
            print(f"IMPORT ERROR {f.name}")
            traceback.print_exc()
            failed += 1
            continue
        for name in sorted(dir(mod)):
            if not name.startswith("test_"):
                continue
            fn = getattr(mod, name)
            if not callable(fn):
                continue
            if fn.__code__.co_argcount:
                skipped += 1  # needs fixtures -> pytest only
                continue
            try:
                fn()
                passed += 1
                print(f"PASS {f.name}::{name}")
            except Exception as e:  # noqa: BLE001
                if type(e).__name__ == "Skipped" or "skip" in type(e).__name__.lower():
                    skipped += 1
                    print(f"SKIP {f.name}::{name}: {e}")
                    continue
                failed += 1
                print(f"FAIL {f.name}::{name}")
                traceback.print_exc()
    print(f"\n{passed} passed, {failed} failed, {skipped} skipped")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
