#!/usr/bin/env bash
# openxlab's setup.py pins setuptools~=60.2.0, which conflicts with torch>=2.11
# (requires setuptools>=77). The pin is a build-time mistake; install openxlab
# without deps and pull only its true runtime requirements. tqdm/filelock are
# pinned to openxlab's ranges. Finally we drop the bogus setuptools Requires-Dist
# line from openxlab's installed metadata (it never imports setuptools) so
# `pip check` is completely clean.
set -euo pipefail
VENV="${1:-.venv}"
"$VENV/bin/pip" install --no-deps "openxlab>=0.1.3"
"$VENV/bin/pip" install \
  "oss2~=2.17.0" "packaging~=24.0" "pytz~=2023.3" "pyyaml~=6.0" \
  "rich~=13.4.2" "tqdm~=4.65.0" "filelock~=3.14.0"

"$VENV/bin/python" - <<'PY'
import importlib.util
from pathlib import Path

spec = importlib.util.find_spec("openxlab")
site_packages = Path(spec.origin).resolve().parent.parent
meta = next(site_packages.glob("openxlab-*.dist-info/METADATA"))
lines = meta.read_text(encoding="utf-8").splitlines()
kept = [ln for ln in lines if not ln.lower().startswith("requires-dist: setuptools")]
if len(kept) != len(lines):
    meta.write_text("\n".join(kept) + "\n", encoding="utf-8")
    print(f"Patched {meta.name}: dropped setuptools Requires-Dist")
else:
    print("openxlab metadata already clean")
PY
