#!/usr/bin/env bash
# openxlab's setup.py pins setuptools~=60.2.0, which conflicts with torch>=2.11
# (requires setuptools>=77). The pin is a build-time mistake; install openxlab
# without deps and pull only its true runtime requirements. tqdm/filelock are
# pinned to openxlab's ranges to keep `pip check` quiet; only the setuptools
# mismatch remains, which is impossible to satisfy alongside torch and harmless
# (openxlab never imports setuptools at runtime).
set -euo pipefail
VENV="${1:-.venv}"
"$VENV/bin/pip" install --no-deps "openxlab>=0.1.3"
"$VENV/bin/pip" install \
  "oss2~=2.17.0" "packaging~=24.0" "pytz~=2023.3" "pyyaml~=6.0" \
  "rich~=13.4.2" "tqdm~=4.65.0" "filelock~=3.14.0"
