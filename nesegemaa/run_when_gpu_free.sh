#!/usr/bin/env bash
# Queued GPU work for the nesegemaa track. DO NOT RUN while others' experiments
# are active (e.g. PID 62741 hdsemg calibration on cuda:0 as of 2026-10-14).
# The script refuses to start when another GPU compute process exists, unless
# --force is given. CPU-only viewer exports are NOT queued here (already done).
#
# Contents:
#  1. TripoSR full 260-object test split x {masked,raw,crop} + masked+largest
#     mesh cleanup  (~0.5 GPU-h total, deterministic, seed 0)
#  2. Point-E default config on the full 260-object split (~3.3 GPU-h)
#  3. Rebuild ablation_triposr.csv (+ pointe row) via build_final.py
#
# Usage:
#   ./nesegemaa/run_when_gpu_free.sh            # dry run: prints plan + GPU state
#   ./nesegemaa/run_when_gpu_free.sh --go       # execute (refuses if GPU busy)
#   ./nesegemaa/run_when_gpu_free.sh --go --force  # execute even if GPU busy
set -u
ROOT=/home/susan/agentic_workspace/home-omni3d
cd "$ROOT" || exit 1
GO=0; FORCE=0
for a in "$@"; do
  case "$a" in --go) GO=1;; --force) FORCE=1;; *) echo "unknown arg $a"; exit 2;; esac
done
PANEL=$(cat /tmp/opencode/test_panel.txt 2>/dev/null || python3 -c "
import json; s=json.load(open('AtlasNet/dataset/data/splits.json'))
print(' '.join(f'{c}:{o}' for c in s for o in s[c].get('test',[])))")
N=$(echo "$PANEL" | wc -w)
echo "test objects: $N"
echo "--- GPU state ---"
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
echo "-----------------"
if [ "$GO" -ne 1 ]; then
  cat <<'EOF'
Plan (nothing executed - dry run):
  [1] TripoSR full split: default + raw + crop + masked/largest-cleanup
      python nesegemaa/triposr/eval_triposr.py --objects <260> [--preprocessing raw|crop] [--mesh-cleanup largest --tag masked_clean]
  [2] Point-E default full split (~3.3 GPU-h, ~46 s/obj):
      python nesegemaa/pointe/faithful_eval.py --objects <260> --tag full
  [3] python nesegemaa/build_final.py   (refreshes ablation_triposr.csv etc.)
  [4] Re-export viewer runs (CPU-only, separate step).
Re-run with --go to execute. AtlasNet unfreeze fine-tune (when scheduled):
  train.py ... --nb_primitives 25 --template_type SPHERE (drop --freeze_encoder;
  a3_frozen proved decoder-only fits; full 38.4M params @batch 8 fits 11 GB).
EOF
  exit 0
fi
if [ "$FORCE" -ne 1 ] && [ "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)" -gt 0 ]; then
  echo "REFUSING: other GPU processes active. Wait or re-run with --force."; exit 3
fi
source .venv/bin/activate
# shellcheck disable=SC2086
python nesegemaa/triposr/eval_triposr.py --objects $PANEL 2>&1 | tail -n 2
# shellcheck disable=SC2086
python nesegemaa/triposr/eval_triposr.py --objects $PANEL --preprocessing raw --tag raw 2>&1 | tail -n 2
# shellcheck disable=SC2086
python nesegemaa/triposr/eval_triposr.py --objects $PANEL --preprocessing crop --tag crop 2>&1 | tail -n 2
# shellcheck disable=SC2086
python nesegemaa/triposr/eval_triposr.py --objects $PANEL --mesh-cleanup largest --tag masked_clean 2>&1 | tail -n 2
# shellcheck disable=SC2086
python nesegemaa/pointe/faithful_eval.py --objects $PANEL --tag full 2>&1 | tail -n 2
python nesegemaa/build_final.py 2>&1 | tail -n 5
echo "queued GPU work complete."
