#!/usr/bin/env bash
# Server runner: executes the queued nesegemaa GPU work on GPU index 1 ONLY
# (GPU0 belongs to the hdsemg calibration runs - never touch it).
# Run ON the server:  ./nesegemaa/run_server_gpu1.sh [--go] [--force]
# Without --go it only prints the plan + GPU state (same contract as
# run_when_gpu_free.sh). Refuses when GPU1 is busy unless --force.
set -u
cd ~/home-omni3d || exit 1
GO=0; FORCE=0
for a in "$@"; do
  case "$a" in --go) GO=1;; --force) FORCE=1;; *) echo "unknown arg $a"; exit 2;; esac
done
export CUDA_VISIBLE_DEVICES=1   # the free RTX 3090; GPU0 stays untouched
export TRIPOSR_SRC="$HOME/home-omni3d/thirdparty/TripoSR"
PANEL=$(python3 -c "
import json; s=json.load(open('AtlasNet/dataset/data/splits.json'))
print(' '.join(f'{c}:{o}' for c in s for o in s[c].get('test',[])))")
echo "test objects: $(echo "$PANEL" | wc -w) | visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "--- GPU state ---"
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used --format=csv
nvidia-smi --query-compute-apps=gpu_uuid,pid,process_name --format=csv
echo "-----------------"
if [ "$GO" -ne 1 ]; then
  cat <<'EOF'
Plan (nothing executed - dry run), all on GPU index 1:
  [1] TripoSR full split: default + raw + crop + masked/largest-cleanup
  [2] Point-E default full split (~3.3 GPU-h-equivalent; 3090 is ~2x faster
      than the local 2080 Ti, expect ~1.5-2 h)
  [3] build_final.py refresh; sync triplets/CSVs back to the PC for viewer
      export (viewer export itself runs on the PC, CPU-only).
Re-run with --go to execute.
EOF
  exit 0
fi
BUSY=$(nvidia-smi --query-compute-apps=gpu_index --format=csv,noheader 2>/dev/null | grep -c "^1$" || true)
if [ "$FORCE" -ne 1 ] && [ "$BUSY" -gt 0 ]; then
  echo "REFUSING: GPU index 1 has active processes. Wait or use --force."; exit 3
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
echo "server GPU work complete."
