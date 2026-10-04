#!/usr/bin/env bash
# Generic AtlasNet sweep phase runner.
# Usage: run_phase_atlas.sh <phase-tag>   (reads spec lines from stdin)
# Spec line format:  dirname|extra train.py args
# Per run: fresh log dir -> train (50ep base + extras, stdout saved) ->
# summarize -> full-split qual -> ablation row. Failures are recorded with
# status=FAILED (+ reason) and the phase CONTINUES. Never touches the dataset.
set -u
ROOT=/home/susan/agentic_workspace/home-omni3d
cd "$ROOT" || exit 1
source .venv/bin/activate
PHASE="$1"
STATUS_FILE="/tmp/opencode/phase_${PHASE}.status"
: > "$STATUS_FILE"
echo "phase=$PHASE started $(date)" >> "$STATUS_FILE"

BASE="--SVR --no_metro --nepoch 50 --batch_size 8 --nb_primitives 25 --template_type SQUARE --workers 2"
CLASSES=$(python -c "import json;print(' '.join(d['synsetId'] for d in json.load(open('AtlasNet/dataset/data/taxonomy.json'))))")

while IFS='|' read -r DIR EXTRA; do
  case "$DIR" in ""|\#*) continue;; esac
  echo "=== [$PHASE] run $DIR :: $EXTRA ==="
  rm -rf "AtlasNet/log/$DIR"
  mkdir -p "AtlasNet/log/$DIR"
  echo "$EXTRA" > "AtlasNet/log/$DIR/extra_args.txt"
  cd AtlasNet || exit 1
  # shellcheck disable=SC2086
  if python train.py --class_choice $CLASSES $BASE --dir_name "log/$DIR" ${EXTRA:-} \
      > "log/$DIR/train_stdout.log" 2>&1; then
    echo "TRAIN_OK $DIR" >> "$STATUS_FILE"
  else
    echo "TRAIN_FAIL $DIR (see AtlasNet/log/$DIR/train_stdout.log)" >> "$STATUS_FILE"
    cd "$ROOT" || exit 1
    python nesegemaa/append_ablation_row.py --run "$DIR" --status FAILED \
      --notes "training exited non-zero (possibly OOM; see train_stdout.log)" 2>/dev/null || true
    continue
  fi
  cd "$ROOT" || exit 1
  if python nesegemaa/summarize_run.py --run "$DIR" \
      && python nesegemaa/atlas_qual.py --run "$DIR" --weights best >/dev/null 2>&1 \
      && python nesegemaa/append_ablation_row.py --run "$DIR" --status OK; then
    echo "RECORDED $DIR" >> "$STATUS_FILE"
  else
    echo "RECORD_FAIL $DIR" >> "$STATUS_FILE"
    python nesegemaa/append_ablation_row.py --run "$DIR" --status FAILED \
      --notes "summarize/qual step failed" 2>/dev/null || true
  fi
done

echo "phase=$PHASE finished $(date)" >> "$STATUS_FILE"
cat "$STATUS_FILE"
