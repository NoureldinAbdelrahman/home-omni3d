#!/usr/bin/env bash
# Autonomous experiment queue for the nesegemaa branch (AtlasNet + Point-E).
#
# Runs every remaining phase with zero supervision, committing + pushing after
# each phase. Idempotent: completed runs (existing summary.json) are skipped,
# so re-running this script resumes where it stopped.
#
# Order: P0 panel (if missing) -> A1 -> A2 -> A3 -> A4 -> winner seeds x2 ->
#        P1/P2/P3 -> triposr -> refresh default panels -> final build.
# Estimated total GPU: ~15h. GPU for one job at a time (this box is shared).
#
# Failure policy: per-run failures are RECORDED (status=FAILED) and the queue
# continues. The queue aborts only on infrastructure failure. Progress and
# failures land in /tmp/opencode/queue.log and nesegemaa/ablation_*.csv.
set -u
ROOT=/home/susan/agentic_workspace/home-omni3d
cd "$ROOT" || exit 1
source .venv/bin/activate
QLOG=/tmp/opencode/queue.log
mkdir -p /tmp/opencode
log() { echo "[$(date '+%F %T')] $*" | tee -a "$QLOG"; }
GIT="git -c user.name=lab-agent -c user.email=lab-agent@local"
PUSH="git push git@github.com:NoureldinAbdelrahman/home-omni3d.git nesegemaa:nesegemaa"

commit_phase() { # $1 = message
  git add nesegemaa/ results/atlasnet/ results/qualitative/atlasnet/ 2>/dev/null
  $GIT commit -m "$1" 2>&1 | tail -1 >> "$QLOG" || true
  $PUSH >> "$QLOG" 2>&1 || log "PUSH FAILED (will retry at end)"
}

atlas_phase() { # $1 = phase tag; spec lines on stdin
  local phase="$1"
  log "phase $phase: start"
  ./nesegemaa/run_phase_atlas.sh "$phase"
  commit_phase "Phase $phase AtlasNet sweep results"
  log "phase $phase: done"
}

# ---- P0: Point-E default panel (skip if present; a previous launch may finish it)
if [ ! -f nesegemaa/pointe/summary.json ]; then
  log "P0: running default panel"
  python nesegemaa/pointe/faithful_eval.py >> /tmp/opencode/queue.log 2>&1 \
    || log "P0 FAILED (see log)"
else
  log "P0: summary.json present, skipped"
fi

# ---- A1: multi-view encoder ----
atlas_phase a1 <<'SPEC'
a1_k3_max|--n_views 3 --views_pool max
a1_k3_attn|--n_views 3 --views_pool attn
a1_k5_max|--n_views 5 --views_pool max
a1_k5_attn|--n_views 5 --views_pool attn
a1_k8_max|--n_views 8 --views_pool max
a1_k8_attn|--n_views 8 --views_pool attn
SPEC

# ---- A2: decoder capacity/topology ----
atlas_phase a2 <<'SPEC'
a2_p1_sq|--nb_primitives 1
a2_p10_sq|--nb_primitives 10
a2_p25_sphere|--template_type SPHERE
a2_p25_4096|--number_points 4096 --number_points_eval 4096
a2_p25_bn512|--bottleneck_size 512
SPEC

# ---- A3: training ----
atlas_phase a3 <<'SPEC'
a3_lr3e4|--lrate 0.0003
a3_bs16|--batch_size 16
a3_bs32|--batch_size 32
a3_aug_rot|--random_rotation
a3_aug_flip|--data_augmentation_random_flips
a3_aug_aniso|--anisotropic_scaling
a3_frozen|--freeze_encoder
SPEC

# ---- A4: loss ----
atlas_phase a4 <<'SPEC'
a4_edge1e3|--loss_reg edge --loss_reg_w 0.001
a4_edge1e2|--loss_reg edge --loss_reg_w 0.01
SPEC

# ---- winner seeds x2 (winner = min best_chamfer among OK rows) ----
WINNER=$(python - <<'PY'
import csv
best, brow = float("inf"), None
try:
    with open("nesegemaa/ablation_atlasnet.csv") as f:
        for r in csv.DictReader(f):
            try:
                if r.get("status") == "OK" and float(r["best_chamfer"]) < best:
                    best, brow = float(r["best_chamfer"]), r["run"]
            except (ValueError, TypeError):
                pass
except FileNotFoundError:
    pass
print(brow or "")
PY
)
if [ -n "$WINNER" ]; then
  log "winner seeds: $WINNER"
  EXTRA=$(cat "AtlasNet/log/$WINNER/extra_args.txt" 2>/dev/null || echo "")
  printf '%s\n' "${WINNER}_s7|${EXTRA} --seed 7" "${WINNER}_s123|${EXTRA} --seed 123" \
    | ./nesegemaa/run_phase_atlas.sh seeds
  commit_phase "Winner seeds x2 ($WINNER)"
else
  log "winner seeds: SKIPPED (no OK rows)"
fi

# ---- P1/P2/P3: Point-E sweeps (panel objects) ----
pointe_eval() { # tag + extra args
  local tag="$1"; shift
  if [ -f "nesegemaa/pointe/summary${tag:+_$tag}.json" ]; then
    log "pointe $tag: present, skipped"; return 0
  fi
  log "pointe $tag: start"
  if [ -n "$tag" ]; then
    python nesegemaa/pointe/faithful_eval.py --tag "$tag" "$@" >> /tmp/opencode/queue.log 2>&1 \
      || log "pointe $tag FAILED"
  else
    python nesegemaa/pointe/faithful_eval.py "$@" >> /tmp/opencode/queue.log 2>&1 \
      || log "pointe default FAILED"
  fi
}
pointe_eval p1_best --views "0,8,16" --fusion best
pointe_eval p1_median --views "0,8,16" --fusion median
pointe_eval p1_v12 --view 12
pointe_eval p2_g1 --guidance 1.0
pointe_eval p2_g5 --guidance 5.0
pointe_eval p2_1k --points 1024
WG=$(python - <<'PY'
import json, glob
best, wg = float("inf"), 3.0
for p in glob.glob("nesegemaa/pointe/summary*.json"):
    try:
        s = json.load(open(p))
        g = s.get("config", {}).get("guidance")
        m = s.get("micro_chamfer")
        if g in (1.0, 3.0, 5.0) and m is not None and m < best:
            best, wg = m, g
    except Exception:
        pass
print(wg)
PY
)
pointe_eval "p2_g${WG}_s012_1k" --guidance "$WG" --seeds "0,1,2" --points 1024
pointe_eval "p2_g${WG}_s012_4k" --guidance "$WG" --seeds "0,1,2" --points 4096
pointe_eval p3_raw --preprocessing raw
pointe_eval p3_crop --preprocessing crop
pointe_eval p4_den --guidance "$WG" --denoise
commit_phase "Phase P: Point-E sweeps (views, guidance, resolution, preprocessing, denoise)"

# ---- TripoSR track ----
triposr_eval() { # tag + extra args
  local tag="$1"; shift
  if [ -f "nesegemaa/triposr/summary${tag:+_$tag}.json" ]; then
    log "triposr $tag: present, skipped"; return 0
  fi
  log "triposr $tag: start"
  if [ -n "$tag" ]; then
    python nesegemaa/triposr/eval_triposr.py --tag "$tag" "$@" >> /tmp/opencode/queue.log 2>&1 \
      || log "triposr $tag FAILED"
  else
    python nesegemaa/triposr/eval_triposr.py "$@" >> /tmp/opencode/queue.log 2>&1 \
      || log "triposr default FAILED"
  fi
}
triposr_eval "" 
triposr_eval raw --preprocessing raw
triposr_eval crop --preprocessing crop
commit_phase "Phase T: TripoSR faithful evals (masked/raw/crop)"

# ---- final refresh: default panels with FINAL code (triplets for figures) ----
log "final refresh: default panels"
python nesegemaa/pointe/faithful_eval.py >> /tmp/opencode/queue.log 2>&1 || log "pointe refresh FAILED"
python nesegemaa/triposr/eval_triposr.py >> /tmp/opencode/queue.log 2>&1 || log "triposr refresh FAILED"
if [ -n "${WINNER:-}" ]; then
  python nesegemaa/atlas_qual.py --run "$WINNER" --weights best >> /tmp/opencode/queue.log 2>&1 || log "winner qual FAILED"
else
  log "winner qual SKIPPED (no OK rows to pick a winner from)"
fi

# ---- final build ----
log "final build"
python nesegemaa/build_final.py >> /tmp/opencode/queue.log 2>&1 || log "build_final FAILED"
commit_phase "Final: ablation tables, cross table, figures, REPORT skeleton"
$PUSH >> "$QLOG" 2>&1 || log "FINAL PUSH FAILED"
log "QUEUE COMPLETE"
