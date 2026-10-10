#!/usr/bin/env bash
# Post-dataset-fix rework (force re-run of everything in scope).
#
# Context: the point clouds were renamed to their true objects
# (fix_point_cloud_names.py; 1608/1695 were shuffled). Every AtlasNet run was
# trained on shuffled targets and every eval scored shuffled GT, so all
# results in the branch are redone here on corrected data. Pre-fix numbers
# are preserved in nesegemaa/*_prefix.csv for the paper's before/after table.
#
# Order: Point-E re-evals -> TripoSR re-evals -> AtlasNet A0a/A0b -> phases
# A1-A4 -> winner seeds -> rebuild tables/figures. Commits after each stage.
set -u
ROOT=/home/susan/agentic_workspace/home-omni3d
cd "$ROOT" || exit 1
source .venv/bin/activate
QLOG=/tmp/opencode/rework.log
mkdir -p /tmp/opencode
log() { echo "[$(date '+%F %T')] $*" | tee -a "$QLOG"; }
GIT="git -c user.name=lab-agent -c user.email=lab-agent@local"
PUSH="git push git@github.com:NoureldinAbdelrahman/home-omni3d.git nesegemaa:nesegemaa"
commit_stage() { # $1 = message
  git add nesegemaa/ results/atlasnet/ results/qualitative/atlasnet/ 2>/dev/null
  $GIT commit -m "$1" 2>&1 | tail -1 >> "$QLOG" || true
  $PUSH >> "$QLOG" 2>&1 || log "PUSH FAILED (retry at end)"
}

# ---- 1) Point-E: predictions are image-only, but scores use fixed GT ----
pointe() { # tag extra...
  local tag="$1"; shift
  if [ -n "$tag" ]; then
    python nesegemaa/pointe/faithful_eval.py --tag "$tag" "$@" >> "$QLOG" 2>&1 \
      || log "pointe $tag FAILED"
  else
    python nesegemaa/pointe/faithful_eval.py "$@" >> "$QLOG" 2>&1 \
      || log "pointe default FAILED"
  fi
}
log "stage: Point-E re-evals (fixed GT)"
pointe ""
pointe p1_best --views "0,8,16" --fusion best
pointe p1_median --views "0,8,16" --fusion median
pointe p1_v12 --view 12
pointe p2_1k --points 1024
pointe p2_g1 --guidance 1.0
pointe p2_g3.0_s012_1k --guidance 3.0 --seeds "0,1,2" --points 1024
pointe p2_g3.0_s012_4k --guidance 3.0 --seeds "0,1,2" --points 4096
pointe p2_g5 --guidance 5.0
pointe p3_raw --preprocessing raw
pointe p3_crop --preprocessing crop
pointe p4_den --guidance 3.0 --denoise
commit_stage "Rework (dataset fix): Point-E re-evals on corrected point clouds"

# ---- 2) TripoSR: same logic ----
log "stage: TripoSR re-evals (fixed GT)"
python nesegemaa/triposr/eval_triposr.py >> "$QLOG" 2>&1 || log "triposr default FAILED"
python nesegemaa/triposr/eval_triposr.py --preprocessing raw --tag raw >> "$QLOG" 2>&1 || log "triposr raw FAILED"
python nesegemaa/triposr/eval_triposr.py --preprocessing crop --tag crop >> "$QLOG" 2>&1 || log "triposr crop FAILED"
commit_stage "Rework (dataset fix): TripoSR re-evals on corrected point clouds"

# ---- 3) AtlasNet: full retrain grid ----
log "stage: AtlasNet A0a"
rm -rf AtlasNet/log/a0_svr25_imgnet
mkdir -p AtlasNet/log/a0_svr25_imgnet
CLASSES=$(python -c "import json;print(' '.join(d['synsetId'] for d in json.load(open('AtlasNet/dataset/data/taxonomy.json'))))")
( cd AtlasNet && python train.py --class_choice $CLASSES --SVR --no_metro --nepoch 50 \
    --batch_size 8 --nb_primitives 25 --template_type SQUARE \
    --dir_name log/a0_svr25_imgnet --workers 2 > log/a0_svr25_imgnet/train_stdout.log 2>&1 )
python nesegemaa/summarize_run.py --run a0_svr25_imgnet >> "$QLOG" 2>&1 || log "summarize a0a FAILED"
python nesegemaa/atlas_qual.py --run a0_svr25_imgnet --weights best >> "$QLOG" 2>&1 || log "qual a0a FAILED"

log "stage: AtlasNet A0b (warm decoder)"
rm -rf AtlasNet/log/a0_svr25_warmdec
mkdir -p AtlasNet/log/a0_svr25_warmdec
( cd AtlasNet && python train.py --class_choice $CLASSES --SVR --no_metro --nepoch 50 \
    --batch_size 8 --nb_primitives 25 --template_type SQUARE \
    --reload_decoder_path /media/susan/429428ec-710b-483c-9aaa-d0c4b6968baa/home-omni3d/downloads/official/trained_models/atlasnet_singleview_25_squares/network.pth \
    --dir_name log/a0_svr25_warmdec --workers 2 > log/a0_svr25_warmdec/train_stdout.log 2>&1 )
python nesegemaa/summarize_run.py --run a0_svr25_warmdec >> "$QLOG" 2>&1 || log "summarize a0b FAILED"
python nesegemaa/atlas_qual.py --run a0_svr25_warmdec --weights best >> "$QLOG" 2>&1 || log "qual a0b FAILED"
commit_stage "Rework (dataset fix): AtlasNet faithful baselines retrained"

atlas_phase() {
  local phase="$1"
  log "phase $phase: start"
  ./nesegemaa/run_phase_atlas.sh "$phase"
  commit_stage "Rework (dataset fix): phase $phase retrained"
  log "phase $phase: done"
}
log "stage: AtlasNet phases A1-A4"
atlas_phase a1 <<'SPEC'
a1_k3_max|--n_views 3 --views_pool max
a1_k3_attn|--n_views 3 --views_pool attn
a1_k5_max|--n_views 5 --views_pool max
a1_k5_attn|--n_views 5 --views_pool attn
a1_k8_max|--n_views 8 --views_pool max
a1_k8_attn|--n_views 8 --views_pool attn
SPEC
atlas_phase a2 <<'SPEC'
a2_p1_sq|--nb_primitives 1
a2_p10_sq|--nb_primitives 10
a2_p25_sphere|--template_type SPHERE
a2_p25_4096|--number_points 4096 --number_points_eval 4096
a2_p25_bn512|--bottleneck_size 512
SPEC
atlas_phase a3 <<'SPEC'
a3_lr3e4|--lrate 0.0003
a3_bs16|--batch_size 16
a3_bs32|--batch_size 32
a3_aug_rot|--random_rotation
a3_aug_flip|--data_augmentation_random_flips
a3_aug_aniso|--anisotropic_scaling
a3_frozen|--freeze_encoder
SPEC
atlas_phase a4 <<'SPEC'
a4_edge1e3|--loss_reg edge --loss_reg_w 0.001
a4_edge1e2|--loss_reg edge --loss_reg_w 0.01
SPEC

log "stage: winner seeds (from post-fix ablation CSV)"
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
  log "winner: $WINNER"
  EXTRA=$(cat "AtlasNet/log/$WINNER/extra_args.txt" 2>/dev/null || echo "")
  printf '%s\n' "${WINNER}_s7|${EXTRA} --seed 7" "${WINNER}_s123|${EXTRA} --seed 123" \
    | ./nesegemaa/run_phase_atlas.sh seeds
  commit_stage "Rework (dataset fix): winner seeds"
else
  log "winner seeds SKIPPED (no OK rows)"
fi

# ---- 4) rebuild tables + figures ----
log "stage: rebuild tables/figures"
python nesegemaa/build_final.py >> "$QLOG" 2>&1 || log "build_final FAILED"
python nesegemaa/figures/make_report_figures.py >> "$QLOG" 2>&1 || log "report figures FAILED"
commit_stage "Rework (dataset fix): rebuilt ablation tables, cross table, figures"
$PUSH >> "$QLOG" 2>&1 || log "FINAL PUSH FAILED"
log "REWORK COMPLETE"
