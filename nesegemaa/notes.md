# Nesegemaa — Point-E, AtlasNet

Branch: `nesegemaa`. Work only in `nesegemaa/` (+ `results/atlasnet/`, `results/qualitative/atlasnet/` for outputs).
Post-fix winner: AtlasNet `a2_p25_sphere_s123` (see `nesegemaa/PAPER.md` §6).

## Log

- 2026-10-11: all tracks retrained/re-evaluated on repaired clouds; PAPER.md rewritten with post-fix tables + §5.2 bug impact.
- 2026-10-14 (prior session): input-ablation retest on winner (weak conditioning confirmed); sphere density audit (64k vs 2.5k confound); full-split Atlas viewer export (260 objs); per-run triplets fix.
- 2026-10-14 (viewer-quality pass):
  - Atlas blob diagnosed: 11/25 sphere patches exploded to object scale (cup) / 6 collapsed (hammer) + 16× point overdraw. Viewer run re-exported capped at 4096 pts (8.6 MB, was 128 MB). Model behavior itself unchanged — good Chamfer via dense overlapping coverage, not clean tiling.
  - New `align.py` (Kabsch + ICP) + `export_to_viewer --points/--align`; `*_aligned` viewer runs for Point-E/TripoSR (visual only). Point-E hammer/shampoo collapses are ~90° pose errors (0.87→0.10, 0.56→0.11 in viewer frame); TripoSR gaps shrink less (shape-level).
  - TripoSR `eval_triposr.py --mesh-cleanup largest` (GT-blind, default none to preserve protocol) for flash/fin artifacts — needs GPU re-run.
  - Point-E viewer enriched with p1_best_3views + p3_raw configs (same 5-obj panel).
  - GPU work QUEUED, not run (local cuda:0 busy with hdsemg calibration, PID 62741): `run_when_gpu_free.sh --go` → TripoSR full-260 ×4 (~0.5 GPU-h) + Point-E default full-260 (~3.3 GPU-h) + rebuild tables.
- Open (2026-10-17 status): queued GPU jobs are DONE (local free day). Server Master-7536 provisioned and parity-proven (see below). Next: TripoSR finetune needs a new training loop + LoRA (engineering task, staged for server GPU1); AtlasNet is near ceiling (finetune +2%).
- 2026-10-17 (free-GPU day):
  - TripoSR full-260 x4 (default 0.1289/0.0892, raw 0.1322/0.0715, crop 0.1375/0.0715, masked_clean 0.1297/0.0886 = no-op, flashes are attached fins). Fixed 2 eval bugs: NumPy-2 `.ptp()` via trimesh.split (scipy incidence-graph components instead) + `rows` shadowing.
  - Point-E default full-260 (~3.3 h): 0.1859/0.0821. Tables rebuilt; cross_table kept at 5-obj panel (panel CSVs/triplets restored from git, full runs in `*_full*`; CPU rescore verified identical to GPU numbers).
  - AtlasNet finetune `a2_p25_sphere_ft1e4` (lr 1e-4 x15 from winner best-model): qual 0.0326/0.1610, marginal. Fixed `--reload_model_path` DataParallel-prefix crash in trainer_model.py.
  - Viewer now 14 runs: full-260 raw + ICP-aligned for all three tracks + finetune run.
  - Server Master-7536 (cdrives-pc-master@10.42.0.1, 2x3090 24GB, GPU1 free) provisioned: repo+data+weights synced, .venv torch 2.14+cu130, transformers pinned to 4.35.0 (v5 breaks 2024 TripoSR ckpt). Fixed absolute-symlink landmine (rsync -L; test views + all point clouds materialized, full 32GB renders pending for train time). Server TripoSR full-260 matches local to 1e-3. Guards: `run_when_gpu_free.sh` (local), `run_server_gpu1.sh` (server GPU1-only).
