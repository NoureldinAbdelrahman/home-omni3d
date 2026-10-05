# TripoSR track (nesegemaa)

TripoSR (Tochilkin et al. 2024, "TripoSR: Fast 3D Object Reconstruction from a
Single Image", arXiv:2410.06769): image → triplane-NeRF → mesh, feed-forward
(single forward pass, no per-scene optimization — unlike our NeRF oracle).

Ownership moved here from the `hamdy` branch (which held only a README title
and no code; see `hamdy@7cd999e` neutralizing that claim). Third-party source
lives OUTSIDE this repo at `<hdd>/home-omni3d/thirdparty/TripoSR` (commit
`107cefdc`, recorded in eval summaries); only our eval harness lives here.

## Status

- [x] install: lean deps from requirements (minus GUI/demo-only pkgs);
  `torchmcubes` cannot compile here (CMake needs a CUDA>=13 toolchain
  dialect; only nvcc 12.8 present) -> vendored CPU shim
  (`nesegemaa/triposr/shim/`, PyMCubes backend, installed as
  `torchmcubes-cpu-shim`; same Lewiner algorithm, CPU only)
- [ ] weights prefetch (`stabilityai/TripoSR`, SHA recorded in summary)
- [ ] `eval_triposr.py` (image-only input, mesh→points sampling, UnitBall +
      squared units + tau 0.001 — identical scoring to the AtlasNet track)
- [ ] panel eval on the shared test objects + `results.csv`/`summary.json`

## Design notes

- `torchmcubes` needs a CUDA build (nvcc 12.8 vs torch cu130 — may fail).
  If it fails, mesh extraction falls back to multi-view depth fusion from
  TripoSR's own renderer (no new deps); the fallback is recorded, never silent.
- Preprocessing mirrors the Point-E track variants (`raw`, `masked`,
  `crop`) so the two feed-forward tracks stay comparable.
- GT clouds are scoring-only, never fed to the model (rule 1).
