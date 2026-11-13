# MS2 deck outline — `ms2_problem_definition.pptx`

16 slides, 16:9. This file lists the source for every claim in the deck.

**Scope rule (presenter, 2026-11-07):** each member's planned work is exactly what is on their branch, nothing more. The plan slides (13–15) therefore have every track finish and evaluate its own branch content on its own protocol. Results are shown side by side, never ranked, and there is no new shared evaluator or re-scoring. Additive-subtractive modelling (named in the bones README, never implemented) is listed as dropped. Branches were read with

**Ownership rule (presenter, 2026-11-07):** each model has exactly one owner, and its results come only from that owner. Noureldin owns AtlasNet, Point-E and TripoSR (nesegemaa branch, plus his MS1 runs on main). Mohamed Ayman owns Pix2Vox and Pix2Vox++. Bones owns COLMAP, Ashry owns 3DGS, NeRF and NeuS (the NeRF oracle results live on ashry), and Hamdy owns LGM (`refs/heads/hamdy:LGM/`, ee0b1cc, 2026-11-07). Mohamed's multi-view AtlasNet and Point-E baseline are not shown. **No README result tables are used.**
`git show refs/heads/<branch>:<path>` and none were checked out or changed. "main@f33b704" means
the file as committed on that branch at that commit.

| # | Slide | Claims → source |
|---|-------|-----------------|
| 1 | Title | Course, instructor (Dr. Mohamed Karam): `docs/project_guidelines.pdf`. "Team 9": `refs/heads/mohamed-ayman:mohamed_ayman/README.md`, confirmed by the presenter. Presenter (Noureldin S. Abdelrahman) and TA (Eng. Rawan Darwish): given by the presenter. No date is shown on the slide |
| 2 | Agenda | Mirrors the deck order. The MS2 pillars come from `docs/project_guidelines.pdf` §3 |
| 3 | Problem statement | Task, input and output: `nesegemaa/PAPER.md` §1, §3, §4 and `docs/ROADMAP.md` §1. 224² input: `milestone-1:Pix2Vox/config.py` (IMG_W/H) and `refs/heads/mohamed-ayman:mohamed_ayman/configs/config.yaml` (image_size 224). "Interior staging": the mohamed-ayman README. The AR and manual-CAD motivation is framing, not a repo fact; wording shortened per a colleague's edit (`ms2_problem_definition (1).pptx`) |
| 4 | Dataset at a glance | 76 / 1,694 / 260 / split 1177·257·260: `refs/heads/ashry:ashry/benchmark/results/benchmark_v1.json` (1bf8436, 2026-10-06; ashry was rebased on 2026-11-07, and the old hash d49469e is no longer on any branch). 24 views, 1024² RGBA, 4,096 pts, 70/15/15, laptop_002 dropped: `nesegemaa/PAPER.md` §3 and `README.md`. Sample image: `main:results/comparison/figures/dataset_examples.png` (f33b704). Licence: CC BY 4.0, as stated by the presenter (no file in the repo states it) |
| 5 | Long tail / challenges / proxy | Chart: per-category counts computed at build time from `benchmark_v1.json` (ashry, 1bf8436). doll 84; bed, canned_beverage and plug 2; median 19.5; top-10 share 31.05%; 35/76 categories with <3 test objects. All are computed from that file, and 35 matches `PAPER.md` §2 |
| 6 | From MS1 to MS2 | MS1 coverage: `git show milestone-1:README.md`, `git log milestone-1` (f33b704, 2026-09-26), and the MS1 branch table in `README.md` (ashry: Pixel2Mesh, DeepSDF, 3DGS, NeRF; hamdy: TripoSR). Current scope: `refs/heads/ashry:ashry/README.md` (3DGS/NeRF/NeuS) and `refs/heads/hamdy:hamdy/README.md` (TripoSR moved). Ablation numbers: `main:results/ablation/image_usage.json` (96c724d): AtlasNet F real 0.00286 / constant 0.00319; Pix2Vox IoU@0.3 real 0.312 / constant 0.214 |
| 7 | Models — single view | Owner column follows the ownership rule. Architecture lines: `nesegemaa/PAPER.md` §2 and §4, plus Mohamed's model code (`mohamed_ayman/src/models/pix2vox.py`, `configs/config.yaml`); no README content. 1,177 train objects: `benchmark_v1.json`. LGM row: `refs/heads/hamdy:LGM/infer.py` (rembg → ImageDream 4 views → U-Net → 3D Gaussians `.ply`, pretrained weights) |
| 8 | Multi-view references | `refs/heads/bones:bones/README.md`, `refs/heads/ashry:ashry/{README.md,3dgs,nerf,neus}/README.md`. Views 000–019 / 020–023: `benchmark_v1.json` |
| 9 | Tools & environment | `requirements.txt` (torch 2.14, torchvision 0.29, open3d 0.20, …), `README.md` setup, the bones README (pycolmap 4.2.1), `viewer/README.md`. 16 GB GPU: the ashry README. 2× RTX 3090 24 GB server: `nesegemaa/notes.md` |
| 10 | Evaluation plan | `docs/METRICS.md` §1–§3 and `PAPER.md` §2 metric box. Callout wording from the colleague's edit. Held-out views: `benchmark_v1.json` |
| 11 | Progress 1 (trained) | Noureldin: MS1 AtlasNet-SVR best F 0.0793–0.0911 over 5 runs, from `main:results/comparison/metrics_table.csv` (f33b704); all MS1 commits are his (same author email). His current AtlasNet/Point-E/TripoSR work is status only, per the no-post-fix-numbers rule. Mohamed Ayman: `mohamed_ayman/outputs/evaluations/{pix2vox,pix2vox++}_3views_report.md` micro iou@0.4 0.3327 / 0.3272 and CD-L1 0.1054 / 0.0992, checked against the per-object CSVs (263 objects, 76 categories); commits b46946f (2026-10-02) and 324240c (2026-10-10). Mohamed's AtlasNet evaluation was removed under the ownership rule |
| 12 | Progress 2 (multi-view / pending) | bones: `results/colmap/sparse_known/summary.json` and `sparse_sfm/summary.json` (b2b9ad3, 2026-10-09). 294 objects, 286 scored, micro chamfer_l1 0.1341 / 0.3705, f_5 0.5393 / 0.3779, registered_mean 24.0 / 14.3. Figure: `results/colmap/sparse_known/figures/qualitative.png`. ashry (0443f31, 2026-11-07): 3DGS/NeRF/NeuS trained and tested, plus a lab app (`ashry/app/`) that extracts clouds and scores Chamfer and F@1%/2%. It rescales each cloud to the reference's robust 1–99% extent, which is a protocol difference to note at MS3. No scores are committed. NeRF oracle 19.11 dB / SSIM 0.804: `refs/heads/ashry:results/oracle/summary.json` (8a23409; 24 objects, 128 px, 2,500 iterations, 20 train / 4 test views). hamdy: LGM pipeline code pushed (ee0b1cc, 2026-11-07, official LGM implementation + `infer.py`); no outputs or scores committed |
| 13 | Who does what (phases + member table) | Member → branch → models → status: Mohamed Ayman / mohamed-ayman (Pix2Vox, Pix2Vox++, Streamlit `app/app.py`), Noureldin / nesegemaa (AtlasNet-SVR, Point-E, TripoSR), Bones / bones (COLMAP known/SfM), Ashry / ashry (3DGS, NeRF, NeuS, `app/`), Hamdy / hamdy (LGM, ee0b1cc) |
| 14 | Timeline & risks | Weeks are relative to MS2, since the MS3 date isn't announced yet. Tasks are per track only. Risks come from `PAPER.md` §2–§3 (long tail), the ashry README (16 GB GPU, sequential runs), `nesegemaa/notes.md` (server), `benchmark_v1.json` (SHA-256 per cloud) and the bones README (GT-alignment reliability) |
| 15 | Summary | Recap of slides 3–14 |
| 16 | Thank you | Repo: `README.md` / git remote |

## Deliberately excluded (scope rules)
- The nesegemaa post-fix rework: `nesegemaa/ablation_*.csv`, `cross_table*.csv`, `nesegemaa/figures/*`, and `PAPER.md` §5.2 and §6+.
- Any "best model" or leaderboard claim, and the point-cloud pairing repair story.
- The optional sneak-peek slide was dropped. No committed viewer screenshot exists, and rendering nesegemaa triplets would surface its rework outputs.
- main's MS1 figures `qualitative/pix2vox_compare.png` and `atlasnet_compare.png` were not used. The Pix2Vox track is mohamed-ayman's, and the AtlasNet figure shows only the MS1-era run.

## [TEAM TO CONFIRM] — to check with the team
0. Improvement proposals for Mohamed (Pix2Vox/++), Noureldeen A. (COLMAP), Ashry, Hamdy.
1. ashry 3DGS / NeRF / NeuS scores. None are committed on `ashry` (HEAD 0443f31, 2026-11-07); the lab app computes them locally only.
2. hamdy LGM results: code is on `origin/hamdy` (ee0b1cc); no outputs, scores or evaluation protocol committed yet.
3. ~~mohamed-ayman README table~~ Resolved: README result tables are not used; only committed evaluation outputs are.
4. ~~NeRF-oracle PSNR~~ Resolved: `ashry:results/oracle/summary.json` gives 19.11 dB / SSIM 0.804 over 24 objects. `docs/METRICS.md`'s 25.2 dB / 0.916 is stale, so the team may want to fix that doc.

Resolved by the presenter: presenter name, TA name (Rawan Darwish), Team 9, licence (CC BY 4.0), dates (left off the deck), Pix2Vox (one track, Mohamed's), the one-owner-per-model rule, and no README tables.

## Qualitative slides (2026-11-10)
Slides 12 and 16 recreate the team viewer's panels (grey = GT, teal = prediction) with matplotlib, from committed viewer exports:
- Slide 12 (Noureldin S.): `refs/heads/nesegemaa:viewer/runs/nesegemaa__atlasnet__a2_p25_sphere_s123`, `..._triposr__default_masked_full_aligned`, `..._point_e__default_masked_g3_full_aligned` (ICP-aligned for display only, marked *).
- Slide 16 (Mohamed Ayman): `refs/heads/mohamed-ayman:viewer/runs/mohamed_ayman__pix2vox__base`, `..._pix2voxplusplus__refined`.
- GT and input photos: local `dataset/point_clouds` and `dataset/renders/<obj>/000.png`; all 470 local clouds verified against `fix_point_cloud_names.json` (array SHA-1). Objects: chair, sofa, teddy bear, vase; per class, the median object by the owner's viewer F@5% (not the best).
- Finding stated on slides 12–14: AtlasNet's outputs of different objects are 3–4× more alike (0.064 mean NN distance between outputs, 40 local test objects) than TripoSR's (0.201) or Point-E's (0.275); its output fits a random other object's GT almost as well as its own (0.126 vs 0.114). This is a presenter-side computation from the viewer exports, consistent with `nesegemaa/notes.md` ("Atlas blob diagnosed").

## Rework for the MS2 email (2026-11-10)
The course email (presentations 14–26 Nov) asks for: run the MS1 model, setup/run configuration, results + discussion, and an improvement. The deck is now 18 slides:
1 Title · 2 Agenda · 3 Problem · 4 Dataset · 5 Long tail · 6 MS1→MS2 · 7 Models explored · 8 Evaluation protocol ·
9 Setup (AtlasNet/Point-E/TripoSR) · 10 AtlasNet results · 11 AtlasNet vs frozen priors · 12 Discussion · 13 Proposed improvement ·
14 Pix2Vox/++ setup & results · 15 Multi-view setup & results · 16 Who does what · 17 Summary · 18 Thank you.
Dropped: tools, multi-view cards, old progress slide, timeline & risks.

**Noureldin S.'s results are now included** (owner rule: his models are reported only from his branch). Sources:
- Setup: `nesegemaa/PAPER.md` §4, `nesegemaa/ablation_atlasnet.csv` (lr, batch, epochs, points), `results/atlasnet/a2_p25_sphere_s123/options.json`.
- Slide 10: full-split micro Chamfer from `nesegemaa/ablation_atlasnet.csv` `qual_micro_ch`; baseline 0.0467 / F 0.083 and winner 0.0332 / F 0.161 from `results/atlasnet/{a0_svr25_imgnet,a2_p25_sphere_s123}/qual_agg.json`. Density caveat (64,050 vs 2,500 pts; 0.0409 at matched 2,500) from `PAPER.md` §6.1 audit note.
- Slide 11: Point-E full 0.1859/0.0821 (`ablation_pointe.csv` tag full), TripoSR full 0.1289/0.0892 and raw/crop (`ablation_triposr.csv`); AtlasNet at matched 4,096 pts 0.0384/0.0896 from `PAPER.md` §6.1.
- Slide 12: `PAPER.md` §6.5–6.10; input test from `nesegemaa/atlas_input_ablation_s123.json`.
- Slide 13: tested gain as above; proposals from `PAPER.md` §7 (class-balanced sampling, fixed eval point counts) plus combining SPHERE + frozen encoder (not yet run).
- Slide 14: `mohamed-ayman:mohamed_ayman/configs/config.yaml`; epochs 15 / 30 = length of `outputs/checkpoints/*_3views_home_decor_history.json` (config says 25, so the logs are used); best val IoU 0.408 / 0.418 from the same histories; test numbers from the evaluation reports. Improvement: [TEAM TO CONFIRM — Mohamed].
- Slide 15: COLMAP setup from `bones/README.md`; Ashry screening settings from `ashry/README.md`.
The pairing-bug story is still not told on the slides.

## Names and sources (2026-11-09)
All on-slide "Source: …" lines, path captions and commit-hash footnotes were removed at the presenter's request; the sources live in this outline and the speaker notes. Two members share a first name, so the slides say **Noureldin S.** (Noureldin Salaheldin, nesegemaa, the presenter) and **Noureldeen A.** (Noureldeen Amr, bones branch). People are named instead of branches on slides 8, 12 and 13.

## Colleague's comments (2026-11-08)
- Slide 3: pipeline now reads 1 RGB view (224²) → network → 32³ voxels / points / mesh.
- Slide 5: on-slide source line removed (source stays in the notes and in this outline).
- Slide 7: retitled "Single-view models we explored" (summary of what was explored).
- Slide 11: one row per model under the member who works on it (Noureldin: AtlasNet-SVR, Point-E, TripoSR; Mohamed: Pix2Vox, Pix2Vox++; Hamdy: LGM); views and split shown per row; the duplicate Pix2Vox stat panel was removed.
- Slide 12: now multi-view only (bones, ashry); "Loading...." is gone and LGM sits on slide 11.
- Slide 13: Models column lists models only (apps dropped).

## Wording pass (2026-11-08)
On-slide text was cut for brevity on every slide; numbers and sources are unchanged. Slide 7's "Why it fits" column became Type (trained / frozen). Slide 13 is now a full-width member table with a Status column; the workflow bullets moved to its speaker notes. Slide 6 lists LGM as a frozen prior, and the screened MS1 candidates are named in its notes.

## Colleague's edited deck (`ms2_problem_definition (1).pptx`)
It was built on an older copy of this deck (pre scope/ownership rules: Branch column, d49469e hashes, shared-evaluator plan, Mohamed's AtlasNet row). Merged: the slide 3 wording, the slide 10 callout wording, Hamdy = LGM (slides 7, 12, 13, 16). Not merged, because it would undo later presenter decisions: the Branch columns, Mohamed's AtlasNet row and the README note, the shared protocol / re-scoring plan (slides 10, 13–15), "COLMAP, additive-subtractive", the stale d49469e hashes, and moving the NeRF oracle back to main. "Loading...." became "Pipeline code pushed; no results yet".

## Rebuild / verify
The deck was built with python-pptx 1.0.2 by a script kept outside the repo
(`build_deck.py` in the session scratchpad). It reads `benchmark_v1.json` directly from
`refs/heads/ashry` and figures from `refs/heads/main` and `refs/heads/bones`. The render check used
LibreOffice → PDF → `pdftoppm -r 80`, with output in `/tmp/opencode/ms2/`.
