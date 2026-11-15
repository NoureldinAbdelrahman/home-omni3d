# Noureldin S. Abdelrahman (58-7007): AtlasNet-SVR, Point-E, TripoSR

| | |
|---|---|
| Branch | `nesegemaa` (a branch/folder name, not a person) |
| Models | AtlasNet-SVR (trained); Point-E, TripoSR (frozen priors) |
| MS2 status | AtlasNet-SVR trained + evaluated; Point-E, TripoSR evaluated (frozen prior) |
| Protocol | 1 view (00), team split: 260 test objects; UnitBall normalisation, squared Chamfer, F@τ=0.001; micro + macro. AtlasNet SPHERE outputs 64,050 points; Point-E and TripoSR 4,096 |
| Results | `results/atlasnet/<run>/qual_agg.json`, `nesegemaa/ablation_{atlasnet,pointe,triposr}.csv`, `nesegemaa/PAPER.md` §6; viewer runs `viewer/runs/nesegemaa__*` |
| Headline (micro) | AtlasNet SPHERE 0.0332 / 0.161 (64,050 pts), 0.0384 / 0.090 at 4,096 pts; baseline 0.0467 / 0.083; TripoSR masked 0.129 / 0.089; Point-E default 0.186 / 0.082 |
| Focus (slide 20, proposed) | keep AtlasNet-SVR (SPHERE) and Point-E; drop TripoSR |

Caveats: Chamfer/F depend on point count (see [`docs/METRICS.md`](../docs/METRICS.md#atlasnet-sphere-point-density)). `a3_aug_*` runs never augmented. ICP-aligned viewer scores are an oracle.

Code lives on `nesegemaa` (`AtlasNet/`, `nesegemaa/`), not on `main`. Work only in `nesegemaa/`, `AtlasNet/` and `results/atlasnet/`. Full index: [`results/README.md`](../results/README.md).
