# Point-E track (Nesegemaa)

Reference copied from `origin/mohamed-ayman:mohamed_ayman/src/models/baseline_point_e.py`
(see `point_e_reference_mohamed.py`) — starter wrapper around the official
`point-e` package (`base40M` + `PointCloudSampler`), with a geometric
visual-hull fallback when weights can't download.

Verdict: good starting API pattern, NOT a faithful Point-E yet. Missing:
upSampler stage (1k -> 4k), proper CLIP image preprocessing / conditioning,
guidance-scale handling, UnitBall normalization matching our eval. The lab
agent's first job is a faithful implementation (see branch plan), keeping this
file as reference only.
