# Noureldeen A. Younes (58-1632): COLMAP

| | |
|---|---|
| Branch | `bones` |
| Models | COLMAP: known poses (triangulate with GT cameras) and blind incremental SfM |
| MS2 status | scored; multi-view reference |
| Protocol | 24 views (+ GT cameras for known poses), masked SIFT, exhaustive matching, 4 random objects per class = 294; 286 scored (8 fail the GT-alignment filter); viewer scorer: CD-L1, F@1/2/5% |
| Results | `results/colmap/sparse_known/summary.json`, `results/colmap/sparse_sfm/summary.json` (+ `metrics.csv`, `objects.json`, `figures/qualitative.png`); viewer runs `viewer/runs/bones__colmap__*` (also on `main`) |
| Headline (micro) | known poses CD-L1 0.134, F@5% 0.539, 24.0 registered views; blind SfM CD-L1 0.370, F@5% 0.378, 14.3 registered views |
| Focus (slide 20, proposed) | fixed reference, no new runs |

Dropped: additive-subtractive modelling (named here at MS1, never implemented).

Not to be confused with Noureldin S. (`nesegemaa`). Code lives on `bones` (`bones/`). Work only in `bones/` and `results/colmap/`. Full index: [`results/README.md`](../results/README.md).
