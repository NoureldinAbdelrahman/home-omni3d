# Docs map

| Doc | What it answers |
|---|---|
| [`ms2/ms2_problem_definition.pptx`](ms2/ms2_problem_definition.pptx) | The MS2 deck (23 slides): scope, owners, status, headline numbers |
| [`ms2/ms2_outline.md`](ms2/ms2_outline.md) | Source path / commit for every slide claim, plus the deck's changelog |
| [`ms2/ERRATA.md`](ms2/ERRATA.md) | Places where the deck or outline differs from the committed artifacts |
| [`ms2/README.md`](ms2/README.md) | How to read or extract the deck |
| [`DATASET.md`](DATASET.md) | Dataset scopes, full-download completeness, licence, per-category counts |
| [`METRICS.md`](METRICS.md) | Metric definitions, the scale each track uses, why tracks are not comparable |
| [`ROADMAP.md`](ROADMAP.md) | Pipeline: what `main` does, where each owner's code and results live, how to reproduce |
| [`BRANCH_ALIGNMENT.md`](BRANCH_ALIGNMENT.md) | Docs-only follow-up per owner branch |
| [`../results/README.md`](../results/README.md) | MS2 result index (owner, branch, path, metric, n, protocol, status) |
| [`../viewer/runs/README.md`](../viewer/runs/README.md) | Viewer run index |
| [`../CONTRIBUTING.md`](../CONTRIBUTING.md) | Branch rules, naming, provenance, no leaderboard |
| `project_guidelines.pdf`, `project_ms_1.pdf` | Course guidelines, MS1 brief |

## Source-of-truth order

When two places disagree:

1. **The MS2 deck**: team-facing claims, status, ownership, scope, headline numbers.
2. **Committed result artifacts** on the owner's branch: exact values. If the deck rounds or differs, the artifact value is kept and the difference goes to [`ms2/ERRATA.md`](ms2/ERRATA.md).
3. **[`ms2/ms2_outline.md`](ms2/ms2_outline.md)**: provenance (paths, commits, dates, settings, caveats).
4. **Other README / doc prose**: check it against 1–3 before trusting it.

Owner branches are behind or ahead of `main` at different points. Cite remote refs (`origin/ashry`, `origin/hamdy`) when a local branch may be stale.
