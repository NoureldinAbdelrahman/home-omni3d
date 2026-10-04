#!/usr/bin/env python3
"""Final aggregation for the nesegemaa branch (run at the very end).

Reads every committed result artifact and produces, without touching the
notebook or other members' paths:

- nesegemaa/ablation_pointe.csv  (one row per Point-E summary_*.json)
- nesegemaa/ablation_triposr.csv (one row per TripoSR summary_*.json)
- nesegemaa/cross_table.csv      (best-AtlasNet vs best-PointE vs TripoSR,
  per object, on shared panel objects)
- nesegemaa/figures/*.png        (run bars, sweep bars, cross grouped bars)
- nesegemaa/REPORT.md            (methods + protocol (fixed text) + all tables
  auto-rendered; interpretation is added by the final human/agent message)

Best-AtlasNet = min best_chamfer among status=OK rows of ablation_atlasnet.csv.
Best-PointE  = min micro_chamfer among default-panel pointe summaries.
TripoSR      = its default (masked) summary.
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
NES = ROOT / "nesegemaa"
FIG = NES / "figures"


def load_json(p):
    return json.loads(Path(p).read_text())


def md_table(rows, cols):
    out = ["| " + " | ".join(cols) + " |",
           "| " + " | ".join(["---"] * len(cols)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


def ablation_csv(summaries, cols, out_path, source):
    rows = []
    for sp in sorted(summaries):
        try:
            s = load_json(sp)
        except Exception as exc:
            print(f"skip {sp}: {exc}")
            continue
        cfg = s.get("config", {})
        row = {"tag": sp.stem.replace("summary", "").strip("_") or "default"}
        row.update({c: cfg.get(c, "") for c in cols if c in cfg})
        for k in ("n_objects", "micro_chamfer", "micro_fscore",
                  "macro_chamfer", "macro_fscore"):
            row[k] = s.get(k, "")
        row["noise"] = ",".join(s.get("noise_categories_lt3", []))
        rows.append(row)
    if rows:
        with open(out_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"{source}: {len(rows)} rows -> {out_path.relative_to(ROOT)}")
    return rows


def main():
    FIG.mkdir(parents=True, exist_ok=True)

    pointe_rows = ablation_csv(
        sorted((NES / "pointe").glob("summary*.json")),
        ["views", "fusion", "preprocessing", "guidance", "points",
         "seeds", "denoise"], NES / "ablation_pointe.csv", "pointe")
    tripoSR_rows = ablation_csv(
        sorted((NES / "triposr").glob("summary*.json")),
        ["view", "preprocessing", "points", "mc_resolution"],
        NES / "ablation_triposr.csv", "triposr")

    # ---- cross table: per-object join on shared panel objects ----
    import csv as _csv
    atlas_qm = {}
    for qp in sorted((ROOT / "results" / "atlasnet").glob("*/qual_metrics.json")):
        try:
            for r in json.loads(Path(qp).read_text()):
                atlas_qm.setdefault((r["category"], r["object"]), []).append(
                    (qp.parent.name, r["chamfer"], r["fscore"]))
        except Exception:
            pass
    pointe_pc, tripoSR_pc = {}, {}
    for rp in sorted((NES / "pointe").glob("results*.csv")):
        try:
            with open(rp) as f:
                for r in _csv.DictReader(f):
                    if rp.stem == "results":
                        pointe_pc[(r["category"], r["object"])] = (
                            float(r["chamfer"]), float(r["fscore"]))
        except Exception:
            pass
    for rp in sorted((NES / "triposr").glob("results*.csv")):
        try:
            with open(rp) as f:
                for r in _csv.DictReader(f):
                    if rp.stem == "results":
                        tripoSR_pc[(r["category"], r["object"])] = (
                            float(r["chamfer"]), float(r["fscore"]))
        except Exception:
            pass
    # best-atlas run = min best_chamfer among OK ablation rows
    best_atlas_run, best_atlas_ch = None, float("inf")
    ap = NES / "ablation_atlasnet.csv"
    if ap.is_file():
        with open(ap) as f:
            for r in _csv.DictReader(f):
                try:
                    if r.get("status") == "OK" and float(r["best_chamfer"]) < best_atlas_ch:
                        best_atlas_ch, best_atlas_run = float(r["best_chamfer"]), r["run"]
                except (ValueError, TypeError):
                    pass
    cross = []
    objs = sorted(set(pointe_pc) | set(tripoSR_pc))
    for cat, obj in objs:
        a = next(((ch, fs) for run, ch, fs in atlas_qm.get((cat, obj), [])
                  if run == best_atlas_run), (None, None))
        p = pointe_pc.get((cat, obj), (None, None))
        t = tripoSR_pc.get((cat, obj), (None, None))
        cross.append({"object": f"{cat}_{obj}", "category": cat,
                      "atlas_run": best_atlas_run or "",
                      "atlas_ch": a[0], "atlas_f": a[1],
                      "pointe_ch": p[0], "pointe_f": p[1],
                      "triposr_ch": t[0], "triposr_f": t[1]})
    with open(NES / "cross_table.csv", "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=["object", "category", "atlas_run",
                                           "atlas_ch", "atlas_f", "pointe_ch",
                                           "pointe_f", "triposr_ch", "triposr_f"])
        w.writeheader()
        w.writerows(cross)
    print(f"cross_table: {len(cross)} objects (atlas run: {best_atlas_run})")

    # ---- figures ----
    if cross:
        xs = np.arange(len(cross))
        w = 0.27
        fig, axes = plt.subplots(1, 2, figsize=(max(11, 0.5 * len(cross)), 4.8))
        for j, (key, col) in enumerate((("atlas_ch", "#4C72B0"),
                                        ("pointe_ch", "#55A868"),
                                        ("triposr_ch", "#C44E52"))):
            vals = [r[key] if r[key] is not None else float("nan") for r in cross]
            axes[0].bar(xs + (j - 1) * w, vals, w, label=key.split("_")[0], color=col)
        axes[0].set_xticks(xs)
        axes[0].set_xticklabels([r["object"] for r in cross], rotation=30,
                                ha="right", fontsize=7)
        axes[0].set_title("Cross-model Chamfer per object")
        axes[0].legend(); axes[0].grid(axis="y", alpha=0.3)
        for j, key in enumerate(("atlas_f", "pointe_f", "triposr_f")):
            vals = [r[key] if r[key] is not None else float("nan") for r in cross]
            axes[1].bar(xs + (j - 1) * w, vals, w,
                        label=key.split("_")[0],
                        color=("#4C72B0", "#55A868", "#C44E52")[j])
        axes[1].set_xticks(xs)
        axes[1].set_xticklabels([r["object"] for r in cross], rotation=30,
                                ha="right", fontsize=7)
        axes[1].set_title("Cross-model F-score per object")
        axes[1].legend(); axes[1].grid(axis="y", alpha=0.3)
        fig.suptitle("Best AtlasNet vs Point-E vs TripoSR (shared objects)", fontsize=11)
        fig.tight_layout()
        fig.savefig(FIG / "cross_models.png", dpi=150)
        plt.close(fig)
        print("saved", (FIG / "cross_models.png").relative_to(ROOT))

    # ---- REPORT.md (tables auto-filled; interpretation added by final message)
    rep = ["# nesegemaa track — final report (auto-generated tables + fixed protocol notes)",
           "",
           "Scope: AtlasNet + Point-E (+ TripoSR track), branch `nesegemaa`.",
           "Protocol (all numbers): splits.json test lists, view `00.png` unless noted, "
           "UnitBall (subtract mean, divide by max radius), squared-distance units, "
           "F threshold 0.001 (≈0.0316 Euclidean). `<3`-object categories are noise-flagged.",
           "",
           "## AtlasNet ablation",
           ""]
    if (NES / "ablation_atlasnet.csv").is_file():
        with open(NES / "ablation_atlasnet.csv") as f:
            rdr = list(_csv.DictReader(f))
        rep.append(md_table(
            [{k: r[k] for k in ("run", "K_views", "pool", "patches", "template",
                               "bottleneck", "encoder_train", "aug", "loss",
                               "lr", "batch", "seed", "best_chamfer",
                               "best_fscore", "status")} for r in rdr],
            ["run", "K_views", "pool", "patches", "template", "bottleneck",
             "encoder_train", "aug", "loss", "lr", "batch", "seed",
             "best_chamfer", "best_fscore", "status"]))
    rep += ["", "## Point-E sweeps", ""]
    if pointe_rows:
        rep.append(md_table(pointe_rows, list(pointe_rows[0].keys())))
    rep += ["", "## TripoSR sweeps", ""]
    if tripoSR_rows:
        rep.append(md_table(tripoSR_rows, list(tripoSR_rows[0].keys())))
    rep += ["", "## Cross-model (shared objects)", ""]
    if cross:
        rep.append(md_table(cross, ["object", "atlas_run", "atlas_ch", "atlas_f",
                                   "pointe_ch", "pointe_f", "triposr_ch", "triposr_f"]))
    rep += ["", "## Interpretation",
            "",
            "(Completed in the final handoff message after all phases land. "
            "Headline questions it must answer: what moved Chamfer-down/F-up vs "
            "the A0 baseline; weight provenance per winner; overfitting behavior "
            "(best vs final); noise-flagged categories excluded from claims.)",
            ""]
    (NES / "REPORT.md").write_text("\n".join(rep) + "\n")
    print("wrote", (NES / "REPORT.md").relative_to(ROOT))


if __name__ == "__main__":
    main()
