#!/usr/bin/env python3
"""Generate report figures for the nesegemaa tracks (AtlasNet, Point-E, TripoSR).

Reads committed run ledgers (no network, no dataset needed) and writes PNGs to
nesegemaa/figures/. Source data:
  nesegemaa/ablation_atlasnet.csv   (22 runs)
  nesegemaa/ablation_pointe.csv     (12 evals)
  nesegemaa/ablation_triposr.csv    (3 evals)
  nesegemaa/cross_table.csv         (5 shared objects x 3 models)
  results/atlasnet/a0_*/summary.json (two faithful baselines)
"""

import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent
ATLAS_CSV = ROOT / "ablation_atlasnet.csv"
POINTE_CSV = ROOT / "ablation_pointe.csv"
TRIPOSR_CSV = ROOT / "ablation_triposr.csv"
CROSS_CSV = ROOT / "cross_table.csv"

C_ATLAS, C_POINTE, C_TRIPOSR = "#2b6cb0", "#dd6b20", "#2f855a"


def read_csv(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def load_atlas_runs():
    rows = read_csv(ATLAS_CSV)
    for tag in ("a0_svr25_imgnet", "a0_svr25_warmdec"):
        p = ROOT.parent / "results" / "atlasnet" / tag / "summary.json"
        if p.exists():
            s = json.loads(p.read_text())
            rows.append({
                "run": tag,
                "best_chamfer": s["best_chamfer"],
                "best_fscore": s["best_fscore"],
                "final_chamfer": s["final_chamfer"],
                "final_fscore": s["final_fscore"],
            })
    return rows


def family(run):
    if run.startswith("a0"):
        return "baseline"
    if run.startswith("a1"):
        return "views"
    if run.startswith("a2"):
        return "topology"
    if run.startswith("a3"):
        return "training"
    if run.startswith("a4"):
        return "loss"
    return "seed"


def fig_headline(atlas, pointe, triposr):
    best_a = min(atlas, key=lambda r: float(r["best_chamfer"]))
    best_p = min(pointe, key=lambda r: float(r["micro_chamfer"]))
    best_t = min(triposr, key=lambda r: float(r["micro_chamfer"]))
    labels = [
        f"AtlasNet (trained)\n{best_a['run']}",
        f"Point-E (frozen)\n{best_p['tag']}",
        f"TripoSR (frozen)\n{best_t['tag']}",
    ]
    ch = [float(best_a["best_chamfer"]), float(best_p["micro_chamfer"]), float(best_t["micro_chamfer"])]
    fs = [float(best_a["best_fscore"]), float(best_p["micro_fscore"]), float(best_t["micro_fscore"])]
    colors = [C_ATLAS, C_POINTE, C_TRIPOSR]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, vals, title, better in (
        (axes[0], ch, "Chamfer (squared units)", "lower is better"),
        (axes[1], fs, "F-score @ τ=0.001", "higher is better"),
    ):
        bars = ax.barh(labels, vals, color=colors, height=0.55)
        ax.invert_yaxis()
        for b, v in zip(bars, vals):
            ax.text(b.get_width() * 1.02, b.get_y() + b.get_height() / 2,
                    f"{v:.4f}", va="center", fontsize=10)
        ax.set_title(f"{title}\n({better})", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
        ax.set_xlim(0, max(vals) * 1.22)
    fig.suptitle("Headline: best config per model, shared protocol (260 test objects)",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "fig_headline.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_ablation(atlas):
    fam_colors = {
        "baseline": "#718096", "views": "#2b6cb0", "topology": "#2f855a",
        "training": "#805ad5", "loss": "#b7791f", "seed": "#c53030",
    }
    rows = sorted(atlas, key=lambda r: float(r["best_chamfer"]))
    runs = [r["run"] for r in rows]
    ch = [float(r["best_chamfer"]) for r in rows]
    fs = [float(r["best_fscore"]) for r in rows]
    cols = [fam_colors[family(r["run"])] for r in rows]
    y = np.arange(len(rows))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 8.6), sharey=True)
    axes[0].barh(y, ch, color=cols, height=0.62)
    axes[1].barh(y, fs, color=cols, height=0.62)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(runs, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].axvline(0.0572, color="black", ls="--", lw=1, label="faithful baseline 0.057")
    axes[1].axvline(0.0709, color="black", ls="--", lw=1, label="faithful baseline 0.071")
    axes[0].set_xlabel("best Chamfer (squared) — lower is better")
    axes[1].set_xlabel("best F-score @0.001 — higher is better")
    axes[0].legend(loc="lower right", fontsize=8)
    axes[1].legend(loc="lower right", fontsize=8)
    for ax, vals in zip(axes, (ch, fs)):
        for i, v in enumerate(vals):
            ax.text(v * 1.02, y[i], f"{v:.4f}", va="center", fontsize=7)
        ax.spines[["top", "right"]].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in fam_colors.values()]
    fig.legend(handles, fam_colors.keys(), loc="lower center", ncol=6, fontsize=9,
               frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("AtlasNet: every run, sorted by Chamfer — which knob mattered", fontsize=12)
    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
    fig.savefig(OUT / "fig_atlasnet_ablation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_best_vs_final(atlas):
    fig, ax = plt.subplots(figsize=(6.2, 5.4))
    for r in atlas:
        b, f = float(r["best_chamfer"]), float(r["final_chamfer"])
        c = C_ATLAS if family(r["run"]) != "seed" else "#c53030"
        ax.scatter(b, f, color=c, s=55, zorder=3, alpha=0.85)
        if f / b > 3:
            ax.annotate(r["run"], (b, f), fontsize=7, xytext=(4, 4),
                        textcoords="offset points")
    lo = min(float(r["best_chamfer"]) for r in atlas) * 0.7
    hi = max(float(r["final_chamfer"]) for r in atlas) * 2
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="best = final (stable)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("best-epoch Chamfer")
    ax.set_ylabel("final-epoch Chamfer")
    ax.set_title("Overfitting: best vs final epoch (each dot = one run)\n"
                 "dots on the dashed line are stable; upper-right runs exploded")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(OUT / "fig_best_vs_final.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def fig_cross_objects():
    rows = read_csv(CROSS_CSV)
    objs = [r["category"] for r in rows]
    x = np.arange(len(rows))
    w = 0.26
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    data = {
        "AtlasNet (trained)": (C_ATLAS, [float(r["atlas_ch"]) for r in rows],
                               [float(r["atlas_f"]) for r in rows]),
        "Point-E (frozen)": (C_POINTE, [float(r["pointe_ch"]) for r in rows],
                             [float(r["pointe_f"]) for r in rows]),
        "TripoSR (frozen)": (C_TRIPOSR, [float(r["triposr_ch"]) for r in rows],
                             [float(r["triposr_f"]) for r in rows]),
    }
    for i, (name, (color, ch, fs)) in enumerate(data.items()):
        axes[0].bar(x + (i - 1) * w, ch, w, label=name, color=color)
        axes[1].bar(x + (i - 1) * w, fs, w, label=name, color=color)
    for ax, title, better in (
        (axes[0], "Chamfer (squared)", "lower is better"),
        (axes[1], "F-score @0.001", "higher is better"),
    ):
        ax.set_xticks(x)
        ax.set_xticklabels(objs, rotation=20, ha="right")
        ax.set_title(f"{title}\n({better})", fontsize=11)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].legend(fontsize=8)
    fig.suptitle("Per-object head-to-head (5 shared objects, same protocol)", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "fig_cross_objects.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    atlas = load_atlas_runs()
    pointe = read_csv(POINTE_CSV)
    triposr = read_csv(TRIPOSR_CSV)
    fig_headline(atlas, pointe, triposr)
    fig_ablation(atlas)
    fig_best_vs_final(atlas)
    fig_cross_objects()
    print(f"wrote 4 figures to {OUT}")


if __name__ == "__main__":
    main()
