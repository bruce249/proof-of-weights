"""Figures for the arXiv build. Every value is read from results/*.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent / "figures"
RES = ROOT / "results"

INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e4e3df"
HONEST = "#8a8984"
# reference categorical slots 1-4, fixed order: 8-bit, 4-bit, LoRA, fine-tune
C = {"quant_8bit": "#2a78d6", "quant_4bit": "#eb6834", "lora_merged": "#1baf7a", "ft_drift": "#eda100"}
NAME = {"quant_8bit": "8-bit", "quant_4bit": "4-bit (NF4)", "lora_merged": "LoRA (merged)", "ft_drift": "Full fine-tune"}
VARIANTS = list(C)
COL_W = 3.35  # single column, inches
FULL_W = 6.9


def style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.titlesize": 8.5,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.5,
        "axes.axisbelow": True,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
    })


def load(name):
    return json.loads((RES / name).read_text(encoding="utf-8"))


def fig_detection():
    """Last-hidden boolean vs per-layer binary score, per tamper class."""
    gate = load("phase_a_gate.json")
    pa = load("paper_analysis.json")["matched"]["separate_axes_binary"]
    last = [100 * gate["variants"][v]["detect_rate"] for v in VARIANTS]
    per = [100 * pa[v]["tpr_at_fpr_0.01"] for v in VARIANTS]
    x = np.arange(len(VARIANTS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(COL_W, 2.0))
    b1 = ax.bar(x - w / 2 - 0.01, last, w, color="white", edgecolor=INK2, hatch="////", linewidth=0.6,
                label="Last-hidden TOPLOC check (2% held-out FPR)")
    b2 = ax.bar(x + w / 2 + 0.01, per, w, color=[C[v] for v in VARIANTS], linewidth=0,
                label="Per-layer profile (1% FPR, out-of-fold)")
    for bars, vals in ((b1, last), (b2, per)):
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.0f}", ha="center", va="bottom", fontsize=6.5, color=INK)
    ax.set_xticks(x, [NAME[v] for v in VARIANTS])
    ax.set_ylim(0, 108)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("Tampered sequences detected (%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower left", bbox_to_anchor=(0, 1.0), frameon=False, ncol=1, handlelength=1.2, borderaxespad=0.1)
    fig.savefig(OUT / "detection.pdf")
    plt.close(fig)


def fig_profiles():
    """Mean per-layer feature, 3 honest axes + 4 tamper classes, matched text."""
    lm = load("paper_analysis.json")["matched"]["layer_means"]
    feats = [("idx_mismatch", "Top-$k$ index mismatch"), ("mantissa_mean", r"Mean $|\Delta|$ mantissa ($\times10^5$)")]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.2))
    L = np.arange(28)
    honest = [("clean_batch4", "-", "Honest: batch 4"), ("clean_padded_batch_length", ":", "Honest: padded batch"),
              ("clean_attn_eager", "--", "Honest: eager attention")]
    for ax, (key, title) in zip(axes, feats):
        scale = 1e-5 if key == "mantissa_mean" else 1.0
        for src, ls, lab in honest:
            ax.plot(L, np.array(lm[src][key]) * scale, color=HONEST, ls=ls, lw=1.2, label=lab)
        for v in VARIANTS:
            ax.plot(L, np.array(lm[v][key]) * scale, color=C[v], lw=1.6, label=NAME[v])
        ax.set_title(title, loc="left")
        ax.set_xlabel("Decoder layer")
        ax.set_xlim(0, 27)
        ax.set_xticks([0, 7, 14, 21, 27])
    h, l = axes[0].get_legend_handles_labels()
    fig.tight_layout(w_pad=2.0)
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.5, -0.07), frameon=False, ncol=7, fontsize=6.5,
               handlelength=2.0, columnspacing=1.0)
    fig.savefig(OUT / "profiles.pdf")
    plt.close(fig)


def fig_ablation():
    ab = load("paper_analysis.json")["matched"]["ablations"]
    rows = [
        ("all_layers_all_features", "All 28 layers, 4 features", "full"),
        ("last_layer_only", "Last layer only", "layer"),
        ("first_layer_only", "First layer only", "layer"),
        ("layers_0_13", "Layers 0-13", "layer"),
        ("layers_14_27", "Layers 14-27", "layer"),
        ("all_but_last_layer", "All but last layer", "layer"),
        ("idx_mismatch_only", "Index mismatch only", "feat"),
        ("exp_mismatch_only", "Exponent mismatch only", "feat"),
        ("mantissa_only", "Mantissa (mean, median) only", "feat"),
        ("checkpoints_1", "1 checkpoint (32 tokens)", "ckpt"),
        ("checkpoints_2", "2 checkpoints", "ckpt"),
        ("checkpoints_4", "4 checkpoints", "ckpt"),
    ]
    fig, ax = plt.subplots(figsize=(COL_W, 2.9))
    y = np.arange(len(rows))[::-1]
    vals = [100 * ab[k]["balanced_accuracy"] for k, _, _ in rows]
    eager = [100 * ab[k]["honest_flag_rate"]["clean_attn_eager"] for k, _, _ in rows]
    colors = [INK if g == "full" else "#2a78d6" for _, _, g in rows]
    ax.barh(y, vals, color=colors, height=0.62, linewidth=0)
    for yy, v, e in zip(y, vals, eager):
        ax.text(v - 0.6, yy, f"{v:.1f}", va="center", ha="right", fontsize=6.5, color="white")
        ax.text(103, yy, f"{e:.1f}", va="center", ha="right", fontsize=6.5, color=INK2, clip_on=False)
    ax.text(103, y[0] + 0.9, "Eager FPR (%)", fontsize=6.3, color=INK2, ha="right", clip_on=False)
    ax.set_yticks(y, [r[1] for r in rows])
    ax.set_xlim(50, 100)
    ax.set_xlabel("Five-class balanced accuracy (%)")
    ax.grid(axis="y", visible=False)
    for yy in (y[5] - 0.5, y[8] - 0.5):
        ax.axhline(yy, color=GRID, lw=0.8)
    fig.savefig(OUT / "ablation.pdf")
    plt.close(fig)


def fig_roc():
    pts = load("paper_analysis.json")["matched"]["roc_points"]
    auc = load("paper_analysis.json")["matched"]["separate_axes_binary"]
    fig, ax = plt.subplots(figsize=(COL_W, 2.2))
    for v in VARIANTS:
        ax.plot(100 * np.array(pts[v]["fpr"]), 100 * np.array(pts[v]["tpr"]), color=C[v], lw=1.5,
                label=f"{NAME[v]} (AUC {auc[v]['auc']:.3f})")
    ax.axvline(1, color=INK2, lw=0.6, ls="--")
    ax.text(1.3, 3, "1% FPR", fontsize=6.5, color=INK2)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 1, 10, 100], ["0", "1", "10", "100"])
    ax.set_ylim(0, 102)
    ax.set_xlabel("Honest sequences flagged (%)")
    ax.set_ylabel("Tampered detected (%)")
    ax.legend(loc="lower right", frameon=False)
    fig.savefig(OUT / "roc.pdf")
    plt.close(fig)


def fig_honest_noise():
    """Phase A honest-axis noise vs tamper medians on the exponent statistic."""
    audit = load("clean_perturbation_audit.json")["axes"]
    gate = load("phase_a_gate.json")["variants"]
    th = load("thresholds.json")
    names = ["Honest: padded batch", "Honest: batch 4", "8-bit", "4-bit (NF4)", "Full fine-tune", "LoRA (merged)", "Honest: eager attention"]
    vals = [audit["padded_batch_length"]["seq_stats_summary"]["exp_mismatch_max_mean"],
            audit["batch4"]["seq_stats_summary"]["exp_mismatch_max_mean"],
            gate["quant_8bit"]["exp_mismatches_max_median"], gate["quant_4bit"]["exp_mismatches_max_median"],
            gate["ft_drift"]["exp_mismatches_max_median"], gate["lora_merged"]["exp_mismatches_max_median"],
            audit["attn_eager"]["seq_stats_summary"]["exp_mismatch_max_mean"]]
    cols = [HONEST, HONEST, C["quant_8bit"], C["quant_4bit"], C["ft_drift"], C["lora_merged"], HONEST]
    fig, ax = plt.subplots(figsize=(COL_W, 1.9))
    y = np.arange(len(names))[::-1]
    ax.barh(y, vals, color=cols, height=0.6, linewidth=0)
    for yy, v in zip(y, vals):
        ax.text(v + 0.8, yy, f"{v:.1f}", va="center", fontsize=6.5)
    ax.axvline(th["calibrated"]["T_exp"], color=INK, lw=0.8, ls="--")
    ax.text(th["calibrated"]["T_exp"] + 0.6, y[0] + 0.55, r"calibrated $T_{\exp}$", fontsize=6.3)
    ax.axvline(th["paper"]["T_exp"], color=INK2, lw=0.8, ls=":")
    ax.text(th["paper"]["T_exp"] - 0.6, y[0] + 0.55, r"published $T_{\exp}$", fontsize=6.3, ha="right", color=INK2)
    ax.set_yticks(y, names)
    ax.set_xlim(0, 56)
    ax.set_xlabel("Max exponent mismatches per sequence")
    ax.grid(axis="y", visible=False)
    fig.savefig(OUT / "noise.pdf")
    plt.close(fig)


def main():
    style()
    OUT.mkdir(parents=True, exist_ok=True)
    fig_detection()
    fig_profiles()
    fig_ablation()
    fig_roc()

    print(OUT)


if __name__ == "__main__":
    main()
