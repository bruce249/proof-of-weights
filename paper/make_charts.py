"""Charts for the PDF. Every bar is read from results/*.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = Path(__file__).resolve().parent / "charts"
INK = "#1c1917"
MUTED = "#78716c"
COLORS = {
    "quant_8bit": "#d6a35c",
    "quant_4bit": "#c2410c",
    "lora_merged": "#9f1239",
    "ft_drift": "#1d4ed8",
    "bf16": "#1c1917",
}


def _style():
    plt.rcParams.update({
        "font.family": "serif",
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "text.color": INK,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
    })


def detection() -> None:
    gate = json.loads((ROOT / "results" / "phase_a_gate.json").read_text(encoding="utf-8"))
    order = ["quant_8bit", "quant_4bit", "lora_merged", "ft_drift"]
    labels = ["8-bit", "4-bit", "LoRA", "fine-tune"]
    rates = [100 * gate["variants"][k]["detect_rate"] for k in order]
    fig, ax = plt.subplots(figsize=(6.6, 3.1), dpi=160)
    bars = ax.barh(labels[::-1], rates[::-1], color=[COLORS[k] for k in order][::-1], height=0.62)
    ax.axvline(95, color=INK, linestyle="--", linewidth=1)
    ax.text(95, 3.45, "95% gate", ha="center", va="bottom", fontsize=8, color=MUTED)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Prompts flagged, out of 200")
    ax.set_title("Last-hidden proof does not clear the gate", loc="left", fontsize=12, pad=10)
    for bar, rate in zip(bars, rates[::-1]):
        ax.text(rate + 1.5, bar.get_y() + bar.get_height() / 2, f"{rate:.1f}%", va="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "detection.png")
    plt.close(fig)


def fpr() -> None:
    th = json.loads((ROOT / "results" / "thresholds.json").read_text(encoding="utf-8"))
    labels = ["Calibrated\nlimits", "Published\nTOPLOC limits"]
    vals = [100 * th["realized_fpr_heldout"], 100 * th["paper_fpr_heldout"]]
    fig, ax = plt.subplots(figsize=(6.6, 2.8), dpi=160)
    bars = ax.bar(labels, vals, color=["#1c1917", "#c2410c"], width=0.55)
    ax.axhline(3, color=MUTED, linestyle="--", linewidth=1)
    ax.text(1.35, 3.4, "3% cap", fontsize=8, color=MUTED)
    ax.set_ylabel("Held-out honest prompts flagged")
    ax.set_ylim(0, 30)
    ax.set_title("Same honest set, two threshold choices", loc="left", fontsize=12, pad=10)
    for bar, val in zip(bars, vals):
        label_y = 6.2 if val < 8 else val + 1.1
        ax.text(bar.get_x() + bar.get_width() / 2, label_y, f"{val:.0f}%", ha="center", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "fpr.png")
    plt.close(fig)


def serving() -> None:
    cost = json.loads((ROOT / "results" / "cost_asymmetry.json").read_text(encoding="utf-8"))
    order = ["bf16", "quant_8bit", "quant_4bit"]
    labels = ["bf16", "8-bit", "4-bit"]
    speed = [cost["serving"][k]["tokens_per_sec_mean"] for k in order]
    mem = [cost["serving"][k]["peak_vram_mib"] for k in order]
    colors = [COLORS["bf16"], COLORS["quant_8bit"], COLORS["quant_4bit"]]
    fig, axes = plt.subplots(1, 2, figsize=(6.6, 3.0), dpi=160)
    for ax, values, title, ylab in (
        (axes[0], speed, "Tokens per second", "tokens / s"),
        (axes[1], mem, "Peak memory", "MiB"),
    ):
        bars = ax.bar(labels, values, color=colors, width=0.62)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel(ylab)
        top = max(values) * 1.18
        ax.set_ylim(0, top)
        for bar, val in zip(bars, values):
            label = f"{val:.1f}" if title.startswith("Tokens") else f"{val:.0f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + top * 0.02, label, ha="center", fontsize=8)
    fig.suptitle("On this A10G, quantized decode is slower", fontsize=12, x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "serving.png")
    plt.close(fig)


def main() -> None:
    _style()
    OUT.mkdir(parents=True, exist_ok=True)
    detection()
    fpr()
    serving()
    print(OUT)


if __name__ == "__main__":
    main()
