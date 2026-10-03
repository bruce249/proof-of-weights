"""Render paper/proof-of-weights.md as a PDF. Numbers match the markdown draft."""

from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
import struct

from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image,
)

OUT = Path(__file__).resolve().parent / "proof-of-weights.pdf"
CHARTS = Path(__file__).resolve().parent / "charts"
FIGS = Path(__file__).resolve().parents[1] / "figures"


def styles():
    base = getSampleStyleSheet()
    base.add(ParagraphStyle(
        "PaperTitle", parent=base["Title"], fontName="Times-Bold", fontSize=16,
        leading=20, alignment=TA_LEFT, spaceAfter=8, textColor=colors.HexColor("#1a1a1a"),
    ))
    base.add(ParagraphStyle(
        "Kicker", parent=base["Normal"], fontName="Times-Italic", fontSize=10,
        leading=13, textColor=colors.HexColor("#444444"), spaceAfter=14,
    ))
    base.add(ParagraphStyle(
        "H", parent=base["Heading2"], fontName="Times-Bold", fontSize=12.5,
        leading=16, spaceBefore=12, spaceAfter=6, textColor=colors.HexColor("#1a1a1a"),
    ))
    base.add(ParagraphStyle(
        "Body", parent=base["Normal"], fontName="Times-Roman", fontSize=10.5,
        leading=14, alignment=TA_LEFT, spaceAfter=8,
    ))
    base.add(ParagraphStyle(
        "BulletBody", parent=base["Normal"], fontName="Times-Roman", fontSize=10.5,
        leading=14, alignment=TA_LEFT,
    ))
    base.add(ParagraphStyle(
        "Cell", parent=base["Normal"], fontName="Times-Roman", fontSize=9.5,
        leading=12, alignment=TA_LEFT,
    ))
    base.add(ParagraphStyle(
        "CellHead", parent=base["Normal"], fontName="Times-Bold", fontSize=9.5,
        leading=12, alignment=TA_LEFT,
    ))
    base.add(ParagraphStyle(
        "Caption", parent=base["Normal"], fontName="Times-Italic", fontSize=9,
        leading=12, textColor=colors.HexColor("#444444"), spaceBefore=2, spaceAfter=10,
    ))
    return base


def table(headers, rows, col_widths):
    s = styles()
    data = [[Paragraph(h, s["CellHead"]) for h in headers]]
    for row in rows:
        data.append([Paragraph(c, s["Cell"]) for c in row])
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#f2f0ea")),
        ("FONTNAME", (0, 0), (-1, 0), "Times-Bold"),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#cccccc")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 6),
        ("RIGHTPADDING", (0, 0), (-1, -1), 6),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))
    return t


def bullets(items):
    s = styles()
    flow = []
    for item in items:
        flow.append(Paragraph("&#8211;  " + item, s["BulletBody"]))
        flow.append(Spacer(1, 3))
    flow.append(Spacer(1, 4))
    return flow


def png_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        handle.read(16)
        return struct.unpack(">II", handle.read(8))


def figure(path: Path, caption: str, width: float) -> list:
    s = styles()
    px_w, px_h = png_size(path)
    image = Image(str(path), width=width, height=width * px_h / px_w)
    image.hAlign = "CENTER"
    return [image, Paragraph(caption, s["Caption"])]


def footer(canvas, doc):
    canvas.saveState()
    canvas.setFont("Times-Roman", 8)
    canvas.setFillColor(colors.HexColor("#666666"))
    canvas.drawString(0.9 * inch, 0.5 * inch, "Proof of Weights - draft from committed result files")
    canvas.drawRightString(7.6 * inch, 0.5 * inch, str(doc.page))
    canvas.restoreState()


def main():
    s = styles()
    doc = SimpleDocTemplate(
        str(OUT), pagesize=letter,
        leftMargin=0.9 * inch, rightMargin=0.9 * inch,
        topMargin=0.8 * inch, bottomMargin=0.75 * inch,
        title="Proof of Weights",
    )
    usable = 6.7 * inch
    story = [
        Paragraph("Proof of Weights: per-layer attribution and an optimistic slash on TOPLOC", s["PaperTitle"]),
        Paragraph("Draft from the committed result files. Numbers below are copied from those files.", s["Kicker"]),
        Paragraph("Abstract", s["H"]),
        Paragraph(
            "Ong et al. (2025) note that detecting slight gradient updates in fine-tuned models is "
            "significantly harder than detecting model swaps, and leave it as future work. We extend "
            "their top-k activation scheme from the last hidden layer to a per-layer profile on "
            "Qwen2.5-1.5B-Instruct. The last-hidden proof does not reproduce their detection claim on "
            "8-bit or 4-bit weights. The per-layer profile does separate clean, 8-bit, 4-bit, merged LoRA, "
            "and a small full fine-tune under grouped cross-validation. We add the stake and slash their "
            "protocol-agnostic primitive does not have. On the A10G used here, the quantized servers are "
            "slower than bf16, so the measured time-cost stake floor is zero.",
            s["Body"],
        ),
        Paragraph("Setup", s["H"]),
        *bullets([
            "Model Qwen/Qwen2.5-1.5B-Instruct, revision 989aa7980e4cf806f80c7fef2b1adb7bc71aa306, 28 layers, hidden size 1536 (results/environment.json).",
            "Library toploc 0.1.6. Proofs are built from CPU bf16 activations. CUDA activations are not used.",
            "200 UltraChat eval prompts. Length 256, checkpoint every 32 tokens, EOS suppressed, natural EOS stored. A 20-prompt probe hit the cap on 19 prompts, so 256 was kept (results/length_probe.json).",
            "Training text for LoRA and the full fine-tune is Alpaca, 2400 rows. It is not UltraChat.",
            "Hardware: NVIDIA A10G (g5.xlarge) for the main run. NVIDIA L4, compute capability 8.9, for 50 honest sequences after the A10G was stopped.",
            "Honest noise used for calibration and for the clean class: eager attention, batch 4, and padded batch length. allow_tf32 was not used.",
        ]),
        Paragraph("Last-hidden proofs", s["H"]),
        Paragraph(
            "Thresholds are the 99th percentile of honest exponent mismatch, mantissa mean, and mantissa "
            "median on prompts 0-99, three honest axes. A sequence fails if any statistic exceeds its "
            "threshold. Held-out prompts 100-199 give false-positive rate 0.02, under the 0.03 cap "
            "(results/thresholds.json). Calibrated limits are T_exp 46.02, T_mean 25.80, T_median 20.0. "
            "The paper's limits {38, 10, 8} give held-out false-positive rate 0.23 and are reported, not used.",
            s["Body"],
        ),
        *figure(
            CHARTS / "fpr.png",
            "Figure 1. Held-out honest prompts. Calibrated limits stay under the 3% cap. The published TOPLOC limits flag 23%.",
            6.5 * inch,
        ),
        Paragraph(
            "Detection on 200 prompts, variant decode proofs checked by a baseline prefill (results/phase_a_gate.json):",
            s["Body"],
        ),
        table(
            ["Variant", "Detect rate", "Median max exponent mismatch"],
            [
                ["quant_8bit", "0.00", "9"],
                ["quant_4bit", "0.00", "20"],
                ["lora_merged", "0.595", "43.5"],
                ["ft_drift", "0.545", "41"],
            ],
            [2.2 * inch, 1.6 * inch, 2.9 * inch],
        ),
        Spacer(1, 8),
        Paragraph("The gate requires every variant at or above 0.95. It fails. passed is false.", s["Body"]),
        *figure(
            CHARTS / "detection.png",
            "Figure 2. Last-hidden detection on 200 prompts. The dashed line is the 95% gate. 8-bit and 4-bit sit at zero.",
            6.5 * inch,
        ),
        Paragraph("Per-layer profiles", s["H"]),
        Paragraph(
            "Each checkpoint and layer contributes four numbers: top-k index mismatch, exponent mismatch "
            "count, and the mean and median absolute mantissa difference. Vectors are the mean over "
            "checkpoints, flattened to 112 dimensions, then z-scored inside each GroupKFold training split "
            "(5 folds, grouped by prompt, shuffle off). The saved classifier uses one z-score fit on all "
            "matched rows. Live and offline vectors on 10 quant_4bit sequences match with max absolute "
            "difference 0 (results/feature_path_parity.json).",
            s["Body"],
        ),
        Paragraph(
            "Primary model is logistic regression with class_weight balanced and random_state 20260923, "
            "chosen before the run. k-NN is k=5, unweighted.",
            s["Body"],
        ),
        Paragraph("Matched, all checkpoints, 200 prompts per class (results/metrics.json):", s["Body"]),
        *bullets([
            "Logistic regression balanced accuracy 0.995, std 0.0035, 95% interval 0.991-0.999.",
            "k-NN balanced accuracy 0.988, std 0.0045.",
            "Out-of-fold logistic counts, rows true class clean / 8-bit / 4-bit / LoRA / ft_drift: clean 198, 1, 1, 0, 0; 8-bit 1, 199, 0, 0, 0; 4-bit 0, 0, 200, 0, 0; LoRA 0, 1, 1, 198, 0; ft_drift 0, 0, 0, 0, 200.",
            "ft_drift F1 is 1.0. It is not confused with clean.",
        ]),
        Paragraph(
            "The same logistic regression on matched pre-EOS rows is 0.988. On shared text, all checkpoints, "
            "it is 0.998. On shared pre-EOS rows it is 0.988. Clean index mismatch averages 0.0316, not zero. "
            "Layer means: 8-bit is small and flat, 4-bit is higher and flatter than the trained variants, "
            "LoRA and ft_drift rise with depth and spike at layer 27 (results/attribution_status.json, "
            "figures/layer_profiles.png).",
            s["Body"],
        ),
        *figure(
            FIGS / "layer_profiles.png",
            "Figure 3. Mean per-layer delta on matched text. LoRA and fine-tune rise with depth and spike at layer 27. 8-bit stays near the clean line.",
            6.5 * inch,
        ),
        *figure(
            FIGS / "confusion_matrix.png",
            "Figure 4. Out-of-fold counts. Rows are the true class. Fine-tune drift is not called clean.",
            6.5 * inch,
        ),
        Paragraph("Enforcement", s["H"]),
        Paragraph(
            "ProofOfWeights keeps a stake, a proof hash, a challenge bond, and a uint8 tamper class. "
            "Verification is off-chain. Only the validator address may resolve a challenge. An upheld "
            "challenge zeroes the stake, splits the stake token, and appends the class.",
            s["Body"],
        ),
        Paragraph(
            "The recorded slash is local Anvil, chain id 31337, contract "
            "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512. Provider 1 served ft_drift on prompt 3. "
            "Exponent mismatch 57 exceeded 46.02. The classifier returned class 4. Resolve transaction "
            "0x6acf4f1e5fdcd88138307ed744a13a108108082a15d0f63b9b5dfb5a6cc6dd4d. Stake after is 0. "
            "This is one prompt that the last-hidden check did catch. The 200-prompt detect rate for "
            "that variant is 0.545.",
            s["Body"],
        ),
        Paragraph("Cost", s["H"]),
        Paragraph(
            "Serving cost was measured on the A10G, three requests of 256 tokens after one excluded warmup "
            "(results/cost_asymmetry.json). The dollar rate is the plan's on-demand price, $1.006 per hour, "
            "applied to measured seconds.",
            s["Body"],
        ),
        table(
            ["Mode", "Tokens/s", "Seconds", "Peak VRAM"],
            [
                ["bf16", "38.4", "6.66", "3183 MiB"],
                ["quant_8bit", "7.9", "32.3", "1936 MiB"],
                ["quant_4bit", "29.6", "8.66", "1387 MiB"],
            ],
            [1.7 * inch, 1.5 * inch, 1.5 * inch, 2.0 * inch],
        ),
        Spacer(1, 8),
        Paragraph(
            "Validator prefill mean is 0.046 s, which is 0.0069 of the bf16 autoregressive time. Seconds "
            "saved per request are -25.7 for 8-bit and -2.0 for 4-bit. With one proof as the epoch, the "
            "stake that makes honesty dominate is $0 at sampling rates 0.01, 0.05, and 0.10, because there "
            "is no time saving. Peak memory does drop.",
            s["Body"],
        ),
        *figure(
            CHARTS / "serving.png",
            "Figure 5. Three timed requests of 256 tokens on the A10G, warmup excluded. bf16 is fastest. Quantized modes use less memory.",
            6.5 * inch,
        ),
        Paragraph("Cross-hardware", s["H"]),
        Paragraph(
            "Fifty baseline sequences on an NVIDIA L4, bf16 supported, compute capability 8.9 "
            "(results/cross_hardware.json). The L4 verified its own decode proofs by an L4 prefill: 0 of 50 "
            "flagged, exponent mismatch from 2 to 9, median 5. The A10G classifier, given L4 prefill top-k "
            "against the stored A10G baseline top-k on the same tokens, predicted clean for all 50. The "
            "proof false-positive rate is not an A10G-proof versus L4-activation comparison. Those proof "
            "bytes were not copied off the A10G.",
            s["Body"],
        ),
        Paragraph("Claim, with the gate failure kept", s["H"]),
        Paragraph(
            "The per-layer profile attributes the modification among the five classes above. The last-hidden "
            "boolean does not detect 8-bit or 4-bit here, and it detects LoRA and the fine-tune on only about "
            "half of the prompts. The slash contract records a class when a check does fire. The time-cost "
            "argument for a large stake is not supported on this GPU: quantized decoding was slower.",
            s["Body"],
        ),
        Paragraph("Limitations", s["H"]),
        Paragraph(
            "Trusted validator. Speculative decoding is out of scope. EOS suppression. One 1.5B model family. "
            "L4 sample size 50, and the L4 proof check is self-consistency. Local Anvil, not a public testnet. "
            "No claim that quantization is globally slower; the timing is this A10G, this model, batch size 1, "
            "greedy decode.",
            s["Body"],
        ),
    ]
    # usable width is documented so a later edit does not silently overflow the tables
    assert sum([2.2, 1.6, 2.9]) * inch <= usable + 1
    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    print(OUT)


if __name__ == "__main__":
    main()
