# Writing review (research-paper-writing skill)

Not part of the arXiv submission. Records the outline, self-review and claim-evidence check from the rewrite of `main.tex`.

## Story

- **Task:** check that an inference provider serves the committed open-weight model.
- **Challenge:** TOPLOC's last-layer check needs thresholds loose enough for honest kernel noise. At the last layer, eager-attention noise is as large as quantisation, so one threshold cannot both accept honest kernels and catch tampering.
- **Insight:** kernel noise and weight changes differ across depth. Kernel noise is early and fading, quantisation is flat, fine-tuning grows with depth (Fig. 1).
- **Contribution:** a per-layer disagreement profile with a linear classifier, which detects and names the modification. The on-chain record and cost measurement are secondary.
- **Boundary:** the classifier rejects unseen honest kernels.

## Templates used

| Section | Template |
|---|---|
| Abstract | Version 2: challenge → insight → contribution |
| Intro, part A | Version 2: application first |
| Intro, part B | Technical challenge v1: ZK, TEE and text tests → TOPLOC → threshold trade-off |
| Intro, part C | Pipeline v4: observation-driven, with the profile figure as the teaser (Fig. 1) |
| Method | Overview, then per module: motivation → design → advantage |
| Experiments | Setup → comparison with TOPLOC → main result → ablation → generalisation and failures → cost |
| Conclusion | Problem + idea → evidence → impact → limitations → next steps |

## Paragraph roles

| Paragraph | Role |
|---|---|
| Intro ¶1 | opening: task and why it matters |
| Intro ¶2 | challenge: prior methods and the TOPLOC trade-off |
| Intro ¶3 | challenge, with the technical reason (eager noise ≥ quantisation) |
| Intro ¶4 | method: observation → per-layer profile → classifier |
| Intro ¶5 | evidence |
| Intro ¶6 | contributions |
| §4.1, §4.2, §4.3 | each opens with its motivation, then design, then advantage |
| §5.2 | evidence: the baseline fails, with the reason |
| §5.3 | evidence: the main result |
| §5.4 | evidence: ablation |
| §5.5 | limitation: failure cases |
| Conclusion, limitations ¶ | limitation |

## Self-review (5 dimensions)

| # | Question | Status | Note |
|---|---|---|---|
| 1 | Contribution: new knowledge? | pass | Depth profile separates kernel noise from tampering; the last-layer check provably can't with mixed kernels (Tab. 2). |
| 1 | Non-obvious? | pass | A single-kernel setup would hide the problem; the recalibration row shows it. |
| 2 | Reproducible? | pass | Features defined, split, seeds, versions and code link given. |
| 2 | Terms consistent? | pass | "honest configuration", "modification", "profile", "last-hidden check" used throughout. |
| 3 | Strong vs baseline? | pass | 0% vs 90–100% detection on quantisation with both kernels. |
| 3 | Absolute performance? | pass | 99.3% balanced accuracy, 1% FPR operating point. |
| 3 | Consistent across settings? | pass | Matched, shared and pre-EOS all ≥ 98.8%. |
| 3 | Failures reported? | pass | Unseen kernel 100% flagged; eager FPR 6.5%. |
| 4 | Ablations for key choices? | pass | Layers, features and checkpoints. |
| 4 | Strong baselines? | needs new experiment | Only TOPLOC is compared. SVIP or text tests target model swaps; a reviewer may still ask. |
| 4 | Challenging enough? | needs new experiment | One 1.5B model. A 7–8B model is the most valuable addition. |
| 5 | Realistic setting? | needs revision (scoped) | Prefill on both sides, not decode-time per-layer proofs. Stated in limitations; no claim of a deployable proof. |
| 5 | Net benefit positive? | pass, with scope | Proof size grows up to 28×; stated. The benefit holds only for a known kernel set; stated. |

## Claim-evidence map

| Claim | Evidence | Status |
|---|---|---|
| Last-hidden check: 0% quantised, 55–60% fine-tuned at 2% FPR | `results/phase_a_gate.json`, `results/thresholds.json`, Tab. 2 row 1 | supported |
| Published thresholds give 23% FPR | `results/thresholds.json` | supported |
| Without eager: 100% detection, 4% FPR, 100% eager rejected | `results/phase_a_recalibrated.json`, Tab. 2 row 2 | supported |
| Eager noise ≈ 6× batch-shape, above 8-bit/4-bit | `results/clean_perturbation_audit.json`, Tab. 3 | supported |
| Depth profiles differ by source | `results/paper_analysis.json` (layer_means), Fig. 1 | supported |
| 99.3% balanced accuracy, honest configurations separate | `paper_analysis.json` (separate_axes), Tab. 4 | supported |
| 90–100% detection at 1% FPR, AUC 0.991 / 0.998 | `paper_analysis.json` (separate_axes_binary) | supported |
| Last layer 78.0%; layers 0–13 98.6%; layers 14–27 91.6% | `paper_analysis.json` (ablations) | supported |
| Shared text 99.1%; pre-EOS 98.8% | `paper_analysis.json`, `results/metrics.json` | supported |
| Unseen eager kernel 100% flagged | `paper_analysis.json` (leave_one_axis_out), Tab. 7 | supported |
| No-eager classifier: 0% batch FPR, 800/800 detected, 799 correct | `paper_analysis.json` (without_eager_training) | supported |
| L4: 50/50 clean, exponent mismatches 2–9 | `results/cross_hardware.json` | supported |
| 8-bit 4.9× and 4-bit 1.3× slower; 39–56% less memory | `results/cost_asymmetry.json` | supported |
| End-to-end slash recorded class 4 | `results/live_slash.json` | supported |

All claims in the paper map to a committed result file.

## Revision after external review (2026-10-04)

| Claim | Evidence | Status |
|---|---|---|
| Decode-time: 99.0% (98.4–99.6); ref decode 0% flagged; eager 12%; last layer 80.6%; prefill-trained transfer 99.2% | `results/decode_analysis.json` (decode_cv, decode_cv_last_layer_only, prefill_trained_on_decode) | supported |
| Kernel attack: eager evades per-layer on 77% (8-bit), 66% (4-bit); batch 4 detected 96–100% | `results/decode_analysis.json` (kernel_attack_per_layer), `results/kernel_attack_seq_stats.json` | supported |
| Training on attack: eager 8-bit 62%, others 98–100%, honest eager FPR 38% vs 12% | `results/decode_analysis.json` (kernel_attack_augmented) | supported |
| Table 3 tampered rows and TOPLOC range | `results/honest_seq_stats.json`, `results/variant_seq_stats.json`, TOPLOC §5.6 Table 2 | supported |
| Separate-setting fold interval 98.7–99.9 | `results/paper_analysis.json` (fold_balanced_accuracy) | supported |
