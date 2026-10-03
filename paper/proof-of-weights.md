# Proof of Weights: per-layer attribution and an optimistic slash on TOPLOC

Draft from the committed result files. Numbers below are copied from those files, not recalled.

## Abstract

Ong et al. (2025) note that detecting slight gradient updates in fine-tuned models is significantly harder than detecting model swaps, and leave it as future work. We extend their top-k activation scheme from the last hidden layer to a per-layer profile on Qwen2.5-1.5B-Instruct. The last-hidden proof does not reproduce their detection claim on 8-bit or 4-bit weights. The per-layer profile does separate clean, 8-bit, 4-bit, merged LoRA, and a small full fine-tune under grouped cross-validation. We add the stake and slash their protocol-agnostic primitive does not have. On the A10G used here, the quantized servers are slower than bf16, so the measured time-cost stake floor is zero.

## Setup

- Model `Qwen/Qwen2.5-1.5B-Instruct`, revision `989aa7980e4cf806f80c7fef2b1adb7bc71aa306`, 28 layers, hidden size 1536 (`results/environment.json`).
- Library `toploc==0.1.6`. Proofs are built from CPU bf16 activations. CUDA activations are not used.
- 200 UltraChat eval prompts. Length 256, checkpoint every 32 tokens, EOS suppressed, natural EOS stored. A 20-prompt probe hit the cap on 19 prompts, so 256 was kept (`results/length_probe.json`).
- Training text for LoRA and the full fine-tune is Alpaca, 2400 rows. It is not UltraChat.
- Hardware: NVIDIA A10G (g5.xlarge) for the main run. NVIDIA L4, compute capability 8.9, for 50 honest sequences after the A10G was stopped.
- Honest noise used for calibration and for the clean class: eager attention, batch 4, and padded batch length. `allow_tf32` was not used.

## Last-hidden proofs

Thresholds are the 99th percentile of honest exponent mismatch, mantissa mean, and mantissa median on prompts 0–99, three honest axes. A sequence fails if any statistic exceeds its threshold. Held-out prompts 100–199 give false-positive rate 0.02, under the 0.03 cap (`results/thresholds.json`). Calibrated limits are T_exp 46.02, T_mean 25.80, T_median 20.0. The paper's limits `{38, 10, 8}` give held-out false-positive rate 0.23 and are reported, not used.

Detection on 200 prompts, variant decode proofs checked by a baseline prefill (`results/phase_a_gate.json`):

| Variant | Detect rate | Median max exponent mismatch |
|---|---:|---:|
| quant_8bit | 0.00 | 9 |
| quant_4bit | 0.00 | 20 |
| lora_merged | 0.595 | 43.5 |
| ft_drift | 0.545 | 41 |

The gate requires every variant at or above 0.95. It fails. `passed` is false.

## Per-layer profiles

Each checkpoint and layer contributes four numbers: top-k index mismatch, exponent mismatch count, and the mean and median absolute mantissa difference. Vectors are the mean over checkpoints, flattened to 112 dimensions, then z-scored inside each GroupKFold training split (5 folds, grouped by prompt, shuffle off). The saved classifier uses one z-score fit on all matched rows. Live and offline vectors on 10 quant_4bit sequences match with max absolute difference 0 (`results/feature_path_parity.json`).

Primary model is logistic regression with `class_weight=balanced` and `random_state=20260923`, chosen before the run. k-NN is k=5, unweighted.

Matched, all checkpoints, 200 prompts per class (`results/metrics.json`):

- Logistic regression balanced accuracy 0.995, std 0.0035, 95% interval 0.991–0.999.
- k-NN balanced accuracy 0.988, std 0.0045.
- Out-of-fold logistic counts, rows true class clean / 8-bit / 4-bit / LoRA / ft_drift: clean 198, 1, 1, 0, 0; 8-bit 1, 199, 0, 0, 0; 4-bit 0, 0, 200, 0, 0; LoRA 0, 1, 1, 198, 0; ft_drift 0, 0, 0, 0, 200.
- ft_drift F1 is 1.0. It is not confused with clean.

The same logistic regression on matched pre-EOS rows is 0.988. On shared text, all checkpoints, it is 0.998. On shared pre-EOS rows it is 0.988. Clean index mismatch averages 0.0316, not zero. Layer means: 8-bit is small and flat, 4-bit is higher and flatter than the trained variants, LoRA and ft_drift rise with depth and spike at layer 27 (`results/attribution_status.json`, `figures/layer_profiles.png`).

## Enforcement

`ProofOfWeights` keeps a stake, a proof hash, a challenge bond, and a `uint8` tamper class. Verification is off-chain. Only the validator address may resolve a challenge. An upheld challenge zeroes the stake, splits the stake token, and appends the class.

The recorded slash is local Anvil, chain id 31337, contract `0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512`. Provider 1 served `ft_drift` on prompt 3. Exponent mismatch 57 exceeded 46.02. The classifier returned class 4. Resolve transaction `0x6acf4f1e5fdcd88138307ed744a13a108108082a15d0f63b9b5dfb5a6cc6dd4d`. Stake after is 0. This is one prompt that the last-hidden check did catch. The 200-prompt detect rate for that variant is 0.545.

## Cost

Serving cost was measured on the A10G, three requests of 256 tokens after one excluded warmup (`results/cost_asymmetry.json`). The dollar rate is the plan's on-demand price, $1.006 per hour, applied to measured seconds.

| Mode | Tokens/s | Seconds | Peak VRAM |
|---|---:|---:|---:|
| bf16 | 38.4 | 6.66 | 3183 MiB |
| quant_8bit | 7.9 | 32.3 | 1936 MiB |
| quant_4bit | 29.6 | 8.66 | 1387 MiB |

Validator prefill mean is 0.046 s, which is 0.0069 of the bf16 autoregressive time. Seconds saved per request are −25.7 for 8-bit and −2.0 for 4-bit. With one proof as the epoch, the stake that makes honesty dominate is $0 at sampling rates 0.01, 0.05, and 0.10, because there is no time saving. Peak memory does drop.

## Cross-hardware

Fifty baseline sequences on an NVIDIA L4, bf16 supported, compute capability 8.9 (`results/cross_hardware.json`). The L4 verified its own decode proofs by an L4 prefill: 0 of 50 flagged, exponent mismatch from 2 to 9, median 5. The A10G classifier, given L4 prefill top-k against the stored A10G baseline top-k on the same tokens, predicted clean for all 50. The proof false-positive rate is not an A10G-proof versus L4-activation comparison. Those proof bytes were not copied off the A10G.

## Claim, with the gate failure kept

The per-layer profile attributes the modification among the five classes above. The last-hidden boolean does not detect 8-bit or 4-bit here, and it detects LoRA and the fine-tune on only about half of the prompts. The slash contract records a class when a check does fire. The time-cost argument for a large stake is not supported on this GPU: quantized decoding was slower.

## Limitations

Trusted validator. Speculative decoding is out of scope. EOS suppression. One 1.5B model family. L4 sample size 50, and the L4 proof check is self-consistency. Local Anvil, not a public testnet. No claim that quantization is globally slower; the timing is this A10G, this model, batch size 1, greedy decode.
