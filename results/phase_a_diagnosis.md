# Phase A gate diagnosis (hard-gate table, "TASK-04 detect": diagnose on failure)

**Gate result:** FAILED. `results/phase_a_gate.json` — detect rates
`quant_8bit=0%`, `quant_4bit=0%`, `lora_merged=59.5%`, `ft_drift=54.5%`,
against thresholds `T_exp=46.02, T_mean=25.80, T_median=20.0` calibrated in
`results/thresholds.json`.

## Cause

Calibration pools three honest noise axes (`attn_eager`, `batch4`,
`padded_batch_length`) into one 100-prompt calibration set, then takes the
99th percentile across all three pooled. `attn_eager` (eager vs sdpa
attention kernel) is far noisier than the other two axes:

`results/clean_perturbation_audit.json`, per-axis `seq_stats_summary`:

| Axis | exp_mismatch_max_mean | mant_mean_max_mean |
|---|---|---|
| attn_eager | 25.55 | 14.22 |
| batch4 | 4.35 | 1.35 |
| padded_batch_length | 4.31 | 1.37 |

Pooling drags the 99th-percentile thresholds up to cover `attn_eager`'s
tail, so they are loose for the other two axes and loose relative to the
actual tamper deltas. `quant_8bit` and `quant_4bit` sit at or below the
`attn_eager` noise floor on the same statistics — that is the 0% detect
rate. This is not a broken hook: `data/features_matched.parquet` (the same
proof-comparison statistics, per layer per checkpoint) confirms the
ordering structurally:

| Class | median exp_mismatch (per-prompt max, all layers/ckpts) |
|---|---|
| honest `attn_eager` | 66.0 |
| `lora_merged` | 28.0 |
| `ft_drift` | 27.0 |
| `quant_4bit` | 22.0 |
| `quant_8bit` | 10.0 |
| honest `batch4` | 5.0 |
| honest `padded_batch_length` | 4.0 |

Honest `attn_eager` noise medians sit *above* every tamper class on this
statistic. `batch4` and `padded_batch_length` sit well below all four
tamper classes. `attn_eager` alone is why the pooled threshold cannot
separate `quant_8bit`/`quant_4bit` from honest noise.

Caveat: this table is a CPU-only proxy built from the per-layer feature
table (`fingerprint.extract_deltas`), not an exact replay of the
`toploc.VerificationResult` objects `task02.py`/`task04.py` computed on the
GPU (those per-prompt raw stats were not persisted, only the aggregates in
`clean_perturbation_audit.json`, `thresholds.json`, `phase_a_gate.json`).
The absolute values differ in scale from `phase_a_gate.json`'s
`exp_mismatches_max_median` column; only the *ordering* (attn_eager >
trained/quant variants > batch4/padded) is asserted here, and it matches
the median columns already in `phase_a_gate.json` closely enough (attn_eager
excluded there, since Phase A ran the gate under `sdpa` attention only — see
`results/phase_a_gate.json`'s `"attn": "sdpa"` field) to support the
diagnosis without a GPU re-run.

## Conclusion

**Do not retune the reported thresholds** (per the plan's locked rule). The
0%/0% miss is real and stays the headline. The mechanism is: pooling a
noisy honest axis (`attn_eager`) with two quiet ones (`batch4`,
`padded_batch_length`) into one calibration set produces thresholds too
loose for 8-bit/4-bit quantization deltas, which are smaller than
`attn_eager`'s own honest noise. An exact re-run of Phase A calibrated on
`batch4` + `padded_batch_length` only (dropping `attn_eager`) would need
GPU time and is future work, not a retune of the reported result.

## Downstream classifier is not affected the same way

TASK-06's classifier is a separate model trained on the same three honest
axes as three *distinct* per-prompt vectors (not pooled into thresholds).
Cross-validated with each axis kept separate (not averaged), logistic
regression balanced accuracy is 0.993 on `matched`/`all_checkpoints`. But
honest `attn_eager` traffic is still the weak point: 13 of 200 honest
`attn_eager` prompts (6.5%) are misclassified as a tamper class in that
run (8 called `quant_8bit`, 5 called `quant_4bit`, 0 called `lora_merged`
or `ft_drift`). `batch4` and `padded_batch_length` are clean: 199/200 and
199/200 respectively. This belongs in the paper's limitations: a provider
running eager attention looks statistically closer to 8-bit/4-bit
quantization than a provider running sdpa does.
