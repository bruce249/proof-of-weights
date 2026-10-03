# Decisions

Record underspecified choices here at the moment they are made. One line of rationale each.

## Locked before Day 1

- **EOS suppression:** `min_new_tokens` equals `tokens_per_generation` so arrays are not ragged. Natural EOS position is still recorded. Limitation in the paper.
- **C7 FPR:** thresholds fit on 100 honest prompts; gate is realized FPR on a disjoint 100, accept if `< 3%`. Same-set pass is not a gate. 99th percentile from a few hundred samples has wide error bars; report realized FPR.
- **D6 first axis:** padded batch length (changes matmul shape under bf16). If inert, shuffle-in-batch, then `use_deterministic_algorithms`. Never `allow_tf32`.
- **k-NN:** `k=5`, unweighted. Logistic regression uses `class_weight=balanced`. Headline is balanced accuracy and per-class F1.
- **tamperClass IDs:** 0=clean, 1=quant_8bit, 2=quant_4bit, 3=lora_merged, 4=ft_drift, 255=TAMPERED.
- **D9 atol:** `1e-4` on z-scored feature vectors, 10 Phase B sequences.
- **Day-2 cut order (only if a gate forces it):** `shared` first, then `quant_8bit`. Never Phase A, `matched`, or `ft_drift`. 3-day sprint default is no cut.
- **If `shared` is cut:** paper must weaken the claim: signal may be partly output-distribution, not only weights.
- **E3:** if TASK-05 fails twice by end of the features day, binary `TAMPERED=255` that evening. No third feature attempt. Skip TASK-06.
- **SFT OOM:** Llama-3.2-1B for all variants. Do not turn `ft_drift` into LoRA.
- **Train data:** Alpaca (fallback Dolly). Never UltraChat (that is eval).

## Filled on the GPU

- Instance `i-0bf57fb51728022b` `g5.xlarge` us-east-1, public IP `100.52.191.96` (changes on Stop/Start). SSH user `ubuntu`, key `D:\toploc.pem`. AMI floor forced **128 GB gp3** root (50 GB rejected).
- Venv `~/pow-venv`: torch `2.11.0+cu128`, toploc `0.1.6`, transformers `5.17.0`. Post-pip `torch.cuda.is_bf16_supported() == True` on **NVIDIA A10G**.
- Dummy proof: official CPU bf16 activations → `build_proofs_bytes(..., decode_batching_size=3, topk=4)` returns `list[bytes]`. **CUDA tensors segfault** in `ProofPoly.from_points_tensor`. All later proofs must `.cpu()` activations first. Do not rebuild toploc.
- `hf_revision`: `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` (`Qwen/Qwen2.5-1.5B-Instruct`). `model.model.layers` has **28** layers, hidden **1536**.
- `tokens_per_generation` / `n_checkpoints` after length probe: **256 / 8**. Seed `20260923`. 20 long-answer UltraChat prompts, no EOS suppress, `max_new_tokens=256`. 19/20 ran to the cap (`hit_eos=False`); one EOS at 134. Observed median 256 is right-censored → true median **> 256**. Keep 256. Do not re-filter.
- `seeds.prompt_sampling`: 20260923
- `seeds.lora`: 202609231
- `seeds.sft`: 202609232
- `seeds.cv_folds`:
- `seeds.calib_val_split`: index 0-99 calibration, 100-199 validation
- **TASK-03 seeds:** `seeds.lora=202609231` and `seeds.sft=202609232`, distinct int32 values independent of prompt sampling `20260923`, so LoRA data order and full-SFT data order are reproducible and not the same shuffle.
- **TASK-03 LoRA extras:** attention QKV means `q_proj`/`k_proj`/`v_proj`; `lora_alpha=32`, dropout `0.05`, lr `2e-4`, `adamw_torch`, constant schedule. Config locks lr `1e-5` and `adamw_bnb_8bit` only for full SFT. Both runs use microbatch 1, accumulation 8, `max_seq_length=512`, 2400 Alpaca train rows, sequential order, and gradient checkpointing. TF32 stays off.
- **TASK-03 quant load:** transformers 5.17 `from_pretrained` has no `load_in_8bit`/`load_in_4bit` kwargs, so the same flags go through `BitsAndBytesConfig(load_in_8bit=True)` and `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=bfloat16)`.
- **TASK-06 z-score, two paths, do not conflate:** (1) `results/feature_zscore_stats.json` is a population per-layer mean/std over the whole table, written once by `extract_deltas.py` for the `z_*` inspection columns only — not used by CV or the saved classifier. (2) CV and the saved `artifacts/classifier.pkl` each fit their own mean/std: CV refits per fold from that fold's training prompts only (`evaluate.py::_scale`); the saved artifact fits once over all matched all-checkpoint rows (`train_classifier.py`, `zscore_mean`/`zscore_std`) so the live validator can reuse those exact numbers. `flatten_zscored` (misleadingly named) only flattens raw per-layer features into one vector — it does not z-score.
- **TASK-06 CV:** `GroupKFold` 5 folds by `prompt_id`, shuffle off. One vector per prompt is the mean of checkpoint rows, then flattened, then per-fold z-scored. Logistic regression `random_state=20260923`. k-NN stays `k=5`, unweighted. Primary saved model is logistic regression, chosen before the run because it is the interpretable model, not because it scored higher.
- **TASK-06 classes:** labels 0–4. Clean is the three honest Phase B axes pooled. Fold z-score uses training prompts only.
