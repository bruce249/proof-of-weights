# Proof of Weights — Agent execution plan

What I (the coding agent) will do, in order, to turn
`proof-of-weights-1-week-plan (1).md` into a working Devcon demo and an honest
arXiv preprint.

This file already incorporates C1–C10, D1–D9, and E1–E3 as **settled rules**.
Those letters are citations, not a changelog. Follow this file. Do not revert
to the original spec on extraction, hardware, gates, length, or the Day 3
fallback.

**Base paper:** TOPLOC (Ong et al., ICML 2025, arXiv:2501.16007)
**Compute:** AWS `g5.xlarge` (1× NVIDIA A10G 24GB, Ampere, CC 8.6) plus one
30-min `g6.xlarge` (1× NVIDIA L4 22GB, Ada, CC 8.9) for cross-hardware. Region
`us-east-1`, on-demand, not spot. **Never** `g4dn.xlarge` (T4, CC 7.5): T4
cannot do native bf16.
**Cash target:** ~$13 if every GPU instance is Stopped after the session
**Ceiling:** $100 / 20 GPU-hours
**GPU plan:** ~11.9 h A10G + 0.5 h L4 ≈ **$13**
**Status as of 19 Sep 2026:** AWS G/VT quota approved at **8 vCPU** in us-east-1.
`g5.xlarge` = 4 vCPU. `g6.xlarge` = 4 vCPU. **Never run both at once.** Stop
the A10G and confirm it is stopped before launching the L4.
Azure A10 quota is optional backup. Do not run Azure and AWS GPUs together.
Verify live `g5.xlarge` and `g6.xlarge` prices in the EC2 console before launch.

Unchanged on purpose: do not rebuild TOPLOC; do not tune features on reported
data; keep a negative `ft_drift` result; proof hash on-chain / blob off-chain;
bonded challenges + dispute window; state the trusted-validator caveat first;
split by prompt; interpretable classifier only; hard gates remain stop conditions.

---

## 0. What success is

Two coupled deliverables. A demo without honest numbers is not success.

| Deliverable | Done when |
|---|---|
| Hackathon demo | One tampered response → on-chain slash → stake drops, visible in 10 seconds. If TASK-05 separates: slash carries the diagnosed `tamperClass`. If the Day 3 binary fallback fires: slash carries `tamperClass = 255` (`TAMPERED`); the contract field is kept. Backup video exists. Pitch states a measured stake threshold from TASK-08b. Live features match offline features when a classifier exists. |
| arXiv preprint | If features separate: honest confusion matrix with **5-fold grouped-CV mean ± std**, **balanced accuracy** and per-class F1 as headline, `matched` vs `shared` (or a **weakened weight-vs-text claim** in the paper body if `shared` was cut — not only a limitation list). If the binary fallback fires: a **negative-result** paper — per-layer top-k profiles do not attribute tamper type at this scale — with calibration, held-out FPR, and L4 numbers still reported. Either way: pinned configs, realized FPR, clean-delta audit, L4 transfer, length path, TOPLOC §6.5 positioning, limitations first. |

I will not fabricate a metric. If a gate fails twice I write the failure into
`results/` and stop for you.

---

## 1. What I will not do

- Rebuild TOPLOC. I will `pip install toploc`.
- Detect speculative decoding (TOPLOC §6.2: undetectable). I will name it as a limitation.
- Build a decentralized validator set. One validator wallet is enough.
- Claim results beyond `Qwen2.5-1.5B-Instruct` (or the Llama 1B fallback if Qwen / SFT fails).
- Call the system cryptographically trustless. Verification is off-chain; the chain is optimistic.
- Tune features on the same data we then report, to force `ft_drift` to separate.
- Run `g5.xlarge` and `g6.xlarge` at the same time (8 vCPU quota).
- Launch any second GPU instance while the A10G is still Running.
- Use `g4dn.xlarge` / T4 for any job.
- Use `generate(..., output_hidden_states=True)` for per-layer extraction.
- Treat `clean` as “model vs itself, same kernels, same batch.”
- Compare activations across **different token sequences** as if they measured weights.
- Calibrate thresholds and then treat a pass on those **same** honest prompts as validation.
- Use `allow_tf32` as a perturbation (no-op under bf16).
- Train LoRA / SFT on UltraChat.
- Turn `ft_drift` into a LoRA run if SFT OOMs. Fall back to Llama-3.2-1B for **all** variants instead.
- Reimplement feature extraction inside `validator.py`.
- Copy a last-token-only hook into Phase B. Prefill must slice `positions`.
- Loop prompt-filtering with no exit. Prefer 128 tokens / 4 checkpoints when the median is 128–256.
- Spend Day 4 on a third feature variant after TASK-05 fails twice.
- Spend OpenAI/Anthropic/Gemini credits as a substitute.

---

## 2. Split of work: you vs me

| You (human, console / wallet) | Me (agent, repo + SSH if you give access) |
|---|---|
| Launch, Stop, Terminate `g5.xlarge` | Scaffold repo, `config.yaml`, all Python/Solidity |
| Launch/Stop `g6.xlarge` on Day 5 **only after** A10G is Stopped | TASK-01…11 code, gates, figures, paper skeleton |
| Hugging Face login / gated-model license | Foundry tests, deploy scripts, README, pitch outline |
| Testnet wallet + Base Sepolia ETH | `results/decisions.md` for every underspecified choice |
| Confirm every GPU instance is Stopped after the session | Write numbers only from committed artifacts |
| Record the backup demo video | |

---

## 3. Before any GPU hour (local, 30–60 min)

I will create this tree if it does not exist:

```
proof-of-weights/
├── config.yaml
├── requirements.txt          # empty until pip freeze on AWS
├── inference/
│   ├── hooks.py              # TopKRecorder mode="decode"|"prefill"
│   ├── build_variants.py
│   ├── generate.py           # Phase A: autoregressive + TOPLOC proofs
│   ├── prefill_deltas.py     # Phase B: teacher-forced prefill
│   └── providers/
│       ├── honest_provider.py
│       └── malicious_provider.py
├── fingerprint/
│   ├── extract_deltas.py     # single path used by train AND validator
│   ├── train_classifier.py
│   └── evaluate.py
├── contracts/                # forge init
│   ├── src/ProofOfWeights.sol
│   └── test/ProofOfWeights.t.sol
├── validator/
│   └── validator.py          # imports extract_deltas.py; no second implementation
├── dashboard/
├── results/
│   └── decisions.md
├── README.md
└── data/                     # created when we generate; never fake results
```

I will write `config.yaml` now, with placeholders, so every seed/hash is
committed at the moment we touch it:

```yaml
model: Qwen/Qwen2.5-1.5B-Instruct
fallback: meta-llama/Llama-3.2-1B-Instruct   # used for ALL variants if SFT OOMs
pin_revision: true                            # HF commit hash filled after first download
precision: bfloat16
require_native_bf16: true                     # assert torch.cuda.is_bf16_supported() after pip, every GPU job
topk: 128
n_prompts: 200
tokens_per_generation: 256                    # Day 1 length check may set 128
proof_interval_tokens: 32
# n_checkpoints = tokens_per_generation // proof_interval_tokens   # 8 or 4; never hardcode 8
min_new_tokens: <same as tokens_per_generation>
suppress_eos: true
# also record natural_eos_pos per sequence
paper_thresholds: {T_exp: 38, T_mean: 10, T_median: 8}   # reference only
calibration_prompts: 100
validation_prompts: 100                       # disjoint from calibration
target_fpr: 0.01
max_realized_fpr: 0.03                        # gate is realized FPR on held-out 100, not same-set pass
binary_tamper_class: 255                      # used if Day 3 fallback fires
tamper_class_ids:
  clean: 0
  quant_8bit: 1
  quant_4bit: 2
  lora_merged: 3
  ft_drift: 4
  TAMPERED: 255                               # binary fallback only
knn_k: 5                                      # unweighted; LR uses class_weight=balanced
feature_path_parity_atol: 1.0e-4              # max abs diff on z-scored vectors, 10 Phase B seqs
clean_perturbations:
  batch_sizes: [1, 4, 8]
  attn: [sdpa, eager]                         # plus flash_attention_2 if it imports
  reduction_order_try_first: padded_batch_length   # changes matmul shape under bf16
  reduction_order_fallback: [shuffle_in_batch, use_deterministic_algorithms]
  forbidden: [allow_tf32]
sft_lora:
  train_data: tatsu-lab/alpaca                # fallback databricks/databricks-dolly-15k; never UltraChat
  lora: {r: 16, targets: attention QKV, steps: 300, merge_and_unload: true}
  sft:
    steps: 300
    lr: 1.0e-5
    optim: adamw_bnb_8bit
    gradient_checkpointing: true
    per_device_train_batch_size: 1
    gradient_accumulation_steps: 8
    max_seq_length: 512
seeds: [prompt_sampling, lora, sft, cv_folds, calib_val_split]
chain: {framework: foundry, network: Base Sepolia, dispute_window_hours: 2, validator_sample_rate: 0.10}
day2_cut_order: [shared first, then quant_8bit; never Phase A, never matched, never ft_drift]
day3_fallback: if TASK-05 fails twice by end of Day 3 → binary TAMPERED; skip TASK-06
```

README stub will already say: verification is off-chain / optimistic, not
trustless; one trusted validator wallet; EOS is suppressed for fixed-length
checkpoints; length check may set 128 tokens / 4 checkpoints; we report pre-EOS
metrics when the 256 path is used; the binary fallback is pre-decided.

**You:** launch the A10G only after this scaffold exists, so Day 1 on AWS
is install + generate, not mkdir.

---

## 4. Week sequence

```
Day 1  AWS A10G   TASK-01 + 20-prompt length check + TASK-02
                  (baseline + clean noise + perturbation audit + 100/100 split)
                  then STOP
Day 2  AWS A10G   TASK-03 + TASK-04 Phase A + Phase B (cut line: Section 7)
                  then STOP
         laptop   start TASK-07 (parallel)
Day 3  CPU        If Phase B slipped: do it this morning.
                  TASK-05 (matched [+ shared if not cut]) + finish TASK-07
                  If TASK-05 fails twice: binary TAMPERED fallback that evening
Day 4  CPU        TASK-06 only if TASK-05 separated. Else skip classifier;
                  wire binary validator + start paper-as-negative-result
Day 5  AWS A10G   TASK-08 (incl. feature-path parity if a classifier exists)
                  + TASK-08b, then STOP g5.xlarge and confirm stopped
         AWS L4   only after A10G is Stopped: g6.xlarge ~30 min, 50 sequences,
                  bf16 assert after pip, then STOP
Day 6  CPU        TASK-09 + TASK-10 + paper skeleton
Day 7             TASK-11 pitch + freeze arXiv writeup
```

| Task | Hours | Why |
|---|---|---|
| TASK-01 | 0.5 | unchanged |
| Prompt-length pre-check | 0.1 | 20 prompts, median natural EOS |
| TASK-02 + clean + C7 split + audit | 2.25 | 100/100 honest split + perturbation audit |
| TASK-03 | 3.0 | 8-bit Adam + checkpointing is slower |
| TASK-04 Phase A | 2.0 | autoregressive proofs per variant |
| TASK-04 Phase B | 1.0 | teacher-forced prefill only (cheap) |
| TASK-08 + 08b + parity | 2.5 | plus feature-path parity when a classifier exists |
| C8 L4 | 0.5 | `g6.xlarge` ~$0.80/h → ~$0.40 |
| **Total** | **~11.9 A10G + 0.5 L4** | ~$13, inside $100 |

Hour rows are estimates; ~11.9 h includes slack. Do not launch a second GPU
to “save time.”

---

## 5. Extraction rule — applies to every generation and prefill

I will **not** call `output_hidden_states=True`. That accumulates every layer’s
full hidden state for every step and OOMs the 24 GB card.

I will use `TopKRecorder` with an explicit **`mode`**. Decode records the last
position on interval steps. Prefill records `h[:, pos, :]` at
`positions = [prompt_len + 32 * k for k in range(n_checkpoints)]` in **one**
forward pass. A last-token-only hook on prefill records the final token once
per layer and silently drops every earlier checkpoint. The token-ID assertion
does **not** catch that (it checks IDs at target positions, not which positions
the hook read).

`self.step` is incremented **once per decode step from the generation loop**,
not from inside the hook. Call `reset("decode")` or `reset("prefill", positions=...)`
**every prompt**.

```python
class TopKRecorder:
    def __init__(self, interval=32, k=128):
        self.interval, self.k = interval, k
        self.mode = None          # "decode" | "prefill"
        self.step, self.buf = 0, []
        self.positions = None     # prefill only

    def reset(self, mode, positions=None):
        assert mode in ("decode", "prefill")
        if mode == "prefill":
            assert positions is not None
        self.mode, self.positions = mode, positions
        self.step, self.buf = 0, []

    def _record(self, vec, ckpt_idx, layer_idx):
        vec = vec.float()
        _, idxs = torch.topk(vec.abs(), k=self.k, dim=-1)
        self.buf.append((ckpt_idx, layer_idx, idxs.cpu(), vec.gather(-1, idxs).cpu()))

    def make_hook(self, layer_idx):
        def hook(module, inputs, output):
            if self.mode is None:
                return
            h = output[0] if isinstance(output, tuple) else output   # [B, T, H]
            if self.mode == "decode":
                if self.step % self.interval != 0:
                    return
                self._record(h[:, -1, :], self.step // self.interval, layer_idx)
            else:
                for ckpt_idx, pos in enumerate(self.positions):
                    self._record(h[:, pos, :], ckpt_idx, layer_idx)
        return hook
```

Usage that must match the class (the hook never increments `step`):

```python
# Phase A — decode, per prompt
recorder.reset("decode")
for _ in range(tokens_per_generation):
    # one decode forward; hook records h[:, -1, :] when step % 32 == 0
    recorder.step += 1          # after the forward that produced this token; start at 0

# Phase B — prefill, per prompt
n_checkpoints = tokens_per_generation // 32          # 8 or 4
positions = [prompt_len + 32 * k for k in range(n_checkpoints)]
recorder.reset("prefill", positions=positions)
model(full_prompt_plus_generated)                    # one forward
# then assert: checkpoints per layer == len(positions)
```

- Verify `model.model.layers` with `print(model)` before relying on it (Qwen2 / Llama).
- `n_layers = len(model.model.layers)` (28 on Qwen2.5-1.5B, 16 on Llama-3.2-1B).
- Decode (Phase A): `reset("decode")`; last position on interval; increment `step`
  after each generated token’s forward, starting at 0.
- Prefill (Phase B): `reset("prefill", positions=...)`. Store `prompt_len` in
  `data/prompts.json` at generation time. Assert that token IDs at those
  positions in the prefill pass **equal** the tokens recorded during decode.
  **Also assert** Phase B produced **exactly `len(positions)` checkpoints per
  layer per prompt**. A run that recorded 1 instead of `n_checkpoints` fails
  loudly.
- Save **two** arrays in each `layer_topk.npz`: `values` (float32) and `indices`
  (int32), each shape `n_prompts × n_checkpoints × n_layers × 128`.
  `n_checkpoints` is 8 at 256 tokens or 4 at 128. Never hardcode 8 in shapes.
- `min_new_tokens` matches the chosen `tokens_per_generation`; suppress EOS.
  Also record the **natural** EOS position per prompt.
- Size check before five variants: two arrays × 200 × n_checkpoints × n_layers
  × 128 × 4 bytes. Fits 50 GB; I will confirm free disk first.

---

## 6. Day 1 — AWS A10G: bootstrap + length check + honest baseline + calibration

### TASK-01 (0.5h GPU)

On the box, after `nvidia-smi` shows **NVIDIA A10G**:

```bash
pip install -U toploc transformers accelerate bitsandbytes peft datasets scikit-learn web3
python -c "import toploc, torch; assert torch.cuda.is_bf16_supported(); print(torch.cuda.get_device_name(0))"
```

The bf16 assert runs **after** `pip install`. AMI torch is not the torch this
job will use. A pre-install assert does not count.

**I will:**

- Assert native bf16 on the installed torch. Fail loudly if not.
- Confirm `toploc.build_proofs_bytes` on a dummy bf16 tensor returns bytes.
- `pip freeze > requirements.txt`
- Write GPU name, driver, CUDA, `bf16_supported=true` into `results/environment.json`
- Pin the Qwen HF revision hash into `config.yaml`
- Print the model and confirm the decoder-layer path for hooks

**Gate:** dummy proof returns bytes. If it fails, fix install. Do not generate data.

### Prompt-length pre-check (0.1h) — bounded, no filter loop

On **20** UltraChat candidates (explanations, step-by-step, lists), generate
**without** EOS suppression and record natural EOS positions.

**Decision (this order; one retry max):**

| Probe median natural EOS | Action |
|---|---|
| **> 256** | Keep `tokens_per_generation = 256` (`n_checkpoints = 8`). |
| **128–256** | Set `tokens_per_generation = 128` (`n_checkpoints = 4`). **Do not** re-filter. Fewer checkpoints is cheaper than a biased prompt set, and every checkpoint stays pre-EOS, which **removes** the length confound instead of only measuring it. Halves per-prompt samples — note the C10 CI impact in `decisions.md`. |
| **< 128** | Re-filter toward long-answer prompts **once**. If still < 128, keep 128 tokens and state the short-answer limitation. **No unbounded retry loop.** |

Record the path taken and the final `tokens_per_generation` / `n_checkpoints`
in `decisions.md` and the paper’s experimental setup. Update `config.yaml`.

Commit `data/prompts.json` after this decision. Each record includes
`prompt_id`, text, `prompt_len` (token count of the prompt). After the main
run, each record also gets `natural_eos_pos` even though EOS was suppressed.

### TASK-02 (2.25h GPU) — baseline + clean noise + audit + disjoint thresholds

**I will implement `inference/generate.py` and `inference/hooks.py` to:**

1. Load the pinned model in bf16. Assert `torch.cuda.is_bf16_supported()`
   (installed torch, post-pip).
2. Use the committed 200 UltraChat prompts.
3. Autoregressive generate `tokens_per_generation` tokens/prompt with KV cache,
   `min_new_tokens` matching that value, EOS suppressed; record `natural_eos_pos`
   and `prompt_len`.
4. Every 32 tokens: TOPLOC proof from the **last** hidden state (`topk=128`).
5. At the same checkpoints: per-layer top-128 **values and indices** via
   `TopKRecorder` in **decode** mode.

**`clean` is not “model vs itself,” and tf32 is not a perturbation.**
I will re-run the unmodified model with at least three of:

- batch size 1 vs 4 vs 8
- attention: `sdpa` vs `eager`, plus `flash_attention_2` if it imports
- a **bf16-real** reduction-order change. Try **padded batch length** first
  (changes matmul shape under bf16). If that axis is inert in the audit, swap
  to shuffle-in-batch, then `torch.use_deterministic_algorithms(True)` vs `False`.

**Not** `torch.backends.cuda.matmul.allow_tf32` (inert under bf16).

Label **all** of these `clean`. Then run each perturbation **individually** and
write delta magnitudes to `results/clean_perturbation_audit.json`. Any
perturbation whose delta is **exactly zero** is inert: swap it out and re-audit.
The “mean clean delta ≠ 0” gate is necessary but not sufficient; the audit
is what proves three real axes of noise.

**Calibrate and validate on disjoint honest sets.**

1. **Calibration set:** 100 honest prompts under the kept perturbations. Fit
   `T_exp`, `T_mean`, `T_median` independently at the target FPR (default 99th
   percentile / 1% FPR).
2. **Validation set:** a **disjoint** 100 honest prompts under the same
   perturbations. Measure **realized** FPR here.
3. **Gate:** realized FPR on the held-out 100 is **< 3%**. Same-set “honest
   passes the thresholds we just fit on it” is **not** a gate.
4. Write target, calibrated values, realized FPR, and set sizes into
   `results/thresholds.json`. Also store the paper’s `{38, 10, 8}` for comparison.
5. Note in `decisions.md` that a 99th percentile from a few hundred samples has
   wide error bars. The paper reports **realized FPR**, not just the nominal target.

A broken hook that produced garbage would still “pass” a same-set gate, because
the threshold would be calibrated to the garbage. The held-out FPR is the test.

**Outputs:**

- `data/prompts.json` (`prompt_len`, later `natural_eos_pos`)
- `data/baseline/proofs.jsonl`
- `data/baseline/layer_topk.npz` with `values` and `indices`
- `data/clean_perturbed/`
- `results/thresholds.json`
- `results/clean_perturbation_audit.json`

**Gates:**

- Hook path is `TopKRecorder` decode mode; no `output_hidden_states=True`;
  buffer reset per prompt.
- Arrays have both `values` and `indices`, shape
  `200 × n_checkpoints × n_layers × 128` (`n_checkpoints` is 8 or 4).
- Mean clean-class delta is **not exactly zero**, **and** every kept
  perturbation has nonzero delta in the audit.
- Realized FPR on the **held-out** 100 < 3%.

**You:** Stop the instance. Confirm with `aws ec2 describe-instances` or the console.

---

## 7. Day 2 — AWS A10G: tamper variants + Phase A/B; contracts in parallel

Day 2 is the most failure-prone day and has no natural slack. The cut line
is decided **now**, not at 2am:

| Priority | Item | Cut? |
|---|---|---|
| 1 | Phase A + `matched` Phase B | **Never.** This is the paper. |
| 2 | `ft_drift` variant | **Never.** It is the §6.5 claim. |
| 3 | `quant_8bit` | Cut to a 4-class problem if needed. Say so in the paper. |
| 4 | `shared` control config | **Cut first.** If cut, weaken the weight claim in the paper body: the signal may be partly text-distribution, not only weights. |

If Day 2 ends without Phase B, push Phase B to **Day 3 morning** and TASK-05 to
Day 3 afternoon. TASK-06 has slack; TASK-03 does not.

### TASK-03 (3.0h GPU)

**I will implement `inference/build_variants.py`:**

| ID | Construction |
|---|---|
| `quant_8bit` | bitsandbytes `load_in_8bit=True` (first optional cut after `shared`) |
| `quant_4bit` | bitsandbytes `load_in_4bit=True`, nf4 |
| `lora_merged` | LoRA r=16 on attention QKV, 300 steps on **Alpaca** (or Dolly), then `merge_and_unload()` |
| `ft_drift` | full SFT, 300 steps, lr `1e-5`, no LoRA, 8-bit Adam + checkpointing |

C5 settings (in `config.yaml` and the trainer):

- `optim="adamw_bnb_8bit"`
- `gradient_checkpointing=True`
- `per_device_train_batch_size=1`, `gradient_accumulation_steps=8`
- `max_seq_length=512`

Train **only** on Alpaca or Dolly. Slice indices, dataset name, and split go in
`variant_manifest.json`. Eval prompts stay UltraChat. State disjointness in
the paper.

If SFT still OOMs: switch **every** variant to `Llama-3.2-1B-Instruct`. Do **not**
turn `ft_drift` into LoRA. Recompute `n_layers` from the loaded model.

**Gate:** each variant loads and generates **coherent** text. Garbage is a broken
build, not a tamper signal.

### TASK-04 — two phases

#### Phase A — autoregressive, per variant (2.0h)

Each variant generates its own `tokens_per_generation`-token continuation and
emits provider-side TOPLOC proofs. Needed for the binary reproduction gate and
the demo.

**Outputs:** `data/<variant_id>/proofs.jsonl` and `data/<variant_id>/layer_topk.npz`
(`values` + `indices`). Record `natural_eos_pos` per variant sequence too.

#### Phase B — teacher-forced prefill, classifier data (1.0h)

For each sequence, run a **single prefill** through that variant **and** through
the baseline, extracting per-layer top-k at **`prompt_len + 32 * k`**. Assert
token IDs match decode. Assert exactly `len(positions)` checkpoints per layer.
The delta is then attributable to weights, not to different text.

I will run and report **both** configs unless Day 2 cuts `shared`:

| Config | Sequences used | Purpose |
|---|---|---|
| `matched` | each variant’s own generated output | Deployment-realistic, **primary** result. Never cut. |
| `shared` | baseline outputs, prefilled through every variant | Control: identical text across classes. Cut first if Day 2 slips. |

If the classifier works in `shared` as well as `matched`, the signal is from
weights. If it only works in `matched`, I will say so in the paper. If `shared`
is cut, I will **weaken the claim in the paper body**, not only list a
limitation: the signal **may be partly attributable to output-distribution
differences rather than weights**.

Clean-class rows use this same Phase B prefill path. Do not mix Phase A
decode-time activations into the classifier table. For `matched`, prefill each
clean-perturbed sequence through the perturbed honest model and through the
unperturbed baseline. For `shared` (if present), prefill baseline sequences
through the perturbed honest model and through the baseline.

Phase B is prefill-only and cheaper than another 3h of generation.

#### Binary reproduction gate (deployment procedure)

1. Variant V generates sequence S and emits proof P from its own activations.
2. Validator runs a single prefill of S through the **baseline** model.
3. Validator compares prefill activations against P using **calibrated** thresholds.
4. Gate passes if this **fails** for all remaining tamper variants, **and** the
   **held-out** honest set from Day 1 stays under the realized-FPR cap.

Both directions are required. A detector that only flags tampering, without
honest-pass on unseen honest noise, is a detector that flags everything.

**Signal check:** record `ft_drift` error stats either way. If it does not
separate, **keep the negative result**. I will not retune extraction.

**You:** Stop the instance. Remaining work until TASK-08 is CPU except Day 5.

### TASK-07 started the same day (laptop, no GPU)

**I will implement `ProofOfWeights.sol` with exactly:**

- `registerProvider` — ERC20 stake
- `submitProof` — proof **hash** on-chain, blob off-chain (URI)
- `challenge` — requires a bond
- ~2 hour dispute window before proofs finalize
- `resolveChallenge(challengeId, upheld, tamperClass)` — validator-only
- upheld → slash, split challenger/validator; rejected → challenger forfeits bond
- `getTamperHistory` → `uint8[]` classes
- no double-slash; challenge outside window reverts

**Tests that must pass:** honest not slashed, tampered slashed, false challenger
loses bond, window revert, double-slash revert, history returns classes
(`0=clean, 1=quant_8bit, 2=quant_4bit, 3=lora_merged, 4=ft_drift`, or `255=TAMPERED`).

Deploy to **Base Sepolia**. Write address + ABI to `contracts/deployment.json`.
Trusted-validator caveat stays in the README.

**You:** testnet wallet, Sepolia ETH (Coinbase Base faucet is fine). I will not
use mainnet.

---

## 8. Day 3 — local CPU: features (the scientific gate)

If Phase B was not finished, do it this morning. Then TASK-05.

### TASK-05

**I will implement `fingerprint/extract_deltas.py`.** This is the **only**
feature implementation. The validator imports it.

For `matched` and `shared` (if present) separately, every variant, prompt,
checkpoint, vs the corresponding baseline prefill:

- per-layer top-k index mismatch ratio
- per-layer exponent mismatch count
- per-layer mean and median mantissa difference

Long format, one row per `(prompt, checkpoint, layer)`. Z-score each feature
**per layer**. Flatten to `4 × n_layers` only at classifier-training time.

Include `clean` rows from the Phase B path above. Their mean delta must be
nonzero (small) and far from the tamper classes if the features work.

Also tag checkpoints `pre_eos` vs `post_eos` using `natural_eos_pos`.

**Outputs:**

- `data/features_matched.parquet`
- `data/features_shared.parquet` (omit if `shared` was cut)
- columns: `[prompt_id, checkpoint, layer, idx_mismatch, exp_mismatch, mantissa_mean, mantissa_median, label, config, pre_eos]`

**Hard gate:** `figures/layer_profiles.png` (mean delta vs layer depth, one line
per variant, **matched**). **Look at it before I train anything.**

Hypothesis:

- quantization ≈ depth-uniform
- `lora_merged` ≈ localized at LoRA-targeted modules
- `ft_drift` ≈ drift increasing with depth
- `clean` ≈ near-zero but **not** a delta spike at exactly 0

If there is no structure: try per-layer normalization or restrict to top-32.
Do **not** train a classifier on a blob with no structure.

### Binary fallback if features do not separate

After **two** attempts, if `layer_profiles.png` still shows no per-variant
structure by **end of Day 3**: switch to the binary fallback **that evening**.
Do **not** spend Day 4 on a third feature variant. Demo and paper both survive;
only the headline changes.

| Piece | If TASK-05 separates | If fallback fires |
|---|---|---|
| Demo shape | Unchanged: detect → challenge → slash | **Unchanged in shape.** Still detect, challenge, slash. |
| `tamperClass` | Predicted 5-class (or 4 if `quant_8bit` was cut) | Binary **`TAMPERED` (`255`)**. Contract **keeps** the field. |
| Validator | Compare proofs + run classifier | Compare proofs only. No classifier call. |
| Dashboard | “CAUGHT” + diagnosed type | “CAUGHT” + `TAMPERED` |
| Pitch lead | Attribution + typed reputation | **Economic layer + TASK-08b stake number.** Typed reputation is future work, not a claim. |
| Paper | Attribution + §6.5 measurement | **Negative result.** “We test whether depth-wise activation signatures carry tamper-type information and find they do not at this scale.” State what a proper test needs (larger models, more severities). Calibration, held-out FPR, and L4 numbers still go in. |
| Day 4 | TASK-06 | **Skip TASK-06.** Wire binary validator; start the negative-result paper. |
| Day 5 D9 | Parity on 10 Phase B sequences | **Skip D9** (no classifier path). Detection + slash still required. |
| Day 5 C8 | Classifier-boundary transfer + L4 honest FPR | L4 honest runs still required: treat as extra `clean` noise; apply A10G-calibrated thresholds; record realized FPR. No classifier-transfer number. |

Write `results/attribution_status.json` either way, for example
`{ "separated": false, "attempts": 2, "fallback": "binary_tampered" }`,
so Day 4+ do not improvise.

Finish TASK-07 deploy if not already done. The contract already has `tamperClass`;
binary mode writes `255`.

---

## 9. Day 4 — local CPU: classifier (or skip)

### TASK-06

**Skip this task if the Day 3 fallback fired.** Do not invent a classifier on
unstructured features. Spend the day on the binary validator path and the
negative-result paper skeleton.

**Otherwise I will implement `fingerprint/train_classifier.py` and `evaluate.py`.**

Classes: `clean | quant_8bit | quant_4bit | lora_merged | ft_drift`
(4-class if `quant_8bit` was cut).

- Logistic regression and k-NN only. No neural net. Default `k=5`; if I pick
  another k, one line in `decisions.md`.
- `class_weight="balanced"` on logistic regression only. k-NN is **unweighted**;
  say so in `decisions.md`. Headline metrics are balanced accuracy and per-class
  F1, so class imbalance is covered without weighting k-NN.
- Split by **prompt**, never by checkpoint.
- 200 prompts are the independent units, not `200 × n_checkpoints`. Feature dim
  is `4 × n_layers` (112 if 28 layers, 64 if 16). I will use **5-fold grouped
  cross-validation by prompt** and report **mean ± std**, plus a confidence
  interval. I will not publish a single-split accuracy.
- Headline: **balanced accuracy** and **per-class F1**, not raw accuracy.
  Record per-class sample counts in `metrics.json`.
- Report the same metrics on **all checkpoints** and on **pre-EOS checkpoints
  only**. If they differ substantially, degeneration is a confound and belongs
  in the paper.
- Train/eval **matched** as primary. Also report **shared** if it exists.

**Outputs:** `artifacts/classifier.pkl`, `figures/confusion_matrix.png`,
`results/metrics.json` (clean-delta distribution, matched vs shared, fold
mean/std, balanced accuracy, per-class F1 and counts, all vs pre-EOS).

**Gates:**

- Confusion matrix committed and honest. If `ft_drift` is confused with
  `clean`, that **quantifies** TOPLOC §6.5.
- `metrics.json` contains the clean-class delta magnitude distribution. Mean
  exactly 0 → invalid.
- Per-class counts recorded.
- I will not fish for a better number on the reported folds.

Any remaining underspecified choice (normalization details) goes into
`results/decisions.md` with one line of rationale. C7 FPR is already specified
(target 1%, accept realized < 3% on the held-out 100) and is not a free knob.

---

## 10. Day 5 — AWS: demo, cost number, then L4 (sequential, not together)

**You:** start the **same** `g5.xlarge`. Do not launch `g6.xlarge` or any
second G-instance while this box is Running.

### TASK-08 (2h GPU) + feature-path parity

**I will implement:**

- `honest_provider.py` — real weights, proof, `submitProof`
- `malicious_provider.py` — same API, randomly swaps one of the remaining variants
- `validator.py` — sample 10% of traffic, re-run as a **single prefill pass**,
  compare proofs with **calibrated** thresholds, on mismatch run classifier
  (if it exists), `challenge` then `resolveChallenge` with predicted
  `tamperClass` or `255`

If TASK-05 separated: `validator.py` calls `fingerprint.extract_deltas` for live
features, then the classifier. I will **not** reimplement extraction in the
validator.

If the fallback fired: validator compares proofs only and
`resolveChallenge(..., tamperClass=255)`. Feature-path parity is N/A.
Detection + slash still required.

**Parity gate, before the live slash (classifier path only):** run the
validator’s live extraction on **10** sequences already in the Phase B dataset.
Assert feature vectors match the offline vectors with
`max abs diff ≤ 1e-4` on the z-scored vectors (`feature_path_parity_atol`).
Write `results/feature_path_parity.json`. If this fails, fix the path — do not demo.

**Gate:** one complete cycle. Tampered response → confirmed slash → class
on-chain (typed or `255`). **Save the transaction hash.**

### TASK-08b (0.5h, same A10G session)

**I will measure, not estimate:**

- Left: serving cost/request for `quant_4bit` and `quant_8bit` (if present) vs bf16
  (tokens/sec and peak VRAM on this A10G)
- Right: validator overhead per verified request (prefill vs original
  autoregressive). Measure the ratio on this box.

Honesty dominates when:

```
savings_per_request  <  p_sample × stake_slashed / requests_per_epoch
```

**Output:** `results/cost_asymmetry.json` plus minimum stake at 1%, 5%, 10% sampling.

**Gate:** one sentence: *“At a 10% sampling rate, stake must exceed X for honesty
to dominate.”*

**You:** **Stop `g5.xlarge`.** Confirm it is stopped (`aws ec2 describe-instances`
or the console). Quota is 8 vCPU; the L4 **cannot start** while the A10G is
Running. Do not overlap them “briefly.”

### Cross-hardware (0.5h on `g6.xlarge` L4, ~$0.40)

**You:** only after the A10G is Stopped, launch **one** `g6.xlarge` (NVIDIA L4,
Ada, CC 8.9, 22 GB), Deep Learning AMI, 50 GB, on-demand, same key /
SSH-from-my-IP. Not T4. Not a second `g5`.

**I will:**

1. Install from the committed `requirements.txt` (or the same `pip install` line).
2. **Then** `assert torch.cuda.is_bf16_supported()` and record GPU name / CC.
   If someone later swaps the instance type, fail loudly rather than emulate bf16.
3. Re-run validation on **50 baseline sequences** on the L4.
4. Treat L4 honest runs as additional `clean` noise. Apply the **A10G-calibrated**
   thresholds and record realized FPR on these 50.
5. If a classifier exists: test whether the **A10G-trained** boundary transfers
   to L4-generated clean samples.
6. If the fallback fired: skip classifier transfer; still write the L4 honest
   FPR / environment numbers. The negative-result paper still needs them.

**Output:** `results/cross_hardware.json` — GPU name, CC, bf16, L4 realized FPR,
and classifier-transfer metrics **only if** `attribution_status.json` says
separated. Merge L4 fields into `results/environment.json`.

**You:** Stop (then later Terminate) the L4. Do not leave it running.

This is the highest-value dollar in the budget: Ampere A10G vs Ada L4 is real
architecture variation with native bf16 on both sides.

---

## 11. Day 6 — dashboard, rehearsal, paper skeleton

### TASK-09

**I will build a minimal dashboard** polling the contract and validator logs.
Live: proof stream, a **CAUGHT** event (diagnosed type if TASK-05 separated,
`TAMPERED` if the fallback fired), provider stake decreasing.

**Gate:** a non-technical viewer understands in 10 seconds that a cheater was
caught and penalized.

### TASK-10

**We will:**

1. Run the full demo **three** times end to end.
2. You record a backup video of one clean run.
3. I write the README: problem, three gaps, architecture diagram,
   speculative-decoding limitation, trusted-validator caveat, EOS suppression
   + length path, disjoint train/eval data, Day 2 cuts if any (with a
   **weakened weight claim** if `shared` was cut), binary fallback if it
   fired, reproduction instructions, contract address.
4. You confirm **no** AWS GPU instance is still **Running** (A10G and L4).

**I will start the paper the same day** from committed artifacts only.
Always: `thresholds.json`, `clean_perturbation_audit.json`, `environment.json`,
`variant_manifest.json`, `prompts.json`, `cost_asymmetry.json`,
`cross_hardware.json`, `attribution_status.json`, `layer_profiles.png`.
If TASK-05 separated, also: `metrics.json`, `confusion_matrix.png`,
`feature_path_parity.json`.

I will not write numbers from memory.

---

## 12. Day 7 — pitch + arXiv

### TASK-11 — 3-minute pitch (I will draft the script in this order)

1. Problem: providers silently swap cheaper models; major DePIN GPU networks
   have conceded verification is unsolved.
2. Landscape: ZKML is orders of magnitude too expensive; TOPLOC detects but
   returns a Boolean and names subtle fine-tunes as open; nothing has enforcement.
3. Contribution: if TASK-05 separated — attribution + slashing + typed reputation.
   If the fallback fired — **slashing + measured stake threshold**; attribution
   is future work. The paper is then economics + the negative result.
4. Security claim with TASK-08b numbers (independent of the classifier).
5. Live demo (video fallback).
6. Limitations **before anyone asks**: trusted validator, speculative decoding
   undetectable, one model family, EOS / length path, L4 transfer is 50
   sequences, `shared` or `quant_8bit` cut if Day 2 slipped (and if `shared`
   was cut: signal may be partly text-distribution, not only weights),
   binary fallback if it fired.
7. Next: real validator set, more tamper classes, more architectures.

Then freeze the paper and post to arXiv. The contract address in the README is
part of reproducibility: anyone can read testnet state and verify the slash.

---

## 13. Paper I will write (this week = arXiv only)

**Allowed claim if TASK-05 separated:**

> Ong et al. (2025) note that detecting slight gradient updates in fine-tuned
> models is significantly harder than detecting model swaps, and leave it as
> future work. We extend their top-k activation scheme from the last hidden
> layer to a per-layer profile, and show that the resulting depth-wise signature
> both detects and attributes the modification. We additionally provide the
> economic enforcement layer their protocol-agnostic primitive lacks.

**Allowed claim if the Day 3 fallback fired (use this wording, not the one above):**

> Ong et al. (2025) leave detection of slight fine-tunes as future work. We
> test whether extending their top-k scheme to a per-layer profile attributes
> tamper type at the 1.5B scale and find that it does not. We still provide
> the economic enforcement layer their protocol-agnostic primitive lacks, with
> a measured stake threshold.

If `shared` was cut, the first claim is further weakened in the method and
experiments sections: the signal may be partly output-distribution, not only
weights.

**Sections:**

1. Abstract — detection + optimistic slashing; if TASK-05 separated:
   attribution, **balanced accuracy** (mean ± std), `ft_drift`, `matched` vs
   `shared`. If fallback: **negative attribution result** at 1.5B, economic
   layer still claimed.
2. Introduction — the three gaps table
3. Background — TOPLOC mechanics; we use `toploc`, we do not reimplement it
4. Method
   - Per-layer top-k via **`TopKRecorder` decode | prefill modes**
   - Teacher-forced deltas at `prompt_len + 32 * k` (`matched` [+ `shared`])
   - `clean` under kernel / batch / padding-or-deterministic noise, not
     self-comparison and not tf32
   - Four delta features, z-scored per layer
   - Interpretable classifier, `class_weight=balanced` on LR, k-NN unweighted,
     prompt-grouped 5-fold CV — **omitted if fallback**
   - Optimistic protocol; verification off-chain
5. Experiments
   - UltraChat eval (200 prompts, 256 **or 128** tokens, EOS suppressed),
     Alpaca/Dolly train
   - Qwen2.5-1.5B bf16, A10G; L4 transfer; optional Llama-1B fallback
   - Binary reproduction with **calibrated** thresholds; **realized FPR** on
     held-out honest prompts; paper thresholds reported
   - Layer profiles; confusion matrix **or** “no separation” figure
   - If `shared` was cut: **explicit** “signal may be partly output-distribution,
     not weights” — not a footnote limitation
   - Clean-delta distribution + perturbation audit
   - All-checkpoint vs pre-EOS metrics (if 256-token path)
   - L4 numbers (`cross_hardware.json`) — FPR always; classifier transfer only
     if a classifier exists
   - Feature-path parity (if a classifier exists)
   - Cost asymmetry and minimum stake
6. Limitations — speculative decoding; trusted validator; single model family;
   EOS / length path; L4 check is small; Day 2 cuts; fallback if fired; no
   bit-exact claim beyond the L4 check; GPU nondeterminism; 99th-percentile
   sample error
7. Conclusion — what this week showed; workshop-tier follow-ups

**Reproducibility block (non-negotiable):**

- HF revision hash
- `data/prompts.json` (`prompt_len`, `natural_eos_pos`) and `variant_manifest.json`
- every seed, including calib/val split
- `requirements.txt` from `pip freeze`
- `results/environment.json` (A10G and L4: GPU, CC, driver, CUDA, bf16)
- `results/thresholds.json` (paper + calibrated + realized FPR)
- `results/clean_perturbation_audit.json`
- `results/attribution_status.json`
- `results/cross_hardware.json`
- `results/feature_path_parity.json` **if** a classifier exists
- contract address in README

I will not claim workshop or ICML readiness from one 1.5B model.

| Tier | When | Extra work (not this week) |
|---|---|---|
| arXiv | this week | Careful writeup, honest matrix or honest negative, §6.5 positioning |
| Workshop | +2–4 weeks | 2–3 model families, multiple severities, held-out generalization, evasion |
| ICML / NeurIPS | months | Adaptive-attacker argument, theory, broad architectures |

---

## 14. Hard gates — I stop if these fail

These are the stop conditions. Section gates above match this table.

| After | Gate | If it fails |
|---|---|---|
| TASK-01 | `toploc.build_proofs_bytes` returns bytes; `is_bf16_supported()` **after** pip | Fix install. Do not generate data. |
| Length check | Path chosen (256 / 128 / one re-filter); `n_checkpoints` written in `config.yaml` and `decisions.md` | No unbounded filter loop. Prefer 128 tokens if median is 128–256. |
| TASK-02 hook | `TopKRecorder` decode mode; reset per prompt; `values`+`indices`; shape uses `n_checkpoints` | Fix hooks. Do not use `output_hidden_states=True`. |
| TASK-02 audit | Every kept clean perturbation has nonzero delta in `clean_perturbation_audit.json` | Swap inert axes (`allow_tf32` is already banned). Redo. |
| TASK-02 clean | Mean clean-class delta is **not** exactly zero | Perturbations were not applied. Redo. Run is invalid. |
| TASK-02 FPR | Realized FPR on **held-out** 100 honest prompts < 3% | Recalibrate or fix hook. Same-set pass is **not** a gate. |
| TASK-02 lengths | `prompt_len` stored for all 200 | Do not start Phase B. |
| TASK-03 | Each variant generates coherent text | Broken build. Rebuild. If SFT OOMs, Llama-1B for all variants, not LoRA-`ft_drift`. |
| TASK-03 data | Train dataset ≠ UltraChat; slices in manifest | Retrain. Do not eval on the train distribution. |
| TASK-04 detect | Remaining tamper variants fail calibrated compare (V’s proof vs baseline prefill of S) | We have not reproduced TOPLOC. Diagnose. |
| TASK-04 honest | Held-out honest FPR stays under cap | Detector flags everything. Fix before claiming a detector. |
| TASK-04 tokens | Prefill token IDs at `prompt_len + 32 * k` match decode | Off-by-N. Fix before any features. |
| TASK-04 count | Phase B has **exactly `len(positions)` checkpoints per layer per prompt** | Last-token-only prefill. Fix `mode="prefill"`. Do not proceed with a short array. |
| TASK-04 signal | Record `ft_drift` stats either way | Do not tune until it separates. Negative is valid. |
| TASK-04 matched | `matched` Phase B exists (`shared` optional) | Do not train on autoregressive mismatched text only. |
| TASK-05 | `layer_profiles.png` shows some per-variant structure | One more attempt (norm / top-32). If it fails again by **end of Day 3**, binary fallback. No Day 4 third try. Write `attribution_status.json` either way. |
| TASK-06 | Honest matrix + 5-fold grouped **balanced acc / F1** mean ± std + class counts | Report real numbers. Raw accuracy is not the headline. **Skip if fallback.** |
| TASK-06 EOS | All-checkpoint and pre-EOS metrics both written | Do not hide degeneration. |
| TASK-07 | `forge test` + testnet deploy | Do not wire providers until this passes. |
| TASK-08 parity | `feature_path_parity.json` matches on 10 Phase B sequences | Fix the single extraction path. Do not demo. **Skip if fallback** (no classifier). |
| TASK-08 demo | One slash + class on-chain (typed or `255`) + tx hash | Demo is not done. |
| TASK-08b | Measured stake threshold at 1%, 5%, 10% | Do not estimate. Re-measure on the same A10G. |
| C8 | `assert is_bf16_supported()` **after** pip on **L4**; `cross_hardware.json` written | Do not use T4. Do not claim cross-hardware without the L4 number. Required even if fallback fired (honest FPR / environment). |

---

## 15. Money I will protect

| Item | Plan |
|---|---|
| `g5.xlarge` Running | $1.006/h — TASK-01–04 and TASK-08/08b (~11.9 h → ~$12) |
| `g6.xlarge` Running | ~$0.80/h × 0.5 h → ~$0.40 (verify in console) |
| 50 GB gp3 while Stopped | ~$1 for the week (A10G disk; L4 disk deleted after C8) |
| Testnet | $0 |
| Idle overnight | I will remind you to Stop. 24h A10G ≈ $24 |

If YC AWS $10k credits are on the account, cash is $0. Still Stop both boxes.

---

## 16. First actions in this repo

1. This file is the execution plan.
2. Next: scaffold the repo + `config.yaml` (fields in Section 3) +
   `inference/hooks.py` (`TopKRecorder` with decode/prefill modes) + Foundry
   skeleton + `results/decisions.md` (EOS suppression, Day 2 cut order, FPR
   error bars, length path, k-NN unweighted, binary-fallback rule).
3. Stop and wait for you to launch the `g5.xlarge` and confirm `nvidia-smi`
   shows A10G. Do **not** treat an AMI-level `is_bf16_supported()` check as
   the official assert — that runs in TASK-01 **after** `pip install`.
4. Then TASK-01 on that machine.
