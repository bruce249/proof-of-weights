---
project: proof-of-weights
type: agent-executable build spec
timeline: 7 days
target: Devcon Mumbai hackathon + arXiv preprint
base_paper: TOPLOC (Ong et al., ICML 2025, arXiv:2501.16007)
---

# Proof of Weights: Build Spec

> This document is written to be executed by a coding agent or a human working
> alongside one. Every task has explicit inputs, outputs, and acceptance criteria.
> Do not proceed to the next task until the current task's acceptance criteria pass.

## Agent execution protocol

Read before starting any task.

1. **Never fabricate a result.** If a measurement fails or a number is unavailable,
   write the failure into `results/` and stop. A fabricated accuracy number destroys
   the entire point of this project.
2. **Acceptance criteria are gates, not suggestions.** If a gate fails, do not
   proceed. Diagnose, fix, re-run. If it still fails after two attempts, record the
   failure and escalate to the human.
3. **Negative results are valid outputs.** Several tasks explicitly say to record a
   negative result rather than tune until it looks good. Tuning the feature extraction
   until `ft_drift` separates, on the same data you then report metrics on, is leakage.
4. **GPU cost discipline.** After any task marked `Where: AWS`, stop the instance
   before starting a CPU-only task. Verify with `aws ec2 describe-instances`.
5. **Ambiguity policy.** If a spec detail is underspecified, choose the simplest
   option, write the choice into `results/decisions.md` with a one-line rationale,
   and continue. Do not stall on it.
6. **Everything reproducible.** Any seed, revision hash, or version you touch goes
   into a committed config file at the moment you touch it, not retroactively.

---

## 0. Context the agent must load first

### 0.1 What the base method does

TOPLOC (Ong et al., ICML 2025) verifies LLM inference by locality-sensitive hashing
of intermediate activations. Concretely:

- Takes the **last hidden layer** activation tensor.
- Extracts **top-k values and indices** (paper uses k=128).
- Encodes them as a **polynomial congruence** in an integer field (k coefficients, 2 bytes each).
- Saves one proof **every 32 generated tokens** (258 bytes per 32 tokens).
- Validation re-runs the sequence as a **single prefill pass** (much faster than autoregressive
  generation) and compares exponent bits and mantissa differences against the proof.
- Returns a **single Boolean**: valid or invalid.

Paper thresholds for bf16: `T_exp = 38`, `T_mean = 10`, `T_median = 8`.
Validation fails if exponent mismatches exceed T_exp, or mean mantissa diff exceeds
T_mean, or median mantissa diff exceeds T_median.

Why top-k works (paper Section 3.3): in bf16, most error comes from catastrophic
cancellation in the residual stream, which disproportionately affects small-magnitude
values. Large-magnitude activations stay stable across GPU nondeterminism and algebraic
reordering. This is the theoretical basis for our per-layer extension too.

### 0.2 The three gaps we are filling

| Gap | Evidence in the paper | What we build |
|---|---|---|
| Binary output, no attribution | Returns only a Boolean validity flag | Multi-class tamper-type classifier |
| Subtle tampering undetected | Section 6.5 states that slight gradient updates in fine-tuned models are significantly harder to detect and names this as future work | Per-layer feature extraction targeted at exactly this case |
| No economic enforcement | No protocol, registry, stake, or penalty anywhere in the paper | On-chain staking and slashing registry |

### 0.3 Explicit non-goals

- **Do not** attempt to detect speculative decoding (cheap model for decode, real model
  for prefill). Paper Section 6.2 states this is undetectable by the method. Out of scope,
  name it as a known limitation in the README.
- **Do not** rebuild TOPLOC. Use `pip install toploc`.
- **Do not** build a decentralized validator set. One validator wallet is sufficient for the demo.
- **Do not** claim generalization beyond the model family actually tested.

### 0.4 Key methodological difference from the base paper

TOPLOC hashes the **last hidden layer only**. We extract top-k features from **every
decoder layer** and build a per-layer delta vector. The per-layer pattern, not the
scalar pass/fail, is what carries tamper-type information. State this explicitly
in the README and any writeup.

Implementation note: obtain per-layer states via `output_hidden_states=True`, then
immediately reduce to top-128 values and indices per layer and discard the full tensor.
Do not buffer full hidden states across the generation loop; on a 1.5B model with 28
layers that will exhaust the 24GB of VRAM. Reduce inside the generation hook, per step.

### 0.5 Why a blockchain is in this design

State this precisely. The imprecise version does not survive questioning.

**The verification is entirely off-chain.** This is an optimistic verification system
with cryptoeconomic guarantees, not a cryptographically trustless one. Say so directly
in the README and the pitch.

The chain provides exactly three things that no centralized alternative provides:

1. **Stake from an anonymous counterparty.** In a permissionless GPU marketplace the
   provider is a wallet address. There is no contract to enforce and no entity to sue,
   so capital at risk is the only enforceable commitment available. This is the actual
   reason crypto belongs in this problem.
2. **A neutral, append-only public record.** A provider cannot delete their own slash
   history and re-register clean.
3. **Permissionless challenging.** Anyone can challenge, without being an auditor the
   network appointed.

**The unique consequence: typed reputation.** Because we attribute the tamper type
rather than just flagging it, the on-chain record becomes queryable in a way binary
detection cannot produce. "Provider 0x4f was slashed" is weak. "Provider 0x4f served
4-bit weights while claiming bf16, three times" is a portable, machine-readable
reputation primitive that any router, network, or agent can query before routing
traffic. This is a crypto-native consequence of the ML contribution, not crypto bolted on.

---

## 1. Environment and constraints

```yaml
compute:
  provider: AWS            # Azure has no GPU allocation available
  instance: g5.xlarge      # 1x NVIDIA A10G, 24GB VRAM, 4 vCPU, 16GB RAM
  region: us-east-1
  ami: AWS Deep Learning AMI (PyTorch, GPU, Ubuntu 22.04)
  storage: 50GB gp3 EBS
  on_demand_rate_usd_per_hour: 1.006   # verify in console before launch
  budget_usd: 100
  gpu_hours_budgeted: 20
  pricing_mode: on-demand              # NOT spot; interruptions cost debug time

model:
  primary: Qwen/Qwen2.5-1.5B-Instruct
  fallback: meta-llama/Llama-3.2-1B-Instruct
  pin_revision: true                   # record the exact HF commit hash in config.yaml
  precision: bfloat16                  # matches paper experiment setup

dataset:
  prompts: HuggingFaceH4/ultrachat_200k    # paper used UltraChat
  n_prompts: 200
  tokens_per_generation: 256
  proof_interval_tokens: 32
  topk: 128

chain:
  framework: foundry
  network: <sponsor testnet, default Base Sepolia>
  rpc: Alchemy free tier
```

**Hard rule:** stop the EC2 instance after every session. An idle instance is the
only realistic way to exceed the budget.

---

## 2. Repository layout

```
proof-of-weights/
├── config.yaml                  # all hyperparameters, model revision, thresholds
├── requirements.txt             # pinned versions
├── inference/
│   ├── build_variants.py        # produces the 4 model variants
│   ├── generate.py              # runs inference, emits TOPLOC proofs + per-layer activations
│   └── providers/
│       ├── honest_provider.py
│       └── malicious_provider.py
├── fingerprint/
│   ├── extract_deltas.py        # per-layer delta feature extraction
│   ├── train_classifier.py
│   └── evaluate.py              # confusion matrix, generalization test
├── contracts/                   # foundry project
│   ├── src/ProofOfWeights.sol
│   └── test/ProofOfWeights.t.sol
├── validator/
│   └── validator.py             # sampling, re-run, classify, on-chain challenge
├── dashboard/
└── README.md
```

---

## 3. Task graph

### TASK-01: Environment bootstrap
**Where:** AWS g5.xlarge
**Depends on:** none
**Estimated GPU time:** 0.5h

```bash
pip install -U toploc transformers accelerate bitsandbytes peft datasets scikit-learn web3
python -c "import toploc, torch; print(torch.cuda.get_device_name(0))"
```

**Acceptance:** `toploc.build_proofs_bytes` runs on a dummy bf16 tensor and returns bytes.
Record the exact installed versions into `requirements.txt` with `pip freeze`.

---

### TASK-02: Baseline generation and proofs
**Where:** AWS
**Depends on:** TASK-01
**Estimated GPU time:** 1.5h

1. Load the pinned model in bf16.
2. Sample 200 prompts from UltraChat, fix the seed, save to `data/prompts.json`.
3. Generate 256 tokens per prompt with KV caching, autoregressive.
4. Every 32 tokens, emit a TOPLOC proof from the last hidden state (topk=128).
5. Simultaneously save **per-layer** hidden states at the same 32-token checkpoints
   (top-128 values and indices per layer, not the full tensor).

**Outputs:**
- `data/baseline/proofs.jsonl` (one record per prompt per checkpoint)
- `data/baseline/layer_topk.npz` (shape: n_prompts x n_checkpoints x n_layers x 128)

**Acceptance:** re-running validation on the baseline model passes TOPLOC's own thresholds
(`T_exp=38`, `T_mean=10`, `T_median=8`). If it fails, the extraction hook is wrong,
fix before continuing.

---

### TASK-03: Build tampered variants
**Where:** AWS
**Depends on:** TASK-02
**Estimated GPU time:** 2.5h

Produce four tampered models from the same base:

| variant_id | construction |
|---|---|
| `quant_8bit` | bitsandbytes `load_in_8bit=True` |
| `quant_4bit` | bitsandbytes `load_in_4bit=True`, nf4 |
| `lora_merged` | LoRA r=16 on attention QKV, 300 steps on a small instruction set, then `merge_and_unload()` |
| `ft_drift` | full SFT, 300 steps, lr 1e-5, no LoRA |

Fix all training seeds and commit the training config. Reproducibility depends on this.

**Outputs:** `models/<variant_id>/` plus a `variant_manifest.json` recording seed,
steps, lr, and dataset slice for each.

**Acceptance:** each variant loads and generates coherent text. A variant that produces
garbage is a broken build, not a tamper signal.

---

### TASK-04: Tampered generation pass
**Where:** AWS
**Depends on:** TASK-03
**Estimated GPU time:** 3h

Run TASK-02's exact pipeline on the same 200 prompts for each of the four variants.

**Outputs:** `data/<variant_id>/proofs.jsonl` and `data/<variant_id>/layer_topk.npz`

**Acceptance (two separate checks, both required):**
1. **Binary check:** validating each variant's generation against the *baseline* model
   fails TOPLOC's thresholds. This reproduces the paper's result on your setup.
2. **Signal check:** `ft_drift` is the hard case. If it fails to separate cleanly, that
   is itself the interesting result, since the paper names this as the difficult case.
   Record the actual error statistics either way. Do not discard the negative result.

**Then stop the EC2 instance.** Everything from TASK-05 to TASK-08 runs on CPU.

---

### TASK-05: Per-layer delta features
**Where:** Local (CPU)
**Depends on:** TASK-04

For each variant and each checkpoint, compute against baseline:
- per-layer top-k index mismatch ratio
- per-layer exponent mismatch count
- per-layer mean and median mantissa difference

Store in long format, one row per (prompt, checkpoint, layer), with the four features
as columns. Flatten to a `4 × n_layers` vector per (prompt, checkpoint) only at
classifier-training time in TASK-06. Z-score each feature per layer across the dataset
so that raw magnitude differences between layers do not dominate.

**Output:** `data/features.parquet` with columns
`[prompt_id, checkpoint, layer, idx_mismatch, exp_mismatch, mantissa_mean, mantissa_median, label]`

**Acceptance:** produce `figures/layer_profiles.png` plotting mean delta by layer depth,
one line per variant. **Look at this plot before continuing.** The hypothesis is that
quantization shows a roughly depth-uniform profile, `lora_merged` shows localization at
the LoRA-targeted modules, and `ft_drift` shows depth-increasing drift. If the plot shows
no separation, fix the feature (try per-layer normalization choices, or restrict to the
top-32 activations) before training any classifier.

---

### TASK-06: Tamper-type classifier
**Where:** Local (CPU)
**Depends on:** TASK-05

Train a 5-class classifier: `clean | quant_8bit | quant_4bit | lora_merged | ft_drift`.

- Start with logistic regression and k-NN. Do not use a neural network. An interpretable
  model with a clear feature story is worth more here than a point of accuracy.
- Split by **prompt**, never by checkpoint, to avoid leakage across checkpoints of the
  same generation.
- Report: accuracy, per-class precision and recall, full confusion matrix.

**Outputs:** `artifacts/classifier.pkl`, `figures/confusion_matrix.png`, `results/metrics.json`

**Acceptance:** confusion matrix is committed and honest. Report the real number even if
`ft_drift` is confused with `clean`. That confusion is a publishable finding, since it
quantifies the difficulty the base paper only asserted qualitatively.

---

### TASK-07: On-chain registry
**Where:** Local, Foundry
**Depends on:** none (can run in parallel with TASK-05/06)

`contracts/src/ProofOfWeights.sol`:

```solidity
function registerProvider(bytes32 modelHash) external;      // requires ERC20 stake deposit

// Store the proof HASH on-chain, the proof blob off-chain (IPFS / blob store).
// 258 bytes per 32 tokens is small until multiplied by real traffic and calldata cost.
function submitProof(uint256 providerId, bytes32 promptHash, bytes32 proofHash, string calldata blobURI) external;

// Challenger must post a bond; bond is forfeited on a false challenge.
function challenge(uint256 providerId, uint256 proofId) external payable;

function resolveChallenge(uint256 challengeId, bool upheld, uint8 tamperClass) external;
    // validator-only; on upheld, slash stake and split between challenger and validator
    // on rejected, challenger forfeits bond

// Typed reputation: the queryable primitive that binary detection cannot produce.
function getTamperHistory(uint256 providerId) external view returns (uint8[] memory);
```

Required mechanism details, not optional:

- **Bonded challenges.** Without a bond, challenging is free and the system is
  griefable. This is the first thing a judge will attack.
- **Dispute window.** Proofs finalize only after a window elapses. Standard optimistic
  shape, roughly two hours of Solidity.
- **Trusted-validator caveat.** A single validator wallet calling `resolveChallenge`
  is a trusted oracle inside a trust-minimized system. This is a known limitation of
  the demo. Either implement 3-validator commit-reveal verdicts if time allows, or
  name it explicitly in the README and pitch. Do not let a judge find it first.
- `tamperClass` recorded on-chain is the diagnostic output made publicly auditable,
  which is the part no existing system does.

**Acceptance:** `forge test` passes covering: honest provider not slashed, tampered
provider slashed, false challenger forfeits bond, challenge outside the dispute window
reverts, double-slashing reverts, `getTamperHistory` returns the recorded classes.
Contract deployed to testnet, address and ABI written to `contracts/deployment.json`.

---

### TASK-08: Provider and validator wiring
**Where:** Local orchestration, AWS for inference
**Depends on:** TASK-06, TASK-07
**Estimated GPU time:** 2h

- `honest_provider.py`: serves real weights, emits proof, calls `submitProof`.
- `malicious_provider.py`: identical interface, randomly swaps in one of the four variants.
- `validator.py`: samples 10% of traffic, re-runs it as a **single prefill pass** (per the
  paper, this is what makes validation cheaper than generation), compares proofs, and on
  mismatch runs the classifier and calls `challenge` then `resolveChallenge` with the
  predicted `tamperClass`.

**Acceptance:** one complete end-to-end cycle where a tampered response results in a
confirmed on-chain slash with the correct tamper class recorded. Capture the transaction hash.

---

### TASK-08b: Cost-asymmetry measurement
**Where:** AWS (piggyback on TASK-08's session)
**Depends on:** TASK-08
**Estimated GPU time:** 0.5h

This is the core mechanism-design result and the part a crypto-native judge will care
about most. Cheating is unprofitable when:

```
savings_per_request  <  p_sample × stake_slashed / requests_per_epoch
```

Measure both sides empirically, do not estimate:

- **Left side:** serving cost per request for `quant_4bit` and `quant_8bit` versus
  the bf16 baseline. Measure tokens/sec and peak VRAM on the same instance.
- **Right side:** validator overhead per verified request. Per the base paper,
  validation is a single prefill pass and is cheaper than the original autoregressive
  generation. Measure the actual ratio on your setup rather than citing theirs.

**Output:** `results/cost_asymmetry.json` with measured numbers, plus the minimum
stake required for honesty to dominate at sampling rates of 1%, 5%, and 10%.

**Acceptance:** a parameterized security claim stated with measured numbers on both
sides, in the form "at a 10% sampling rate, stake must exceed X for honesty to dominate."
Nobody in this space currently states this cleanly with real measurements.

---

### TASK-09: Demo dashboard
**Where:** Local
**Depends on:** TASK-08

Minimal frontend polling the contract and validator logs. Must show, in real time:
live proof stream, a "CAUGHT" event labelled with the diagnosed tamper type, and the
provider's stake balance decreasing.

**Acceptance:** a non-technical viewer can tell within 10 seconds that a cheater was
caught and penalized.

---

### TASK-10: Rehearsal and hardening
**Depends on:** TASK-09

1. Run the full demo three times end to end.
2. Record a backup video of one clean run (venue wifi and RPC access will fail at the worst time).
3. Write the README with: problem, the three gaps from Section 0.2, architecture diagram,
   the speculative-decoding limitation from Section 0.3, and reproduction instructions.
4. Confirm no AWS resources are still billing.

---

### TASK-11: Buffer and pitch
**Depends on:** TASK-10

3-minute pitch structure:
1. Problem: providers silently swap in cheaper models; the largest DePIN GPU networks
   have publicly conceded that verification is unsolved.
2. Existing landscape: ZKML is orders of magnitude too expensive; TOPLOC detects but
   only returns a Boolean and its own paper names subtle fine-tune detection as open;
   nothing has an enforcement layer.
3. Our contribution: attribution plus slashing, and the typed reputation primitive
   that falls out of it (Section 0.5).
4. The security claim with measured numbers from TASK-08b.
5. Live demo.
6. Known limitations, stated before anyone asks: trusted validator in the demo,
   speculative decoding undetectable by the underlying method, single model family tested.
7. What is next (real validator set, more tamper classes, more architectures).

---

## 4. Reproducibility requirements

Non-negotiable, these are what make it a paper rather than a demo:

- Pin the HF model revision hash, not just the model name.
- Commit `data/prompts.json` and `variant_manifest.json`.
- Fix and record every seed (prompt sampling, LoRA training, SFT training).
- Commit `requirements.txt` from `pip freeze`, since toploc, bitsandbytes, and peft move fast.
- Record GPU type, driver, and CUDA version in `results/environment.json`. Activation
  values drift across hardware, so do not claim bit-exact cross-hardware reproducibility
  without testing it.
- The contract is inherently reproducible: anyone can read testnet state and verify the
  slash occurred. Put the address in the README.

---

## 5. Path to publication

| Tier | Feasible from | Additional work required |
|---|---|---|
| arXiv preprint | this week's output | Careful writeup, honest confusion matrix, explicit positioning against TOPLOC Section 6.5 |
| Workshop paper | 2 to 4 weeks after | 2-3 model families (the base paper used Llama, INTELLECT-1, Gemma), multiple tamper severities, held-out generalization (train on one severity, test on another), adversarial evasion check |
| Main track (ICML/NeurIPS) | months | Full security argument against adaptive attackers, theory for why depth-wise signatures separate by tamper type, broad architecture coverage |

**Framing for the writeup:** Ong et al. (2025) note that detecting slight gradient
updates in fine-tuned models is significantly harder than detecting model swaps, and
leave it as future work. We extend their top-k activation scheme from the last hidden
layer to a per-layer profile, and show that the resulting depth-wise signature both
detects and attributes the modification. We additionally provide the economic
enforcement layer their protocol-agnostic primitive lacks.

---

## 6. Cost checklist

- [ ] Stop the EC2 instance after every session
- [ ] On-demand, not spot
- [ ] Verify current EC2 pricing before launch
- [ ] GPU work confined to TASK-01 through TASK-04 and TASK-08 (about 10h total)
- [ ] Testnet gas only, no mainnet deployment
