"""TASK-02: 200-prompt baseline, clean-noise audit, disjoint TOPLOC thresholds.

Never use output_hidden_states=True. TOPLOC proofs from CPU last-hidden only.
"""

from __future__ import annotations

import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from fingerprint.extract_deltas import extract_from_topk_pair
from inference.hooks import TopKRecorder, decoder_layers, stack_buf


ROOT = Path(__file__).resolve().parents[1]
LONG_MARKERS = (
    "explain",
    "step by step",
    "step-by-step",
    "list",
    "why",
    "how",
    "describe",
    "detailed",
    "write a",
    "analyze",
)
PROMPT_SEED = 20260923


def load_config() -> dict:
    with (ROOT / "config.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def n_checkpoints(cfg: dict) -> int:
    return cfg["tokens_per_generation"] // cfg["proof_interval_tokens"]


def user_text(example: dict) -> str:
    msgs = example.get("messages") or example.get("conversation") or []
    for m in msgs:
        role = (m.get("role") or m.get("from") or "").lower()
        if role in ("user", "human"):
            return (m.get("content") or m.get("value") or "").strip()
    if msgs:
        return str(msgs[0].get("content") or msgs[0].get("value") or "").strip()
    return str(example.get("prompt") or example.get("text") or "").strip()


def is_long_answer(text: str) -> bool:
    low = text.lower()
    return any(m in low for m in LONG_MARKERS) and len(text.split()) >= 12


def pick_prompts(n: int, seed: int) -> list[str]:
    ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft")
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    out: list[str] = []
    for i in idxs:
        text = user_text(ds[int(i)])
        if is_long_answer(text):
            out.append(text)
        if len(out) >= n:
            break
    if len(out) < n:
        raise RuntimeError(f"only found {len(out)} long-answer prompts")
    return out


def encode_prompt(tok, text: str) -> torch.Tensor:
    rendered = tok.apply_chat_template(
        [{"role": "user", "content": text}],
        add_generation_prompt=True,
        tokenize=False,
    )
    ids = tok(rendered, return_tensors="pt", add_special_tokens=False).input_ids
    if ids.dim() == 1:
        ids = ids.unsqueeze(0)
    return ids


def load_model(cfg: dict, attn: str = "sdpa"):
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["hf_revision"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        revision=cfg["hf_revision"],
        dtype=torch.bfloat16,
        device_map="cuda",
        attn_implementation=attn,
    )
    model.eval()
    return tok, model


class LastHiddenTap:
    """Final-norm last token. Used for TOPLOC, not per-layer top-k."""

    def __init__(self):
        self.h = None
        self._handle = None

    def attach(self, model) -> None:
        def hook(_m, _i, out):
            t = out[0] if isinstance(out, tuple) else out
            self.h = t[:, -1, :].detach()

        self._handle = model.model.norm.register_forward_hook(hook)

    def detach(self) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None


def _proofs_from_hiddens(hiddens: list[torch.Tensor], cfg: dict) -> list[bytes]:
    import toploc

    cpu = [h.detach().to("cpu").contiguous().view(-1).to(torch.bfloat16) for h in hiddens]
    return toploc.build_proofs_bytes(
        cpu,
        decode_batching_size=cfg["proof_interval_tokens"],
        topk=cfg["topk"],
        skip_prefill=True,
    )


def _verify(hiddens: list[torch.Tensor], proofs: list[bytes], cfg: dict):
    import toploc

    cpu = [h.detach().to("cpu").contiguous().view(-1).to(torch.bfloat16) for h in hiddens]
    return toploc.verify_proofs_bytes(
        cpu,
        proofs,
        decode_batching_size=cfg["proof_interval_tokens"],
        topk=cfg["topk"],
        skip_prefill=True,
    )


def _seq_stats(results) -> dict:
    # Installed toploc reports exp_mismatches (error; higher is worse), not intersections.
    exps = [float(r.exp_mismatches) for r in results]
    means = [float(r.mant_err_mean) for r in results]
    meds = [float(r.mant_err_median) for r in results]
    return {
        "exp_mismatches_max": max(exps),
        "mant_err_mean_max": max(means),
        "mant_err_median_max": max(meds),
        "n_proofs": len(results),
    }


def generate_one(
    model,
    tok,
    rec: TopKRecorder,
    tap: LastHiddenTap,
    prompt_ids: torch.Tensor,
    cfg: dict,
    teacher_ids: list[int] | None = None,
) -> dict:
    """Prefill without recording, then 256 decode steps. step starts at 0; increment after each decode."""
    device = model.device
    prompt_ids = prompt_ids.to(device)
    n_new = cfg["tokens_per_generation"]
    n_ck = n_checkpoints(cfg)
    n_layers = len(decoder_layers(model))
    eos_id = tok.eos_token_id

    rec.mode = None
    tap.h = None
    with torch.no_grad():
        out = model(input_ids=prompt_ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :]

    natural_eos = None
    gen: list[int] = []
    last_hiddens: list[torch.Tensor] = []
    rec.reset("decode")

    for t in range(n_new):
        if teacher_ids is None:
            if eos_id is not None:
                would = int(logits[0].argmax().item())
                if would == eos_id and natural_eos is None:
                    natural_eos = t
                logits = logits.clone()
                logits[0, eos_id] = -float("inf")
            next_id = int(logits[0].argmax().item())
        else:
            next_id = int(teacher_ids[t])
        gen.append(next_id)
        step_in = torch.tensor([[next_id]], device=device, dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids=step_in, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
        if tap.h is None:
            raise RuntimeError("LastHiddenTap missed a decode step")
        last_hiddens.append(tap.h[0].detach().cpu())
        rec.step += 1

    vals, idxs = stack_buf(rec.buf, n_ck, n_layers, cfg["topk"])
    if vals.shape[0] != n_ck:
        raise RuntimeError(f"expected {n_ck} checkpoints, got values {tuple(vals.shape)}")
    proofs = _proofs_from_hiddens(last_hiddens, cfg)
    if len(proofs) != n_ck:
        raise RuntimeError(f"expected {n_ck} proofs, got {len(proofs)}")
    return {
        "prompt_len": int(prompt_ids.shape[1]),
        "tokens": gen,
        "natural_eos_pos": natural_eos if natural_eos is not None else n_new,
        "hit_eos": natural_eos is not None,
        "values": vals.numpy(),
        "indices": idxs.numpy(),
        "proofs": proofs,
        "hiddens": last_hiddens,
    }


def generate_baseline(cfg: dict) -> None:
    seed = PROMPT_SEED
    cfg.setdefault("seeds", {})["prompt_sampling"] = seed
    tok, model = load_model(cfg, attn="sdpa")
    rec = TopKRecorder(interval=cfg["proof_interval_tokens"], k=cfg["topk"])
    rec.attach(decoder_layers(model))
    tap = LastHiddenTap()
    tap.attach(model)

    texts = pick_prompts(cfg["n_prompts"], seed)
    n_ck = n_checkpoints(cfg)
    n_layers = len(decoder_layers(model))
    k = cfg["topk"]
    values = np.zeros((len(texts), n_ck, n_layers, k), dtype=np.float32)
    indices = np.zeros((len(texts), n_ck, n_layers, k), dtype=np.int32)
    prompts = []
    proofs_path = ROOT / "data" / "baseline" / "proofs.jsonl"
    proofs_path.parent.mkdir(parents=True, exist_ok=True)
    hid_dir = ROOT / "data" / "baseline" / "hiddens"
    hid_dir.mkdir(parents=True, exist_ok=True)

    with proofs_path.open("w", encoding="utf-8") as pf:
        for i, text in enumerate(texts):
            ids = encode_prompt(tok, text)
            row = generate_one(model, tok, rec, tap, ids, cfg)
            values[i] = row["values"]
            indices[i] = row["indices"]
            hid_path = hid_dir / f"{i:04d}.pt"
            torch.save(row["hiddens"], hid_path)
            recs = {
                "prompt_id": i,
                "text": text,
                "prompt_len": row["prompt_len"],
                "natural_eos_pos": row["natural_eos_pos"],
                "hit_eos": row["hit_eos"],
                "tokens": row["tokens"],
            }
            prompts.append(recs)
            pf.write(
                json.dumps(
                    {
                        "prompt_id": i,
                        "proofs_hex": [p.hex() for p in row["proofs"]],
                    }
                )
                + "\n"
            )
            print(
                f"baseline [{i:03d}/{len(texts)}] prompt_len={row['prompt_len']} "
                f"eos={row['natural_eos_pos']} proofs={len(row['proofs'])}",
                flush=True,
            )

    rec.detach()
    tap.detach()
    np.savez_compressed(
        ROOT / "data" / "baseline" / "layer_topk.npz",
        values=values,
        indices=indices,
    )
    (ROOT / "data" / "prompts.json").write_text(
        json.dumps({"seed": seed, "prompts": prompts}, indent=2) + "\n",
        encoding="utf-8",
    )
    with (ROOT / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print("wrote data/baseline/layer_topk.npz", values.shape, indices.shape)


def _set_attn(model, attn: str) -> None:
    if hasattr(model, "set_attn_implementation"):
        model.set_attn_implementation(attn)
    else:
        model.config._attn_implementation = attn


def replay_one(
    model,
    tok,
    rec: TopKRecorder,
    tap: LastHiddenTap,
    prompt_ids: torch.Tensor,
    tokens: list[int],
    cfg: dict,
) -> dict:
    return generate_one(model, tok, rec, tap, prompt_ids, cfg, teacher_ids=tokens)


def replay_batch(
    model,
    tok,
    rec: TopKRecorder,
    tap: LastHiddenTap,
    prompt_ids_list: list[torch.Tensor],
    tokens_list: list[list[int]],
    cfg: dict,
    extra_pad: int = 0,
) -> list[dict]:
    """Left-pad a batch and teacher-force the same tokens. Changes matmul shape under bf16."""
    device = model.device
    n_new = cfg["tokens_per_generation"]
    n_ck = n_checkpoints(cfg)
    n_layers = len(decoder_layers(model))
    k = cfg["topk"]
    bsz = len(tokens_list)
    max_len = max(int(p.shape[1]) for p in prompt_ids_list) + extra_pad
    padded = []
    for p in prompt_ids_list:
        need = max_len - int(p.shape[1])
        if need:
            pad = torch.full((1, need), tok.pad_token_id, dtype=p.dtype)
            p = torch.cat([pad, p], dim=1)
        padded.append(p)
    batch = torch.cat(padded, dim=0).to(device)
    attn = (batch != tok.pad_token_id).long()

    rec.mode = None
    tap.h = None
    with torch.no_grad():
        out = model(input_ids=batch, attention_mask=attn, use_cache=True)
    past = out.past_key_values
    rec.reset("decode")
    hiddens_per: list[list[torch.Tensor]] = [[] for _ in range(bsz)]
    for t in range(n_new):
        step = torch.tensor([[tokens_list[b][t] for b in range(bsz)]], device=device, dtype=torch.long).T
        attn = torch.cat([attn, torch.ones(bsz, 1, device=device, dtype=attn.dtype)], dim=1)
        with torch.no_grad():
            out = model(
                input_ids=step,
                past_key_values=past,
                attention_mask=attn,
                use_cache=True,
            )
        past = out.past_key_values
        if tap.h is None or tap.h.shape[0] != bsz:
            raise RuntimeError("LastHiddenTap batch mismatch")
        for b in range(bsz):
            hiddens_per[b].append(tap.h[b].detach().cpu())
        rec.step += 1

    rows = []
    for b in range(bsz):
        vals, idxs = stack_buf(rec.buf, n_ck, n_layers, k, batch_index=b)
        proofs = _proofs_from_hiddens(hiddens_per[b], cfg)
        rows.append(
            {
                "values": vals.numpy(),
                "indices": idxs.numpy(),
                "hiddens": hiddens_per[b],
                "proofs": proofs,
            }
        )
    return rows


def run_clean_and_calibrate(cfg: dict) -> None:
    prompts = json.loads((ROOT / "data" / "prompts.json").read_text(encoding="utf-8"))["prompts"]
    base = np.load(ROOT / "data" / "baseline" / "layer_topk.npz")
    base_v, base_i = base["values"], base["indices"]
    proofs = []
    with (ROOT / "data" / "baseline" / "proofs.jsonl").open(encoding="utf-8") as f:
        for line in f:
            proofs.append([bytes.fromhex(h) for h in json.loads(line)["proofs_hex"]])

    tok, model = load_model(cfg, attn="sdpa")
    rec = TopKRecorder(interval=cfg["proof_interval_tokens"], k=cfg["topk"])
    rec.attach(decoder_layers(model))
    tap = LastHiddenTap()
    tap.attach(model)

    axes = [
        {"name": "attn_eager", "attn": "eager", "batch": 1, "extra_pad": 0},
        {"name": "batch4", "attn": "sdpa", "batch": 4, "extra_pad": 0},
        {"name": "padded_batch_length", "attn": "sdpa", "batch": 4, "extra_pad": 17},
    ]
    fallbacks = [
        {"name": "batch8", "attn": "sdpa", "batch": 8, "extra_pad": 0},
        {"name": "deterministic", "attn": "sdpa", "batch": 1, "extra_pad": 0, "deterministic": True},
    ]

    hid_dir = ROOT / "data" / "baseline" / "hiddens"
    clean_dir = ROOT / "data" / "clean_perturbed"
    clean_dir.mkdir(parents=True, exist_ok=True)
    audit = {}

    def run_axis(axis: dict) -> dict:
        name = axis["name"]
        _set_attn(model, axis["attn"])
        det = bool(axis.get("deterministic"))
        deltas = []
        seq_stats = []
        n_ck = n_checkpoints(cfg)
        n_layers = base_v.shape[2]
        k = cfg["topk"]
        out_v = np.zeros_like(base_v)
        out_i = np.zeros_like(base_i)
        bsz = int(axis["batch"])
        extra_pad = int(axis["extra_pad"])
        for start in range(0, len(prompts), bsz):
            chunk = prompts[start : start + bsz]
            ids_list = [encode_prompt(tok, rec_p["text"]) for rec_p in chunk]
            toks_list = [rec_p["tokens"] for rec_p in chunk]
            if det:
                torch.use_deterministic_algorithms(True)
            try:
                if bsz == 1:
                    ids = ids_list[0]
                    if extra_pad:
                        pad = torch.full((1, extra_pad), tok.pad_token_id, dtype=ids.dtype)
                        ids = torch.cat([pad, ids], dim=1)
                    rows = [replay_one(model, tok, rec, tap, ids, toks_list[0], cfg)]
                else:
                    rows = replay_batch(
                        model, tok, rec, tap, ids_list, toks_list, cfg, extra_pad=extra_pad
                    )
            finally:
                if det:
                    torch.use_deterministic_algorithms(False)
            for off, row in enumerate(rows):
                i = start + off
                out_v[i] = row["values"]
                out_i[i] = row["indices"]
                feat = extract_from_topk_pair(base_v[i], base_i[i], row["values"], row["indices"])
                deltas.append(float(np.mean(np.abs(feat))))
                stats = _seq_stats(_verify(row["hiddens"], proofs[i], cfg))
                seq_stats.append(stats)
                print(f"clean {name} [{i:03d}] mean_abs_feat={deltas[-1]:.6g}", flush=True)
        np.savez_compressed(clean_dir / f"{name}_layer_topk.npz", values=out_v, indices=out_i)
        mean_delta = float(np.mean(deltas))
        return {
            "name": name,
            "mean_abs_feature_delta": mean_delta,
            "min_abs_feature_delta": float(np.min(deltas)),
            "max_abs_feature_delta": float(np.max(deltas)),
            "inert": mean_delta == 0.0,
            "n": len(deltas),
            "seq_stats": seq_stats,
            "axis": axis,
        }

    kept = []
    for axis in axes:
        info = run_axis(axis)
        audit[info["name"]] = {k: v for k, v in info.items() if k != "seq_stats"}
        audit[info["name"]]["seq_stats_summary"] = {
            "exp_mismatch_max_mean": float(np.mean([s["exp_mismatches_max"] for s in info["seq_stats"]])),
            "mant_mean_max_mean": float(np.mean([s["mant_err_mean_max"] for s in info["seq_stats"]])),
        }
        if info["inert"]:
            print("INERT axis", info["name"], "— swapping")
            if fallbacks:
                fb = fallbacks.pop(0)
                info = run_axis(fb)
                audit[info["name"]] = {k: v for k, v in info.items() if k != "seq_stats"}
        if not info["inert"]:
            kept.append(info)
        else:
            print("still inert after swap", info["name"])

    if len(kept) < 3:
        raise RuntimeError(f"need 3 live clean axes, kept { [k['name'] for k in kept] }")

    # Calibrate on first 100, validate on last 100 of the union of kept axes
    # (each prompt × each kept axis is one honest observation).
    calib_idx = list(range(cfg["calibration_prompts"]))
    val_idx = list(range(cfg["calibration_prompts"], cfg["n_prompts"]))
    cfg.setdefault("seeds", {})["calib_val_split"] = "index_0_99_calib_100_199_val"

    def collect(indices: list[int]) -> tuple[list[float], list[float], list[float]]:
        exps, means, meds = [], [], []
        for info in kept:
            for i in indices:
                s = info["seq_stats"][i]
                exps.append(s["exp_mismatches_max"])
                means.append(s["mant_err_mean_max"])
                meds.append(s["mant_err_median_max"])
        return exps, means, meds

    c_exp, c_mean, c_med = collect(calib_idx)
    # 1% FPR: all three stats are errors. Fail if any exceeds its 99th percentile.
    t_exp = float(np.percentile(c_exp, 99))
    t_mean = float(np.percentile(c_mean, 99))
    t_med = float(np.percentile(c_med, 99))

    def flagged(s: dict, te: float, tm: float, td: float) -> bool:
        return (
            s["exp_mismatches_max"] > te
            or s["mant_err_mean_max"] > tm
            or s["mant_err_median_max"] > td
        )

    def fpr(indices: list[int], te: float, tm: float, td: float) -> float:
        flags = 0
        n = 0
        for info in kept:
            for i in indices:
                flags += int(flagged(info["seq_stats"][i], te, tm, td))
                n += 1
        return flags / n if n else 0.0

    realized = fpr(val_idx, t_exp, t_mean, t_med)
    paper = cfg["paper_thresholds"]
    paper_fpr = fpr(val_idx, paper["T_exp"], paper["T_mean"], paper["T_median"])

    thresholds = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "target_fpr": cfg["target_fpr"],
        "max_realized_fpr": cfg["max_realized_fpr"],
        "calibration_n_prompts": len(calib_idx),
        "validation_n_prompts": len(val_idx),
        "n_kept_axes": len(kept),
        "kept_axes": [k["name"] for k in kept],
        "calibrated": {"T_exp": t_exp, "T_mean": t_mean, "T_median": t_med},
        "paper": paper,
        "realized_fpr_heldout": realized,
        "paper_fpr_heldout": paper_fpr,
        "same_set_fpr_not_a_gate": fpr(calib_idx, t_exp, t_mean, t_med),
        "gate_passed": realized < cfg["max_realized_fpr"],
        "note": "99th-percentile from a few hundred samples has wide error bars; report realized FPR.",
    }
    (ROOT / "results" / "thresholds.json").write_text(
        json.dumps(thresholds, indent=2) + "\n", encoding="utf-8"
    )
    (ROOT / "results" / "clean_perturbation_audit.json").write_text(
        json.dumps(
            {
                "written_at": thresholds["written_at"],
                "axes": audit,
                "kept": [k["name"] for k in kept],
                "mean_clean_delta": float(np.mean([k["mean_abs_feature_delta"] for k in kept])),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    rec.detach()
    tap.detach()
    with (ROOT / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print("thresholds", json.dumps(thresholds, indent=2))
    if realized >= cfg["max_realized_fpr"]:
        raise SystemExit(f"GATE FAIL held-out FPR {realized} >= {cfg['max_realized_fpr']}")
    print("TASK-02 gate passed")


def main() -> int:
    cfg = load_config()
    stage = "all"
    if len(__import__("sys").argv) > 1:
        stage = __import__("sys").argv[1]
    if stage in ("all", "baseline"):
        generate_baseline(cfg)
    if stage in ("all", "clean"):
        run_clean_and_calibrate(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
