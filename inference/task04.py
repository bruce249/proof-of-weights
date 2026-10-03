"""TASK-04 Phase A (autoregressive proofs) and Phase B (teacher-forced prefill).

Proofs go through toploc on CPU tensors. No output_hidden_states=True.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from transformers import BitsAndBytesConfig

from inference.build_variants import (
    VARIANT_IDS,
    input_device,
    load_causal,
    load_tokenizer,
    log,
    release,
)
from inference.hooks import TopKRecorder, decoder_layers, prefill_positions, stack_buf
from inference.task02 import (
    LastHiddenTap,
    _seq_stats,
    _verify,
    encode_prompt,
    generate_one,
    load_config,
    n_checkpoints,
)

ROOT = Path(__file__).resolve().parents[1]


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _prompts() -> list[dict]:
    payload = json.loads((ROOT / "data" / "prompts.json").read_text(encoding="utf-8"))
    return payload["prompts"]


def _thresholds() -> dict:
    return json.loads((ROOT / "results" / "thresholds.json").read_text(encoding="utf-8"))["calibrated"]


def _flagged(stats: dict, th: dict) -> bool:
    return (
        stats["exp_mismatches_max"] > th["T_exp"]
        or stats["mant_err_mean_max"] > th["T_mean"]
        or stats["mant_err_median_max"] > th["T_median"]
    )


def _load_variant(cfg: dict, variant_id: str):
    model_id = cfg["model"]
    revision = cfg["hf_revision"]
    if variant_id == "quant_8bit":
        q = BitsAndBytesConfig(load_in_8bit=True)
        model, attn = load_causal(model_id, revision, q)
    elif variant_id == "quant_4bit":
        q = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model, attn = load_causal(model_id, revision, q)
    elif variant_id in ("lora_merged", "ft_drift"):
        path = ROOT / "artifacts" / "variants" / variant_id
        tok = load_tokenizer(str(path), None)
        model, attn = load_causal(str(path), None, None)
        model.eval()
        return tok, model, attn
    else:
        raise ValueError(variant_id)
    tok = load_tokenizer(model_id, revision)
    model.eval()
    return tok, model, attn


def _load_baseline(cfg: dict, attn: str = "sdpa"):
    tok = load_tokenizer(cfg["model"], cfg["hf_revision"])
    model, used = load_causal(cfg["model"], cfg["hf_revision"], None)
    if attn == "eager" and used != "eager":
        if hasattr(model, "set_attn_implementation"):
            model.set_attn_implementation("eager")
        else:
            model.config._attn_implementation = "eager"
        used = "eager"
    model.eval()
    return tok, model, used


def _full_ids(tok, text: str, tokens: list[int], device) -> tuple[torch.Tensor, int]:
    prompt = encode_prompt(tok, text)
    gen = torch.tensor([tokens], dtype=prompt.dtype)
    ids = torch.cat([prompt, gen], dim=1).to(device)
    return ids, int(prompt.shape[1])


def phase_a(cfg: dict) -> None:
    prompts = _prompts()
    n_ck = n_checkpoints(cfg)
    for variant_id in VARIANT_IDS:
        out = ROOT / "data" / variant_id
        done = out / "layer_topk.npz"
        if done.exists() and (out / "proofs.jsonl").exists() and (out / "sequences.json").exists():
            log(f"phase A skip {variant_id}")
            continue
        log(f"phase A load {variant_id}")
        tok, model, attn = _load_variant(cfg, variant_id)
        n_layers = len(decoder_layers(model))
        rec = TopKRecorder(interval=cfg["proof_interval_tokens"], k=cfg["topk"])
        rec.attach(decoder_layers(model))
        tap = LastHiddenTap()
        tap.attach(model)
        values = np.zeros((len(prompts), n_ck, n_layers, cfg["topk"]), dtype=np.float32)
        indices = np.zeros((len(prompts), n_ck, n_layers, cfg["topk"]), dtype=np.int32)
        sequences = []
        out.mkdir(parents=True, exist_ok=True)
        proofs_path = out / "proofs.jsonl"
        with proofs_path.open("w", encoding="utf-8") as pf:
            for row in prompts:
                ids = encode_prompt(tok, row["text"]).to(input_device(model))
                gen = generate_one(model, tok, rec, tap, ids, cfg)
                i = row["prompt_id"]
                values[i] = gen["values"]
                indices[i] = gen["indices"]
                sequences.append(
                    {
                        "prompt_id": i,
                        "prompt_len": gen["prompt_len"],
                        "tokens": gen["tokens"],
                        "natural_eos_pos": gen["natural_eos_pos"],
                        "hit_eos": gen["hit_eos"],
                    }
                )
                pf.write(json.dumps({"prompt_id": i, "proofs_hex": [p.hex() for p in gen["proofs"]]}) + "\n")
                if i % 10 == 0:
                    log(f"phase A {variant_id} [{i:03d}] prompt_len={gen['prompt_len']} eos={gen['natural_eos_pos']}")
        rec.detach()
        tap.detach()
        np.savez_compressed(done, values=values, indices=indices)
        (out / "sequences.json").write_text(json.dumps(sequences) + "\n", encoding="utf-8")
        log(f"phase A wrote {variant_id} {values.shape}")
        del model, tok, rec, tap
        release()


def _prefill_decode_hiddens(model, ids: torch.Tensor, prompt_len: int, n_new: int) -> list[torch.Tensor]:
    captured = {}

    def hook(_m, _i, out):
        t = out[0] if isinstance(out, tuple) else out
        captured["h"] = t.detach()

    handle = model.model.norm.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids=ids)
    finally:
        handle.remove()
    if "h" not in captured:
        raise RuntimeError("prefill hook missed the final norm")
    h = captured["h"][0, prompt_len : prompt_len + n_new].to("cpu", dtype=torch.bfloat16).contiguous()
    if h.shape[0] != n_new:
        raise RuntimeError(f"prefill hidden length {h.shape[0]} != {n_new}")
    return [h[t] for t in range(n_new)]


def phase_a_gate(cfg: dict) -> None:
    th = _thresholds()
    prompts = _prompts()
    log("phase A gate load baseline")
    tok, model, attn = _load_baseline(cfg, "sdpa")
    report = {"written_at": _now(), "attn": attn, "thresholds": th, "variants": {}}
    n_new = cfg["tokens_per_generation"]
    for variant_id in VARIANT_IDS:
        sequences = json.loads((ROOT / "data" / variant_id / "sequences.json").read_text(encoding="utf-8"))
        by_id = {s["prompt_id"]: s for s in sequences}
        proofs = []
        with (ROOT / "data" / variant_id / "proofs.jsonl").open(encoding="utf-8") as f:
            for line in f:
                proofs.append([bytes.fromhex(h) for h in json.loads(line)["proofs_hex"]])
        flags = []
        stats_rows = []
        for row in prompts:
            seq = by_id[row["prompt_id"]]
            ids, prompt_len = _full_ids(tok, row["text"], seq["tokens"], input_device(model))
            if prompt_len != seq["prompt_len"]:
                raise RuntimeError(f"{variant_id} prompt_len {prompt_len} != {seq['prompt_len']}")
            hiddens = _prefill_decode_hiddens(model, ids, prompt_len, n_new)
            stats = _seq_stats(_verify(hiddens, proofs[row["prompt_id"]], cfg))
            flag = _flagged(stats, th)
            flags.append(flag)
            stats_rows.append(stats)
            if row["prompt_id"] % 20 == 0:
                log(f"gate {variant_id} [{row['prompt_id']:03d}] flagged={flag} exp={stats['exp_mismatches_max']:.2f}")
        detect = float(np.mean(flags))
        report["variants"][variant_id] = {
            "detect_rate": detect,
            "n": len(flags),
            "exp_mismatches_max_median": float(np.median([s["exp_mismatches_max"] for s in stats_rows])),
            "mant_err_mean_max_median": float(np.median([s["mant_err_mean_max"] for s in stats_rows])),
            "mant_err_median_max_median": float(np.median([s["mant_err_median_max"] for s in stats_rows])),
            "gate_passed": detect >= 0.95,
        }
        log(f"gate {variant_id} detect_rate={detect:.3f}")
    report["heldout_honest_fpr"] = json.loads((ROOT / "results" / "thresholds.json").read_text(encoding="utf-8"))[
        "realized_fpr_heldout"
    ]
    report["passed"] = all(v["gate_passed"] for v in report["variants"].values()) and report["heldout_honest_fpr"] < 0.03
    path = ROOT / "results" / "phase_a_gate.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {path} passed={report['passed']}")
    del model, tok
    release()
    if not report["passed"]:
        raise SystemExit("Phase A reproduction gate did not pass for every variant. Numbers are in phase_a_gate.json")


def _prefill_topk(model, rec: TopKRecorder, ids: torch.Tensor, prompt_len: int, tokens: list[int], cfg: dict):
    n_ck = n_checkpoints(cfg)
    interval = cfg["proof_interval_tokens"]
    positions = prefill_positions(prompt_len, cfg["tokens_per_generation"], interval)
    expected = [tokens[interval * k] for k in range(n_ck)]
    got = ids[0, positions].detach().cpu().tolist()
    if got != expected:
        raise AssertionError(f"token IDs at {positions} mismatch: decode {expected} vs prefill {got}")
    rec.reset("prefill", positions=positions)
    with torch.no_grad():
        model(input_ids=ids)
    n_layers = len(decoder_layers(model))
    rec.assert_prefill_complete(n_layers)
    return stack_buf(rec.buf, n_ck, n_layers, cfg["topk"])


def _save_pair(path: Path, values: np.ndarray, indices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, values=values, indices=indices)
    log(f"wrote {path} {values.shape}")


def phase_b(cfg: dict) -> None:
    prompts = _prompts()
    n_ck = n_checkpoints(cfg)
    n_new = cfg["tokens_per_generation"]

    def alloc(n_layers: int):
        v = np.zeros((len(prompts), n_ck, n_layers, cfg["topk"]), dtype=np.float32)
        i = np.zeros((len(prompts), n_ck, n_layers, cfg["topk"]), dtype=np.int32)
        return v, i

    # Shared text is the baseline generation already stored on each prompt.
    log("phase B shared baseline")
    tok, base, _attn = _load_baseline(cfg, "sdpa")
    rec = TopKRecorder(interval=cfg["proof_interval_tokens"], k=cfg["topk"])
    rec.attach(decoder_layers(base))
    n_layers = len(decoder_layers(base))
    shared_base_v, shared_base_i = alloc(n_layers)
    shared_path = ROOT / "data" / "phase_b" / "shared" / "baseline.npz"
    if not shared_path.exists():
        for row in prompts:
            ids, prompt_len = _full_ids(tok, row["text"], row["tokens"], input_device(base))
            values, indices = _prefill_topk(base, rec, ids, prompt_len, row["tokens"], cfg)
            shared_base_v[row["prompt_id"]] = values.numpy()
            shared_base_i[row["prompt_id"]] = indices.numpy()
            if row["prompt_id"] % 20 == 0:
                log(f"phase B shared baseline [{row['prompt_id']:03d}]")
        _save_pair(shared_path, shared_base_v, shared_base_i)
    rec.detach()
    rec.attach(decoder_layers(base))

    for variant_id in VARIANT_IDS:
        sequences = json.loads((ROOT / "data" / variant_id / "sequences.json").read_text(encoding="utf-8"))
        by_id = {s["prompt_id"]: s for s in sequences}
        matched_path = ROOT / "data" / "phase_b" / "matched" / f"{variant_id}.npz"
        matched_base_path = ROOT / "data" / "phase_b" / "matched" / f"baseline_on_{variant_id}.npz"
        shared_var_path = ROOT / "data" / "phase_b" / "shared" / f"{variant_id}.npz"

        if not matched_base_path.exists():
            log(f"phase B matched baseline on {variant_id} text")
            bv, bi = alloc(n_layers)
            for row in prompts:
                seq = by_id[row["prompt_id"]]
                ids, prompt_len = _full_ids(tok, row["text"], seq["tokens"], input_device(base))
                values, indices = _prefill_topk(base, rec, ids, prompt_len, seq["tokens"], cfg)
                bv[row["prompt_id"]] = values.numpy()
                bi[row["prompt_id"]] = indices.numpy()
            rec.detach()
            rec.attach(decoder_layers(base))
            _save_pair(matched_base_path, bv, bi)

        del_base = False
        if matched_path.exists() and shared_var_path.exists():
            log(f"phase B skip {variant_id}")
            continue
        del base, rec
        release()
        del_base = True
        log(f"phase B load {variant_id}")
        vtok, vmodel, _attn = _load_variant(cfg, variant_id)
        vrec = TopKRecorder(interval=cfg["proof_interval_tokens"], k=cfg["topk"])
        vrec.attach(decoder_layers(vmodel))
        vn_layers = len(decoder_layers(vmodel))
        if vn_layers != n_layers:
            raise RuntimeError(f"{variant_id} n_layers {vn_layers} != baseline {n_layers}")
        if not matched_path.exists():
            mv, mi = alloc(n_layers)
            for row in prompts:
                seq = by_id[row["prompt_id"]]
                ids, prompt_len = _full_ids(vtok, row["text"], seq["tokens"], input_device(vmodel))
                values, indices = _prefill_topk(vmodel, vrec, ids, prompt_len, seq["tokens"], cfg)
                mv[row["prompt_id"]] = values.numpy()
                mi[row["prompt_id"]] = indices.numpy()
                if row["prompt_id"] % 20 == 0:
                    log(f"phase B matched {variant_id} [{row['prompt_id']:03d}]")
            _save_pair(matched_path, mv, mi)
        if not shared_var_path.exists():
            sv, si = alloc(n_layers)
            for row in prompts:
                ids, prompt_len = _full_ids(vtok, row["text"], row["tokens"], input_device(vmodel))
                values, indices = _prefill_topk(vmodel, vrec, ids, prompt_len, row["tokens"], cfg)
                sv[row["prompt_id"]] = values.numpy()
                si[row["prompt_id"]] = indices.numpy()
                if row["prompt_id"] % 20 == 0:
                    log(f"phase B shared {variant_id} [{row['prompt_id']:03d}]")
            _save_pair(shared_var_path, sv, si)
        vrec.detach()
        del vmodel, vtok, vrec
        release()
        if del_base:
            tok, base, _attn = _load_baseline(cfg, "sdpa")
            rec = TopKRecorder(interval=cfg["proof_interval_tokens"], k=cfg["topk"])
            rec.attach(decoder_layers(base))

    log("phase B complete")
    _ = n_new


def main() -> int:
    cfg = load_config()
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    log(f"TASK-04 stage={stage}")
    if stage in ("all", "a"):
        phase_a(cfg)
    if stage in ("all", "gate"):
        phase_a_gate(cfg)
    if stage in ("all", "b"):
        phase_b(cfg)
    log("TASK-04 done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
