"""Measured serving time and validator prefill cost on this GPU. No estimates."""

from __future__ import annotations

import json
import time

import torch

from inference.build_variants import input_device, log, release
from inference.task02 import encode_prompt, load_config
from inference.task04 import ROOT, _full_ids, _load_baseline, _load_variant, _prompts

# Locked on-demand price from the experiment plan. Not a live bill.
HOURLY_USD = 1.006
N_TIMED = 3
REQUESTS_PER_EPOCH = 1


def _sync() -> None:
    torch.cuda.synchronize()


def _autoregressive(model, tok, prompt_ids: torch.Tensor, n_new: int) -> tuple[list[int], float]:
    device = input_device(model)
    prompt_ids = prompt_ids.to(device)
    eos_id = tok.eos_token_id
    _sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model(input_ids=prompt_ids, use_cache=True)
    past = out.past_key_values
    logits = out.logits[:, -1, :]
    gen: list[int] = []
    for _ in range(n_new):
        if eos_id is not None:
            logits = logits.clone()
            logits[0, eos_id] = -float("inf")
        next_id = int(logits[0].argmax().item())
        gen.append(next_id)
        step = torch.tensor([[next_id]], device=device, dtype=torch.long)
        with torch.no_grad():
            out = model(input_ids=step, past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[:, -1, :]
    _sync()
    return gen, time.perf_counter() - t0


def _prefill(model, ids: torch.Tensor) -> float:
    _sync()
    t0 = time.perf_counter()
    with torch.no_grad():
        model(input_ids=ids)
    _sync()
    return time.perf_counter() - t0


def _measure_variant(cfg: dict, name: str, prompts: list[dict]) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    if name == "bf16":
        tok, model, attn = _load_baseline(cfg, "sdpa")
    else:
        tok, model, attn = _load_variant(cfg, name)
    n_new = int(cfg["tokens_per_generation"])
    warmup_ids = encode_prompt(tok, prompts[0]["text"])
    _autoregressive(model, tok, warmup_ids, n_new)
    torch.cuda.reset_peak_memory_stats()
    rows = []
    for row in prompts[:N_TIMED]:
        ids = encode_prompt(tok, row["text"])
        tokens, seconds = _autoregressive(model, tok, ids, n_new)
        full, prompt_len = _full_ids(tok, row["text"], tokens, input_device(model))
        prefill_s = _prefill(model, full)
        rows.append(
            {
                "prompt_id": row["prompt_id"],
                "prompt_len": prompt_len,
                "new_tokens": n_new,
                "autoregressive_seconds": seconds,
                "tokens_per_sec": n_new / seconds,
                "prefill_seconds": prefill_s,
            }
        )
        log(f"task08b {name} prompt {row['prompt_id']} ar={seconds:.3f}s prefill={prefill_s:.3f}s")
    peak = int(torch.cuda.max_memory_allocated())
    summary = {
        "attn": attn,
        "n_timed": len(rows),
        "warmup_excluded": 1,
        "autoregressive_seconds_mean": sum(r["autoregressive_seconds"] for r in rows) / len(rows),
        "tokens_per_sec_mean": sum(r["tokens_per_sec"] for r in rows) / len(rows),
        "prefill_seconds_mean": sum(r["prefill_seconds"] for r in rows) / len(rows),
        "peak_vram_bytes": peak,
        "peak_vram_mib": peak / (1024**2),
        "requests": rows,
    }
    del model, tok
    release()
    return summary


def _stake(savings_usd: float) -> dict:
    out = {}
    for rate in (0.01, 0.05, 0.10):
        if savings_usd <= 0:
            minimum = 0.0
        else:
            minimum = savings_usd * REQUESTS_PER_EPOCH / rate
        out[f"{rate:.2f}"] = minimum
    return out


def main() -> int:
    cfg = load_config()
    prompts = _prompts()
    measured = {}
    for name in ("bf16", "quant_8bit", "quant_4bit"):
        log(f"task08b load {name}")
        measured[name] = _measure_variant(cfg, name, prompts)
    bf16_s = measured["bf16"]["autoregressive_seconds_mean"]
    usd_per_s = HOURLY_USD / 3600.0
    comparisons = {}
    for name in ("quant_8bit", "quant_4bit"):
        sec = measured[name]["autoregressive_seconds_mean"]
        saved_s = bf16_s - sec
        saved_usd = saved_s * usd_per_s
        comparisons[name] = {
            "seconds_saved_per_request": saved_s,
            "usd_saved_per_request": saved_usd,
            "minimum_stake_usd": _stake(saved_usd),
        }
    validator_s = measured["bf16"]["prefill_seconds_mean"]
    report = {
        "gpu": "NVIDIA A10G",
        "hourly_usd": HOURLY_USD,
        "hourly_usd_source": "experiment plan on-demand g5.xlarge rate",
        "requests_per_epoch": REQUESTS_PER_EPOCH,
        "epoch_meaning": "one submitted proof; one upheld challenge slashes the full stake",
        "inequality": "savings_per_request < p_sample * stake_slashed / requests_per_epoch",
        "serving": measured,
        "validator_prefill_seconds_mean": validator_s,
        "validator_over_autoregressive": validator_s / bf16_s,
        "vs_bf16": comparisons,
    }
    path = ROOT / "results" / "cost_asymmetry.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {path}")
    q4 = comparisons["quant_4bit"]["minimum_stake_usd"]["0.10"]
    q8 = comparisons["quant_8bit"]["minimum_stake_usd"]["0.10"]
    log(f"task08b 10% stake floor usd 8bit={q8:.8f} 4bit={q4:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
