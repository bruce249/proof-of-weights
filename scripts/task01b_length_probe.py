"""20-prompt natural-EOS length probe. No filter loop. Run on A10G after TASK-01."""

from __future__ import annotations

import json
import random
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


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


def load_config() -> dict:
    with (ROOT / "config.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


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


def eos_pos(seq: list[int], eos_id: int | None) -> int | None:
    if eos_id is None:
        return None
    for i, tid in enumerate(seq):
        if tid == eos_id:
            return i
    return None


def decide(median_eos: float, n_capped: int, n_total: int) -> dict:
    # max_new_tokens=256 censors longer answers. A majority that never hit EOS
    # means the true median is > 256.
    if n_capped * 2 >= n_total or median_eos > 256:
        return {"tokens_per_generation": 256, "n_checkpoints": 8, "path": "median_gt_256_keep"}
    if median_eos >= 128:
        return {"tokens_per_generation": 128, "n_checkpoints": 4, "path": "median_128_256_no_refilter"}
    return {"tokens_per_generation": 128, "n_checkpoints": 4, "path": "median_lt_128_keep_128_one_refilter_max"}


def main() -> int:
    assert torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    cfg = load_config()
    seed = 20260923
    cfg.setdefault("seeds", {})["length_probe"] = seed

    print("loading", cfg["model"], cfg["hf_revision"])
    tok = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["hf_revision"])
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model"],
        revision=cfg["hf_revision"],
        dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.eval()

    prompts = pick_prompts(20, seed)
    rows = []
    eos_id = tok.eos_token_id
    for i, text in enumerate(prompts):
        if hasattr(tok, "apply_chat_template"):
            rendered = tok.apply_chat_template(
                [{"role": "user", "content": text}],
                add_generation_prompt=True,
                tokenize=False,
            )
            enc = tok(rendered, return_tensors="pt", add_special_tokens=False).input_ids
        else:
            enc = tok(text, return_tensors="pt").input_ids
        if enc.dim() == 1:
            enc = enc.unsqueeze(0)
        enc = enc.to(model.device)
        prompt_len = int(enc.shape[1])
        with torch.no_grad():
            out = model.generate(
                enc,
                max_new_tokens=256,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        new_ids = out[0, prompt_len:].tolist()
        pos = eos_pos(new_ids, eos_id)
        natural = pos if pos is not None else len(new_ids)
        rows.append(
            {
                "i": i,
                "prompt_len": prompt_len,
                "n_new": len(new_ids),
                "natural_eos_pos": natural,
                "hit_eos": pos is not None,
                "prompt_preview": text[:240],
            }
        )
        print(f"[{i:02d}] prompt_len={prompt_len} eos={natural} hit={pos is not None}")

    eos_vals = [r["natural_eos_pos"] for r in rows]
    median = float(statistics.median(eos_vals))
    n_capped = sum(1 for r in rows if not r["hit_eos"])
    decision = decide(median, n_capped, len(rows))
    # median < 128: one refilter already encoded in pick_prompts (long-answer markers).
    payload = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "median_natural_eos": median,
        "mean_natural_eos": float(statistics.mean(eos_vals)),
        "min_natural_eos": min(eos_vals),
        "max_natural_eos": max(eos_vals),
        "n_hit_eos": sum(1 for r in rows if r["hit_eos"]),
        **decision,
        "rows": rows,
    }
    out_path = ROOT / "results" / "length_probe.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    cfg["tokens_per_generation"] = decision["tokens_per_generation"]
    cfg["min_new_tokens"] = decision["tokens_per_generation"]
    with (ROOT / "config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    print("wrote", out_path, "median", median, decision)
    return 0


if __name__ == "__main__":
    sys.exit(main())
