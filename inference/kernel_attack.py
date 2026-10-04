"""Tampered weights served under a non-default kernel.

A provider that quantises or fine-tunes could also switch attention kernel or
batch shape to hide inside honest noise. This replays each tampered model's own
tokens (teacher-forced decode, as in Phase A) under eager attention at batch 1
and under SDPA at batch 4, on held-out prompts 100-199. For each sequence it
keeps the last-hidden TOPLOC statistics (verified against a baseline SDPA
prefill) and the decode-time per-layer top-k.

  python -m inference.kernel_attack           # GPU, resumable per (variant, config)

Writes data/kernel_attack/<variant>_<config>.npz and results/kernel_attack_seq_stats.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from inference.build_variants import input_device, log, release
from inference.hooks import TopKRecorder, decoder_layers
from inference.task02 import (
    LastHiddenTap,
    _seq_stats,
    _set_attn,
    _verify,
    encode_prompt,
    load_config,
    replay_batch,
    replay_one,
)
from inference.task04 import _full_ids, _load_baseline, _load_variant, _prefill_decode_hiddens

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "data" / "kernel_attack"
STATS = ROOT / "results" / "kernel_attack_seq_stats.json"
VARIANTS = ("quant_8bit", "quant_4bit", "lora_merged", "ft_drift")
CONFIGS = {
    "eager_b1": {"attn": "eager", "batch": 1},
    "sdpa_b4": {"attn": "sdpa", "batch": 4},
}
PROMPT_IDS = list(range(100, 200))


def _read() -> dict:
    return json.loads(STATS.read_text(encoding="utf-8")) if STATS.exists() else {}


def main() -> int:
    cfg = load_config()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prompts = {p["prompt_id"]: p for p in json.loads((ROOT / "data" / "prompts.json").read_text(encoding="utf-8"))["prompts"]}
    n_new = cfg["tokens_per_generation"]
    saved = _read()

    btok, base, _ = _load_baseline(cfg, "sdpa")
    for vid in VARIANTS:
        todo = [c for c in CONFIGS if f"{vid}_{c}" not in saved or not (OUT_DIR / f"{vid}_{c}.npz").exists()]
        if not todo:
            log(f"{vid} already done")
            continue
        seqs = {s["prompt_id"]: s for s in json.loads((ROOT / "data" / vid / "sequences.json").read_text(encoding="utf-8"))}
        tok, model, _ = _load_variant(cfg, vid)
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        rec = TopKRecorder(interval=cfg["proof_interval_tokens"], k=cfg["topk"])
        rec.attach(decoder_layers(model))
        tap = LastHiddenTap()
        tap.attach(model)
        n_layers = len(decoder_layers(model))
        n_ck = n_new // cfg["proof_interval_tokens"]
        for cname in todo:
            c = CONFIGS[cname]
            _set_attn(model, c["attn"])
            bsz = c["batch"]
            values = np.zeros((len(PROMPT_IDS), n_ck, n_layers, cfg["topk"]), dtype=np.float32)
            indices = np.zeros_like(values, dtype=np.int32)
            rows_out = []
            for start in range(0, len(PROMPT_IDS), bsz):
                pids = PROMPT_IDS[start : start + bsz]
                ids_list = [encode_prompt(tok, prompts[p]["text"]) for p in pids]
                toks_list = [seqs[p]["tokens"] for p in pids]
                if bsz == 1:
                    rows = [replay_one(model, tok, rec, tap, ids_list[0].to(input_device(model)), toks_list[0], cfg)]
                else:
                    rows = replay_batch(model, tok, rec, tap, ids_list, toks_list, cfg)
                for off, row in enumerate(rows):
                    pid = pids[off]
                    j = start + off
                    values[j] = row["values"]
                    indices[j] = row["indices"]
                    full, plen = _full_ids(btok, prompts[pid]["text"], seqs[pid]["tokens"], input_device(base))
                    ref = _prefill_decode_hiddens(base, full, plen, n_new)
                    s = _seq_stats(_verify(ref, row["proofs"], cfg))
                    s["prompt_id"] = pid
                    rows_out.append(s)
                    if pid % 20 == 0:
                        log(f"{vid} {cname} [{pid:03d}] exp={s['exp_mismatches_max']:.0f}")
                del rows
                torch.cuda.empty_cache()
            np.savez_compressed(OUT_DIR / f"{vid}_{cname}.npz", values=values, indices=indices,
                                prompt_ids=np.array(PROMPT_IDS, dtype=np.int32))
            saved = _read()
            saved[f"{vid}_{cname}"] = {"variant": vid, "config": c, "seq_stats": rows_out}
            STATS.write_text(json.dumps(saved, indent=1) + "\n", encoding="utf-8")
            log(f"wrote {vid}_{cname}")
        rec.detach()
        tap.detach()
        del model, tok, rec, tap
        release()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
