"""50 honest bf16 sequences on this GPU. Score them on the laptop."""

from __future__ import annotations

import json

import numpy as np
import torch

from inference.build_variants import input_device, log, release
from inference.hooks import TopKRecorder, decoder_layers
from inference.task02 import (
    LastHiddenTap,
    _seq_stats,
    _verify,
    encode_prompt,
    load_config,
    replay_one,
)
from inference.task04 import (
    ROOT,
    _full_ids,
    _load_baseline,
    _prefill_decode_hiddens,
    _prefill_topk,
)


N = 50


def main() -> int:
    cfg = load_config()
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("bf16 is not supported on this GPU")
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    log(f"cross gpu={name} cc={major}.{minor} bf16=True")
    prompts = json.loads((ROOT / "data" / "prompts.json").read_text(encoding="utf-8"))["prompts"][:N]
    tok, model, attn = _load_baseline(cfg, "sdpa")
    rec = TopKRecorder(interval=cfg["proof_interval_tokens"], k=cfg["topk"])
    rec.attach(decoder_layers(model))
    tap = LastHiddenTap()
    tap.attach(model)
    n_layers = len(decoder_layers(model))
    n_ck = cfg["tokens_per_generation"] // cfg["proof_interval_tokens"]
    values = np.zeros((N, n_ck, n_layers, cfg["topk"]), dtype=np.float32)
    indices = np.zeros((N, n_ck, n_layers, cfg["topk"]), dtype=np.int32)
    rows = []
    for row in prompts:
        ids = encode_prompt(tok, row["text"])
        gen = replay_one(model, tok, rec, tap, ids, row["tokens"], cfg)
        rec.mode = None
        full, prompt_len = _full_ids(tok, row["text"], row["tokens"], input_device(model))
        if prompt_len != row["prompt_len"]:
            raise RuntimeError(f"prompt {row['prompt_id']} len {prompt_len} != {row['prompt_len']}")
        hiddens = _prefill_decode_hiddens(model, full, prompt_len, cfg["tokens_per_generation"])
        stats = _seq_stats(_verify(hiddens, gen["proofs"], cfg))
        pv, pi = _prefill_topk(model, rec, full, prompt_len, row["tokens"], cfg)
        values[row["prompt_id"]] = pv.numpy()
        indices[row["prompt_id"]] = pi.numpy()
        rows.append({"prompt_id": row["prompt_id"], **stats})
        log(
            f"cross [{row['prompt_id']:03d}] exp={stats['exp_mismatches_max']:.2f} "
            f"mean={stats['mant_err_mean_max']:.2f} med={stats['mant_err_median_max']:.2f}"
        )
    rec.detach()
    tap.detach()
    del model, tok
    release()
    out = ROOT / "results"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "l4_prefill_topk.npz", values=values, indices=indices)
    payload = {
        "gpu_name": name,
        "compute_capability": f"{major}.{minor}",
        "bf16": True,
        "attn": attn,
        "n": N,
        "protocol": "L4 teacher-forced decode proof verified by an L4 prefill of the same baseline tokens",
        "sequences": rows,
    }
    (out / "cross_hardware_raw.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    log("cross wrote raw results")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
