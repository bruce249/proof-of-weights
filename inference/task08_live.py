"""One tampered serve on the A10G. Chain txs are sent from the laptop."""

from __future__ import annotations

import json

import numpy as np

from inference.hooks import TopKRecorder, decoder_layers
from inference.task02 import (
    LastHiddenTap,
    _proofs_from_hiddens,
    _seq_stats,
    _verify,
    encode_prompt,
    generate_one,
    load_config,
)
from inference.task04 import (
    ROOT,
    _flagged,
    _full_ids,
    _load_baseline,
    _load_variant,
    _prefill_decode_hiddens,
    _prefill_topk,
    _prompts,
    _thresholds,
    release,
)
from inference.build_variants import input_device, log


def serve_until_flagged(cfg: dict, variant_id: str = "ft_drift", limit: int = 16) -> dict:
    prompts = _prompts()[:limit]
    th = _thresholds()
    log(f"task08 load {variant_id}")
    tok, model, _attn = _load_variant(cfg, variant_id)
    rec = TopKRecorder(interval=cfg["proof_interval_tokens"], k=cfg["topk"])
    rec.attach(decoder_layers(model))
    tap = LastHiddenTap()
    tap.attach(model)
    served = []
    for row in prompts:
        ids = encode_prompt(tok, row["text"]).to(input_device(model))
        gen = generate_one(model, tok, rec, tap, ids, cfg)
        served.append(
            {
                "prompt_id": row["prompt_id"],
                "text": row["text"],
                "prompt_len": gen["prompt_len"],
                "tokens": gen["tokens"],
                "values": gen["values"],
                "indices": gen["indices"],
                "proofs": gen["proofs"],
            }
        )
        log(f"task08 served prompt {row['prompt_id']}")
    rec.detach()
    tap.detach()
    del model, tok, rec, tap
    release()

    log("task08 load baseline")
    btok, base, attn = _load_baseline(cfg, "sdpa")
    brec = TopKRecorder(interval=cfg["proof_interval_tokens"], k=cfg["topk"])
    brec.attach(decoder_layers(base))
    chosen = None
    for item in served:
        ids, prompt_len = _full_ids(btok, item["text"], item["tokens"], input_device(base))
        if prompt_len != item["prompt_len"]:
            raise RuntimeError(f"prompt_len {prompt_len} != {item['prompt_len']}")
        brec.mode = None
        hiddens = _prefill_decode_hiddens(base, ids, prompt_len, cfg["tokens_per_generation"])
        stats = _seq_stats(_verify(hiddens, item["proofs"], cfg))
        flagged = _flagged(stats, th)
        values, indices = _prefill_topk(base, brec, ids, prompt_len, item["tokens"], cfg)
        log(f"task08 prompt {item['prompt_id']} flagged={flagged} exp={stats['exp_mismatches_max']:.2f}")
        if flagged and chosen is None:
            chosen = {
                "prompt_id": item["prompt_id"],
                "variant": variant_id,
                "attn": attn,
                "thresholds": th,
                "proof_stats": stats,
                "flagged": True,
                "provider_values": item["values"],
                "provider_indices": item["indices"],
                "baseline_values": values.numpy(),
                "baseline_indices": indices.numpy(),
                "proofs_hex": [p.hex() for p in item["proofs"]],
            }
    brec.detach()
    del base, btok, brec
    release()
    if chosen is None:
        raise RuntimeError(f"no {variant_id} proof exceeded calibrated thresholds in {limit} prompts")

    out_dir = ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / "live_topk.npz",
        provider_values=chosen.pop("provider_values"),
        provider_indices=chosen.pop("provider_indices"),
        baseline_values=chosen.pop("baseline_values"),
        baseline_indices=chosen.pop("baseline_indices"),
    )
    path = out_dir / "live_decision.json"
    path.write_text(json.dumps(chosen, indent=2) + "\n", encoding="utf-8")
    log(f"wrote {path}")
    return chosen


def main() -> int:
    serve_until_flagged(load_config())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
