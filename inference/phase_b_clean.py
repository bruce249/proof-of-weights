"""Phase B prefill for the three kept honest-noise axes. Label 0.

Same baseline tokens through the perturbed honest forward and, on the other
side, the unperturbed baseline prefill already stored in shared/baseline.npz.
Not a model-vs-itself comparison.
"""

from __future__ import annotations

import shutil

import numpy as np
import torch

from inference.hooks import TopKRecorder, decoder_layers, stack_buf
from inference.task02 import encode_prompt, n_checkpoints
from inference.task04 import ROOT, _full_ids, _load_baseline, _prompts, log
from inference.build_variants import input_device, release

AXES = (
    {"name": "attn_eager", "attn": "eager", "batch": 1, "extra_pad": 0},
    {"name": "batch4", "attn": "sdpa", "batch": 4, "extra_pad": 0},
    {"name": "padded_batch_length", "attn": "sdpa", "batch": 4, "extra_pad": 17},
)


def _prefill_chunk(model, tok, rec, rows, cfg, extra_pad: int):
    n_new = cfg["tokens_per_generation"]
    interval = cfg["proof_interval_tokens"]
    n_ck = n_checkpoints(cfg)
    device = input_device(model)
    full = []
    for row in rows:
        ids, _prompt_len = _full_ids(tok, row["text"], row["tokens"], "cpu")
        full.append(ids.cpu())
    max_len = max(int(x.shape[1]) for x in full) + extra_pad
    padded = []
    for ids in full:
        need = max_len - int(ids.shape[1])
        if need:
            pad = torch.full((1, need), tok.pad_token_id, dtype=ids.dtype)
            ids = torch.cat([pad, ids], dim=1)
        padded.append(ids)
    batch = torch.cat(padded, dim=0).to(device)
    attn = (batch != tok.pad_token_id).long()
    gen_start = max_len - n_new
    positions = [gen_start + interval * k for k in range(n_ck)]
    for b, row in enumerate(rows):
        expected = [row["tokens"][interval * k] for k in range(n_ck)]
        got = batch[b, positions].detach().cpu().tolist()
        if got != expected:
            raise AssertionError(f"token ids at {positions} mismatch: {expected} vs {got}")
    rec.reset("prefill", positions=positions)
    with torch.no_grad():
        model(input_ids=batch, attention_mask=attn)
    n_layers = len(decoder_layers(model))
    rec.assert_prefill_complete(n_layers)
    out = []
    for b in range(len(rows)):
        values, indices = stack_buf(rec.buf, n_ck, n_layers, cfg["topk"], batch_index=b)
        out.append((values.numpy(), indices.numpy()))
    return out


def _save(path, values, indices) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, values=values, indices=indices)
    log(f"wrote {path} {values.shape}")


def _mean_idx_mismatch(base_i: np.ndarray, other_i: np.ndarray) -> float:
    # Mean fraction of other top-k indices absent from the baseline top-k.
    n = base_i.shape[0] * base_i.shape[1] * base_i.shape[2]
    miss = 0.0
    for p in range(base_i.shape[0]):
        for c in range(base_i.shape[1]):
            for ell in range(base_i.shape[2]):
                miss += 1.0 - np.mean(np.isin(other_i[p, c, ell], base_i[p, c, ell]))
    return float(miss / n)


def run(cfg: dict) -> None:
    prompts = _prompts()
    base_path = ROOT / "data" / "phase_b" / "shared" / "baseline.npz"
    with np.load(base_path) as z:
        base_i = np.asarray(z["indices"])
    for axis in AXES:
        name = axis["name"]
        matched = ROOT / "data" / "phase_b" / "matched" / f"clean_{name}.npz"
        shared = ROOT / "data" / "phase_b" / "shared" / f"clean_{name}.npz"
        base_copy = ROOT / "data" / "phase_b" / "matched" / f"baseline_on_clean_{name}.npz"
        if matched.exists() and shared.exists() and base_copy.exists():
            log(f"clean skip {name}")
            continue
        log(f"clean load {name} attn={axis['attn']}")
        tok, model, used = _load_baseline(cfg, axis["attn"])
        if used != axis["attn"]:
            raise RuntimeError(f"{name} attn {used} != {axis['attn']}")
        rec = TopKRecorder(interval=cfg["proof_interval_tokens"], k=cfg["topk"])
        rec.attach(decoder_layers(model))
        n_layers = len(decoder_layers(model))
        values = np.zeros((len(prompts), n_checkpoints(cfg), n_layers, cfg["topk"]), dtype=np.float32)
        indices = np.zeros_like(values, dtype=np.int32)
        bsz = int(axis["batch"])
        for start in range(0, len(prompts), bsz):
            chunk = prompts[start : start + bsz]
            rows = _prefill_chunk(model, tok, rec, chunk, cfg, int(axis["extra_pad"]))
            for off, (v, i) in enumerate(rows):
                values[start + off] = v
                indices[start + off] = i
            if start % 20 == 0:
                log(f"clean {name} [{start:03d}]")
        delta = _mean_idx_mismatch(base_i, indices)
        log(f"clean {name} mean_idx_mismatch={delta:.6g}")
        if delta == 0.0:
            raise RuntimeError(f"{name} is inert against the unperturbed baseline")
        _save(matched, values, indices)
        _save(shared, values, indices)
        if not base_copy.exists():
            shutil.copyfile(base_path, base_copy)
            log(f"copied unperturbed baseline to {base_copy}")
        rec.detach()
        del model, tok, rec
        release()
    log("clean phase B complete")


if __name__ == "__main__":
    from inference.task02 import load_config

    run(load_config())
