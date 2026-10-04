"""Phase A recalibration without the eager-attention axis.

The original run kept only aggregate honest statistics, so this replays honest
traffic and re-verifies variant proofs once more, and this time keeps every
per-sequence statistic. Nothing in results/thresholds.json or
results/phase_a_gate.json is overwritten.

Stages:
  python -m inference.recalibrate honest    # GPU: 3 honest axes -> results/honest_seq_stats.json
  python -m inference.recalibrate variants  # GPU: 4 variants    -> results/variant_seq_stats.json
  python -m inference.recalibrate analyze   # CPU: -> results/phase_a_recalibrated.json

Each GPU stage skips work already saved, so it can be restarted.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
HONEST = ROOT / "results" / "honest_seq_stats.json"
VARIANT = ROOT / "results" / "variant_seq_stats.json"
OUT = ROOT / "results" / "phase_a_recalibrated.json"
AXES = [
    {"name": "attn_eager", "attn": "eager", "batch": 1, "extra_pad": 0},
    {"name": "batch4", "attn": "sdpa", "batch": 4, "extra_pad": 0},
    {"name": "padded_batch_length", "attn": "sdpa", "batch": 4, "extra_pad": 17},
]
VARIANTS = ("quant_8bit", "quant_4bit", "lora_merged", "ft_drift")
STATS = ("exp_mismatches_max", "mant_err_mean_max", "mant_err_median_max")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _write(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def honest(cfg: dict) -> None:
    """Same replay as task02.run_clean_and_calibrate, keeping per-sequence stats."""
    import torch

    from inference.hooks import TopKRecorder, decoder_layers
    from inference.task02 import (
        LastHiddenTap,
        _seq_stats,
        _set_attn,
        _verify,
        encode_prompt,
        load_model,
        replay_batch,
        replay_one,
    )

    prompts = json.loads((ROOT / "data" / "prompts.json").read_text(encoding="utf-8"))["prompts"]
    proofs = []
    with (ROOT / "data" / "baseline" / "proofs.jsonl").open(encoding="utf-8") as f:
        for line in f:
            proofs.append([bytes.fromhex(h) for h in json.loads(line)["proofs_hex"]])

    tok, model = load_model(cfg, attn="sdpa")
    rec = TopKRecorder(interval=cfg["proof_interval_tokens"], k=cfg["topk"])
    rec.attach(decoder_layers(model))
    tap = LastHiddenTap()
    tap.attach(model)

    saved = _read(HONEST)
    saved.setdefault("axes", {})
    for axis in AXES:
        name = axis["name"]
        if len(saved["axes"].get(name, {}).get("seq_stats", [])) == len(prompts):
            print(f"honest {name} already saved", flush=True)
            continue
        _set_attn(model, axis["attn"])
        bsz, extra_pad = int(axis["batch"]), int(axis["extra_pad"])
        rows_out = []
        for start in range(0, len(prompts), bsz):
            chunk = prompts[start : start + bsz]
            ids_list = [encode_prompt(tok, p["text"]) for p in chunk]
            toks_list = [p["tokens"] for p in chunk]
            if bsz == 1:
                rows = [replay_one(model, tok, rec, tap, ids_list[0], toks_list[0], cfg)]
            else:
                rows = replay_batch(model, tok, rec, tap, ids_list, toks_list, cfg, extra_pad=extra_pad)
            for off, row in enumerate(rows):
                i = start + off
                s = _seq_stats(_verify(row["hiddens"], proofs[i], cfg))
                s["prompt_id"] = i
                rows_out.append(s)
                if i % 20 == 0:
                    print(f"honest {name} [{i:03d}] exp={s['exp_mismatches_max']:.0f}", flush=True)
            del rows
            torch.cuda.empty_cache()
        saved["axes"][name] = {"axis": axis, "seq_stats": rows_out}
        saved["written_at"] = _now()
        _write(HONEST, saved)
        print(f"wrote {HONEST} axis={name}", flush=True)
    rec.detach()
    tap.detach()


def variants(cfg: dict) -> None:
    """Same check as task04.phase_a_gate, keeping per-sequence stats."""
    from inference.build_variants import input_device, release
    from inference.task02 import _seq_stats, _verify
    from inference.task04 import _full_ids, _load_baseline, _prefill_decode_hiddens

    prompts = json.loads((ROOT / "data" / "prompts.json").read_text(encoding="utf-8"))["prompts"]
    saved = _read(VARIANT)
    saved.setdefault("variants", {})
    tok, model, attn = _load_baseline(cfg, "sdpa")
    n_new = cfg["tokens_per_generation"]
    for vid in VARIANTS:
        if len(saved["variants"].get(vid, [])) == len(prompts):
            print(f"variant {vid} already saved", flush=True)
            continue
        seqs = {s["prompt_id"]: s for s in json.loads((ROOT / "data" / vid / "sequences.json").read_text(encoding="utf-8"))}
        proofs = []
        with (ROOT / "data" / vid / "proofs.jsonl").open(encoding="utf-8") as f:
            for line in f:
                proofs.append([bytes.fromhex(h) for h in json.loads(line)["proofs_hex"]])
        rows_out = []
        for row in prompts:
            pid = row["prompt_id"]
            ids, prompt_len = _full_ids(tok, row["text"], seqs[pid]["tokens"], input_device(model))
            hiddens = _prefill_decode_hiddens(model, ids, prompt_len, n_new)
            s = _seq_stats(_verify(hiddens, proofs[pid], cfg))
            s["prompt_id"] = pid
            rows_out.append(s)
            if pid % 20 == 0:
                print(f"variant {vid} [{pid:03d}] exp={s['exp_mismatches_max']:.0f}", flush=True)
        saved["variants"][vid] = rows_out
        saved["attn"] = attn
        saved["written_at"] = _now()
        _write(VARIANT, saved)
        print(f"wrote {VARIANT} variant={vid}", flush=True)
    del model, tok
    release()


def _arr(rows: list[dict], ids: range) -> np.ndarray:
    by = {r["prompt_id"]: r for r in rows}
    return np.array([[by[i][k] for k in STATS] for i in ids], dtype=np.float64)


def _flag(x: np.ndarray, t: np.ndarray) -> np.ndarray:
    return np.any(x > t[None, :], axis=1)


def analyze(cfg: dict) -> None:
    h = _read(HONEST)["axes"]
    v = _read(VARIANT)["variants"]
    n = int(cfg["n_prompts"])
    calib, val, every = range(cfg["calibration_prompts"]), range(cfg["calibration_prompts"], n), range(n)
    paper = np.array([cfg["paper_thresholds"][k] for k in ("T_exp", "T_mean", "T_median")], dtype=np.float64)
    old = json.loads((ROOT / "results" / "thresholds.json").read_text(encoding="utf-8"))["calibrated"]

    def fit(axes: list[str]) -> np.ndarray:
        pooled = np.vstack([_arr(h[a]["seq_stats"], calib) for a in axes])
        return np.percentile(pooled, 99, axis=0)

    def report(t: np.ndarray, axes: list[str]) -> dict:
        return {
            "thresholds": {"T_exp": float(t[0]), "T_mean": float(t[1]), "T_median": float(t[2])},
            "calibrated_on_axes": axes,
            "heldout_fpr_calibration_axes": float(np.mean(np.concatenate(
                [_flag(_arr(h[a]["seq_stats"], val), t) for a in axes]))),
            "heldout_fpr_by_axis": {a: float(np.mean(_flag(_arr(h[a]["seq_stats"], val), t))) for a in h},
            "detect_rate": {vid: float(np.mean(_flag(_arr(v[vid], every), t))) for vid in VARIANTS if vid in v},
        }

    all_axes = [a["name"] for a in AXES]
    quiet = ["batch4", "padded_batch_length"]
    t_all = fit(all_axes)
    out = {
        "written_at": _now(),
        "split": "calibrate on prompts 0-99, held-out FPR on 100-199, detect rate on 0-199",
        "sanity_original_thresholds": old,
        "all_three_axes": report(t_all, all_axes),
        "without_eager": report(fit(quiet), quiet),
        "published_thresholds": report(paper, all_axes),
        "per_variant_medians": {vid: dict(zip(STATS, np.median(_arr(v[vid], every), axis=0).tolist())) for vid in v},
        "per_axis_medians": {a: dict(zip(STATS, np.median(_arr(h[a]["seq_stats"], every), axis=0).tolist())) for a in h},
    }
    _write(OUT, out)
    print(json.dumps({k: out[k] for k in ("all_three_axes", "without_eager", "published_thresholds")}, indent=1))
    print(f"wrote {OUT}")


def main() -> int:
    import yaml

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    if stage in ("all", "honest"):
        honest(cfg)
    if stage in ("all", "variants"):
        variants(cfg)
    if stage in ("all", "analyze"):
        analyze(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
