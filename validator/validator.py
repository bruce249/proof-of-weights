"""Sample traffic, prefill, compare proofs, classify via extract_deltas."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from fingerprint.extract_deltas import ROOT, extract_from_topk_pair, layer_features
from fingerprint.train_classifier import _vectors


def live_features(base_values, base_indices, other_values, other_indices):
    """Must match the offline path. Do not copy feature math into this file."""
    return extract_from_topk_pair(base_values, base_indices, other_values, other_indices)


def _zscore(vec: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return (vec - mu) / sd


def feature_path_parity(cfg: dict, n: int = 10) -> dict:
    """Live extraction vs offline parquet vectors on n Phase B sequences."""
    import joblib
    import pandas as pd

    atol = float(cfg["feature_path_parity_atol"])
    n_layers = int(cfg["n_layers"])
    payload = joblib.load(ROOT / "artifacts" / "classifier.pkl")
    mu = np.asarray(payload["zscore_mean"], dtype=np.float64)
    sd = np.asarray(payload["zscore_std"], dtype=np.float64)
    df = pd.read_parquet(ROOT / "data" / "features_matched.parquet")
    offline_x, offline_y, offline_p = _vectors(df, n_layers)
    # One tamper class, first n prompt ids. Clean pools three axes, so it is not a single npz pair.
    label = int(cfg["tamper_class_ids"]["quant_4bit"])
    mask = offline_y == label
    prompt_ids = [int(p) for p in offline_p[mask][:n]]
    if len(prompt_ids) != n:
        raise RuntimeError(f"need {n} quant_4bit prompts, found {len(prompt_ids)}")

    phase = ROOT / "data" / "phase_b" / "matched"
    with np.load(phase / "quant_4bit.npz") as other, np.load(phase / "baseline_on_quant_4bit.npz") as base:
        diffs = []
        rows = []
        for pid in prompt_ids:
            live = live_features(base["values"][pid], base["indices"][pid], other["values"][pid], other["indices"][pid])
            off = offline_x[(offline_p == pid) & (offline_y == label)]
            if off.shape != (1, live.shape[0]):
                raise RuntimeError(f"offline rows for prompt {pid}: {off.shape}")
            live_z = _zscore(live, mu, sd)
            off_z = _zscore(off[0], mu, sd)
            delta = float(np.max(np.abs(live_z - off_z)))
            diffs.append(delta)
            rows.append({"prompt_id": pid, "label": label, "max_abs_diff": delta})

    report = {
        "n": n,
        "variant": "quant_4bit",
        "label": label,
        "atol": atol,
        "max_abs_diff": float(max(diffs)),
        "passed": bool(max(diffs) <= atol),
        "prompts": rows,
    }
    path = ROOT / "results" / "feature_path_parity.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not report["passed"]:
        raise RuntimeError(f"feature path parity failed: max abs diff {report['max_abs_diff']} > {atol}")
    return report


def run(cfg: dict) -> None:
    raise NotImplementedError("TASK-08 live cycle")


__all__ = ["live_features", "run", "layer_features", "feature_path_parity"]
