"""CPU-only checks for the shared feature path. No GPU."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fingerprint.extract_deltas import extract_from_topk_pair, layer_features
from inference.hooks import prefill_positions


def test_identical_topk_near_zero_mismatch():
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    i = np.array([0, 1, 2], dtype=np.int32)
    f = layer_features(v, i, v, i)
    assert f["idx_mismatch"] == 0.0
    assert f["exp_mismatch"] == 0.0
    assert f["mantissa_mean"] == 0.0


def test_extract_shape():
    rng = np.random.default_rng(0)
    c, l, k = 4, 3, 8
    bv = rng.standard_normal((c, l, k)).astype(np.float32)
    ov = bv + 0.01
    bi = rng.integers(0, 100, (c, l, k), dtype=np.int32)
    oi = bi.copy()
    vec = extract_from_topk_pair(bv, bi, ov, oi)
    assert vec.shape == (4 * l,)


def test_prefill_positions():
    pos = prefill_positions(prompt_len=10, tokens_per_generation=128, interval=32)
    assert pos == [10, 42, 74, 106]


if __name__ == "__main__":
    test_identical_topk_near_zero_mismatch()
    test_extract_shape()
    test_prefill_positions()
    print("ok")
