"""Decode-time per-layer analysis and the kernel attack. CPU only.

Provider side: per-layer top-k recorded during autoregressive decode (Phase A
and the honest replays), which is what a provider would commit to.
Validator side: baseline SDPA prefill over the same tokens (data/phase_b).

Honest configurations (validator reference = baseline prefill on baseline text):
  decode_sdpa_b1   baseline decode itself (same kernel, decode vs prefill)
  attn_eager, batch4, padded_batch_length   honest decode replays
Tampered (reference = baseline prefill on the variant's own text):
  <variant>        default kernel (SDPA, batch 1) decode
  <variant>_eager_b1, <variant>_sdpa_b4   kernel attack, prompts 100-199

Writes results/decode_analysis.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, roc_curve
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fingerprint.extract_deltas import FEATURE_KEYS, _features_for_pair, _load_pair  # noqa: E402

SEED = 20260923
VARIANTS = ("quant_8bit", "quant_4bit", "lora_merged", "ft_drift")
LABEL = {"quant_8bit": 1, "quant_4bit": 2, "lora_merged": 3, "ft_drift": 4}
HONEST = ("decode_sdpa_b1", "attn_eager", "batch4", "padded_batch_length")
DEC = ROOT / "data" / "decode"
PB = ROOT / "data" / "phase_b"
ATT = ROOT / "data" / "kernel_attack"
CACHE = ROOT / "data" / "decode_cubes.npz"


def cube(base_path: Path, other_v, other_i, rows=None) -> np.ndarray:
    bv, bi = _load_pair(base_path)
    if rows is not None:
        bv, bi = bv[rows], bi[rows]
    f = _features_for_pair(bv, bi, other_v, other_i)
    return np.stack([f[k] for k in FEATURE_KEYS], axis=2)  # [P, C, 4, L]


def load() -> dict[str, np.ndarray]:
    cubes = dict(np.load(CACHE)) if CACHE.exists() else {}
    base_text = PB / "shared" / "baseline.npz"
    want = {
        "decode_sdpa_b1": (base_text, DEC / "baseline" / "layer_topk.npz", None),
        **{a: (base_text, DEC / "clean_perturbed" / f"{a}_layer_topk.npz", None) for a in HONEST[1:]},
        **{v: (PB / "matched" / f"baseline_on_{v}.npz", DEC / v / "layer_topk.npz", None) for v in VARIANTS},
    }
    for v in VARIANTS:
        for c in ("eager_b1", "sdpa_b4"):
            p = ATT / f"{v}_{c}.npz"
            if p.exists():
                want[f"{v}_{c}"] = (PB / "matched" / f"baseline_on_{v}.npz", p, "attack")
    changed = False
    for name, (bp, op, kind) in want.items():
        if name in cubes:
            continue
        print("features", name, flush=True)
        ov, oi = _load_pair(op)
        rows = None
        if kind == "attack":
            with np.load(op) as z:
                rows = z["prompt_ids"]
        cubes[name] = cube(bp, ov, oi, rows)
        changed = True
    if changed:
        np.savez_compressed(CACHE, **cubes)
    return cubes


def vec(c: np.ndarray, layers=None) -> np.ndarray:
    m = c.mean(axis=1)
    if layers is not None:
        m = m[:, :, layers]
    return m.reshape(len(m), -1)


def table(cubes, layers=None):
    X, y, g, src = [], [], [], []
    for h in HONEST:
        x = vec(cubes[h], layers)
        X.append(x); y += [0] * len(x); g += list(range(len(x))); src += [h] * len(x)
    for v in VARIANTS:
        x = vec(cubes[v], layers)
        X.append(x); y += [LABEL[v]] * len(x); g += list(range(len(x))); src += [v] * len(x)
    return np.vstack(X), np.array(y), np.array(g), np.array(src)


def fit(X, y):
    mu, sd = X.mean(0), X.std(0)
    sd[sd == 0] = 1.0
    clf = LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=5000).fit((X - mu) / sd, y)
    return lambda Z: clf.predict((Z - mu) / sd), lambda Z: clf.predict_proba((Z - mu) / sd)


def cv(X, y, g):
    pred = np.zeros(len(y), int)
    proba = np.zeros((len(y), 5))
    folds = []
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        p, pp = fit(X[tr], y[tr])
        pred[te] = p(X[te])
        proba[te] = pp(X[te])
        folds.append(float(balanced_accuracy_score(y[te], pred[te])))
    m, sd = float(np.mean(folds)), float(np.std(folds, ddof=1))
    half = 2.776 * sd / np.sqrt(5)
    return pred, proba, {"mean": m, "ci95": [m - half, m + half], "folds": folds}


def tpr_at(honest_scores, tamper_scores, fpr_target=0.01) -> float:
    lab = np.r_[np.zeros(len(honest_scores)), np.ones(len(tamper_scores))]
    fpr, tpr, _ = roc_curve(lab, np.r_[honest_scores, tamper_scores])
    return float(np.interp(fpr_target, fpr, tpr))


def main() -> None:
    cubes = load()
    out = {"note": "provider decode-time per-layer top-k vs validator SDPA prefill; GroupKFold 5 by prompt"}

    X, y, g, src = table(cubes)
    pred, proba, folds = cv(X, y, g)
    score = 1 - proba[:, 0]
    out["decode_cv"] = {
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "fold": folds,
        "honest_flag_rate": {h: float(np.mean(pred[src == h] != 0)) for h in HONEST},
        "correct_class_rate": {v: float(np.mean(pred[src == v] == LABEL[v])) for v in VARIANTS},
        "tpr_at_1pct_fpr": {v: tpr_at(score[y == 0], score[src == v]) for v in VARIANTS},
    }
    Xl, yl, gl, sl = table(cubes, layers=[27])
    pl, _, fl = cv(Xl, yl, gl)
    out["decode_cv_last_layer_only"] = {
        "balanced_accuracy": float(balanced_accuracy_score(yl, pl)), "fold": fl,
        "honest_flag_rate": {h: float(np.mean(pl[sl == h] != 0)) for h in HONEST},
    }

    # transfer: trained on prefill-vs-prefill profiles (paper_analysis cubes), tested on decode
    pc = ROOT / "data" / "paper_cube_matched.npz"
    if pc.exists():
        with np.load(pc) as z:
            pre = {k: z[k] for k in z.files}
        Xp = np.vstack([vec(pre[f"clean_{a}"]) for a in HONEST[1:]] + [vec(pre[v]) for v in VARIANTS])
        yp = np.r_[np.zeros(600, int), np.concatenate([np.full(200, LABEL[v]) for v in VARIANTS])]
        p, _ = fit(Xp, yp)
        pt = p(X)
        out["prefill_trained_on_decode"] = {
            "balanced_accuracy": float(balanced_accuracy_score(y, pt)),
            "honest_flag_rate": {h: float(np.mean(pt[src == h] != 0)) for h in HONEST},
            "correct_class_rate": {v: float(np.mean(pt[src == v] == LABEL[v])) for v in VARIANTS},
        }

    # kernel attack: train on decode data from prompts 0-99, test attack sequences (prompts 100-199)
    attack = [k for k in cubes if k.endswith(("_eager_b1", "_sdpa_b4"))]
    if attack:
        tr = g < 100
        p, pp = fit(X[tr], y[tr])
        te_h = (~tr) & (y == 0)
        thr = np.quantile(1 - pp(X[te_h])[:, 0], 0.99)
        res = {"train": "decode data, prompts 0-99, all honest configs + default-kernel tampered",
               "test_honest_flag_rate": float(np.mean(p(X[te_h]) != 0)),
               "attacks": {}}
        for k in sorted(attack):
            v = k.rsplit("_", 2)[0]
            Xa = vec(cubes[k])
            pa = p(Xa)
            sa = 1 - pp(Xa)[:, 0]
            res["attacks"][k] = {
                "n": len(Xa),
                "detected_rate": float(np.mean(pa != 0)),
                "correct_class_rate": float(np.mean(pa == LABEL[v])),
                "predicted_counts": np.bincount(pa, minlength=5).tolist(),
                "detected_at_1pct_heldout_fpr": float(np.mean(sa > thr)),
            }
        out["kernel_attack_per_layer"] = res

    st = ROOT / "results" / "kernel_attack_seq_stats.json"
    if st.exists():
        stats = json.loads(st.read_text(encoding="utf-8"))
        th = json.loads((ROOT / "results" / "phase_a_recalibrated.json").read_text(encoding="utf-8"))
        sets = {k: th[k]["thresholds"] for k in ("all_three_axes", "without_eager", "published_thresholds")}
        lh = {}
        for k, row in stats.items():
            S = np.array([[r["exp_mismatches_max"], r["mant_err_mean_max"], r["mant_err_median_max"]] for r in row["seq_stats"]])
            lh[k] = {
                "n": len(S),
                "mean_stats": S.mean(0).tolist(),
                "detected": {n: float(np.mean(np.any(S > np.array([t["T_exp"], t["T_mean"], t["T_median"]]), axis=1)))
                             for n, t in sets.items()},
            }
        out["kernel_attack_last_hidden"] = lh

    (ROOT / "results" / "decode_analysis.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "note"}, indent=1))


if __name__ == "__main__":
    main()
