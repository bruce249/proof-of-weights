"""Offline analyses for the paper. CPU only, reads data/phase_b/*.npz.

Adds what results/metrics.json does not have:
  1. clean axes kept separate (one honest vector per prompt per axis, not averaged)
  2. leave-one-axis-out honest false-positive rate
  3. layer and feature-group ablations
  4. binary clean-vs-tampered ROC from out-of-fold probabilities
  5. number of checkpoints

Same feature path (fingerprint.extract_deltas.layer_features), same split
(GroupKFold 5 by prompt, shuffle off), same logistic regression settings.
Writes results/paper_analysis.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, roc_auc_score, roc_curve
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fingerprint.extract_deltas import FEATURE_KEYS, _features_for_pair, _load_pair  # noqa: E402

PHASE_B = ROOT / "data" / "phase_b"
SEED = 20260923
N_LAYERS = 28
AXES = ("attn_eager", "batch4", "padded_batch_length")
VARIANTS = ("quant_8bit", "quant_4bit", "lora_merged", "ft_drift")
LABEL = {"clean": 0, "quant_8bit": 1, "quant_4bit": 2, "lora_merged": 3, "ft_drift": 4}


def _cube(config: str, stem: str) -> np.ndarray:
    """[prompts, checkpoints, 4, layers] raw features."""
    if config == "matched":
        bv, bi = _load_pair(PHASE_B / "matched" / f"baseline_on_{stem}.npz")
    else:
        bv, bi = _load_pair(PHASE_B / "shared" / "baseline.npz")
    ov, oi = _load_pair(PHASE_B / config / f"{stem}.npz")
    f = _features_for_pair(bv, bi, ov, oi)
    return np.stack([f[k] for k in FEATURE_KEYS], axis=2)


def load(config: str) -> dict[str, np.ndarray]:
    cache = ROOT / "data" / f"paper_cube_{config}.npz"
    if cache.exists():
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}
    out = {}
    for axis in AXES:
        print(f"{config} clean_{axis}", flush=True)
        out[f"clean_{axis}"] = _cube(config, f"clean_{axis}")
    for v in VARIANTS:
        print(f"{config} {v}", flush=True)
        out[v] = _cube(config, v)
    np.savez_compressed(cache, **out)
    return out


def table(cubes, ckpts=None, layers=None, feats=None, clean_mode="separate"):
    """Rows: one per prompt per source. Returns X, y, groups, source names."""
    xs, ys, gs, src = [], [], [], []

    def vec(c):
        c = c if ckpts is None else c[:, ckpts]
        c = c.mean(axis=1)  # [P, 4, L]
        if feats is not None:
            c = c[:, feats]
        if layers is not None:
            c = c[:, :, layers]
        return c.reshape(c.shape[0], -1)

    if clean_mode == "pooled":
        x = np.mean([vec(cubes[f"clean_{a}"]) for a in AXES], axis=0)
        xs.append(x); ys.append(np.zeros(len(x), int)); gs.append(np.arange(len(x))); src += ["clean"] * len(x)
    else:
        for a in AXES:
            x = vec(cubes[f"clean_{a}"])
            xs.append(x); ys.append(np.zeros(len(x), int)); gs.append(np.arange(len(x))); src += [f"clean_{a}"] * len(x)
    for v in VARIANTS:
        x = vec(cubes[v])
        xs.append(x); ys.append(np.full(len(x), LABEL[v])); gs.append(np.arange(len(x))); src += [v] * len(x)
    return np.vstack(xs), np.concatenate(ys), np.concatenate(gs), np.array(src)


def oof(X, y, g, train_mask=None):
    """Out-of-fold predictions and probabilities. train_mask limits training rows."""
    pred = np.full(len(y), -1)
    proba = np.zeros((len(y), 5))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        if train_mask is not None:
            tr = tr[train_mask[tr]]
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd[sd == 0] = 1.0
        clf = LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=5000)
        clf.fit((X[tr] - mu) / sd, y[tr])
        Z = (X[te] - mu) / sd
        pred[te] = clf.predict(Z)
        p = np.zeros((len(te), 5))
        p[:, clf.classes_] = clf.predict_proba(Z)
        proba[te] = p
    return pred, proba


def fold_scores(y, pred, g):
    """Balanced accuracy per GroupKFold test fold, with a t-based 95% interval (as in results/metrics.json)."""
    folds = []
    for _, te in GroupKFold(n_splits=5).split(pred, y, g):
        folds.append(float(balanced_accuracy_score(y[te], pred[te])))
    m, sd = float(np.mean(folds)), float(np.std(folds, ddof=1))
    half = 2.776 * sd / np.sqrt(len(folds))
    return {"folds": folds, "mean": m, "std": sd, "ci95": [m - half, m + half]}


def summary(y, pred, src, g=None):
    honest = {s: float(np.mean(pred[src == s] != 0)) for s in np.unique(src) if s.startswith("clean")}
    missed = {v: float(np.mean(pred[src == v] == 0)) for v in VARIANTS}
    return {
        "fold_balanced_accuracy": None if g is None else fold_scores(y, pred, g),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "honest_flag_rate": honest,
        "tamper_called_clean": missed,
        "confusion": confusion_matrix(y, pred, labels=range(5)).tolist(),
    }


def binary(y, proba, src):
    """Score = 1 - P(clean). ROC over all honest rows vs each tamper class."""
    s = 1.0 - proba[:, 0]
    honest = s[y == 0]
    out = {}
    for v in VARIANTS:
        t = s[src == v]
        lab = np.r_[np.zeros(len(honest)), np.ones(len(t))]
        sc = np.r_[honest, t]
        fpr, tpr, _ = roc_curve(lab, sc)
        out[v] = {
            "auc": float(roc_auc_score(lab, sc)),
            "tpr_at_fpr_0.01": float(np.interp(0.01, fpr, tpr)),
            "tpr_at_fpr_0.05": float(np.interp(0.05, fpr, tpr)),
        }
    return out


def roc_points(y, proba, src):
    s = 1.0 - proba[:, 0]
    honest = s[y == 0]
    pts = {}
    for v in VARIANTS:
        t = s[src == v]
        fpr, tpr, _ = roc_curve(np.r_[np.zeros(len(honest)), np.ones(len(t))], np.r_[honest, t])
        pts[v] = {"fpr": fpr.tolist(), "tpr": tpr.tolist()}
    return pts


def main() -> None:
    report = {"seed": SEED, "split": "GroupKFold 5 by prompt_id, shuffle off", "model": "LogisticRegression balanced"}
    for config in ("matched", "shared"):
        cubes = load(config)
        r = {}
        X, y, g, src = table(cubes, clean_mode="pooled")
        pred, _ = oof(X, y, g)
        r["pooled_reproduction"] = summary(y, pred, src, g)

        X, y, g, src = table(cubes)
        pred, proba = oof(X, y, g)
        r["separate_axes"] = summary(y, pred, src, g)
        r["separate_axes_binary"] = binary(y, proba, src)
        if config == "matched":
            r["roc_points"] = roc_points(y, proba, src)

        # leave one honest axis out of training; score it at test time
        loo = {}
        for a in AXES:
            mask = src != f"clean_{a}"
            p, _ = oof(X, y, g, train_mask=mask)
            loo[a] = {
                "held_out_axis_flag_rate": float(np.mean(p[src == f"clean_{a}"] != 0)),
                "held_out_axis_predicted": np.bincount(p[src == f"clean_{a}"], minlength=5).tolist(),
            }
        r["leave_one_axis_out"] = loo

        # single honest kernel: train without eager, compare with the batch-shape-only last-hidden thresholds
        p, _ = oof(X, y, g, train_mask=src != "clean_attn_eager")
        r["without_eager_training"] = {
            "honest_flag_rate": {s: float(np.mean(p[src == s] != 0)) for s in np.unique(src) if s.startswith("clean")},
            "detect_rate": {v: float(np.mean(p[src == v] != 0)) for v in VARIANTS},
            "correct_class_rate": {v: float(np.mean(p[src == v] == LABEL[v])) for v in VARIANTS},
            "tampered_correct": int(sum(np.sum(p[src == v] == LABEL[v]) for v in VARIANTS)),
        }

        # ablations, separate axes
        abl = {}
        specs = {
            "all_layers_all_features": {},
            "last_layer_only": {"layers": [N_LAYERS - 1]},
            "first_layer_only": {"layers": [0]},
            "layers_0_13": {"layers": list(range(14))},
            "layers_14_27": {"layers": list(range(14, 28))},
            "all_but_last_layer": {"layers": list(range(N_LAYERS - 1))},
            "idx_mismatch_only": {"feats": [0]},
            "exp_mismatch_only": {"feats": [1]},
            "mantissa_only": {"feats": [2, 3]},
            "checkpoints_1": {"ckpts": [0]},
            "checkpoints_2": {"ckpts": [0, 1]},
            "checkpoints_4": {"ckpts": [0, 1, 2, 3]},
        }
        for name, kw in specs.items():
            Xa, ya, ga, sa = table(cubes, **kw)
            pa, pr = oof(Xa, ya, ga)
            s = summary(ya, pa, sa)
            abl[name] = {
                "balanced_accuracy": s["balanced_accuracy"],
                "honest_flag_rate": s["honest_flag_rate"],
                "tamper_called_clean": s["tamper_called_clean"],
                "binary_auc": {v: b["auc"] for v, b in binary(ya, pr, sa).items()},
            }
            print(config, name, round(s["balanced_accuracy"], 4), flush=True)
        r["ablations"] = abl

        # per-layer separation profile: mean feature by layer and source
        prof = {}
        for name, c in cubes.items():
            m = c.mean(axis=(0, 1))  # [4, L]
            prof[name] = {k: m[i].tolist() for i, k in enumerate(FEATURE_KEYS)}
        r["layer_means"] = prof
        report[config] = r

    out = ROOT / "results" / "paper_analysis.json"
    out.write_text(json.dumps(report, indent=1) + "\n", encoding="utf-8")
    print(out)


if __name__ == "__main__":
    main()
