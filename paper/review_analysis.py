"""Analyses requested by the conference review. CPU only, existing data.

1. Operating point on a held-out prompt split: train prompts 0-99, threshold
   chosen on training data only (inner prompt-grouped CV), test prompts 100-199.
   FAR (tampered accepted as clean) and FRR (honest rejected), Wilson 95% CIs.
2. Open set: hold out one modification type from training, measure how often it
   is accepted as clean.
3. Per-class precision/recall of the main out-of-fold result.
4. Calibration (ECE, Brier) of P(clean) on the test split.
5. Kernel policy: unrestricted / declared-unenforced / enforced, on the
   eager-served attack data (train prompts <150, test 150-199).
6. Interpretation: logistic-regression weight by layer; top-k sensitivity.

Sampling unit for every interval is the sequence (one prompt under one source).
Writes results/review_analysis.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from fingerprint.extract_deltas import FEATURE_KEYS, _features_for_pair, _load_pair  # noqa: E402

SEED = 20260923
VARIANTS = ("quant_8bit", "quant_4bit", "lora_merged", "ft_drift")
LABEL = {"clean": 0, "quant_8bit": 1, "quant_4bit": 2, "lora_merged": 3, "ft_drift": 4}
PRE_HONEST = ("clean_attn_eager", "clean_batch4", "clean_padded_batch_length")
DEC_HONEST = ("decode_sdpa_b1", "attn_eager", "batch4", "padded_batch_length")
SDPA_HONEST_DEC = ("decode_sdpa_b1", "batch4", "padded_batch_length")


# ---------------------------------------------------------------- helpers
def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [float(max(0.0, c - h)), float(min(1.0, c + h))]


def rate(mask: np.ndarray) -> dict:
    k, n = int(mask.sum()), int(mask.size)
    return {"rate": k / n if n else float("nan"), "k": k, "n": n, "ci95_wilson": wilson(k, n)}


def vec(c: np.ndarray, k: int | None = None) -> np.ndarray:
    m = c.mean(axis=1)  # [P, 4, L]
    return m.reshape(len(m), -1)


class Clf:
    def __init__(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0)
        self.sd[self.sd == 0] = 1.0
        self.m = LogisticRegression(class_weight="balanced", random_state=SEED, max_iter=5000)
        self.m.fit((X - self.mu) / self.sd, y)

    def proba(self, X):
        P = np.zeros((len(X), 5))
        P[:, self.m.classes_] = self.m.predict_proba((X - self.mu) / self.sd)
        return P

    def predict(self, X):
        return self.m.classes_[np.argmax(self.m.predict_proba((X - self.mu) / self.sd), axis=1)]


def pick_threshold(X, y, g, target_frr=0.01) -> float:
    """Accept when P(clean) >= tau. tau from inner prompt-grouped OOF on training data only."""
    pc = np.zeros(len(y))
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        pc[te] = Clf(X[tr], y[tr]).proba(X[te])[:, 0]
    return float(np.quantile(pc[y == 0], target_frr))


def build(cubes, honest, variants, rows=None):
    X, y, g, s = [], [], [], []
    for h in honest:
        x = vec(cubes[h]); idx = np.arange(len(x)) if rows is None else rows
        X.append(x[idx]); y += [0] * len(idx); g += list(idx); s += [h] * len(idx)
    for v in variants:
        x = vec(cubes[v]); idx = np.arange(len(x)) if rows is None else rows
        X.append(x[idx]); y += [LABEL[v]] * len(idx); g += list(idx); s += [v] * len(idx)
    return np.vstack(X), np.array(y), np.array(g), np.array(s)


def operating_point(X, y, g, s, honest, variants, test_from=100, target_frr=0.01) -> dict:
    tr, te = g < test_from, g >= test_from
    tau = pick_threshold(X[tr], y[tr], g[tr], target_frr)
    clf = Clf(X[tr], y[tr])
    pc = clf.proba(X[te])[:, 0]
    acc = pc >= tau
    st, yt = s[te], y[te]
    out = {"threshold_P_clean": tau, "target_train_frr": target_frr,
           "frr_honest_all": rate(~acc[yt == 0]),
           "frr_by_config": {h: rate(~acc[st == h]) for h in honest},
           "far_by_variant": {v: rate(acc[st == v]) for v in variants},
           "far_all_tampered": rate(acc[yt != 0])}
    pred = clf.predict(X[te])
    out["argmax_rule"] = {"frr_honest_all": rate(pred[yt == 0] != 0),
                          "far_by_variant": {v: rate(pred[st == v] == 0) for v in variants}}
    # calibration of binary P(clean)
    is_clean = (yt == 0).astype(float)
    bins = np.linspace(0, 1, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (pc >= lo) & (pc < hi if hi < 1 else pc <= hi)
        if m.any():
            ece += m.mean() * abs(pc[m].mean() - is_clean[m].mean())
    out["calibration_P_clean"] = {"ece_10bin": float(ece), "brier": float(np.mean((pc - is_clean) ** 2))}
    return out


def open_set(X, y, g, s, honest, variants, test_from=100, target_frr=0.01) -> dict:
    res = {}
    for held in variants:
        keep = [v for v in variants if v != held]
        tr = (g < test_from) & (s != held)
        te = g >= test_from
        tau = pick_threshold(X[tr], y[tr], g[tr], target_frr)
        clf = Clf(X[tr], y[tr])
        pc = clf.proba(X[te])[:, 0]
        acc = pc >= tau
        st = s[te]
        pred = clf.predict(X[te])
        res[held] = {"trained_on": keep,
                     "far_heldout_threshold": rate(acc[st == held]),
                     "far_heldout_argmax": rate(pred[st == held] == 0),
                     "heldout_predicted_counts": np.bincount(pred[st == held], minlength=5).tolist(),
                     "frr_honest_threshold": rate(~acc[y[te] == 0])}
    return res


# ---------------------------------------------------------------- main
def main() -> None:
    out = {"sampling_unit": "sequence (one prompt under one source)",
           "interval": "Wilson score 95% for rates; train/test split by prompt id"}
    with np.load(ROOT / "data" / "paper_cube_matched.npz") as z:
        pre = {k: z[k] for k in z.files}
    with np.load(ROOT / "data" / "decode_cubes.npz") as z:
        dec = {k: z[k] for k in z.files}

    # 1-2. operating point and open set, prefill and decode
    for name, cubes, honest in (("prefill", pre, PRE_HONEST), ("decode", dec, DEC_HONEST)):
        X, y, g, s = build(cubes, honest, VARIANTS)
        out[f"operating_point_{name}"] = operating_point(X, y, g, s, honest, VARIANTS)
        out[f"open_set_{name}"] = open_set(X, y, g, s, honest, VARIANTS)

    # 3. per-class precision/recall, main OOF setting (prefill, separate)
    X, y, g, s = build(pre, PRE_HONEST, VARIANTS)
    pred = np.zeros(len(y), int)
    for tr, te in GroupKFold(n_splits=5).split(X, y, g):
        pred[te] = Clf(X[tr], y[tr]).predict(X[te])
    p, r, f, n = precision_recall_fscore_support(y, pred, labels=range(5), zero_division=0)
    names = ["clean", *VARIANTS]
    out["per_class_prefill_oof"] = {names[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(n[i])} for i in range(5)}
    out["balanced_accuracy_prefill_oof"] = float(balanced_accuracy_score(y, pred))

    # 4. interpretation: |standardised coefficient| summed per layer (all classes, all features)
    clf = Clf(X, y)
    W = np.abs(clf.m.coef_).reshape(len(clf.m.classes_), 4, 28)
    out["coef_mass_by_layer"] = (W.sum(axis=(0, 1)) / W.sum()).tolist()
    out["coef_mass_by_feature"] = dict(zip(FEATURE_KEYS, (W.sum(axis=(0, 2)) / W.sum()).tolist()))

    # 5. kernel policy, decode data, test prompts 150-199
    att = {k: v for k, v in dec.items() if k.endswith(("_eager_b1", "_sdpa_b4"))}
    rows_att = np.arange(100, 200)

    def att_vec(key, lo, hi):
        x = vec(att[key])
        m = (rows_att >= lo) & (rows_att < hi)
        return x[m]

    def honest_rows(h, lo, hi):
        return vec(dec[h])[lo:hi]

    def default_rows(v, lo, hi):
        return vec(dec[v])[lo:hi]

    def fit_detector(honest_cfgs, tamper_rows):
        Xs, ys, gs = [], [], []
        for h in honest_cfgs:
            x = honest_rows(h, 0, 150); Xs.append(x); ys += [0] * len(x); gs += list(range(150))
        for lab, x, pids in tamper_rows:
            Xs.append(x); ys += [lab] * len(x); gs += list(pids)
        X_, y_, g_ = np.vstack(Xs), np.array(ys), np.array(gs)
        return Clf(X_, y_), pick_threshold(X_, y_, g_)

    def accept(clf, tau, x):
        return clf.proba(x)[:, 0] >= tau

    pol = {}
    # S1 unrestricted: one detector, honest = all kernels, tampered = default-kernel only
    tam_default = [(LABEL[v], default_rows(v, 0, 150), range(150)) for v in VARIANTS]
    c1, t1 = fit_detector(DEC_HONEST, tam_default)
    pol["S1_unrestricted"] = {
        "description": "one detector accepting all honest kernels; trained on default-kernel tampering",
        "frr_honest": {h: rate(~accept(c1, t1, honest_rows(h, 150, 200))) for h in DEC_HONEST},
        "far_attack": {k: rate(accept(c1, t1, att_vec(k, 150, 200))) for k in sorted(att)}}
    # SDPA-pinned detector: honest SDPA configs only; tampered = default kernel + sdpa_b4 attack rows (<150)
    tam_sdpa = tam_default + [(LABEL[k.rsplit("_", 2)[0]], att_vec(k, 100, 150), range(100, 150)) for k in att if k.endswith("_sdpa_b4")]
    c_sdpa, t_sdpa = fit_detector(SDPA_HONEST_DEC, tam_sdpa)
    # eager-declared detector: honest eager only; tampered = eager attack rows (<150)
    Xs = [honest_rows("attn_eager", 0, 150)]; ys = [0] * 150; gs = list(range(150))
    for k in att:
        if k.endswith("_eager_b1"):
            x = att_vec(k, 100, 150); Xs.append(x); ys += [LABEL[k.rsplit("_", 2)[0]]] * len(x); gs += list(range(100, 150))
    Xe, ye, ge = np.vstack(Xs), np.array(ys), np.array(gs)
    c_eag, t_eag = Clf(Xe, ye), pick_threshold(Xe, ye, ge)
    s2 = {}
    for k in sorted(k for k in att if k.endswith("_eager_b1")):
        x = att_vec(k, 150, 200)
        a_e = accept(c_eag, t_eag, x)
        a_s = accept(c_sdpa, t_sdpa, x)
        s2[k] = {"declare_eager": rate(a_e), "declare_sdpa": rate(a_s),
                 "attacker_best": rate(a_e if a_e.mean() >= a_s.mean() else a_s)}
    pol["S2_declared_unenforced"] = {
        "description": "provider declares a kernel; validator uses that kernel's detector; attacker runs eager and picks the declaration that is accepted more often",
        "frr_honest_eager_declares_eager": rate(~accept(c_eag, t_eag, honest_rows("attn_eager", 150, 200))),
        "frr_honest_sdpa_configs": {h: rate(~accept(c_sdpa, t_sdpa, honest_rows(h, 150, 200))) for h in SDPA_HONEST_DEC},
        "far_eager_attack": s2,
        "note": "validator reference is an SDPA prefill even when eager is declared; an eager reference prefill was not collected"}
    pol["S3_enforced_sdpa"] = {
        "description": "kernel enforced to SDPA; attacker limited to SDPA serving (default batch 1 and batch 4)",
        "frr_honest": {h: rate(~accept(c_sdpa, t_sdpa, honest_rows(h, 150, 200))) for h in SDPA_HONEST_DEC},
        "far_default_kernel": {v: rate(accept(c_sdpa, t_sdpa, default_rows(v, 150, 200))) for v in VARIANTS},
        "far_sdpa_b4_attack": {k: rate(accept(c_sdpa, t_sdpa, att_vec(k, 150, 200))) for k in sorted(att) if k.endswith("_sdpa_b4")}}
    out["kernel_policy"] = pol

    # 6. top-k sensitivity (prefill matched, separate honest), recomputed from stored top-128
    pb = ROOT / "data" / "phase_b"
    src = {"clean_attn_eager": ("shared/baseline", "matched/clean_attn_eager"),
           "clean_batch4": ("shared/baseline", "matched/clean_batch4"),
           "clean_padded_batch_length": ("shared/baseline", "matched/clean_padded_batch_length"),
           **{v: (f"matched/baseline_on_{v}", f"matched/{v}") for v in VARIANTS}}
    cache = ROOT / "data" / "topk_sensitivity_cubes.npz"
    store = dict(np.load(cache)) if cache.exists() else {}
    topk = {}
    for k in (16, 32, 64, 128):
        cubes = {}
        for name, (b, o) in src.items():
            key = f"k{k}_{name}"
            if key not in store:
                bv, bi = _load_pair(pb / f"{b}.npz")
                ov, oi = _load_pair(pb / f"{o}.npz")
                f = _features_for_pair(bv[..., :k], bi[..., :k], ov[..., :k], oi[..., :k])
                store[key] = np.stack([f[q] for q in FEATURE_KEYS], axis=2)
                print("topk", key, flush=True)
            cubes[name] = store[key]
        X, y, g, s = build(cubes, PRE_HONEST, VARIANTS)
        pred = np.zeros(len(y), int)
        for tr, te in GroupKFold(n_splits=5).split(X, y, g):
            pred[te] = Clf(X[tr], y[tr]).predict(X[te])
        topk[k] = {"balanced_accuracy": float(balanced_accuracy_score(y, pred)),
                   "eager_flag_rate": float(np.mean(pred[s == "clean_attn_eager"] != 0)),
                   "tamper_called_clean": float(np.mean(pred[y != 0] == 0))}
    np.savez_compressed(cache, **store)
    out["topk_sensitivity_prefill"] = topk

    (ROOT / "results" / "review_analysis.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print("wrote results/review_analysis.json")


if __name__ == "__main__":
    main()
