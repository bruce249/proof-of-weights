"""Analysis of the review GPU run (inference/review_gpu.py). CPU only.

Detectors are the decode-time detectors of paper/review_analysis.py, trained on
prompts 0-149 with a threshold chosen on training data only:
  unrestricted : honest = all four decode configurations
  sdpa_pinned  : honest = reference decode, batch 4, batch 4 + padding
New Qwen modifications are evaluated on test prompts 150-199, which no detector
saw. The detector-aware attacker picks, among the five selective quantisations,
the one the pinned detector accepts most often on calibration prompts 0-19.

Writes results/review_gpu_analysis.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "paper"))
from fingerprint.extract_deltas import FEATURE_KEYS, _features_for_pair  # noqa: E402
from review_analysis import (  # noqa: E402
    DEC_HONEST, LABEL, SDPA_HONEST_DEC, VARIANTS, Clf, pick_threshold, rate, vec,
)

REV = ROOT / "data" / "review"
SELECTIVE = ("nf4_first14", "nf4_last14", "nf4_even", "nf4_odd", "nf4_mlp")


def cube_from(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as z:
        f = _features_for_pair(z["ref_values"], z["ref_indices"], z["prov_values"], z["prov_indices"])
        pids = z["prompt_ids"]
    return np.stack([f[k] for k in FEATURE_KEYS], axis=2), pids


def detectors():
    with np.load(ROOT / "data" / "decode_cubes.npz") as z:
        dec = {k: z[k] for k in z.files}

    def fit(honest):
        Xs, ys, gs = [], [], []
        for h in honest:
            Xs.append(vec(dec[h])[:150]); ys += [0] * 150; gs += list(range(150))
        for v in VARIANTS:
            Xs.append(vec(dec[v])[:150]); ys += [LABEL[v]] * 150; gs += list(range(150))
        X, y, g = np.vstack(Xs), np.array(ys), np.array(gs)
        return Clf(X, y), pick_threshold(X, y, g)

    return {"unrestricted": fit(DEC_HONEST), "sdpa_pinned": fit(SDPA_HONEST_DEC)}


def last_hidden(rows, pids_keep):
    th = json.loads((ROOT / "results" / "phase_a_recalibrated.json").read_text(encoding="utf-8"))
    sets = {k: th[k]["thresholds"] for k in ("all_three_axes", "without_eager", "published_thresholds")}
    S = np.array([[r["exp_mismatches_max"], r["mant_err_mean_max"], r["mant_err_median_max"]] for r in rows if r["prompt_id"] in pids_keep])
    return {n: rate(np.any(S > np.array([t["T_exp"], t["T_mean"], t["T_median"]]), axis=1)) for n, t in sets.items()}


def qwen_new(out):
    stats = json.loads((ROOT / "results" / "review_gpu_qwen_new.json").read_text(encoding="utf-8"))
    det = detectors()
    res = {}
    for name, info in stats.items():
        c, pids = cube_from(REV / f"qwen_{name}.npz")
        X = vec(c)
        test, calib = pids >= 150, pids < 20
        r = {"quantisation": info["quantisation"], "n_test": int(test.sum())}
        for dn, (clf, tau) in det.items():
            pc = clf.proba(X)[:, 0]
            pred = clf.predict(X)
            r[dn] = {"far_test": rate(pc[test] >= tau),
                     "calib_acceptance": float(np.mean(pc[calib] >= tau)),
                     "calib_mean_P_clean": float(np.mean(pc[calib])),
                     "test_predicted_counts": np.bincount(pred[test], minlength=5).tolist()}
        r["last_hidden_test"] = last_hidden(info["seq_stats"], set(int(p) for p in pids[test]))
        res[name] = r
    # detector-aware choice: highest calibration acceptance under the pinned detector, ties by mean P(clean)
    cand = [n for n in SELECTIVE if n in res]
    best = max(cand, key=lambda n: (res[n]["sdpa_pinned"]["calib_acceptance"], res[n]["sdpa_pinned"]["calib_mean_P_clean"]))
    out["qwen_new"] = res
    out["detector_aware_choice"] = {"candidates": cand, "chosen": best,
                                    "criterion": "acceptance by the SDPA-pinned detector on prompts 0-19",
                                    "chosen_far_test_pinned": res[best]["sdpa_pinned"]["far_test"],
                                    "chosen_far_test_unrestricted": res[best]["unrestricted"]["far_test"]}


def smol(out):
    meta = json.loads((ROOT / "results" / "review_gpu_smol.json").read_text(encoding="utf-8"))
    names = {"ref_decode": "smol_ref_decode", "eager": "smol_honest_eager", "batch4": "smol_honest_batch4",
             "nf4": "smol_nf4", "int8": "smol_int8", "nf4_eager": "smol_nf4_eager", "int8_eager": "smol_int8_eager"}
    cubes, pids = {}, {}
    for k, f in names.items():
        cubes[k], pids[k] = cube_from(REV / f"{f}.npz")
    lab = {"nf4": 1, "int8": 2}
    honest = ("ref_decode", "eager", "batch4")

    def table(hon, layers=None):
        Xs, ys, gs, ss = [], [], [], []
        for h in hon:
            x = vec(cubes[h][:, :, :, layers] if layers is not None else cubes[h])
            Xs.append(x); ys += [0] * len(x); gs += list(pids[h]); ss += [h] * len(x)
        for v in lab:
            x = vec(cubes[v][:, :, :, layers] if layers is not None else cubes[v])
            Xs.append(x); ys += [lab[v]] * len(x); gs += list(pids[v]); ss += [v] * len(x)
        return np.vstack(Xs), np.array(ys), np.array(gs), np.array(ss)

    def cv(X, y, g):
        pred = np.zeros(len(y), int)
        for tr, te in GroupKFold(n_splits=5).split(X, y, g):
            pred[te] = Clf(X[tr], y[tr]).predict(X[te])
        return pred

    L = meta["n_layers"]
    X, y, g, s = table(honest)
    pred = cv(X, y, g)
    Xl, yl, gl, sl = table(honest, layers=[L - 1])
    predl = cv(Xl, yl, gl)
    r = {"model": meta["model"], "revision": meta["revision"], "n_layers": L, "hidden_size": meta["hidden_size"],
         "tokens": meta["tokens_per_generation"], "quantisation": meta.get("quantisation"),
         "cv_balanced_accuracy_all_layers": float(balanced_accuracy_score(y, pred)),
         "cv_balanced_accuracy_last_layer": float(balanced_accuracy_score(yl, predl)),
         "cv_honest_flag_rate": {h: float(np.mean(pred[s == h] != 0)) for h in honest},
         "cv_tamper_called_clean": {v: float(np.mean(pred[s == v] == 0)) for v in lab},
         "cv_last_layer_honest_flag_rate": {h: float(np.mean(predl[sl == h] != 0)) for h in honest}}
    # operating point + eager attack: train prompts 0-49, test 50-99
    for dn, hon in (("unrestricted", honest), ("sdpa_pinned", ("ref_decode", "batch4"))):
        Xd, yd, gd, sd = table(hon)
        tr = gd < 50
        clf, tau = Clf(Xd[tr], yd[tr]), pick_threshold(Xd[tr], yd[tr], gd[tr])
        acc = clf.proba(Xd)[:, 0] >= tau
        te = ~tr
        r[dn] = {"frr_test": {h: rate(~acc[te & (sd == h)]) for h in hon},
                 "far_test": {v: rate(acc[te & (sd == v)]) for v in lab},
                 "far_eager_attack": {k: rate(clf.proba(vec(cubes[k]))[:, 0] >= tau) for k in ("nf4_eager", "int8_eager")}}
        if dn == "sdpa_pinned":
            r[dn]["frr_honest_eager_if_pinned"] = rate(~(clf.proba(vec(cubes["eager"])[pids["eager"] >= 50])[:, 0] >= tau))
    # mean per-layer index mismatch, for the depth-profile comparison
    r["layer_mean_idx_mismatch"] = {k: cubes[k].mean(axis=(0, 1))[0].tolist() for k in ("ref_decode", "eager", "batch4", "nf4", "int8")}
    out["smol"] = r


def overhead(out):
    o = json.loads((ROOT / "results" / "review_gpu_overhead.json").read_text(encoding="utf-8"))
    rows = o["per_seq"]
    m = lambda k: float(np.mean([r[k] for r in rows]))
    out["overhead"] = {"n": len(rows), "bytes_all_layers": m("bytes_total_all_layers"), "bytes_last_layer": m("bytes_last_layer"),
                       "build_seconds_all_layers": m("build_seconds_all_layers"), "verify_seconds_all_layers": m("verify_seconds_all_layers"),
                       "prefill_seconds_plain": m("prefill_seconds_plain"), "prefill_seconds_capture": m("prefill_seconds_capture_all_layers"),
                       "honest_exp_max_by_layer_max": np.max([r["honest_exp_mismatch_max_by_layer"] for r in rows], axis=0).tolist()}


def throughput(out):
    out["throughput"] = json.loads((ROOT / "results" / "review_gpu_throughput.json").read_text(encoding="utf-8"))


def main():
    out = {}
    for fn in (qwen_new, smol, overhead, throughput):
        try:
            fn(out)
        except FileNotFoundError as e:
            print("skip", fn.__name__, e)
    (ROOT / "results" / "review_gpu_analysis.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1)[:6000])


if __name__ == "__main__":
    main()
