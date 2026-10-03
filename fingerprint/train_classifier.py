"""5-class LR (balanced) + k-NN k=5 unweighted. Split by prompt.

Clean (label 0) is the three honest Phase B axes (attn_eager, batch4,
padded_batch_length) pooled. This scores clean vs all four tamper classes.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from fingerprint.evaluate import KNN_K, LR_RANDOM_STATE, evaluate_splits
from fingerprint.extract_deltas import FEATURE_KEYS, ROOT, VARIANTS, flatten_zscored

CLASS_IDS = [0, 1, 2, 3, 4]
CLASS_NAMES = ("clean", *VARIANTS)


def _vectors(df: pd.DataFrame, n_layers: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    grouped = df.groupby(["prompt_id", "label", "layer"], sort=True)[list(FEATURE_KEYS)].mean()
    xs: list[np.ndarray] = []
    ys: list[int] = []
    prompts: list[int] = []
    for (pid, lab), sub in grouped.groupby(level=[0, 1], sort=True):
        sub = sub.reset_index()
        rows = []
        for ell in range(n_layers):
            hit = sub[sub["layer"] == ell]
            if len(hit) != 1:
                raise RuntimeError(f"prompt {pid} label {lab} layer {ell} rows={len(hit)}")
            rec = hit.iloc[0]
            rows.append({key: float(rec[key]) for key in FEATURE_KEYS})
        xs.append(flatten_zscored(rows, n_layers))
        ys.append(int(lab))
        prompts.append(int(pid))
    return np.stack(xs), np.asarray(ys, dtype=np.int32), np.asarray(prompts, dtype=np.int32)


def _public(report: dict) -> dict:
    return {k: v for k, v in report.items() if not k.startswith("oof_")}


def _plot(y_true: np.ndarray, y_lr: np.ndarray, y_knn: np.ndarray, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix

    names = list(CLASS_NAMES)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    for ax, pred, title in (
        (axes[0], y_lr, "logistic regression"),
        (axes[1], y_knn, "k-NN k=5"),
    ):
        cm = confusion_matrix(y_true, pred, labels=CLASS_IDS)
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(title)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
        ax.set_xticks(range(len(names)), names, rotation=30, ha="right")
        ax.set_yticks(range(len(names)), names)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(int(cm[i, j])), ha="center", va="center", color="black", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("Matched, all checkpoints, out-of-fold")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def train(cfg: dict) -> None:
    n_layers = int(cfg["n_layers"])
    metrics: dict = {
        "classes": {name: int(cfg["tamper_class_ids"][name]) for name in ("clean", *VARIANTS)},
        "trained_class_ids": CLASS_IDS,
        "clean_present": True,
        "feature_dim": 4 * n_layers,
        "checkpoint_reduce": "mean of per-layer raw features, then flatten",
        "zscore_cv": "per fold, mean and std of each feature from training prompts only, applied to that fold's test rows",
        "zscore_saved_artifact": "population mean/std over all matched all-checkpoint rows, saved in artifacts/classifier.pkl (zscore_mean/zscore_std) for the live validator",
        "split": "GroupKFold n_splits=5 by prompt_id, shuffle off",
        "logistic_regression": {
            "class_weight": "balanced",
            "random_state": LR_RANDOM_STATE,
        },
        "knn": {"k": KNN_K, "weights": "uniform"},
        "primary_model": "logistic_regression",
        "configs": {},
    }

    matched_all = None
    for config in ("matched", "shared"):
        df = pd.read_parquet(ROOT / "data" / f"features_{config}.parquet")
        block = {}
        for subset, frame in (
            ("all_checkpoints", df),
            ("pre_eos", df[df["pre_eos"]]),
        ):
            X, y, groups = _vectors(frame, n_layers)
            if X.shape[1] != 4 * n_layers:
                raise RuntimeError(f"feature dim {X.shape[1]} != {4 * n_layers}")
            report = evaluate_splits(X, y, groups, CLASS_IDS)
            block[subset] = _public(report)
            if config == "matched" and subset == "all_checkpoints":
                matched_all = (X, y, report)
                clean = frame[frame["label"] == 0]
                metrics["clean_delta"] = {
                    key: float(clean[key].mean()) for key in FEATURE_KEYS
                }
                if metrics["clean_delta"]["idx_mismatch"] == 0.0:
                    raise RuntimeError("clean mean index mismatch is exactly 0")
            print(
                f"{config} {subset} LR bal_acc="
                f"{report['logistic_regression']['balanced_accuracy']['mean']:.3f} "
                f"kNN bal_acc={report['knn']['balanced_accuracy']['mean']:.3f}",
                flush=True,
            )
        metrics["configs"][config] = block

    if matched_all is None:
        raise RuntimeError("matched all-checkpoint matrix was not built")
    X, y, report = matched_all
    lr = LogisticRegression(class_weight="balanced", max_iter=2000, random_state=LR_RANDOM_STATE)
    knn = KNeighborsClassifier(n_neighbors=KNN_K, weights="uniform")
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd = np.where(sd == 0.0, 1.0, sd)
    xs = (X - mu) / sd
    lr.fit(xs, y)
    knn.fit(xs, y)
    payload = {
        "model": lr,
        "knn": knn,
        "primary": "logistic_regression",
        "class_ids": CLASS_IDS,
        "class_names": list(CLASS_NAMES),
        "n_layers": n_layers,
        "feature_keys": list(FEATURE_KEYS),
        "checkpoint_reduce": "mean",
        "zscore_mean": mu,
        "zscore_std": sd,
        "random_state": LR_RANDOM_STATE,
    }
    out = ROOT / "artifacts" / "classifier.pkl"
    joblib.dump(payload, out)
    fig = ROOT / "figures" / "confusion_matrix.png"
    _plot(report["oof_true"], report["oof_lr"], report["oof_knn"], fig)
    metrics_path = ROOT / "results" / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}", flush=True)
    print(f"wrote {fig}", flush=True)
    print(f"wrote {metrics_path}", flush=True)


if __name__ == "__main__":
    import yaml

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    train(cfg)
