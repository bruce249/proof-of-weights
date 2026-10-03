"""5-fold grouped CV, balanced accuracy, per-class F1, all-ckpt vs pre-EOS."""

from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import KNeighborsClassifier

LR_RANDOM_STATE = 20260923
N_FOLDS = 5
KNN_K = 5


def _t_crit(df: int) -> float:
    from scipy import stats

    return float(stats.t.ppf(0.975, df))


def _summarize(scores: list[float]) -> dict:
    arr = np.asarray(scores, dtype=np.float64)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1))
    half = _t_crit(len(arr) - 1) * std / np.sqrt(len(arr))
    return {
        "mean": mean,
        "std": std,
        "ci95": [mean - half, mean + half],
        "folds": [float(x) for x in arr],
    }


def evaluate_splits(X: np.ndarray, y: np.ndarray, groups: np.ndarray, class_ids: list[int]) -> dict:
    """GroupKFold by prompt. Returns fold metrics and out-of-fold predictions."""
    if len(X) != len(y) or len(y) != len(groups):
        raise RuntimeError(f"length mismatch X={len(X)} y={len(y)} groups={len(groups)}")
    def _scale(train_idx, test_idx):
        mu = X[train_idx].mean(axis=0)
        sd = X[train_idx].std(axis=0)
        sd = np.where(sd == 0.0, 1.0, sd)
        return (X[train_idx] - mu) / sd, (X[test_idx] - mu) / sd

    gkf = GroupKFold(n_splits=N_FOLDS)
    ba_lr, ba_knn = [], []
    f1_lr, f1_knn = [], []
    oof_true = np.empty(len(y), dtype=np.int32)
    oof_lr = np.empty(len(y), dtype=np.int32)
    oof_knn = np.empty(len(y), dtype=np.int32)
    filled = np.zeros(len(y), dtype=bool)

    for train_idx, test_idx in gkf.split(X, y, groups):
        lr = LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=LR_RANDOM_STATE,
        )
        knn = KNeighborsClassifier(n_neighbors=KNN_K, weights="uniform")
        x_tr, x_te = _scale(train_idx, test_idx)
        lr.fit(x_tr, y[train_idx])
        knn.fit(x_tr, y[train_idx])
        pred_lr = lr.predict(x_te)
        pred_knn = knn.predict(x_te)
        y_te = y[test_idx]
        ba_lr.append(float(balanced_accuracy_score(y_te, pred_lr)))
        ba_knn.append(float(balanced_accuracy_score(y_te, pred_knn)))
        f1_lr.append(f1_score(y_te, pred_lr, labels=class_ids, average=None, zero_division=0))
        f1_knn.append(f1_score(y_te, pred_knn, labels=class_ids, average=None, zero_division=0))
        oof_true[test_idx] = y_te
        oof_lr[test_idx] = pred_lr
        oof_knn[test_idx] = pred_knn
        filled[test_idx] = True

    if not filled.all():
        raise RuntimeError("GroupKFold left some rows without an out-of-fold prediction")

    def pack(ba: list[float], f1_rows: list[np.ndarray], pred: np.ndarray) -> dict:
        f1_mat = np.vstack(f1_rows)
        per_class = {}
        for i, cid in enumerate(class_ids):
            per_class[str(cid)] = _summarize([float(row[i]) for row in f1_mat])
        return {
            "balanced_accuracy": _summarize(ba),
            "per_class_f1": per_class,
            "confusion_matrix": confusion_matrix(oof_true, pred, labels=class_ids).tolist(),
        }

    counts = {str(cid): int(np.sum(y == cid)) for cid in class_ids}
    return {
        "n": int(len(y)),
        "n_prompts": int(len(np.unique(groups))),
        "per_class_counts": counts,
        "logistic_regression": pack(ba_lr, f1_lr, oof_lr),
        "knn": pack(ba_knn, f1_knn, oof_knn),
        "oof_true": oof_true,
        "oof_lr": oof_lr,
        "oof_knn": oof_knn,
    }
