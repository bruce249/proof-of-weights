"""Single feature path. Validator must import this module. Do not reimplement."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FEATURE_KEYS = ("idx_mismatch", "exp_mismatch", "mantissa_mean", "mantissa_median")
VARIANTS = ("quant_8bit", "quant_4bit", "lora_merged", "ft_drift")


def index_mismatch_ratio(base_idx: np.ndarray, other_idx: np.ndarray) -> float:
    """base_idx, other_idx: [k] int."""
    return float(1.0 - np.mean(np.isin(other_idx, base_idx)))


def _split_exp_mant(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = x.astype(np.float32)
    bits = x.view(np.uint32)
    exp = (bits >> 23) & np.uint32(0xFF)
    mant = bits & np.uint32(0x7FFFFF)
    return exp.astype(np.int32), mant.astype(np.int32)


def exp_mismatch_count(base_val: np.ndarray, other_val: np.ndarray) -> int:
    be, _ = _split_exp_mant(base_val)
    oe, _ = _split_exp_mant(other_val)
    return int(np.sum(be != oe))


def mantissa_mean_median_diff(base_val: np.ndarray, other_val: np.ndarray) -> tuple[float, float]:
    _, bm = _split_exp_mant(base_val)
    _, om = _split_exp_mant(other_val)
    d = np.abs(bm.astype(np.float64) - om.astype(np.float64))
    return float(np.mean(d)), float(np.median(d))


def layer_features(
    base_values: np.ndarray,
    base_indices: np.ndarray,
    other_values: np.ndarray,
    other_indices: np.ndarray,
) -> Dict[str, float]:
    """One checkpoint, one layer: arrays shape [k]."""
    mean_d, med_d = mantissa_mean_median_diff(base_values, other_values)
    return {
        "idx_mismatch": index_mismatch_ratio(base_indices, other_indices),
        "exp_mismatch": float(exp_mismatch_count(base_values, other_values)),
        "mantissa_mean": mean_d,
        "mantissa_median": med_d,
    }


def flatten_zscored(rows_by_layer: list[dict], n_layers: int) -> np.ndarray:
    """4 x n_layers vector used at classifier-train and live validator time."""
    keys = ("idx_mismatch", "exp_mismatch", "mantissa_mean", "mantissa_median")
    vec = np.zeros(4 * n_layers, dtype=np.float64)
    for layer, feat in enumerate(rows_by_layer):
        for i, key in enumerate(keys):
            vec[i * n_layers + layer] = feat[key]
    return vec


def extract_from_topk_pair(
    base_values: np.ndarray,
    base_indices: np.ndarray,
    other_values: np.ndarray,
    other_indices: np.ndarray,
) -> np.ndarray:
    """
    values/indices: [n_checkpoints, n_layers, k]
    Returns mean-over-checkpoints flattened features (live path). Offline
    training keeps long-format rows; both must call layer_features.
    """
    n_ckpts, n_layers, _k = base_values.shape
    acc = np.zeros(4 * n_layers, dtype=np.float64)
    for c in range(n_ckpts):
        rows = [
            layer_features(
                base_values[c, ell],
                base_indices[c, ell],
                other_values[c, ell],
                other_indices[c, ell],
            )
            for ell in range(n_layers)
        ]
        acc += flatten_zscored(rows, n_layers)
    return acc / n_ckpts


def _load_pair(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path) as z:
        values = np.asarray(z["values"], dtype=np.float32)
        indices = np.asarray(z["indices"], dtype=np.int32)
    if values.shape != indices.shape or values.ndim != 4:
        raise RuntimeError(f"{path} expected values/indices [N,C,L,K], got {values.shape} {indices.shape}")
    return values, indices


def _eos_by_prompt(path: Path) -> dict[int, int]:
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = data["prompts"] if isinstance(data, dict) else data
    return {int(row["prompt_id"]): int(row["natural_eos_pos"]) for row in rows}


def _features_for_pair(
    base_values: np.ndarray,
    base_indices: np.ndarray,
    other_values: np.ndarray,
    other_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Call layer_features for every prompt, checkpoint, and layer. Arrays [N,C,L]."""
    if base_values.shape != other_values.shape or base_indices.shape != other_indices.shape:
        raise RuntimeError(f"shape mismatch {base_values.shape} vs {other_values.shape}")
    n_prompts, n_ckpts, n_layers, _k = base_values.shape
    out = {key: np.empty((n_prompts, n_ckpts, n_layers), dtype=np.float64) for key in FEATURE_KEYS}
    for p in range(n_prompts):
        for c in range(n_ckpts):
            for ell in range(n_layers):
                feat = layer_features(
                    base_values[p, c, ell],
                    base_indices[p, c, ell],
                    other_values[p, c, ell],
                    other_indices[p, c, ell],
                )
                for key in FEATURE_KEYS:
                    out[key][p, c, ell] = feat[key]
    return out


def _zscore_per_layer(columns: dict[str, np.ndarray], layers: np.ndarray) -> tuple[dict[str, np.ndarray], dict]:
    """Population z-score of each feature inside each layer. Stats are for the live path."""
    zcols: dict[str, np.ndarray] = {}
    stats: dict[str, dict[str, list[float]]] = {key: {"mean": [], "std": []} for key in FEATURE_KEYS}
    for key in FEATURE_KEYS:
        z = np.empty_like(columns[key])
        for ell in range(int(layers.max()) + 1):
            mask = layers == ell
            x = columns[key][mask]
            mu = float(np.mean(x))
            sd = float(np.std(x))
            stats[key]["mean"].append(mu)
            stats[key]["std"].append(sd)
            z[mask] = 0.0 if sd == 0.0 else (x - mu) / sd
        zcols[key] = z
    return zcols, stats


def _write_table(path: Path, frame_cols: dict[str, np.ndarray]) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(frame_cols).to_parquet(path, index=False)


def _plot_matched(df, labels: dict, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    name_of = {int(v): k for k, v in labels.items() if k in (*VARIANTS, "clean")}
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), sharex=True)
    for ax, key in zip(axes.ravel(), FEATURE_KEYS):
        for label, name in name_of.items():
            sub = df[df["label"] == label]
            g = sub.groupby("layer", sort=True)[key].mean()
            ax.plot(g.index.to_numpy(), g.to_numpy(), label=name, linewidth=1.6)
        ax.set_title(key)
        ax.set_xlabel("layer")
        ax.set_ylabel("mean")
        ax.grid(True, alpha=0.3)
    axes[0, 0].legend(frameon=False, fontsize=8)
    fig.suptitle("Matched Phase B: mean delta vs layer depth")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def build_feature_tables(cfg: dict) -> None:
    """Long-format per-layer deltas for matched and shared Phase B arrays.

    Required columns stay in raw units. z_* columns are a full-table scale for
    inspection. Classifier folds refit that scale on training prompts only.
    Label 0 is the three honest axes pooled: attn_eager, batch4, padded_batch_length.
    """
    import pandas as pd

    interval = int(cfg["proof_interval_tokens"])
    labels = {k: int(v) for k, v in cfg["tamper_class_ids"].items()}
    phase_b = ROOT / "data" / "phase_b"
    shared_eos = _eos_by_prompt(ROOT / "data" / "prompts.json")

    tables: dict[str, list[dict[str, np.ndarray]]] = {"matched": [], "shared": []}
    base_shared_v, base_shared_i = _load_pair(phase_b / "shared" / "baseline.npz")

    jobs = [(variant, labels[variant], ROOT / "data" / variant / "sequences.json", variant) for variant in VARIANTS]
    for axis in ("attn_eager", "batch4", "padded_batch_length"):
        jobs.append(("clean", labels["clean"], ROOT / "data" / "prompts.json", f"clean_{axis}"))
    for variant, label, eos_path, stem in jobs:
        eos = _eos_by_prompt(eos_path)
        other_m_v, other_m_i = _load_pair(phase_b / "matched" / f"{stem}.npz")
        base_m_v, base_m_i = _load_pair(phase_b / "matched" / f"baseline_on_{stem}.npz")
        other_s_v, other_s_i = _load_pair(phase_b / "shared" / f"{stem}.npz")
        print(f"features {stem}", flush=True)
        pairs = {
            "matched": (_features_for_pair(base_m_v, base_m_i, other_m_v, other_m_i), eos, other_m_v.shape[0]),
            "shared": (_features_for_pair(base_shared_v, base_shared_i, other_s_v, other_s_i), shared_eos, other_s_v.shape[0]),
        }
        for config, (feats, eos_map, n_prompts) in pairs.items():
            n_ckpts = next(iter(feats.values())).shape[1]
            n_layers = next(iter(feats.values())).shape[2]
            prompt_ids = np.arange(n_prompts, dtype=np.int32)
            missing = [int(p) for p in prompt_ids if int(p) not in eos_map]
            if missing:
                raise RuntimeError(f"{variant} {config} missing natural_eos_pos for {missing[:5]}")
            eos_arr = np.array([eos_map[int(p)] for p in prompt_ids], dtype=np.int32)
            ckpt = np.arange(n_ckpts, dtype=np.int32)
            pre = (ckpt[None, :] * interval) < eos_arr[:, None]
            p_grid = np.broadcast_to(prompt_ids[:, None, None], (n_prompts, n_ckpts, n_layers)).reshape(-1)
            c_grid = np.broadcast_to(ckpt[None, :, None], (n_prompts, n_ckpts, n_layers)).reshape(-1)
            l_grid = np.broadcast_to(np.arange(n_layers, dtype=np.int32)[None, None, :], (n_prompts, n_ckpts, n_layers)).reshape(-1)
            pre_grid = np.broadcast_to(pre[:, :, None], (n_prompts, n_ckpts, n_layers)).reshape(-1)
            block = {
                "prompt_id": p_grid,
                "checkpoint": c_grid,
                "layer": l_grid,
                "label": np.full(p_grid.shape, label, dtype=np.int16),
                "config": np.full(p_grid.shape, config, dtype=object),
                "pre_eos": pre_grid,
            }
            for key in FEATURE_KEYS:
                block[key] = feats[key].reshape(-1)
            tables[config].append(block)

    stats_out = {}
    matched_df = None
    for config, blocks in tables.items():
        cols = {}
        for key in ("prompt_id", "checkpoint", "layer", "label", "pre_eos", *FEATURE_KEYS):
            cols[key] = np.concatenate([b[key] for b in blocks])
        cols["config"] = np.concatenate([b["config"] for b in blocks])
        zcols, stats = _zscore_per_layer({k: cols[k] for k in FEATURE_KEYS}, cols["layer"])
        for key in FEATURE_KEYS:
            cols[f"z_{key}"] = zcols[key]
        stats_out[config] = stats
        path = ROOT / "data" / f"features_{config}.parquet"
        _write_table(path, cols)
        print(f"wrote {path} rows={len(cols['prompt_id'])}", flush=True)
        if config == "matched":
            matched_df = pd.DataFrame(
                {
                    "layer": cols["layer"],
                    "label": cols["label"],
                    **{k: cols[k] for k in FEATURE_KEYS},
                }
            )

    (ROOT / "results" / "feature_zscore_stats.json").write_text(
        json.dumps(stats_out) + "\n", encoding="utf-8"
    )
    if matched_df is None:
        raise RuntimeError("matched table was not built")
    _plot_matched(matched_df, labels, ROOT / "figures" / "layer_profiles.png")
    print(f"wrote {ROOT / 'figures' / 'layer_profiles.png'}", flush=True)


if __name__ == "__main__":
    import yaml

    cfg = yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))
    build_feature_tables(cfg)
