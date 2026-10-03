"""Phase A: autoregressive generation + TOPLOC proofs + decode-mode top-k.

Do not call generate(..., output_hidden_states=True).
Increment recorder.step after each decode forward, starting at 0.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from inference.hooks import TopKRecorder, decoder_layers, prefill_positions


ROOT = Path(__file__).resolve().parents[1]


def load_config(path: Path | None = None) -> dict:
    path = path or ROOT / "config.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def n_checkpoints(cfg: dict) -> int:
    return cfg["tokens_per_generation"] // cfg["proof_interval_tokens"]


def attach_recorder(model, cfg: dict) -> TopKRecorder:
    rec = TopKRecorder(interval=cfg["proof_interval_tokens"], k=cfg["topk"])
    rec.attach(decoder_layers(model))
    return rec


def run_length_probe(cfg: dict) -> dict:
    """Already run on A10G; results in results/length_probe.json."""
    path = ROOT / "results" / "length_probe.json"
    if not path.exists():
        raise FileNotFoundError("run scripts/task01b_length_probe.py on the A10G first")
    return json.loads(path.read_text(encoding="utf-8"))


def generate_baseline(cfg: dict) -> None:
    from inference.task02 import generate_baseline as _run

    _run(cfg)


def generate_variant(cfg: dict, variant_id: str) -> None:
    raise NotImplementedError("TASK-04 Phase A runs on the A10G")


if __name__ == "__main__":
    import sys

    cfg = load_config()
    print("config loaded", cfg["model"], "n_checkpoints", n_checkpoints(cfg))
    print("prefill example positions", prefill_positions(40, cfg["tokens_per_generation"], cfg["proof_interval_tokens"]))
    if len(sys.argv) > 1 and sys.argv[1] == "task02":
        from inference.task02 import main as task02_main

        raise SystemExit(task02_main())
