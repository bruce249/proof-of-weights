"""TASK-01: post-pip bf16 assert, dummy TOPLOC proof, env dump, pin Qwen revision.

Run on the A10G after: pip install -U toploc transformers accelerate bitsandbytes
peft datasets scikit-learn web3 huggingface_hub pyyaml numpy pandas pyarrow
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import yaml


ROOT = Path(__file__).resolve().parents[1]


def _toploc_build_proofs_bytes(tensor):
    import toploc

    for name in ("build_proofs_bytes", "build_proof_bytes", "build_proofs"):
        fn = getattr(toploc, name, None)
        if callable(fn):
            return fn, name, fn(tensor)
    # Older / namespaced API
    for mod_name in ("toploc.core", "toploc.proofs", "toploc.csrc"):
        try:
            mod = __import__(mod_name, fromlist=["*"])
        except ImportError:
            continue
        for name in ("build_proofs_bytes", "build_proof_bytes", "build_proofs"):
            fn = getattr(mod, name, None)
            if callable(fn):
                return fn, f"{mod_name}.{name}", fn(tensor)
    raise RuntimeError(
        "Could not find toploc.build_proofs_bytes. "
        f"toploc attrs: {sorted(a for a in dir(toploc) if not a.startswith('_'))}"
    )


def dummy_proof() -> dict:
    import toploc

    # Official API: list[Tensor] on CPU. CUDA tensors segfault in ProofPoly.
    dummy = [torch.randn(5, 16, dtype=torch.bfloat16)]
    dummy += [torch.randn(16, dtype=torch.bfloat16) for _ in range(10)]
    proofs = toploc.build_proofs_bytes(dummy, decode_batching_size=3, topk=4)
    if not isinstance(proofs, list) or not proofs:
        raise TypeError(f"build_proofs_bytes returned {type(proofs)}, expected non-empty list")
    proof = proofs[0]
    if not isinstance(proof, (bytes, bytearray)):
        raise TypeError(f"proof item is {type(proof)}, expected bytes")
    if len(proof) == 0:
        raise RuntimeError("build_proofs_bytes returned empty bytes")
    return {
        "api": "toploc.build_proofs_bytes",
        "n_proofs": len(proofs),
        "nbytes": len(proof),
    }


def nvidia_smi_query() -> dict:
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total",
            "--format=csv,noheader",
        ],
        text=True,
    ).strip()
    parts = [p.strip() for p in out.split(",")]
    return {
        "gpu_name": parts[0] if parts else out,
        "driver_version": parts[1] if len(parts) > 1 else "",
        "memory_total": parts[2] if len(parts) > 2 else "",
        "nvidia_smi_raw": out,
    }


def pin_qwen(cfg: dict) -> dict:
    from huggingface_hub import model_info
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    model_id = cfg["model"]
    info = model_info(model_id)
    revision = info.sha
    print("HF revision", model_id, revision)

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    config = AutoConfig.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        revision=revision,
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    print(model)
    layers = model.model.layers
    n_layers = len(layers)
    print("decoder path model.model.layers n_layers", n_layers)
    if n_layers <= 0:
        raise RuntimeError("model.model.layers is empty")

    cfg_path = ROOT / "config.yaml"
    cfg["hf_revision"] = revision
    cfg["n_layers"] = n_layers
    cfg["hidden_size"] = int(getattr(config, "hidden_size", 0) or 0)
    with cfg_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)

    # Free VRAM so later tasks start clean
    del model, tokenizer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "model_id": model_id,
        "hf_revision": revision,
        "n_layers": n_layers,
        "hidden_size": cfg["hidden_size"],
        "decoder_path": "model.model.layers",
    }


def main() -> int:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA not available")
    if not torch.cuda.is_bf16_supported():
        raise SystemExit("native bf16 not supported on installed torch — stop")

    gpu_name = torch.cuda.get_device_name(0)
    print("gpu", gpu_name)
    if "A10G" not in gpu_name.replace(" ", ""):
        # Accept "NVIDIA A10G" / "A10G"
        if "A10G" not in gpu_name:
            raise SystemExit(f"expected A10G, got {gpu_name}")

    proof_info = dummy_proof()
    print("dummy proof", proof_info)

    cfg_path = ROOT / "config.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    model_info_out = pin_qwen(cfg)
    smi = nvidia_smi_query()

    env = {
        "written_at": datetime.now(timezone.utc).isoformat(),
        "gpu_name": gpu_name,
        "bf16_supported": True,
        "torch_version": torch.__version__,
        "cuda_version": getattr(torch.version, "cuda", None),
        "python": sys.version,
        **smi,
        **proof_info,
        **model_info_out,
    }
    out_path = ROOT / "results" / "environment.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(env, indent=2) + "\n", encoding="utf-8")
    print("wrote", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
