"""Build quant_8bit, quant_4bit, lora_merged, ft_drift. Train on Alpaca/Dolly only."""

from __future__ import annotations

import gc
import json
import os
import random
import re
import shutil
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")

import torch
import yaml
from datasets import Dataset, load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)

ROOT = Path(__file__).resolve().parents[1]
VARIANT_IDS = ("quant_8bit", "quant_4bit", "lora_merged", "ft_drift")
PINNED_QWEN_REVISION = "989aa7980e4cf806f80c7fef2b1adb7bc71aa306"
SEED_LORA = 202609231
SEED_SFT = 202609232
QKV = ("q_proj", "k_proj", "v_proj")
GATE_PROMPT = "In one or two sentences, explain why ice floats on water."
MAX_NEW_TOKENS = 64
N_ROWS = 300 * 1 * 8

SAMPLES_PATH = ROOT / "results" / "variant_samples.json"
MANIFEST_PATH = ROOT / "data" / "variant_manifest.json"


class SftOOM(Exception):
    pass


def sanitize(text: str) -> str:
    text = re.sub(r"hf_[A-Za-z0-9_\-]{8,}", "hf_REDACTED", text)
    text = re.sub(r"(?i)(authorization:\s*bearer\s+)\S+", r"\1REDACTED", text)
    return text


def log(msg: str) -> None:
    stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
    print(f"[{stamp}] {msg}", flush=True)


def load_config() -> dict:
    with (ROOT / "config.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    tmp.replace(path)


def release() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def mem() -> str:
    if not torch.cuda.is_available():
        return "no-cuda"
    alloc = torch.cuda.memory_allocated() / (1024**2)
    reserved = torch.cuda.memory_reserved() / (1024**2)
    return f"alloc_mib={alloc:.0f} reserved_mib={reserved:.0f}"


def claim_pid() -> None:
    pid_path = ROOT / "results" / "task03.pid"
    pid_path.parent.mkdir(parents=True, exist_ok=True)
    if pid_path.exists():
        raw = pid_path.read_text(encoding="utf-8").strip()
        old = int(raw) if raw.isdigit() else 0
        if old and old != os.getpid() and Path(f"/proc/{old}").exists():
            cmd = Path(f"/proc/{old}/cmdline").read_bytes()
            if b"build_variants" in cmd:
                log(f"another build_variants is already running pid={old}")
                raise SystemExit(0)
    pid_path.write_text(str(os.getpid()), encoding="utf-8")


def pin_seeds(cfg: dict) -> tuple[int, int]:
    seeds = cfg.setdefault("seeds", {})
    path = ROOT / "config.yaml"
    text = path.read_text(encoding="utf-8")
    changed = False
    if not seeds.get("lora"):
        seeds["lora"] = SEED_LORA
        text2, n = re.subn(r"(?m)^(\s*lora:\s*)(?:null|~)\s*$", rf"\g<1>{SEED_LORA}", text, count=1)
        if n != 1:
            raise RuntimeError("could not pin seeds.lora in config.yaml")
        text = text2
        changed = True
    if not seeds.get("sft"):
        seeds["sft"] = SEED_SFT
        text2, n = re.subn(r"(?m)^(\s*sft:\s*)(?:null|~)\s*$", rf"\g<1>{SEED_SFT}", text, count=1)
        if n != 1:
            raise RuntimeError("could not pin seeds.sft in config.yaml")
        text = text2
        changed = True
    if changed:
        path.write_text(text, encoding="utf-8")
        log(f"pinned seeds.lora={SEED_LORA} seeds.sft={SEED_SFT}")
    return int(seeds["lora"]), int(seeds["sft"])


def patch_yaml(updates: dict) -> None:
    path = ROOT / "config.yaml"
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    data.update(updates)
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def append_decision(line: str) -> None:
    path = ROOT / "results" / "decisions.md"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    if line in text:
        return
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text + line + "\n", encoding="utf-8")


def auth_failure(exc: BaseException) -> bool:
    name = type(exc).__name__
    msg = sanitize(str(exc)).lower()
    if name == "GatedRepoError":
        return True
    return any(s in msg for s in ("401", "403", "gated", "unauthorized", "restricted", "awaiting a review"))


def switch_to_llama(cfg: dict) -> dict:
    model_id = cfg["fallback"]
    log(f"resolving gated fallback {model_id}")
    try:
        from huggingface_hub import model_info

        info = model_info(model_id)
        revision = info.sha
    except Exception as exc:
        if auth_failure(exc):
            log("STOP: meta-llama/Llama-3.2-1B-Instruct is gated or unauthorized.")
            log("The user must run `hf auth login` on the box. No token was read or invented.")
            log(f"error_type={type(exc).__name__} error={sanitize(str(exc))[:500]}")
            raise SystemExit(4) from exc
        raise
    cfg["model"] = model_id
    cfg["hf_revision"] = revision
    patch_yaml({"model": model_id, "hf_revision": revision})
    append_decision(
        "- **TASK-03 Llama fallback:** full SFT OOM'd on Qwen2.5-1.5B-Instruct, so every variant moved to "
        f"`meta-llama/Llama-3.2-1B-Instruct` revision `{revision}`. `ft_drift` stays full SFT, not LoRA."
    )
    log(f"pinned llama revision {revision}")
    return cfg


def wipe_variant_outputs() -> None:
    var = ROOT / "artifacts" / "variants"
    if var.exists():
        for child in var.iterdir():
            if child.name.startswith("."):
                continue
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    for path in (SAMPLES_PATH, MANIFEST_PATH):
        if path.exists():
            path.unlink()


def is_coherent(text: str) -> tuple[bool, str]:
    sample = " ".join(text.split())
    if len(sample) < 4:
        return False, "too short"
    letters = [c for c in sample if c.isalpha()]
    if len(letters) < 4:
        return False, "too few letters"
    if len(set(c.lower() for c in letters)) < 3 and len(letters) >= 8:
        return False, "low letter variety"
    words = sample.split()
    if len(words) >= 6:
        counts: dict[str, int] = {}
        for word in words:
            key = word.lower()
            counts[key] = counts.get(key, 0) + 1
        if max(counts.values()) / len(words) > 0.6:
            return False, "repetitive"
    vowels = sum(c.lower() in "aeiou" for c in letters)
    if vowels / len(letters) < 0.15:
        return False, "vowel-starved"
    weird = sum(not (c.isalnum() or c.isspace() or c in ".,;:'\"?!-()") for c in sample)
    if weird / len(sample) > 0.15:
        return False, "noisy symbols"
    return True, "ok"


def input_device(model) -> torch.device:
    return next(model.parameters()).device


def load_tokenizer(model_id: str, revision: str | None):
    kwargs = {"revision": revision} if revision else {}
    tok = AutoTokenizer.from_pretrained(model_id, **kwargs)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def load_causal(model_id: str, revision: str | None, quantization_config=None):
    kwargs = {
        "dtype": torch.bfloat16,
        "device_map": "cuda",
        "attn_implementation": "sdpa",
    }
    if revision:
        kwargs["revision"] = revision
    if quantization_config is not None:
        kwargs["quantization_config"] = quantization_config
    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        return model, "sdpa"
    except Exception as exc:
        msg = sanitize(str(exc)).lower()
        if quantization_config is None or not any(s in msg for s in ("sdpa", "attention", "attn")):
            raise
        log(f"sdpa load failed ({sanitize(str(exc))[:240]}); retrying eager")
        release()
        kwargs["attn_implementation"] = "eager"
        model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        return model, "eager"


def greedy_sample(model, tok) -> str:
    model.eval()
    enc = tok.apply_chat_template(
        [{"role": "user", "content": GATE_PROMPT}],
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
    )
    device = input_device(model)
    input_ids = enc["input_ids"].to(device)
    attention = enc["attention_mask"].to(device) if "attention_mask" in enc else None
    with torch.no_grad():
        out = model.generate(
            input_ids=input_ids,
            attention_mask=attention,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    new_tokens = out[0, input_ids.shape[-1] :]
    return tok.decode(new_tokens, skip_special_tokens=True)


def remember_arch(cfg: dict, model) -> None:
    n_layers = len(model.model.layers)
    hidden = int(model.config.hidden_size)
    cfg["n_layers"] = n_layers
    cfg["hidden_size"] = hidden
    log(f"architecture layers={n_layers} hidden={hidden} {mem()}")


def quant_spec(variant_id: str):
    if variant_id == "quant_8bit":
        return BitsAndBytesConfig(load_in_8bit=True)
    if variant_id == "quant_4bit":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    raise KeyError(variant_id)


def jsonable_quant(qcfg: BitsAndBytesConfig) -> dict:
    raw = qcfg.to_dict()
    out = {}
    for key, value in raw.items():
        if isinstance(value, torch.dtype):
            out[key] = str(value).replace("torch.", "")
        else:
            try:
                json.dumps(value)
            except TypeError:
                out[key] = str(value)
            else:
                out[key] = value
    return out


def record_sample(samples: dict, variant_id: str, text: str, extra: dict | None = None) -> None:
    ok, reason = is_coherent(text)
    row = {"passed": ok, "reason": reason, "sample": text, "decode": "greedy", "max_new_tokens": MAX_NEW_TOKENS}
    if extra:
        row.update(extra)
    samples["variants"][variant_id] = row
    write_json(SAMPLES_PATH, samples)
    log(f"gate {variant_id} passed={ok} reason={reason} sample={text!r}")
    if not ok:
        raise RuntimeError(f"{variant_id} failed the coherent-text gate: {reason}: {text!r}")


def gate_quant(cfg: dict, samples: dict, manifest: dict, variant_id: str) -> None:
    log(f"loading {variant_id}")
    qcfg = quant_spec(variant_id)
    tok = load_tokenizer(cfg["model"], cfg["hf_revision"])
    model, attn = load_causal(cfg["model"], cfg["hf_revision"], qcfg)
    remember_arch(cfg, model)
    text = greedy_sample(model, tok)
    del model
    release()
    record_sample(samples, variant_id, text, {"attn": attn})
    manifest["variants"][variant_id] = {
        "tamper_class": cfg["tamper_class_ids"][variant_id],
        "weights_dir": None,
        "attn": attn,
        "quantization_config": jsonable_quant(qcfg),
        "coherent": True,
    }
    write_json(MANIFEST_PATH, manifest)


def row_fields(row: dict) -> tuple[str, str, str]:
    instruction = str(row.get("instruction") or "").strip()
    extra = str(row.get("input") or row.get("context") or "").strip()
    output = str(row.get("output") or row.get("response") or "").strip()
    return instruction, extra, output


def load_train(cfg: dict):
    primary = cfg["sft_lora"]["train_data"]
    fallback = cfg["sft_lora"]["train_data_fallback"]
    for name in (primary, fallback):
        if "ultrachat" in name.lower():
            raise RuntimeError(f"refusing eval set as train data: {name}")
    last = None
    for name in (primary, fallback):
        try:
            log(f"loading dataset {name}")
            ds = load_dataset(name, split="train")
            log(f"dataset {name} rows={len(ds)} cols={ds.column_names}")
            return name, "train", ds
        except Exception as exc:
            last = exc
            log(f"dataset {name} failed: {type(exc).__name__}: {sanitize(str(exc))[:400]}")
    raise RuntimeError(f"no train dataset: {sanitize(str(last))[:400]}")


def select_indices(ds, seed: int, n: int) -> list[int]:
    rng = random.Random(seed)
    order = list(range(len(ds)))
    rng.shuffle(order)
    chosen: list[int] = []
    for index in order:
        instruction, _, output = row_fields(ds[int(index)])
        if instruction and output:
            chosen.append(int(index))
        if len(chosen) >= n:
            break
    if len(chosen) < n:
        raise RuntimeError(f"only {len(chosen)} usable train rows, need {n}")
    return chosen


def build_lm_dataset(ds, indices: list[int], tok, max_len: int) -> Dataset:
    texts = []
    for index in indices:
        instruction, extra, output = row_fields(ds[int(index)])
        user = instruction if not extra else f"{instruction}\n\n{extra}"
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": output},
        ]
        text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        if not isinstance(text, str):
            text = text[0] if isinstance(text, list) else str(text)
        texts.append(text)
    log(f"formatted example chars={len(texts[0])} head={texts[0][:180]!r}")
    enc = tok(texts, truncation=True, max_length=max_len, padding=False, add_special_tokens=False)
    return Dataset.from_dict(
        {
            "input_ids": enc["input_ids"],
            "attention_mask": enc["attention_mask"],
            "labels": [list(ids) for ids in enc["input_ids"]],
        }
    )


class StepLog(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        keep = {k: logs[k] for k in ("loss", "learning_rate", "epoch", "grad_norm") if k in logs}
        if keep:
            log(f"train step={state.global_step} {keep} {mem()}")


def assert_qkv(model) -> None:
    names = {module_name.split(".")[-1] for module_name, _ in model.named_modules()}
    missing = [name for name in QKV if name not in names]
    if missing:
        raise RuntimeError(f"attention QKV modules missing: {missing}")


def last_loss(trainer: Trainer):
    for entry in reversed(trainer.state.log_history):
        if "loss" in entry:
            return float(entry["loss"])
    return None


def save_weights(model, tok, variant_id: str) -> Path:
    out = ROOT / "artifacts" / "variants" / variant_id
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    model.config.use_cache = True
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.use_cache = True
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    names = sorted(p.name for p in out.iterdir())
    log(f"saved {variant_id} files={names}")
    if not any(name.endswith(".safetensors") or name.endswith(".bin") for name in names):
        raise RuntimeError(f"{variant_id} save produced no weight file")
    return out


def train_one(
    cfg: dict,
    ds,
    tok,
    variant_id: str,
    seed: int,
    indices: list[int],
) -> tuple[object, float | None]:
    sft = cfg["sft_lora"]["sft"]
    lora = cfg["sft_lora"]["lora"]
    max_len = int(sft["max_seq_length"])
    steps = int(lora["steps"] if variant_id == "lora_merged" else sft["steps"])
    if steps != 300:
        raise RuntimeError(f"locked step count is 300, got {steps} for {variant_id}")
    train_ds = build_lm_dataset(ds, indices, tok, max_len)
    set_seed(seed)
    model, attn = load_causal(cfg["model"], cfg["hf_revision"])
    assert_qkv(model)
    model.config.use_cache = False
    if variant_id == "lora_merged":
        model = get_peft_model(
            model,
            LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=int(lora["r"]),
                lora_alpha=32,
                lora_dropout=0.05,
                bias="none",
                target_modules=list(QKV),
            ),
        )
        lr = 2.0e-4
        optim = "adamw_torch"
    else:
        lr = float(sft["lr"])
        optim = str(sft["optim"])
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    tmp = ROOT / "artifacts" / "variants" / f"_trainer_{variant_id}"
    if tmp.exists():
        shutil.rmtree(tmp)
    args = TrainingArguments(
        output_dir=str(tmp),
        max_steps=steps,
        per_device_train_batch_size=int(sft["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(sft["gradient_accumulation_steps"]),
        learning_rate=lr,
        bf16=True,
        optim=optim,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type="constant",
        warmup_steps=0,
        logging_steps=10,
        logging_first_step=True,
        save_strategy="no",
        report_to="none",
        seed=seed,
        data_seed=seed,
        train_sampling_strategy="sequential",
        dataloader_num_workers=0,
        disable_tqdm=True,
        remove_unused_columns=True,
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        data_collator=DataCollatorForLanguageModeling(tokenizer=tok, mlm=False),
        processing_class=tok,
        callbacks=[StepLog()],
    )
    log(f"training {variant_id} steps={steps} lr={lr} optim={optim} attn={attn} rows={len(indices)} {mem()}")
    try:
        trainer.train()
    except (torch.cuda.OutOfMemoryError, torch.OutOfMemoryError) as exc:
        if variant_id == "ft_drift":
            raise SftOOM(sanitize(str(exc))[:400]) from exc
        raise
    except RuntimeError as exc:
        if variant_id == "ft_drift" and "out of memory" in str(exc).lower():
            raise SftOOM(sanitize(str(exc))[:400]) from exc
        raise
    loss = last_loss(trainer)
    trained = trainer.model
    if hasattr(trained, "module"):
        trained = trained.module
    if variant_id == "lora_merged":
        trained = trained.merge_and_unload()
        log("lora merge_and_unload done")
    del trainer
    release()
    if tmp.exists():
        shutil.rmtree(tmp)
    return trained, loss


def gate_saved(cfg: dict, samples: dict, variant_id: str, weights: Path) -> None:
    log(f"reloading saved {variant_id} for the coherent-text gate")
    tok = load_tokenizer(str(weights), None)
    model, attn = load_causal(str(weights), None)
    text = greedy_sample(model, tok)
    del model
    release()
    record_sample(samples, variant_id, text, {"attn": attn, "weights_dir": str(weights.relative_to(ROOT)).replace("\\", "/")})


def train_and_gate(
    cfg: dict,
    ds,
    dataset_name: str,
    split: str,
    samples: dict,
    manifest: dict,
    variant_id: str,
    seed: int,
) -> None:
    log(f"start {variant_id} seed={seed}")
    tok = load_tokenizer(cfg["model"], cfg["hf_revision"])
    indices = select_indices(ds, seed, N_ROWS)
    model, loss = train_one(cfg, ds, tok, variant_id, seed, indices)
    weights = save_weights(model, tok, variant_id)
    del model
    release()
    rel = str(weights.relative_to(ROOT)).replace("\\", "/")
    spec = cfg["sft_lora"]["sft"]
    lora = cfg["sft_lora"]["lora"]
    row = {
        "tamper_class": cfg["tamper_class_ids"][variant_id],
        "weights_dir": rel,
        "dataset": dataset_name,
        "split": split,
        "seed": seed,
        "slice_indices": indices,
        "steps": 300,
        "max_seq_length": int(spec["max_seq_length"]),
        "per_device_train_batch_size": int(spec["per_device_train_batch_size"]),
        "gradient_accumulation_steps": int(spec["gradient_accumulation_steps"]),
        "gradient_checkpointing": True,
        "lr_scheduler_type": "constant",
        "train_loss_last": loss,
        "coherent": False,
    }
    if variant_id == "lora_merged":
        row.update(
            {
                "r": int(lora["r"]),
                "lora_alpha": 32,
                "lora_dropout": 0.05,
                "target_modules": list(QKV),
                "merge_and_unload": True,
                "learning_rate": 2.0e-4,
                "optim": "adamw_torch",
            }
        )
    else:
        row.update(
            {
                "lora": False,
                "learning_rate": float(spec["lr"]),
                "optim": str(spec["optim"]),
            }
        )
    manifest["variants"][variant_id] = row
    write_json(MANIFEST_PATH, manifest)
    gate_saved(cfg, samples, variant_id, weights)
    manifest["variants"][variant_id]["coherent"] = True
    write_json(MANIFEST_PATH, manifest)


def build_all(cfg: dict) -> None:
    if "Qwen2.5-1.5B" in str(cfg["model"]) and str(cfg["hf_revision"]) != PINNED_QWEN_REVISION:
        raise RuntimeError(f"refusing to move pinned Qwen revision: {cfg['hf_revision']}")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("need a CUDA device with native bf16")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    lora_seed, sft_seed = pin_seeds(cfg)
    samples = {
        "gate_prompt": GATE_PROMPT,
        "decode": "greedy",
        "max_new_tokens": MAX_NEW_TOKENS,
        "base_model": cfg["model"],
        "hf_revision": cfg["hf_revision"],
        "variants": {},
    }
    manifest = {
        "complete": False,
        "base_model": cfg["model"],
        "hf_revision": cfg["hf_revision"],
        "precision": "bfloat16",
        "eval_dataset_not_used_for_training": cfg["eval_prompts"],
        "seeds": {"lora": lora_seed, "sft": sft_seed},
        "n_rows_per_train": N_ROWS,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "variants": {},
    }
    write_json(SAMPLES_PATH, samples)
    write_json(MANIFEST_PATH, manifest)
    gate_quant(cfg, samples, manifest, "quant_8bit")
    gate_quant(cfg, samples, manifest, "quant_4bit")
    dataset_name, split, ds = load_train(cfg)
    manifest["train_dataset"] = dataset_name
    manifest["train_split"] = split
    manifest["train_rows_full"] = len(ds)
    manifest["train_fingerprint"] = getattr(ds, "_fingerprint", None)
    manifest["n_layers"] = cfg.get("n_layers")
    manifest["hidden_size"] = cfg.get("hidden_size")
    write_json(MANIFEST_PATH, manifest)
    train_and_gate(cfg, ds, dataset_name, split, samples, manifest, "ft_drift", sft_seed)
    train_and_gate(cfg, ds, dataset_name, split, samples, manifest, "lora_merged", lora_seed)
    manifest["complete"] = True
    manifest["finished_at"] = datetime.now(timezone.utc).isoformat()
    write_json(MANIFEST_PATH, manifest)
    log("TASK03_DONE " + json.dumps({k: samples["variants"][k]["passed"] for k in VARIANT_IDS}))


def assert_trainer_args() -> None:
    tmp = ROOT / "artifacts" / "variants" / "_argcheck"
    if tmp.exists():
        shutil.rmtree(tmp)
    TrainingArguments(
        output_dir=str(tmp),
        max_steps=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=1e-5,
        bf16=True,
        optim="adamw_bnb_8bit",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        lr_scheduler_type="constant",
        warmup_steps=0,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        seed=1,
        train_sampling_strategy="sequential",
        dataloader_num_workers=0,
        disable_tqdm=True,
    )
    if tmp.exists():
        shutil.rmtree(tmp)


def restart_on_llama() -> None:
    append_decision(
        "- **TASK-03 SFT OOM:** full SFT ran out of memory; restarting every variant on "
        "meta-llama/Llama-3.2-1B-Instruct. ft_drift stays full SFT, not LoRA."
    )
    log("SFT OOM. Restarting all four variants on Llama-3.2-1B-Instruct.")
    env = os.environ.copy()
    env["TASK03_FORCE_LLAMA"] = "1"
    os.chdir(ROOT)
    os.execve(sys.executable, [sys.executable, "-u", "-m", "inference.build_variants"], env)


def main() -> int:
    claim_pid()
    cfg = load_config()
    if os.environ.get("TASK03_FORCE_LLAMA") == "1":
        cfg = switch_to_llama(cfg)
        wipe_variant_outputs()
    log(
        f"TASK-03 start model={cfg['model']} revision={cfg['hf_revision']} "
        f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'}"
    )
    assert_trainer_args()
    try:
        build_all(cfg)
    except SftOOM as exc:
        log(f"caught SFT OOM: {sanitize(str(exc))[:400]}")
        if os.environ.get("TASK03_FORCE_LLAMA") == "1":
            log("SFT OOM on Llama-3.2-1B as well. Not converting ft_drift to LoRA.")
            return 5
        restart_on_llama()
    except SystemExit:
        raise
    except Exception:
        log(sanitize(traceback.format_exc())[-4000:])
        return 1
    return 0


if __name__ == "__main__":
    t0 = time.time()
    code = main()
    log(f"exit={code} elapsed_s={time.time() - t0:.1f}")
    raise SystemExit(code)
