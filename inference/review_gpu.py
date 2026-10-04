"""GPU experiments requested by the conference review.

Stages (each resumable; outputs under data/review/ and results/review_gpu_*.json):

  qwen_new    Experiment B/C. New Qwen2.5-1.5B modifications never seen in training:
              lora_nf4 (LoRA-merged weights loaded in NF4) and five selective NF4
              quantisations (first 14 layers, last 14, even layers, odd layers,
              MLP-only). Prompts 0-19 (attacker's calibration set) and 150-199
              (test). Provider side: decode-time per-layer top-k. Validator side:
              baseline SDPA prefill on the same tokens. Also last-hidden TOPLOC stats.
  overhead    Experiment E. Real per-layer TOPLOC commitments (28 proofs per
              sequence): bytes, build time, verify time; validator prefill time
              with and without per-layer hooks. 10 prompts.
  throughput  bf16 / 8-bit / NF4 decoding at batch 1, 8, 32 (128 new tokens).
  smol        Experiment D. SmolLM2-1.7B-Instruct (Llama architecture, 24 layers),
              128-token answers, prompts 0-99: honest (reference decode, eager,
              batch 4), tampered (NF4, 8-bit), eager-served tampering (prompts 50-99).

  python -m inference.review_gpu smoke      # 2 prompts per stage, writes nothing permanent
  python -m inference.review_gpu all
"""

from __future__ import annotations

import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from inference.build_variants import input_device, log, release
from inference.hooks import TopKRecorder, decoder_layers, stack_buf
from inference.task02 import (
    LastHiddenTap,
    _proofs_from_hiddens,
    _seq_stats,
    _set_attn,
    _verify,
    encode_prompt,
    generate_one,
    load_config,
    replay_batch,
    replay_one,
)
from inference.task04 import _full_ids, _prefill_decode_hiddens, _prefill_topk

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "review"
SMOKE = False
LINEAR = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
          "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj")
SMOL_ID = "HuggingFaceTB/SmolLM2-1.7B-Instruct"


def prompts() -> dict[int, dict]:
    return {p["prompt_id"]: p for p in json.loads((ROOT / "data" / "prompts.json").read_text(encoding="utf-8"))["prompts"]}


def write_json(path: Path, data) -> None:
    if SMOKE:
        log(f"[smoke] would write {path.name}")
        return
    path.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def save_npz(path: Path, **arrays) -> None:
    if SMOKE:
        log(f"[smoke] would write {path.name} " + str({k: v.shape for k, v in arrays.items()}))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def load_tok(model_id: str, revision: str | None):
    kw = {"revision": revision} if revision else {}
    tok = AutoTokenizer.from_pretrained(model_id, **kw)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok


def load_lm(model_id: str, revision: str | None, attn: str = "sdpa", q=None):
    kw = {"dtype": torch.bfloat16, "device_map": "cuda", "attn_implementation": attn}
    if revision:
        kw["revision"] = revision
    if q is not None:
        kw["quantization_config"] = q
    m = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    m.eval()
    return m


def nf4(skip: list[str] | None = None) -> BitsAndBytesConfig:
    kw = dict(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    if skip is not None:
        kw["llm_int8_skip_modules"] = skip
    return BitsAndBytesConfig(**kw)


def quantised_layers(model) -> dict:
    """Which decoder layers / sublayers hold bitsandbytes 4- or 8-bit linears, and parameter share."""
    import bitsandbytes as bnb

    q_layers, q_names, q_params, all_params = set(), [], 0, 0
    for name, mod in model.named_modules():
        if not name.startswith("model.layers.") or not isinstance(mod, torch.nn.Linear):
            continue
        n = mod.in_features * mod.out_features
        all_params += n
        if isinstance(mod, (bnb.nn.Linear4bit, bnb.nn.Linear8bitLt)):
            q_params += n
            q_names.append(name)
            q_layers.add(int(name.split(".")[2]))
    return {"layers": sorted(q_layers), "n_modules": len(q_names), "param_fraction": q_params / max(all_params, 1),
            "attn_quantised": any(".self_attn." in n for n in q_names), "mlp_quantised": any(".mlp." in n for n in q_names)}


# ------------------------------------------------------------------ Qwen new modifications
def qwen_specs(n_layers: int = 28) -> dict:
    def skip_layers(layers):
        return [f"model.layers.{i}.{l}" for i in layers for l in LINEAR] + ["lm_head"]

    half = n_layers // 2
    return {
        "lora_nf4": {"weights": "artifacts/variants/lora_merged", "skip": None,
                     "expect_layers": list(range(n_layers)), "expect_attn": True, "expect_mlp": True},
        "nf4_first14": {"weights": None, "skip": skip_layers(range(half, n_layers)),
                        "expect_layers": list(range(half)), "expect_attn": True, "expect_mlp": True},
        "nf4_last14": {"weights": None, "skip": skip_layers(range(half)),
                       "expect_layers": list(range(half, n_layers)), "expect_attn": True, "expect_mlp": True},
        "nf4_even": {"weights": None, "skip": skip_layers(range(1, n_layers, 2)),
                     "expect_layers": list(range(0, n_layers, 2)), "expect_attn": True, "expect_mlp": True},
        "nf4_odd": {"weights": None, "skip": skip_layers(range(0, n_layers, 2)),
                    "expect_layers": list(range(1, n_layers, 2)), "expect_attn": True, "expect_mlp": True},
        "nf4_mlp": {"weights": None, "skip": [f"model.layers.{i}.{l}" for i in range(n_layers) for l in LINEAR[:4]] + ["lm_head"],
                    "expect_layers": list(range(n_layers)), "expect_attn": False, "expect_mlp": True},
    }


def stage_qwen_new(cfg: dict) -> None:
    P = prompts()
    pids = [0, 150] if SMOKE else list(range(0, 20)) + list(range(150, 200))
    stats_path = ROOT / "results" / "review_gpu_qwen_new.json"
    saved = read_json(stats_path)
    btok = load_tok(cfg["model"], cfg["hf_revision"])
    base = load_lm(cfg["model"], cfg["hf_revision"])
    brec = TopKRecorder(interval=cfg["proof_interval_tokens"], k=cfg["topk"])
    brec.attach(decoder_layers(base))
    n_new = cfg["tokens_per_generation"]
    for name, spec in qwen_specs().items():
        if name in saved and (OUT / f"qwen_{name}.npz").exists():
            log(f"qwen_new {name} done")
            continue
        src = str(ROOT / spec["weights"]) if spec["weights"] else cfg["model"]
        rev = None if spec["weights"] else cfg["hf_revision"]
        tok = load_tok(src, rev)
        model = load_lm(src, rev, q=nf4(spec["skip"]))
        info = quantised_layers(model)
        ok = (info["layers"] == spec["expect_layers"] and info["attn_quantised"] == spec["expect_attn"]
              and info["mlp_quantised"] == spec["expect_mlp"])
        log(f"qwen_new {name} quantised={info} expected_layers={spec['expect_layers']} ok={ok}")
        if not ok:
            raise SystemExit(f"{name}: quantised layers do not match the specification; stopping")
        info["memory_footprint_bytes"] = int(model.get_memory_footprint())
        rec = TopKRecorder(interval=cfg["proof_interval_tokens"], k=cfg["topk"])
        rec.attach(decoder_layers(model))
        tap = LastHiddenTap()
        tap.attach(model)
        L, K, C = len(decoder_layers(model)), cfg["topk"], n_new // cfg["proof_interval_tokens"]
        pv = np.zeros((len(pids), C, L, K), np.float32); pi = np.zeros_like(pv, np.int32)
        rv = np.zeros_like(pv); ri = np.zeros_like(pi)
        rows, t0 = [], time.time()
        for j, pid in enumerate(pids):
            ids = encode_prompt(tok, P[pid]["text"]).to(input_device(model))
            g = generate_one(model, tok, rec, tap, ids, cfg)
            pv[j], pi[j] = g["values"], g["indices"]
            full, plen = _full_ids(btok, P[pid]["text"], g["tokens"], input_device(base))
            if plen != g["prompt_len"]:
                raise SystemExit(f"{name} prompt {pid}: prompt length mismatch {plen} vs {g['prompt_len']}")
            v, i = _prefill_topk(base, brec, full, plen, g["tokens"], cfg)
            rv[j], ri[j] = v.numpy(), i.numpy()
            s = _seq_stats(_verify(_prefill_decode_hiddens(base, full, plen, n_new), g["proofs"], cfg))
            s.update({"prompt_id": pid, "natural_eos_pos": g["natural_eos_pos"]})
            rows.append(s)
            if pid % 10 == 0:
                log(f"qwen_new {name} [{pid:03d}] exp={s['exp_mismatches_max']:.0f}")
        info["seconds"] = time.time() - t0
        save_npz(OUT / f"qwen_{name}.npz", prov_values=pv, prov_indices=pi, ref_values=rv, ref_indices=ri,
                 prompt_ids=np.array(pids, np.int32))
        saved = read_json(stats_path)
        saved[name] = {"quantisation": info, "seq_stats": rows}
        write_json(stats_path, saved)
        rec.detach(); tap.detach()
        del model, tok, rec, tap
        release()
    brec.detach()
    del base
    release()


# ------------------------------------------------------------------ overhead
class AllLayerTap:
    """Last-position output of every decoder layer at every decode step (CPU bf16)."""

    def __init__(self, layers):
        self.buf = [[] for _ in layers]
        self.on = False
        self.h = [l.register_forward_hook(self._mk(i)) for i, l in enumerate(layers)]

    def _mk(self, i):
        def hook(_m, _i, out):
            if self.on:
                t = out[0] if isinstance(out, tuple) else out
                self.buf[i].append(t[0, -1, :].detach().to("cpu", torch.bfloat16))
        return hook

    def remove(self):
        for h in self.h:
            h.remove()


def stage_overhead(cfg: dict) -> None:
    import toploc

    path = ROOT / "results" / "review_gpu_overhead.json"
    if path.exists() and not SMOKE:
        log("overhead done")
        return
    P = prompts()
    pids = [0] if SMOKE else list(range(10))
    tok = load_tok(cfg["model"], cfg["hf_revision"])
    model = load_lm(cfg["model"], cfg["hf_revision"])
    layers = decoder_layers(model)
    n_new, iv, k = cfg["tokens_per_generation"], cfg["proof_interval_tokens"], cfg["topk"]
    res = {"prompts": pids, "per_seq": []}
    for pid in pids:
        ids = encode_prompt(tok, P[pid]["text"]).to(input_device(model))
        plen = ids.shape[1]
        tap = AllLayerTap(layers)
        with torch.no_grad():
            out = model(input_ids=ids, use_cache=True)
            past, logits = out.past_key_values, out.logits[:, -1, :]
            gen = []
            tap.on = True
            for _ in range(n_new):
                logits = logits.clone(); logits[0, tok.eos_token_id] = -float("inf")
                nxt = int(logits[0].argmax()); gen.append(nxt)
                o = model(input_ids=torch.tensor([[nxt]], device=ids.device), past_key_values=past, use_cache=True)
                past, logits = o.past_key_values, o.logits[:, -1, :]
            tap.on = False
        tap.remove()
        # provider: build one TOPLOC proof stream per layer
        t = time.time()
        proofs = [toploc.build_proofs_bytes(tap.buf[l], decode_batching_size=iv, topk=k, skip_prefill=True) for l in range(len(layers))]
        build_s = time.time() - t
        nbytes = [sum(len(p) for p in pl) for pl in proofs]
        # validator: one prefill, plain and with per-layer capture
        full = torch.cat([ids, torch.tensor([gen], device=ids.device)], dim=1)
        torch.cuda.synchronize(); t = time.time()
        with torch.no_grad():
            model(input_ids=full)
        torch.cuda.synchronize(); plain_s = time.time() - t
        cap = {}
        hs = [l.register_forward_hook((lambda i: lambda _m, _i, o: cap.__setitem__(i, (o[0] if isinstance(o, tuple) else o)[0, plen:plen + n_new].detach()))(i))
              for i, l in enumerate(layers)]
        torch.cuda.synchronize(); t = time.time()
        with torch.no_grad():
            model(input_ids=full)
        torch.cuda.synchronize(); hooked_s = time.time() - t
        for h in hs:
            h.remove()
        t = time.time()
        exp_max = []
        for l in range(len(layers)):
            ref = [cap[l][j].to("cpu", torch.bfloat16) for j in range(n_new)]
            r = toploc.verify_proofs_bytes(ref, proofs[l], decode_batching_size=iv, topk=k, skip_prefill=True)
            exp_max.append(max(float(x.exp_mismatches) for x in r))
        verify_s = time.time() - t
        res["per_seq"].append({"prompt_id": pid, "prompt_len": plen, "bytes_per_layer": nbytes,
                               "bytes_total_all_layers": sum(nbytes), "bytes_last_layer": nbytes[-1],
                               "build_seconds_all_layers": build_s, "verify_seconds_all_layers": verify_s,
                               "prefill_seconds_plain": plain_s, "prefill_seconds_capture_all_layers": hooked_s,
                               "honest_exp_mismatch_max_by_layer": exp_max})
        log(f"overhead [{pid}] bytes={sum(nbytes)} build={build_s:.2f}s verify={verify_s:.2f}s prefill={plain_s:.3f}/{hooked_s:.3f}s")
        del tap, proofs, cap
    write_json(path, res)
    del model
    release()


# ------------------------------------------------------------------ throughput
def stage_throughput(cfg: dict) -> None:
    path = ROOT / "results" / "review_gpu_throughput.json"
    saved = read_json(path)
    P = prompts()
    modes = {"bf16": None, "int8": BitsAndBytesConfig(load_in_8bit=True), "nf4": nf4()}
    batches = [1, 8] if SMOKE else [1, 8, 32]
    n_new = 16 if SMOKE else 128
    for mode, q in modes.items():
        if mode in saved and not SMOKE:
            log(f"throughput {mode} done")
            continue
        tok = load_tok(cfg["model"], cfg["hf_revision"])
        model = load_lm(cfg["model"], cfg["hf_revision"], q=q)
        rows = {}
        for b in batches:
            texts = [tok.apply_chat_template([{"role": "user", "content": P[i]["text"]}], add_generation_prompt=True, tokenize=False) for i in range(b)]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(input_device(model))
            times = []
            torch.cuda.reset_peak_memory_stats()
            for rep in range(3):
                torch.cuda.synchronize(); t = time.time()
                with torch.no_grad():
                    model.generate(**enc, max_new_tokens=n_new, min_new_tokens=n_new, do_sample=False, pad_token_id=tok.pad_token_id)
                torch.cuda.synchronize()
                if rep > 0:
                    times.append(time.time() - t)
            s = float(np.mean(times))
            rows[str(b)] = {"seconds": s, "tokens_per_second": b * n_new / s, "peak_memory_mib": torch.cuda.max_memory_allocated() / 2**20}
            log(f"throughput {mode} batch={b} tok/s={b * n_new / s:.1f}")
        saved[mode] = {"new_tokens": n_new, "timed_runs": 2, "warmup_runs": 1, "by_batch": rows}
        write_json(path, saved)
        del model, tok
        release()


# ------------------------------------------------------------------ second model
def stage_smol(cfg: dict) -> None:
    from huggingface_hub import model_info

    scfg = copy.deepcopy(cfg)
    scfg["tokens_per_generation"] = 32 if SMOKE else 128
    n_new, iv, K = scfg["tokens_per_generation"], scfg["proof_interval_tokens"], scfg["topk"]
    C = n_new // iv
    P = prompts()
    pids = [0, 1] if SMOKE else list(range(100))
    att_pids = [0, 1] if SMOKE else list(range(50, 100))
    rev = model_info(SMOL_ID).sha
    meta_path = ROOT / "results" / "review_gpu_smol.json"
    meta = read_json(meta_path)
    meta.update({"model": SMOL_ID, "revision": rev, "tokens_per_generation": n_new, "prompts": [pids[0], pids[-1]]})
    tok = load_tok(SMOL_ID, rev)
    base = load_lm(SMOL_ID, rev)
    L = len(decoder_layers(base))
    meta["n_layers"] = L
    meta["hidden_size"] = int(base.config.hidden_size)
    brec = TopKRecorder(interval=iv, k=K); brec.attach(decoder_layers(base))
    btap = LastHiddenTap(); btap.attach(base)

    def ref_topk(tokens_by_pid, sel):
        rv = np.zeros((len(sel), C, L, K), np.float32); ri = np.zeros_like(rv, np.int32)
        for j, pid in enumerate(sel):
            full, plen = _full_ids(tok, P[pid]["text"], tokens_by_pid[pid], input_device(base))
            v, i = _prefill_topk(base, brec, full, plen, tokens_by_pid[pid], scfg)
            rv[j], ri[j] = v.numpy(), i.numpy()
        return rv, ri

    # reference generation (honest decode on the reference kernel)
    seq_path = OUT / "smol_sequences.json"
    seqs = {int(k): v for k, v in read_json(seq_path).items()}
    if not (OUT / "smol_ref_decode.npz").exists() or SMOKE:
        pv = np.zeros((len(pids), C, L, K), np.float32); pi = np.zeros_like(pv, np.int32)
        for j, pid in enumerate(pids):
            g = generate_one(base, tok, brec, btap, encode_prompt(tok, P[pid]["text"]).to(input_device(base)), scfg)
            pv[j], pi[j] = g["values"], g["indices"]
            seqs[pid] = g["tokens"]
            if pid % 20 == 0:
                log(f"smol ref [{pid:03d}]")
        rv, ri = ref_topk(seqs, pids)
        save_npz(OUT / "smol_ref_decode.npz", prov_values=pv, prov_indices=pi, ref_values=rv, ref_indices=ri, prompt_ids=np.array(pids, np.int32))
        if not SMOKE:
            seq_path.write_text(json.dumps({str(k): v for k, v in seqs.items()}) + "\n", encoding="utf-8")
    ref_v, ref_i = None, None
    with np.load(OUT / "smol_ref_decode.npz") if not SMOKE else _nullctx() as z:
        if z is not None:
            ref_v, ref_i = z["ref_values"], z["ref_indices"]
    if SMOKE:
        ref_v, ref_i = ref_topk(seqs, pids)

    # honest replays of the reference tokens
    for name, attn, bsz in (("eager", "eager", 1), ("batch4", "sdpa", 4)):
        out = OUT / f"smol_honest_{name}.npz"
        if out.exists() and not SMOKE:
            continue
        _set_attn(base, attn)
        pv = np.zeros((len(pids), C, L, K), np.float32); pi = np.zeros_like(pv, np.int32)
        bsz_eff = min(bsz, len(pids))
        for s0 in range(0, len(pids), bsz_eff):
            chunk = pids[s0:s0 + bsz_eff]
            ids_list = [encode_prompt(tok, P[p]["text"]) for p in chunk]
            if bsz_eff == 1:
                rows = [replay_one(base, tok, brec, btap, ids_list[0].to(input_device(base)), seqs[chunk[0]], scfg)]
            else:
                rows = replay_batch(base, tok, brec, btap, ids_list, [seqs[p] for p in chunk], scfg)
            for off, row in enumerate(rows):
                pv[s0 + off], pi[s0 + off] = row["values"], row["indices"]
        _set_attn(base, "sdpa")
        save_npz(out, prov_values=pv, prov_indices=pi, ref_values=ref_v, ref_indices=ref_i, prompt_ids=np.array(pids, np.int32))
        log(f"smol honest {name} done")

    # tampered models: own generations (reference kernel), then eager replay of their tokens
    for name, q in (("nf4", nf4()), ("int8", BitsAndBytesConfig(load_in_8bit=True))):
        out = OUT / f"smol_{name}.npz"
        tseq_path = OUT / f"smol_{name}_sequences.json"
        vmodel = load_lm(SMOL_ID, rev, q=q)
        meta.setdefault("quantisation", {})[name] = quantised_layers(vmodel)
        vrec = TopKRecorder(interval=iv, k=K); vrec.attach(decoder_layers(vmodel))
        vtap = LastHiddenTap(); vtap.attach(vmodel)
        tseqs = {int(k): v for k, v in read_json(tseq_path).items()}
        if not out.exists() or SMOKE:
            pv = np.zeros((len(pids), C, L, K), np.float32); pi = np.zeros_like(pv, np.int32)
            for j, pid in enumerate(pids):
                g = generate_one(vmodel, tok, vrec, vtap, encode_prompt(tok, P[pid]["text"]).to(input_device(vmodel)), scfg)
                pv[j], pi[j] = g["values"], g["indices"]
                tseqs[pid] = g["tokens"]
                if pid % 20 == 0:
                    log(f"smol {name} [{pid:03d}]")
            rv, ri = ref_topk(tseqs, pids)
            save_npz(out, prov_values=pv, prov_indices=pi, ref_values=rv, ref_indices=ri, prompt_ids=np.array(pids, np.int32))
            if not SMOKE:
                tseq_path.write_text(json.dumps({str(k): v for k, v in tseqs.items()}) + "\n", encoding="utf-8")
        aout = OUT / f"smol_{name}_eager.npz"
        if not aout.exists() or SMOKE:
            _set_attn(vmodel, "eager")
            pv = np.zeros((len(att_pids), C, L, K), np.float32); pi = np.zeros_like(pv, np.int32)
            for j, pid in enumerate(att_pids):
                row = replay_one(vmodel, tok, vrec, vtap, encode_prompt(tok, P[pid]["text"]).to(input_device(vmodel)), tseqs[pid], scfg)
                pv[j], pi[j] = row["values"], row["indices"]
            rv, ri = ref_topk(tseqs, att_pids)
            save_npz(aout, prov_values=pv, prov_indices=pi, ref_values=rv, ref_indices=ri, prompt_ids=np.array(att_pids, np.int32))
            log(f"smol {name} eager attack done")
        vrec.detach(); vtap.detach()
        del vmodel, vrec, vtap
        release()
    write_json(meta_path, meta)
    brec.detach(); btap.detach()
    del base
    release()


class _nullctx:
    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


def main() -> int:
    global SMOKE
    stage = sys.argv[1] if len(sys.argv) > 1 else "all"
    SMOKE = stage == "smoke"
    cfg = load_config()
    OUT.mkdir(parents=True, exist_ok=True)
    order = ["qwen_new", "overhead", "throughput", "smol"]
    run = order if stage in ("all", "smoke") else [stage]
    for s in run:
        log(f"=== stage {s}{' (smoke)' if SMOKE else ''}")
        {"qwen_new": stage_qwen_new, "overhead": stage_overhead, "throughput": stage_throughput, "smol": stage_smol}[s](cfg)
    log("review_gpu finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
