"""Send one slash to the local Anvil deployment. Dev keys only. Not a public chain."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CAST = Path.home() / ".foundry" / "bin" / "cast.exe"
RPC = "http://127.0.0.1:8545"

# Anvil default accounts. Localhost only.
PROVIDER_KEY = "0x5de4111afa1a4b94908f83103eb1f1706367c2e68ca870fc3fb9a804cdab365a"
PROVIDER = "0x3C44CdDdB6a900fa2b585dd299e03d12FA4293BC"
CHALLENGER_KEY = "0x7c852118294e51e653712a81e05800f419141751be58f605c371e15141b007a6"
VALIDATOR_KEY = "0x59c6995e998f97a5a0044966f0945389dc9e86dae88c7a8412f4603b6b78690d"


def _cast(*args: str, stdin: bytes | None = None) -> str:
    cmd = [str(CAST), *args]
    if args[0] != "keccak":
        cmd.extend(["--rpc-url", RPC])
    proc = subprocess.run(cmd, input=stdin, check=True, capture_output=True, text=stdin is None)
    out = proc.stdout.decode() if isinstance(proc.stdout, bytes) else proc.stdout
    return out.strip()


def _send(*args: str) -> str:
    raw = _cast("send", "--json", *args)
    return json.loads(raw)["transactionHash"]


def _classify() -> tuple[int, dict]:
    from validator.validator import live_features

    decision = json.loads((ROOT / "results" / "live_decision.json").read_text(encoding="utf-8"))
    if not decision["flagged"]:
        raise RuntimeError("proof was not flagged; refusing to slash")
    with np.load(ROOT / "results" / "live_topk.npz") as z:
        raw = live_features(z["baseline_values"], z["baseline_indices"], z["provider_values"], z["provider_indices"])
    payload = joblib.load(ROOT / "artifacts" / "classifier.pkl")
    mu = np.asarray(payload["zscore_mean"], dtype=np.float64)
    sd = np.asarray(payload["zscore_std"], dtype=np.float64)
    pred = int(payload["model"].predict(((raw - mu) / sd).reshape(1, -1))[0])
    names = payload["class_names"]
    decision["predicted_class"] = pred
    decision["predicted_name"] = names[pred]
    return pred, decision


def main() -> int:
    deploy = json.loads((ROOT / "contracts" / "deployment.json").read_text(encoding="utf-8"))
    pow_addr = deploy["address"]
    token = deploy["stakeToken"]
    stake = deploy["requiredStake"]
    bond = deploy["challengeBond"]
    pred, decision = _classify()
    blob = bytes.fromhex("".join(decision["proofs_hex"]))
    proof_hash = _cast("keccak", stdin=blob)

    _send("--private-key", PROVIDER_KEY, token, "mint(address,uint256)", PROVIDER, stake)
    _send("--private-key", PROVIDER_KEY, token, "approve(address,uint256)", pow_addr, stake)
    _send("--private-key", PROVIDER_KEY, pow_addr, "registerProvider(bytes32)", proof_hash)
    provider_id = _cast("call", pow_addr, "providerIdOf(address)(uint256)", PROVIDER)
    _send(
        "--private-key",
        PROVIDER_KEY,
        pow_addr,
        "submitProof(uint256,bytes32,bytes32,string)",
        provider_id,
        proof_hash,
        proof_hash,
        "local://live_decision.json",
    )
    proof_id = _cast("call", pow_addr, "nextProofId()(uint256)")
    # nextProofId is the next unused id; the submitted proof is that minus one.
    submitted = str(int(proof_id) - 1)
    _send("--private-key", CHALLENGER_KEY, "--value", bond, pow_addr, "challenge(uint256,uint256)", provider_id, submitted)
    challenge_id = _cast("call", pow_addr, "nextChallengeId()(uint256)")
    cid = str(int(challenge_id) - 1)
    tx = _send("--private-key", VALIDATOR_KEY, pow_addr, "resolveChallenge(uint256,bool,uint8)", cid, "true", str(pred))
    stake_after = _cast("call", pow_addr, "providerStake(uint256)(uint256)", provider_id)
    history = _cast("call", pow_addr, "getTamperHistory(uint256)(uint8[])", provider_id)
    decision["provider_id"] = int(provider_id)
    decision["proof_id"] = int(submitted)
    decision["challenge_id"] = int(cid)
    decision["slash_tx"] = tx
    decision["stake_after"] = stake_after
    decision["tamper_history"] = history
    decision["proof_hash"] = proof_hash
    (ROOT / "results" / "live_slash.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"slash_tx": tx, "class": pred, "name": decision["predicted_name"], "stake_after": stake_after}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
