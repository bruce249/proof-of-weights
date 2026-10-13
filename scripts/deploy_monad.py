"""Deploy ProofOfWeights to Monad testnet and replay the recorded slash there.

Reads the funded key from MONAD_PRIVATE_KEY. That wallet deploys and acts as validator.
Provider and challenger wallets are generated once, stored in .env.monad (gitignored),
and funded with a little MON from the deployer.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from eth_account import Account
from web3 import Web3

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

RPC = os.environ.get("MONAD_RPC", "https://testnet-rpc.monad.xyz")
CHAIN_ID = 10143
EXPLORER = "https://testnet.monadvision.com"
REQUIRED_STAKE = 100 * 10**18  # mock STK tokens, same as the Anvil run
CHALLENGE_BOND = 10**16  # 0.01 MON; faucet amounts are small
FUND = Web3.to_wei(0.5, "ether")  # Monad bills the gas limit; 0.05 ran out
KEYS = ROOT / ".env.monad"


def _artifact(name: str) -> tuple[list, str]:
    art = json.loads((ROOT / "contracts" / "out" / f"{name}.sol" / f"{name}.json").read_text(encoding="utf-8"))
    return art["abi"], art["bytecode"]["object"]


def _role_keys() -> dict[str, str]:
    if KEYS.exists():
        pairs = (line.split("=", 1) for line in KEYS.read_text().splitlines() if "=" in line)
        return {k.strip(): v.strip() for k, v in pairs}
    keys = {"PROVIDER_KEY": Account.create().key.hex(), "CHALLENGER_KEY": Account.create().key.hex()}
    KEYS.write_text("".join(f"{k}={v}\n" for k, v in keys.items()))
    return keys


class Chain:
    def __init__(self) -> None:
        self.w3 = Web3(Web3.HTTPProvider(RPC, request_kwargs={"timeout": 60}))
        if self.w3.eth.chain_id != CHAIN_ID:
            raise RuntimeError(f"unexpected chain id {self.w3.eth.chain_id}")

    def send(self, acct, tx: dict) -> dict:
        tx = {
            "from": acct.address,
            "chainId": CHAIN_ID,
            "nonce": self.w3.eth.get_transaction_count(acct.address, "pending"),
            "gasPrice": self.w3.eth.gas_price,
            **tx,
        }
        tx.setdefault("gas", int(self.w3.eth.estimate_gas(tx) * 1.2))
        signed = acct.sign_transaction(tx)
        h = self.w3.eth.send_raw_transaction(signed.raw_transaction)
        rcpt = self.w3.eth.wait_for_transaction_receipt(h, timeout=180)
        if rcpt.status != 1:
            raise RuntimeError(f"tx reverted: {h.hex()}")
        print(f"  ok {('0x' + h.hex().removeprefix('0x'))}", flush=True)
        return rcpt

    def call_tx(self, acct, fn, value: int = 0) -> dict:
        gas = int(fn.estimate_gas({"from": acct.address, "value": value}) * 1.2)
        tx = fn.build_transaction({"from": acct.address, "value": value, "gas": gas})
        return self.send(acct, {"to": tx["to"], "data": tx["data"], "value": value, "gas": gas})

    def deploy(self, acct, name: str, *args):
        abi, bytecode = _artifact(name)
        factory = self.w3.eth.contract(abi=abi, bytecode=bytecode)
        data = factory.constructor(*args).build_transaction({"from": acct.address, "gas": 0})["data"]
        rcpt = self.send(acct, {"data": data})
        return self.w3.eth.contract(address=rcpt.contractAddress, abi=abi), rcpt


def main() -> int:
    key = os.environ.get("MONAD_PRIVATE_KEY")
    deployer_file = ROOT / ".env.monad_deployer"
    if not key and deployer_file.exists():
        key = deployer_file.read_text().split("=", 1)[1].strip()
    if not key:
        print("set MONAD_PRIVATE_KEY to a funded Monad testnet key", file=sys.stderr)
        return 1
    from slash_anvil import _classify

    pred, decision = _classify()
    blob = bytes.fromhex("".join(decision["proofs_hex"]))
    proof_hash = Web3.keccak(blob)

    chain = Chain()
    w3 = chain.w3
    validator = Account.from_key(key)
    roles = _role_keys()
    provider = Account.from_key(roles["PROVIDER_KEY"])
    challenger = Account.from_key(roles["CHALLENGER_KEY"])
    bal = w3.eth.get_balance(validator.address)
    print(f"deployer/validator {validator.address}  balance {w3.from_wei(bal, 'ether')} MON")

    print("funding provider and challenger")
    for who in (provider, challenger):
        short = FUND - w3.eth.get_balance(who.address)
        if short > 0:
            chain.send(validator, {"to": who.address, "value": short, "gas": 21000})

    print("deploying MockERC20")
    token, _ = chain.deploy(validator, "MockERC20")
    print("deploying ProofOfWeights")
    pow_c, deploy_rcpt = chain.deploy(validator, "ProofOfWeights", token.address, validator.address, REQUIRED_STAKE, CHALLENGE_BOND)

    print("provider stakes and submits proof")
    chain.call_tx(provider, token.functions.mint(provider.address, REQUIRED_STAKE))
    chain.call_tx(provider, token.functions.approve(pow_c.address, REQUIRED_STAKE))
    chain.call_tx(provider, pow_c.functions.registerProvider(proof_hash))
    provider_id = pow_c.functions.providerIdOf(provider.address).call()
    chain.call_tx(provider, pow_c.functions.submitProof(provider_id, proof_hash, proof_hash, "github:results/live_decision.json"))
    proof_id = pow_c.functions.nextProofId().call() - 1

    print("challenger posts bond")
    chain.call_tx(challenger, pow_c.functions.challenge(provider_id, proof_id), value=CHALLENGE_BOND)
    challenge_id = pow_c.functions.nextChallengeId().call() - 1

    print(f"validator resolves: upheld, class {pred} ({decision['predicted_name']})")
    rcpt = chain.call_tx(validator, pow_c.functions.resolveChallenge(challenge_id, True, pred))
    slash_tx = "0x" + rcpt.transactionHash.hex().removeprefix("0x")

    stake_after = pow_c.functions.providerStake(provider_id).call()
    history = pow_c.functions.getTamperHistory(provider_id).call()

    deployment = {
        "network": "monad-testnet",
        "chainId": CHAIN_ID,
        "rpc": RPC,
        "explorer": EXPLORER,
        "address": pow_c.address,
        "abi": pow_c.abi,
        "stakeToken": token.address,
        "validator": validator.address,
        "provider": provider.address,
        "challenger": challenger.address,
        "deployTx": "0x" + deploy_rcpt.transactionHash.hex().removeprefix("0x"),
        "requiredStake": str(REQUIRED_STAKE),
        "challengeBond": str(CHALLENGE_BOND),
        "note": "Monad testnet. Stake token is a mock ERC20. One trusted validator.",
    }
    (ROOT / "contracts" / "deployment.monad.json").write_text(json.dumps(deployment, indent=2) + "\n", encoding="utf-8")
    decision |= {
        "chain": "monad-testnet",
        "contract": pow_c.address,
        "provider_id": provider_id,
        "proof_id": proof_id,
        "challenge_id": challenge_id,
        "slash_tx": slash_tx,
        "stake_after": str(stake_after),
        "tamper_history": str(history),
        "proof_hash": "0x" + proof_hash.hex().removeprefix("0x"),
    }
    (ROOT / "results" / "live_slash_monad.json").write_text(json.dumps(decision, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"contract": pow_c.address, "slash_tx": slash_tx, "explorer": f"{EXPLORER}/tx/{slash_tx}",
                      "class": pred, "stake_after": str(stake_after)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
