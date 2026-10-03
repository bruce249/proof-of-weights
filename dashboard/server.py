"""Serve the slash dashboard and poll the local Anvil contract."""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
RPC = "http://127.0.0.1:8545"
CONTRACT = "0xe7f1725E7734CE288F8367e1Bb143E90bb3F0512"
NAMES = {0: "clean", 1: "quant_8bit", 2: "quant_4bit", 3: "lora_merged", 4: "ft_drift", 255: "TAMPERED"}


def _rpc(method: str, params: list) -> object:
    body = json.dumps({"jsonrpc": "2.0", "id": 1, "method": method, "params": params}).encode()
    req = Request(RPC, data=body, headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=3) as resp:
        payload = json.loads(resp.read().decode())
    if "error" in payload:
        raise RuntimeError(payload["error"])
    return payload["result"]


def _call(selector: str, arg: int | None = None) -> bytes:
    data = selector if arg is None else selector + f"{arg:064x}"
    raw = _rpc("eth_call", [{"to": CONTRACT, "data": data}, "latest"])
    return bytes.fromhex(raw[2:])


def _word(blob: bytes, index: int) -> int:
    return int.from_bytes(blob[index * 32 : (index + 1) * 32], "big")


def _uint8_array(blob: bytes) -> list[int]:
    if len(blob) < 64:
        return []
    offset = _word(blob, 0) // 32
    length = _word(blob, offset)
    return [_word(blob, offset + 1 + i) for i in range(length)]


def state() -> dict:
    stake = _word(_call("0xd3ddf59c", 1), 0)
    required = _word(_call("0x19cb355d"), 0)
    history = _uint8_array(_call("0x33ffff1d", 1))
    proof = _call("0x9ddaf5aa", 1)
    slashed = bool(_word(proof, 7))
    challenged = bool(_word(proof, 6))
    log = json.loads((ROOT / "results" / "live_slash.json").read_text(encoding="utf-8"))
    klass = history[-1] if history else None
    return {
        "chain": "anvil",
        "rpc": RPC,
        "contract": CONTRACT,
        "provider_id": 1,
        "proof_id": int(log["proof_id"]),
        "required_stake": str(required),
        "stake": str(stake),
        "challenged": challenged,
        "slashed": slashed,
        "tamper_class": klass,
        "tamper_name": NAMES.get(klass, "unknown"),
        "slash_tx": log["slash_tx"],
        "exp_mismatches_max": log["proof_stats"]["exp_mismatches_max"],
        "threshold_exp": log["thresholds"]["T_exp"],
        "variant": log["variant"],
    }


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        if self.path.startswith("/api/state"):
            try:
                body = json.dumps(state()).encode()
                status = 200
            except Exception as exc:
                body = json.dumps({"error": str(exc)}).encode()
                status = 503
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        page = (Path(__file__).parent / "index.html").read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(page)))
        self.end_headers()
        self.wfile.write(page)

    def log_message(self, fmt: str, *args) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8765), Handler)
    print("dashboard http://127.0.0.1:8765", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
