"""Per-layer top-k recorder. Never use output_hidden_states=True."""

from __future__ import annotations

from typing import Iterable, List, Optional, Sequence, Tuple

try:
    import torch
    import torch.nn as nn
except ImportError:  # laptop scaffold; torch is on the GPU AMI
    torch = None  # type: ignore
    nn = None  # type: ignore


class TopKRecorder:
    def __init__(self, interval: int = 32, k: int = 128):
        self.interval = interval
        self.k = k
        self.mode: Optional[str] = None
        self.step = 0
        self.buf: list = []
        self.positions: Optional[Sequence[int]] = None
        self._handles: list = []

    def reset(self, mode: str, positions: Optional[Sequence[int]] = None) -> None:
        assert mode in ("decode", "prefill")
        if mode == "prefill":
            assert positions is not None
        self.mode = mode
        self.positions = positions
        self.step = 0
        self.buf = []

    def _record(self, vec, ckpt_idx: int, layer_idx: int) -> None:
        assert torch is not None
        vec = vec.float()
        _, idxs = torch.topk(vec.abs(), k=self.k, dim=-1)
        self.buf.append((ckpt_idx, layer_idx, idxs.cpu(), vec.gather(-1, idxs).cpu()))

    def make_hook(self, layer_idx: int):
        def hook(module, inputs, output):
            if self.mode is None:
                return
            h = output[0] if isinstance(output, tuple) else output
            if self.mode == "decode":
                if self.step % self.interval != 0:
                    return
                self._record(h[:, -1, :], self.step // self.interval, layer_idx)
            else:
                assert self.positions is not None
                for ckpt_idx, pos in enumerate(self.positions):
                    self._record(h[:, pos, :], ckpt_idx, layer_idx)

        return hook

    def attach(self, layers: Iterable) -> None:
        self.detach()
        for i, layer in enumerate(layers):
            self._handles.append(layer.register_forward_hook(self.make_hook(i)))

    def detach(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def checkpoints_per_layer(self, n_layers: int) -> List[int]:
        counts = [0] * n_layers
        for _ckpt, layer_idx, _idx, _val in self.buf:
            counts[layer_idx] += 1
        return counts

    def assert_prefill_complete(self, n_layers: int) -> None:
        if self.mode != "prefill" or self.positions is None:
            raise AssertionError("assert_prefill_complete requires mode=prefill")
        expected = len(self.positions)
        counts = self.checkpoints_per_layer(n_layers)
        if any(c != expected for c in counts):
            raise AssertionError(
                f"Phase B checkpoint counts {counts} != {expected} per layer. "
                "Hook likely recorded last-token only."
            )


def decoder_layers(model):
    """Qwen2 / Llama: model.model.layers. Verify with print(model) on first load."""
    return model.model.layers


def prefill_positions(prompt_len: int, tokens_per_generation: int, interval: int) -> List[int]:
    n_checkpoints = tokens_per_generation // interval
    return [prompt_len + interval * k for k in range(n_checkpoints)]


def stack_buf(buf, n_checkpoints: int, n_layers: int, k: int, batch_index: int = 0):
    """Turn recorder.buf into (values float32, indices int32) [C, L, k]."""
    assert torch is not None
    values = torch.zeros(n_checkpoints, n_layers, k, dtype=torch.float32)
    indices = torch.zeros(n_checkpoints, n_layers, k, dtype=torch.int32)
    for ckpt_idx, layer_idx, idxs, vals in buf:
        if vals.dim() == 1:
            values[ckpt_idx, layer_idx] = vals
            indices[ckpt_idx, layer_idx] = idxs.to(torch.int32)
        else:
            values[ckpt_idx, layer_idx] = vals[batch_index]
            indices[ckpt_idx, layer_idx] = idxs[batch_index].to(torch.int32)
    return values, indices
