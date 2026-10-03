"""Phase B: teacher-forced prefill through variant and baseline at prompt_len + 32k."""

from __future__ import annotations

from inference.hooks import TopKRecorder, prefill_positions


def run_prefill_prompt(
    recorder: TopKRecorder,
    model,
    input_ids,
    prompt_len: int,
    tokens_per_generation: int,
    interval: int,
    decode_token_ids_at_positions,
) -> None:
    positions = prefill_positions(prompt_len, tokens_per_generation, interval)
    if decode_token_ids_at_positions is not None:
        got = input_ids[0, positions].tolist()
        if list(decode_token_ids_at_positions) != got:
            raise AssertionError(
                f"token IDs at positions {positions} mismatch decode vs prefill: "
                f"{decode_token_ids_at_positions} vs {got}"
            )
    recorder.reset("prefill", positions=positions)
    model(input_ids)
    n_layers = len(model.model.layers)
    recorder.assert_prefill_complete(n_layers)


def run_matched_and_shared(cfg: dict) -> None:
    raise NotImplementedError("TASK-04 Phase B runs on the A10G")
