"""Compare independent synthetic fixtures from pinned upstream helper execution."""

import hashlib
import json
import sys
from pathlib import Path

import pytest
import torch
from src.data.diffusion_batch import (
    DenoiseSample,
    TextTokenContract,
    TextTurn,
    compile_text_batch,
)

FIXTURES = json.loads(
    (Path(__file__).parent / "fixtures" / "denoising_text_u2_v1.json").read_text()
)


@pytest.mark.parametrize("case", FIXTURES["cases"], ids=lambda case: case["name"])
def test_pinned_upstream_pretokenized_turn_and_packing(case):
    inputs = case["input"]
    batch = compile_text_batch(
        (
            DenoiseSample(
                17,
                (
                    TextTurn(tuple(inputs["prompt_ids_for_pack"]), kind="context"),
                    TextTurn(
                        tuple(inputs["token_ids"]),
                        kind="answer" if inputs["add_eoa"] else "think",
                        add_boa=inputs["add_boa"],
                    ),
                ),
            ),
        ),
        contract=TextTokenContract(
            tokenizer_digest="synthetic-upstream-fixture-v1",
            tokenizer_special_ids=tuple(inputs["caller_all_special_ids"]),
        ),
        block_size=inputs["block_size"],
        seed=inputs["seed"],
    )
    turn = case["expected_turn"]
    prompt_length = len(inputs["prompt_ids_for_pack"])
    noisy_stop = prompt_length + len(turn["noisy_ids"])
    noisy = slice(prompt_length, noisy_stop)
    clean = slice(noisy_stop, None)
    assert batch.input_ids[noisy].tolist() == turn["noisy_ids"]
    assert batch.input_ids[clean].tolist() == turn["clean_ids"]
    assert batch.labels[noisy].tolist() == turn["labels"]
    assert batch.type_ids[noisy].tolist() == turn["type_ids"]
    assert batch.block_ids[noisy].tolist() == turn["block_ids"]
    torch.testing.assert_close(batch.loss_weights[noisy], torch.tensor(turn["loss_weights"]))
    assert batch.labels[:prompt_length].eq(-100).all()
    assert batch.labels[clean].eq(-100).all()
    assert torch.equal(batch.supervision_mask, batch.labels.ne(-100))
    assert torch.equal(batch.corruption_mask, batch.supervision_mask)

    packed = case["expected_pack"]
    for source_name, batch_name in (
        ("position_ids", "position_ids"),
        ("turn_ids", "turn_ids"),
        ("is_clean", "is_clean"),
        ("type_position_ids", "type_ids"),
        ("attention_mask", "valid_mask"),
    ):
        assert getattr(batch, batch_name).tolist() == packed[source_name]
    assert len(batch.input_ids) == packed["all_len"]
    # The fixture passes a manually constructed prompt with block ID 0 into
    # _pack_turns. Real _make_user_turn uses -1 (pinned dataset lines 438-439).
    # Preserve the source fixture and compare only its target block metadata.
    assert batch.block_ids[:prompt_length].eq(-1).all()
    assert batch.block_ids[prompt_length:].tolist() == packed["block_ids"][prompt_length:]
    # Python does not promise random.sample sequences across versions. Compare
    # the complete state only in the independently recorded source environment.
    if ".".join(map(str, sys.version_info[:3])) == FIXTURES["provenance"]["environment"]["python"]:
        digest = hashlib.sha256(repr(batch.rng_state_after).encode("utf-8")).hexdigest()
        assert digest == case["local_rng_final_state_sha256"]
