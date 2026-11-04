"""Hand-computable CPU tests for the pretokenized compiler contract."""

from __future__ import annotations

import math
import random
from dataclasses import replace
from unittest.mock import patch

import pytest
import torch
from src.data.diffusion_batch import (
    SCHEDULE_VERSION,
    SCHEMA_VERSION,
    DenoiseSample,
    TextTokenContract,
    TextTurn,
    compile_text_batch,
)


@pytest.fixture
def contract():
    return TextTokenContract(
        tokenizer_digest="synthetic-fixture-v1",
        vocab_size=64,
        mask_token_id=0,
        pad_token_id=1,
        boa_token_id=2,
        eoa_token_id=3,
        think_open_token_id=4,
        think_close_token_id=5,
        tokenizer_special_ids=(1, 2, 3, 4, 5, 6),
    )


def sample(*turns, sample_id=7):
    return DenoiseSample(sample_id, turns)


class ScriptedRandom:
    """Ratio/branch uniforms and deterministic sample selection for arithmetic."""

    def __init__(self, values):
        self.values = iter(values)
        self.calls = []

    def random(self):
        value = next(self.values)
        self.calls.append(("random", value))
        return value

    def uniform(self, low, high):
        assert (low, high) == (0, 1)
        value = next(self.values)
        self.calls.append(("uniform", value))
        return value

    def sample(self, candidates, count):
        self.calls.append(("sample", tuple(candidates), count))
        return candidates[-count:]

    def getstate(self):
        return ("scripted", len(self.calls))


def scripted_compile(contract, turns, values, *, block_size=8, **kwargs):
    rng = ScriptedRandom(values)
    with patch("src.data.diffusion_batch.random.Random", return_value=rng):
        batch = compile_text_batch(
            [sample(*turns)], contract=contract, block_size=block_size, **kwargs
        )
    return batch, rng


def test_pinned_default_ids_are_declared_without_loading_assets():
    contract = TextTokenContract("caller-declared")
    assert contract.vocab_size == 157184
    assert contract.mask_token_id == 156895
    assert contract.pad_token_id == 156892
    assert (contract.boa_token_id, contract.eoa_token_id) == (157165, 157166)
    assert (contract.think_open_token_id, contract.think_close_token_id) == (157175, 157176)
    assert contract.eoa_token_id not in contract.never_mask_ids
    assert contract.think_close_token_id not in contract.never_mask_ids


def test_source_registry_is_retained_when_caller_adds_tokenizer_specials(contract):
    real = TextTokenContract("caller-declared", tokenizer_special_ids=(42,))
    assert 157153 in real.never_mask_ids  # Source <system>, absent from caller list.
    assert 42 in real.never_mask_ids
    assert 157166 not in real.never_mask_ids
    assert 157176 not in real.never_mask_ids
    assert not any(token >= contract.vocab_size for token in contract.never_mask_ids)


@pytest.mark.parametrize("protect_pad", [False, True])
def test_valid_pad_corruption_is_controlled_by_caller_special_registry(contract, protect_pad):
    special_ids = (contract.pad_token_id,) if protect_pad else ()
    contract = replace(contract, tokenizer_special_ids=special_ids)
    batch, _ = scripted_compile(
        contract,
        [TextTurn((contract.pad_token_id,), "answer")],
        [0.1, 0.9, 0.9],
    )
    assert batch.valid_mask[0].item()
    assert batch.labels[0].item() == (-100 if protect_pad else contract.pad_token_id)
    assert batch.corruption_mask[0].item() is not protect_pad
    assert batch.labels[1].item() == contract.eoa_token_id
    assert (contract.pad_token_id in contract.never_mask_ids) is protect_pad


def test_declared_special_pad_equal_to_eoa_is_still_a_maskable_target(contract):
    contract = replace(
        contract,
        pad_token_id=contract.eoa_token_id,
        tokenizer_special_ids=(contract.eoa_token_id, contract.think_close_token_id),
    )
    batch, _ = scripted_compile(contract, [TextTurn((), "answer")], [0.1, 0.9, 0.9])
    assert contract.eoa_token_id not in contract.never_mask_ids
    assert contract.think_close_token_id not in contract.never_mask_ids
    assert batch.labels[0].item() == contract.eoa_token_id
    assert batch.corruption_mask[0].item()


@pytest.mark.parametrize("length", [0, 1, 4, 5, 31, 32, 33])
def test_answer_boundary_lengths_and_tail_weight(contract, length):
    batch = compile_text_batch(
        [sample(TextTurn((10,) * length, "answer", add_boa=True))], contract=contract
    )
    noisy, clean = batch.spans
    natural = length + 2  # BOA, supplied answer and natural EOA.
    padded_core = math.ceil((length + 1) / 32) * 32
    assert noisy.stop - noisy.start == 1 + padded_core
    assert clean.stop - clean.start == natural
    assert clean.natural_length == natural
    tail_start = noisy.start + natural
    tail_length = noisy.stop - tail_start
    if tail_length:
        assert batch.loss_weights[tail_start : noisy.stop].sum().item() == pytest.approx(1.0)
        assert batch.original_position_ids[tail_start : noisy.stop].tolist() == [-1] * tail_length
    assert batch.loss_weights[noisy.start : tail_start].tolist() == [1.0] * natural
    assert batch.input_ids[clean.stop - 1].item() == contract.eoa_token_id
    assert batch.labels[clean.start : clean.stop].tolist() == [-100] * natural


def test_natural_eoa_is_not_appended_twice(contract):
    batch = compile_text_batch(
        [sample(TextTurn((10, 3), "answer"))], contract=contract, block_size=4
    )
    assert batch.spans[0].natural_length == 2
    assert batch.input_ids[batch.spans[1].start :].tolist() == [10, 3]
    assert batch.original_to_merged[1].merged_positions == (1, 5)


def test_same_position_labels_match_exact_corrupted_positions(contract):
    batch, rng = scripted_compile(
        contract, [TextTurn((10, 11, 12, 13, 14, 15), "answer")], [0.029, 0.515], block_size=4
    )
    # Block zero: 3% full branch. Block one: cos(pi/4), floor(4*.707)=2.
    assert batch.input_ids[:8].tolist() == [0, 0, 0, 0, 14, 15, 0, 0]
    assert batch.labels[:8].tolist() == [10, 11, 12, 13, -100, -100, 3, 3]
    assert batch.supervision_mask[:8].tolist() == [True] * 4 + [False] * 2 + [True] * 2
    assert torch.equal(batch.corruption_mask, batch.supervision_mask)
    assert [draw.region for draw in batch.rng_trace] == ["full", "full"]
    assert rng.calls == [
        ("uniform", 0.029),
        ("sample", (0, 1, 2, 3), 4),
        ("uniform", 0.515),
        ("sample", (4, 5, 6, 7), 2),
    ]


@pytest.mark.parametrize("length", [1, 4])
def test_short_terminal_forced_branch_and_independent_tail_draw(contract, length):
    batch, rng = scripted_compile(
        contract, [TextTurn((10,) * length, "answer", add_boa=True)], [0.599, 0.999, 0.999]
    )
    natural = length + 1  # Answer and EOA; BOA is never a candidate.
    assert batch.labels[0].item() == -100
    assert batch.labels[1 : natural + 1].tolist() == [10] * length + [3]
    assert batch.supervision_mask[natural + 1 : 9].sum().item() == 1
    assert [draw.region for draw in batch.rng_trace] == ["short_answer", "natural", "tail"]
    assert batch.rng_trace[1].mask_ratio == 1.0
    assert rng.calls[1] == ("uniform", 0.999)  # Still consumed in forced branch.
    assert rng.calls[2] == ("sample", tuple(range(1, natural + 1)), natural)


def test_short_branch_at_sixty_percent_uses_ordinary_cosine(contract):
    batch, _ = scripted_compile(
        contract, [TextTurn((10, 11, 12, 13), "answer")], [0.6, 0.999, 0.999]
    )
    assert batch.labels[:5].tolist() == [-100, -100, -100, -100, 3]
    assert batch.supervision_mask[:5].sum().item() == 1


def test_length_five_does_not_consume_short_branch_draw(contract):
    batch, rng = scripted_compile(contract, [TextTurn((10,) * 5, "answer")], [0.999])
    assert [draw.region for draw in batch.rng_trace] == ["full"]
    assert rng.calls[0] == ("uniform", 0.999)
    assert batch.labels[:8].tolist() == [-100] * 7 + [3]


def test_three_percent_boundary_is_cosine_and_at_least_one_is_masked(contract):
    batch, _ = scripted_compile(
        contract, [TextTurn((10,) * 7, "answer")], [0.03, 1.0], block_size=4
    )
    assert batch.rng_trace[0].mask_ratio == pytest.approx(1.0)
    assert len(batch.rng_trace[0].selected_positions) == 4
    assert len(batch.rng_trace[1].selected_positions) == 1


def test_never_mask_specials_and_candidate_free_blocks_consume_no_ratio(contract):
    batch, rng = scripted_compile(
        contract, [TextTurn((6,), "answer", add_boa=True)], [0.1, 0.9], block_size=1
    )
    assert batch.labels[:3].tolist() == [-100, -100, 3]
    assert [call[0] for call in rng.calls] == ["random", "uniform", "sample"]
    assert batch.rng_trace[1].candidate_positions == (2,)


def test_think_tail_is_absorbing_close_without_eoa(contract):
    batch, _ = scripted_compile(
        contract, [TextTurn((4, 10), "think", add_boa=True)], [0.0], block_size=4
    )
    noisy, clean = batch.spans
    assert batch.input_ids[clean.start : clean.stop].tolist() == [2, 4, 10, 5]
    assert batch.labels[noisy.start : noisy.stop].tolist() == [-100, -100, 10, 5, 5, 5]
    assert 3 not in batch.labels.tolist()
    assert batch.loss_weights[4:6].tolist() == [0.5, 0.5]
    assert batch.type_ids[:6].tolist() == [1, 1, 3, 3, 3, 3]
    assert batch.block_ids[:6].tolist() == [0, 0, 0, 0, 0, 0]


def test_unmarked_think_does_not_invent_a_natural_close(contract):
    batch = compile_text_batch(
        [sample(TextTurn((10, 11), "think"))], contract=contract, block_size=4
    )
    assert batch.spans[0].natural_length == 2
    assert batch.input_ids[batch.spans[1].start :].tolist() == [10, 11]
    assert batch.loss_weights[:4].tolist() == [1.0, 1.0, 0.5, 0.5]


def test_think_segment_markers_restart_text_block_offsets(contract):
    batch = compile_text_batch(
        [sample(TextTurn((10, 11, 12, 4, 13, 14), "answer"))],
        contract=contract,
        block_size=2,
    )
    assert batch.block_ids[:8].tolist() == [0, 0, 1, 0, 2, 2, 3, 3]
    assert batch.type_ids[:8].tolist() == [3, 3, 3, 1, 3, 3, 3, 3]


def test_chronological_first_stream_then_trimmed_clean_memory(contract):
    batch = compile_text_batch(
        [
            sample(
                TextTurn((10,), "context"),
                TextTurn((11,), "answer"),
                TextTurn((12, 13), "context"),
                TextTurn((14,), "think"),
            )
        ],
        contract=contract,
        block_size=4,
    )
    assert [(span.turn_id, span.stream) for span in batch.spans] == [
        (0, "context"),
        (1, "noisy"),
        (2, "context"),
        (3, "noisy"),
        (1, "clean"),
        (3, "clean"),
    ]
    assert batch.position_ids[:11].tolist() == list(range(11))
    assert batch.position_ids[11:].tolist() == [1, 2, 7]
    assert batch.input_ids[11:].tolist() == [11, 3, 14]


def test_packed_sample_positions_reset_and_identity_is_preserved(contract):
    batch = compile_text_batch(
        [
            sample(TextTurn((10,), "answer"), sample_id=31),
            sample(TextTurn((11,), "context"), sample_id=52),
        ],
        contract=contract,
        block_size=4,
    )
    assert batch.sample_ids.tolist() == [31] * 6 + [52]
    assert batch.position_ids.tolist() == [0, 1, 2, 3, 0, 1, 0]
    assert batch.turn_ids.tolist() == [0] * 7


def test_original_mapping_includes_only_supplied_tokens(contract):
    batch = compile_text_batch(
        [sample(TextTurn((10, 11), "context"), TextTurn((12, 3), "answer", add_boa=True))],
        contract=contract,
        block_size=4,
        pad_to_length=12,
    )
    assert [entry.merged_positions for entry in batch.original_to_merged] == [
        (0,),
        (1,),
        (3, 8),
        (4, 9),
    ]
    assert [
        (entry.sample_id, entry.turn_id, entry.token_index) for entry in batch.original_to_merged
    ] == [
        (7, 0, 0),
        (7, 0, 1),
        (7, 1, 0),
        (7, 1, 1),
    ]
    assert batch.original_position_ids.tolist() == [0, 1, -1, 2, 3, -1, -1, -1, 2, 3, -1, -1]


def test_pad_equals_eoa_still_has_explicit_invalid_and_valid_states(contract):
    contract = replace(contract, pad_token_id=contract.eoa_token_id)
    batch = compile_text_batch(
        [sample(TextTurn((10,), "answer"))], contract=contract, block_size=4, pad_to_length=8
    )
    assert batch.input_ids[-3:].tolist() == [3, 3, 3]
    assert batch.valid_mask[-3:].tolist() == [True, False, False]
    assert batch.loss_weights[-2:].tolist() == [0.0, 0.0]
    for field in (
        batch.sample_ids,
        batch.turn_ids,
        batch.type_ids,
        batch.block_ids,
        batch.position_ids,
    ):
        assert field[-2:].tolist() == [-1, -1]
    assert not batch.attention_allowed()[-2:, :].any()
    assert not batch.attention_allowed()[:, -2:].any()


def test_zero_target_context_keeps_finite_weights_and_rng_unchanged(contract):
    batch = compile_text_batch([sample(TextTurn((10, 11), "context"))], contract=contract)
    assert not batch.supervision_mask.any()
    assert batch.labels.tolist() == [-100, -100]
    assert batch.block_ids.tolist() == [-1, -1]
    assert batch.rng_trace == ()
    assert batch.rng_state_before == batch.rng_state_after


def test_deterministic_seed_and_global_rng_isolation(contract):
    inputs = [sample(TextTurn((10,) * 33, "answer"))]
    state = random.getstate()
    first = compile_text_batch(inputs, contract=contract, seed=123)
    second = compile_text_batch(inputs, contract=contract, seed=123)
    assert random.getstate() == state
    assert torch.equal(first.input_ids, second.input_ids)
    assert first.rng_trace == second.rng_trace
    assert first.rng_state_after == second.rng_state_after
    assert first.schema_version == SCHEMA_VERSION
    assert first.schedule_version == SCHEDULE_VERSION
    assert len(first.rng_python_version) == 3
    restored = random.Random()
    restored.setstate(first.rng_state_before)
    assert restored.getstate() == random.Random(123).getstate()


def test_expanded_cap_counts_noisy_padding_and_clean_copies_before_rng(contract):
    inputs = [sample(TextTurn((10,) * 700, "context"), TextTurn((11,) * 32, "answer"))]
    with patch("src.data.diffusion_batch.random.Random", side_effect=AssertionError("RNG touched")):
        with pytest.raises(ValueError, match="expanded batch length 797"):
            compile_text_batch(inputs, contract=contract)


def test_exact_768_position_boundary_and_padding_cap(contract):
    batch = compile_text_batch(
        [sample(TextTurn((10,) * 704, "context"), TextTurn((11,) * 31, "answer"))],
        contract=contract,
    )
    assert len(batch.input_ids) == 768
    with pytest.raises(ValueError, match="pad_to_length exceeds"):
        compile_text_batch(
            [sample(TextTurn((10,), "context"))], contract=contract, pad_to_length=769
        )
    with pytest.raises(ValueError, match="cannot truncate"):
        compile_text_batch([sample(TextTurn((10,), "answer"))], contract=contract, pad_to_length=1)


def test_tensor_shapes_dtypes_and_cpu(contract):
    batch = compile_text_batch([sample(TextTurn((10,), "answer"))], contract=contract)
    flags = {"valid_mask", "is_clean", "corruption_mask", "supervision_mask"}
    for name, value in vars(batch).items():
        if isinstance(value, torch.Tensor):
            assert value.shape == batch.input_ids.shape
            assert value.device.type == "cpu"
            assert value.dtype == (
                torch.bool
                if name in flags
                else torch.float32
                if name == "loss_weights"
                else torch.long
            )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"seed": True},
        {"seed": -1},
        {"seed": 1.5},
        {"block_size": 0},
        {"block_size": True},
        {"max_positions": 769},
        {"max_positions": 0},
        {"pad_to_length": False},
        {"pad_to_length": 0},
    ],
)
def test_invalid_compiler_options_fail(contract, kwargs):
    with pytest.raises(ValueError):
        compile_text_batch([sample(TextTurn((10,), "context"))], contract=contract, **kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"tokenizer_digest": ""},
        {"tokenizer_digest": " spaced "},
        {"vocab_size": True},
        {"pad_token_id": 64},
        {"mask_token_id": -1},
        {"boa_token_id": 3},
        {"tokenizer_special_ids": (True,)},
        {"tokenizer_special_ids": (64,)},
    ],
)
def test_invalid_token_contract_fails(contract, kwargs):
    with pytest.raises(ValueError):
        replace(contract, **kwargs)


@pytest.mark.parametrize(
    "tokens,kind,add_boa",
    [
        ((True,), "answer", False),
        ((-1,), "answer", False),
        ((), "context", False),
        ((), "think", False),
        ((10,), "image", False),
        ((10,), "context", True),
        ((10,), "answer", 1),
        ("text", "answer", False),
    ],
)
def test_invalid_turns_fail(tokens, kind, add_boa):
    with pytest.raises(ValueError):
        TextTurn(tokens, kind, add_boa)


def test_invalid_sample_and_input_ids_fail(contract):
    with pytest.raises(ValueError):
        DenoiseSample(True, (TextTurn((10,), "context"),))
    with pytest.raises(ValueError):
        DenoiseSample(1, ())
    with pytest.raises(ValueError, match="unique"):
        compile_text_batch([sample(TextTurn((10,), "context"))] * 2, contract=contract)
    with pytest.raises(ValueError, match="within vocab_size"):
        compile_text_batch([sample(TextTurn((64,), "answer"))], contract=contract)
    with pytest.raises(ValueError, match="mask token"):
        compile_text_batch([sample(TextTurn((0,), "answer"))], contract=contract)
    with pytest.raises(ValueError, match="nonempty"):
        compile_text_batch([], contract=contract)


def test_callers_lists_are_snapshotted_before_compilation(contract):
    tokens = [10]
    turn = TextTurn(tokens, "answer")
    turns = [turn]
    item = DenoiseSample(7, turns)
    tokens.append(11)
    turns.append(TextTurn((12,), "context"))
    batch = compile_text_batch([item], contract=contract, block_size=4)
    assert batch.spans[0].natural_length == 2
    assert len(batch.spans) == 2
