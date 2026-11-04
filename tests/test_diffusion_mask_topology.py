"""Independent literal and multi-layer oracles for text denoising visibility."""

import torch
from src.data.diffusion_batch import (
    DenoiseSample,
    TextTokenContract,
    TextTurn,
    compile_text_batch,
)


def toy_contract():
    return TextTokenContract(
        tokenizer_digest="synthetic-topology-v1",
        vocab_size=32,
        mask_token_id=31,
        pad_token_id=0,
        boa_token_id=28,
        eoa_token_id=29,
        think_open_token_id=26,
        think_close_token_id=27,
        tokenizer_special_ids=(0, 26, 28),
    )


def two_block_sample(sample_id):
    # P, two noisy blocks, then their natural clean copies: 1 + 4 + 4 positions.
    return DenoiseSample(
        sample_id,
        (
            TextTurn((1,), kind="context"),
            TextTurn((5, 6, 7, 29), kind="answer"),
        ),
    )


def literal_two_block_visibility():
    # Query rows/key columns: P, N1a, N1b, N2a, N2b, C1a, C1b, C2a, C2b.
    return torch.tensor(
        [
            [1, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 1, 1, 1, 1, 0, 0],
            [1, 0, 0, 1, 1, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 1, 1, 0, 0],
            [1, 0, 0, 0, 0, 1, 1, 1, 1],
            [1, 0, 0, 0, 0, 1, 1, 1, 1],
        ],
        dtype=torch.bool,
    )


def test_two_packed_samples_match_literal_visibility_and_positions():
    batch = compile_text_batch(
        (two_block_sample(10), two_block_sample(20)),
        contract=toy_contract(),
        block_size=2,
        seed=5,
        pad_to_length=20,
    )
    expected = torch.zeros((20, 20), dtype=torch.bool)
    expected[:18, :18] = torch.block_diag(
        literal_two_block_visibility(), literal_two_block_visibility()
    )
    torch.testing.assert_close(batch.attention_allowed(), expected)
    torch.testing.assert_close(
        batch.position_ids[:18], torch.tensor([0, 1, 2, 3, 4, 1, 2, 3, 4] * 2)
    )
    assert batch.valid_mask.tolist() == [True] * 18 + [False] * 2
    assert batch.sample_ids.tolist() == [10] * 9 + [20] * 9 + [-1, -1]


def test_reachability_cannot_leak_current_or_future_clean_targets():
    batch = compile_text_batch(
        (two_block_sample(0),), contract=toy_contract(), block_size=2, seed=8
    )
    reachable = batch.attention_allowed().clone()
    # Boolean transitive closure independently checks arbitrarily many layers.
    for intermediate in range(len(batch.input_ids)):
        reachable |= reachable[:, intermediate, None] & reachable[intermediate, None, :]
    assert not reachable[1:3, 3:9].any()  # N1 cannot reach N2 or either clean block.
    assert not reachable[3:5, 7:9].any()  # N2 cannot reach its own clean target.
    assert not reachable[5:9, 1:5].any()  # Clean memory never reads noisy targets.
    assert reachable[3:5, 5:7].all()  # Positive control: earlier clean blocks are visible.


def test_three_layer_aggregation_preserves_no_leakage_and_allowed_dependencies():
    batch = compile_text_batch(
        (two_block_sample(0), two_block_sample(1)),
        contract=toy_contract(),
        block_size=2,
        seed=9,
    )
    weights = batch.attention_allowed().to(torch.float64)
    weights /= weights.sum(dim=-1, keepdim=True).clamp_min(1)

    def three_layers(values):
        for _ in range(3):
            values = weights @ values
        return values

    baseline = torch.arange(18, dtype=torch.float64)
    original = three_layers(baseline)
    future = baseline.clone()
    future[7:9] += 1000
    torch.testing.assert_close(three_layers(future)[:5], original[:5])
    earlier = baseline.clone()
    earlier[5:7] += 1000
    altered = three_layers(earlier)
    torch.testing.assert_close(altered[1:3], original[1:3])
    assert torch.all(altered[3:5] > original[3:5])
    other_sample = baseline.clone()
    other_sample[9:] -= 1000
    torch.testing.assert_close(three_layers(other_sample)[:9], original[:9])


def test_boa_and_think_specials_do_not_create_indirect_clean_target_paths():
    batch = compile_text_batch(
        (
            DenoiseSample(
                0,
                (
                    TextTurn((1,), kind="context"),
                    TextTurn((26, 5, 6, 7, 29), kind="answer", add_boa=True),
                ),
            ),
        ),
        contract=toy_contract(),
        block_size=2,
        seed=12,
    )
    allowed = batch.attention_allowed()
    noisy_specials = (~batch.is_clean) & (batch.type_ids == 1) & batch.valid_mask
    same_turn_text = (batch.turn_ids == 1) & (batch.type_ids == 3)
    assert noisy_specials.sum() == 2
    assert not allowed[noisy_specials][:, same_turn_text].any()
    reachable = allowed.clone()
    for intermediate in range(len(batch.input_ids)):
        reachable |= reachable[:, intermediate, None] & reachable[intermediate, None, :]
    first_noisy = (~batch.is_clean) & (batch.type_ids == 3) & (batch.block_ids == 0)
    same_turn_clean = batch.is_clean & (batch.turn_ids == 1) & (batch.type_ids == 3)
    assert not reachable[first_noisy][:, same_turn_clean].any()
