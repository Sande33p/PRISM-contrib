"""Compiler-to-objective checks with independent loss and gradient arithmetic."""

import math

import torch
from src.data.diffusion_batch import DenoiseSample, TextTokenContract, TextTurn, compile_text_batch
from src.objectives.masked_categorical import masked_categorical_sums, reduce_masked_categorical


def contract():
    return TextTokenContract(
        tokenizer_digest="synthetic-objective-integration-v1",
        vocab_size=32,
        mask_token_id=31,
        pad_token_id=0,
        boa_token_id=28,
        eoa_token_id=29,
        think_open_token_id=26,
        think_close_token_id=27,
        tokenizer_special_ids=(0, 26, 28),
    )


def test_compiler_natural_and_tail_weights_reach_same_position_ce():
    batch = compile_text_batch(
        (DenoiseSample(0, (TextTurn((0, 2), kind="context"), TextTurn((5,), kind="answer"))),),
        contract=contract(),
        block_size=8,
        pad_to_length=16,
        seed=1,
    )
    logits = torch.zeros((16, 32), dtype=torch.float64, requires_grad=True)
    sums = masked_categorical_sums(
        logits, batch.labels, batch.loss_weights, valid=batch.valid_mask, text_coefficient=2.5
    )
    # The answer's two natural positions precede six artificial tail positions.
    # The seed forces both natural positions; each selected tail has weight 1/6.
    assert batch.labels[2:4].tolist() == [5, 29]
    expected_count = 2 + int(batch.supervision_mask[4:10].sum()) / 6
    torch.testing.assert_close(
        sums.effective_weight, torch.tensor(expected_count, dtype=torch.float64)
    )
    result = reduce_masked_categorical([[sums]], policy="prism_global_weighted_masked_ce_v1")
    torch.testing.assert_close(result.loss, torch.tensor(2.5 * math.log(32), dtype=torch.float64))
    result.loss.backward()
    assert logits.grad[~batch.supervision_mask].eq(0).all()
    assert logits.grad[batch.supervision_mask].abs().sum() > 0
    assert batch.valid_mask[0]  # A real PAD-valued prompt token remains valid.
    assert not batch.valid_mask[12:].any()


def test_compiled_pack_repartition_preserves_global_value_and_gradients():
    batch = compile_text_batch(
        (
            DenoiseSample(10, (TextTurn((1,), kind="context"), TextTurn((4,), kind="answer"))),
            DenoiseSample(
                20, (TextTurn((2,), kind="context"), TextTurn((5, 6, 7, 8, 9), kind="answer"))
            ),
        ),
        contract=contract(),
        block_size=4,
        seed=5,
        pad_to_length=32,
    )
    scores = torch.linspace(-1, 1, 32 * 32, dtype=torch.float64).reshape(32, 32).requires_grad_()

    def term(selection):
        return masked_categorical_sums(
            scores[selection],
            batch.labels[selection],
            batch.loss_weights[selection],
            valid=batch.valid_mask[selection],
        )

    packed = reduce_masked_categorical(
        [[term(slice(None))]], policy="prism_global_weighted_masked_ce_v1"
    )
    regrouped = reduce_masked_categorical(
        [
            [term(batch.sample_ids.eq(20)), term(~batch.valid_mask)],
            [],
            [term(batch.sample_ids.eq(10))],
        ],
        policy="prism_global_weighted_masked_ce_v1",
    )
    torch.testing.assert_close(regrouped.loss, packed.loss)
    expected_grad = torch.autograd.grad(packed.loss, scores, retain_graph=True)[0]
    averaged_local_grad = (
        sum(
            torch.autograd.grad(loss, scores, retain_graph=True)[0]
            for loss in regrouped.per_rank_backward
        )
        / 3
    )
    torch.testing.assert_close(averaged_local_grad, expected_grad)
    assert expected_grad[~batch.supervision_mask].eq(0).all()


def test_context_only_compiler_batch_skips_supervision_with_connected_zero():
    batch = compile_text_batch(
        (DenoiseSample(0, (TextTurn((0, 1), kind="context"),)),),
        contract=contract(),
        pad_to_length=4,
    )
    scores = torch.ones((4, 32), requires_grad=True)
    sums = masked_categorical_sums(scores, batch.labels, batch.loss_weights, valid=batch.valid_mask)
    result = reduce_masked_categorical([[sums], []], policy="prism_global_weighted_masked_ce_v1")
    assert result.supervised_skip and result.loss.item() == 0
    result.loss.backward()
    assert torch.isfinite(scores.grad).all() and scores.grad.eq(0).all()
