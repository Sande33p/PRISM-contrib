"""Hand CE and algebraic distributed-gradient oracles on tiny CPU tensors."""

import math

import pytest
import torch
from src.objectives.masked_categorical import (
    GLOBAL_POLICY,
    GLOBAL_ROUTER,
    REFERENCE_POLICY,
    MaskedCategoricalSums,
    masked_categorical_sums,
    reduce_masked_categorical,
)
from torch.utils._python_dispatch import TorchDispatchMode

pytestmark = pytest.mark.unit
LN2 = math.log(2)


def _pack(probabilities, weights=None):
    logits = torch.tensor(
        [[math.log(p), math.log(1 - p)] for p in probabilities],
        dtype=torch.float64,
        requires_grad=True,
    )
    labels = torch.zeros(len(probabilities), dtype=torch.long)
    weights = torch.tensor(weights or [1.0] * len(probabilities), dtype=torch.float64)
    return masked_categorical_sums(logits, labels, weights)


def _zero_pack():
    logits = torch.zeros(1, 2, dtype=torch.float64, requires_grad=True)
    return masked_categorical_sums(logits, torch.tensor([-100]), torch.ones(1, dtype=torch.float64))


def test_same_position_weighted_ce_and_exact_gradient():
    logits = torch.tensor(
        [[[math.log(3), 0.0], [0.0, math.log(3)], [0.0, math.log(3)]]],
        dtype=torch.float64,
        requires_grad=True,
    )
    terms = masked_categorical_sums(
        logits,
        torch.tensor([[0, 1, -100]]),
        torch.tensor([[1.0, 0.25, 8.0]], dtype=torch.float64),
        valid=torch.tensor([[True, True, False]]),
    )
    assert terms.numerator.item() == pytest.approx(1.25 * math.log(4 / 3))
    assert terms.effective_weight.item() == 1.25
    assert terms.supervised_tokens == 2
    assert not terms.effective_weight.requires_grad
    terms.numerator.backward()
    torch.testing.assert_close(
        logits.grad,
        torch.tensor([[[-0.25, 0.25], [0.0625, -0.0625], [0.0, 0.0]]], dtype=torch.float64),
    )
    # A causal shift would compare the first logits with label 1 and give ln(4).
    assert terms.numerator.item() != pytest.approx(math.log(4))


def test_text_coefficient_changes_numerator_only():
    logits = torch.zeros(2, 2, requires_grad=True)
    labels = torch.tensor([0, 1])
    weights = torch.tensor([1.0, 3.0])
    first = masked_categorical_sums(logits, labels, weights)
    scaled = masked_categorical_sums(logits, labels, weights, text_coefficient=2.5)
    assert scaled.numerator.item() == pytest.approx(2.5 * first.numerator.item())
    assert scaled.effective_weight.item() == first.effective_weight.item() == 4


@pytest.mark.parametrize("world_size,weight", [(1, 1e-38), (4, 2e-38)])
def test_tiny_weights_and_text_coefficient_have_finite_analytic_gradients(world_size, weight):
    logits = torch.zeros((1, 2), requires_grad=True)
    weights = torch.tensor([weight])
    terms = masked_categorical_sums(logits, torch.tensor([0]), weights, text_coefficient=4.0)
    ranks = [[terms]] + [[] for _ in range(world_size - 1)]
    result = reduce_masked_categorical(ranks, policy=GLOBAL_POLICY)
    assert result.loss.item() == pytest.approx(4 * LN2)
    assert terms.effective_weight.item() == weights.sum().item()
    scalar_grad = torch.autograd.grad(result.loss, logits, retain_graph=True)[0]
    local_grad = torch.autograd.grad(result.per_rank_backward[0], logits)[0]
    torch.testing.assert_close(scalar_grad, torch.tensor([[-2.0, 2.0]]))
    torch.testing.assert_close(local_grad, torch.tensor([[-2.0, 2.0]]) * world_size)


class FloatAllocationRecorder(TorchDispatchMode):
    """Observe produced FP32 tensors without allocating a large model-shaped input."""

    def __init__(self):
        self.shapes = []

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        result = func(*args, **(kwargs or {}))
        outputs = result if isinstance(result, tuple | list) else (result,)
        for output in outputs:
            if isinstance(output, torch.Tensor) and output.dtype == torch.float32:
                self.shapes.append(tuple(output.shape))
        return result


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("kind", ["active", "ignored", "zero_weight", "zero_coefficient"])
def test_float32_temporaries_scale_with_active_rows_not_full_logits(dtype, kind):
    # Exercise row selection and connected-zero behavior on noncontiguous input.
    logits = torch.zeros((16, 8), dtype=dtype).T.requires_grad_()
    labels = torch.full((8,), -100 if kind in ("active", "ignored") else 0)
    weights = torch.zeros(8) if kind == "zero_weight" else torch.ones(8)
    if kind == "active":
        labels[0] = 0
    coefficient = 0.0 if kind == "zero_coefficient" else 1.0
    recorder = FloatAllocationRecorder()
    with recorder:
        terms = masked_categorical_sums(logits, labels, weights, text_coefficient=coefficient)
    # Only one vocabulary row is active; the other cases need scalar/count
    # metadata, never a dense FP32 copy or multiplication of the full logits.
    assert all(math.prod(shape) <= 16 for shape in recorder.shapes), recorder.shapes
    assert terms.numerator.dtype == terms.effective_weight.dtype == torch.float32
    terms.numerator.backward()
    assert torch.isfinite(logits.grad).all()
    if kind == "active":
        torch.testing.assert_close(logits.grad[0].float(), torch.tensor([-15 / 16] + [1 / 16] * 15))
        assert logits.grad[1:].eq(0).all()
    else:
        assert terms.numerator.item() == 0
        assert logits.grad.eq(0).all()


@pytest.mark.parametrize("mixed", [False, True])
def test_zero_weight_extreme_logits_do_not_evaluate_unneeded_ce(mixed):
    values = [[-3e38, 3e38]] + ([[0.0, 0.0]] if mixed else [])
    logits = torch.tensor(values, requires_grad=True)
    labels = torch.zeros(len(values), dtype=torch.long)
    weights = torch.tensor([0.0] + ([1.0] if mixed else []))
    terms = masked_categorical_sums(logits, labels, weights)
    result = reduce_masked_categorical([[terms]], policy=GLOBAL_POLICY)
    assert terms.supervised_tokens == len(values)
    assert result.loss.item() == pytest.approx(LN2 if mixed else 0.0)
    result.loss.backward()
    assert torch.isfinite(logits.grad).all()
    torch.testing.assert_close(logits.grad[0], torch.zeros(2))
    if mixed:
        torch.testing.assert_close(logits.grad[1], torch.tensor([-0.5, 0.5]))


def test_zero_text_coefficient_retains_count_and_finite_zero_gradient():
    logits = torch.tensor([[-3e38, 3e38]], requires_grad=True)
    terms = masked_categorical_sums(logits, torch.tensor([0]), torch.ones(1), text_coefficient=0.0)
    assert terms.effective_weight.item() == 1
    assert terms.numerator.item() == 0
    terms.numerator.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


@pytest.mark.parametrize("policy", [REFERENCE_POLICY, GLOBAL_POLICY])
def test_zero_router_coefficient_skips_overflowing_unused_sum(policy):
    auxiliaries = [torch.tensor(1e308, dtype=torch.float64, requires_grad=True) for _ in range(2)]
    result = reduce_masked_categorical(
        [[_zero_pack(), _zero_pack()]],
        policy=policy,
        router_aux=[auxiliaries],
        router_weight=0.0,
    )
    assert result.loss.item() == result.router_loss.item() == 0
    result.loss.backward()
    assert all(value.grad.item() == 0 for value in auxiliaries)


def test_global_backward_scale_must_fit_computation_dtype():
    logits = torch.zeros((1, 2), requires_grad=True)
    term = masked_categorical_sums(logits, torch.tensor([0]), torch.tensor([1e-38]))
    # 1/C fits FP32, but the exposed four-rank local backward scale 4/C does not.
    with pytest.raises(ValueError, match="finite per-rank backward scaling"):
        reduce_masked_categorical([[term], [], [], []], policy=GLOBAL_POLICY)
    reference = reduce_masked_categorical([[term]], policy=REFERENCE_POLICY)
    assert torch.isfinite(reference.loss)
    precise = torch.zeros((1, 2), dtype=torch.float64, requires_grad=True)
    term64 = masked_categorical_sums(
        precise, torch.tensor([0]), torch.tensor([1e-40], dtype=torch.float64)
    )
    result = reduce_masked_categorical([[term64]], policy=GLOBAL_POLICY)
    result.loss.backward()
    assert result.loss.item() == pytest.approx(LN2)
    torch.testing.assert_close(precise.grad, torch.tensor([[-0.5, 0.5]], dtype=torch.float64))


def test_positive_supervised_weights_cannot_silently_underflow_to_zero():
    with pytest.raises(ValueError, match="underflow the computation dtype"):
        masked_categorical_sums(
            torch.zeros((1, 2)), torch.tensor([0]), torch.tensor([1e-50], dtype=torch.float64)
        )


@pytest.mark.parametrize(
    "weight,coefficient,message",
    [
        (1e-20, 1e39, "text_coefficient exceeds"),
        (1e-38, 1e-10, "coefficient-scaled weights underflow"),
        (1e38, 4.0, "coefficient-scaled weights exceed"),
    ],
)
def test_coefficient_scaled_weights_must_fit_computation_dtype(weight, coefficient, message):
    with pytest.raises(ValueError, match=message):
        masked_categorical_sums(
            torch.zeros((1, 2)),
            torch.tensor([0]),
            torch.tensor([weight]),
            text_coefficient=coefficient,
        )
    precise = torch.zeros((1, 2), dtype=torch.float64, requires_grad=True)
    terms = masked_categorical_sums(
        precise,
        torch.tensor([0]),
        torch.tensor([weight], dtype=torch.float64),
        text_coefficient=coefficient,
    )
    result = reduce_masked_categorical([[terms]], policy=GLOBAL_POLICY)
    result.loss.backward()
    assert result.loss.item() == pytest.approx(coefficient * LN2, rel=1e-12, abs=0)
    torch.testing.assert_close(
        precise.grad, torch.tensor([[-0.5, 0.5]], dtype=torch.float64) * coefficient
    )


@pytest.mark.parametrize("kind", ["ignored", "zero_weight", "zero_coefficient"])
def test_scalar_prefix_zero_branches_retain_connected_zero(kind):
    logits = torch.zeros(3, dtype=torch.float16, requires_grad=True)
    terms = masked_categorical_sums(
        logits,
        torch.tensor(-100 if kind == "ignored" else 0),
        torch.tensor(0.0 if kind == "zero_weight" else 1.0),
        text_coefficient=0.0 if kind == "zero_coefficient" else 1.0,
    )
    assert terms.numerator.dtype == torch.float32
    assert terms.numerator.item() == 0
    terms.numerator.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_compute_dtype_and_finite_cpu_gradients(dtype):
    logits = torch.tensor([[0.1, -0.2]], dtype=dtype, requires_grad=True)
    terms = masked_categorical_sums(logits, torch.tensor([0]), torch.ones(1, dtype=dtype))
    expected = torch.float64 if dtype == torch.float64 else torch.float32
    assert terms.numerator.dtype == terms.effective_weight.dtype == expected
    terms.numerator.backward()
    assert logits.grad.dtype == dtype
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("policy", [REFERENCE_POLICY, GLOBAL_POLICY])
@pytest.mark.parametrize("kind", ["empty", "ignored", "zero_weight"])
def test_zero_target_cases_have_connected_finite_zero(policy, kind):
    size = 0 if kind == "empty" else 2
    logits = torch.zeros(size, 3, dtype=torch.float64, requires_grad=True)
    labels = torch.full((size,), -100 if kind != "zero_weight" else 1, dtype=torch.long)
    weights = torch.full((size,), 0.0 if kind == "zero_weight" else 1.0, dtype=torch.float64)
    terms = masked_categorical_sums(logits, labels, weights)
    result = reduce_masked_categorical([[terms], []], policy=policy)
    assert result.supervised_skip
    assert result.main_loss.item() == result.loss.item() == result.global_weight.item() == 0
    result.loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_reference_rank_mean_and_global_mean_are_distinct():
    ranks = [[_pack([0.5])], [_pack([0.25, 0.25, 0.25])]]
    reference = reduce_masked_categorical(ranks, policy=REFERENCE_POLICY)
    global_mean = reduce_masked_categorical(ranks, policy=GLOBAL_POLICY)
    assert reference.loss.item() == pytest.approx(1.5 * LN2)
    assert global_mean.loss.item() == pytest.approx(1.75 * LN2)
    assert [value.item() for value in reference.per_rank_backward] == pytest.approx([LN2, 2 * LN2])
    assert [value.item() for value in global_mean.per_rank_backward] == pytest.approx(
        [0.5 * LN2, 3 * LN2]
    )


def test_zero_target_rank_is_included_in_reference_averaging():
    ranks = [[_pack([0.25])], [_zero_pack()]]
    reference = reduce_masked_categorical(ranks, policy=REFERENCE_POLICY)
    global_mean = reduce_masked_categorical(ranks, policy=GLOBAL_POLICY)
    assert reference.loss.item() == pytest.approx(LN2)
    assert global_mean.loss.item() == pytest.approx(2 * LN2)
    assert global_mean.per_rank_backward[1].item() == 0


def test_reference_preserves_both_fractional_clamps():
    split = [[_pack([0.5], [0.25]), _pack([0.5], [0.25])]]
    packed = [[_pack([0.5, 0.5], [0.25, 0.25])]]
    assert reduce_masked_categorical(split, policy=REFERENCE_POLICY).loss.item() == pytest.approx(
        0.125 * LN2
    )
    assert reduce_masked_categorical(packed, policy=REFERENCE_POLICY).loss.item() == pytest.approx(
        0.25 * LN2
    )
    for ranks in (split, packed):
        result = reduce_masked_categorical(ranks, policy=GLOBAL_POLICY)
        assert result.loss.item() == pytest.approx(LN2)
        assert result.global_weight.item() == 0.5


def test_global_reduction_is_pack_rank_and_accumulation_invariant():
    logits = torch.tensor(
        [[0.1, 0.7], [0.5, -0.2], [-0.9, 0.2], [0.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    labels = torch.tensor([0, 1, 0, -100])
    weights = torch.tensor([0.25, 0.5, 2.0, 99.0], dtype=torch.float64)
    one = masked_categorical_sums(logits, labels, weights)
    parts = [
        masked_categorical_sums(logits[i : i + 1], labels[i : i + 1], weights[i : i + 1])
        for i in range(4)
    ]
    single = reduce_masked_categorical([[one]], policy=GLOBAL_POLICY)
    split = reduce_masked_categorical([[parts[0], parts[3]], [], parts[1:3]], policy=GLOBAL_POLICY)
    torch.testing.assert_close(single.loss, split.loss)
    expected_gradient = torch.autograd.grad(single.loss, logits, retain_graph=True)[0]
    actual_gradient = torch.autograd.grad(split.loss, logits)[0]
    torch.testing.assert_close(expected_gradient, actual_gradient)
    assert split.global_weight.item() == 2.75


@pytest.mark.parametrize("policy", [REFERENCE_POLICY, GLOBAL_POLICY])
def test_averaged_rank_backward_gradients_equal_scalar_objective(policy):
    parameter = torch.tensor(0.3, dtype=torch.float64, requires_grad=True)
    router_parameter = torch.tensor(0.4, dtype=torch.float64, requires_grad=True)
    offsets = [-0.2, 0.7, 0.1]
    weights = [0.25, 1.5, 0.5]
    parts = []
    for offset, weight in zip(offsets, weights, strict=True):
        scores = torch.stack((parameter + offset, parameter * 0)).unsqueeze(0)
        parts.append(
            masked_categorical_sums(
                scores,
                torch.tensor([0]),
                torch.tensor([weight], dtype=torch.float64),
                text_coefficient=2.0,
            )
        )
    ranks = [[parts[0]], [parts[1], parts[2]], []]
    auxiliaries = [[router_parameter * 2], [router_parameter * 3, router_parameter * 4], []]
    result = reduce_masked_categorical(
        ranks,
        policy=policy,
        router_aux=auxiliaries,
        router_weight=0.1,
        router_reduction=GLOBAL_ROUTER if policy == GLOBAL_POLICY else None,
    )
    direct = torch.autograd.grad(result.loss, (parameter, router_parameter), retain_graph=True)
    rank_gradients = [
        torch.autograd.grad(value, (parameter, router_parameter), retain_graph=True)
        for value in result.per_rank_backward
    ]
    for index, gradient in enumerate(direct):
        torch.testing.assert_close(
            gradient, sum(values[index] for values in rank_gradients) / len(ranks)
        )
    if policy == GLOBAL_POLICY:
        hand_gradient = (
            2
            * sum(
                weight * (1 / (1 + math.exp(-(parameter.item() + offset))) - 1)
                for offset, weight in zip(offsets, weights, strict=True)
            )
            / sum(weights)
        )
        assert direct[0].item() == pytest.approx(hand_gradient)
        assert direct[1].item() == pytest.approx(0.1 * (2 + 3.5 + 0) / 3)


def test_router_contribution_is_separate_and_global_requires_named_policy():
    ranks = [[_pack([0.5]), _pack([0.5] * 3), _zero_pack()], [_pack([0.5] * 2)]]
    auxiliaries = [
        [torch.tensor(x, dtype=torch.float64) for x in [2.0, 4.0, 100.0]],
        [torch.tensor(6.0, dtype=torch.float64)],
    ]
    reference = reduce_masked_categorical(
        ranks, policy=REFERENCE_POLICY, router_aux=auxiliaries, router_weight=0.1
    )
    assert reference.router_loss.item() == pytest.approx(0.1 * (3.5 + 6) / 2)
    with pytest.raises(ValueError, match="explicit router reduction"):
        reduce_masked_categorical(
            ranks, policy=GLOBAL_POLICY, router_aux=auxiliaries, router_weight=0.1
        )
    global_mean = reduce_masked_categorical(
        ranks,
        policy=GLOBAL_POLICY,
        router_aux=auxiliaries,
        router_weight=0.1,
        router_reduction=GLOBAL_ROUTER,
    )
    assert global_mean.router_loss.item() == pytest.approx(0.1 * (106 / 3 + 6) / 2)
    assert global_mean.main_loss.item() == pytest.approx(LN2)
    torch.testing.assert_close(global_mean.loss, global_mean.main_loss + global_mean.router_loss)


def test_fractional_reference_router_uses_local_count_clamp_without_pack_division():
    result = reduce_masked_categorical(
        [[_pack([0.5], [0.25]), _pack([0.5], [0.25])]],
        policy=REFERENCE_POLICY,
        router_aux=[
            [torch.tensor(2.0, dtype=torch.float64), torch.tensor(4.0, dtype=torch.float64)]
        ],
        router_weight=0.1,
    )
    assert result.router_loss.item() == pytest.approx(0.15)


def test_global_supervised_skip_does_not_silently_disable_explicit_router_objective():
    router = torch.tensor(2.0, dtype=torch.float64, requires_grad=True)
    result = reduce_masked_categorical(
        [[_zero_pack()]],
        policy=GLOBAL_POLICY,
        router_aux=[[router]],
        router_weight=0.1,
        router_reduction=GLOBAL_ROUTER,
    )
    assert result.supervised_skip
    assert result.main_loss.item() == 0
    assert result.loss.item() == pytest.approx(0.2)
    result.loss.backward()
    assert router.grad.item() == pytest.approx(0.1)


@pytest.mark.parametrize("policy", [REFERENCE_POLICY, GLOBAL_POLICY])
def test_all_empty_ranks_with_and_without_graph_anchor(policy):
    detached = reduce_masked_categorical([[], []], policy=policy)
    assert detached.loss.item() == 0 and detached.supervised_skip
    assert not detached.loss.requires_grad
    anchor = torch.tensor(2.0, requires_grad=True)
    connected = reduce_masked_categorical([[], []], policy=policy, zero_anchor=anchor)
    connected.loss.backward()
    assert anchor.grad.item() == 0


@pytest.mark.parametrize(
    "override",
    [
        {"labels": torch.tensor([0.0, 1.0])},
        {"labels": torch.tensor([0, 2])},
        {"labels": torch.tensor([0, -1])},
        {"labels": torch.tensor([[0, 1]])},
        {"labels": torch.tensor([0, 1]).to_sparse()},
        {"weights": torch.tensor([1.0])},
        {"weights": torch.tensor([1, 1])},
        {"weights": torch.tensor([1.0, -1.0])},
        {"weights": torch.tensor([1.0, float("nan")])},
        {"weights": torch.ones(2, requires_grad=True)},
        {"logits": torch.ones(2, 2, dtype=torch.long)},
        {"logits": torch.full((2, 2), float("inf"))},
        {"logits": torch.zeros(2, 0)},
        {"logits": torch.ones(2, 2).to_sparse()},
        {"valid": torch.tensor([True, False])},
        {"valid": torch.tensor([1, 1])},
        {"valid": torch.tensor([True])},
        {"valid": torch.tensor([True, True]).to_sparse()},
    ],
)
def test_invalid_supervision_and_tensors_are_rejected(override):
    args = {
        "logits": torch.zeros(2, 2),
        "labels": torch.tensor([0, 1]),
        "weights": torch.ones(2),
        **override,
    }
    with pytest.raises(ValueError):
        masked_categorical_sums(**args)


@pytest.mark.parametrize("value", [-1.0, float("nan"), float("inf"), True, "1"])
def test_invalid_coefficients_rejected(value):
    with pytest.raises(ValueError):
        masked_categorical_sums(
            torch.zeros(1, 2), torch.tensor([0]), torch.ones(1), text_coefficient=value
        )
    with pytest.raises(ValueError):
        reduce_masked_categorical([[]], policy=GLOBAL_POLICY, router_weight=value)


@pytest.mark.parametrize("policy", ["unknown", "", None, True, 1])
def test_unknown_policy_is_rejected(policy):
    with pytest.raises(ValueError):
        reduce_masked_categorical([[]], policy=policy)


@pytest.mark.parametrize("ranks", [[], "bad", [None], [[1]]])
def test_malformed_rank_structure_is_rejected(ranks):
    with pytest.raises(ValueError):
        reduce_masked_categorical(ranks, policy=GLOBAL_POLICY)


def test_malformed_auxiliaries_counts_and_anchor_are_rejected():
    terms = _pack([0.5])
    for kwargs in (
        {"router_reduction": "unknown"},
        {"router_reduction": True},
        {"router_weight": 1.0, "router_reduction": GLOBAL_ROUTER},
        {"router_aux": []},
        {"router_aux": [[torch.tensor([1.0], dtype=torch.float64)]]},
        {"router_aux": [[torch.tensor(float("nan"), dtype=torch.float64)]]},
        {"router_aux": [[torch.tensor(1.0, dtype=torch.float32)]]},
        {"zero_anchor": torch.tensor([1.0], dtype=torch.float64)},
    ):
        with pytest.raises(ValueError):
            reduce_masked_categorical([[terms]], policy=GLOBAL_POLICY, **kwargs)
    with pytest.raises(ValueError, match="detached"):
        MaskedCategoricalSums(torch.tensor(1.0), torch.tensor(1.0, requires_grad=True), 1)
    with pytest.raises(ValueError, match="zero effective weight"):
        MaskedCategoricalSums(torch.tensor(1.0), torch.tensor(0.0), 1)
    with pytest.raises(ValueError, match="zero supervised tokens"):
        MaskedCategoricalSums(torch.tensor(1.0), torch.tensor(1.0), 0)
    terms.effective_weight.fill_(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        reduce_masked_categorical([[terms]], policy=GLOBAL_POLICY)


def test_reduction_and_sum_overflow_reject_nonfinite_results():
    terms = MaskedCategoricalSums(torch.tensor(3e38), torch.tensor(1.0), 1)
    with pytest.raises(ValueError, match="finite"):
        reduce_masked_categorical([[terms, terms]], policy=GLOBAL_POLICY)
    with pytest.raises(ValueError, match="finite"):
        masked_categorical_sums(torch.tensor([[-3e38, 3e38]]), torch.tensor([0]), torch.ones(1))
