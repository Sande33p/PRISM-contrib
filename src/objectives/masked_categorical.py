"""Offline same-position text CE and explicitly named reduction policies.

The reference math follows InternLumina-U2 commit
70f21c1df8fa26e514df26ba31bc061be8d3d018, tasks/train_sft.py:1180-1224,
1415-1440 and 1520-1532. This independently authored text-only oracle does not
implement U2 visual heads, router computation, collectives, or trainer dispatch.
"""

import math
from collections.abc import Sequence
from dataclasses import dataclass
from numbers import Real

import torch
import torch.nn.functional as F
from torch import Tensor

REFERENCE_POLICY = "u2_reference_rank_mean_v1"
GLOBAL_POLICY = "prism_global_weighted_masked_ce_v1"
REFERENCE_ROUTER = "u2_reference_count_weighted_v1"
GLOBAL_ROUTER = "rank_mean_of_microbatch_means_v1"


def _coefficient(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite nonnegative real number")
    result = float(value)
    if not math.isfinite(result) or result < 0:
        raise ValueError(f"{name} must be a finite nonnegative real number")
    return result


def _floating(value: Tensor, name: str, *, scalar: bool = False) -> None:
    if not isinstance(value, Tensor) or not value.is_floating_point():
        raise ValueError(f"{name} must be a floating-point tensor")
    if value.layout != torch.strided or value.device.type == "meta":
        raise ValueError(f"{name} must be a concrete dense tensor")
    if scalar and value.ndim != 0:
        raise ValueError(f"{name} must be a scalar tensor")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")


@dataclass(frozen=True)
class MaskedCategoricalSums:
    """One pack's weighted CE numerator and detached effective-weight count.

    The numerator includes the text coefficient; the count never does.
    ``supervised_tokens`` counts nonignored labels, including zero-weight ones.
    Tensor contents remain mutable, so reduction revalidates these fields.
    """

    numerator: Tensor
    effective_weight: Tensor
    supervised_tokens: int

    def __post_init__(self) -> None:
        _floating(self.numerator, "numerator", scalar=True)
        _floating(self.effective_weight, "effective_weight", scalar=True)
        if self.effective_weight.requires_grad:
            raise ValueError("effective_weight must be detached")
        if self.numerator.device != self.effective_weight.device:
            raise ValueError("sum and count must use the same device")
        if self.numerator.dtype != self.effective_weight.dtype:
            raise ValueError("sum and count must use the same dtype")
        if bool(self.numerator < 0) or bool(self.effective_weight < 0):
            raise ValueError("sum and count must be nonnegative")
        if type(self.supervised_tokens) is not int or self.supervised_tokens < 0:
            raise ValueError("supervised_tokens must be a nonnegative integer")
        if bool(self.effective_weight == 0) and bool(self.numerator != 0):
            raise ValueError("zero effective weight requires a zero numerator")
        if self.supervised_tokens == 0 and bool(self.effective_weight != 0):
            raise ValueError("zero supervised tokens require zero effective weight")


@dataclass(frozen=True)
class ReductionResult:
    """Scalar objective and local backward losses for an averaging reducer.

    ``router_loss`` is the already weighted auxiliary contribution to ``loss``.
    The mean of ``per_rank_backward`` has the same value and shared-parameter
    gradients as ``loss``. This is an algebraic oracle, not a collective launch.
    ``supervised_skip`` describes zero global supervised weight only: an
    explicitly chosen router objective can still have a nonzero contribution.
    """

    policy: str
    loss: Tensor
    main_loss: Tensor
    router_loss: Tensor
    per_rank_backward: tuple[Tensor, ...]
    global_weight: Tensor
    supervised_skip: bool


def masked_categorical_sums(
    logits: Tensor,
    labels: Tensor,
    weights: Tensor,
    *,
    valid: Tensor | None = None,
    text_coefficient: float = 1.0,
) -> MaskedCategoricalSums:
    """Compute one pack's same-position weighted text CE without any shift.

    ``logits`` has shape ``[..., vocabulary]``; long ``labels`` and fixed
    floating ``weights`` match its prefix shape. Labels of -100 are ignored.
    An optional bool validity mask has the same shape; every invalid position
    must already have label -100. Malformed supervision is never silently
    discarded. Weights must be finite, nonnegative and non-trainable.

    FP16/BF16/FP32 logits use FP32 CE as in the reference. FP64 remains FP64
    for high-precision numerical fixtures; this is not model/dtype parity.
    Positive supervised weights must survive conversion to that compute dtype.
    An empty/all-ignored pack retains a finite zero connected to its logits.
    """
    coefficient = _coefficient(text_coefficient, "text_coefficient")
    _floating(logits, "logits")
    _floating(weights, "weights")
    if logits.ndim < 1 or logits.shape[-1] < 1:
        raise ValueError("logits must have a nonempty vocabulary dimension")
    if (
        not isinstance(labels, Tensor)
        or labels.dtype != torch.long
        or labels.layout != torch.strided
    ):
        raise ValueError("labels must be a dense long tensor")
    if labels.shape != logits.shape[:-1] or weights.shape != labels.shape:
        raise ValueError("labels and weights must match the logits prefix shape")
    if logits.device != labels.device or logits.device != weights.device:
        raise ValueError("logits, labels and weights must use the same device")
    if weights.requires_grad or bool((weights < 0).any()):
        raise ValueError("weights must be nonnegative fixed metadata")
    supervised = labels != -100
    if valid is not None:
        if (
            not isinstance(valid, Tensor)
            or valid.dtype != torch.bool
            or valid.layout != torch.strided
            or valid.shape != labels.shape
            or valid.device != labels.device
        ):
            raise ValueError("valid must be a matching bool tensor on the same device")
        if bool((supervised & ~valid).any()):
            raise ValueError("invalid positions must have ignored labels")
    if bool(((labels < 0) & supervised).any()) or bool((labels >= logits.shape[-1]).any()):
        raise ValueError("nonignored labels must be valid vocabulary IDs")

    dtype = torch.float64 if logits.dtype == torch.float64 else torch.float32
    scores = logits.to(dtype=dtype)
    fixed_weights = weights.detach().to(dtype=dtype)
    if bool((supervised & weights.gt(0) & fixed_weights.eq(0)).any()):
        raise ValueError("positive weights underflow the computation dtype; use float64 logits")
    target_weights = fixed_weights[supervised]
    count = target_weights.sum().detach()
    active = supervised & fixed_weights.gt(0)
    if bool(active.any()) and coefficient > 0:
        # Exclude zero-weight targets before CE: finite extreme logits can
        # produce an infinite CE, and multiplying that by zero would be NaN.
        losses = F.cross_entropy(scores[active], labels[active], reduction="none")
        numerator = (losses * fixed_weights[active]).sum() * coefficient
    else:
        numerator = (scores * 0.0).sum()
    return MaskedCategoricalSums(numerator, count, int(supervised.sum().item()))


def _as_sequence(value: Sequence, name: str) -> tuple:
    if not isinstance(value, Sequence) or isinstance(value, str | bytes | bytearray):
        raise ValueError(f"{name} must be a sequence")
    return tuple(value)


def reduce_masked_categorical(
    ranks: Sequence[Sequence[MaskedCategoricalSums]],
    *,
    policy: str,
    router_aux: Sequence[Sequence[Tensor]] | None = None,
    router_weight: float = 0.0,
    router_reduction: str | None = None,
    zero_anchor: Tensor | None = None,
) -> ReductionResult:
    """Reduce explicit per-rank/per-pack terms without distributed side effects.

    For numerator N, effective weight C and local G=max(sum(C),1), reference
    main loss on each rank is sum((N/max(C,1)) * C/G). Rank losses are averaged;
    both clamps are intentional. Router scalars use sum(aux * C/G), then rank
    averaging and ``router_weight``. Its optional name is
    ``u2_reference_count_weighted_v1``. Fractional packs need not be invariant
    to repartitioning under this reference policy.

    The global main policy uses sum(N)/sum(C) for any positive global count,
    including counts below one. Each rank's backward main loss is world_size
    times its numerator divided by that global count. Do not divide again by
    accumulation count. An averaging shared-model gradient reducer then agrees
    with the scalar objective; no collective or trainer behavior is implemented.
    If world_size/count cannot fit the backward dtype, reduction rejects it
    explicitly instead of silently clamping or returning nonfinite gradients.

    Nonzero global-policy router weight requires the explicit auxiliary policy
    ``rank_mean_of_microbatch_means_v1``: average supplied router scalars within
    each rank, including zero-target microbatches, then average all ranks.
    Empty ranks contribute zero; unequal microbatch counts still give each rank
    equal weight. This router policy is not a global supervised-token mean.

    At least one rank is required; each rank may have no packs. Zero global
    supervised weight returns finite connected zero main loss and sets
    ``supervised_skip``. With no terms/aux/anchor at all, there is no supplied
    graph to connect, so zeros are detached. ``zero_anchor`` is an optional
    finite floating scalar graph hook. All supplied tensors use one dtype and
    device; explicit sums/counts are revalidated before use.
    """
    if type(policy) is not str or policy not in (REFERENCE_POLICY, GLOBAL_POLICY):
        raise ValueError("unknown masked-categorical reduction policy")
    coefficient = _coefficient(router_weight, "router_weight")
    expected_router = REFERENCE_ROUTER if policy == REFERENCE_POLICY else GLOBAL_ROUTER
    if router_reduction is not None and (
        type(router_reduction) is not str or router_reduction != expected_router
    ):
        raise ValueError("unsupported router reduction for the selected main policy")
    if coefficient > 0 and router_aux is None:
        raise ValueError("nonzero router_weight requires explicit router_aux")
    if policy == GLOBAL_POLICY and coefficient > 0 and router_reduction != GLOBAL_ROUTER:
        raise ValueError("global main policy requires an explicit router reduction")

    rank_packs = tuple(_as_sequence(rank, "rank packs") for rank in _as_sequence(ranks, "ranks"))
    if not rank_packs:
        raise ValueError("at least one rank is required")
    tensors: list[Tensor] = []
    for rank in rank_packs:
        for pack in rank:
            if not isinstance(pack, MaskedCategoricalSums):
                raise ValueError("rank packs must contain MaskedCategoricalSums")
            pack.__post_init__()
            tensors.extend((pack.numerator, pack.effective_weight))
    auxiliaries = None
    if router_aux is not None:
        auxiliaries = tuple(
            _as_sequence(rank, "rank router auxiliaries")
            for rank in _as_sequence(router_aux, "router_aux")
        )
        if len(auxiliaries) != len(rank_packs) or any(
            len(aux) != len(packs) for aux, packs in zip(auxiliaries, rank_packs, strict=True)
        ):
            raise ValueError("router auxiliaries must match every rank and pack")
        for rank in auxiliaries:
            for auxiliary in rank:
                _floating(auxiliary, "router auxiliary", scalar=True)
                tensors.append(auxiliary)
    if zero_anchor is not None:
        _floating(zero_anchor, "zero_anchor", scalar=True)
        tensors.append(zero_anchor)
    anchor = tensors[0] if tensors else torch.zeros(())
    if any(value.device != anchor.device or value.dtype != anchor.dtype for value in tensors):
        raise ValueError("all reduction tensors must use one dtype and device")
    # Every explicitly supplied graph has a finite zero contribution, including
    # on empty ranks. This lets the offline oracle materialize zero gradients
    # for an unused head without inventing a model or collective operation.
    zero = sum((value * 0.0 for value in tensors), torch.zeros_like(anchor))
    count_zero = zero.detach()
    rank_counts = [sum((pack.effective_weight for pack in rank), count_zero) for rank in rank_packs]
    global_weight = sum(rank_counts, count_zero)
    _floating(global_weight, "global effective weight", scalar=True)
    skip = bool(global_weight == 0)
    world_size = len(rank_packs)
    if (
        policy == GLOBAL_POLICY
        and not skip
        and float(global_weight) < world_size / torch.finfo(global_weight.dtype).max
    ):
        raise ValueError(
            "global effective weight is too small for finite per-rank backward scaling; "
            "use float64 logits and weights or rescale the weights"
        )
    main_backwards = []
    router_backwards = []
    for index, packs in enumerate(rank_packs):
        rank_count = rank_counts[index].clamp_min(1)
        if policy == REFERENCE_POLICY:
            main = sum(
                (
                    pack.numerator
                    / pack.effective_weight.clamp_min(1)
                    * (pack.effective_weight / rank_count)
                    for pack in packs
                ),
                zero,
            )
        else:
            numerator = sum((pack.numerator for pack in packs), zero)
            main = numerator * 0.0 if skip else numerator / global_weight * world_size
        if auxiliaries is None or coefficient == 0:
            router = zero
        elif policy == REFERENCE_POLICY:
            router = sum(
                (
                    auxiliary * (pack.effective_weight / rank_count)
                    for auxiliary, pack in zip(auxiliaries[index], packs, strict=True)
                ),
                zero,
            )
        else:
            router = sum(auxiliaries[index], zero) / len(packs) if packs else zero
        main_backwards.append(main)
        router_backwards.append(router * coefficient)
    main_loss = sum(main_backwards, zero) / world_size
    router_loss = sum(router_backwards, zero) / world_size
    per_rank = tuple(
        main + router for main, router in zip(main_backwards, router_backwards, strict=True)
    )
    loss = main_loss + router_loss
    for value in (main_loss, router_loss, loss, *per_rank):
        _floating(value, "reduced loss", scalar=True)
    return ReductionResult(policy, loss, main_loss, router_loss, per_rank, global_weight, skip)
