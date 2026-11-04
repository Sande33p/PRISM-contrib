"""Explicit objective oracles; these do not change existing training dispatch."""

from .masked_categorical import (
    MaskedCategoricalSums,
    ReductionResult,
    masked_categorical_sums,
    reduce_masked_categorical,
)

__all__ = [
    "MaskedCategoricalSums",
    "ReductionResult",
    "masked_categorical_sums",
    "reduce_masked_categorical",
]
