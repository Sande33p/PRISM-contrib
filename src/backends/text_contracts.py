"""Immutable token-level contracts for an explicitly injected text runtime.

These types declare a protocol, not verified artifacts or pretrained support.
They have no model, tokenizer, tensor, or serving dependencies.
"""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

FinishReason = Literal["eoa", "length", "budget", "cancelled", "error"]


def _integer(value: int, name: str, minimum: int = 0, maximum: int | None = None) -> None:
    if type(value) is not int or value < minimum or (maximum is not None and value > maximum):
        raise ValueError(f"{name} must be an integer in the supported range")


def _text(value: str, name: str, maximum: int = 256) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > maximum:
        raise ValueError(f"{name} must be a nonempty string of at most {maximum} characters")


def _token_ids(values: Iterable[int], name: str, maximum: int) -> tuple[int, ...]:
    if isinstance(values, str | bytes | bytearray | Mapping):
        raise ValueError(f"{name} must be an iterable of integer token IDs")
    try:
        iterator = iter(values)
    except TypeError as exc:
        raise ValueError(f"{name} must be an iterable of integer token IDs") from exc
    result = []
    for value in iterator:
        _integer(value, name)
        result.append(value)
        if len(result) > maximum:
            raise ValueError(f"{name} exceeds the supported token limit")
    return tuple(result)


@dataclass(frozen=True, slots=True)
class RuntimeIdentity:
    """Driver-declared identity; it does not prove artifact or tokenizer parity."""

    runtime_id: str
    source_revision: str
    tokenizer_digest: str
    vocab_size: int
    eoa_token_id: int
    mask_token_id: int

    def __post_init__(self) -> None:
        for name in ("runtime_id", "source_revision", "tokenizer_digest"):
            _text(getattr(self, name), name)
        _integer(self.vocab_size, "vocab_size", 2)
        for name in ("eoa_token_id", "mask_token_id"):
            _integer(getattr(self, name), name, maximum=self.vocab_size - 1)
        if self.eoa_token_id == self.mask_token_id:
            raise ValueError("EOA and mask token IDs must be distinct")


@dataclass(frozen=True, slots=True)
class TextRequest:
    """One immutable token-level request with bounded API resource settings.

    Limits are protocol bounds, not a claim of model or accelerator capacity.
    Prompt IDs must already use the declared tokenizer; no rendering occurs.
    """

    request_id: str
    prompt_ids: tuple[int, ...]
    tokenizer_digest: str
    max_new_tokens: int = 128
    max_model_calls: int = 128
    block_size: int = 32
    steps_per_block: int = 32
    seed: int = 0
    contract_version: int = 1

    def __post_init__(self) -> None:
        _text(self.request_id, "request_id", 128)
        _text(self.tokenizer_digest, "tokenizer_digest")
        object.__setattr__(self, "prompt_ids", _token_ids(self.prompt_ids, "prompt_ids", 4096))
        if not self.prompt_ids:
            raise ValueError("prompt_ids must not be empty")
        for name in ("max_new_tokens", "max_model_calls", "block_size", "steps_per_block"):
            _integer(getattr(self, name), name, 1, 128)
        _integer(self.seed, "seed", maximum=2**64 - 1)
        _integer(self.contract_version, "contract_version", 1, 1)


@dataclass(frozen=True, slots=True)
class UsageReceipt:
    """Observed call starts and completed-operation positions, never estimates.

    An incomplete receipt retains observed lower bounds but leaves aggregate
    model_calls/processed_positions unknown. Token counts describe submitted
    prompt IDs and committed output IDs, not a billing or hardware metric.
    """

    prompt_tokens: int
    completion_tokens: int
    observed_model_calls: int
    observed_processed_positions: int
    model_calls: int | None
    processed_positions: int | None
    complete: bool

    def __post_init__(self) -> None:
        for name in (
            "prompt_tokens",
            "completion_tokens",
            "observed_model_calls",
            "observed_processed_positions",
        ):
            _integer(getattr(self, name), name)
        if type(self.complete) is not bool:
            raise ValueError("complete must be a bool")
        if self.complete:
            if (
                self.model_calls != self.observed_model_calls
                or self.processed_positions != self.observed_processed_positions
            ):
                raise ValueError("Complete usage totals must equal observed counters")
            _integer(self.model_calls, "model_calls")
            _integer(self.processed_positions, "processed_positions")
        elif self.model_calls is not None or self.processed_positions is not None:
            raise ValueError("Incomplete usage aggregate totals must be unknown")


@dataclass(frozen=True, slots=True)
class TextResult:
    """Terminal receipt containing only stable, committed, output-only IDs."""

    contract_version: int
    request_id: str
    identity: RuntimeIdentity
    output_ids: tuple[int, ...]
    committed_block_ends: tuple[int, ...]
    finish_reason: FinishReason
    usage: UsageReceipt
    seed: int
    eoa_observed: bool
    error_code: str | None = None

    def __post_init__(self) -> None:
        _integer(self.contract_version, "contract_version", 1, 1)
        _text(self.request_id, "request_id", 128)
        if not isinstance(self.identity, RuntimeIdentity) or not isinstance(
            self.usage, UsageReceipt
        ):
            raise ValueError("identity and usage must use the canonical contracts")
        object.__setattr__(self, "output_ids", _token_ids(self.output_ids, "output_ids", 128))
        ends = _token_ids(self.committed_block_ends, "committed_block_ends", 128)
        object.__setattr__(self, "committed_block_ends", ends)
        if any(token >= self.identity.vocab_size for token in self.output_ids):
            raise ValueError("output_ids contain out-of-vocabulary tokens")
        if any(
            token in (self.identity.eoa_token_id, self.identity.mask_token_id)
            for token in self.output_ids
        ):
            raise ValueError("Output must not contain mask or EOA IDs")
        previous = 0
        for end in ends:
            if end <= previous or end > len(self.output_ids):
                raise ValueError("Committed block ends must be increasing output offsets")
            previous = end
        if (self.output_ids and previous != len(self.output_ids)) or (not self.output_ids and ends):
            raise ValueError("Committed block ends must cover the output exactly")
        if not isinstance(self.finish_reason, str) or self.finish_reason not in (
            "eoa",
            "length",
            "budget",
            "cancelled",
            "error",
        ):
            raise ValueError("Unsupported finish_reason")
        if self.usage.completion_tokens != len(self.output_ids):
            raise ValueError("completion_tokens must match committed output")
        _integer(self.seed, "seed", maximum=2**64 - 1)
        if type(self.eoa_observed) is not bool:
            raise ValueError("eoa_observed must be a bool")
        if self.finish_reason == "eoa" and not self.eoa_observed:
            raise ValueError("EOA finish requires observed EOA")
        if self.finish_reason == "error":
            _text(self.error_code, "error_code", 128)  # type: ignore[arg-type]
            if self.usage.complete:
                raise ValueError("Error receipts must leave aggregate usage unknown")
        elif self.error_code is not None:
            raise ValueError("Only error receipts may carry an error_code")
