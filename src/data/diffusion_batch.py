"""Offline, pretokenized text denoising batches and a visibility oracle.

The schedule and two-stream layout follow InternLM/InternLumina-U2 commit
70f21c1df8fa26e514df26ba31bc061be8d3d018, specifically
``tasks/dataset/xllmx_dataset_sft_multitask_blkpad.py`` (lines 507-610,
638-701, 862-1019), and the text attention rules in
``models/internluminau2/modeling_internluminau2.py`` (lines 1411-1512).
This independently authored oracle does not load a tokenizer, render chat,
run a model, or establish runtime parity. Tokenizer identity is caller-declared.

RNG calls deliberately follow the source's Python ``random`` call order, using
an isolated generator. States and a draw trace are retained; seeded identity
across different Python versions is not promised. Artificial absorbing tails
are valid supervised positions, distinct from invalid batch padding.
"""

from __future__ import annotations

import math
import random
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal

import torch

SCHEMA_VERSION = "prism_denoise_batch_v1"
SCHEDULE_VERSION = "u2_text_block_cosine_v1"
MAX_COMPILED_POSITIONS = 768
IGNORE_INDEX = -100

# Upstream's explicit special-token registry; tokenizer-specific specials are
# additional caller-supplied metadata. EOA and think-close remain maskable.
U2_SPECIAL_TOKEN_IDS = (
    157153,
    157154,
    157155,
    157156,
    157157,
    157158,
    157160,
    157161,
    157165,
    157166,
    157167,
    157168,
    157169,
    157170,
    157171,
    157172,
    157175,
    157176,
)

TurnKind = Literal["context", "answer", "think"]
StreamKind = Literal["context", "noisy", "clean"]


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum or value > 2**63 - 1:
        raise ValueError(f"{name} must be an integer in [{minimum}, {2**63 - 1}]")
    return value


def _ids(values: object, name: str) -> tuple[int, ...]:
    if not isinstance(values, tuple | list):
        raise ValueError(f"{name} must be a list or tuple of integer token IDs")
    return tuple(_integer(value, name) for value in values)


@dataclass(frozen=True)
class TextTokenContract:
    """Caller-declared token identity; no artifact verification is performed.

    The defaults are the pinned U2 IDs. Small synthetic vocabularies can supply
    their own IDs and ``tokenizer_special_ids``. ``pad_token_id`` may equal EOA:
    validity is always represented separately from token identity. A PAD ID
    occurring as valid text is protected only when declared in
    ``tokenizer_special_ids``; merely choosing the batch-padding ID does not
    change corruption eligibility. EOA and think-close stay maskable even if
    declared special or used for batch padding.
    """

    tokenizer_digest: str
    vocab_size: int = 157184
    mask_token_id: int = 156895
    pad_token_id: int = 156892
    boa_token_id: int = 157165
    eoa_token_id: int = 157166
    think_open_token_id: int = 157175
    think_close_token_id: int = 157176
    tokenizer_special_ids: tuple[int, ...] = U2_SPECIAL_TOKEN_IDS

    def __post_init__(self) -> None:
        if (
            not isinstance(self.tokenizer_digest, str)
            or not self.tokenizer_digest.strip()
            or self.tokenizer_digest != self.tokenizer_digest.strip()
        ):
            raise ValueError("tokenizer_digest must be a nonempty trimmed string")
        size = _integer(self.vocab_size, "vocab_size", minimum=1)
        structural = []
        for name in (
            "mask_token_id",
            "pad_token_id",
            "boa_token_id",
            "eoa_token_id",
            "think_open_token_id",
            "think_close_token_id",
        ):
            value = _integer(getattr(self, name), name)
            if value >= size:
                raise ValueError(f"{name} must be smaller than vocab_size")
            if name != "pad_token_id":
                structural.append(value)
        if len(set(structural)) != len(structural):
            raise ValueError("mask, BOA, EOA and think token IDs must be distinct")
        special = _ids(self.tokenizer_special_ids, "tokenizer_special_ids")
        if any(value >= size for value in special):
            raise ValueError("tokenizer_special_ids must be within vocab_size")
        object.__setattr__(self, "tokenizer_special_ids", special)

    @property
    def never_mask_ids(self) -> frozenset[int]:
        """Special IDs that remain visible, excluding absorbing targets."""
        # U2's source registry is independent of tokenizer.all_special_ids.
        # Custom synthetic vocabularies must not inherit U2's numeric IDs.
        canonical = (
            self.vocab_size,
            self.mask_token_id,
            self.boa_token_id,
            self.eoa_token_id,
            self.think_open_token_id,
            self.think_close_token_id,
        ) == (157184, 156895, 157165, 157166, 157175, 157176)
        source_specials = set(U2_SPECIAL_TOKEN_IDS) if canonical else set()
        return frozenset(
            (
                source_specials
                | set(self.tokenizer_special_ids)
                | {self.boa_token_id, self.think_open_token_id}
            )
            - {self.eoa_token_id, self.think_close_token_id}
        )


@dataclass(frozen=True)
class TextTurn:
    """Explicit text IDs, without implicit chat rendering or token truncation.

    Context is copied once without supervision. Answers get a natural EOA if
    missing. Think turns get a closing token only when starting with think-open
    and lacking a final think-close. Empty think/context turns are rejected;
    an empty answer is a valid EOA-only target. BOA insertion is explicit.
    """

    token_ids: tuple[int, ...]
    kind: TurnKind
    add_boa: bool = False

    def __post_init__(self) -> None:
        tokens = _ids(self.token_ids, "token_ids")
        if not isinstance(self.kind, str) or self.kind not in ("context", "answer", "think"):
            raise ValueError("kind must be context, answer or think")
        if type(self.add_boa) is not bool:
            raise ValueError("add_boa must be a bool")
        if self.kind == "context" and self.add_boa:
            raise ValueError("context turns cannot insert BOA")
        if not tokens and self.kind != "answer":
            raise ValueError("context and think turns must not be empty")
        object.__setattr__(self, "token_ids", tokens)


@dataclass(frozen=True)
class DenoiseSample:
    """A packed sample with a unique nonnegative integer identity.

    Turn kinds are explicit: no conversation grammar is inferred. Context-only
    samples are supported and have zero supervised targets.
    """

    sample_id: int
    turns: tuple[TextTurn, ...]

    def __post_init__(self) -> None:
        _integer(self.sample_id, "sample_id")
        if not isinstance(self.turns, tuple | list) or not self.turns:
            raise ValueError("turns must be a nonempty list or tuple of TextTurn")
        if any(not isinstance(turn, TextTurn) for turn in self.turns):
            raise ValueError("turns must contain only TextTurn instances")
        object.__setattr__(self, "turns", tuple(self.turns))


@dataclass(frozen=True)
class BatchSpan:
    """Half-open physical span; natural_length excludes artificial tails."""

    sample_id: int
    turn_id: int
    kind: TurnKind
    stream: StreamKind
    start: int
    stop: int
    natural_length: int


@dataclass(frozen=True)
class OriginalPosition:
    """Mapping of one supplied input token to its compiled physical copies."""

    original_index: int
    sample_id: int
    turn_id: int
    token_index: int
    merged_positions: tuple[int, ...]


@dataclass(frozen=True)
class CorruptionDraw:
    """One upstream-order draw; block positions are local to the padded turn.

    ``short_answer`` records the 60% branch uniform and has no block/selection.
    Other records include the ratio uniform, eligible positions, and the exact
    order returned by ``random.sample``. All candidates are selected even when
    the forced-full branch applies, preserving RNG consumption.
    """

    sample_id: int
    turn_id: int
    region: Literal["short_answer", "natural", "tail", "full"]
    block_id: int | None
    candidate_positions: tuple[int, ...]
    uniform: float
    mask_ratio: float | None
    selected_positions: tuple[int, ...]


@dataclass(frozen=True)
class DenoiseBatch:
    """A one-row packed CPU oracle batch; tensor contents remain mutable.

    All tensors are one-dimensional. Validity/corruption/supervision/clean
    flags are bool, IDs are int64, and weights are float32. Labels occupy the
    *same positions* as their corrupted inputs; there is no causal shift.
    """

    input_ids: torch.Tensor
    labels: torch.Tensor
    valid_mask: torch.Tensor
    sample_ids: torch.Tensor
    turn_ids: torch.Tensor
    type_ids: torch.Tensor
    block_ids: torch.Tensor
    position_ids: torch.Tensor
    is_clean: torch.Tensor
    corruption_mask: torch.Tensor
    supervision_mask: torch.Tensor
    loss_weights: torch.Tensor
    original_position_ids: torch.Tensor
    contract: TextTokenContract
    block_size: int
    seed: int
    rng_state_before: tuple[Any, ...]
    rng_state_after: tuple[Any, ...]
    rng_python_version: tuple[int, int, int]
    rng_trace: tuple[CorruptionDraw, ...]
    spans: tuple[BatchSpan, ...]
    original_to_merged: tuple[OriginalPosition, ...]
    schema_version: str = SCHEMA_VERSION
    schedule_version: str = SCHEDULE_VERSION

    def attention_allowed(self) -> torch.Tensor:
        """Return bool[S,S], where True permits query-row to key-column access.

        This is the pinned text-only graph, including restricted special-token
        queries. It is a reference oracle, not a fused attention implementation.
        Batch padding has no incoming or outgoing edges.
        """
        valid = self.valid_mask[:, None] & self.valid_mask[None, :]
        same_sample = self.sample_ids[:, None] == self.sample_ids[None, :]
        same_turn = self.turn_ids[:, None] == self.turn_ids[None, :]
        previous_turn = self.turn_ids[None, :] < self.turn_ids[:, None]
        clean_q, clean_k = self.is_clean[:, None], self.is_clean[None, :]
        text_q = self.type_ids[:, None] == 3
        text_k = self.type_ids[None, :] == 3
        special_k = self.type_ids[None, :] == 1
        earlier_block = self.block_ids[None, :] < self.block_ids[:, None]
        same_block = self.block_ids[None, :] == self.block_ids[:, None]
        prior_or_same_block = earlier_block | same_block

        noisy_text = (
            ~clean_q
            & text_q
            & (
                (~clean_k & same_turn & ((text_k & same_block) | (special_k & prior_or_same_block)))
                | (clean_k & previous_turn)
                | (clean_k & same_turn & text_k & earlier_block)
            )
        )
        noisy_special = (
            ~clean_q & ~text_q & ((~clean_k & same_turn & ~text_k) | (clean_k & previous_turn))
        )
        clean_text = (
            clean_q
            & text_q
            & clean_k
            & (previous_turn | (same_turn & (text_k | special_k) & prior_or_same_block))
        )
        clean_special = clean_q & ~text_q & clean_k & (previous_turn | (same_turn & ~text_k))
        return valid & same_sample & (noisy_text | noisy_special | clean_text | clean_special)


@dataclass
class _CompiledTurn:
    kind: TurnKind
    clean_ids: list[int]
    noisy_ids: list[int]
    labels: list[int]
    blocks: list[int]
    types: list[int]
    weights: list[float]
    origins: list[int]
    natural_length: int


def _core_blocks(tokens: Sequence[int], block_size: int, think_open: int) -> list[int]:
    # Segments separated by think-open each start a new text block. The marker
    # itself remains block zero and does not consume an ordinary text slot.
    blocks: list[int] = []
    next_block = 0
    segment_length = 0
    for token in tokens:
        if token == think_open:
            if segment_length:
                next_block += (segment_length + block_size - 1) // block_size
            segment_length = 0
            blocks.append(0)
        else:
            blocks.append(next_block + segment_length // block_size)
            segment_length += 1
    return blocks


def _prepare_turn(
    turn: TextTurn, contract: TextTokenContract, block_size: int, original_start: int
) -> _CompiledTurn:
    core = list(turn.token_ids)
    origins = list(range(original_start, original_start + len(core)))
    if turn.kind == "context":
        return _CompiledTurn(
            turn.kind,
            core,
            core.copy(),
            [IGNORE_INDEX] * len(core),
            [-1] * len(core),
            [1] * len(core),
            [1.0] * len(core),
            origins,
            len(core),
        )
    if turn.kind == "answer" and (not core or core[-1] != contract.eoa_token_id):
        core.append(contract.eoa_token_id)
        origins.append(-1)
    if (
        turn.kind == "think"
        and core[0] == contract.think_open_token_id
        and core[-1] != contract.think_close_token_id
    ):
        core.append(contract.think_close_token_id)
        origins.append(-1)

    natural_core_length = len(core)
    blocks = _core_blocks(core, block_size, contract.think_open_token_id)
    last_block = blocks[-1]
    last_count = sum(
        token != contract.think_open_token_id and block == last_block
        for token, block in zip(core, blocks, strict=True)
    )
    tail_length = (-last_count) % block_size
    tail_id = contract.eoa_token_id if turn.kind == "answer" else contract.think_close_token_id
    core.extend([tail_id] * tail_length)
    blocks.extend([last_block] * tail_length)
    origins.extend([-1] * tail_length)
    types = [1 if token == contract.think_open_token_id else 3 for token in core]
    weights = (
        [1.0] * natural_core_length + [1.0 / tail_length] * tail_length
        if tail_length
        else [1.0] * len(core)
    )
    if turn.add_boa:
        core.insert(0, contract.boa_token_id)
        blocks.insert(0, 0)
        types.insert(0, 1)
        weights.insert(0, 1.0)
        origins.insert(0, -1)
    return _CompiledTurn(
        turn.kind,
        core,
        core.copy(),
        [IGNORE_INDEX] * len(core),
        blocks,
        types,
        weights,
        origins,
        natural_core_length + int(turn.add_boa),
    )


def _corrupt_turn(
    turn: _CompiledTurn,
    *,
    sample_id: int,
    turn_id: int,
    contract: TextTokenContract,
    rng: random.Random,
    trace: list[CorruptionDraw],
    add_boa: bool,
) -> None:
    if turn.kind == "context":
        return

    def draw_region(
        start: int,
        stop: int,
        region: Literal["natural", "tail", "full"],
        force_short: bool,
    ) -> None:
        by_block: dict[int, list[int]] = {}
        for pos in range(start, stop):
            by_block.setdefault(turn.blocks[pos], []).append(pos)
        for block, positions in by_block.items():
            candidates = [
                pos for pos in positions if turn.clean_ids[pos] not in contract.never_mask_ids
            ]
            if not candidates:
                continue
            uniform = rng.uniform(0, 1)
            full = (force_short and len(candidates) <= 5) or uniform < 0.03
            ratio = 1.0 if full else math.cos(((uniform - 0.03) / 0.97) * math.pi / 2)
            selected = rng.sample(candidates, max(1, int(len(candidates) * ratio)))
            for pos in selected:
                turn.noisy_ids[pos] = contract.mask_token_id
                turn.labels[pos] = turn.clean_ids[pos]
            trace.append(
                CorruptionDraw(
                    sample_id,
                    turn_id,
                    region,
                    block,
                    tuple(candidates),
                    uniform,
                    ratio,
                    tuple(selected),
                )
            )

    # The terminal EOA is part of the natural corruption region but excluded
    # from the '<5 answer tokens' eligibility decision, as is optional BOA.
    answer_length = turn.natural_length - int(add_boa) - 1
    if turn.kind == "answer" and answer_length < 5:
        uniform = rng.random()
        trace.append(
            CorruptionDraw(sample_id, turn_id, "short_answer", None, (), uniform, None, ())
        )
        draw_region(0, turn.natural_length, "natural", uniform < 0.6)
        if turn.natural_length < len(turn.clean_ids):
            draw_region(turn.natural_length, len(turn.clean_ids), "tail", False)
    else:
        draw_region(0, len(turn.clean_ids), "full", False)


def compile_text_batch(
    samples: Sequence[DenoiseSample],
    *,
    contract: TextTokenContract,
    seed: int = 0,
    block_size: int = 32,
    max_positions: int = MAX_COMPILED_POSITIONS,
    pad_to_length: int | None = None,
) -> DenoiseBatch:
    """Compile explicit text turns into one packed, two-stream CPU batch.

    Each sample retains chronological context/noisy turns, followed by natural
    clean answer/think copies. Clean copies reuse the corresponding noisy
    position coordinates. Each sample restarts positions and turn IDs at zero.
    The cap includes every physical position, including optional batch padding;
    oversized inputs fail before corruption and are never truncated.
    """
    if not isinstance(contract, TextTokenContract):
        raise ValueError("contract must be a TextTokenContract")
    if not isinstance(samples, tuple | list) or not samples:
        raise ValueError("samples must be a nonempty list or tuple of DenoiseSample")
    if any(not isinstance(sample, DenoiseSample) for sample in samples):
        raise ValueError("samples must contain only DenoiseSample instances")
    _integer(seed, "seed")
    _integer(block_size, "block_size", minimum=1)
    _integer(max_positions, "max_positions", minimum=1)
    if block_size > MAX_COMPILED_POSITIONS or max_positions > MAX_COMPILED_POSITIONS:
        raise ValueError("block_size and max_positions must not exceed 768")
    if pad_to_length is not None:
        _integer(pad_to_length, "pad_to_length", minimum=1)
        if pad_to_length > max_positions:
            raise ValueError("pad_to_length exceeds max_positions")
    if len({sample.sample_id for sample in samples}) != len(samples):
        raise ValueError("sample IDs must be unique within a packed batch")

    prepared: list[list[_CompiledTurn]] = []
    original_coords: list[tuple[int, int, int]] = []
    expanded = 0
    for sample in samples:
        sample_turns = []
        for turn_id, turn in enumerate(sample.turns):
            if any(token >= contract.vocab_size for token in turn.token_ids):
                raise ValueError("input token IDs must be within vocab_size")
            if contract.mask_token_id in turn.token_ids:
                raise ValueError("supplied clean text cannot contain the corruption mask token")
            # Bound source input before constructing duplicated/padded lists.
            if len(turn.token_ids) > max_positions:
                raise ValueError("supplied turn exceeds max_positions before expansion")
            compiled = _prepare_turn(turn, contract, block_size, len(original_coords))
            original_coords.extend(
                (sample.sample_id, turn_id, i) for i in range(len(turn.token_ids))
            )
            expanded += len(compiled.clean_ids)
            if turn.kind != "context":
                expanded += compiled.natural_length
            if expanded > max_positions:
                raise ValueError(
                    f"expanded batch length {expanded} exceeds max_positions={max_positions}"
                )
            sample_turns.append(compiled)
        prepared.append(sample_turns)
    if pad_to_length is not None and pad_to_length < expanded:
        raise ValueError("pad_to_length cannot truncate the expanded batch")

    rng = random.Random(seed)
    before = rng.getstate()
    trace: list[CorruptionDraw] = []
    columns: dict[str, list[int | float | bool]] = {
        name: []
        for name in (
            "input_ids",
            "labels",
            "valid_mask",
            "sample_ids",
            "turn_ids",
            "type_ids",
            "block_ids",
            "position_ids",
            "is_clean",
            "corruption_mask",
            "supervision_mask",
            "loss_weights",
            "original_position_ids",
        )
    }
    spans: list[BatchSpan] = []
    merged: list[list[int]] = [[] for _ in original_coords]
    for sample, turns in zip(samples, prepared, strict=True):
        sample_start = len(columns["input_ids"])
        noisy_positions: dict[int, list[int]] = {}

        def append(
            turn_id: int,
            turn: _CompiledTurn,
            stream: StreamKind,
            sample_id: int = sample.sample_id,
            start_offset: int = sample_start,
            noisy_coords: dict[int, list[int]] = noisy_positions,
        ) -> None:
            start = len(columns["input_ids"])
            clean = stream != "noisy"
            count = turn.natural_length if stream == "clean" else len(turn.clean_ids)
            tokens = turn.clean_ids if clean else turn.noisy_ids
            positions = (
                noisy_coords[turn_id][:count]
                if stream == "clean"
                else list(range(start - start_offset, start - start_offset + count))
            )
            if stream == "noisy":
                noisy_coords[turn_id] = positions
            for i in range(count):
                label = IGNORE_INDEX if clean else turn.labels[i]
                supervised = label != IGNORE_INDEX
                values = {
                    "input_ids": tokens[i],
                    "labels": label,
                    "valid_mask": True,
                    "sample_ids": sample_id,
                    "turn_ids": turn_id,
                    "type_ids": turn.types[i],
                    "block_ids": turn.blocks[i],
                    "position_ids": positions[i],
                    "is_clean": clean,
                    "corruption_mask": supervised,
                    "supervision_mask": supervised,
                    "loss_weights": 1.0 if clean else turn.weights[i],
                    "original_position_ids": turn.origins[i],
                }
                for name, value in values.items():
                    columns[name].append(value)
                if turn.origins[i] >= 0:
                    merged[turn.origins[i]].append(start + i)
            spans.append(
                BatchSpan(
                    sample_id,
                    turn_id,
                    turn.kind,
                    stream,
                    start,
                    start + count,
                    turn.natural_length,
                )
            )

        for turn_id, turn in enumerate(turns):
            _corrupt_turn(
                turn,
                sample_id=sample.sample_id,
                turn_id=turn_id,
                contract=contract,
                rng=rng,
                trace=trace,
                add_boa=sample.turns[turn_id].add_boa,
            )
            append(turn_id, turn, "context" if turn.kind == "context" else "noisy")
        for turn_id, turn in enumerate(turns):
            if turn.kind != "context":
                append(turn_id, turn, "clean")

    for _ in range((pad_to_length or expanded) - expanded):
        for name in columns:
            value: int | float | bool
            if name == "input_ids":
                value = contract.pad_token_id
            elif name == "labels":
                value = IGNORE_INDEX
            elif name == "loss_weights":
                value = 0.0
            elif name in ("valid_mask", "is_clean", "corruption_mask", "supervision_mask"):
                value = False
            else:
                value = -1
            columns[name].append(value)
    boolean_names = {"valid_mask", "is_clean", "corruption_mask", "supervision_mask"}
    tensors = {
        name: torch.tensor(
            values,
            device="cpu",
            dtype=torch.bool
            if name in boolean_names
            else torch.float32
            if name == "loss_weights"
            else torch.long,
        )
        for name, values in columns.items()
    }
    mapping = tuple(
        OriginalPosition(i, sid, tid, token_index, tuple(merged[i]))
        for i, (sid, tid, token_index) in enumerate(original_coords)
    )
    return DenoiseBatch(
        input_ids=tensors["input_ids"],
        labels=tensors["labels"],
        valid_mask=tensors["valid_mask"],
        sample_ids=tensors["sample_ids"],
        turn_ids=tensors["turn_ids"],
        type_ids=tensors["type_ids"],
        block_ids=tensors["block_ids"],
        position_ids=tensors["position_ids"],
        is_clean=tensors["is_clean"],
        corruption_mask=tensors["corruption_mask"],
        supervision_mask=tensors["supervision_mask"],
        loss_weights=tensors["loss_weights"],
        original_position_ids=tensors["original_position_ids"],
        contract=contract,
        block_size=block_size,
        seed=seed,
        rng_state_before=before,
        rng_state_after=rng.getstate(),
        rng_python_version=(sys.version_info.major, sys.version_info.minor, sys.version_info.micro),
        rng_trace=tuple(trace),
        spans=tuple(spans),
        original_to_merged=mapping,
    )
