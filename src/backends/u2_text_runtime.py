"""Injected, greedy text block denoising; no loader, downloads or global selector.

Adapted from InternLM/InternLumina-U2 (Apache-2.0), revision
70f21c1df8fa26e514df26ba31bc061be8d3d018:
generators/text_understanding_generator.py::generate_text_bd and
build_text_attention_mask_4d_from_meta; utils/generation_utils.py's transfer
quotas and temperature-zero branch. The text-only mask omits visual rules.
Every forward is separately controlled; only completed blocks are committed.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from types import MappingProxyType
from typing import Protocol

import torch
from torch import Tensor

from .text_contracts import RuntimeIdentity, TextRequest
from .u2_preflight import U2PreflightReport
from .u2_reference import ExecutionControl
from .u2_text_renderer import EXPECTED_SPECIAL_TOKEN_IDS, SOURCE_REVISION

BOA = 157165
EOA = 157166
MASK = 156895
_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


@dataclass(frozen=True)
class LocalModelSpec:
    """Caller-declared binding of one local model to inspected metadata.

    This is not proof of checkpoint loading or artifact-to-object provenance.
    The caller owns that qualification and must keep the model unchanged while
    in use. Synthetic fixtures must identify themselves as such in runtime_id.
    The explicit model boundary must use split layout and eight text channels.
    """

    identity: RuntimeIdentity
    special_token_ids: Mapping[str, int]
    dtype: torch.dtype
    device: torch.device
    max_position_embeddings: int
    num_codebooks: int = 8
    layout: str = "split"
    prompt_vocab_size: int | None = None

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "special_token_ids", MappingProxyType(dict(self.special_token_ids))
        )


class LocalTextModel(Protocol):
    """Explicit local callable; no HF tuple/.logits guessing or hidden loader.

    spec describes the actual bound model; eval() must disable training. Return
    a dense floating [current_block_length, vocab_size] tensor on spec.device
    in spec.dtype, covering *all* noisy type-3 rows, including revealed rows.
    Implementations must not mutate inputs or perform hidden extra forwards.
    """

    @property
    def spec(self) -> LocalModelSpec:
        """Return the immutable declaration for this bound local model."""
        ...

    @property
    def training(self) -> bool:
        """Report whether the underlying model is in training mode."""
        ...

    def eval(self) -> object:
        """Put the underlying model in evaluation mode without loading assets."""
        ...

    def __call__(
        self,
        input_ids: Tensor,
        type_position_ids: Tensor,
        *,
        attention_mask: Tensor,
        infer_mmu: bool,
        is_clean: Tensor,
        position_ids: Tensor,
    ) -> Tensor:
        """Run exactly one forward using the pinned positional/keyword boundary."""
        ...


def _text_attention_mask(
    types: Tensor, turns: Tensor, clean: Tensor, blocks: Tensor, dtype: torch.dtype
) -> Tensor:
    """Pinned metadata visibility restricted to the admitted type-1/type-3 path."""
    type_q, type_k = types[:, None], types[None, :]
    turn_q, turn_k = turns[:, None], turns[None, :]
    clean_q, clean_k = clean[:, None].bool(), clean[None, :].bool()
    block_q, block_k = blocks[:, None], blocks[None, :]
    same_turn, prev_turn = turn_q == turn_k, turn_k < turn_q
    text_q, text_k = type_q == 3, type_k == 3
    special_k = type_k == 1
    noisy_text = (~clean_q) & text_q
    allowed = (
        (
            noisy_text
            & (~clean_k)
            & same_turn
            & ((text_k & (block_q == block_k)) | (special_k & (block_k <= block_q)))
        )
        | (noisy_text & clean_k & prev_turn)
        | (noisy_text & clean_k & same_turn & text_k & (block_k < block_q))
        | ((~clean_q) & (~text_q) & (((~clean_k) & same_turn & (~text_k)) | (clean_k & prev_turn)))
        | (
            clean_q
            & text_q
            & clean_k
            & (prev_turn | (same_turn & (text_k | special_k) & (block_k <= block_q)))
        )
        | (clean_q & (~text_q) & clean_k & (prev_turn | (same_turn & (~text_k))))
    )
    mask = torch.zeros((1, 1, types.numel(), types.numel()), device=types.device, dtype=dtype)
    return mask.masked_fill_(~allowed[None, None, :, :], torch.finfo(dtype).min)


class U2TextRuntime:
    """Source-backed T=0, cfg=0, low-confidence text loop for explicit injection.

    A successful preflight is metadata evidence only. Construction compares the
    supplied model's full special-token declaration, source/tokenizer/vocabulary
    identity, dtype, layout and capacity with that evidence. It never loads or
    qualifies weights. Request seed is retained by the adapter; this greedy
    runtime draws no randomness and does not alter global RNG state.
    """

    def __init__(self, model: LocalTextModel, *, preflight: U2PreflightReport) -> None:
        if not isinstance(preflight, U2PreflightReport):
            raise ValueError("an explicit U2 preflight report is required")
        preflight.require_metadata()
        spec = model.spec
        if not isinstance(spec, LocalModelSpec) or not isinstance(spec.identity, RuntimeIdentity):
            raise ValueError("an explicit local model specification is required")
        identity = spec.identity
        if (
            not preflight.source_verified
            or preflight.source_revision != SOURCE_REVISION
            or identity.source_revision != SOURCE_REVISION
            or identity.tokenizer_digest != preflight.tokenizer_digest
            or not identity.tokenizer_digest.startswith("sha256:")
            or len(identity.tokenizer_digest) != 71
            or any(c not in "0123456789abcdef" for c in identity.tokenizer_digest[7:])
            or identity.vocab_size != preflight.vocab_size
            or identity.vocab_size <= max(EXPECTED_SPECIAL_TOKEN_IDS.values())
            or identity.mask_token_id != MASK
            or identity.eoa_token_id != EOA
            or dict(spec.special_token_ids) != dict(EXPECTED_SPECIAL_TOKEN_IDS)
            or any(type(value) is not int for value in spec.special_token_ids.values())
        ):
            raise ValueError("incompatible source, tokenizer, vocabulary or special-token identity")
        if (
            type(spec.max_position_embeddings) is not int
            or spec.max_position_embeddings <= 0
            or spec.max_position_embeddings != preflight.max_position_embeddings
            or type(spec.num_codebooks) is not int
            or spec.num_codebooks != 8
            or spec.layout != "split"
            or preflight.split_layout != "split_candidate"
            or spec.dtype not in _DTYPES.values()
            or spec.dtype != _DTYPES.get(preflight.dtype or "")
            or not isinstance(spec.device, torch.device)
            or spec.device.type not in ("cpu", "cuda", "xpu")
            or (spec.device.type == "cpu" and spec.device.index is not None)
            or (spec.device.type in ("cuda", "xpu") and spec.device.index is None)
            or not callable(model)
            or (
                spec.prompt_vocab_size is not None
                and (
                    type(spec.prompt_vocab_size) is not int
                    or not max(EXPECTED_SPECIAL_TOKEN_IDS.values())
                    < spec.prompt_vocab_size
                    <= identity.vocab_size
                    or spec.prompt_vocab_size != preflight.tokenizer_vocab_size
                )
            )
        ):
            raise ValueError("incompatible local model layout, dtype, device or context capacity")
        self._model = model
        self._spec = spec
        self.identity = identity
        model.eval()
        self._check_binding()

    def _check_binding(self) -> None:
        if (
            self._model.spec != self._spec
            or self.identity != self._spec.identity
            or self._model.training is not False
        ):
            raise ValueError("local model binding changed or model is not in eval mode")

    def _logits(self, value: object, block_length: int) -> Tensor:
        if (
            not isinstance(value, Tensor)
            or value.layout != torch.strided
            or value.shape != (block_length, self.identity.vocab_size)
            or value.dtype != self._spec.dtype
            or value.device != self._spec.device
            or not bool(torch.isfinite(value).all())
        ):
            raise ValueError("invalid current-block model logits")
        return value

    def generate(self, request: TextRequest, control: ExecutionControl) -> str:
        """Run every scheduled forward; propagate control stops and runtime errors.

        Admission reserves prefix + BOA + the complete requested output against
        model capacity, even if EOA or a call budget might stop generation early.
        Invalid returned logits fail after that returned operation is counted.
        """
        self._check_binding()
        if request.tokenizer_digest != self.identity.tokenizer_digest or any(
            token >= (self._spec.prompt_vocab_size or self.identity.vocab_size)
            for token in request.prompt_ids
        ):
            raise ValueError("request does not match the bound tokenizer")
        if BOA in request.prompt_ids or EOA in request.prompt_ids or MASK in request.prompt_ids:
            raise ValueError("prompt must precede BOA and contain no answer or mask IDs")
        prefix_length = len(request.prompt_ids)
        if prefix_length + 1 + request.max_new_tokens > self._spec.max_position_embeddings:
            raise ValueError("requested sequence exceeds inspected context capacity")
        device, dtype = self._spec.device, self._spec.dtype
        decoded: list[int] = []
        with torch.inference_mode():
            for offset in range(0, request.max_new_tokens, request.block_size):
                control.check_can_continue()
                n = min(request.block_size, request.max_new_tokens - offset)
                code_start = prefix_length + 1 + offset
                sequence = [*request.prompt_ids, BOA, *decoded, *([MASK] * n)]
                length = len(sequence)
                x = torch.tensor(sequence, dtype=torch.long, device=device)[None, :, None].repeat(
                    1, 1, 8
                )
                types = torch.tensor(
                    [1] * (prefix_length + 1) + [3] * (offset + n), dtype=torch.long, device=device
                )
                turns = torch.tensor(
                    [0] * prefix_length + [1] * (1 + offset + n), dtype=torch.long, device=device
                )
                blocks = torch.tensor(
                    [-1] * prefix_length
                    + [0]
                    + [j // request.block_size for j in range(offset + n)],
                    dtype=torch.long,
                    device=device,
                )
                clean = torch.tensor(
                    [1] * prefix_length + [0] + [1] * offset + [0] * n,
                    dtype=torch.long,
                    device=device,
                )
                positions = torch.arange(length, dtype=torch.long, device=device)[None, :]
                attention = _text_attention_mask(types, turns, clean, blocks, dtype)
                for step in range(request.steps_per_block):
                    self._check_binding()
                    # Exactly one local forward per operation, including quota=0.
                    result = control.run_model_call(
                        length,
                        partial(
                            self._model,
                            x,
                            types[None, :],
                            attention_mask=attention,
                            infer_mmu=True,
                            is_clean=clean[None, :],
                            position_ids=positions,
                        ),
                    )
                    logits = self._logits(result, n)
                    quota = n // request.steps_per_block + int(step < n % request.steps_per_block)
                    if quota == 0:
                        continue
                    candidates = logits.argmax(dim=-1)
                    probabilities = torch.softmax(logits.to(torch.float64), dim=-1)
                    confidence = probabilities.gather(1, candidates[:, None]).squeeze(1)
                    current = x[0, code_start:, 0]
                    confidence = torch.where(current == MASK, confidence, -torch.inf)
                    selected = torch.topk(confidence, k=quota).indices
                    # Already revealed tokens are stable, even if later logits differ.
                    predictions = torch.where(current == MASK, candidates, current)
                    x[0, code_start + selected, :] = predictions[selected, None].expand(-1, 8)
                block: list[int] = x[0, code_start:, 0].tolist()
                if MASK in block:
                    raise ValueError("completed block contains unresolved mask IDs")
                if EOA in block:
                    control.commit(block[: block.index(EOA) + 1])
                    return "eoa"
                control.commit(block)
                decoded.extend(block)
        return "length"
