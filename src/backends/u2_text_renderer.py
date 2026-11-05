"""Text-only U2 SFT prefixes, with no model imports or asset loading.

Source: InternLM/InternLumina-U2 at 70f21c1df8fa26e514df26ba31bc061be8d3d018,
infer_mmu_sft.py:113-140 and utils/tokenizer_contract.py (Apache-2.0).
The driver owns BOA insertion and generated answer/think/EOA tokens. This
renderer supports one explicit system/user turn, without media or chat history.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

SOURCE_REVISION = "70f21c1df8fa26e514df26ba31bc061be8d3d018"
RENDERER_REVISION = f"u2-text-sft-v1:{SOURCE_REVISION}"
MAX_PROMPT_TOKENS = 4096
REQUIRED_TOKENIZER_ARTIFACTS = ("tokenizer.json", "tokenizer_config.json")
OPTIONAL_TOKENIZER_ARTIFACTS = ("special_tokens_map.json", "added_tokens.json")
TOKENIZER_ARTIFACT_NAMES = frozenset(REQUIRED_TOKENIZER_ARTIFACTS + OPTIONAL_TOKENIZER_ARTIFACTS)
EXPECTED_SPECIAL_TOKEN_IDS: Mapping[str, int] = MappingProxyType(
    {
        "<|mask|>": 156895,
        "<system>": 157153,
        "</system>": 157154,
        "<user>": 157155,
        "<uncondition>": 157156,
        "</user>": 157157,
        "<3D_CONTEXT>": 157158,
        "<IMG_CONTEXT>": 157160,
        "<VIDEO_CONTEXT>": 157161,
        "<answer>": 157165,
        "</answer>": 157166,
        "<BOI>": 157167,
        "<EOI>": 157168,
        "<BO3d>": 157169,
        "<EO3d>": 157170,
        "<BOV>": 157171,
        "<EOV>": 157172,
    }
)
_INPUT_MARKERS = (*EXPECTED_SPECIAL_TOKEN_IDS, "<think>", "</think>")
_FORBIDDEN_PROMPT_IDS = frozenset(
    value
    for token, value in EXPECTED_SPECIAL_TOKEN_IDS.items()
    if token not in {"<system>", "</system>", "<user>", "</user>"}
)


class TokenizedText(Protocol):
    """Minimal scalar tokenizer output; tensors and batches are unsupported."""

    @property
    def input_ids(self) -> list[int]:
        """Return token IDs for one string."""
        ...


class TextTokenizer(Protocol):
    """The local tokenizer operations needed by the rendering boundary."""

    def __len__(self) -> int: ...

    def convert_tokens_to_ids(self, token: str) -> int:
        """Look up a token without encoding a conversation."""
        ...

    def __call__(
        self,
        text: str,
        *,
        truncation: bool,
        max_length: int,
        padding: bool,
        **kwargs: bool,
    ) -> TokenizedText:
        """Encode a scalar text string."""
        ...


@dataclass(frozen=True, slots=True)
class RenderedTextPrompt:
    """Immutable prefix identity for TextRequest; prompt_ids exclude BOA."""

    prompt_ids: tuple[int, ...]
    tokenizer_digest: str
    renderer_revision: str


def tokenizer_digest(artifacts: Mapping[str, bytes]) -> str:
    """Hash named, exact tokenizer bytes, independent of mapping iteration order.

    The admitted set is tokenizer.json, tokenizer_config.json and optional
    special_tokens_map.json/added_tokens.json. Unknown names are rejected,
    never silently omitted. No files are loaded.
    This is content identity, not proof that the supplied tokenizer was loaded
    from those bytes; the runtime/preflight owner must establish that binding.
    """
    if not isinstance(artifacts, Mapping) or not set(REQUIRED_TOKENIZER_ARTIFACTS).issubset(
        artifacts
    ):
        raise ValueError("tokenizer.json and tokenizer_config.json bytes are required")
    entries = []
    for name, content in artifacts.items():
        if (
            not isinstance(name, str)
            or not name
            or name in {".", ".."}
            or "/" in name
            or "\\" in name
            or name not in TOKENIZER_ARTIFACT_NAMES
            or not isinstance(content, bytes)
            or not content
        ):
            raise ValueError("tokenizer artifacts require basenames and nonempty bytes")
        entries.append((name, hashlib.sha256(content).hexdigest()))
    encoded = json.dumps(sorted(entries), ensure_ascii=True, separators=(",", ":")).encode()
    return "sha256:" + hashlib.sha256(b"u2-tokenizer-artifacts-v1\0" + encoded).hexdigest()


def _ids(encoded: TokenizedText, vocab_size: int) -> tuple[int, ...]:
    values = getattr(encoded, "input_ids", None)
    if not isinstance(values, list | tuple) or not 0 < len(values) <= MAX_PROMPT_TOKENS:
        raise ValueError("tokenizer must return 1..4096 scalar input_ids")
    if any(type(value) is not int or not 0 <= value < vocab_size for value in values):
        raise ValueError("tokenizer returned invalid or out-of-vocabulary input_ids")
    return tuple(values)


def validate_tokenizer_contract(tokenizer: TextTokenizer) -> None:
    """Check all pinned IDs and their atomic encoding before rendering.

    Think markers have no numeric IDs in the pinned contract. No IDs are
    invented for them, and this module neither inserts nor decodes them.
    """
    vocab_size = len(tokenizer)
    if vocab_size <= max(EXPECTED_SPECIAL_TOKEN_IDS.values()):
        raise ValueError("tokenizer vocabulary cannot contain the pinned special IDs")
    for token, expected in EXPECTED_SPECIAL_TOKEN_IDS.items():
        actual = tokenizer.convert_tokens_to_ids(token)
        if type(actual) is not int or actual != expected:
            raise ValueError(f"incompatible tokenizer: {token} expected {expected}, got {actual}")
        encoded = tokenizer(
            token,
            truncation=False,
            max_length=MAX_PROMPT_TOKENS,
            padding=False,
            add_special_tokens=False,
        )
        if _ids(encoded, vocab_size) != (expected,):
            raise ValueError(f"special token {token} must encode atomically as {expected}")


def render_text_prompt(
    tokenizer: TextTokenizer,
    *,
    system_prompt: str,
    user_text: str,
    tokenizer_artifacts: Mapping[str, bytes],
) -> RenderedTextPrompt:
    """Render the pinned first-user text path, without appending answer tokens.

    Text is preserved verbatim, including empty strings and whitespace. Reserved
    control/media/answer/think markers in either input are rejected. The source
    tokenizer defaults and 4096-token truncation are retained; truncation may
    remove the closing user marker. No HF chat template or BOS/EOS override is
    applied. Media and multi-turn requests require a separately qualified path.
    """
    for name, value in (("system_prompt", system_prompt), ("user_text", user_text)):
        if not isinstance(value, str):
            raise ValueError(f"{name} must be a string")
        if any(marker in value for marker in _INPUT_MARKERS):
            raise ValueError(f"{name} contains a reserved U2 marker")
    digest = tokenizer_digest(tokenizer_artifacts)
    validate_tokenizer_contract(tokenizer)
    wrapped = f"<system>{system_prompt}</system><user>{user_text}</user>"
    encoded = tokenizer(wrapped, truncation=True, max_length=MAX_PROMPT_TOKENS, padding=False)
    ids = _ids(encoded, len(tokenizer))
    if _FORBIDDEN_PROMPT_IDS.intersection(ids):
        raise ValueError("text prefix contains an unsupported control, answer, or media ID")
    return RenderedTextPrompt(ids, digest, RENDERER_REVISION)
