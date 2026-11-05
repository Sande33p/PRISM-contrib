"""Bind the U2 text tokenizer to an owned snapshot of verified local bytes.

Construction uses Tokenizer.from_str and the installed PreTrainedTokenizerFast
constructor, never AutoTokenizer, from_pretrained, a path, or artifact code.
The generic fast backend supports serialized Qwen2 pipelines; class-specific
Python tokenizers and unrecognized configuration require separate qualification.
No temporary files, hub/cache lookup, network operation or model is needed.

Call verify_binding immediately before handing tokenizer to the oracle. It
checks object/backend identity, serialized backend semantics, Python state and
class attributes. Backend padding/truncation caches (reset by explicit renderer
arguments) and HF warning bookkeeping are excluded. Arbitrary concurrent access,
hostile in-process Python/library patching and mutation during external use are
outside this boundary; callers must serialize use and verify again afterward.
Real tokenizer/default-prefix equivalence still requires Aurora qualification.
"""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import import_module
from importlib.metadata import version
from types import MappingProxyType
from typing import Any

from src.backends.u2_text_renderer import (
    EXPECTED_SPECIAL_TOKEN_IDS,
    RENDERER_REVISION,
    RenderedTextPrompt,
    TextTokenizer,
    render_text_prompt,
    tokenizer_digest,
    validate_tokenizer_contract,
)

_SPECIAL_KEYS = frozenset(
    {
        "bos_token",
        "eos_token",
        "unk_token",
        "sep_token",
        "pad_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
    }
)
# Reviewed Aurora metadata for the explicit generic fast-tokenizer class only.
# These exact legacy values are inert: remove them before library construction.
# In particular, trust_remote_code=True is NOT executable-code permission, and
# merges_file may only be null. Other values/classes require separate review.
_INERT_LEGACY_CONFIG = {
    "extra_special_tokens": {},
    "fast_tokenizer": True,
    "gmask_token": "[gMASK]",
    "merges_file": None,
    "trust_remote_code": True,
}
_CONFIG_KEYS = (
    _SPECIAL_KEYS
    | _INERT_LEGACY_CONFIG.keys()
    | {
        "tokenizer_class",
        "added_tokens_decoder",
        "model_max_length",
        "padding_side",
        "truncation_side",
        "clean_up_tokenization_spaces",
        "split_special_tokens",
        "chat_template",
        "model_input_names",
        "add_prefix_space",
        "add_bos_token",
        "add_eos_token",
    }
)
_CLASSES = {"PreTrainedTokenizerFast", "TokenizersBackend", "Qwen2Tokenizer", "Qwen2TokenizerFast"}


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(content: bytes, name: str) -> dict[str, Any]:
    try:
        value = json.loads(content.decode("utf-8"), object_pairs_hook=_pairs)
    except (UnicodeError, ValueError) as exc:
        raise ValueError(f"invalid tokenizer JSON: {name}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"tokenizer JSON must be an object: {name}")
    return value


def _added_token(value: Any) -> Any:
    AddedToken = import_module("tokenizers").AddedToken

    if value is None or isinstance(value, str):
        return value
    if not isinstance(value, dict) or not isinstance(value.get("content"), str):
        raise ValueError("special token must be a string or AddedToken record")
    flags = {"single_word", "lstrip", "rstrip", "normalized", "special"}
    if set(value) - flags - {"content", "__type"}:
        raise ValueError("unsupported AddedToken fields")
    if value.get("__type", "AddedToken") != "AddedToken":
        raise ValueError("unsupported token type")
    if any(type(value[key]) is not bool for key in flags & value.keys()):
        raise ValueError("AddedToken flags must be boolean")
    return AddedToken(**{key: val for key, val in value.items() if key != "__type"})


def _construct(artifacts: Mapping[str, bytes]) -> Any:
    """Construct only from data; no injectable public constructor or object."""
    Tokenizer = import_module("tokenizers").Tokenizer
    PreTrainedTokenizerFast = import_module("transformers").PreTrainedTokenizerFast

    # Parse every admitted file strictly, including the Rust backend serialization.
    parsed = {name: _json(content, name) for name, content in artifacts.items()}
    config = dict(parsed["tokenizer_config.json"])
    if set(config) - _CONFIG_KEYS:
        raise ValueError("unsupported tokenizer configuration fields")
    for key, expected in _INERT_LEGACY_CONFIG.items():
        if key not in config:
            continue
        if config.get("tokenizer_class") != "PreTrainedTokenizerFast":
            raise ValueError("unsupported legacy metadata tokenizer class")
        value = config.pop(key)
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f"unsupported legacy tokenizer metadata: {key}")
    if config.pop("tokenizer_class", "PreTrainedTokenizerFast") not in _CLASSES:
        raise ValueError("unsupported tokenizer class")
    for key in ("add_bos_token", "add_eos_token", "add_prefix_space", "split_special_tokens"):
        if key in config and config[key] is not False:
            raise ValueError(f"unsupported tokenizer behavior: {key}")
    if "model_max_length" in config and (
        type(config["model_max_length"]) is not int or config["model_max_length"] < 4096
    ):
        raise ValueError("model_max_length must admit the 4096-token prefix")
    for key in ("padding_side", "truncation_side"):
        if key in config and config[key] != "right":
            raise ValueError(f"unsupported tokenizer behavior: {key}")
    specials = parsed.get("special_tokens_map.json", {})
    if set(specials) - _SPECIAL_KEYS:
        raise ValueError("unsupported special_tokens_map fields")
    for key, value in specials.items():
        if key in config and _state(_special_value(key, config[key])) != _state(
            _special_value(key, value)
        ):
            raise ValueError(f"conflicting special token configuration: {key}")
        config[key] = value
    for key in _SPECIAL_KEYS & config.keys():
        config[key] = _special_value(key, config[key])
    decoder = config.get("added_tokens_decoder", {})
    if not isinstance(decoder, dict):
        raise ValueError("added_tokens_decoder must be a mapping")
    converted = {}
    for key, value in decoder.items():
        if (
            not isinstance(key, str)
            or not key.isascii()
            or not key.isdecimal()
            or str(int(key)) != key
        ):
            raise ValueError("added token IDs must be canonical nonnegative integers")
        converted[int(key)] = _added_token(value)
        if converted[int(key)] is None:
            raise ValueError("added tokens cannot be null")
    config["added_tokens_decoder"] = converted
    backend = Tokenizer.from_str(artifacts["tokenizer.json"].decode("utf-8"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, local_files_only=True, **config)
    vocab = tokenizer.get_vocab()
    declared = parsed.get("added_tokens.json", {})
    for token, token_id in declared.items():
        if type(token_id) is not int or token_id < 0:
            raise ValueError("added_tokens.json requires nonnegative integer IDs")
        # A legacy sidecar may confirm serialized tokens, never silently add them
        # in an order that changes IDs or invents missing AddedToken attributes.
        if vocab.get(token) != token_id:
            raise ValueError("added_tokens.json disagrees with constructed vocabulary")
    for token_id, token in converted.items():
        if vocab.get(str(token)) != token_id:
            raise ValueError("added_tokens_decoder disagrees with constructed vocabulary")
    if set(vocab.values()) != set(range(len(tokenizer))):
        raise ValueError("tokenizer vocabulary must have dense unique IDs")
    return tokenizer


def _special_value(key: str, value: Any) -> Any:
    if key == "additional_special_tokens":
        if not isinstance(value, list) or any(item is None for item in value):
            raise ValueError("additional_special_tokens must be a token list")
        return [_added_token(item) for item in value]
    return _added_token(value)


def _state(value: Any) -> Any:
    """Serialize only data types created by the trusted local constructor."""
    AddedToken = import_module("tokenizers").AddedToken

    if value is None or type(value) in (str, bool, int, float):
        return (type(value).__name__, value)
    if type(value) is AddedToken:
        return (
            "AddedToken",
            {
                key: getattr(value, key)
                for key in ("content", "single_word", "lstrip", "rstrip", "normalized", "special")
            },
        )
    if type(value) in (list, tuple):
        return (type(value).__name__, [_state(item) for item in value])
    if type(value) is dict:
        return (
            "dict",
            sorted(
                [(_state(key), _state(item)) for key, item in value.items()],
                key=lambda pair: json.dumps(pair[0], sort_keys=True),
            ),
        )
    raise ValueError("unsupported tokenizer state or object mutation")


def _fingerprint(tokenizer: Any) -> str:
    backend = _json(tokenizer.backend_tokenizer.to_str().encode(), "constructed backend")
    # HF rewrites these caches on every call using the explicit call arguments.
    backend.pop("padding", None)
    backend.pop("truncation", None)
    state = {
        key: value
        for key, value in vars(tokenizer).items()
        if key not in {"_tokenizer", "deprecation_warnings"}
    }
    encoded = json.dumps([backend, _state(state)], sort_keys=True, allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _class_state(tokenizer: Any) -> tuple[tuple[type, str, int], ...]:
    return tuple(
        (cls, name, id(value))
        for cls in type(tokenizer).__mro__
        for name, value in sorted(vars(cls).items())
    )


@dataclass(frozen=True, slots=True, init=False)
class BoundTextTokenizer:
    """Factory-only binding with immutable provenance and checked object state.

    The underlying HF object is intentionally shared with the oracle. A frozen
    wrapper does not freeze that object: verify_binding detects unsupported
    changes, including token additions, normalizer/postprocessor changes,
    special maps, truncation side, methods and backend/object replacement.
    """

    tokenizer_digest: str
    renderer_revision: str
    vocab_size: int
    artifact_hashes: tuple[tuple[str, str], ...]
    special_token_ids: Mapping[str, int]
    construction_provenance: Mapping[str, str]
    _artifacts: Mapping[str, bytes]
    _tokenizer: Any
    _object_id: int
    _backend_id: int
    _fingerprint: str
    _classes: tuple[tuple[type, str, int], ...]

    def __init__(self) -> None:
        raise TypeError("use bind_local_tokenizer with verified artifact bytes")

    @property
    def tokenizer(self) -> TextTokenizer:
        """Return the actual constructed object after checking the binding."""
        self.verify_binding()
        return self._tokenizer

    def verify_binding(self) -> None:
        """Fail closed if the certified construction or object state changed."""
        try:
            valid = (
                id(self._tokenizer) == self._object_id
                and id(self._tokenizer.backend_tokenizer) == self._backend_id
                and _class_state(self._tokenizer) == self._classes
                and _fingerprint(self._tokenizer) == self._fingerprint
                and tokenizer_digest(self._artifacts) == self.tokenizer_digest
                and len(self._tokenizer) == self.vocab_size
            )
        except Exception as exc:
            raise ValueError("tokenizer binding was invalidated") from exc
        if not valid:
            raise ValueError("tokenizer binding was invalidated")

    def render(self, *, system_prompt: str, user_text: str) -> RenderedTextPrompt:
        """Render the existing first-turn prefix, preserving BOA ownership."""
        self.verify_binding()
        try:
            return render_text_prompt(
                self._tokenizer,
                system_prompt=system_prompt,
                user_text=user_text,
                tokenizer_artifacts=self._artifacts,
            )
        finally:
            self.verify_binding()


def bind_local_tokenizer(
    artifacts: Mapping[str, bytes], *, expected_digest: str
) -> BoundTextTokenizer:
    """Verify exact bytes before constructing and certifying a local tokenizer.

    The caller must obtain expected_digest independently from an admitted asset
    manifest. Self-hashing an untrusted input is not provenance verification.
    Unknown files/configuration, incomplete bytes and incompatible IDs fail
    closed. Dependencies are imported lazily, after digest verification.
    """
    if not isinstance(artifacts, Mapping):
        raise ValueError("tokenizer artifacts must be a mapping")
    snapshot = MappingProxyType(dict(artifacts))
    if any(
        type(name) is not str or type(content) is not bytes for name, content in snapshot.items()
    ):
        raise ValueError("tokenizer artifacts require plain str names and immutable bytes")
    digest = tokenizer_digest(snapshot)
    if type(expected_digest) is not str or digest != expected_digest:
        raise ValueError("tokenizer artifact digest does not match expected_digest")
    try:
        tokenizer = _construct(snapshot)
        validate_tokenizer_contract(tokenizer)
        fingerprint = _fingerprint(tokenizer)
    except Exception as exc:
        raise ValueError(f"cannot bind local tokenizer: {exc}") from exc
    bound = object.__new__(BoundTextTokenizer)
    fields = {
        "tokenizer_digest": digest,
        "renderer_revision": RENDERER_REVISION,
        "vocab_size": len(tokenizer),
        "artifact_hashes": tuple(
            sorted(
                (name, hashlib.sha256(content).hexdigest()) for name, content in snapshot.items()
            )
        ),
        "special_token_ids": MappingProxyType(dict(EXPECTED_SPECIAL_TOKEN_IDS)),
        "construction_provenance": MappingProxyType(
            {
                "constructor": "tokenizers.Tokenizer.from_str -> transformers.PreTrainedTokenizerFast",
                "tokenizers_version": version("tokenizers"),
                "transformers_version": version("transformers"),
                "constructed_class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
                "object_state_sha256": fingerprint,
                "artifact_digest": digest,
                "network": "no path, hub, cache or from_pretrained calls; local_files_only=True",
            }
        ),
        "_artifacts": snapshot,
        "_tokenizer": tokenizer,
        "_object_id": id(tokenizer),
        "_backend_id": id(tokenizer.backend_tokenizer),
        "_fingerprint": fingerprint,
        "_classes": _class_state(tokenizer),
    }
    for name, value in fields.items():
        object.__setattr__(bound, name, value)
    bound.verify_binding()
    return bound
