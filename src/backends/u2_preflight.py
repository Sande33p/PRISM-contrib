"""Local U2 metadata inspection; never import checkpoint code or load tensors.

Source semantics: InternLumina-U2 revision 70f21c1df8fa26e514df26ba31bc061be8d3d018,
README.md (split-only inference) and utils/tokenizer_contract.py (special IDs).
Index key patterns provide a split *candidate*, not proof of tensor layout.
Strict key/shape/dtype loading and direct-upstream parity remain separate gates.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Literal

from .u2_text_renderer import (
    OPTIONAL_TOKENIZER_ARTIFACTS,
    REQUIRED_TOKENIZER_ARTIFACTS,
    TOKENIZER_ARTIFACT_NAMES,
    tokenizer_digest,
)

SOURCE_REVISION = "70f21c1df8fa26e514df26ba31bc061be8d3d018"
SOURCE_HASHES = {
    "infer_mmu_sft.py": "82779ccc3107ef2d604b1dff2765a7b4536eaf881665da66cfe6a29bf086cc35",
    "generators/text_understanding_generator.py": "e9e9bc3d6f59c7baac3a740c37b4992de796c1f91dba7b1ef2a769d2befeec8a",
    "utils/generation_utils.py": "bfe97c1dec4cbbd1cef1c52ea0d115655130cb7004305cc5eff8feeb7a7a5d76",
    "utils/tokenizer_contract.py": "cf0890cb1dc0fc47e794bc350ca93fb8bb51a367eac2cf4bff02775893218916",
    "LICENSE": "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30",
    "README.md": "a6906235946ff0d342f27fa8339d95598e47ca2031386ca22f412971539adc44",
}
SPECIAL_TOKEN_IDS = {
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
_CHECKPOINT_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "model.safetensors.index.json",
    "configuration_internluminau2.py",
    "modeling_internluminau2.py",
    "tokenbridge.py",
    "parallel_plan.py",
)
EXECUTION_REQUIREMENTS = (
    "strict checkpoint key, shape and tensor dtype validation by an admitted loader",
    "verified split tensor layout and complete architecture configuration",
    "tokenizer encode/decode contract and direct-upstream parity",
    "authorized platform, allocation, resource ceilings and qualified runtime",
)


@dataclass(frozen=True)
class U2PreflightRequirements:
    """Expected identities supplied by the admitting caller, never inferred.

    tokenizer_digest uses the renderer's sha256: identity: SHA256 of
    b"u2-tokenizer-artifacts-v1\\0" plus canonical JSON of sorted filename/hash
    pairs. Includes tokenizer.json, tokenizer_config.json and any present
    special_tokens_map.json/added_tokens.json. Per-file artifact_hashes and
    FileEvidence.sha256 remain bare lowercase 64-hex SHA256 values.
    config_key selects an explicit nested text config; None uses the root object.
    artifact_hashes optionally pins checkpoint-relative files, including shards.
    """

    tokenizer_digest: str
    model_type: str
    dtype: str = "bfloat16"
    config_key: str | None = None
    artifact_hashes: Mapping[str, str] | None = None
    required_packages: tuple[str, ...] = ("torch", "transformers", "safetensors")


@dataclass(frozen=True)
class FileEvidence:
    """Hash observed on disk, with optional comparison to an expected digest."""

    area: str
    path: str
    sha256: str
    size_bytes: int
    matches_expected: bool | None


@dataclass(frozen=True)
class U2PreflightReport:
    """Metadata result; metadata_complete never grants execution permission."""

    source_revision: str
    source_verified: bool
    files: tuple[FileEvidence, ...]
    required_artifacts: tuple[str, ...]
    available_artifacts: tuple[str, ...]
    missing: tuple[str, ...]
    incompatible: tuple[str, ...]
    tokenizer_digest: str | None
    vocab_size: int | None
    max_position_embeddings: int | None
    dtype: str | None
    split_layout: Literal["split_candidate", "packed", "unknown"]
    package_versions: tuple[tuple[str, str | None], ...]
    metadata_complete: bool
    execution_qualified: Literal[False] = False
    execution_requirements: tuple[str, ...] = EXECUTION_REQUIREMENTS

    def require_metadata(self) -> None:
        """Raise with actionable failures; success still does not qualify execution."""
        if not self.metadata_complete:
            raise ValueError("U2 preflight failed: " + "; ".join(self.missing + self.incompatible))


def _local(root: Path, name: str) -> Path:
    if not name or "\\" in name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError(f"unsafe relative path: {name!r}")
    result = (root / name).resolve()
    if not result.is_relative_to(root):
        raise ValueError(f"path escapes artifact root: {name!r}")
    return result


def _pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def _metadata_bytes(path: Path) -> bytes:
    with path.open("rb") as stream:
        content = stream.read(64 * 1024 * 1024 + 1)
    if len(content) > 64 * 1024 * 1024:
        raise ValueError("metadata exceeds 64 MiB inspection limit")
    return content


def _json(path: Path) -> object:
    return json.loads(
        _metadata_bytes(path).decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant
    )


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _positive(value: object, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _digest(value: str) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"[0-9a-f]{64}", value))


def _tokenizer_identity(value: str) -> bool:
    return isinstance(value, str) and bool(re.fullmatch(r"sha256:[0-9a-f]{64}", value))


def preflight_u2(
    source_root: Path | str,
    checkpoint_root: Path | str,
    requirements: U2PreflightRequirements,
) -> U2PreflightReport:
    """Inspect local regular files and installed distribution metadata only.

    Missing or incompatible artifacts accumulate in the returned report. Unsafe
    paths, malformed JSON and identities fail closed. Shards are streamed for
    hashing, never deserialized; no filesystem writes or network operations occur.
    Only indexed safetensors and JSON tokenizers are supported by this inspector.
    """
    source = Path(source_root).resolve()
    checkpoint = Path(checkpoint_root).resolve()
    files: list[FileEvidence] = []
    required: set[str] = set()
    available: set[str] = set()
    missing: list[str] = []
    incompatible: list[str] = []
    inspected: dict[tuple[str, str], Path | None] = {}
    tokenizer_artifacts: dict[str, bytes] = {}

    def inspect(area: str, name: str, expected: str | None = None) -> Path | None:
        label = f"{area}/{name}"
        required.add(label)
        if (area, name) in inspected:
            return inspected[area, name]
        inspected[area, name] = None
        try:
            path = _local(source if area == "source" else checkpoint, name)
            if not path.is_file():
                missing.append(label)
                return None
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            actual = digest.hexdigest()
            if path.stat().st_size == 0:
                incompatible.append(f"{label}: empty artifact")
            matches = None if expected is None else actual == expected
            files.append(FileEvidence(area, name, actual, path.stat().st_size, matches))
            available.add(label)
            inspected[area, name] = path
            if expected is not None and (not _digest(expected) or not matches):
                incompatible.append(f"{label}: SHA256 mismatch")
            return path
        except (OSError, ValueError, RuntimeError) as exc:
            incompatible.append(f"{label}: {exc}")
            return None

    def read_object(area: str, name: str) -> dict[str, object] | None:
        path = inspect(area, name, expected_hashes.get(name) if area == "checkpoint" else None)
        if path is None:
            return None
        try:
            if area == "checkpoint" and name in TOKENIZER_ARTIFACT_NAMES:
                content = _metadata_bytes(path)
                evidence = next(f for f in files if f.area == area and f.path == name)
                if not content or hashlib.sha256(content).hexdigest() != evidence.sha256:
                    raise ValueError("empty or changed tokenizer artifact during inspection")
                parsed = _object(
                    json.loads(
                        content.decode("utf-8"), object_pairs_hook=_pairs, parse_constant=_constant
                    ),
                    name,
                )
                tokenizer_artifacts[name] = content
                return parsed
            return _object(_json(path), name)
        except (OSError, ValueError) as exc:
            incompatible.append(f"{area}/{name}: {exc}")
            return None

    expected_hashes = requirements.artifact_hashes or {}
    manifest = inspect("source", "source-manifest.json")
    source_start = len(incompatible)
    if manifest is not None:
        try:
            entries = _json(manifest)
            if not isinstance(entries, list):
                raise ValueError("source manifest must be a list")
            observed = {}
            for entry in entries:
                item = _object(entry, "source manifest entry")
                name = item.get("path")
                if not isinstance(name, str) or name in observed:
                    raise ValueError("invalid or duplicate source path")
                _local(source, name)
                if item.get("revision") != SOURCE_REVISION:
                    raise ValueError("incompatible source revision")
                observed[name] = item.get("sha256")
            if observed != SOURCE_HASHES:
                raise ValueError("source manifest differs from pinned source evidence")
        except (OSError, ValueError, RuntimeError) as exc:
            incompatible.append(f"source manifest: {exc}")
    for name, expected in SOURCE_HASHES.items():
        inspect("source", name, expected)
    source_verified = (
        manifest is not None
        and len(incompatible) == source_start
        and not any(item.startswith("source/") for item in missing)
    )
    for name in sorted(set(_CHECKPOINT_FILES) | set(expected_hashes)):
        inspect("checkpoint", name, expected_hashes.get(name))
    vocab: int | None = None
    positions: int | None = None
    dtype: str | None = None
    config = read_object("checkpoint", "config.json")
    if config is not None:
        try:
            if not requirements.model_type or config.get("model_type") != requirements.model_type:
                raise ValueError("incompatible model_type identity")
            if requirements.config_key is not None:
                config = _object(config.get(requirements.config_key), requirements.config_key)
            vocab = _positive(config.get("vocab_size"), "vocab_size")
            positions = _positive(config.get("max_position_embeddings"), "max_position_embeddings")
            declared = config.get("torch_dtype", config.get("dtype"))
            if declared != requirements.dtype or declared not in ("float32", "float16", "bfloat16"):
                raise ValueError("missing or incompatible declared dtype")
            if "dtype" in config and config["dtype"] != declared:
                raise ValueError("conflicting dtype declarations")
            dtype = str(declared)
        except ValueError as exc:
            incompatible.append(f"config.json: {exc}")
    tokenizer = read_object("checkpoint", "tokenizer.json")
    read_object("checkpoint", "tokenizer_config.json")
    tokenizer_names = list(REQUIRED_TOKENIZER_ARTIFACTS)
    for name in OPTIONAL_TOKENIZER_ARTIFACTS:
        # lexists also catches broken/escaping optional symlinks.
        if (checkpoint / name).exists() or (checkpoint / name).is_symlink():
            tokenizer_names.append(name)
            read_object("checkpoint", name)
    token_digest: str | None = None
    if len(tokenizer_artifacts) == len(tokenizer_names):
        token_digest = tokenizer_digest(tokenizer_artifacts)
    if (
        not _tokenizer_identity(requirements.tokenizer_digest)
        or token_digest != requirements.tokenizer_digest
    ):
        incompatible.append("tokenizer digest missing or incompatible with expected identity")
    if tokenizer is not None:
        try:
            model = _object(tokenizer.get("model"), "tokenizer.model")
            tokens = _object(model.get("vocab"), "tokenizer.model.vocab").copy()
            added = tokenizer.get("added_tokens", [])
            if not isinstance(added, list):
                raise ValueError("added_tokens must be a list")
            for entry in added:
                item = _object(entry, "added token")
                token = item.get("content")
                if not isinstance(token, str) or not token:
                    raise ValueError("invalid added token content")
                if token in tokens and tokens[token] != item.get("id"):
                    raise ValueError(f"conflicting token ID: {token}")
                tokens[token] = item.get("id")
            for token, token_id in tokens.items():
                if type(token_id) is not int or token_id < 0 or vocab is None or token_id >= vocab:
                    raise ValueError(f"token ID outside configured vocabulary: {token}")
            if len(set(tokens.values())) != len(tokens):
                raise ValueError("duplicate tokenizer IDs")
            for token, expected_id in SPECIAL_TOKEN_IDS.items():
                if tokens.get(token) != expected_id:
                    raise ValueError(
                        f"incompatible token identity: {token}, expected {expected_id}"
                    )
        except (ValueError, TypeError) as exc:
            incompatible.append(f"tokenizer.json: {exc}")
    split: Literal["split_candidate", "packed", "unknown"] = "unknown"
    index = read_object("checkpoint", "model.safetensors.index.json")
    if index is not None:
        try:
            if "metadata" in index:
                index_metadata = _object(index["metadata"], "index metadata")
                if "total_size" in index_metadata:
                    _positive(index_metadata["total_size"], "index total_size")
            weight_map = _object(index.get("weight_map"), "weight_map")
            if not weight_map:
                raise ValueError("weight_map must not be empty")
            for key, shard in weight_map.items():
                if not key or not isinstance(shard, str) or not shard.endswith(".safetensors"):
                    raise ValueError(
                        "weight_map must map nonempty tensor keys to safetensors paths"
                    )
                inspect("checkpoint", shard, expected_hashes.get(shard))
            if any(
                re.search(r"\.experts\.(?:gate_up_proj|down_proj|w[123])(?:\.weight)?$", key)
                for key in weight_map
            ):
                split = "packed"
                raise ValueError("packed expert keys are incompatible with split-only inference")
            if any(re.search(r"\.experts\.\d+\.", key) for key in weight_map):
                split = "split_candidate"
            else:
                raise ValueError("split layout cannot be established from index keys")
        except ValueError as exc:
            incompatible.append(f"checkpoint index: {exc}")
    versions: list[tuple[str, str | None]] = []
    for package in requirements.required_packages:
        try:
            version = metadata.version(package)
        except metadata.PackageNotFoundError:
            version = None
            missing.append(f"distribution/{package}")
        versions.append((package, version))
    return U2PreflightReport(
        SOURCE_REVISION,
        source_verified,
        tuple(files),
        tuple(sorted(required)),
        tuple(sorted(available)),
        tuple(missing),
        tuple(incompatible),
        token_digest,
        vocab,
        positions,
        dtype,
        split,
        tuple(versions),
        not missing and not incompatible,
    )
