"""Strict offline U2 loading, with separate artifact, source and object evidence.

No hub loader or checkpoint-supplied executable code is used. Callers must admit
immutable manifest hashes independently. CPU fixture tests do not qualify the
pinned model import graph. The disclosed split-only profile excludes two unused
fused/distributed imports while preserving all pinned computation and dependencies.
"""

from __future__ import annotations

import hashlib
import importlib
import importlib.abc
import importlib.machinery
import inspect
import json
import re
import stat
import struct
import sys
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import partial
from pathlib import Path, PurePosixPath
from types import FunctionType, MappingProxyType, MethodType, ModuleType
from typing import Any, cast

import torch
from torch import Tensor, nn

from .text_contracts import RuntimeIdentity
from .u2_preflight import SOURCE_HASHES, SOURCE_REVISION, U2PreflightReport
from .u2_text_renderer import TOKENIZER_ARTIFACT_NAMES, tokenizer_digest
from .u2_text_runtime import EOA, MASK, LocalModelSpec
from .u2_text_tokenizer import BoundTextTokenizer, bind_local_tokenizer

CHECKPOINT_REVISION = "8e3d57f3cb190323334752952a622bb5b97086c1"
_MODEL_SOURCE = "models/internluminau2/modeling_internluminau2.py"
_MODEL_SHA256 = "badb75508d4cda0964f76a19aa6dd32d20e72f5a52f4615abff1b78d6ded18b0"
_SPLIT_IMPORTS = (
    b"from veomni.ops import fused_moe_forward\n"
    b"from veomni.distributed.parallel_state import get_parallel_state\n"
)
_SPLIT_REPLACEMENT = (
    b"from src.backends.u2_checkpoint import _unsupported_split_optional as fused_moe_forward\n"
    b"from src.backends.u2_checkpoint import _unsupported_split_optional as get_parallel_state\n"
)
_PACKED = re.compile(
    r"\.experts\.(?:gate_proj|up_proj|gate_up_proj|down_proj|w[123])(?:\.weight)?$"
)
_SPLIT = re.compile(r"\.mlp\.experts\.\d+\.(?:gate_proj|up_proj|down_proj)\.weight$")
_DTYPES = {
    "F32": torch.float32,
    "F16": torch.float16,
    "BF16": torch.bfloat16,
    "I64": torch.int64,
    "I32": torch.int32,
    "BOOL": torch.bool,
}


class CheckpointError(ValueError):
    """Integrity, layout or binding failure; no usable model is returned."""


class UnsupportedRuntimeError(RuntimeError):
    """Execution cannot be qualified in the selected runtime."""


def _unsupported_split_optional(*args: Any, **kwargs: Any) -> Any:
    raise UnsupportedRuntimeError("packed/fused/distributed execution excluded by split-only-v1")


def split_source(content: bytes) -> bytes:
    """Adapt exactly two import statements; all pinned function bodies stay intact."""
    if sha256(content) != _MODEL_SHA256 or content.count(_SPLIT_IMPORTS) != 1:
        raise CheckpointError("split-only import profile requires exact pinned model source")
    return content.replace(_SPLIT_IMPORTS, _SPLIT_REPLACEMENT, 1)


def split_adaptation(source: VerifiedFiles) -> dict[str, Any]:
    """Disclose original and compiled source identities, without editing upstream."""
    original = source.read(_MODEL_SOURCE)
    compiled = split_source(original)
    return {
        "profile": "split-only-v1",
        "path": _MODEL_SOURCE,
        "original_sha256": sha256(original),
        "compiled_sha256": sha256(compiled),
        "removed_imports": _SPLIT_IMPORTS.decode(),
        "replacement_imports": _SPLIT_REPLACEMENT.decode(),
        "excluded_symbols": ["fused_moe_forward", "get_parallel_state"],
        "computation_changes": [],
        "optional_packages": "veomni_patch and liger_kernel must be absent",
    }


def _reject_optional_code() -> None:
    # Called only after installing the effective source + VeOmni search path.
    for name in ("veomni_patch", "liger_kernel"):
        if any(k == name or k.startswith(name + ".") for k in sys.modules):
            raise UnsupportedRuntimeError(f"unadmitted optional {name} already loaded")
        if importlib.machinery.PathFinder.find_spec(name, sys.path) is not None:
            raise UnsupportedRuntimeError(f"unadmitted optional {name} on effective import path")


def sha256(content: bytes) -> str:
    """Return a bare SHA256 content identity."""
    return hashlib.sha256(content).hexdigest()


def _pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in items:
        if key in result:
            raise CheckpointError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _nonfinite(value: str) -> Any:
    raise CheckpointError(f"nonfinite JSON: {value}")


def json_object(content: bytes) -> dict[str, Any]:
    """Parse an unambiguous finite JSON object."""
    try:
        value = json.loads(content, object_pairs_hook=_pairs, parse_constant=_nonfinite)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise CheckpointError("malformed JSON input") from exc
    if not isinstance(value, dict):
        raise CheckpointError("expected JSON object")
    return cast(dict[str, Any], value)


def safe_path(root: Path, relative: str = "") -> Path:
    """Reject aliases, traversal and every symlink ancestor, including the root."""
    if ".." in root.parts:
        raise CheckpointError("traversal in root path")
    if relative and (
        "\\" in relative
        or "\x00" in relative
        or PurePosixPath(relative).is_absolute()
        or any(p in ("", ".", "..") for p in relative.split("/"))
    ):
        raise CheckpointError(f"unsafe relative path: {relative!r}")
    path = root.absolute() / relative
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise CheckpointError(f"symlink input: {path}")
    if not path.resolve().is_relative_to(root.absolute().resolve()):
        raise CheckpointError("path escapes root")
    return path


def _hash_file(path: Path) -> tuple[int, str, str]:
    info = path.stat()
    if not stat.S_ISREG(info.st_mode):
        raise CheckpointError("input is not a regular file")
    size = info.st_size
    git = hashlib.sha1(f"blob {size}\0".encode())
    digest = hashlib.sha256()
    count = 0
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            count += len(chunk)
            digest.update(chunk)
            git.update(chunk)
    if count != size:
        raise CheckpointError("file changed while hashing")
    return size, digest.hexdigest(), git.hexdigest()


@dataclass(frozen=True)
class VerifiedFiles:
    """Admitted original identities plus observed hashes of local regular files."""

    root: Path
    manifest_sha256: str
    files: Mapping[str, Mapping[str, Any]]
    identity: Mapping[str, str]

    def read(self, name: str) -> bytes:
        """Read an owned snapshot and recheck the consumed bytes."""
        if name not in self.files:
            raise CheckpointError(f"required artifact absent from manifest: {name}")
        expected = self.files[name]
        path = safe_path(self.root, name)
        if not path.is_file():
            raise CheckpointError(f"required regular artifact missing: {name}")
        with path.open("rb") as stream:
            content = stream.read(expected["bytes"] + 1)
        if len(content) != expected["bytes"] or sha256(content) != expected["sha256"]:
            raise CheckpointError(f"artifact changed: {name}")
        return content

    def verify(self, name: str) -> Path:
        """Recheck a large file without retaining its bytes in memory."""
        path = safe_path(self.root, name)
        size, digest, _ = _hash_file(path)
        if size != self.files[name]["bytes"] or digest != self.files[name]["sha256"]:
            raise CheckpointError(f"artifact changed: {name}")
        return path


def verify_manifest(
    root: Path,
    manifest: Path,
    *,
    expected_sha256: str,
    kind: str,
    variant_prefix: str | None = None,
) -> VerifiedFiles:
    """Verify Git-blob SHA1 and/or LFS SHA256 originals, recording fresh SHA256s.

    The only prefix mapping is explicit ``nvidia/`` to a root named ``nvidia``.
    A prior verification receipt is never accepted in place of a trust manifest.
    Source manifests are separate from the six-entry preflight manifest.
    """
    safe_path(root)
    if not root.is_dir():
        raise CheckpointError("artifact root must be a directory")
    manifest_path = safe_path(manifest.parent, manifest.name)
    if not manifest_path.is_file() or manifest_path.stat().st_size > 64 * 1024 * 1024:
        raise CheckpointError("manifest must be a bounded regular file")
    raw = manifest_path.read_bytes()
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256) or sha256(raw) != expected_sha256:
        raise CheckpointError("trusted manifest SHA256 mismatch")
    data = json_object(raw)
    entries: dict[str, dict[str, Any]] = {}
    if kind == "checkpoint":
        if (
            variant_prefix != "nvidia/"
            or data.get("schema") != 1
            or root.name != "nvidia"
            or data.get("repository") != "internlm/InternLumina-U2"
            or data.get("revision") != CHECKPOINT_REVISION
            or not isinstance(data.get("files"), list)
        ):
            raise CheckpointError("checkpoint identity or exact variant-prefix mapping mismatch")
        for item in data["files"]:
            name = item.get("path")
            if not isinstance(name, str) or not name.startswith(variant_prefix):
                raise CheckpointError("manifest path lacks exact variant prefix")
            relative = name[len(variant_prefix) :]
            safe_path(root, relative)
            if relative in entries:
                raise CheckpointError("duplicate manifest path")
            entries[relative] = item
        actual: set[str] = set()
        for path in root.rglob("*"):
            safe_path(root, path.relative_to(root).as_posix())
            if path.is_file():
                actual.add(path.relative_to(root).as_posix())
            elif not path.is_dir():
                raise CheckpointError("nonregular checkpoint input")
        if actual != set(entries):
            raise CheckpointError(
                f"checkpoint file set mismatch: missing={set(entries)-actual}, extra={actual-set(entries)}"
            )
        identity = {
            "repository": data["repository"],
            "revision": data["revision"],
            "variant": "nvidia",
        }
    elif kind == "source":
        if data.get("source_revision") != SOURCE_REVISION or not isinstance(
            data.get("files"), dict
        ):
            raise CheckpointError("source manifest revision or files mismatch")
        entries = data["files"]
        if any(
            name not in entries or entries[name].get("sha256") != digest
            for name, digest in SOURCE_HASHES.items()
        ):
            raise CheckpointError("source manifest does not preserve six pinned files")
        identity = {"revision": SOURCE_REVISION}
    else:
        raise CheckpointError("unknown manifest kind")
    if not entries:
        raise CheckpointError("empty manifest")
    observed: dict[str, Mapping[str, Any]] = {}
    for name, item in entries.items():
        path = safe_path(root, name)
        if not path.is_file() or type(item.get("bytes")) is not int:
            raise CheckpointError(f"missing or invalid artifact: {name}")
        size, digest, git = _hash_file(path)
        expected_git = item.get("git_blob_sha1", item.get("git_blob"))
        expected_digest = item.get("sha256")
        if (
            size != item["bytes"]
            or not (expected_git or expected_digest)
            or (expected_git is not None and expected_git != git)
            or (expected_digest is not None and expected_digest != digest)
        ):
            raise CheckpointError(f"original artifact identity mismatch: {name}")
        observed[name] = MappingProxyType(
            {
                "bytes": size,
                "sha256": digest,
                "git_blob_sha1": git,
                "verified_original": "git_blob_sha1" if expected_git else "sha256",
            }
        )
    if kind == "checkpoint" and data.get("total_bytes") != sum(
        v["bytes"] for v in observed.values()
    ):
        raise CheckpointError("manifest total byte count mismatch")
    return VerifiedFiles(
        root.absolute(), expected_sha256, MappingProxyType(observed), MappingProxyType(identity)
    )


@dataclass(frozen=True)
class TensorEntry:
    """Validated safetensors header entry bound to one shard."""

    shard: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    offsets: tuple[int, int]


def inspect_tensors(files: VerifiedFiles) -> dict[str, TensorEntry]:
    """Validate duplicate-free headers, exact index membership and dense ranges."""
    index = json_object(files.read("model.safetensors.index.json"))
    weight_map = index.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise CheckpointError("missing weight_map")
    if any(
        not isinstance(k, str) or not k or not isinstance(v, str) for k, v in weight_map.items()
    ):
        raise CheckpointError("invalid weight_map")
    if any(_PACKED.search(key) for key in weight_map) or not any(
        _SPLIT.search(key) for key in weight_map
    ):
        raise CheckpointError("only split expert layout is admitted")
    shards = set(weight_map.values())
    if shards != {name for name in files.files if name.endswith(".safetensors")}:
        raise CheckpointError("missing or extra shards")
    result: dict[str, TensorEntry] = {}
    total = 0
    for name in sorted(shards):
        path = files.verify(name)
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise CheckpointError("truncated safetensors prefix")
            length = struct.unpack("<Q", prefix)[0]
            if length > 64 * 1024 * 1024 or length + 8 > path.stat().st_size:
                raise CheckpointError("invalid safetensors header length")
            header = json_object(stream.read(length))
        ranges: list[tuple[int, int]] = []
        for key, value in header.items():
            if key == "__metadata__":
                if not isinstance(value, dict) or any(
                    not isinstance(v, str) for v in value.values()
                ):
                    raise CheckpointError("invalid safetensors metadata")
                continue
            if key in result or weight_map.get(key) != name or not isinstance(value, dict):
                raise CheckpointError(f"duplicate, extra or misplaced tensor: {key}")
            shape, offsets, dtype_name = (
                value.get("shape"),
                value.get("data_offsets"),
                value.get("dtype"),
            )
            if (
                not isinstance(shape, list)
                or any(type(n) is not int or n < 0 for n in shape)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or any(type(n) is not int or n < 0 for n in offsets)
                or dtype_name not in _DTYPES
            ):
                raise CheckpointError(f"invalid tensor metadata: {key}")
            dtype = _DTYPES[dtype_name]
            count = 1
            for dim in shape:
                count *= dim
            size = count * torch.empty((), dtype=dtype).element_size()
            if offsets[1] - offsets[0] != size:
                raise CheckpointError(f"tensor byte extent mismatch: {key}")
            ranges.append((offsets[0], offsets[1]))
            total += size
            result[key] = TensorEntry(name, tuple(shape), dtype, (offsets[0], offsets[1]))
        cursor = 0
        for start, end in sorted(ranges):
            if start != cursor:
                raise CheckpointError("overlapping or noncontiguous tensor data")
            cursor = end
        if cursor + length + 8 != path.stat().st_size:
            raise CheckpointError("unindexed trailing or missing tensor bytes")
    if set(result) != set(weight_map):
        raise CheckpointError("index tensor set mismatch")
    if "total_size" in index.get("metadata", {}) and index["metadata"]["total_size"] != total:
        raise CheckpointError("index total tensor size mismatch")
    return result


def tensor_hash(tensor: Tensor) -> str:
    """Hash every original tensor byte without a dtype conversion."""
    return sha256(
        tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
    )


def strict_load(
    model: nn.Module, files: VerifiedFiles, entries: Mapping[str, TensorEntry]
) -> dict[str, Any]:
    """Bind every constructed state key to exact checkpoint tensors, without casts.

    Full key/shape/dtype equality is checked before the first assignment. Shard
    assignment uses one verified owned shard buffer plus deserializer allocations;
    the largest production shard is approximately 5 GB. Model storage and tensor
    hashing are additional. Allocation failure propagates; no partial object is
    returned. All state bytes are read back and compared after assignment.
    """
    from safetensors.torch import load

    expected = model.state_dict()
    if set(expected) != set(entries):
        raise CheckpointError(
            f"model key mismatch: missing={set(expected)-set(entries)}, extra={set(entries)-set(expected)}"
        )
    for key, tensor in expected.items():
        entry = entries[key]
        if (
            tensor.layout != torch.strided
            or tuple(tensor.shape) != entry.shape
            or tensor.dtype != entry.dtype
        ):
            raise CheckpointError(f"constructed model shape/dtype mismatch: {key}")
    del expected  # Do not retain the replaced constructor tensors across all shards.
    load_safetensors = cast(Callable[[bytes], dict[str, Tensor]], load)
    hashes: dict[str, str] = {}
    for name in sorted({v.shard for v in entries.values()}):
        # The immutable owned buffer is both hashed and deserialized. No second
        # pathname open can substitute A->B->A bytes between trust and use.
        owned = files.read(name)
        state = load_safetensors(owned)
        del owned
        keys = {k for k, v in entries.items() if v.shard == name}
        if set(state) != keys:
            raise CheckpointError("shard keys changed")
        for key, tensor in state.items():
            entry = entries[key]
            if tuple(tensor.shape) != entry.shape or tensor.dtype != entry.dtype:
                raise CheckpointError("shard tensor metadata changed")
            hashes[key] = tensor_hash(tensor)
        incompatible = model.load_state_dict(state, strict=False, assign=True)
        if incompatible.unexpected_keys or set(incompatible.missing_keys) != set(entries) - keys:
            raise CheckpointError("model state assignment deviated from checked keys")
        del state
    actual = model.state_dict()
    if set(actual) != set(hashes) or any(
        t.is_meta or tensor_hash(t) != hashes[k] for k, t in actual.items()
    ):
        raise CheckpointError("loaded tensor byte binding mismatch")
    if any(t.is_meta for t in model.buffers()):
        raise CheckpointError("uninitialized model buffer")
    return {
        "complete": True,
        "missing_keys": [],
        "unexpected_keys": [],
        "conversions": [],
        "consumption": "sha256-verified immutable per-shard bytes -> safetensors.torch.load",
        "largest_owned_shard_bytes": max(files.files[v.shard]["bytes"] for v in entries.values()),
        "tensors": {
            k: {
                "shape": list(v.shape),
                "dtype": str(v.dtype),
                "sha256": hashes[k],
                "shard": v.shard,
            }
            for k, v in entries.items()
        },
    }


class _PinnedFinder(importlib.abc.MetaPathFinder):
    def __init__(self, source: VerifiedFiles, roots: tuple[str, ...], model_import: bool) -> None:
        self.source, self.roots, self.model_import = source, roots, model_import

    def find_spec(
        self, fullname: str, path: Sequence[str] | None = None, target: ModuleType | None = None
    ) -> importlib.machinery.ModuleSpec | None:
        pinned = fullname.split(".")[0] in self.roots
        if pinned:
            _reject_optional_code()
        if fullname.startswith(("veomni.ops", "veomni.distributed")) and self.model_import:
            raise UnsupportedRuntimeError("split-only profile excludes veomni ops/distributed")
        spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
        if not pinned and (
            spec is None
            or spec.origin is None
            or not Path(spec.origin).absolute().is_relative_to(self.source.root)
        ):
            return None
        if spec is None:
            raise UnsupportedRuntimeError(f"pinned module unavailable: {fullname}")
        if spec.origin is None:
            if not spec.submodule_search_locations or any(
                not Path(p).absolute().is_relative_to(self.source.root)
                for p in spec.submodule_search_locations
            ):
                raise CheckpointError(f"foreign pinned namespace: {fullname}")
        else:
            file = Path(spec.origin).absolute()
            if not file.is_relative_to(self.source.root):
                raise CheckpointError(f"foreign pinned module before import: {fullname}")
            relative = file.relative_to(self.source.root).as_posix()
            if relative not in self.source.files or file.suffix != ".py":
                raise CheckpointError(f"unmanifested pinned code before import: {relative}")
            self.source.verify(relative)
            # Import the verified source bytes, never a stale timestamp-valid pyc.
            spec.loader = _VerifiedSourceLoader(
                fullname, str(file), self.source, relative, self.model_import
            )
        return spec


class _VerifiedSourceLoader(importlib.machinery.SourceFileLoader):
    def __init__(
        self, name: str, path: str, source: VerifiedFiles, relative: str, model_import: bool
    ) -> None:
        super().__init__(name, path)
        self.source, self.relative = source, relative
        self.model_import = model_import

    def get_code(self, fullname: str) -> Any:
        content = self.source.read(self.relative)
        if self.relative == _MODEL_SOURCE:
            if not self.model_import:
                raise UnsupportedRuntimeError("model import requires the explicit split profile")
            content = split_source(content)
        return compile(content, self.path, "exec", dont_inherit=True)


def require_import_runtime(device: torch.device) -> None:
    """Admit explicit device metadata, separately from actual device availability."""
    if (
        device.type not in ("cpu", "cuda", "xpu")
        or (device.type != "cpu" and device.index is None)
        or (device.type == "cpu" and device.index is not None)
    ):
        raise UnsupportedRuntimeError("explicit supported device required")


@contextmanager
def pinned_imports(
    source: VerifiedFiles, *, device: torch.device, model_import: bool = True
) -> Iterator[None]:
    """Use only the admitted vendored packages; never fabricate module stubs."""
    if model_import:
        require_import_runtime(device)
    roots = ("models", "generators", "utils", "veomni")
    if any(name.split(".")[0] in roots for name in sys.modules):
        raise UnsupportedRuntimeError("pinned package names already imported; use a fresh process")
    original = list(sys.path)
    finder = _PinnedFinder(source, roots, model_import)
    sys.path[:0] = [str(source.root), str(source.root / "VeOmni")]
    sys.meta_path.insert(0, finder)
    try:
        importlib.invalidate_caches()
        _reject_optional_code()
        if model_import:
            split_adaptation(source)
        yield
        _reject_optional_code()
    except ModuleNotFoundError as exc:
        raise UnsupportedRuntimeError(
            f"genuine runtime dependency unavailable: {exc.name}; provision a separately "
            "authorized compatible environment (pinned tokenbridge requires timm==1.0.22)"
        ) from exc
    finally:
        sys.meta_path.remove(finder)
        sys.path[:] = original


def imported_code(source: VerifiedFiles) -> list[dict[str, Any]]:
    """Record every imported module file; strictly bind pinned-package origins."""
    records: list[dict[str, Any]] = []
    for name, module in sorted(tuple(sys.modules.items())):
        filename = getattr(module, "__file__", None)
        if not filename:
            continue
        path = Path(filename).absolute()
        pinned = name.split(".")[0] in {"models", "generators", "utils", "veomni"}
        if path.is_relative_to(source.root):
            pinned = True
        if pinned:
            if not path.is_relative_to(source.root):
                raise CheckpointError(f"foreign pinned module: {name}")
            relative = path.relative_to(source.root).as_posix()
            if relative not in source.files:
                raise CheckpointError(f"unmanifested imported code: {relative}")
            source.verify(relative)
        if path.is_file():
            size, digest, _ = _hash_file(path)
            records.append(
                {
                    "module": name,
                    "path": str(path),
                    "bytes": size,
                    "sha256": digest,
                    "pinned_source": pinned,
                }
            )
    return records


def _value_binding(value: Any) -> Any:
    """Snapshot ordinary execution attributes without calling arbitrary objects."""
    if isinstance(value, Tensor):
        return (id(value), value.data_ptr(), value._version, tuple(value.shape), value.dtype)
    if isinstance(value, nn.Module):
        return (id(value), type(value))
    if isinstance(value, MethodType):
        return (id(value.__self__), _value_binding(value.__func__))
    if isinstance(value, FunctionType):
        return (
            id(value),
            id(value.__code__),
            _value_binding(value.__defaults__),
            _value_binding(value.__kwdefaults__),
        )
    if isinstance(value, partial):
        return (
            id(value),
            _value_binding(value.func),
            _value_binding(value.args),
            _value_binding(value.keywords),
        )
    if isinstance(value, tuple | list):
        return (type(value), tuple(_value_binding(v) for v in value))
    if isinstance(value, dict):
        return tuple((k, _value_binding(v)) for k, v in value.items())
    if value is None or isinstance(value, bool | int | float | str | torch.dtype | torch.device):
        return (type(value), value)
    return (id(value), type(value))


class BoundCheckpoint:
    """Actual constructed model and state binding, distinct from LocalModelSpec.

    Version/pointer checks guard each call; full byte checks bracket comparisons
    and detect ordinary .data mutation. This is not a hostile-Python boundary.
    """

    def __init__(
        self,
        model: nn.Module,
        spec: LocalModelSpec,
        tokenizer: BoundTextTokenizer,
        provenance: dict[str, Any],
    ) -> None:
        self.model = model
        self.spec = spec
        self.tokenizer = tokenizer
        self.provenance = provenance
        self._spec_binding = spec
        self._tokenizer_id = id(tokenizer)
        self._model_id = id(model)
        self._class = type(model)
        self._forward = type(model).forward
        self._state = self._snapshot()
        self._modules = self._module_snapshot()
        self._classes = self._class_snapshot()
        self._hashes = {k: tensor_hash(t) for k, t in self._tensors().items()}
        self._config = json.dumps(cast(Any, model).config.to_dict(), sort_keys=True, default=str)
        self.verify_binding(full=True)

    def _tensors(self) -> dict[str, Tensor]:
        return {**dict(self.model.named_parameters()), **dict(self.model.named_buffers())}

    def _module_snapshot(self) -> tuple[tuple[Any, ...], ...]:
        return tuple(
            (
                name,
                id(module),
                type(module),
                type(module).forward,
                tuple((key, _value_binding(value)) for key, value in module.__dict__.items()),
            )
            for name, module in self.model.named_modules()
        )

    def _class_snapshot(self) -> tuple[Any, ...]:
        # Shared methods are bound once per class, not thousands of times per
        # expert. Include helper methods, descriptors and callable defaults.
        classes = dict.fromkeys(
            cls for module in self.model.modules() for cls in type(module).__mro__
        )
        return tuple(
            (cls, key, _value_binding(value))
            for cls in classes
            for key, value in cls.__dict__.items()
            if callable(value) or isinstance(value, staticmethod | classmethod | property)
        )

    def _snapshot(self) -> dict[str, tuple[Any, ...]]:
        return {
            k: (id(t), t.data_ptr(), t._version, tuple(t.shape), t.dtype, t.device)
            for k, t in self._tensors().items()
        }

    def verify_binding(self, *, full: bool = False) -> None:
        """Check the actual object, configuration and loaded state."""
        self.tokenizer.verify_binding()
        if (
            id(self.model) != self._model_id
            or self.spec is not self._spec_binding
            or id(self.tokenizer) != self._tokenizer_id
            or type(self.model) is not self._class
            or type(self.model).forward is not self._forward
            or "forward" in self.model.__dict__
            or any(m.training for m in self.model.modules())
            or self._snapshot() != self._state
            or self._module_snapshot() != self._modules
            or self._class_snapshot() != self._classes
            or (
                getattr(self.spec, "prompt_vocab_size", None) is not None
                and self.spec.prompt_vocab_size != self.tokenizer.vocab_size
            )
            or json.dumps(cast(Any, self.model).config.to_dict(), sort_keys=True, default=str)
            != self._config
        ):
            raise CheckpointError("actual model construction/object binding changed")
        if full and any(tensor_hash(t) != self._hashes[k] for k, t in self._tensors().items()):
            raise CheckpointError("actual model parameter bytes changed")

    @property
    def training(self) -> bool:
        """Report the bound model training flag."""
        return bool(self.model.training)

    def eval(self) -> BoundCheckpoint:
        """Put the existing model in evaluation mode."""
        self.model.eval()
        return self

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
        """Execute one pinned forward, retaining its full logits domain."""
        self.verify_binding()
        result = self.model(
            input_ids,
            type_position_ids,
            attention_mask=attention_mask,
            infer_mmu=infer_mmu,
            is_clean=is_clean,
            position_ids=position_ids,
        )
        self.verify_binding()
        if not isinstance(result, Tensor):
            raise CheckpointError("pinned model did not return a logits tensor")
        return result


def validate_split_config(config: Mapping[str, Any]) -> None:
    """Admit the selected eight-channel split architecture, including fused metadata."""
    required = {
        "model_type": "llada2_moe",
        "num_codebooks": 8,
        "torch_dtype": "bfloat16",
        "vocab_size": 157184,
        "num_hidden_layers": 20,
        "first_k_dense_replace": 1,
        "num_experts": 256,
        "num_shared_experts": 1,
        "hidden_size": 2048,
        "intermediate_size": 5120,
        "moe_intermediate_size": 512,
        "num_attention_heads": 16,
        "num_key_value_heads": 4,
        "head_dim": 128,
        "num_experts_per_tok": 8,
        "n_group": 8,
        "topk_group": 4,
        "max_position_embeddings": 16384,
        "_moe_implementation": "fused",
        "use_cache": False,
        "use_sliding_window": False,
    }
    if any(type(config.get(k)) is not type(v) or config.get(k) != v for k, v in required.items()):
        raise CheckpointError("incompatible split/eight-channel/model-dimension config")
    if config.get("_attn_implementation", "sdpa") != "sdpa":
        raise UnsupportedRuntimeError("split-only profile requires SDPA")
    if config.get("use_segmented_attention", False) is not False:
        raise UnsupportedRuntimeError("split-only profile excludes segmented attention")


def validate_split_model(model: Any, module: ModuleType, config: Any) -> None:
    """Check the actual full constructed split object, never pruning image weights."""
    if (
        type(model) is not module.LLaDA2MoeModelLM
        or model.config is not config
        or config.model_type != "llada2_moe"
        or config._attn_implementation != "sdpa"
        or model.num_codebooks != 8
        or model.model.num_codebooks != 8
        or len(model.model.layers) != config.num_hidden_layers
        or tuple(model.lm_head.weight.shape) != (157184, config.hidden_size)
        or tuple(model.model.word_embeddings.weight.shape) != (157184, config.hidden_size)
        or len(model.lm_img_head) != 8
        or len(model.model.img_embeddings) != 8
        or module.is_liger_kernel_available()
    ):
        raise CheckpointError("actual constructed split model dimensions/branch differ")
    for i, layer in enumerate(model.model.layers):
        mlp = layer.mlp
        if i < config.first_k_dense_replace:
            if type(mlp) is not module.LLaDA2MoeMLP:
                raise CheckpointError("actual dense layer differs")
        elif (
            type(mlp) is not module.LLaDA2MoeSparseMoeBlock
            or type(mlp.experts) is not nn.ModuleList
            or len(mlp.experts) != config.num_experts
            or any(type(expert) is not module.LLaDA2MoeMLP for expert in mlp.experts)
            or type(mlp.shared_experts) is not module.LLaDA2MoeMLP
        ):
            raise CheckpointError("actual split/shared expert layout differs")
    if any(type(m) is module.LLaDA2MoeExperts for m in model.modules()):
        raise CheckpointError("packed expert object is excluded")


def load_checkpoint(
    files: VerifiedFiles,
    source: VerifiedFiles,
    *,
    device: torch.device,
    dtype: torch.dtype,
    preflight: U2PreflightReport,
) -> tuple[BoundCheckpoint, ModuleType]:
    """Construct the pinned class locally and strictly load all split state.

    BF16/SDPA preserves the pinned loader's dtype/backend. No dtype conversion,
    random fallback, AutoModel, from_pretrained, downloads or bundled code import.
    The CPU construction is intentionally memory-heavy; resource fit is unqualified.
    """
    preflight.require_metadata()
    if dtype != torch.bfloat16:
        raise UnsupportedRuntimeError(
            "pinned loader fixes bfloat16; diagnostic dtype conversion is not implemented"
        )
    if (
        device.type not in ("cpu", "cuda", "xpu")
        or (device.type != "cpu" and device.index is None)
        or (device.type == "cpu" and device.index is not None)
    ):
        raise UnsupportedRuntimeError("explicit supported device required")
    require_import_runtime(device)
    entries = inspect_tensors(files)
    config_data = json_object(files.read("config.json"))
    validate_split_config(config_data)
    artifacts = {name: files.read(name) for name in TOKENIZER_ARTIFACT_NAMES if name in files.files}
    bound_tokenizer = bind_local_tokenizer(artifacts, expected_digest=tokenizer_digest(artifacts))
    if (
        bound_tokenizer.vocab_size != 157177
        or preflight.tokenizer_digest != bound_tokenizer.tokenizer_digest
    ):
        raise CheckpointError("actual bound tokenizer cardinality/digest mismatch")
    with pinned_imports(source, device=device):
        generator = importlib.import_module("generators.text_understanding_generator")
        module = importlib.import_module("models.internluminau2.modeling_internluminau2")
        config_module = importlib.import_module(
            "models.internluminau2.configuration_internluminau2"
        )
        code = imported_code(source)
        config = config_module.LLaDA2MoeConfig(**config_data)
        config._attn_implementation = "sdpa"
        previous = torch.get_default_dtype()
        try:
            torch.set_default_dtype(dtype)
            model = module.LLaDA2MoeModelLM(config)
        finally:
            torch.set_default_dtype(previous)
        validate_split_model(model, module, config)
    report = strict_load(model, files, entries)
    if model.num_codebooks != 8 or model.lm_head.weight.shape[0] != 157184:
        raise CheckpointError("actual constructed model dimensions differ")
    model.to(device=device).eval()
    # Movement must preserve every checkpoint tensor byte and dtype.
    if any(tensor_hash(t) != report["tensors"][k]["sha256"] for k, t in model.state_dict().items()):
        raise CheckpointError("device movement changed loaded tensor bytes")
    spec = LocalModelSpec(
        RuntimeIdentity(
            "strict-local-u2-v1",
            SOURCE_REVISION,
            bound_tokenizer.tokenizer_digest,
            157184,
            EOA,
            MASK,
        ),
        bound_tokenizer.special_token_ids,
        dtype,
        device,
        config.max_position_embeddings,
        prompt_vocab_size=bound_tokenizer.vocab_size,
    )
    code = imported_code(source)
    provenance = {
        "strict_load": report,
        "imported_code": code,
        "adaptations": [split_adaptation(source)],
        "construction": "pinned LLaDA2MoeConfig -> LLaDA2MoeModelLM -> strict tensor assignment",
        "constructed_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "model_source": inspect.getfile(type(model)),
        "config_sha256": files.files["config.json"]["sha256"],
        "tokenizer_cardinality": bound_tokenizer.vocab_size,
        "model_vocab_size": 157184,
        "observed_logit_dimension": None,
        "checkpoint": dict(files.identity),
        "checkpoint_files": {k: dict(v) for k, v in files.files.items()},
        "source": dict(source.identity),
    }
    return BoundCheckpoint(model, spec, bound_tokenizer, provenance), generator
