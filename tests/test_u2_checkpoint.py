"""Tiny CPU safetensors validate strict loading; no real checkpoint/code executes."""

import hashlib
import json
import struct
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
from src.backends import u2_checkpoint as loader
from src.backends.u2_checkpoint import (
    BoundCheckpoint,
    CheckpointError,
    UnsupportedRuntimeError,
    inspect_tensors,
    json_object,
    safe_path,
    strict_load,
    verify_manifest,
)

KEY = "model.layers.0.mlp.experts.0.gate_proj.weight"


def manifest_for(root, path):
    files = []
    for file in sorted(root.iterdir()):
        data = file.read_bytes()
        entry = {"path": "nvidia/" + file.name, "bytes": len(data)}
        if file.suffix == ".safetensors":
            entry["sha256"] = hashlib.sha256(data).hexdigest()
        else:
            entry["git_blob_sha1"] = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        files.append(entry)
    value = {
        "schema": 1,
        "repository": "internlm/InternLumina-U2",
        "revision": loader.CHECKPOINT_REVISION,
        "total_bytes": sum(f["bytes"] for f in files),
        "files": files,
    }
    path.write_text(json.dumps(value))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture
def assets(tmp_path):
    root = tmp_path / "nvidia"
    root.mkdir()
    save_file(
        {KEY: torch.arange(6, dtype=torch.float32).reshape(2, 3), "bias": torch.tensor([2.0, 3.0])},
        str(root / "one.safetensors"),
    )
    (root / "config.json").write_text('{"model_type":"synthetic"}')
    (root / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": 32},
                "weight_map": {KEY: "one.safetensors", "bias": "one.safetensors"},
            }
        )
    )
    manifest = tmp_path / "manifest.json"
    digest = manifest_for(root, manifest)
    return root, manifest, digest


def verified(assets):
    root, manifest, digest = assets
    return verify_manifest(
        root, manifest, expected_sha256=digest, kind="checkpoint", variant_prefix="nvidia/"
    )


def tiny_model():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model = torch.nn.Module()
            self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
            self.model.layers[0].mlp = torch.nn.Module()
            self.model.layers[0].mlp.experts = torch.nn.ModuleList([torch.nn.Module()])
            self.model.layers[0].mlp.experts[0].gate_proj = torch.nn.Linear(3, 2, bias=False)
            self.bias = torch.nn.Parameter(torch.empty(2))
            self.config = SimpleNamespace(to_dict=lambda: {"synthetic": True})

    return Model()


def test_original_identities_fresh_hashes_and_strict_binding(assets):
    files = verified(assets)
    assert files.files["config.json"]["verified_original"] == "git_blob_sha1"
    assert files.files["one.safetensors"]["verified_original"] == "sha256"
    entries = inspect_tensors(files)
    model = tiny_model()
    report = strict_load(model, files, entries)
    assert report["complete"] and report["conversions"] == []
    assert torch.equal(model.state_dict()[KEY], torch.arange(6).reshape(2, 3))
    assert report["tensors"][KEY]["dtype"] == "torch.float32"


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "changed",
        "symlink",
        "duplicate",
        "traversal",
        "wrong_prefix",
        "wrong_git",
        "wrong_sha",
        "wrong_size",
        "receipt",
    ],
)
def test_manifest_rejections(assets, mutation):
    root, manifest, digest = assets
    data = json.loads(manifest.read_bytes())
    if mutation == "missing":
        (root / "config.json").unlink()
    elif mutation == "extra":
        (root / "extra.bin").write_bytes(b"x")
    elif mutation == "changed":
        (root / "config.json").write_bytes(b"not original")
    elif mutation == "symlink":
        target = root.parent / "alias.json"
        (root / "config.json").rename(target)
        (root / "config.json").symlink_to(target)
    elif mutation == "duplicate":
        data["files"].append(data["files"][0])
    elif mutation == "traversal":
        data["files"][0]["path"] = "nvidia/../config.json"
    elif mutation == "wrong_prefix":
        data["files"][0]["path"] = "arbitrary/config.json"
    elif mutation == "wrong_git":
        data["files"][0]["git_blob_sha1"] = "0" * 40
    elif mutation == "wrong_sha":
        next(e for e in data["files"] if "sha256" in e)["sha256"] = "0" * 64
    elif mutation == "wrong_size":
        data["files"][0]["bytes"] += 1
    elif mutation == "receipt":
        data = {"files": data["files"], "verified": True}
    if mutation in {
        "duplicate",
        "traversal",
        "wrong_prefix",
        "wrong_git",
        "wrong_sha",
        "wrong_size",
        "receipt",
    }:
        manifest.write_text(json.dumps(data))
        digest = hashlib.sha256(manifest.read_bytes()).hexdigest()
    with pytest.raises((CheckpointError, FileNotFoundError)):
        verified((root, manifest, digest))


def test_exact_prefix_and_trust_digest_are_mandatory(assets):
    root, manifest, digest = assets
    for prefix in (None, "", "nvidia", "other/"):
        with pytest.raises(CheckpointError):
            verify_manifest(
                root, manifest, expected_sha256=digest, kind="checkpoint", variant_prefix=prefix
            )
    with pytest.raises(CheckpointError, match="manifest SHA256"):
        verified((root, manifest, "0" * 64))


@pytest.mark.parametrize("text", ['{"a":1,"a":2}', '{"a":{"x":1,"x":2}}', '{"a":NaN}', "[]"])
def test_ambiguous_json_fails(text):
    with pytest.raises(ValueError):
        json_object(text.encode())


@pytest.mark.parametrize("name", ["../x", "/x", "a/../x", "a//x", "a/./x", "a\\x"])
def test_unsafe_paths(tmp_path, name):
    with pytest.raises(CheckpointError):
        safe_path(tmp_path, name)


def test_symlink_ancestor_even_when_contained(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "x").write_bytes(b"x")
    (tmp_path / "link").symlink_to(real, target_is_directory=True)
    with pytest.raises(CheckpointError, match="symlink"):
        safe_path(tmp_path / "link", "x")


@pytest.mark.parametrize(
    "mutation",
    [
        "packed",
        "missing_key",
        "extra_key",
        "wrong_shard",
        "duplicate_header",
        "gap",
        "overlap",
        "trailing",
        "dtype",
        "shape",
        "total_size",
    ],
)
def test_index_and_tensor_failures(assets, mutation):
    root, manifest, _ = assets
    index_path = root / "model.safetensors.index.json"
    index = json.loads(index_path.read_bytes())
    shard = root / "one.safetensors"
    raw = shard.read_bytes()
    length = struct.unpack("<Q", raw[:8])[0]
    header, payload = json.loads(raw[8 : 8 + length]), raw[8 + length :]
    if mutation == "packed":
        index["weight_map"][KEY.replace("experts.0.", "experts.")] = index["weight_map"].pop(KEY)
    elif mutation == "missing_key":
        del header["bias"]
    elif mutation == "extra_key":
        header["alien"] = header["bias"]
    elif mutation == "wrong_shard":
        index["weight_map"]["bias"] = "absent.safetensors"
    elif mutation == "gap":
        header[KEY]["data_offsets"] = [20, 44]
    elif mutation == "overlap":
        header[KEY]["data_offsets"] = [0, 24]
    elif mutation == "trailing":
        payload += b"extra"
    elif mutation == "dtype":
        header[KEY]["dtype"] = "UNSUPPORTED"
    elif mutation == "shape":
        header[KEY]["shape"] = [300]
    elif mutation == "total_size":
        index["metadata"]["total_size"] += 1
    encoded = json.dumps(header).encode()
    if mutation == "duplicate_header":
        encoded = encoded[:-1] + b', "bias":' + json.dumps(header["bias"]).encode() + b"}"
    encoded += b" " * (-len(encoded) % 8)
    shard.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    index_path.write_text(json.dumps(index))
    digest = manifest_for(root, manifest)
    with pytest.raises(CheckpointError):
        inspect_tensors(verified((root, manifest, digest)))


@pytest.mark.parametrize("mutation", ["key", "shape", "dtype"])
def test_constructed_model_must_exactly_match_before_assignment(assets, mutation):
    files = verified(assets)
    entries = inspect_tensors(files)
    model = tiny_model()
    if mutation == "key":
        model.extra = torch.nn.Parameter(torch.zeros(1))
    elif mutation == "shape":
        model.bias = torch.nn.Parameter(torch.zeros(3))
    else:
        model.double()
    original = {k: t.clone() for k, t in model.state_dict().items()}
    with pytest.raises(CheckpointError, match="mismatch"):
        strict_load(model, files, entries)
    assert all(torch.equal(t, original[k]) for k, t in model.state_dict().items())


def test_changed_bytes_after_verification_fail(assets):
    files = verified(assets)
    (assets[0] / "config.json").write_text("{}")
    with pytest.raises(CheckpointError, match="changed"):
        files.read("config.json")


@pytest.mark.parametrize(
    "mutation", ["parameter", "data", "forward", "config", "training", "object"]
)
def test_model_binding_detects_mutation(assets, mutation):
    files = verified(assets)
    model = tiny_model()
    strict_load(model, files, inspect_tensors(files))
    model.eval()
    bound = BoundCheckpoint(
        model, SimpleNamespace(), SimpleNamespace(verify_binding=lambda: None), {"synthetic": True}
    )
    if mutation == "parameter":
        with torch.no_grad():
            model.bias.add_(1)
    elif mutation == "data":
        model.bias.data.add_(1)
    elif mutation == "forward":
        model.forward = lambda x: x
    elif mutation == "config":
        model.config = SimpleNamespace(to_dict=lambda: {"changed": True})
    elif mutation == "training":
        model.train()
    else:
        bound.model = tiny_model().eval()
    with pytest.raises(CheckpointError, match="binding|bytes"):
        bound.verify_binding(full=True)


def test_xpu_import_is_explicitly_unavailable_without_cuda_stub():
    with pytest.raises(UnsupportedRuntimeError, match="get_device_capability"):
        with loader.pinned_imports(None, device=torch.device("xpu:0")):
            pytest.fail("must not import model")


def test_import_guard_rejects_unverified_source_before_execution(tmp_path, monkeypatch):
    source = tmp_path / "source"
    (source / "generators").mkdir(parents=True)
    marker = tmp_path / "executed"
    (source / "generators" / "__init__.py").write_text(f"open({str(marker)!r}, 'w').write('bad')")
    files = loader.VerifiedFiles(source, "synthetic", {}, {"revision": loader.SOURCE_REVISION})
    monkeypatch.delitem(__import__("sys").modules, "generators", raising=False)
    with loader.pinned_imports(files, device=torch.device("cpu"), model_import=False):
        with pytest.raises(CheckpointError, match="unmanifested"):
            __import__("generators")
    assert not marker.exists()


def test_import_guard_executes_verified_source_without_stale_bytecode(tmp_path, monkeypatch):
    import importlib
    import sys

    source = tmp_path / "source"
    (source / "generators").mkdir(parents=True)
    path = source / "generators" / "__init__.py"
    path.write_text("value = 37\n")
    files = loader.VerifiedFiles(
        source,
        "synthetic",
        {
            "generators/__init__.py": {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        },
        {"revision": loader.SOURCE_REVISION},
    )
    monkeypatch.delitem(sys.modules, "generators", raising=False)
    try:
        with loader.pinned_imports(files, device=torch.device("cpu"), model_import=False):
            module = importlib.import_module("generators")
            assert module.value == 37
            records = loader.imported_code(files)
            assert next(r for r in records if r["module"] == "generators")["pinned_source"]
    finally:
        sys.modules.pop("generators", None)


def test_traversal_root_and_nonregular_manifest_fail_without_opening(tmp_path):
    import os

    with pytest.raises(CheckpointError, match="traversal"):
        safe_path(tmp_path / ".." / "alias")
    root = tmp_path / "nvidia"
    root.mkdir()
    fifo = tmp_path / "manifest.fifo"
    os.mkfifo(fifo)
    with pytest.raises(CheckpointError, match="regular"):
        verify_manifest(
            root, fifo, expected_sha256="0" * 64, kind="checkpoint", variant_prefix="nvidia/"
        )
