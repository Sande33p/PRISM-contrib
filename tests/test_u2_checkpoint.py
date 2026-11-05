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


def test_indexed_xpu_admission_does_not_query_a_device(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda: pytest.fail("CUDA query"))
    loader.require_import_runtime(torch.device("xpu:0"))
    with pytest.raises(UnsupportedRuntimeError):
        loader.require_import_runtime(torch.device("xpu"))


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
    import os
    import py_compile

    path.write_text("value = 99\n")
    stamp = path.stat()
    py_compile.compile(str(path), doraise=True)
    path.write_text("value = 37\n")
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    ordinary = {}
    exec(
        importlib.machinery.SourceFileLoader("stale_probe", str(path)).get_code("stale_probe"),
        ordinary,
    )
    assert ordinary["value"] == 99  # Timestamp/size-valid pyc really contains other code.
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


@pytest.mark.parametrize("name", ["veomni_patch", "liger_kernel"])
@pytest.mark.parametrize("location", ["root", "vendored", "external", "loaded"])
def test_optional_code_rejected_on_effective_path_before_side_effect(
    tmp_path, monkeypatch, name, location
):
    import sys
    from types import ModuleType

    source = tmp_path / "source"
    (source / "VeOmni").mkdir(parents=True)
    marker = tmp_path / "executed"
    if location == "loaded":
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    else:
        target = source if location == "root" else source / "VeOmni"
        if location == "external":
            target = tmp_path / "external"
            target.mkdir()
            monkeypatch.syspath_prepend(str(target))
        (target / (name + ".py")).write_text(f"open({str(marker)!r}, 'w').write('bad')")
    files = loader.VerifiedFiles(source, "synthetic", {}, {})
    before_path, before_meta = list(sys.path), list(sys.meta_path)
    with pytest.raises(UnsupportedRuntimeError, match=name):
        with loader.pinned_imports(files, device=torch.device("cpu"), model_import=False):
            pytest.fail("optional code admitted")
    assert not marker.exists()
    assert sys.path == before_path and sys.meta_path == before_meta


@pytest.mark.parametrize(
    "mutation",
    ["callable", "child_callable", "child_module", "helper", "partial", "dimension", "buffer"],
)
def test_execution_structure_binding_rejects_ordinary_mutation(assets, mutation):
    from functools import partial

    model = tiny_model().eval()
    model.act_fn = torch.abs
    model.child = torch.nn.Linear(2, 2).eval()
    model.child.act_fn = torch.abs
    model.choice = partial(torch.add, other=2)
    model.register_buffer("inv_freq", torch.tensor([2.0]), persistent=False)
    bound = BoundCheckpoint(
        model, SimpleNamespace(), SimpleNamespace(verify_binding=lambda: None), {}
    )
    bound.verify_binding(full=True)
    if mutation == "callable":
        model.act_fn = torch.neg
    elif mutation == "child_callable":
        model.child.act_fn = torch.neg
    elif mutation == "child_module":
        model.child = torch.nn.Linear(2, 2).eval()
    elif mutation == "helper":
        model.child.extra_repr = lambda: "changed"
    elif mutation == "partial":
        model.choice.keywords["other"] = 9
    elif mutation == "dimension":
        model.child.in_features = 999
    else:
        model.inv_freq.add_(1)
    with pytest.raises(CheckpointError, match="binding|bytes"):
        bound.verify_binding(full=True)


def test_consumed_shard_snapshot_survives_swap_and_restoration(assets, monkeypatch):
    from safetensors import torch as st
    from safetensors.torch import save

    files = verified(assets)
    entries = inspect_tensors(files)
    path = assets[0] / "one.safetensors"
    trusted = path.read_bytes()
    alternate = save({KEY: torch.full((2, 3), 9.0), "bias": torch.full((2,), 9.0)})
    real_load = st.load
    consumed = []

    def swap_during_deserialization(owned):
        path.write_bytes(alternate)
        try:
            consumed.append(hashlib.sha256(owned).hexdigest())
            return real_load(owned)
        finally:
            path.write_bytes(trusted)

    monkeypatch.setattr(st, "load", swap_during_deserialization)
    model = tiny_model()
    report = strict_load(model, files, entries)
    assert report["complete"]
    assert consumed == [hashlib.sha256(trusted).hexdigest()]
    assert torch.equal(model.bias, torch.tensor([2.0, 3.0]))
    assert path.read_bytes() == trusted


def test_shard_replacement_before_snapshot_is_rejected(assets):
    files = verified(assets)
    entries = inspect_tensors(files)
    save_file(
        {KEY: torch.full((2, 3), 9.0), "bias": torch.full((2,), 9.0)},
        str(assets[0] / "one.safetensors"),
    )
    with pytest.raises(CheckpointError, match="changed"):
        strict_load(tiny_model(), files, entries)


def pinned_source():
    import os
    from pathlib import Path

    root = Path(os.environ["U2_PINNED_SOURCE_DIR"])
    manifest = root.parent / "upstream-source-manifest.json"
    return verify_manifest(
        root,
        manifest,
        expected_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
        kind="source",
    )


def test_split_adaptation_changes_only_two_import_ast_nodes():
    import ast

    files = pinned_source()
    before = files.read(loader._MODEL_SOURCE)
    after = loader.split_source(before)
    left, right = ast.parse(before), ast.parse(after)
    changes = [
        (a, b) for a, b in zip(left.body, right.body, strict=True) if ast.dump(a) != ast.dump(b)
    ]
    assert len(changes) == 2
    assert all(isinstance(a, ast.ImportFrom) and isinstance(b, ast.ImportFrom) for a, b in changes)
    assert files.read(loader._MODEL_SOURCE) == before
    with pytest.raises(CheckpointError, match="exact pinned"):
        loader.split_source(before + b"\n")
    for symbol in ("fused_moe_forward", "get_parallel_state"):
        with pytest.raises(UnsupportedRuntimeError, match="excluded"):
            loader._unsupported_split_optional(symbol)


@pytest.mark.parametrize(
    "key,value",
    [
        ("model_type", "llada2_moe_veomni"),
        ("num_codebooks", 7),
        ("num_experts", 2),
        ("num_hidden_layers", 2),
        ("first_k_dense_replace", 0),
        ("num_shared_experts", 0),
    ],
)
def test_production_split_architecture_rejects_other_layouts(key, value):
    config = dict(
        model_type="llada2_moe",
        num_codebooks=8,
        torch_dtype="bfloat16",
        vocab_size=157184,
        num_hidden_layers=20,
        first_k_dense_replace=1,
        num_experts=256,
        num_shared_experts=1,
        _moe_implementation="fused",
    )
    config.update(
        hidden_size=2048,
        intermediate_size=5120,
        moe_intermediate_size=512,
        num_attention_heads=16,
        num_key_value_heads=4,
        head_dim=128,
        num_experts_per_tok=8,
        n_group=8,
        topk_group=4,
        max_position_embeddings=16384,
        use_cache=False,
        use_sliding_window=False,
    )
    loader.validate_split_config(config)
    config[key] = value
    with pytest.raises(CheckpointError):
        loader.validate_split_config(config)


def test_tiny_genuine_pinned_split_cpu_forward_and_weights(tmp_path):
    """Existing genuine dependencies only; fixed <8M parameters, <256 MiB tensor budget.

    Every configurable allocation dimension is explicit. Pinned hardcoded sizes:
    text embedding/head=157184, image embeddings=8*5010, image heads=8*4096,
    tokenbridge depth=6, heads=8, width=16, expansion=4. No pretrained assets.
    Run this one test in the existing compatible prism_env, separately from py310.
    """
    import importlib
    import sys
    from importlib import metadata

    from src.backends.text_contracts import RuntimeIdentity, TextRequest
    from src.backends.u2_preflight import U2PreflightReport
    from src.backends.u2_text_renderer import EXPECTED_SPECIAL_TOKEN_IDS
    from src.backends.u2_text_runtime import LocalModelSpec
    from tools.parity import u2_text as runner

    assert metadata.version("timm") == "1.0.22"
    files = pinned_source()
    before = {k: files.files[k]["sha256"] for k in files.files}
    roots = {"models", "generators", "utils", "veomni"}
    prior = {k: v for k, v in sys.modules.items() if k.split(".")[0] in roots}
    assert not prior, "tiny real-model attempt requires a fresh interpreter"
    threads = torch.get_num_threads()
    torch.set_num_threads(1)
    previous_dtype = torch.get_default_dtype()
    try:
        with loader.pinned_imports(files, device=torch.device("cpu")):
            module = importlib.import_module("models.internluminau2.modeling_internluminau2")
            config_module = importlib.import_module(
                "models.internluminau2.configuration_internluminau2"
            )
            values = dict(
                vocab_size=157184,
                hidden_size=16,
                intermediate_size=32,
                num_hidden_layers=2,
                num_attention_heads=2,
                num_key_value_heads=1,
                hidden_act="silu",
                use_qkv_bias=False,
                use_qk_norm=True,
                use_bias=False,
                rms_norm_eps=1e-6,
                norm_head=False,
                tie_word_embeddings=False,
                embedding_dropout=0.0,
                attention_dropout=0.0,
                output_dropout=0.0,
                initializer_range=0.02,
                max_position_embeddings=32,
                rope_theta=10000.0,
                use_cache=False,
                use_sliding_window=False,
                sliding_window=32,
                max_window_layers=2,
                rope_scaling=None,
                pad_token_id=156892,
                num_experts=2,
                num_shared_experts=1,
                num_experts_per_tok=1,
                n_group=1,
                topk_group=1,
                routed_scaling_factor=2.5,
                moe_intermediate_size=16,
                first_k_dense_replace=1,
                head_dim=8,
                output_router_logits=False,
                partial_rotary_factor=0.5,
                num_codebooks=8,
                codebook_size=4096,
                _moe_implementation="fused",
                use_segmented_attention=False,
            )
            config = config_module.LLaDA2MoeConfig(**values)
            config._attn_implementation = "sdpa"
            torch.set_default_dtype(torch.bfloat16)
            # Inspect all allocations without storage before any CPU model allocation.
            with torch.device("meta"):
                sizing = module.LLaDA2MoeModelLM(config)
            parameter_count = sum(p.numel() for p in sizing.parameters())
            state_bytes = sum(t.numel() * t.element_size() for t in sizing.state_dict().values())
            assert parameter_count < 8_000_000 and state_bytes < 32 * 1024 * 1024
            del sizing
            torch.manual_seed(1729)
            model = module.LLaDA2MoeModelLM(config).eval()
            loader.validate_split_model(model, module, config)
            assert len(model.multi_codebook_head.channel_blocks) == 6
            assert model.multi_codebook_head.channel_blocks[0].mlp.__class__.__module__.startswith(
                "timm."
            )
            assert not any(k.startswith(("veomni.ops", "veomni.distributed")) for k in sys.modules)
            for symbol in ("fused_moe_forward", "get_parallel_state"):
                with pytest.raises(UnsupportedRuntimeError):
                    getattr(module, symbol)()
            root = tmp_path / "nvidia"
            root.mkdir()
            state = model.state_dict()
            save_file(state, str(root / "tiny.safetensors"))
            (root / "model.safetensors.index.json").write_text(
                json.dumps(
                    {
                        "weight_map": {k: "tiny.safetensors" for k in state},
                        "metadata": {"total_size": state_bytes},
                    }
                )
            )
            (root / "config.json").write_text(json.dumps(values))
            manifest = tmp_path / "manifest.json"
            digest = manifest_for(root, manifest)
            synthetic = verified((root, manifest, digest))
            torch.manual_seed(1730)
            loaded = module.LLaDA2MoeModelLM(config).eval()
            receipt = strict_load(loaded, synthetic, inspect_tensors(synthetic))
            assert receipt["complete"] and set(loaded.state_dict()) == set(state)
            del state
            x = torch.tensor([41, 157165, 156895])[None, :, None].repeat(1, 1, 8)
            kwargs = dict(
                attention_mask=torch.zeros(1, 1, 3, 3, dtype=torch.bfloat16),
                infer_mmu=True,
                is_clean=torch.tensor([[1, 0, 0]]),
                position_ids=torch.arange(3)[None, :],
            )
            types = torch.tensor([[1, 1, 3]])
            digest = "sha256:" + "a" * 64
            spec = LocalModelSpec(
                RuntimeIdentity(
                    "synthetic-pinned-cpu", loader.SOURCE_REVISION, digest, 157184, 157166, 156895
                ),
                EXPECTED_SPECIAL_TOKEN_IDS,
                torch.bfloat16,
                torch.device("cpu"),
                32,
                prompt_vocab_size=157177,
            )
            tokenizer = SimpleNamespace(
                verify_binding=lambda: None, vocab_size=157177, tokenizer=object()
            )
            bound = BoundCheckpoint(loaded, spec, tokenizer, {})
            with torch.inference_mode():
                expected = model(x, types, **kwargs)
                actual = bound(x, types, **kwargs)
            assert actual.shape == (1, 157184) and torch.isfinite(actual).all()
            assert torch.equal(actual, expected)
            bound.verify_binding(full=True)
            report = U2PreflightReport(
                loader.SOURCE_REVISION,
                True,
                (),
                (),
                (),
                (),
                (),
                digest,
                157184,
                32,
                "bfloat16",
                "split_candidate",
                (),
                True,
                tokenizer_vocab_size=157177,
            )
            generator = importlib.import_module("generators.text_understanding_generator")
            request = TextRequest(
                request_id="synthetic-real-pinned",
                prompt_ids=(41, 42),
                tokenizer_digest=digest,
                max_new_tokens=2,
                block_size=2,
                steps_per_block=3,
                max_model_calls=8,
                seed=1729,
            )
            from pathlib import Path

            fixture = json.loads(
                (Path(__file__).parent / "fixtures/u2_aurora_text_parity_v1.json").read_bytes()
            )
            comparisons = []
            for repetition, order in enumerate((("oracle", "prism"), ("prism", "oracle"))):
                traces = {
                    path: runner.run_path(
                        bound,
                        generator,
                        report,
                        request,
                        path,
                        tmp_path,
                        f"tiny-{repetition}-{path}",
                        repetition=repetition,
                    )
                    for path in order
                }
                comparison = runner.compare_paths(
                    traces["oracle"],
                    traces["prism"],
                    tmp_path,
                    fixture["numeric_tolerances"]["bfloat16"],
                )
                assert comparison["status"] == "matched", comparison
                assert len(traces["oracle"]["calls"]) == 3
                assert any(c["transfer_quota"] == 0 for c in traces["oracle"]["calls"])
                comparisons.append(comparison["status"])
            # No registered buffer/callable replacement can pass the actual object boundary.
            loaded.model.layers[1].mlp.experts[0].act_fn = torch.nn.ReLU()
            with pytest.raises(CheckpointError, match="binding"):
                bound.verify_binding(full=True)
            print(
                json.dumps(
                    {
                        "synthetic_only": True,
                        "parameters": parameter_count,
                        "state_bytes": state_bytes,
                        "comparison_orders": comparisons,
                        "logits_shape": list(actual.shape),
                        "dtype": str(actual.dtype),
                        "adaptation": loader.split_adaptation(files),
                        "versions": {
                            p: metadata.version(p)
                            for p in ("torch", "transformers", "timm", "safetensors")
                        },
                    }
                )
            )
        for name, digest in before.items():
            assert hashlib.sha256(files.read(name)).hexdigest() == digest
    finally:
        torch.set_default_dtype(previous_dtype)
        torch.set_num_threads(threads)
        for name in list(sys.modules):
            if name.split(".")[0] in roots:
                del sys.modules[name]


def test_missing_genuine_dependency_is_actionable_and_import_state_restored(tmp_path, monkeypatch):
    import importlib
    import sys

    source = tmp_path / "source"
    (source / "generators").mkdir(parents=True)
    path = source / "generators" / "__init__.py"
    path.write_text("import pri28_deliberately_unavailable_dependency\n")
    files = loader.VerifiedFiles(
        source,
        "synthetic",
        {
            "generators/__init__.py": {
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
        },
        {},
    )
    monkeypatch.delitem(sys.modules, "generators", raising=False)
    before_path, before_meta = list(sys.path), list(sys.meta_path)
    with pytest.raises(
        UnsupportedRuntimeError,
        match="genuine runtime dependency unavailable: pri28_deliberately_unavailable_dependency",
    ):
        with loader.pinned_imports(files, device=torch.device("cpu"), model_import=False):
            importlib.import_module("generators")
    assert sys.path == before_path and sys.meta_path == before_meta
    assert "generators" not in sys.modules
