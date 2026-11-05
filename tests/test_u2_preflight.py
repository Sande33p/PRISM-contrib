"""Tiny local metadata fixtures; no model code or tensor deserialization."""

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
from src.backends import u2_preflight as preflight
from src.backends.u2_text_renderer import TOKENIZER_ARTIFACT_NAMES, tokenizer_digest


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def token_digest(root):
    return tokenizer_digest(
        {
            name: (root / name).read_bytes()
            for name in TOKENIZER_ARTIFACT_NAMES
            if (root / name).exists()
        }
    )


@pytest.fixture
def bundle(tmp_path, monkeypatch):
    source = tmp_path / "source"
    checkpoint = tmp_path / "checkpoint"
    source.mkdir()
    checkpoint.mkdir()
    # Substitute only the fixture's content hashes; retain revision/manifest logic.
    source_hashes = {}
    for name in preflight.SOURCE_HASHES:
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("raise RuntimeError('must never execute')\n")
        source_hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(preflight, "SOURCE_HASHES", source_hashes)
    write_json(
        source / "source-manifest.json",
        [
            {"path": name, "sha256": digest, "revision": preflight.SOURCE_REVISION}
            for name, digest in source_hashes.items()
        ],
    )
    write_json(
        checkpoint / "config.json",
        {
            "model_type": "fixture_u2",
            "vocab_size": 160000,
            "max_position_embeddings": 8192,
            "torch_dtype": "bfloat16",
        },
    )
    write_json(
        checkpoint / "tokenizer.json",
        {
            "model": {"vocab": {"hello": 1}},
            "added_tokens": [
                {"content": token, "id": value}
                for token, value in preflight.SPECIAL_TOKEN_IDS.items()
            ],
        },
    )
    write_json(checkpoint / "tokenizer_config.json", {"tokenizer_class": "FixtureTokenizer"})
    write_json(
        checkpoint / "model.safetensors.index.json",
        {
            "weight_map": {"model.layers.0.mlp.experts.0.gate_proj.weight": "part.safetensors"},
        },
    )
    (checkpoint / "part.safetensors").write_bytes(b"not real weights")
    for name in (
        "configuration_internluminau2.py",
        "modeling_internluminau2.py",
        "tokenbridge.py",
        "parallel_plan.py",
    ):
        (checkpoint / name).write_text("raise RuntimeError('must never import')\n")
    requirements = preflight.U2PreflightRequirements(
        token_digest(checkpoint), "fixture_u2", required_packages=()
    )
    return source, checkpoint, requirements


def test_complete_metadata_never_qualifies_execution(bundle):
    result = preflight.preflight_u2(*bundle)
    result.require_metadata()
    assert result.source_verified
    assert result.metadata_complete and not result.execution_qualified
    assert result.split_layout == "split_candidate"
    assert result.vocab_size == 160000 and result.max_position_embeddings == 8192
    assert result.dtype == "bfloat16"
    assert result.execution_requirements
    assert all(
        f.matches_expected
        for f in result.files
        if f.area == "source" and f.path != "source-manifest.json"
    )
    assert "checkpoint/part.safetensors" in result.available_artifacts


@pytest.mark.parametrize(
    "name",
    [
        "config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
        "part.safetensors",
        "modeling_internluminau2.py",
    ],
)
def test_missing_artifacts_are_explicit(bundle, name):
    source, checkpoint, req = bundle
    (checkpoint / name).unlink()
    result = preflight.preflight_u2(source, checkpoint, req)
    assert f"checkpoint/{name}" in result.missing
    with pytest.raises(ValueError, match="preflight failed"):
        result.require_metadata()


@pytest.mark.parametrize(
    "name,content",
    [
        ("config.json", "{"),
        ("config.json", "[]"),
        ("model.safetensors.index.json", '{"weight_map": {"a": "x", "a": "y"}}'),
        ("model.safetensors.index.json", '{"weight_map": {}}'),
        ("model.safetensors.index.json", '{"weight_map": {"x": 1}}'),
    ],
)
def test_malformed_metadata_fails_closed(bundle, name, content):
    source, checkpoint, req = bundle
    (checkpoint / name).write_text(content)
    result = preflight.preflight_u2(source, checkpoint, req)
    assert not result.metadata_complete and result.incompatible


@pytest.mark.parametrize(
    "shard", ["../outside.safetensors", "/tmp/outside.safetensors", "..\\outside.safetensors"]
)
def test_shard_paths_cannot_escape(bundle, shard):
    source, checkpoint, req = bundle
    write_json(
        checkpoint / "model.safetensors.index.json", {"weight_map": {"x.experts.0.weight": shard}}
    )
    result = preflight.preflight_u2(source, checkpoint, req)
    assert not result.metadata_complete
    assert any("unsafe" in error for error in result.incompatible)


def test_symlink_escape_is_rejected_before_read(bundle, tmp_path):
    source, checkpoint, req = bundle
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"must not read")
    (checkpoint / "part.safetensors").unlink()
    (checkpoint / "part.safetensors").symlink_to(outside)
    result = preflight.preflight_u2(source, checkpoint, req)
    assert any("escapes" in error for error in result.incompatible)
    assert not any(f.path == "part.safetensors" for f in result.files)


@pytest.mark.parametrize(
    "field,value",
    [
        ("vocab_size", True),
        ("vocab_size", 100),
        ("max_position_embeddings", 0),
        ("model_type", "wrong"),
        ("torch_dtype", "float16"),
    ],
)
def test_incompatible_config(bundle, field, value):
    source, checkpoint, req = bundle
    config = json.loads((checkpoint / "config.json").read_text())
    config[field] = value
    write_json(checkpoint / "config.json", config)
    assert not preflight.preflight_u2(source, checkpoint, req).metadata_complete


def test_wrong_token_identity_even_with_matching_digest(bundle):
    source, checkpoint, req = bundle
    tokenizer = json.loads((checkpoint / "tokenizer.json").read_text())
    tokenizer["added_tokens"][0]["id"] = 157147
    write_json(checkpoint / "tokenizer.json", tokenizer)
    result = preflight.preflight_u2(
        source, checkpoint, replace(req, tokenizer_digest=token_digest(checkpoint))
    )
    assert any("incompatible token identity" in error for error in result.incompatible)


def test_wrong_digest_and_checkpoint_hash(bundle):
    source, checkpoint, req = bundle
    result = preflight.preflight_u2(
        source,
        checkpoint,
        replace(req, tokenizer_digest="0" * 64, artifact_hashes={"part.safetensors": "0" * 64}),
    )
    assert any("tokenizer digest" in error for error in result.incompatible)
    assert any("SHA256 mismatch" in error for error in result.incompatible)


@pytest.mark.parametrize("mutation", ["revision", "hash", "duplicate", "escape", "content"])
def test_source_identity_failures(bundle, mutation):
    source, checkpoint, req = bundle
    path = source / "source-manifest.json"
    manifest = json.loads(path.read_text())
    if mutation == "revision":
        manifest[0]["revision"] = "wrong"
    elif mutation == "hash":
        manifest[0]["sha256"] = "0" * 64
    elif mutation == "duplicate":
        manifest.append(manifest[0])
    elif mutation == "escape":
        manifest[0]["path"] = "../escape"
    else:
        (source / manifest[0]["path"]).write_text("changed")
    write_json(path, manifest)
    result = preflight.preflight_u2(source, checkpoint, req)
    assert not result.source_verified and not result.metadata_complete


@pytest.mark.parametrize(
    "key,assessment", [("x.experts.gate_up_proj", "packed"), ("x.attention.weight", "unknown")]
)
def test_split_layout_is_not_assumed(bundle, key, assessment):
    source, checkpoint, req = bundle
    write_json(
        checkpoint / "model.safetensors.index.json", {"weight_map": {key: "part.safetensors"}}
    )
    result = preflight.preflight_u2(source, checkpoint, req)
    assert result.split_layout == assessment and not result.metadata_complete


@pytest.mark.parametrize(
    "projection", ["gate_proj", "up_proj", "down_proj", "gate_up_proj", "w1", "w2", "w3"]
)
@pytest.mark.parametrize("suffix", ["", ".weight"])
def test_packed_experts_override_valid_split_candidate(bundle, projection, suffix):
    source, checkpoint, req = bundle
    write_json(
        checkpoint / "model.safetensors.index.json",
        {
            "weight_map": {
                "model.layers.0.mlp.experts.0.gate_proj.weight": "part.safetensors",
                f"model.layers.1.mlp.experts.{projection}{suffix}": "part.safetensors",
            }
        },
    )
    result = preflight.preflight_u2(source, checkpoint, req)
    assert result.split_layout == "packed"
    assert not result.metadata_complete
    assert any("packed expert keys" in error for error in result.incompatible)


@pytest.mark.parametrize("projection", ["gate_proj", "up_proj", "down_proj"])
@pytest.mark.parametrize("suffix", ["", ".weight"])
def test_split_candidate_requires_upstream_projection(bundle, projection, suffix):
    source, checkpoint, req = bundle
    write_json(
        checkpoint / "model.safetensors.index.json",
        {"weight_map": {f"model.layers.0.mlp.experts.12.{projection}{suffix}": "part.safetensors"}},
    )
    result = preflight.preflight_u2(source, checkpoint, req)
    result.require_metadata()
    assert result.split_layout == "split_candidate"
    assert not result.execution_qualified


@pytest.mark.parametrize(
    "key",
    [
        "model.layers.0.mlp.experts.0.unrelated.weight",
        "model.layers.0.mlp.experts.0.gate_up_proj.weight",
        "model.layers.0.mlp.experts.0.gate_proj.bias",
        "model.layers.0.mlp.experts.0.up_proj.weight.extra",
        "model.layers.0.mlp.experts.other.down_proj.weight",
        "model.layers.0.attention.experts.0.gate_proj.weight",
        "model.layers.0.experts.0.up_proj",
        "model.layers.0.mlp.other.0.down_proj.weight",
    ],
)
def test_unrelated_or_misplaced_expert_keys_are_not_split_candidates(bundle, key):
    source, checkpoint, req = bundle
    write_json(
        checkpoint / "model.safetensors.index.json", {"weight_map": {key: "part.safetensors"}}
    )
    result = preflight.preflight_u2(source, checkpoint, req)
    assert result.split_layout == "unknown"
    assert not result.metadata_complete
    assert any("split layout cannot be established" in error for error in result.incompatible)


def test_missing_runtime_distribution(bundle):
    source, checkpoint, req = bundle
    result = preflight.preflight_u2(
        source, checkpoint, replace(req, required_packages=("prism-nonexistent-preflight-fixture",))
    )
    assert result.package_versions == (("prism-nonexistent-preflight-fixture", None),)
    assert "distribution/prism-nonexistent-preflight-fixture" in result.missing


def test_explicit_nested_config(bundle):
    source, checkpoint, req = bundle
    config = json.loads((checkpoint / "config.json").read_text())
    write_json(checkpoint / "config.json", {"model_type": "fixture_u2", "text_config": config})
    result = preflight.preflight_u2(source, checkpoint, replace(req, config_key="text_config"))
    assert result.metadata_complete


def test_import_without_site_packages():
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            "-S",
            "-c",
            "from src.backends.u2_preflight import preflight_u2; import sys; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules",
        ],
        cwd=Path(__file__).resolve().parents[1],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_nonfinite_json_and_empty_shard(bundle):
    source, checkpoint, req = bundle
    (checkpoint / "config.json").write_text('{"vocab_size": NaN}')
    (checkpoint / "part.safetensors").write_bytes(b"")
    result = preflight.preflight_u2(source, checkpoint, req)
    assert any("non-finite" in error for error in result.incompatible)
    assert any("empty artifact" in error for error in result.incompatible)


def test_optional_tokenizer_symlink_escape(bundle, tmp_path):
    source, checkpoint, req = bundle
    outside = tmp_path / "tokens.json"
    outside.write_text("{}")
    (checkpoint / "added_tokens.json").symlink_to(outside)
    result = preflight.preflight_u2(source, checkpoint, req)
    assert not result.metadata_complete
    assert any("escapes" in error for error in result.incompatible)


@pytest.mark.parametrize("name", ["special_tokens_map.json", "added_tokens.json"])
def test_optional_artifact_is_included_in_shared_identity(bundle, name):
    source, checkpoint, req = bundle
    (checkpoint / name).write_text("{}")
    changed = token_digest(checkpoint)
    assert changed != req.tokenizer_digest
    stale = preflight.preflight_u2(source, checkpoint, req)
    assert not stale.metadata_complete and stale.tokenizer_digest == changed
    result = preflight.preflight_u2(source, checkpoint, replace(req, tokenizer_digest=changed))
    result.require_metadata()
    assert result.tokenizer_digest == changed
    assert all(len(f.sha256) == 64 and ":" not in f.sha256 for f in result.files)


def test_bare_legacy_aggregate_is_rejected_and_bare_file_hashes_still_work(bundle):
    source, checkpoint, req = bundle
    hashes = {
        name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
        for name in ("tokenizer.json", "tokenizer_config.json")
    }
    legacy = hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert not preflight.preflight_u2(
        source, checkpoint, replace(req, tokenizer_digest=legacy)
    ).metadata_complete
    result = preflight.preflight_u2(source, checkpoint, replace(req, artifact_hashes=hashes))
    result.require_metadata()
    assert all(
        f.matches_expected is True
        for f in result.files
        if f.area == "checkpoint" and f.path in hashes
    )
    bad = preflight.preflight_u2(
        source,
        checkpoint,
        replace(req, artifact_hashes={"tokenizer.json": "sha256:" + hashes["tokenizer.json"]}),
    )
    assert not bad.metadata_complete
    assert any("SHA256 mismatch" in message for message in bad.incompatible)


@pytest.mark.parametrize(
    "name",
    ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"],
)
@pytest.mark.parametrize("content", [b"", b"{", b'{"x":1,"x":2}', b'{"x":NaN}', b"\xff"])
def test_bad_tokenizer_artifacts_never_get_an_aggregate_identity(bundle, name, content):
    source, checkpoint, req = bundle
    (checkpoint / name).write_bytes(content)
    result = preflight.preflight_u2(source, checkpoint, req)
    assert not result.metadata_complete
    assert result.tokenizer_digest is None


def test_tokenizer_mutation_during_inspection_is_detected(bundle, monkeypatch):
    source, checkpoint, req = bundle
    original = preflight._metadata_bytes

    def mutated(path):
        content = original(path)
        return content + b" " if path.name == "tokenizer.json" else content

    monkeypatch.setattr(preflight, "_metadata_bytes", mutated)
    result = preflight.preflight_u2(source, checkpoint, req)
    assert not result.metadata_complete and result.tokenizer_digest is None
    assert any("changed tokenizer artifact" in error for error in result.incompatible)


@pytest.mark.parametrize("token", preflight.SPECIAL_TOKEN_IDS)
def test_all_special_ids_are_checked_by_preflight(bundle, token):
    source, checkpoint, req = bundle
    tokenizer = json.loads((checkpoint / "tokenizer.json").read_text())
    next(item for item in tokenizer["added_tokens"] if item["content"] == token)["id"] = 42
    write_json(checkpoint / "tokenizer.json", tokenizer)
    result = preflight.preflight_u2(
        source, checkpoint, replace(req, tokenizer_digest=token_digest(checkpoint))
    )
    assert not result.metadata_complete
    assert any("incompatible token identity" in message for message in result.incompatible)
