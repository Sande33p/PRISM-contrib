"""Synthetic local binding fixtures; no checkpoint/tokenizer assets or downloads."""

import hashlib
import json
import socket
from dataclasses import FrozenInstanceError

import pytest
from src.backends.u2_text_renderer import RENDERER_REVISION, tokenizer_digest
from src.backends.u2_text_tokenizer import BoundTextTokenizer, bind_local_tokenizer

# Literal upstream contract, independent of production constants. Dense dummy
# IDs are necessary to exercise these pinned IDs with a real Rust tokenizer.
# The fixture is generated in memory; it is not a pretrained tokenizer asset.
SPECIALS = {
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


def encode_json(value):
    return json.dumps(value, ensure_ascii=True).encode()


@pytest.fixture(scope="module")
def artifacts():
    from tokenizers import Tokenizer, pre_tokenizers
    from tokenizers.models import WordLevel

    tokens = [f"unused_{index}" for index in range(157173)]
    tokens[0] = "[UNK]"
    for index in range(1, 128):
        tokens[index] = chr(index)
    tokens[960] = "π"
    for token, index in SPECIALS.items():
        tokens[index] = token
    backend = Tokenizer(
        WordLevel(dict(zip(tokens, range(len(tokens)), strict=True)), unk_token="[UNK]")
    )
    backend.pre_tokenizer = pre_tokenizers.Split("", behavior="isolated")
    backend.add_special_tokens(list(SPECIALS))
    return {
        "tokenizer.json": backend.to_str().encode(),
        "tokenizer_config.json": encode_json(
            {
                "tokenizer_class": "Qwen2Tokenizer",
                "unk_token": "[UNK]",
                "model_max_length": 4096,
                "clean_up_tokenization_spaces": False,
            }
        ),
    }


@pytest.fixture(scope="module")
def aurora_metadata_artifacts(artifacts):
    """Synthetic vocabulary with the operator-observed JSON metadata shape.

    Only scalar metadata, special-map keys and the decoder count are matched;
    these generated tokens/IDs are not a copy of the checkpoint tokenizer.
    """
    from tokenizers import Tokenizer

    backend = Tokenizer.from_str(artifacts["tokenizer.json"].decode())
    backend.add_special_tokens(
        ["<|startoftext|>", "<|endoftext|>", "[CLS]", "[gMASK]"]
        + [f"<synthetic_{index}>" for index in range(265)]
    )
    serialized = backend.to_str().encode()
    decoder = {
        str(entry["id"]): {key: value for key, value in entry.items() if key != "id"}
        for entry in json.loads(serialized)["added_tokens"]
    }
    assert len(decoder) == 286
    specials = {
        "bos_token": "<|startoftext|>",
        "cls_token": "[CLS]",
        "eos_token": "<|endoftext|>",
        "mask_token": "<|mask|>",
        "pad_token": "<|endoftext|>",
    }
    return {
        "tokenizer.json": serialized,
        "special_tokens_map.json": encode_json(specials),
        "tokenizer_config.json": encode_json(
            {
                **specials,
                "add_bos_token": False,
                "add_eos_token": False,
                "added_tokens_decoder": decoder,
                "clean_up_tokenization_spaces": False,
                "extra_special_tokens": {},
                "fast_tokenizer": True,
                "gmask_token": "[gMASK]",
                "merges_file": None,
                "model_max_length": 32768,
                "padding_side": "right",
                "tokenizer_class": "PreTrainedTokenizerFast",
                "trust_remote_code": True,
            }
        ),
    }


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("network/hub construction is forbidden")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    import transformers

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", forbidden)
    monkeypatch.setattr(transformers.PreTrainedTokenizerFast, "from_pretrained", forbidden)


def bind(artifacts):
    return bind_local_tokenizer(artifacts, expected_digest=tokenizer_digest(artifacts))


def test_constructed_object_prefix_identity_and_owned_snapshot(artifacts):
    caller = dict(artifacts)
    bound = bind(caller)
    tokenizer = bound.tokenizer
    assert bound.tokenizer is tokenizer
    caller.clear()
    result = bound.render(system_prompt="S", user_text="U")
    assert result.prompt_ids == (157153, 83, 157154, 157155, 85, 157157)
    assert 157165 not in result.prompt_ids  # Integration appends BOA once.
    assert result.renderer_revision == bound.renderer_revision == RENDERER_REVISION
    pairs = sorted((name, hashlib.sha256(data).hexdigest()) for name, data in artifacts.items())
    digest = (
        "sha256:"
        + hashlib.sha256(
            b"u2-tokenizer-artifacts-v1\0" + json.dumps(pairs, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert result.tokenizer_digest == bound.tokenizer_digest == digest
    assert bound.artifact_hashes == tuple(pairs)
    assert bound.vocab_size == 157173
    assert dict(bound.special_token_ids) == SPECIALS
    assert bound.construction_provenance["artifact_digest"] == digest
    assert bound.construction_provenance["tokenizers_version"]
    assert bound.construction_provenance["transformers_version"]
    assert tokenizer.convert_tokens_to_ids("S") == 83
    assert tokenizer.decode([83], skip_special_tokens=True) == "S"
    bound.verify_binding()
    with pytest.raises(TypeError):
        bound.special_token_ids["<answer>"] = 2
    with pytest.raises(TypeError):
        bound.construction_provenance["constructor"] = "unrelated"
    with pytest.raises(FrozenInstanceError):
        bound.vocab_size = 3
    with pytest.raises(TypeError, match="bind_local_tokenizer"):
        BoundTextTokenizer()
    with pytest.raises(TypeError):
        bind_local_tokenizer(artifacts, expected_digest=digest, tokenizer=tokenizer)


def test_render_preserves_verbatim_text_truncation_and_repeated_use(artifacts):
    bound = bind(artifacts)
    assert bound.render(system_prompt=" \n", user_text="π\t").prompt_ids == (
        157153,
        32,
        10,
        157154,
        157155,
        960,
        9,
        157157,
    )
    assert (
        bound.render(system_prompt="", user_text="x" * 4096).prompt_ids
        == (
            157153,
            157154,
            157155,
        )
        + (120,) * 4093
    )
    assert bound.render(system_prompt="", user_text="").prompt_ids == (
        157153,
        157154,
        157155,
        157157,
    )
    for marker in ("<answer>", "<think>", "<IMG_CONTEXT>"):
        with pytest.raises(ValueError, match="reserved"):
            bound.render(system_prompt="", user_text=marker)
    bound.verify_binding()


def test_changed_verified_bytes_change_actual_constructed_tokenizer(artifacts):
    changed = json.loads(artifacts["tokenizer.json"])
    vocab = changed["model"]["vocab"]
    vocab["S"], vocab["U"] = vocab["U"], vocab["S"]
    other = {**artifacts, "tokenizer.json": encode_json(changed)}
    with pytest.raises(ValueError, match="expected_digest"):
        bind_local_tokenizer(other, expected_digest=tokenizer_digest(artifacts))
    bound = bind(other)
    assert bound.render(system_prompt="S", user_text="U").prompt_ids == (
        157153,
        85,
        157154,
        157155,
        83,
        157157,
    )
    assert bound.tokenizer_digest != tokenizer_digest(artifacts)


@pytest.mark.parametrize("digest", ["", "sha256:" + "0" * 64, None, 1])
def test_digest_checked_before_any_construction(artifacts, monkeypatch, digest):
    import src.backends.u2_text_tokenizer as module

    def forbidden(*args):
        raise AssertionError("construction happened before verification")

    monkeypatch.setattr(module, "_construct", forbidden)
    with pytest.raises(ValueError, match="expected_digest"):
        bind_local_tokenizer(artifacts, expected_digest=digest)


def test_data_subclasses_cannot_substitute_construction_bytes_or_digest(artifacts):
    class DifferentBytes(bytes):
        def decode(self, *args, **kwargs):
            raise AssertionError("substituted bytes reached construction")

    class EqualDigest(str):
        def __ne__(self, other):
            return False

    with pytest.raises(ValueError, match="immutable bytes"):
        bind_local_tokenizer(
            {**artifacts, "tokenizer.json": DifferentBytes(artifacts["tokenizer.json"])},
            expected_digest=tokenizer_digest(artifacts),
        )
    with pytest.raises(ValueError, match="expected_digest"):
        bind_local_tokenizer(artifacts, expected_digest=EqualDigest("wrong"))


@pytest.mark.parametrize(
    "name", ["../tokenizer.json", "/tokenizer.json", "x\\y", "model.py", "vocab.json"]
)
def test_unexpected_or_unsafe_files_rejected(artifacts, name):
    with pytest.raises(ValueError, match="basenames"):
        bind_local_tokenizer(
            {**artifacts, name: b"{}"}, expected_digest=tokenizer_digest(artifacts)
        )


@pytest.mark.parametrize("name", ["tokenizer.json", "tokenizer_config.json"])
@pytest.mark.parametrize("bad", [None, b"", "{}", bytearray(b"{}")])
def test_missing_empty_or_mutable_required_bytes_rejected(artifacts, name, bad):
    candidate = dict(artifacts)
    if bad is None:
        candidate.pop(name)
    else:
        candidate[name] = bad
    with pytest.raises(ValueError):
        bind_local_tokenizer(candidate, expected_digest=tokenizer_digest(artifacts))


@pytest.mark.parametrize("raw", [b"[]", b"null", b"\xff", b"{", b'{"x": 1, "x": 2}'])
def test_invalid_json_rejected(artifacts, raw):
    with pytest.raises(ValueError, match="JSON"):
        bind({**artifacts, "tokenizer_config.json": raw})


@pytest.mark.parametrize(
    "config",
    [
        {"tokenizer_file": "/unverified/tokenizer.json"},
        {"auto_map": {"AutoTokenizer": "untrusted.Code"}},
        {"trust_remote_code": True},
        {"tokenizer_class": "UntrustedTokenizer"},
        {"truncation_side": "left"},
        {"model_max_length": 8},
        {"split_special_tokens": True},
        {"add_bos_token": True},
        {"added_tokens_decoder": {"01": "S"}},
    ],
)
def test_external_sources_custom_code_and_incompatible_defaults_rejected(artifacts, config):
    with pytest.raises(ValueError):
        bind({**artifacts, "tokenizer_config.json": encode_json(config)})


def test_aurora_metadata_is_inert_and_preserves_binding(aurora_metadata_artifacts, monkeypatch):
    import transformers
    from tokenizers import Tokenizer

    legacy_keys = {
        "extra_special_tokens",
        "fast_tokenizer",
        "gmask_token",
        "merges_file",
        "trust_remote_code",
    }
    calls = []
    original_init = transformers.PreTrainedTokenizerFast.__init__

    def checked_init(self, *args, **kwargs):
        assert not args
        assert legacy_keys.isdisjoint(kwargs)
        assert kwargs["local_files_only"] is True
        assert isinstance(kwargs["tokenizer_object"], Tokenizer)
        calls.append(kwargs)
        original_init(self, **kwargs)

    monkeypatch.setattr(transformers.PreTrainedTokenizerFast, "__init__", checked_init)
    bound = bind(aurora_metadata_artifacts)
    assert len(calls) == 1
    assert type(bound.tokenizer) is transformers.PreTrainedTokenizerFast
    assert bound.render(system_prompt="S", user_text="U").prompt_ids == (
        157153,
        83,
        157154,
        157155,
        85,
        157157,
    )
    assert dict(bound.special_token_ids) == SPECIALS
    assert bound.artifact_hashes == tuple(
        sorted(
            (name, hashlib.sha256(data).hexdigest())
            for name, data in aurora_metadata_artifacts.items()
        )
    )
    # The ignored metadata stays in the verified bytes/digest, not HF kwargs.
    config = json.loads(aurora_metadata_artifacts["tokenizer_config.json"])
    sanitized = {key: value for key, value in config.items() if key not in legacy_keys}
    other = {**aurora_metadata_artifacts, "tokenizer_config.json": encode_json(sanitized)}
    with pytest.raises(ValueError, match="expected_digest"):
        bind_local_tokenizer(other, expected_digest=bound.tokenizer_digest)
    clean = bind(other)
    assert clean.tokenizer_digest != bound.tokenizer_digest
    assert (
        clean.construction_provenance["object_state_sha256"]
        == (bound.construction_provenance["object_state_sha256"])
    )
    assert clean.render(system_prompt="S", user_text="U").prompt_ids == (
        bound.render(system_prompt="S", user_text="U").prompt_ids
    )
    bound.verify_binding()
    for sidecar in (
        {"special_tokens_map.json": encode_json({"eos_token": "different"})},
        {"added_tokens.json": encode_json({"<answer>": 157166})},
    ):
        with pytest.raises(ValueError, match="conflicting|disagrees"):
            bind({**aurora_metadata_artifacts, **sidecar})


@pytest.mark.parametrize(
    "key,value",
    [
        ("extra_special_tokens", []),
        ("extra_special_tokens", None),
        ("extra_special_tokens", {"custom_token": "<custom>"}),
        ("fast_tokenizer", False),
        ("fast_tokenizer", 1),
        ("fast_tokenizer", "true"),
        ("fast_tokenizer", None),
        ("gmask_token", "[MASK]"),
        ("gmask_token", {"content": "[gMASK]"}),
        ("gmask_token", None),
        ("merges_file", "merges.txt"),
        ("merges_file", "../merges.txt"),
        ("merges_file", "/unverified/merges.txt"),
        ("merges_file", "https://example.invalid/merges.txt"),
        ("merges_file", False),
        ("merges_file", []),
        ("trust_remote_code", False),
        ("trust_remote_code", 1),
        ("trust_remote_code", "true"),
        ("trust_remote_code", {"code": "untrusted.Code"}),
        ("trust_remote_code", None),
    ],
)
def test_aurora_metadata_unsupported_variants_rejected(aurora_metadata_artifacts, key, value):
    config = json.loads(aurora_metadata_artifacts["tokenizer_config.json"])
    config[key] = value
    with pytest.raises(ValueError, match=f"unsupported legacy tokenizer metadata: {key}"):
        bind({**aurora_metadata_artifacts, "tokenizer_config.json": encode_json(config)})


@pytest.mark.parametrize("tokenizer_class", [None, "Qwen2Tokenizer", "UntrustedTokenizer"])
def test_aurora_metadata_requires_explicit_generic_class(
    aurora_metadata_artifacts, tokenizer_class
):
    config = json.loads(aurora_metadata_artifacts["tokenizer_config.json"])
    if tokenizer_class is None:
        config.pop("tokenizer_class")
    else:
        config["tokenizer_class"] = tokenizer_class
    with pytest.raises(ValueError, match="unsupported legacy metadata tokenizer class"):
        bind({**aurora_metadata_artifacts, "tokenizer_config.json": encode_json(config)})


@pytest.mark.parametrize(
    "directive",
    [
        {"auto_map": {"AutoTokenizer": ["untrusted.Code", "untrusted.FastCode"]}},
        {"auto_map": {}},
        {"auto_map": None},
        {"tokenizer_file": "tokenizer.json"},
        {"tokenizer_file": "/unverified/tokenizer.json"},
        {"tokenizer_file": "https://example.invalid/tokenizer.json"},
        {"vocab_file": "../vocab.json"},
        {"vocab_file": "/unverified/vocab.json"},
        {"vocab_file": None},
        {"name_or_path": "untrusted/repository"},
        {"custom_code": "untrusted.Code"},
        {"unknown_metadata": None},
    ],
)
def test_aurora_metadata_does_not_admit_code_or_paths(
    aurora_metadata_artifacts, directive, monkeypatch
):
    import transformers

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported directive reached construction")

    monkeypatch.setattr(transformers, "PreTrainedTokenizerFast", forbidden)
    config = json.loads(aurora_metadata_artifacts["tokenizer_config.json"])
    config.update(directive)
    with pytest.raises(ValueError, match="unsupported tokenizer configuration fields"):
        bind({**aurora_metadata_artifacts, "tokenizer_config.json": encode_json(config)})


def test_optional_sidecars_are_consumed_and_bound(artifacts):
    candidate = {
        **artifacts,
        "special_tokens_map.json": encode_json({"unk_token": "[UNK]"}),
        "added_tokens.json": encode_json({"<answer>": 157165}),
    }
    bound = bind(candidate)
    assert len(bound.artifact_hashes) == 4
    assert bound.tokenizer_digest != tokenizer_digest(artifacts)
    assert bound.render(system_prompt="S", user_text="U").prompt_ids[1] == 83
    for extra in [
        {"added_tokens.json": encode_json({"<answer>": 157166})},
        {"added_tokens.json": encode_json({"unknown": 157173})},
        {"special_tokens_map.json": encode_json({"unk_token": "different"})},
        {"special_tokens_map.json": encode_json({"tokenizer_file": "elsewhere"})},
    ]:
        with pytest.raises(ValueError):
            bind({**candidate, **extra})


def test_config_added_token_decoder_enforces_declared_ids(artifacts):
    config = json.loads(artifacts["tokenizer_config.json"])
    config["added_tokens_decoder"] = {"157165": {"content": "<answer>", "special": True}}
    bound = bind({**artifacts, "tokenizer_config.json": encode_json(config)})
    assert bound.tokenizer.convert_tokens_to_ids("<answer>") == 157165
    config["added_tokens_decoder"] = {"157166": {"content": "<answer>", "special": True}}
    with pytest.raises(ValueError, match="disagrees"):
        bind({**artifacts, "tokenizer_config.json": encode_json(config)})


@pytest.mark.parametrize("failure", ["wrong_id", "nonatomic", "small_vocab"])
def test_incompatible_vocab_or_special_behavior_rejected(artifacts, failure):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel

    data = json.loads(artifacts["tokenizer.json"])
    if failure == "wrong_id":
        vocab = data["model"]["vocab"]
        vocab["<answer>"], vocab["<user>"] = vocab["<user>"], vocab["<answer>"]
    elif failure == "nonatomic":
        data["added_tokens"] = [
            entry for entry in data["added_tokens"] if entry["content"] != "<answer>"
        ]
    else:
        data = json.loads(Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")).to_str())
    with pytest.raises(ValueError):
        bind({**artifacts, "tokenizer.json": encode_json(data)})


@pytest.mark.parametrize("fixture_name", ["artifacts", "aurora_metadata_artifacts"])
@pytest.mark.parametrize(
    "mutation",
    [
        "add_token",
        "normalizer",
        "postprocessor",
        "truncation_side",
        "special_map",
        "method",
        "backend",
        "class_method",
    ],
)
def test_object_mutations_invalidate_binding_before_use(
    request, fixture_name, monkeypatch, mutation
):
    from tokenizers import Tokenizer, normalizers, processors

    bound = bind(request.getfixturevalue(fixture_name))
    tokenizer = bound.tokenizer
    if mutation == "add_token":
        tokenizer.add_tokens(["new_token"])
    elif mutation == "normalizer":
        tokenizer.backend_tokenizer.normalizer = normalizers.Lowercase()
    elif mutation == "postprocessor":
        tokenizer.backend_tokenizer.post_processor = processors.TemplateProcessing(
            single="$A <answer>", special_tokens=[("<answer>", 157165)]
        )
    elif mutation == "truncation_side":
        tokenizer.truncation_side = "left"
    elif mutation == "special_map":
        tokenizer.unk_token = "S"
    elif mutation == "method":
        tokenizer.convert_tokens_to_ids = lambda token: 0
    elif mutation == "backend":
        tokenizer._tokenizer = Tokenizer.from_str(tokenizer.backend_tokenizer.to_str())
    else:
        monkeypatch.setattr(type(tokenizer), "__call__", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="invalidated"):
        bound.verify_binding()
    with pytest.raises(ValueError, match="invalidated"):
        bound.render(system_prompt="S", user_text="U")
    with pytest.raises(ValueError, match="invalidated"):
        _ = bound.tokenizer
