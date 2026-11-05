"""Source-derived scalar fixtures; no tokenizer assets or model execution."""

from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest
from src.backends.text_contracts import TextRequest
from src.backends.u2_text_renderer import (
    EXPECTED_SPECIAL_TOKEN_IDS,
    RENDERER_REVISION,
    render_text_prompt,
    tokenizer_digest,
    validate_tokenizer_contract,
)

# Independent literal oracle from pinned utils/tokenizer_contract.py.
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
ARTIFACTS = {"tokenizer.json": b'{"model":"fixture"}', "tokenizer_config.json": b"{}"}


class Tokenizer:
    """Deterministic local fixture with explicit special tokens and character IDs."""

    def __init__(self):
        self.specials = dict(SPECIALS)
        self.calls = []
        self.last_ids = []

    def __len__(self):
        return 160000

    def convert_tokens_to_ids(self, token):
        return self.specials.get(token)

    def __call__(self, text, **kwargs):
        self.calls.append((text, kwargs))
        ids = []
        while text:
            token = next((key for key in SPECIALS if text.startswith(key)), None)
            if token is not None:
                ids.append(SPECIALS[token])
                text = text[len(token) :]
            else:
                ids.append(ord(text[0]))
                text = text[1:]
        if kwargs["truncation"]:
            ids = ids[: kwargs["max_length"]]
        self.last_ids = ids
        return SimpleNamespace(input_ids=ids)

    def apply_chat_template(self, *args, **kwargs):
        raise AssertionError("bundled HF template must never be used")


def render(tokenizer=None, **changes):
    args = dict(system_prompt="S", user_text="U", tokenizer_artifacts=ARTIFACTS)
    args.update(changes)
    return render_text_prompt(tokenizer or Tokenizer(), **args)


def test_literal_source_prefix_and_tokenizer_call():
    tokenizer = Tokenizer()
    result = render(tokenizer)
    assert result.prompt_ids == (157153, 83, 157154, 157155, 85, 157157)
    assert tokenizer.calls[-1] == (
        "<system>S</system><user>U</user>",
        {"truncation": True, "max_length": 4096, "padding": False},
    )
    assert result.renderer_revision == RENDERER_REVISION
    assert "70f21c1df8fa26e514df26ba31bc061be8d3d018" in result.renderer_revision
    assert 157165 not in result.prompt_ids  # BOA belongs to the generation driver.
    assert 157166 not in result.prompt_ids  # EOA belongs to generated output.
    request = TextRequest("fixture", result.prompt_ids, result.tokenizer_digest)
    assert request.prompt_ids == result.prompt_ids
    tokenizer.last_ids.append(999)
    assert result.prompt_ids[-1] == 157157
    with pytest.raises(FrozenInstanceError):
        result.renderer_revision = "changed"


@pytest.mark.parametrize(
    ("system", "user", "expected"),
    [
        ("", "", (157153, 157154, 157155, 157157)),
        (" \n", "π\t", (157153, 32, 10, 157154, 157155, 960, 9, 157157)),
    ],
)
def test_no_whitespace_or_unicode_normalization(system, user, expected):
    assert render(system_prompt=system, user_text=user).prompt_ids == expected


def test_source_truncation_is_preserved_without_restoring_end_marker():
    result = render(system_prompt="", user_text="x" * 4096)
    assert result.prompt_ids == (157153, 157154, 157155) + (120,) * 4093


@pytest.mark.parametrize("marker", [*SPECIALS, "<think>", "</think>"])
@pytest.mark.parametrize("field", ["system_prompt", "user_text"])
def test_reserved_boundary_and_media_input_rejected(marker, field):
    with pytest.raises(ValueError, match="reserved"):
        render(**{field: "before" + marker + "after"})


@pytest.mark.parametrize("bad", [None, 1, True, [], {}, b"text"])
@pytest.mark.parametrize("field", ["system_prompt", "user_text"])
def test_non_text_inputs_rejected(bad, field):
    with pytest.raises(ValueError, match="string"):
        render(**{field: bad})


def test_no_media_history_or_implicit_system_input():
    with pytest.raises(TypeError):
        render(media_token_list=[])
    with pytest.raises(TypeError):
        render(messages=[])
    with pytest.raises(TypeError):
        render_text_prompt(Tokenizer(), user_text="U", tokenizer_artifacts=ARTIFACTS)


@pytest.mark.parametrize("token", SPECIALS)
def test_every_pinned_special_id_is_required(token):
    assert dict(EXPECTED_SPECIAL_TOKEN_IDS) == SPECIALS
    tokenizer = Tokenizer()
    tokenizer.specials[token] = -1
    with pytest.raises(ValueError, match="incompatible"):
        validate_tokenizer_contract(tokenizer)


@pytest.mark.parametrize("bad", [None, True, 157165.0, "157165"])
def test_special_ids_require_exact_integers(bad):
    tokenizer = Tokenizer()
    tokenizer.specials["<answer>"] = bad
    with pytest.raises(ValueError, match="incompatible"):
        validate_tokenizer_contract(tokenizer)


def test_convert_lookup_does_not_hide_nonatomic_encoding():
    class SplitTokenizer(Tokenizer):
        def __call__(self, text, **kwargs):
            return SimpleNamespace(input_ids=[1, 2])

    with pytest.raises(ValueError, match="atomically"):
        render(SplitTokenizer())


@pytest.mark.parametrize(
    "bad",
    [
        [],
        [True],
        [-1],
        [160000],
        [1.5],
        [[1]],
        "1",
        None,
        [1] * 4097,
        [157165],
        [157166],
        [156895],
        [157160],
    ],
)
def test_invalid_rendered_ids_fail_closed(bad):
    class BadOutput(Tokenizer):
        def __call__(self, text, **kwargs):
            if kwargs["truncation"]:
                return SimpleNamespace(input_ids=bad)
            return super().__call__(text, **kwargs)

    with pytest.raises(ValueError):
        render(BadOutput())


def test_identity_covers_artifact_bytes_names_and_configuration():
    digest = tokenizer_digest(ARTIFACTS)
    assert digest == tokenizer_digest(dict(reversed(list(ARTIFACTS.items()))))
    assert digest == render(user_text="different").tokenizer_digest
    for name in ARTIFACTS:
        assert tokenizer_digest({**ARTIFACTS, name: ARTIFACTS[name] + b" "}) != digest
    assert tokenizer_digest({**ARTIFACTS, "special_tokens_map.json": b"{}"}) != digest


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"tokenizer.json": b"{}"},
        {**ARTIFACTS, "tokenizer.json": "{}"},
        {**ARTIFACTS, "tokenizer.json": b""},
        {**ARTIFACTS, "../x": b"x"},
        None,
    ],
)
def test_incomplete_or_malformed_identity_rejected(bad):
    with pytest.raises(ValueError):
        render(tokenizer_artifacts=bad)


def test_artifact_names_are_fixed_and_part_of_the_digest():
    first = {**ARTIFACTS, "special_tokens_map.json": b"{}"}
    renamed = {**ARTIFACTS, "added_tokens.json": b"{}"}
    assert tokenizer_digest(first) != tokenizer_digest(renamed)
    with pytest.raises(ValueError):
        tokenizer_digest({**ARTIFACTS, "unknown-tokenizer-file.json": b"{}"})


def test_literal_domain_separated_artifact_digest():
    # Freeze the algorithm independently from the implementation/helper.
    import hashlib
    import json

    pairs = sorted((name, hashlib.sha256(value).hexdigest()) for name, value in ARTIFACTS.items())
    expected = (
        "sha256:"
        + hashlib.sha256(
            b"u2-tokenizer-artifacts-v1\0"
            + json.dumps(pairs, ensure_ascii=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    assert tokenizer_digest(ARTIFACTS) == expected
