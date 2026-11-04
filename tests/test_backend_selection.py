"""Offline configuration, construction and compatibility checks for backend selection."""

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
import transformers
from hydra import compose, initialize_config_dir
from hydra.errors import ConfigCompositionException
from omegaconf import OmegaConf
from src.backends import resolve_backend
from src.config import ModelConfig
from src.model import UnifiedTransformer

pytestmark = pytest.mark.unit
CONF = Path(__file__).resolve().parents[1] / "src" / "conf"


class _TinyTextEncoder(nn.Module):
    def __init__(self, d_text):
        super().__init__()
        self.embedding = nn.Embedding(32, d_text)

    def forward(self, input_ids):
        return self.embedding(input_ids)


class _TinyTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def __len__(self):
        return 32


class _TinyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=16)
        self.embed = nn.Embedding(32, 16)
        self.head = nn.Linear(16, 32, bias=False)

    @property
    def dtype(self):
        return self.embed.weight.dtype

    def get_input_embeddings(self):
        return self.embed

    def forward(self, inputs_embeds, **kwargs):
        return SimpleNamespace(logits=self.head(inputs_embeds), hidden_states=(inputs_embeds,))


@pytest.fixture(autouse=True)
def _offline_environment(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


@pytest.fixture
def loaders(monkeypatch):
    """Patch the actual importing sites, including the tokenizer cache wrapper."""
    import src.model as model_module

    backbone = Mock(side_effect=lambda *args, **kwargs: _TinyBackbone())
    tokenizer = Mock(side_effect=lambda *args, **kwargs: _TinyTokenizer())
    encoder = Mock(side_effect=_TinyTextEncoder)
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", backbone)
    monkeypatch.setattr(model_module, "load_cached_tokenizer", tokenizer)
    monkeypatch.setattr(model_module, "TextEncoder", encoder)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: False)
    return SimpleNamespace(backbone=backbone, tokenizer=tokenizer, encoder=encoder)


def _small_config(**options):
    fields = {
        "vocab_size": 32,
        "d_model": 16,
        "d_text": 16,
        "num_layers": 1,
        "num_heads": 2,
        "num_experts": 2,
        "num_experts_per_token": 1,
        "dropout": 0.0,
        "max_seq_len": 16,
        "modalities": ["text"],
    }
    fields.update(options)
    return ModelConfig(**fields)


@pytest.mark.parametrize(
    "fields, expected",
    [
        ({}, "local_ar"),
        ({"llm_backbone_id": ""}, "local_ar"),
        ({"backend": "local_ar"}, "local_ar"),
        ({"hf_model_id": "fixture/initialization-only"}, "local_ar"),
        ({"llm_backbone_id": "fixture/backbone"}, "hf_ar"),
        ({"backend": "hf_ar", "llm_backbone_id": "fixture/backbone"}, "hf_ar"),
    ],
)
def test_model_config_preserves_legacy_selection(fields, expected):
    config = ModelConfig(**fields)
    selected = resolve_backend(
        backend=config.backend,
        llm_backbone_id=config.llm_backbone_id,
        contract_version=config.backend_contract_version,
    )
    assert selected.name == expected
    assert config.backend == fields.get("backend")
    assert config.backend_contract_version == 1


@pytest.mark.parametrize(
    "fields, error, message",
    [
        ({"backend": "unknown"}, ValueError, "Unknown backend"),
        ({"backend": ""}, ValueError, "nonempty string"),
        ({"backend": False}, ValueError, "nonempty string"),
        ({"backend": ["local_ar"]}, ValueError, "nonempty string"),
        ({"backend": "hf_ar"}, ValueError, "requires"),
        ({"backend": "local_ar", "llm_backbone_id": "fixture/model"}, ValueError, "conflicts"),
        ({"llm_backbone_id": False}, ValueError, "llm_backbone_id"),
        ({"backend_contract_version": True}, ValueError, "contract_version"),
        ({"backend_contract_version": "1"}, ValueError, "contract_version"),
        ({"backend_contract_version": 1.0}, ValueError, "contract_version"),
        ({"backend_contract_version": 2}, ValueError, "contract_version"),
        ({"backend_contract_version": None}, ValueError, "contract_version"),
        ({"backend": "u2_reference"}, NotImplementedError, "planned and unavailable"),
        ({"backend": "u2_native"}, NotImplementedError, "planned and unavailable"),
    ],
)
def test_invalid_config_and_runtime_mutations_fail_before_construction(
    monkeypatch, loaders, fields, error, message
):
    import src.model as model_module

    blocks = Mock(side_effect=AssertionError("must reject before constructing local blocks"))
    monkeypatch.setattr(model_module, "TransformerBlock", blocks)
    with pytest.raises(error, match=message):
        _small_config(**fields)

    # Config objects are mutable; construction must also validate current values.
    config = _small_config()
    for name, value in fields.items():
        setattr(config, name, value)
    with pytest.raises(error, match=message):
        UnifiedTransformer(config)

    blocks.assert_not_called()
    loaders.backbone.assert_not_called()
    loaders.tokenizer.assert_not_called()
    loaders.encoder.assert_not_called()


@pytest.mark.parametrize("backend", ["local_ar", "hf_ar"])
def test_explicit_selection_preserves_parameters_outputs_and_checkpoint_keys(loaders, backend):
    options = {"llm_backbone_id": "fixture/backbone"} if backend == "hf_ar" else {}
    torch.manual_seed(821)
    implicit = UnifiedTransformer(_small_config(**options)).eval()
    torch.manual_seed(821)
    explicit = UnifiedTransformer(_small_config(backend=backend, **options)).eval()

    implicit_state = implicit.state_dict()
    explicit_state = explicit.state_dict()
    assert list(explicit_state) == list(implicit_state)
    for name in implicit_state:
        torch.testing.assert_close(implicit_state[name], explicit_state[name], rtol=0, atol=0)
    implicit.load_state_dict(explicit_state, strict=True)
    explicit.load_state_dict(implicit_state, strict=True)

    tokens = torch.tensor([[2, 3, 4, 5], [6, 7, 8, 9]])
    with torch.no_grad():
        implicit_logits, implicit_loss = implicit({"text": tokens}, labels=tokens)
        explicit_logits, explicit_loss = explicit({"text": tokens}, labels=tokens)
    assert explicit_logits.shape == (2, 4, 32)
    assert torch.isfinite(explicit_logits).all()
    assert torch.isfinite(explicit_loss)
    torch.testing.assert_close(implicit_logits, explicit_logits, rtol=0, atol=0)
    torch.testing.assert_close(implicit_loss, explicit_loss, rtol=0, atol=0)

    if backend == "local_ar":
        assert implicit.backbone is explicit.backbone is None
        assert "blocks.0.ln1.weight" in explicit_state
        assert "head.decoder.weight" in explicit_state
        assert not any(name.startswith("backbone.") for name in explicit_state)
        loaders.backbone.assert_not_called()
        loaders.tokenizer.assert_not_called()
        assert loaders.encoder.call_count == 2
    else:
        assert isinstance(explicit.backbone, _TinyBackbone)
        assert explicit.backbone_dim == 16
        assert not hasattr(explicit, "blocks")
        assert set(explicit_state) == {"backbone.embed.weight", "backbone.head.weight"}
        assert all(not parameter.requires_grad for parameter in explicit.backbone.parameters())
        loaders.encoder.assert_not_called()
        assert loaders.backbone.call_count == loaders.tokenizer.call_count == 2
        for call in loaders.backbone.call_args_list:
            assert call.args == ("fixture/backbone",)
            assert call.kwargs["local_files_only"] is True
            assert call.kwargs["attn_implementation"] == "sdpa"


def test_hf_selector_preserves_tokenizer_override_and_unfrozen_backbone(loaders):
    config = _small_config(
        backend="hf_ar",
        llm_backbone_id="fixture/backbone",
        llm_tokenizer_id="fixture/tokenizer",
        freeze_backbone=False,
        attn_implementation="eager",
    )
    model = UnifiedTransformer(config)
    loaders.tokenizer.assert_called_once_with("fixture/tokenizer", trust_remote_code=True)
    assert loaders.backbone.call_args.kwargs["attn_implementation"] == "eager"
    assert all(parameter.requires_grad for parameter in model.backbone.parameters())


def test_explicit_hf_load_failure_does_not_fall_back(loaders):
    loaders.backbone.side_effect = OSError("fixture cache missing")
    with pytest.raises(RuntimeError, match="Could not load HF backbone"):
        UnifiedTransformer(_small_config(backend="hf_ar", llm_backbone_id="fixture/model"))
    loaders.tokenizer.assert_not_called()
    loaders.encoder.assert_not_called()


@pytest.mark.parametrize(
    "preset",
    [
        pytest.param(
            path.stem,
            marks=pytest.mark.xfail(
                strict=True,
                raises=ConfigCompositionException,
                reason="Existing AuroraGPT preset has unquoted <project> in an oc.env default",
            ),
        )
        if path.stem == "prism_auroragpt_2b"
        else path.stem
        for path in sorted((CONF / "model").glob("*.yaml"))
    ],
)
def test_existing_hydra_presets_accept_shared_backend_defaults_and_overrides(preset):
    try:
        with initialize_config_dir(version_base=None, config_dir=str(CONF)):
            legacy = compose(config_name="config", overrides=[f"model={preset}"])
            explicit = compose(
                config_name="config", overrides=[f"model={preset}", "model.backend=hf_ar"]
            )
    except ConfigCompositionException as error:
        # The xfail applies only to this pre-existing parser failure, not any
        # future composition regression in the new selector defaults.
        assert preset == "prism_auroragpt_2b"
        assert "token recognition error at: '<'" in str(error.__cause__)
        raise
    assert legacy.model.backend is None
    assert legacy.model.backend_contract_version == 1
    assert explicit.model.backend == "hf_ar"
    before = OmegaConf.to_container(legacy.model, resolve=False)
    after = OmegaConf.to_container(explicit.model, resolve=False)
    assert after == {**before, "backend": "hf_ar"}
    assert legacy.model.backbone_id


def _training_tree(**model_fields):
    fields = {
        "backbone_id": None,
        "freeze_backbone": True,
        "freeze_encoders": True,
        "modalities": ["text"],
    }
    fields.update(model_fields)
    return OmegaConf.create({"model": fields, "training": {}})


@pytest.mark.parametrize(
    "fields, backend, backbone",
    [
        ({}, None, None),
        ({"backend": None}, None, None),
        ({"backend": "local_ar"}, "local_ar", None),
        ({"backbone_id": "fixture/model"}, None, "fixture/model"),
        ({"backend": "hf_ar", "backbone_id": "fixture/model"}, "hf_ar", "fixture/model"),
    ],
)
def test_training_model_config_maps_selector_and_legacy_fallback(fields, backend, backbone):
    from src.train import build_model_config

    config = build_model_config(_training_tree(**fields))
    assert config.backend == backend
    assert config.backend_contract_version == 1
    assert config.llm_backbone_id == backbone
    assert config.llm_tokenizer_id == backbone


def test_hydra_backend_override_reaches_training_model_config():
    from src.train import build_model_config

    with initialize_config_dir(version_base=None, config_dir=str(CONF)):
        cfg = compose(
            config_name="config",
            overrides=["model=prism_image_only", "model.backend=hf_ar"],
        )
    model_config = build_model_config(cfg)
    assert model_config.backend == "hf_ar"
    assert model_config.backend_contract_version == 1
    assert model_config.llm_backbone_id == cfg.model.backbone_id


@pytest.mark.parametrize(
    "fields, error",
    [
        ({"backend": False}, ValueError),
        ({"backend": ["local_ar"]}, ValueError),
        ({"backend": "hf_ar"}, ValueError),
        ({"backend_contract_version": True}, ValueError),
        ({"backend_contract_version": "1"}, ValueError),
        ({"backend_contract_version": 1.0}, ValueError),
        ({"backend_contract_version": None}, ValueError),
        ({"backend": "u2_reference"}, NotImplementedError),
        ({"backend": "u2_native"}, NotImplementedError),
    ],
)
def test_training_mapping_does_not_coerce_invalid_selectors(fields, error):
    from src.train import build_model_config

    with pytest.raises(error):
        build_model_config(_training_tree(**fields))
