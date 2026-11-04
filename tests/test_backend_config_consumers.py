"""Saved selector validation before serving/evaluation loads external assets.

The API module is imported normally. Evaluation and Gradio loader functions
are compiled unchanged from their source, so these CPU contracts do not need
metric packages or instantiate a UI. This does not test those full imports.
"""

import ast
import logging
import os
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
import yaml
from src.backends import resolve_backend
from src.config import ModelConfig
from src.decoders import remap_legacy_decoder_keys

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[1]


def _production_functions(relative_path, names, **namespace):
    """Execute the original function bodies with only their direct dependencies."""
    source_path = ROOT / relative_path
    tree = ast.parse(source_path.read_text(), filename=str(source_path))
    selected = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    assert {node.name for node in selected} == set(names)
    module = ModuleType("isolated_loader")
    module.__dict__.update(
        os=os,
        torch=torch,
        ModelConfig=ModelConfig,
        resolve_backend=resolve_backend,
        logger=logging.getLogger(__name__),
        **namespace,
    )
    exec(
        compile(ast.Module(body=selected, type_ignores=[]), str(source_path), "exec"),
        module.__dict__,
    )
    return module


@pytest.fixture(autouse=True)
def _offline_environment(monkeypatch):
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.delenv("PRISM_BACKBONE", raising=False)


def _saved_run(tmp_path, **fields):
    model = {"backbone_id": "fixture/backbone", "modalities": ["text"], "d_img": 8}
    model.update(fields)
    hydra_dir = tmp_path / ".hydra"
    hydra_dir.mkdir()
    config_path = hydra_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump({"model": model}))
    checkpoint_dir = tmp_path / "checkpoints" / "step_1"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "model.safetensors").touch()
    return checkpoint_dir, config_path


class _ConstructedModel:
    def __init__(self, config):
        self.config = config

    def load_state_dict(self, state, strict):
        return [], []

    def to(self, *args, **kwargs):
        return self

    def eval(self):
        return self


@pytest.fixture
def api(monkeypatch):
    from src.api import server

    monkeypatch.setattr(server, "model", None)
    monkeypatch.setattr(server, "tokenizer", None)
    monkeypatch.setattr(server, "device", "cpu")
    monkeypatch.setattr(server, "model_dtype", torch.float32)
    weights = Mock(return_value={"backbone.model.embed_tokens.weight": torch.zeros(32, 16)})
    tokenizer = Mock(return_value=SimpleNamespace(pad_token="<pad>", eos_token="<eos>"))
    constructor = Mock(side_effect=_ConstructedModel)
    monkeypatch.setattr(server, "load_file", weights)
    monkeypatch.setattr(server.AutoTokenizer, "from_pretrained", tokenizer)
    monkeypatch.setattr(server, "UnifiedTransformer", constructor)
    return SimpleNamespace(module=server, weights=weights, tokenizer=tokenizer, model=constructor)


@pytest.fixture
def evaluator(monkeypatch):
    universal_evaluator = _production_functions(
        "tools/universal_evaluator.py",
        {
            "_load_resolved_model_config",
            "_checkpoint_backbone_vocab_size",
            "_model_tokenizer_len",
            "_set_backbone_vocab_size",
            "_copy_vocab_rows_into_current_tensor",
            "_pad_mismatched_vocab_tensors",
            "_align_checkpoint_backbone_vocab",
            "setup_model",
        },
        remap_legacy_decoder_keys=remap_legacy_decoder_keys,
        load_file=None,
        load_cached_tokenizer=None,
        UnifiedTransformer=None,
        Accelerator=None,
    )

    weights = Mock(return_value={"backbone.model.embed_tokens.weight": torch.zeros(32, 16)})
    tokenizer = Mock(return_value=SimpleNamespace(pad_token="<pad>", eos_token="<eos>"))
    constructor = Mock(side_effect=_ConstructedModel)
    parser = Mock(wraps=universal_evaluator._load_resolved_model_config)
    monkeypatch.setattr(universal_evaluator, "load_file", weights)
    monkeypatch.setattr(universal_evaluator, "load_cached_tokenizer", tokenizer)
    monkeypatch.setattr(universal_evaluator, "UnifiedTransformer", constructor)
    monkeypatch.setattr(universal_evaluator, "_load_resolved_model_config", parser)
    monkeypatch.setattr(
        universal_evaluator, "Accelerator", lambda **kwargs: SimpleNamespace(device="cpu")
    )
    return SimpleNamespace(
        module=universal_evaluator,
        weights=weights,
        tokenizer=tokenizer,
        model=constructor,
        parser=parser,
    )


@pytest.fixture
def ui():
    weights = Mock(return_value={"backbone.model.embed_tokens.weight": torch.zeros(32, 16)})
    tokenizer = Mock(return_value=SimpleNamespace(pad_token="<pad>", eos_token="<eos>"))
    constructor = Mock(side_effect=_ConstructedModel)
    text_loader = Mock(return_value=_ConstructedModel(None))
    module = _production_functions(
        "src/ui/app.py",
        {"_read_hydra_config", "load_model", "load_text_model"},
        model=None,
        text_model=None,
        tokenizer=None,
        device="cpu",
        model_dtype=torch.float32,
        CHECKPOINT_DIR=None,
        load_file=weights,
        UnifiedTransformer=constructor,
        AutoTokenizer=SimpleNamespace(from_pretrained=tokenizer),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=text_loader),
    )
    return SimpleNamespace(
        module=module,
        weights=weights,
        tokenizer=tokenizer,
        model=constructor,
        text_loader=text_loader,
    )


@pytest.mark.parametrize(
    "fields, error",
    [
        ({"backend": "unknown"}, ValueError),
        ({"backend": False}, ValueError),
        ({"backend": ["hf_ar"]}, ValueError),
        ({"backend": "local_ar"}, ValueError),
        ({"backend": "hf_ar", "backbone_id": None}, ValueError),
        ({"backend_contract_version": True}, ValueError),
        ({"backend_contract_version": "1"}, ValueError),
        ({"backend_contract_version": 1.0}, ValueError),
        ({"backend_contract_version": None}, ValueError),
        ({"backend": "u2_reference"}, NotImplementedError),
        ({"backend": "u2_native"}, NotImplementedError),
    ],
)
@pytest.mark.parametrize("consumer", ["api", "evaluator", "ui"])
def test_saved_selector_rejects_before_assets_load(
    tmp_path, monkeypatch, request, consumer, fields, error
):
    target = request.getfixturevalue(consumer)
    checkpoint_dir, config_path = _saved_run(tmp_path, **fields)
    with pytest.raises(error):
        if consumer != "evaluator":
            monkeypatch.setattr(target.module, "CHECKPOINT_DIR", str(checkpoint_dir))
            target.module.load_model()
        else:
            target.module.setup_model(str(checkpoint_dir), model_config=str(config_path))
    target.weights.assert_not_called()
    target.tokenizer.assert_not_called()
    target.model.assert_not_called()


@pytest.mark.parametrize("consumer", ["api", "evaluator", "ui"])
def test_backbone_override_cannot_bypass_saved_local_selector(
    tmp_path, monkeypatch, request, consumer
):
    target = request.getfixturevalue(consumer)
    checkpoint_dir, config_path = _saved_run(tmp_path, backend="local_ar", backbone_id=None)
    with pytest.raises(ValueError, match="conflicts"):
        if consumer != "evaluator":
            monkeypatch.setattr(target.module, "CHECKPOINT_DIR", str(checkpoint_dir))
            monkeypatch.setenv("PRISM_BACKBONE", "fixture/override")
            target.module.load_model()
        else:
            target.module.setup_model(
                str(checkpoint_dir), model_config=str(config_path), backbone_id="fixture/override"
            )
    target.weights.assert_not_called()
    target.tokenizer.assert_not_called()
    target.model.assert_not_called()


@pytest.mark.parametrize("backend", [None, "hf_ar"])
@pytest.mark.parametrize("consumer", ["api", "evaluator", "ui"])
def test_saved_selector_reaches_model_config(tmp_path, monkeypatch, request, consumer, backend):
    target = request.getfixturevalue(consumer)
    fields = {} if backend is None else {"backend": backend, "backend_contract_version": 1}
    checkpoint_dir, config_path = _saved_run(tmp_path, **fields)
    if consumer != "evaluator":
        monkeypatch.setattr(target.module, "CHECKPOINT_DIR", str(checkpoint_dir))
        target.module.load_model()
    else:
        target.module.setup_model(str(checkpoint_dir), model_config=str(config_path))
        target.parser.assert_called_once_with(str(config_path))
    config = target.model.call_args.args[0]
    assert config.backend == backend
    assert config.backend_contract_version == 1
    assert config.llm_backbone_id == "fixture/backbone"
    assert config.d_model == 16
    target.weights.assert_called_once()
    target.tokenizer.assert_called_once()


@pytest.mark.parametrize("consumer", ["api", "ui"])
def test_serving_reads_raw_selector_values_from_saved_hydra_config(tmp_path, request, consumer):
    target = request.getfixturevalue(consumer)
    checkpoint_dir, _ = _saved_run(tmp_path, backend=False, backend_contract_version="1")
    fields = target.module._read_hydra_config(str(checkpoint_dir))
    assert fields["backend"] is False
    assert fields["backend_contract_version"] == "1"
    assert fields["backbone_id"] == "fixture/backbone"


@pytest.mark.parametrize("consumer", ["api", "ui"])
def test_serving_malformed_existing_config_is_not_silently_ignored(
    tmp_path, monkeypatch, request, consumer
):
    target = request.getfixturevalue(consumer)
    checkpoint_dir, config_path = _saved_run(tmp_path)
    config_path.write_text("model: [unterminated\n")
    monkeypatch.setattr(target.module, "CHECKPOINT_DIR", str(checkpoint_dir))
    with pytest.raises(ValueError):
        target.module.load_model()
    target.weights.assert_not_called()
    target.tokenizer.assert_not_called()
    target.model.assert_not_called()


@pytest.mark.parametrize("consumer", ["api", "ui"])
def test_serving_missing_saved_config_retains_empty_mapping(tmp_path, request, consumer):
    target = request.getfixturevalue(consumer)
    assert target.module._read_hydra_config(str(tmp_path)) == {}


def test_evaluator_discovers_saved_hydra_selector_before_loading_weights(tmp_path, evaluator):
    checkpoint_dir, config_path = _saved_run(tmp_path, backend="u2_reference")
    with pytest.raises(NotImplementedError, match="planned and unavailable"):
        evaluator.module.setup_model(str(checkpoint_dir))
    evaluator.parser.assert_called_once_with(str(config_path))
    evaluator.weights.assert_not_called()
    evaluator.tokenizer.assert_not_called()
    evaluator.model.assert_not_called()


def test_evaluator_retains_missing_backbone_error(tmp_path, evaluator):
    checkpoint_dir, config_path = _saved_run(tmp_path, backbone_id=None)
    with pytest.raises(ValueError, match="did not provide model.backbone_id"):
        evaluator.module.setup_model(str(checkpoint_dir), model_config=str(config_path))
    # With no selector, this remains the legacy evaluator-specific error;
    # dimension probing may precede it because local_ar metadata itself is valid.
    evaluator.tokenizer.assert_not_called()
    evaluator.model.assert_not_called()


@pytest.mark.parametrize(
    "fields, error",
    [
        ({"backend": "u2_reference"}, NotImplementedError),
        ({"backend": "u2_native"}, NotImplementedError),
        ({"backend": "unknown"}, ValueError),
        ({"backend_contract_version": "1"}, ValueError),
        ({"backend": "local_ar", "backbone_id": None}, ValueError),
    ],
)
def test_ui_text_comparison_rejects_unsupported_selector_before_loaders(
    tmp_path, ui, fields, error
):
    checkpoint_dir, _ = _saved_run(tmp_path, **fields)
    ui.module.CHECKPOINT_DIR = str(checkpoint_dir)
    with pytest.raises(error):
        ui.module.load_text_model()
    ui.tokenizer.assert_not_called()
    ui.text_loader.assert_not_called()


@pytest.mark.parametrize("backend", [None, "hf_ar"])
def test_ui_text_comparison_preserves_hf_loading(tmp_path, ui, backend):
    checkpoint_dir, _ = _saved_run(tmp_path, backend=backend)
    ui.module.CHECKPOINT_DIR = str(checkpoint_dir)
    ui.module.load_text_model()
    ui.tokenizer.assert_called_once_with("fixture/backbone", trust_remote_code=True)
    assert ui.text_loader.call_args.args == ("fixture/backbone",)
    assert ui.module.text_model is ui.text_loader.return_value
    ui.weights.assert_not_called()
    ui.model.assert_not_called()
