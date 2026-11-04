"""Offline HTTP contracts for an explicitly injected text-reference runtime."""

import ast
import json
import threading
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from fastapi.testclient import TestClient
from src.api import server
from src.api import u2_reference as reference_api
from src.backends import describe_backend, resolve_backend
from src.backends.text_contracts import RuntimeIdentity, TextRequest
from src.backends.u2_reference import U2ReferenceAdapter

pytestmark = pytest.mark.unit
ROUTE = "/v1/backends/u2_reference/generate"
IDENTITY = RuntimeIdentity(
    runtime_id="fixture-runtime",
    source_revision="fixture-revision",
    tokenizer_digest="fixture-tokenizer",
    vocab_size=100,
    eoa_token_id=99,
    mask_token_id=98,
)


class FakeRuntime:
    identity = IDENTITY

    def __init__(self, action=None):
        self.action = action
        self.requests = []
        self.thread_ids = []

    def generate(self, request, control):
        self.requests.append(request)
        self.thread_ids.append(threading.get_ident())
        if self.action is not None:
            return self.action(request, control)
        output = control.run_model_call(len(request.prompt_ids), lambda: (4, 5))
        control.commit(output)
        control.run_model_call(len(request.prompt_ids) + 2, lambda: None)
        control.commit((self.identity.eoa_token_id,))
        return "eoa"


def _payload(**overrides):
    return {
        "request_id": "request-1",
        "prompt_ids": [1, 2, 3],
        "tokenizer_digest": IDENTITY.tokenizer_digest,
        **overrides,
    }


def _app(adapter=None):
    return server.create_app(
        reference_adapter=adapter,
        load_legacy_model_on_startup=False,
    )


@pytest.fixture
def no_model_loads(monkeypatch):
    """Any accidental use of the legacy construction path fails the test."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.delenv("PRISM_CHECKPOINT", raising=False)
    monkeypatch.setattr(server, "model", None)
    monkeypatch.setattr(server, "tokenizer", None)
    forbidden = []
    for name in ("load_model", "load_file", "UnifiedTransformer", "ModelConfig", "resolve_backend"):
        spy = Mock(side_effect=AssertionError(f"unexpected legacy operation: {name}"))
        monkeypatch.setattr(server, name, spy)
        forbidden.append(spy)
    tokenizer_loader = Mock(side_effect=AssertionError("unexpected tokenizer load"))
    monkeypatch.setattr(server.AutoTokenizer, "from_pretrained", tokenizer_loader)
    forbidden.append(tokenizer_loader)
    yield
    for spy in forbidden:
        spy.assert_not_called()


def test_unconfigured_reference_route_is_explicit_and_offline(no_model_loads):
    with TestClient(_app()) as client:
        response = client.post(ROUTE, json=_payload())
    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "reference_adapter_unconfigured"}}


def test_exact_canonical_receipt_and_same_request_replay(no_model_loads):
    runtime = FakeRuntime()
    adapter = U2ReferenceAdapter(runtime)
    body = _payload(seed=27, steps_per_block=8)
    with TestClient(_app(adapter)) as client:
        event_loop_thread = client.portal.call(threading.get_ident)
        response = client.post(ROUTE, json=body)
        replay = client.post(ROUTE, json=body)
    assert response.status_code == replay.status_code == 200
    assert response.json() == replay.json()
    expected = adapter.generate(TextRequest(**body))
    assert response.json() == json.loads(json.dumps(asdict(expected)))
    result = response.json()
    assert result["output_ids"] == [4, 5]
    assert result["committed_block_ends"] == [2]
    assert result["finish_reason"] == "eoa"
    assert result["eoa_observed"] is True
    assert result["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "observed_model_calls": 2,
        "observed_processed_positions": 8,
        "model_calls": 2,
        "processed_positions": 8,
        "complete": True,
    }
    assert len(runtime.requests) == 1
    assert runtime.requests[0].prompt_ids == (1, 2, 3)
    assert runtime.thread_ids[0] != event_loop_thread


@pytest.mark.parametrize(
    "overrides",
    [
        {"request_id": 12},
        {"request_id": ""},
        {"request_id": "a" * 129},
        {"prompt_ids": []},
        {"prompt_ids": "1,2,3"},
        {"prompt_ids": {"0": 1}},
        {"prompt_ids": [True]},
        {"prompt_ids": ["1"]},
        {"prompt_ids": [1.0]},
        {"prompt_ids": [-1]},
        {"prompt_ids": [100]},
        {"prompt_ids": [1] * 4097},
        {"tokenizer_digest": 123},
        {"tokenizer_digest": ""},
        {"tokenizer_digest": "x" * 257},
        {"tokenizer_digest": "different-tokenizer"},
        {"contract_version": True},
        {"contract_version": "1"},
        {"contract_version": 2},
        {"seed": True},
        {"seed": "0"},
        {"seed": -1},
        {"seed": 2**64},
        {"max_new_tokens": True},
        {"max_model_calls": "2"},
        {"block_size": 2.0},
        {"steps_per_block": None},
        {"max_new_tokens": 0},
        {"max_new_tokens": 129},
        {"max_model_calls": 0},
        {"max_model_calls": 129},
        {"block_size": 0},
        {"block_size": 129},
        {"steps_per_block": 0},
        {"steps_per_block": 129},
        {"messages": [{"role": "user", "content": "no implicit rendering"}]},
        {"stream": True},
        {"backend": "hf_ar"},
    ],
)
def test_invalid_request_is_rejected_before_runtime(overrides, no_model_loads):
    runtime = FakeRuntime()
    with TestClient(_app(U2ReferenceAdapter(runtime))) as client:
        response = client.post(ROUTE, json=_payload(**overrides))
    assert response.status_code == 422
    assert runtime.requests == []


def test_missing_required_request_fields_are_rejected(no_model_loads):
    runtime = FakeRuntime()
    with TestClient(_app(U2ReferenceAdapter(runtime))) as client:
        response = client.post(ROUTE, json={"request_id": "empty"})
    assert response.status_code == 422
    assert runtime.requests == []


def test_runtime_failure_is_a_cached_receipt_with_unknown_totals(no_model_loads):
    def fail_after_commit(request, control):
        control.run_model_call(3, lambda: None)
        control.commit((4,))

        def fail():
            raise RuntimeError("private runtime failure detail must not be returned")

        control.run_model_call(9, fail)

    runtime = FakeRuntime(fail_after_commit)
    adapter = U2ReferenceAdapter(runtime)
    with TestClient(_app(adapter)) as client:
        response = client.post(ROUTE, json=_payload())
        replay = client.post(ROUTE, json=_payload())
    assert response.status_code == 200
    assert response.json() == replay.json()
    expected = adapter.generate(TextRequest(**_payload()))
    assert response.json() == json.loads(json.dumps(asdict(expected)))
    result = response.json()
    assert result["output_ids"] == [4]
    assert result["finish_reason"] == "error"
    assert result["error_code"]
    assert "private runtime failure detail" not in response.text
    assert result["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 1,
        "observed_model_calls": 2,
        "observed_processed_positions": 3,
        "model_calls": None,
        "processed_positions": None,
        "complete": False,
    }
    assert len(runtime.requests) == 1


@pytest.mark.parametrize("reason", ["length", "budget", "cancelled"])
def test_control_terminal_reasons_are_preserved_without_openai_mapping(reason, no_model_loads):
    def finish(request, control):
        output = (4, 5) if reason == "length" else (4,)
        control.commit(control.run_model_call(3, lambda: output))
        if reason == "cancelled":
            assert adapter.cancel(request.request_id)
        return reason

    runtime = FakeRuntime(finish)
    adapter = U2ReferenceAdapter(runtime)
    body = _payload(max_new_tokens=2, max_model_calls=1)
    with TestClient(_app(adapter)) as client:
        response = client.post(ROUTE, json=body)
    assert response.status_code == 200
    expected = adapter.generate(TextRequest(**body))
    assert response.json() == json.loads(json.dumps(asdict(expected)))
    result = response.json()
    assert result["finish_reason"] == reason
    assert result["eoa_observed"] is False
    assert result["usage"]["complete"] is True
    assert result["usage"]["model_calls"] == 1
    assert result["usage"]["processed_positions"] == 3
    assert len(runtime.requests) == 1


def test_changed_payload_conflict_and_full_ledger_are_distinct(no_model_loads):
    runtime = FakeRuntime()
    with TestClient(_app(U2ReferenceAdapter(runtime, max_entries=1))) as client:
        assert client.post(ROUTE, json=_payload()).status_code == 200
        conflict = client.post(ROUTE, json=_payload(seed=1))
        full = client.post(ROUTE, json=_payload(request_id="request-2"))
        replay = client.post(ROUTE, json=_payload())
    assert conflict.status_code == 409
    assert conflict.json() == {"detail": {"code": "request_conflict"}}
    assert full.status_code == 503
    assert full.json() == {"detail": {"code": "reference_ledger_capacity"}}
    assert replay.status_code == 200
    assert len(runtime.requests) == 1


def test_in_progress_request_conflict_is_distinct_and_never_reexecutes(no_model_loads):
    started = threading.Event()
    release = threading.Event()

    def blocked(request, control):
        def operation():
            started.set()
            assert release.wait(timeout=10), "test did not release the runtime"
            return (4, IDENTITY.eoa_token_id)

        control.commit(control.run_model_call(3, operation))
        return "eoa"

    runtime = FakeRuntime(blocked)
    with TestClient(_app(U2ReferenceAdapter(runtime))) as client:
        with ThreadPoolExecutor(max_workers=1) as pool:
            initial = pool.submit(client.post, ROUTE, json=_payload())
            try:
                assert started.wait(timeout=10), "runtime never started"
                in_progress = client.post(ROUTE, json=_payload())
                changed = client.post(ROUTE, json=_payload(seed=1))
            finally:
                release.set()
            first = initial.result(timeout=10)
        replay = client.post(ROUTE, json=_payload())
    assert in_progress.status_code == changed.status_code == 409
    assert in_progress.json() == {"detail": {"code": "request_in_progress"}}
    assert changed.json() == {"detail": {"code": "request_conflict"}}
    assert first.status_code == replay.status_code == 200
    assert first.json() == replay.json()
    assert len(runtime.requests) == 1


def test_reference_adapters_are_scoped_to_each_app(no_model_loads):
    first_runtime = FakeRuntime()
    second_runtime = FakeRuntime()
    with (
        TestClient(_app(U2ReferenceAdapter(first_runtime))) as first,
        TestClient(_app(U2ReferenceAdapter(second_runtime))) as second,
        TestClient(_app()) as unconfigured,
    ):
        assert first.post(ROUTE, json=_payload()).status_code == 200
        assert second.post(ROUTE, json=_payload(seed=2)).status_code == 200
        assert unconfigured.post(ROUTE, json=_payload()).status_code == 503
    assert len(first_runtime.requests) == len(second_runtime.requests) == 1
    assert first_runtime.requests[0].seed == 0
    assert second_runtime.requests[0].seed == 2


@pytest.mark.parametrize("use_module_app", [False, True])
def test_default_app_preserves_legacy_startup(monkeypatch, use_module_app):
    load = Mock()
    monkeypatch.setattr(server, "load_model", load)
    application = server.app if use_module_app else server.create_app()
    with TestClient(application) as client:
        assert load.call_count == 1
        assert client.get("/v1/models").status_code == 200
        response = client.post(ROUTE, json=_payload())
        assert response.status_code == 503
    load.assert_called_once_with()


def test_legacy_chat_and_models_remain_available(monkeypatch, no_model_loads):
    tokenizer = Mock()
    tokenizer.return_value = SimpleNamespace(input_ids=torch.tensor([[1, 2, 3]]))
    tokenizer.decode.return_value = "fixture answer"
    model = SimpleNamespace(generate=Mock(return_value=torch.tensor([[4, 5]])))
    monkeypatch.setattr(server, "model", model)
    monkeypatch.setattr(server, "tokenizer", tokenizer)
    monkeypatch.setattr(server, "device", "cpu")
    with TestClient(_app(U2ReferenceAdapter(FakeRuntime()))) as client:
        listing = client.get("/v1/models")
        response = client.post(
            "/v1/chat/completions",
            json={
                "messages": [{"role": "user", "content": "legacy prompt"}],
                "max_tokens": 4,
                "max_completion_tokens": 2,
            },
        )
    assert listing.status_code == response.status_code == 200
    assert listing.json()["data"] == [
        {"id": server.SERVED_MODEL_ID, "object": "model", "owned_by": "prism-team"}
    ]
    result = response.json()
    assert result["object"] == "chat.completion"
    assert result["choices"] == [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "fixture answer"},
            "finish_reason": "stop",
        }
    ]
    assert result["usage"] == {"prompt_tokens": 3, "completion_tokens": 2, "total_tokens": 5}
    assert model.generate.call_args.kwargs["max_new_tokens"] == 2
    tokenizer.assert_called_once_with("legacy prompt", return_tensors="pt")
    assert tokenizer.decode.call_args.args[0].tolist() == [4, 5]


def test_injected_route_does_not_activate_global_backend_resolution(no_model_loads):
    with TestClient(_app(U2ReferenceAdapter(FakeRuntime()))) as client:
        assert client.post(ROUTE, json=_payload()).status_code == 200
    for name in ("u2_reference", "u2_native"):
        assert describe_backend(name).status == "planned"
        with pytest.raises(NotImplementedError):
            resolve_backend(backend=name)


@pytest.fixture(scope="module", params=["pydantic", "pydantic.v1"])
def dto_engine(request):
    """Run the actual DTO class under each installed validation engine.

    This verifies Pydantic-v1 DTO construction and validation, not a full
    Pydantic-v1/FastAPI serving environment. Normal route tests above import
    the real API module with the environment's installed native Pydantic.
    """
    engine = pytest.importorskip(request.param)
    path = Path(reference_api.__file__)
    tree = ast.parse(path.read_text(), filename=str(path))
    selected = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "ReferenceTextRequest"
    ]
    assert len(selected) == 1
    namespace = {
        "__name__": __name__,
        "BaseModel": engine.BaseModel,
        "StrictInt": engine.StrictInt,
        "StrictStr": engine.StrictStr,
        "TextRequest": TextRequest,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        exec(
            compile(ast.Module(body=selected, type_ignores=[]), str(path), "exec"),
            namespace,
        )
    return namespace["ReferenceTextRequest"], engine


def test_dto_defaults_and_extra_rejection_on_both_pydantic_engines(dto_engine):
    dto_class, engine = dto_engine
    assert dto_class(**_payload()).to_contract() == TextRequest(**_payload())
    with pytest.raises(engine.ValidationError):
        dto_class(**_payload(stream=True))


@pytest.mark.parametrize(
    "overrides",
    [
        {"request_id": 123},
        {"prompt_ids": [True]},
        {"max_model_calls": "1"},
        {"contract_version": True},
    ],
)
def test_dto_strict_scalars_on_both_pydantic_engines(dto_engine, overrides):
    dto_class, engine = dto_engine
    with pytest.raises(engine.ValidationError):
        dto_class(**_payload(**overrides))


@pytest.mark.parametrize(
    "overrides",
    [
        {"request_id": "a" * 129},
        {"prompt_ids": []},
        {"prompt_ids": [1] * 4097},
        {"max_new_tokens": 129},
        {"seed": 2**64},
        {"contract_version": 2},
    ],
)
def test_dto_uses_canonical_limits_on_both_pydantic_engines(dto_engine, overrides):
    dto_class, _engine = dto_engine
    dto = dto_class(**_payload(**overrides))
    with pytest.raises(ValueError):
        dto.to_contract()
