"""Frozen U2 text-runtime CPU oracle against the real loop through U2ReferenceAdapter.

All models and artifact snapshots here are explicitly synthetic. Full-size
vocabulary columns preserve canonical special IDs; only a few logits compete.
These tests establish deterministic contracts, not checkpoint/model parity.
"""

import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

import pytest
import torch
from src.backends import u2_text_runtime
from src.backends.text_contracts import RuntimeIdentity, TextRequest
from src.backends.u2_preflight import preflight_u2
from src.backends.u2_reference import (
    ExecutionControl,
    LedgerCapacityError,
    RequestConflictError,
    RequestInProgressError,
    U2ReferenceAdapter,
)
from src.backends.u2_text_renderer import render_text_prompt, tokenizer_digest
from src.backends.u2_text_runtime import LocalModelSpec, U2TextRuntime

from test_u2_preflight import bundle as _bundle_fixture
from test_u2_text_renderer import SPECIALS, Tokenizer

# Literal upstream pin and IDs, independent of runtime constants.
PIN = "70f21c1df8fa26e514df26ba31bc061be8d3d018"
M, BOA, EOA = 156895, 157165, 157166
bundle = _bundle_fixture


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def report(bundle):
    source, checkpoint, requirements = bundle
    config = json.loads((checkpoint / "config.json").read_text())
    config["torch_dtype"] = "float32"
    (checkpoint / "config.json").write_text(json.dumps(config))
    result = preflight_u2(source, checkpoint, replace(requirements, dtype="float32"))
    result.require_metadata()
    assert result.execution_qualified is False
    return result


def request(digest, **changes):
    fields = dict(
        request_id="actual-loop",
        prompt_ids=(41, 42),
        tokenizer_digest=digest,
        max_new_tokens=5,
        block_size=3,
        steps_per_block=2,
        max_model_calls=128,
    )
    fields.update(changes)
    return TextRequest(**fields)


def row_logits(n, vocab, ids=None, wins=None, dtype=torch.float32):
    logits = torch.full((n, vocab), -1000.0, dtype=dtype)
    logits[:, :4] = 0
    for i in range(n):
        logits[i, ids[i] if ids is not None else i % 2 + 1] = (
            wins[i] if wins is not None else n - i + 1
        )
    return logits


class TinyModel:
    """Explicit synthetic local model; captures every actual forward tensor."""

    def __init__(self, report, operation=None):
        self.spec = LocalModelSpec(
            identity=RuntimeIdentity(
                "synthetic-cpu-logits-v1", PIN, report.tokenizer_digest, report.vocab_size, EOA, M
            ),
            special_token_ids=SPECIALS,
            dtype=getattr(torch, report.dtype),
            device=torch.device("cpu"),
            max_position_embeddings=report.max_position_embeddings,
        )
        self.training = True
        self.calls = []
        self.operation = operation

    def eval(self):
        self.training = False
        return self

    def __call__(self, input_ids, type_position_ids, **kwargs):
        assert torch.is_inference_mode_enabled()
        assert not torch.is_grad_enabled() and self.training is False
        assert kwargs["infer_mmu"] is True
        self.calls.append(
            dict(
                input_ids=input_ids.clone(),
                types=type_position_ids.clone(),
                **{k: v.clone() if isinstance(v, torch.Tensor) else v for k, v in kwargs.items()},
            )
        )
        n = int(((type_position_ids == 3) & (kwargs["is_clean"] == 0)).sum())
        if self.operation is not None:
            return self.operation(self, n)
        return row_logits(n, self.spec.identity.vocab_size, dtype=self.spec.dtype)


def setup(report, operation=None, max_entries=128):
    model = TinyModel(report, operation)
    runtime = U2TextRuntime(model, preflight=report)
    adapter = U2ReferenceAdapter(runtime, max_entries=max_entries)
    return model, runtime, adapter


def assert_receipt(result, ids, ends, reason, calls, positions):
    assert result.output_ids == tuple(ids)
    assert result.committed_block_ends == tuple(ends)
    assert result.finish_reason == reason
    assert result.usage.prompt_tokens == 2
    assert result.usage.completion_tokens == len(ids)
    assert result.usage.observed_model_calls == calls
    assert result.usage.observed_processed_positions == positions
    if reason == "error":
        assert not result.usage.complete
        assert result.usage.model_calls is None and result.usage.processed_positions is None
    else:
        assert result.usage.complete
        assert result.usage.model_calls == calls and result.usage.processed_positions == positions


def assert_replay(adapter, model, req, result):
    count = len(model.calls)
    assert adapter.generate(req) is result
    assert len(model.calls) == count
    assert not adapter.cancel(req.request_id)


def assert_mask(call, rows):
    allowed = torch.tensor([[c == "1" for c in row] for row in rows])
    actual = call["attention_mask"]
    expected = torch.where(allowed, 0.0, torch.finfo(actual.dtype).min).to(actual.dtype)
    assert actual.shape == (1, 1, len(rows), len(rows))
    assert torch.equal(actual[0, 0], expected)


def test_literal_trajectory_metadata_counts_and_rng(report):
    model, _, adapter = setup(report)
    req = request(report.tokenizer_digest, seed=2**64 - 1)
    state = torch.random.get_rng_state().clone()
    result = adapter.generate(req)
    assert_receipt(result, [1, 2, 1, 1, 2], [3, 5], "length", 4, 28)
    assert not result.eoa_observed
    assert result.seed == 2**64 - 1
    assert torch.equal(state, torch.random.get_rng_state())
    expected = [
        [41, 42, BOA, M, M, M],
        [41, 42, BOA, 1, 2, M],
        [41, 42, BOA, 1, 2, 1, M, M],
        [41, 42, BOA, 1, 2, 1, 1, M],
    ]
    for i, (call, tokens) in enumerate(zip(model.calls, expected, strict=True)):
        length = len(tokens)
        assert call["input_ids"].shape == (1, length, 8)
        assert call["input_ids"].dtype == torch.long
        assert torch.equal(call["input_ids"][0], torch.tensor(tokens)[:, None].repeat(1, 8))
        assert call["types"].tolist() == [[1, 1, 1] + [3] * (length - 3)]
        assert call["is_clean"].tolist() == (
            [[1, 1, 0, 0, 0, 0]] if i < 2 else [[1, 1, 0, 1, 1, 1, 0, 0]]
        )
        for key in ("types", "is_clean", "position_ids"):
            assert call[key].shape == (1, length) and call[key].dtype == torch.long
        assert call["position_ids"].tolist() == [list(range(length))]
        assert call["attention_mask"].dtype == torch.float32
    for call in model.calls[:2]:
        assert_mask(call, ["110000", "110000", "111000", "111111", "111111", "111111"])
    assert_replay(adapter, model, req, result)


def test_literal_second_block_mask(report):
    model, _, adapter = setup(report)
    result = adapter.generate(request(report.tokenizer_digest, max_new_tokens=4, block_size=2))
    assert result.finish_reason == "length"
    for call in model.calls[2:]:
        assert call["is_clean"].tolist() == [[1, 1, 0, 1, 1, 0, 0]]
        assert_mask(
            call, ["1100000", "1100000", "1110000", "1101100", "1101100", "1111111", "1111111"]
        )


@pytest.mark.parametrize(
    "n,steps,quotas", [(5, 2, [3, 2]), (3, 2, [2, 1]), (2, 4, [1, 1, 0, 0]), (1, 3, [1, 0, 0])]
)
def test_literal_reveal_quotas_include_zero_forwards(report, n, steps, quotas):
    model, _, adapter = setup(report)
    result = adapter.generate(
        request(report.tokenizer_digest, max_new_tokens=n, block_size=n, steps_per_block=steps)
    )
    assert_receipt(result, [i % 2 + 1 for i in range(n)], [n], "length", steps, (n + 3) * steps)
    counts = [int((c["input_ids"][0, 3:, 0] != M).sum()) for c in model.calls] + [n]
    assert [b - a for a, b in zip(counts[:-1], counts[1:], strict=True)] == quotas


def test_zero_quota_forwards_validate_without_sampling(report, monkeypatch):
    model, runtime, adapter = setup(report)
    sampled = {name: [] for name in ("argmax", "to", "softmax", "gather", "where", "topk")}
    validated = []
    original_logits = runtime._logits

    def validate(value, n):
        validated.append(len(model.calls))
        return original_logits(value, n)

    def track(owner, name):
        original = getattr(owner, name)

        def wrapped(*args, **kwargs):
            sampled[name].append(len(model.calls))
            return original(*args, **kwargs)

        monkeypatch.setattr(owner, name, wrapped)

    monkeypatch.setattr(runtime, "_logits", validate)
    for name in ("argmax", "to", "gather"):
        track(torch.Tensor, name)
    for name in ("softmax", "where", "topk"):
        track(torch, name)
    req = request(report.tokenizer_digest, max_new_tokens=3, block_size=2, steps_per_block=4)
    result = adapter.generate(req)
    assert_receipt(result, [1, 2, 1], [2, 3], "length", 8, 44)
    assert validated == list(range(1, 9))
    for name, calls in sampled.items():
        assert calls == ([1, 1, 2, 2, 5, 5] if name == "where" else [1, 2, 5]), name
    assert_replay(adapter, model, req, result)


@pytest.mark.parametrize("bad", ["nan", "shape", "dtype"])
def test_invalid_logits_on_zero_quota_forward_still_fail(report, bad):
    def operation(model, n):
        logits = row_logits(n, model.spec.identity.vocab_size)
        # The first block is committed; call 6 is a zero-quota second-block step.
        if len(model.calls) == 6:
            if bad == "nan":
                logits[0, 0] = torch.nan
            elif bad == "shape":
                return logits[None]
            else:
                return logits.double()
        return logits

    model, _, adapter = setup(report, operation)
    req = request(report.tokenizer_digest, max_new_tokens=3, block_size=2, steps_per_block=4)
    result = adapter.generate(req)
    assert_receipt(result, [1, 2], [2], "error", 6, 32)
    assert result.error_code == "runtime_error"
    assert_replay(adapter, model, req, result)


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("cap", [2, 128])
def test_block_boundary_stops_before_next_allocation(report, monkeypatch, cancel, cap):
    model, _, adapter = setup(report)
    req = request(report.tokenizer_digest, max_model_calls=cap)
    original_mask = u2_text_runtime._text_attention_mask
    original_commit = ExecutionControl.commit
    allocations = []

    def mask(*args, **kwargs):
        allocations.append(len(model.calls))
        if len(allocations) > 1:
            raise MemoryError("next block allocation attempted")
        return original_mask(*args, **kwargs)

    def commit(control, ids):
        original_commit(control, ids)
        if cancel:
            assert adapter.cancel(req.request_id)

    monkeypatch.setattr(u2_text_runtime, "_text_attention_mask", mask)
    monkeypatch.setattr(ExecutionControl, "commit", commit)
    result = adapter.generate(req)
    reason = "cancelled" if cancel else ("budget" if cap == 2 else "error")
    assert_receipt(result, [1, 2, 1], [3], reason, 2, 12)
    assert allocations == ([0, 2] if reason == "error" else [0])
    assert result.error_code == ("runtime_error" if reason == "error" else None)
    assert_replay(adapter, model, req, result)


@pytest.mark.parametrize("shift", [0, 100])
def test_float64_confidence_row_shift_and_prior_token_stability(report, shift):
    def operation(model, n):
        logits = row_logits(n, model.spec.identity.vocab_size, [1, 2, 3, 1, 2], [2, 4, 1, 5, 3])
        logits[2] += shift  # raw winning logit is now largest; probability is unchanged.
        if len(model.calls) == 2:
            logits[1, 3] = 20
            logits[3, 2] = 20
            logits[4, 1] = 20
        return logits

    model, _, adapter = setup(report, operation)
    result = adapter.generate(request(report.tokenizer_digest, block_size=5))
    assert model.calls[1]["input_ids"][0, 3:, 0].tolist() == [M, 2, M, 1, 2]
    assert result.output_ids == (1, 2, 3, 1, 2)


def test_float64_rank_survives_float32_softmax_rounding(report):
    # Both winning probabilities round to 1 in float32, but differ in float64.
    # Put the weaker row at Torch's own tied topk choice to catch float32 ranking.
    tied = int(torch.topk(torch.ones(2), k=1).indices[0])
    strong = 1 - tied

    def operation(model, n):
        logits = row_logits(n, model.spec.identity.vocab_size, [1, 2], [20, 20])
        logits[strong, strong + 1] = 21
        return logits

    model, _, adapter = setup(report, operation)
    result = adapter.generate(request(report.tokenizer_digest, max_new_tokens=2, block_size=2))
    revealed = model.calls[1]["input_ids"][0, 3:, 0] != M
    assert torch.nonzero(revealed).flatten().tolist() == [strong]
    assert result.output_ids == (1, 2)


def test_ties_use_same_runtime_argmax_and_topk(report):
    def operation(model, n):
        return row_logits(n, model.spec.identity.vocab_size, [1] * n, [0] * n)

    model, _, adapter = setup(report, operation)
    result = adapter.generate(request(report.tokenizer_digest, block_size=5))
    logits = operation(model, 5)
    candidates = logits.argmax(-1)
    p = logits.to(torch.float64).softmax(-1).gather(1, candidates[:, None]).squeeze(1)
    selected = torch.topk(p, k=3).indices
    expected = torch.full((5,), M)
    expected[selected] = candidates[selected]
    assert torch.equal(model.calls[1]["input_ids"][0, 3:, 0], expected)
    assert result.output_ids == tuple(candidates.tolist())


@pytest.mark.parametrize(
    "cap,ids,ends,reason,positions",
    [
        (1, [], [], "budget", 6),
        (2, [1, 2, 1], [3], "budget", 12),
        (3, [1, 2, 1], [3], "budget", 20),
        (4, [1, 2, 1, 1, 2], [3, 5], "length", 28),
    ],
)
def test_call_caps_on_real_loop_and_exact_replay(report, cap, ids, ends, reason, positions):
    model, _, adapter = setup(report)
    req = request(report.tokenizer_digest, max_model_calls=cap)
    result = adapter.generate(req)
    assert_receipt(result, ids, ends, reason, cap, positions)
    assert not result.eoa_observed
    assert len(model.calls) == cap
    assert_replay(adapter, model, req, result)


@pytest.mark.parametrize(
    "tokens,expected",
    [([EOA, 7, 8], []), ([7, EOA, 8], [7]), ([7, 8, EOA], [7, 8]), ([7, EOA, EOA], [7])],
)
def test_eoa_only_after_completed_schedule_and_first_eoa_only(report, tokens, expected):
    def operation(model, n):
        return row_logits(
            n, model.spec.identity.vocab_size, tokens, [8 if t == EOA else 6 for t in tokens]
        )

    model, _, adapter = setup(report, operation)
    req = request(report.tokenizer_digest, max_new_tokens=3, max_model_calls=2)
    result = adapter.generate(req)
    assert_receipt(result, expected, [len(expected)] if expected else [], "eoa", 2, 12)
    assert result.eoa_observed
    assert EOA in model.calls[1]["input_ids"][0, 3:, 0].tolist()
    assert_replay(adapter, model, req, result)


def test_eoa_only_block_after_prior_commit_does_not_add_empty_end(report):
    def operation(model, n):
        return row_logits(
            n, model.spec.identity.vocab_size, [EOA, 7] if len(model.calls) > 2 else None
        )

    model, _, adapter = setup(report, operation)
    req = request(report.tokenizer_digest)
    result = adapter.generate(req)
    assert_receipt(result, [1, 2, 1], [3], "eoa", 4, 28)
    assert result.eoa_observed


def test_budget_discards_tentative_eoa(report):
    model, _, adapter = setup(
        report,
        lambda model, n: row_logits(n, model.spec.identity.vocab_size, [EOA, 7, 8], [9, 5, 1]),
    )
    result = adapter.generate(request(report.tokenizer_digest, max_model_calls=1))
    assert_receipt(result, [], [], "budget", 1, 6)
    assert not result.eoa_observed
    assert len(model.calls) == 1


def test_generated_specials_except_eoa_and_mask_are_preserved(report):
    ids = [157153, BOA, 157172]
    _, _, adapter = setup(
        report, lambda model, n: row_logits(n, model.spec.identity.vocab_size, ids)
    )
    result = adapter.generate(request(report.tokenizer_digest, max_new_tokens=3))
    assert result.output_ids == tuple(ids)


@pytest.mark.parametrize("call_number", [1, 3])
@pytest.mark.parametrize("raises", [False, True])
def test_event_cancellation_during_real_call_and_error_precedence(report, call_number, raises):
    entered, release = threading.Event(), threading.Event()

    def operation(model, n):
        if len(model.calls) == call_number:
            entered.set()
            assert release.wait(10)
            if raises:
                raise RuntimeError("private model failure must not escape")
        return row_logits(n, model.spec.identity.vocab_size)

    model, _, adapter = setup(report, operation)
    req = request(report.tokenizer_digest)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(adapter.generate, req)
        try:
            assert entered.wait(10)
            assert adapter.cancel(req.request_id)
            assert not adapter.cancel(req.request_id)
        finally:
            release.set()
        result = future.result(timeout=10)
    ids, ends = ([], []) if call_number == 1 else ([1, 2, 1], [3])
    positions = (0 if call_number == 1 else 12) if raises else (6 if call_number == 1 else 20)
    assert_receipt(result, ids, ends, "error" if raises else "cancelled", call_number, positions)
    if raises:
        assert result.error_code == "model_call_failed"
        assert "private" not in repr(result)
    assert_replay(adapter, model, req, result)


def test_raising_third_forward_without_cancellation(report):
    def operation(model, n):
        if len(model.calls) == 3:
            raise RuntimeError("private failure")
        return row_logits(n, model.spec.identity.vocab_size)

    model, _, adapter = setup(report, operation)
    req = request(report.tokenizer_digest)
    result = adapter.generate(req)
    assert_receipt(result, [1, 2, 1], [3], "error", 3, 12)
    assert result.error_code == "model_call_failed"
    assert_replay(adapter, model, req, result)


@pytest.mark.parametrize("boundary", ["before_call", "after_return", "after_commit"])
def test_event_cancellation_at_control_boundaries(report, monkeypatch, boundary):
    entered, release = threading.Event(), threading.Event()
    model, _, adapter = setup(
        report, lambda model, n: row_logits(n, model.spec.identity.vocab_size, [7, EOA, 8])
    )
    req = request(report.tokenizer_digest, max_new_tokens=3)
    original_call, original_commit = ExecutionControl.run_model_call, ExecutionControl.commit

    def wait():
        entered.set()
        assert release.wait(10)

    def call(control, positions, operation):
        if boundary == "before_call" and not model.calls:
            wait()
        result = original_call(control, positions, operation)
        if boundary == "after_return" and len(model.calls) == 1:
            wait()
        return result

    def commit(control, ids):
        original_commit(control, ids)
        if boundary == "after_commit":
            wait()

    monkeypatch.setattr(ExecutionControl, "run_model_call", call)
    monkeypatch.setattr(ExecutionControl, "commit", commit)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(adapter.generate, req)
        try:
            assert entered.wait(10)
            assert adapter.cancel(req.request_id)
        finally:
            release.set()
        result = future.result(timeout=10)
    calls = {"before_call": 0, "after_return": 1, "after_commit": 2}[boundary]
    committed = boundary == "after_commit"
    assert_receipt(
        result, [7] if committed else [], [1] if committed else [], "cancelled", calls, calls * 6
    )
    assert result.eoa_observed is committed
    assert_replay(adapter, model, req, result)


@pytest.mark.parametrize(
    "bad",
    ["shape", "rows", "integer", "wrong_float", "nan", "inf", "tuple", "object", "device", "mask"],
)
def test_invalid_returned_logits_and_unresolved_masks_fail_closed(report, bad):
    def operation(model, n):
        value = row_logits(n, model.spec.identity.vocab_size)
        if bad == "shape":
            return value[None]
        if bad == "rows":
            return value[:1]
        if bad == "integer":
            return value.long()
        if bad == "wrong_float":
            return value.double()
        if bad in ("nan", "inf"):
            value[0, 0] = float(bad)
        if bad == "tuple":
            return (value,)
        if bad == "object":
            return {"logits": value}
        if bad == "device":
            return value.to("meta")
        if bad == "mask":
            value[:, M] = 100
        return value

    model, _, adapter = setup(report, operation)
    req = request(report.tokenizer_digest)
    result = adapter.generate(req)
    count = 2 if bad == "mask" else 1
    assert_receipt(result, [], [], "error", count, count * 6)
    assert result.error_code == "runtime_error"
    assert not result.eoa_observed
    assert_replay(adapter, model, req, result)


def test_no_forward_context_failure_and_replay(report):
    limited = replace(report, max_position_embeddings=7)
    model, _, adapter = setup(limited)
    req = request(report.tokenizer_digest)
    result = adapter.generate(req)
    assert_receipt(result, [], [], "error", 0, 0)
    assert result.error_code == "runtime_error"
    assert_replay(adapter, model, req, result)


@pytest.mark.parametrize(
    "field,value",
    [
        ("num_codebooks", 4),
        ("num_codebooks", True),
        ("layout", "packed"),
        ("dtype", torch.int64),
        ("dtype", torch.float16),
        ("device", torch.device("meta")),
        ("device", torch.device("cuda")),
        ("max_position_embeddings", 7),
        ("max_position_embeddings", True),
    ],
)
def test_incompatible_model_binding_fails_before_forward(report, field, value):
    model = TinyModel(report)
    model.spec = replace(model.spec, **{field: value})
    with pytest.raises(ValueError, match="incompatible"):
        U2TextRuntime(model, preflight=report)
    assert not model.calls


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_revision", "wrong"),
        ("tokenizer_digest", "0" * 64),
        ("tokenizer_digest", "sha256:" + "0" * 64),
        ("vocab_size", 160001),
        ("eoa_token_id", 157164),
        ("mask_token_id", 157147),
    ],
)
def test_incompatible_identity_fails_before_forward(report, field, value):
    model = TinyModel(report)
    model.spec = replace(model.spec, identity=replace(model.spec.identity, **{field: value}))
    with pytest.raises(ValueError, match="identity"):
        U2TextRuntime(model, preflight=report)
    assert not model.calls


@pytest.mark.parametrize("token", SPECIALS)
def test_full_special_mapping_required_in_local_binding(report, token):
    model = TinyModel(report)
    model.spec = replace(model.spec, special_token_ids={**SPECIALS, token: 0})
    with pytest.raises(ValueError, match="identity"):
        U2TextRuntime(model, preflight=report)


@pytest.mark.parametrize(
    "field,value",
    [
        ("metadata_complete", False),
        ("source_verified", False),
        ("source_revision", "wrong"),
        ("split_layout", "packed"),
        ("dtype", "float64"),
    ],
)
def test_incomplete_or_incompatible_preflight_is_not_execution_admission(report, field, value):
    model = TinyModel(report)
    with pytest.raises(ValueError):
        U2TextRuntime(model, preflight=replace(report, **{field: value}))
    assert not model.calls


def test_binding_mutation_and_training_mode_fail_before_forward(report):
    for mode in ("spec", "training"):
        model, _, adapter = setup(report)
        if mode == "spec":
            model.spec = replace(model.spec, max_position_embeddings=1)
        else:
            model.training = True
        result = adapter.generate(request(report.tokenizer_digest))
        assert_receipt(result, [], [], "error", 0, 0)
        assert not model.calls


@pytest.mark.parametrize("digest", ["0" * 64, "sha256:" + "0" * 64])
def test_request_digest_mismatch_rejected_before_forward(report, digest):
    model, _, adapter = setup(report)
    with pytest.raises(ValueError, match="tokenizer_digest"):
        adapter.generate(request(digest))
    assert not model.calls


def test_out_of_vocabulary_and_preinserted_boa_rejected(report):
    model, _, adapter = setup(report)
    with pytest.raises(ValueError, match="out-of-vocabulary"):
        adapter.generate(request(report.tokenizer_digest, prompt_ids=(160000,)))
    for token in (BOA, EOA, M):
        result = adapter.generate(
            request(report.tokenizer_digest, request_id=str(token), prompt_ids=(41, token))
        )
        assert_receipt(result, [], [], "error", 0, 0)
    assert not model.calls


@pytest.mark.parametrize(
    "field", ["max_new_tokens", "max_model_calls", "block_size", "steps_per_block"]
)
@pytest.mark.parametrize("invalid", [True, "1", 0, 129])
def test_request_cap_types_and_outside_bounds_fail_before_forward(report, field, invalid):
    model, _, adapter = setup(report)
    with pytest.raises(ValueError):
        adapter.generate(request(report.tokenizer_digest, **{field: invalid}))
    assert not model.calls


@pytest.mark.parametrize(
    "changes",
    [
        {"prompt_ids": ()},
        {"prompt_ids": (True,)},
        {"prompt_ids": (1,) * 4097},
        {"contract_version": 2},
        {"contract_version": True},
        {"seed": -1},
        {"seed": 2**64},
        {"seed": True},
    ],
)
def test_request_admission_errors_before_forward(report, changes):
    model, _, adapter = setup(report)
    with pytest.raises(ValueError):
        adapter.generate(request(report.tokenizer_digest, **changes))
    assert not model.calls


@pytest.mark.parametrize(
    "field", ["max_new_tokens", "max_model_calls", "block_size", "steps_per_block"]
)
@pytest.mark.parametrize("endpoint", [1, 128])
def test_request_cap_endpoints_through_actual_loop(report, field, endpoint):
    model, _, adapter = setup(report)
    settings = dict(max_new_tokens=1, max_model_calls=128, block_size=1, steps_per_block=1)
    settings[field] = endpoint
    result = adapter.generate(request(report.tokenizer_digest, **settings))
    assert result.finish_reason == "length"
    assert result.usage.model_calls == len(model.calls)
    assert result.usage.processed_positions == sum(c["input_ids"].shape[1] for c in model.calls)
    assert len(result.output_ids) == settings["max_new_tokens"]


@pytest.mark.parametrize(
    "changes",
    [
        {"prompt_ids": (41,)},
        {"tokenizer_digest": "other"},
        {"max_new_tokens": 4},
        {"max_model_calls": 127},
        {"block_size": 2},
        {"steps_per_block": 1},
        {"seed": 1},
    ],
)
def test_changed_request_replay_conflicts(report, changes):
    model, _, adapter = setup(report)
    req = request(report.tokenizer_digest)
    result = adapter.generate(req)
    with pytest.raises(RequestConflictError):
        adapter.generate(replace(req, **changes))
    assert_replay(adapter, model, req, result)


def test_live_duplicate_and_capacity_without_eviction(report):
    entered, release = threading.Event(), threading.Event()

    def operation(model, n):
        entered.set()
        assert release.wait(10)
        return row_logits(n, model.spec.identity.vocab_size)

    model, _, adapter = setup(report, operation, max_entries=1)
    req = request(report.tokenizer_digest)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(adapter.generate, req)
        try:
            assert entered.wait(10)
            with pytest.raises(RequestInProgressError):
                adapter.generate(req)
            with pytest.raises(RequestConflictError):
                adapter.generate(replace(req, seed=1))
            with pytest.raises(LedgerCapacityError):
                adapter.generate(replace(req, request_id="second"))
        finally:
            release.set()
        result = future.result(timeout=10)
    with pytest.raises(LedgerCapacityError):
        adapter.generate(replace(req, request_id="second"))
    assert_replay(adapter, model, req, result)


def test_renderer_preflight_runtime_identity_same_snapshot(bundle, report):
    _, checkpoint, _ = bundle
    artifacts = {
        name: (checkpoint / name).read_bytes()
        for name in ("tokenizer.json", "tokenizer_config.json")
    }
    rendered = render_text_prompt(
        Tokenizer(), system_prompt="S", user_text="U", tokenizer_artifacts=artifacts
    )
    # Independent canonical digest, not just comparing two calls to one helper.
    pairs = sorted((name, hashlib.sha256(data).hexdigest()) for name, data in artifacts.items())
    expected = (
        "sha256:"
        + hashlib.sha256(
            b"u2-tokenizer-artifacts-v1\0"
            + json.dumps(pairs, ensure_ascii=True, separators=(",", ":")).encode()
        ).hexdigest()
    )
    model, runtime, adapter = setup(report)
    assert (
        rendered.tokenizer_digest
        == report.tokenizer_digest
        == runtime.identity.tokenizer_digest
        == expected
    )
    assert expected == tokenizer_digest(dict(reversed(list(artifacts.items()))))
    req = request(
        expected, prompt_ids=rendered.prompt_ids, max_new_tokens=1, block_size=1, steps_per_block=1
    )
    result = adapter.generate(req)
    assert result.finish_reason == "length" and result.output_ids == (1,)
    assert model.calls[0]["input_ids"][0, :, 0].tolist() == [*rendered.prompt_ids, BOA, M]
    assert result.usage.prompt_tokens == len(rendered.prompt_ids)
    assert result.usage.processed_positions == len(rendered.prompt_ids) + 2
    assert report.execution_qualified is False


@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_model_dtype_controls_masks_and_logits(report, dtype):
    typed = replace(report, dtype=dtype)
    model, _, adapter = setup(typed)
    result = adapter.generate(request(typed.tokenizer_digest))
    assert_receipt(result, [1, 2, 1, 1, 2], [3, 5], "length", 4, 28)
    assert model.calls[0]["attention_mask"].dtype == getattr(torch, dtype)
    assert_mask(model.calls[0], ["110000", "110000", "111000", "111111", "111111", "111111"])


def test_invalid_third_return_retains_prior_block_and_completed_positions(report):
    def operation(model, n):
        value = row_logits(n, model.spec.identity.vocab_size)
        return value[:1] if len(model.calls) == 3 else value

    model, _, adapter = setup(report, operation)
    req = request(report.tokenizer_digest)
    result = adapter.generate(req)
    assert_receipt(result, [1, 2, 1], [3], "error", 3, 20)
    assert_replay(adapter, model, req, result)


def test_cancellation_discards_already_revealed_tentative_eoa(report):
    entered, release = threading.Event(), threading.Event()

    def operation(model, n):
        if len(model.calls) == 2:
            assert model.calls[-1]["input_ids"][0, 3, 0] == EOA
            entered.set()
            assert release.wait(10)
        return row_logits(n, model.spec.identity.vocab_size, [EOA, 7, 8], [9, 5, 1])

    model, _, adapter = setup(report, operation)
    req = request(report.tokenizer_digest)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(adapter.generate, req)
        try:
            assert entered.wait(10)
            assert adapter.cancel(req.request_id)
        finally:
            release.set()
        result = future.result(timeout=10)
    assert_receipt(result, [], [], "cancelled", 2, 12)
    assert not result.eoa_observed
    assert_replay(adapter, model, req, result)


@pytest.mark.parametrize("terminal", ["length", "budget", "eoa"])
def test_cancellation_after_real_loop_before_finalization(report, monkeypatch, terminal):
    entered, release = threading.Event(), threading.Event()
    operation = (
        (lambda model, n: row_logits(n, model.spec.identity.vocab_size, [7, EOA, 8]))
        if terminal == "eoa"
        else None
    )
    model, runtime, adapter = setup(report, operation)
    req = request(report.tokenizer_digest, max_model_calls=1 if terminal == "budget" else 128)
    generate = runtime.generate

    def paused_finish(value, control):
        try:
            return generate(value, control)
        finally:
            entered.set()
            assert release.wait(10)

    monkeypatch.setattr(runtime, "generate", paused_finish)
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(adapter.generate, req)
        try:
            assert entered.wait(10)
            assert adapter.cancel(req.request_id)
        finally:
            release.set()
        result = future.result(timeout=10)
    ids, ends, calls, positions = {
        "length": ([1, 2, 1, 1, 2], [3, 5], 4, 28),
        "budget": ([], [], 1, 6),
        "eoa": ([7], [1], 2, 12),
    }[terminal]
    assert_receipt(result, ids, ends, "cancelled", calls, positions)
    assert result.eoa_observed is (terminal == "eoa")
    assert_replay(adapter, model, req, result)


@pytest.mark.parametrize("device", ["xpu:0", "xpu:7", "cuda:0", "cpu"])
def test_indexed_device_admission_does_not_probe_availability(report, monkeypatch, device):
    model = TinyModel(report)
    model.spec = replace(model.spec, device=torch.device(device))
    monkeypatch.setattr(
        torch.cuda, "is_available", lambda: pytest.fail("metadata must not probe CUDA")
    )
    monkeypatch.setattr(
        torch.xpu, "is_available", lambda: pytest.fail("metadata must not probe XPU")
    )
    runtime = U2TextRuntime(model, preflight=report)
    assert runtime._spec.device == torch.device(device)
    assert not model.calls


@pytest.mark.parametrize("device", ["xpu", "cuda", "cpu:0", "meta"])
def test_unindexed_accelerators_and_cpu_index_remain_rejected(report, device):
    model = TinyModel(report)
    model.spec = replace(model.spec, device=torch.device(device))
    with pytest.raises(ValueError, match="device"):
        U2TextRuntime(model, preflight=report)
