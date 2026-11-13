"""Frozen acceptance inputs and independent synthetic checks of existing contracts.

No checkpoint, tokenizer assets, upstream runner or comparison driver is used.
These checks cannot award Aurora parity: the native acceptance document owns
the deferred evidence/coverage/numeric verdict until a reviewed runner exists.
"""

import hashlib
import json
from collections.abc import Callable, Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch
from src.backends.text_contracts import RuntimeIdentity, TextRequest, UsageReceipt
from src.backends.u2_preflight import U2PreflightReport
from src.backends.u2_reference import U2ReferenceAdapter
from src.backends.u2_text_renderer import EXPECTED_SPECIAL_TOKEN_IDS
from src.backends.u2_text_runtime import LocalModelSpec, U2TextRuntime

FIXTURE = Path(__file__).parent / "fixtures" / "u2_aurora_text_parity_v1.json"
FROZEN_SHA256 = "103aeddaf7175f74513f29ca748d3b331726468b5d35fdeff033079c9843c48f"
PROTOCOL = json.loads(FIXTURE.read_text(encoding="utf-8"))
SYNTHETIC = PROTOCOL["synthetic_cpu"]
DIGEST = "sha256:" + "a" * 64  # Deliberately synthetic declaration, not artifact binding.


@pytest.fixture(scope="module", autouse=True)
def cpu_threads() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _preflight() -> U2PreflightReport:
    """Declare synthetic metadata; this never qualifies real artifact execution."""
    return U2PreflightReport(
        source_revision=PROTOCOL["source_revision"],
        source_verified=True,
        files=(),
        required_artifacts=(),
        available_artifacts=(),
        missing=(),
        incompatible=(),
        tokenizer_digest=DIGEST,
        vocab_size=SYNTHETIC["vocab_size"],
        max_position_embeddings=256,
        dtype="float32",
        split_layout="split_candidate",
        package_versions=(),
        metadata_complete=True,
    )


class SyntheticModel:
    """Return hand-chosen logits and observe actual calls at the model boundary."""

    def __init__(self, case: dict[str, Any]) -> None:
        self.case = case
        self.training = True
        self.calls: list[dict[str, torch.Tensor]] = []
        self.on_call: Callable[[], object] | None = None
        self.spec = LocalModelSpec(
            identity=RuntimeIdentity(
                "synthetic-acceptance-cpu-v1",
                PROTOCOL["source_revision"],
                DIGEST,
                SYNTHETIC["vocab_size"],
                157166,
                156895,
            ),
            special_token_ids=EXPECTED_SPECIAL_TOKEN_IDS,
            dtype=torch.float32,
            device=torch.device("cpu"),
            max_position_embeddings=256,
        )

    def eval(self) -> "SyntheticModel":
        self.training = False
        return self

    def __call__(
        self,
        input_ids: torch.Tensor,
        type_position_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor,
        infer_mmu: bool,
        is_clean: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> torch.Tensor:
        assert infer_mmu and not self.training
        assert torch.is_inference_mode_enabled() and not torch.is_grad_enabled()
        self.calls.append(
            {
                "input": input_ids.clone(),
                "types": type_position_ids.clone(),
                "clean": is_clean.clone(),
                "positions": position_ids.clone(),
                "attention": attention_mask.clone(),
            }
        )
        if len(self.calls) == self.case.get("raise_on_call"):
            raise RuntimeError("synthetic forward failure")
        if len(self.calls) == self.case.get("cancel_on_call"):
            assert self.on_call is not None
            self.on_call()
        rows = int(((type_position_ids == 3) & (is_clean == 0)).sum())
        logits = torch.full((rows, self.spec.identity.vocab_size), -1000.0)
        logits[:, 100] = 0.0
        for row, winner in enumerate(self.case["winner_ids"][:rows]):
            logits[row, winner] = SYNTHETIC["winner_logits"][row]
        if self.case.get("nonfinite"):
            logits[0, 0] = float("nan")
        return logits


def _setup(
    case: dict[str, Any],
) -> tuple[SyntheticModel, U2ReferenceAdapter, TextRequest]:
    model = SyntheticModel(case)
    report = _preflight()
    assert report.execution_qualified is False
    adapter = U2ReferenceAdapter(U2TextRuntime(model, preflight=report))
    request = TextRequest(
        request_id=case["id"],
        prompt_ids=tuple(SYNTHETIC["prompt_ids"]),
        tokenizer_digest=DIGEST,
        seed=SYNTHETIC["seed"],
        **{
            key: case[key]
            for key in ("max_new_tokens", "block_size", "steps_per_block", "max_model_calls")
        },
    )
    model.on_call = lambda: adapter.cancel(request.request_id)
    return model, adapter, request


def test_frozen_bytes_are_unchanged() -> None:
    # Bind the pre-result freeze, not a hash recomputed from mutable expectations.
    assert hashlib.sha256(FIXTURE.read_bytes()).hexdigest() == FROZEN_SHA256


@pytest.mark.parametrize("case", PROTOCOL["cases"], ids=lambda case: case["id"])
def test_frozen_settings_fit_existing_request_bounds(case: dict[str, Any]) -> None:
    settings = dict(case["settings"])
    assert settings.pop("temperature") == 0
    assert settings.pop("cfg_scale") == 0
    assert settings.pop("remasking") == "low_confidence"
    request = TextRequest(
        request_id=case["id"], prompt_ids=(41, 42), tokenizer_digest=DIGEST, **settings
    )
    assert request.seed == 1729


@pytest.mark.parametrize("case", SYNTHETIC["cases"], ids=lambda case: case["id"])
def test_actual_loop_stop_commit_and_observed_receipts(case: dict[str, Any]) -> None:
    model, adapter, request = _setup(case)
    result = adapter.generate(request)
    expected = case["expected"]
    assert result.output_ids == tuple(expected["output_ids"])
    assert result.committed_block_ends == tuple(expected["committed_block_ends"])
    assert result.finish_reason == expected["finish_reason"]
    assert result.eoa_observed is (expected["finish_reason"] == "eoa")
    assert result.seed == request.seed
    assert [call["input"].shape[1] for call in model.calls] == expected["forward_sequence_lengths"]
    assert len(model.calls) == result.usage.observed_model_calls == expected["observed_model_calls"]
    assert result.usage.observed_processed_positions == expected["observed_processed_positions"]
    assert result.usage.complete is expected["usage_complete"]
    assert result.usage.prompt_tokens == 2
    assert result.usage.completion_tokens == len(result.output_ids)
    if expected["usage_complete"]:
        assert result.usage.model_calls == expected["observed_model_calls"]
        assert result.usage.processed_positions == expected["observed_processed_positions"]
    else:
        assert result.usage.model_calls is result.usage.processed_positions is None
        assert result.error_code is not None
    for call in model.calls:
        ids = call["input"]
        assert ids.shape[0] == 1 and ids.shape[2] == 8
        assert torch.equal(ids, ids[:, :, :1].expand_as(ids))
    observed_calls = len(model.calls)
    assert adapter.generate(request) is result
    assert len(model.calls) == observed_calls  # Replay must not perform another forward.


def test_first_forward_prefix_metadata_and_visibility() -> None:
    model, adapter, request = _setup(SYNTHETIC["cases"][0])
    adapter.generate(request)
    call = model.calls[0]
    assert call["input"][0, :, 0].tolist() == [41, 42, 157165, 156895, 156895, 156895]
    assert call["types"].tolist() == [[1, 1, 1, 3, 3, 3]]
    assert call["clean"].tolist() == [[1, 1, 0, 0, 0, 0]]
    assert call["positions"].tolist() == [[0, 1, 2, 3, 4, 5]]
    assert (call["attention"][0, 0] == 0).tolist() == [
        [True, True, False, False, False, False],
        [True, True, False, False, False, False],
        [True, True, True, False, False, False],
        [True, True, True, True, True, True],
        [True, True, True, True, True, True],
        [True, True, True, True, True, True],
    ]
    # Hand-specified reveals [3,4] survive into the next forward.
    assert model.calls[1]["input"][0, :, 0].tolist() == [41, 42, 157165, 101, 102, 156895]


def test_zero_quota_forwards_continue_after_all_tokens_are_revealed() -> None:
    model, adapter, request = _setup(SYNTHETIC["cases"][1])
    adapter.generate(request)
    assert len(model.calls) == 4
    assert model.calls[2]["input"][0, :, 0].tolist() == [41, 42, 157165, 101, 102]
    assert torch.equal(model.calls[2]["input"], model.calls[3]["input"])


def test_nonfinite_return_cannot_commit_or_receive_success() -> None:
    case = {**SYNTHETIC["cases"][0], "nonfinite": True}
    model, adapter, request = _setup(case)
    result = adapter.generate(request)
    assert result.finish_reason == "error" and result.output_ids == ()
    assert len(model.calls) == result.usage.observed_model_calls == 1
    assert result.usage.observed_processed_positions == 6  # Forward returned before validation.
    assert not result.usage.complete
    assert result.usage.model_calls is result.usage.processed_positions is None


def test_mismatched_tokenizer_rejected_before_any_forward() -> None:
    model, adapter, request = _setup(SYNTHETIC["cases"][0])
    with pytest.raises(ValueError, match="tokenizer_digest"):
        adapter.generate(replace(request, tokenizer_digest="sha256:" + "b" * 64))
    assert model.calls == []


@pytest.mark.parametrize("totals", [(3, None), (None, 12), (3, 12)])
def test_incomplete_receipt_cannot_invent_aggregate_totals(
    totals: tuple[int | None, int | None],
) -> None:
    with pytest.raises(ValueError, match="unknown"):
        UsageReceipt(2, 3, 3, 12, totals[0], totals[1], False)
