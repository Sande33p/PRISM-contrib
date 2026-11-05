#!/usr/bin/env python3
"""Private measured comparison of pinned generate_text_bd and PRISM.

This command does not grant execution authority or independent Evaluation
acceptance. No expected checkpoint outputs are generated. All tensor columns,
failed forwards and zero-quota steps are retained. The pinned model's XPU import
is currently unsupported and fails closed before checkpoint/model construction.
"""

from __future__ import annotations

import argparse
import dataclasses
import inspect
import json
import os
import platform
import random
import subprocess
import sys
from collections.abc import Callable, Iterator
from importlib import metadata
from pathlib import Path
from types import FrameType, ModuleType
from typing import Any

# The inherited launcher executes this file, rather than python -m.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import torch
from src.backends.text_contracts import TextRequest
from src.backends.u2_checkpoint import (
    BoundCheckpoint,
    CheckpointError,
    UnsupportedRuntimeError,
    imported_code,
    json_object,
    load_checkpoint,
    require_import_runtime,
    safe_path,
    sha256,
    verify_manifest,
)
from src.backends.u2_preflight import (
    SOURCE_HASHES,
    SOURCE_REVISION,
    U2PreflightRequirements,
    preflight_u2,
)
from src.backends.u2_reference import ExecutionControl, U2ReferenceAdapter
from src.backends.u2_text_renderer import TOKENIZER_ARTIFACT_NAMES, tokenizer_digest
from src.backends.u2_text_runtime import BOA, EOA, MASK, U2TextRuntime
from torch import Tensor

ACCEPTANCE_SHA256 = "103aeddaf7175f74513f29ca748d3b331726468b5d35fdeff033079c9843c48f"
_DTYPES = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}


class EvidenceError(RuntimeError):
    """A required observation is absent; this cannot become a parity pass."""


def write_json(path: Path, value: Any) -> None:
    safe_path(path.parent, path.name)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def tensor_record(root: Path, name: str, tensor: Tensor) -> dict[str, Any]:
    value = tensor.detach().cpu().contiguous()
    raw = value.reshape(-1).view(torch.uint8).numpy().tobytes()
    path = safe_path(root, name)
    with path.open("xb") as stream:
        stream.write(raw)
    return {
        "relative_path": name,
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "byte_order": sys.byteorder,
        "size_bytes": len(raw),
        "sha256": sha256(raw),
    }


def read_tensor(root: Path, record: dict[str, Any]) -> Tensor:
    raw = safe_path(root, record["relative_path"]).read_bytes()
    if (
        len(raw) != record["size_bytes"]
        or sha256(raw) != record["sha256"]
        or record["byte_order"] != sys.byteorder
    ):
        raise EvidenceError("tensor evidence integrity mismatch")
    dtype = getattr(torch, record["dtype"].removeprefix("torch."))
    value = torch.frombuffer(bytearray(raw), dtype=dtype).clone()
    return value.reshape(record["shape"])


def _list(value: Tensor) -> list[Any]:
    return list(value.detach().cpu().tolist())


def reset_rng(seed: int, device: torch.device) -> dict[str, Any]:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    states = {
        "cpu": sha256(torch.get_rng_state().numpy().tobytes()),
        "python": sha256(repr(random.getstate()).encode()),
        "numpy": sha256(repr(np.random.get_state()).encode()),
    }
    if device.type in ("cuda", "xpu"):
        backend = getattr(torch, device.type)
        backend.manual_seed_all(seed)
        states[device.type] = sha256(backend.get_rng_state(device).cpu().numpy().tobytes())
    return {"seed": seed, "state_sha256s": states}


class Observations:
    """Read-only Python line observations plus actual callable entry/return.

    No AST rewriting, sampler substitution or torch monkeypatching. Anchors are
    checked against the pinned function's source; an absent anchor fails closed.
    Line tracing is thread-local, serialized, and refused if another tracer exists.
    Review this instrumentation at the exact PRISM revision before execution.
    """

    def __init__(
        self,
        root: Path,
        label: str,
        path: str,
        function: Callable[..., Any],
        max_calls: int,
        case_id: str = "unspecified",
        repetition: int = 0,
    ) -> None:
        self.root, self.label, self.path = root, label, path
        self.function = inspect.unwrap(function)
        self.code = self.function.__code__
        self.calls: list[dict[str, Any]] = []
        self.blocks: list[dict[str, Any]] = []
        self.pending: dict[str, Any] | None = None
        self.max_calls = max_calls
        self.case_id, self.repetition = case_id, repetition
        self.frame: FrameType | None = None
        lines, start = inspect.getsourcelines(self.function)
        anchors = (
            ("for step_i in range(", "decoded_block =", "all_decoded.append(decoded_block)")
            if path == "oracle"
            else ("for step in range(", "block: list[int] =", "if MASK in block:")
        )
        matched: list[int] = []
        for anchor in anchors:
            hits = [start + i for i, line in enumerate(lines) if line.strip().startswith(anchor)]
            if len(hits) != 1:
                raise EvidenceError(f"unavailable instrumentation anchor: {path}: {anchor}")
            matched.append(hits[0])
        self.step_line, self.end_line, self.block_line = matched
        self.previous: Any = None
        self._journal_created = False

    def __enter__(self) -> Observations:
        self.previous = sys.gettrace()
        if self.previous is not None:
            raise EvidenceError("another trace function is active")
        sys.settrace(self.trace)
        return self

    def __exit__(self, *exc: Any) -> None:
        sys.settrace(self.previous)
        self.frame = None

    def trace(self, frame: FrameType, event: str, arg: Any) -> Any:
        if self.path == "prism" and frame.f_code is ExecutionControl.commit.__code__:
            if event == "return" and self.blocks:
                control = frame.f_locals["self"]
                previous = sum(len(b["committed_output_ids"]) for b in self.blocks[:-1])
                self.blocks[-1]["committed_output_ids"] = list(control._output[previous:])
                self.blocks[-1]["committed_block_ends"] = list(control._block_ends)
                self.blocks[-1]["commit_observation"] = "ExecutionControl.commit return"
            return self.trace
        if frame.f_code is not self.code:
            return None
        self.frame = frame
        if event == "line":
            if frame.f_lineno in (self.step_line, self.end_line):
                self.finish_step(frame.f_locals)
            if frame.f_lineno == self.block_line:
                values = frame.f_locals
                block = (
                    _list(values["decoded_block"])
                    if self.path == "oracle"
                    else list(values["block"])
                )
                eoa = block.index(EOA) if EOA in block else None
                prefix_output = [x for b in self.blocks for x in b["committed_output_ids"]]
                committed = block if eoa is None else block[:eoa]
                ends = [
                    b["committed_block_ends"][-1] for b in self.blocks if b["committed_output_ids"]
                ]
                if committed:
                    ends.append(len(prefix_output) + len(committed))
                if self.path == "prism":
                    committed = []
                    ends = self.blocks[-1]["committed_block_ends"] if self.blocks else []
                self.blocks.append(
                    {
                        "block_index": len(self.blocks),
                        "completed_block_ids": block,
                        "first_eoa_local_index": eoa,
                        "committed_output_ids": committed,
                        "committed_block_ends": ends,
                    }
                )
        return self.trace

    def journal(self, event: str, entry: dict[str, Any]) -> None:
        path = safe_path(self.root, f"{self.label}-events.jsonl")
        with path.open("a" if self._journal_created else "x", encoding="utf-8") as stream:
            self._journal_created = True
            stream.write(json.dumps({"event": event, **entry}, allow_nan=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())

    def finish_step(self, values: dict[str, Any]) -> None:
        entry = self.pending
        if entry is None or not entry["return_observed"]:
            return
        oracle = self.path == "oracle"
        quota_name = "k" if oracle else "quota"
        if quota_name not in values:
            raise EvidenceError("returned forward lacks observed quota")
        quota = int(values[quota_name])
        entry["transfer_quota"] = quota
        if quota:
            selected = (
                values["select_index"] if oracle else values["selected"] + values["code_start"]
            )
            entry["selected_absolute_indices"] = _list(selected)
        else:
            entry["selected_absolute_indices"] = []
        if oracle or quota:
            candidates = values["x0_mask"] if oracle else values["candidates"]
            entry["observed_candidate_ids"] = _list(candidates)
            entry["selection_observation"] = "actual sampler locals"
        else:
            entry["selection_observation"] = "zero-quota: candidates/confidence are diagnostic-only"
        logits = read_tensor(self.root, entry["raw_logits"]).to(torch.float64)
        candidates = logits.argmax(-1)
        confidence = torch.softmax(logits, -1).gather(1, candidates[:, None]).squeeze(1)
        entry["diagnostic_candidate_ids"] = _list(candidates)
        entry["diagnostic_candidate_confidence"] = _list(confidence)
        if oracle or quota:
            actual_ids = values["x0_mask"] if oracle else values["candidates"]
            actual_confidence = (
                values["x0_p_mask"]
                if oracle
                else values["probabilities"].gather(1, actual_ids[:, None]).squeeze(1)
            )
            entry["candidate_ids"] = _list(actual_ids)
            entry["candidate_confidence"] = _list(actual_confidence)
            confidence = actual_confidence.detach().cpu()
        else:
            entry["candidate_ids"] = _list(candidates)
            entry["candidate_confidence"] = _list(confidence)
        top = torch.topk(logits, k=2, dim=-1).values
        entry["argmax_ties"] = [_list(torch.nonzero(row == row.max()).flatten()) for row in logits]
        entry["near_tie_margins"] = {"argmax": _list(top[:, 0] - top[:, 1])}
        current = torch.tensor(entry["input_ids"])[0, entry["code_start"] :, 0]
        eligible = torch.nonzero(current == MASK).flatten()
        ranked = torch.sort(confidence[eligible], descending=True).values
        boundary = None
        tied: list[int] = []
        if 0 < quota < len(eligible):
            kth, following = ranked[quota - 1].item(), ranked[quota].item()
            boundary = {"kth": kth, "next": following, "gap": kth - following}
            tied = _list(eligible[confidence[eligible] == kth] + entry["code_start"])
        entry["selection_boundary_ties"] = tied
        entry["near_tie_margins"]["selection_boundary"] = boundary
        entry["post_selection_ids"] = _list(values["x"])
        self.pending = None

    def forward(
        self, operation: Callable[[], Tensor], x: Tensor, types: Tensor, kwargs: dict[str, Any]
    ) -> Tensor:
        if self.frame is None:
            raise EvidenceError("forward occurred outside the observed generator frame")
        if len(self.calls) >= self.max_calls:
            raise EvidenceError("oracle forward budget exhausted")
        values = self.frame.f_locals
        oracle = self.path == "oracle"
        block = int(
            values["blk_idx"] if oracle else values["offset"] // values["request"].block_size
        )
        step = int(values["step_i"] if oracle else values["step"])
        code_start = int(values["code_start"])
        entry: dict[str, Any] = {
            "case_id": self.case_id,
            "repetition": self.repetition,
            "path": self.path,
            "call_index": len(self.calls),
            "start_observed": True,
            "return_observed": False,
            "exception": None,
            "block_index": block,
            "step_index": step,
            "sequence_length": x.shape[1],
            "processed_positions_on_return": 0,
            "code_start": code_start,
            "input_ids": _list(x),
            "type_ids": _list(types),
            "turn_ids": _list(values["turn_ids"] if oracle else values["turns"]),
            "block_ids": _list(values["block_ids"] if oracle else values["blocks"]),
            "is_clean": _list(kwargs["is_clean"]),
            "position_ids": _list(kwargs["position_ids"]),
            "attention_mask": tensor_record(
                self.root, f"{self.label}-{len(self.calls)}-mask.bin", kwargs["attention_mask"]
            ),
        }
        self.calls.append(entry)
        self.pending = entry
        self.journal("forward_start", entry)
        try:
            result = operation()
        except BaseException as exc:
            entry["exception"] = f"{type(exc).__name__}: {exc}"
            self.journal("forward_exception", entry)
            raise
        entry["return_observed"] = True
        entry["processed_positions_on_return"] = x.shape[1]
        self.journal("forward_return", entry)
        entry["raw_logits"] = tensor_record(
            self.root, f"{self.label}-{entry['call_index']}-logits.bin", result
        )
        if (
            result.ndim != 2
            or result.shape[0] != x.shape[1] - code_start
            or not bool(torch.isfinite(result).all())
        ):
            raise CheckpointError("invalid/nonfinite full logits")
        return result


class ObservedModel:
    """One shared bound object, with independent observers for each path."""

    def __init__(self, bound: BoundCheckpoint, observer: Observations) -> None:
        self.bound, self.observer = bound, observer
        self.spec = bound.spec

    @property
    def training(self) -> bool:
        return self.bound.training

    def eval(self) -> ObservedModel:
        self.bound.eval()
        return self

    def parameters(self) -> Iterator[Tensor]:
        return iter(self.bound.model.parameters())

    def __call__(
        self,
        input_ids: Tensor,
        type_position_ids: Tensor,
        *,
        attention_mask: Tensor,
        infer_mmu: bool,
        is_clean: Tensor,
        position_ids: Tensor,
    ) -> Tensor:
        self.bound.verify_binding()
        kwargs: dict[str, Any] = dict(
            attention_mask=attention_mask,
            infer_mmu=infer_mmu,
            is_clean=is_clean,
            position_ids=position_ids,
        )
        result = self.observer.forward(
            lambda: self.bound.model(input_ids, type_position_ids, **kwargs),
            input_ids,
            type_position_ids,
            kwargs,
        )
        self.bound.verify_binding()
        if (
            result.dtype != self.spec.dtype
            or result.device != self.spec.device
            or result.shape[1] != self.spec.identity.vocab_size
        ):
            raise CheckpointError("actual output dtype/device/model dimension mismatch")
        return result


def run_path(
    bound: BoundCheckpoint,
    generator: ModuleType,
    report: Any,
    request: TextRequest,
    path: str,
    root: Path,
    label: str,
    *,
    case_id: str | None = None,
    repetition: int = 0,
) -> dict[str, Any]:
    oracle = path == "oracle"
    function = generator.generate_text_bd if oracle else U2TextRuntime.generate
    observer = Observations(
        root,
        label,
        path,
        function,
        request.max_model_calls,
        case_id or request.request_id,
        repetition,
    )
    result: dict[str, Any] = {
        "path": path,
        "calls": observer.calls,
        "blocks": observer.blocks,
        "rng": reset_rng(request.seed, bound.spec.device),
    }
    model = ObservedModel(bound, observer)
    interruption: BaseException | None = None
    try:
        bound.verify_binding(full=True)
        with observer:
            if oracle:
                device = bound.spec.device
                prefix = [*request.prompt_ids, BOA]
                n = len(prefix)

                def tensor(xs: list[int]) -> Tensor:
                    return torch.tensor(xs, dtype=torch.long, device=device)

                output = generator.generate_text_bd(
                    model,
                    tensor(prefix)[:, None].repeat(1, 8),
                    tensor([1] * n),
                    gen_length=request.max_new_tokens,
                    block_length=request.block_size,
                    steps_per_block=request.steps_per_block,
                    temperature=0.0,
                    cfg_scale=0.0,
                    remasking="low_confidence",
                    mask_id=MASK,
                    answer_type_id=3,
                    early_stop_tokens=[EOA],
                    tokenizer=bound.tokenizer.tokenizer,
                    prefix_turn_ids=tensor([0] * (n - 1) + [1]),
                    prefix_block_ids=tensor([-1] * (n - 1) + [0]),
                    prefix_is_clean=tensor([1] * (n - 1) + [0]),
                    prefix_position_ids=tensor(list(range(n))),
                    generated_turn_id=1,
                    generated_position_start=n,
                    require_meta_mask=True,
                )
                result["output_ids"] = _list(output)
                result["finish_reason"] = (
                    "eoa"
                    if any(b["first_eoa_local_index"] is not None for b in observer.blocks)
                    else "length"
                )
                result["committed_block_ends"] = (
                    observer.blocks[-1]["committed_block_ends"] if observer.blocks else []
                )
            else:
                terminal = U2ReferenceAdapter(U2TextRuntime(model, preflight=report)).generate(
                    request
                )
                result.update(json.loads(json.dumps(dataclasses.asdict(terminal))))
        bound.verify_binding(full=True)
        ids = result["output_ids"]
        # Never invent decoding for the seven config-only positions. Preserve IDs
        # and logits, then report incomplete if the actual tokenizer cannot map them.
        if any(i >= bound.tokenizer.vocab_size for i in ids):
            raise EvidenceError(
                "selected model position has no serialized tokenizer ID; decoding unavailable"
            )
    except BaseException as exc:
        result["exception"] = f"{type(exc).__name__}: {exc}"
        result.setdefault("finish_reason", "error")
        if not isinstance(exc, Exception):
            interruption = exc
    result.setdefault(
        "output_ids",
        [token for block in observer.blocks for token in block["committed_output_ids"]],
    )
    result.setdefault(
        "committed_block_ends",
        observer.blocks[-1]["committed_block_ends"] if observer.blocks else [],
    )
    result.setdefault("error_code", "comparison_incomplete" if "exception" in result else None)
    result["observed_usage"] = {
        "forward_starts": len(observer.calls),
        "forward_returns": sum(c["return_observed"] for c in observer.calls),
        "processed_positions": sum(c["processed_positions_on_return"] for c in observer.calls),
        "complete": "exception" not in result and result.get("finish_reason") in ("length", "eoa"),
    }
    if oracle:
        usage = result["observed_usage"]
        result["usage"] = {
            "prompt_tokens": len(request.prompt_ids),
            "completion_tokens": len(result["output_ids"]),
            "observed_model_calls": usage["forward_starts"],
            "observed_processed_positions": usage["processed_positions"],
            "model_calls": usage["forward_starts"] if usage["complete"] else None,
            "processed_positions": usage["processed_positions"] if usage["complete"] else None,
            "complete": usage["complete"],
        }
    result["eoa_observed"] = any(b["first_eoa_local_index"] is not None for b in observer.blocks)
    write_json(root / f"{label}-trace.json", result)
    if interruption is not None:
        raise interruption
    return result


def numeric_summary(oracle: Tensor, prism: Tensor, limits: dict[str, float]) -> dict[str, Any]:
    if oracle.shape != prism.shape:
        return {"shape_match": False, "violation_count": 1}
    left, right = oracle.to(torch.float64), prism.to(torch.float64)
    if not bool(torch.isfinite(left).all() & torch.isfinite(right).all()):
        return {"finite": False, "violation_count": 1}
    error = (right - left).abs()
    normalized = error / (limits["atol"] + limits["rtol"] * left.abs())
    bad = torch.nonzero(normalized > 1)
    nonzero = left != 0
    return {
        "compared_scalar_count": left.numel(),
        "max_absolute_error": error.max().item(),
        "max_relative_error_nonzero_oracle": (error[nonzero] / left[nonzero].abs()).max().item()
        if bool(nonzero.any())
        else 0.0,
        "max_normalized_error": normalized.max().item(),
        "violation_count": bad.shape[0],
        "first_violation": _list(bad[0]) if len(bad) else None,
    }


def compare_paths(
    oracle: dict[str, Any], prism: dict[str, Any], root: Path, limits: dict[str, Any]
) -> dict[str, Any]:
    failures: list[str] = []
    missing: list[str] = []
    numeric: list[dict[str, Any]] = []
    forward_fields = (
        "case_id",
        "repetition",
        "path",
        "call_index",
        "start_observed",
        "return_observed",
        "exception",
        "block_index",
        "step_index",
        "sequence_length",
        "processed_positions_on_return",
        "input_ids",
        "type_ids",
        "turn_ids",
        "block_ids",
        "is_clean",
        "position_ids",
        "attention_mask",
        "raw_logits",
        "candidate_ids",
        "candidate_confidence",
        "transfer_quota",
        "selected_absolute_indices",
        "argmax_ties",
        "selection_boundary_ties",
        "near_tie_margins",
    )
    for path in (oracle, prism):
        for key in ("finish_reason", "error_code", "eoa_observed", "usage"):
            if key not in path:
                missing.append(f"{path['path']}: terminal {key}")
        for call in path["calls"]:
            missing.extend(
                f"{path['path']}: call {call.get('call_index')}: {key}"
                for key in forward_fields
                if key not in call
            )
        for block in path["blocks"]:
            missing.extend(
                f"{path['path']}: block: {key}"
                for key in (
                    "block_index",
                    "completed_block_ids",
                    "first_eoa_local_index",
                    "committed_output_ids",
                    "committed_block_ends",
                )
                if key not in block
            )
        if path.get("exception") or not path["observed_usage"]["complete"]:
            missing.append(f"{path['path']}: {path.get('exception', path.get('finish_reason'))}")
    for field in (
        "output_ids",
        "committed_block_ends",
        "finish_reason",
        "eoa_observed",
        "rng",
        "observed_usage",
    ):
        if field not in oracle or field not in prism:
            missing.append(field)
        elif oracle[field] != prism[field]:
            failures.append(field)

    def normalized_blocks(path: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {k: v for k, v in block.items() if k != "commit_observation"}
            for block in path["blocks"]
        ]

    if normalized_blocks(oracle) != normalized_blocks(prism):
        failures.append("blocks")
    if len(oracle["calls"]) != len(prism["calls"]):
        failures.append("forward count")
    for index, (left, right) in enumerate(zip(oracle["calls"], prism["calls"], strict=False)):
        for field in (
            "start_observed",
            "return_observed",
            "block_index",
            "step_index",
            "sequence_length",
            "processed_positions_on_return",
            "input_ids",
            "type_ids",
            "turn_ids",
            "block_ids",
            "is_clean",
            "position_ids",
            "transfer_quota",
            "candidate_ids",
            "post_selection_ids",
        ):
            if field not in left or field not in right:
                missing.append(f"call {index}: {field}")
            elif left[field] != right[field]:
                failures.append(f"call {index}: {field}")
        for entry in (left, right):
            selected = entry.get("selected_absolute_indices", [])
            if len(selected) != len(set(selected)):
                failures.append(f"call {index}: duplicate reveal index")
        if set(left.get("selected_absolute_indices", [])) != set(
            right.get("selected_absolute_indices", [])
        ):
            failures.append(f"call {index}: reveal set")
        margins = left.get("near_tie_margins", {})
        boundary = margins.get("selection_boundary")
        tie_summary = {
            "minimum_argmax_margin": min(margins.get("argmax", []), default=None),
            "minimum_selection_boundary_margin": boundary["gap"] if boundary else None,
            "tie_count": sum(len(ids) > 1 for ids in left.get("argmax_ties", []))
            + int(len(left.get("selection_boundary_ties", [])) > 1),
        }
        for field in ("attention_mask", "raw_logits"):
            if field not in left or field not in right:
                missing.append(f"call {index}: {field}")
                continue
            a, b = read_tensor(root, left[field]), read_tensor(root, right[field])
            if field == "attention_mask":
                if not torch.equal(a, b):
                    failures.append(f"call {index}: mask")
            else:
                summary = numeric_summary(a, b, limits["raw_logits"])
                summary.update({"call_index": index, "quantity": field, **tie_summary})
                numeric.append(summary)
                if summary["violation_count"]:
                    failures.append(f"call {index}: logits")
        if "candidate_confidence" in left and "candidate_confidence" in right:
            summary = numeric_summary(
                torch.tensor(left["candidate_confidence"], dtype=torch.float64),
                torch.tensor(right["candidate_confidence"], dtype=torch.float64),
                limits["candidate_confidence"],
            )
            summary.update({"call_index": index, "quantity": "candidate_confidence", **tie_summary})
            numeric.append(summary)
            if summary["violation_count"]:
                failures.append(f"call {index}: confidence")
        else:
            missing.append(f"call {index}: confidence")
    return {
        "status": "fail" if failures else "incomplete" if missing else "matched",
        "failures": failures,
        "missing": missing,
        "numeric_summaries": numeric,
    }


def observed_coverage(
    paths: list[dict[str, Any]], block_size: int, max_new_tokens: int
) -> set[str]:
    coverage: set[str] = set()
    for path in paths:
        blocks = path["blocks"]
        if len(blocks) > 1:
            coverage.add("multiple_blocks")
        if any(len(b["completed_block_ids"]) == block_size for b in blocks):
            coverage.add("complete_normal_block")
        if any(len(b["completed_block_ids"]) < block_size for b in blocks):
            coverage.add("short_final_block")
        if any(c.get("transfer_quota") == 0 and c["return_observed"] for c in path["calls"]):
            coverage.add("zero_transfer_forward")
        if path.get("finish_reason") == "eoa":
            coverage.add("eoa_stop")
        if (
            path.get("finish_reason") == "length"
            and len(path.get("output_ids", [])) == max_new_tokens
        ):
            coverage.add("length_stop")
    return coverage


def run_cases(
    bound: BoundCheckpoint, generator: ModuleType, report: Any, fixture: dict[str, Any], root: Path
) -> tuple[list[dict[str, Any]], list[str]]:
    cases: list[dict[str, Any]] = []
    coverage: set[str] = set()
    for case in fixture["cases"]:
        rendered = bound.tokenizer.render(
            system_prompt=case["system_prompt"], user_text=case["user_text"]
        )
        settings = case["settings"]
        if (settings["temperature"], settings["cfg_scale"], settings["remasking"]) != (
            0.0,
            0.0,
            "low_confidence",
        ):
            raise EvidenceError("unsupported frozen sampler settings")
        repeats: list[dict[str, Any]] = []
        for repetition, order in enumerate(fixture["execution"]["path_order"]):
            request = TextRequest(
                request_id=f"{case['id']}-{repetition}",
                prompt_ids=rendered.prompt_ids,
                tokenizer_digest=bound.tokenizer.tokenizer_digest,
                **{
                    k: settings[k]
                    for k in (
                        "seed",
                        "block_size",
                        "steps_per_block",
                        "max_new_tokens",
                        "max_model_calls",
                    )
                },
            )
            paths: dict[str, dict[str, Any]] = {}
            for path in order.split("_then_"):
                paths[path] = run_path(
                    bound,
                    generator,
                    report,
                    request,
                    path,
                    root,
                    f"{case['id']}-{repetition}-{path}",
                    case_id=case["id"],
                    repetition=repetition,
                )
            comparison = compare_paths(
                paths["oracle"],
                paths["prism"],
                root,
                fixture["numeric_tolerances"][str(bound.spec.dtype).removeprefix("torch.")],
            )
            observed = observed_coverage(
                list(paths.values()), request.block_size, request.max_new_tokens
            )
            coverage.update(observed)
            repeats.append(
                {
                    "request": {**dataclasses.asdict(request), **settings},
                    "rendered_prefix": dataclasses.asdict(rendered),
                    **paths,
                    "discrete_comparisons": comparison,
                    "numeric_summaries": comparison["numeric_summaries"],
                    "threshold_reference": {
                        "acceptance_sha256": ACCEPTANCE_SHA256,
                        "dtype": str(bound.spec.dtype),
                    },
                    "coverage": sorted(observed),
                }
            )
        # Reset RNG plus reversed execution order must reproduce each path's evidence.
        for path in ("oracle", "prism"):
            repeat = compare_paths(
                repeats[0][path],
                repeats[1][path],
                root,
                fixture["numeric_tolerances"][str(bound.spec.dtype).removeprefix("torch.")],
            )
            if repeat["status"] != "matched":
                repeats[1]["discrete_comparisons"]["failures"].append(
                    f"{path} repeatability: {repeat}"
                )
                repeats[1]["discrete_comparisons"]["status"] = (
                    "fail" if repeat["failures"] else "incomplete"
                )
        cases.append({"id": case["id"], "repetitions": repeats})
    return cases, sorted(set(fixture["coverage"]["required_observed"]) - coverage)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-root",
        "checkpoint-root",
        "output-dir",
        "acceptance-file",
        "asset-manifest",
        "source-manifest",
    ):
        result.add_argument("--" + name, required=True, type=Path)
    result.add_argument("--device", required=True)
    result.add_argument("--dtype", required=True, choices=tuple(_DTYPES))
    result.add_argument(
        "--run-context",
        type=Path,
        help="Private reviewed identity/allocation evidence, never authority by itself",
    )
    return result


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=Path(__file__).resolve().parents[2], text=True
    ).strip()


def execute(args: argparse.Namespace) -> dict[str, Any]:
    result: dict[str, Any] = {
        "schema_version": 1,
        "evidence_level": "proposed",
        "complete": False,
        "status": "incomplete",
        "failure_reasons": [],
        "identities": {},
        "runtime": {},
        "cases": [],
        "artifacts": [],
        "independent_acceptance": "pending Evaluation; this driver cannot self-award acceptance",
    }
    root = safe_path(args.output_dir)
    for input_path in (
        args.source_root,
        args.checkpoint_root,
        args.acceptance_file,
        args.asset_manifest,
        args.source_manifest,
    ):
        safe_path(input_path)
        if root.is_relative_to(input_path.absolute()) or input_path.absolute().is_relative_to(root):
            raise CheckpointError("output/input overlap")
    root.mkdir(parents=True, exist_ok=True)
    if (root / "result.json").exists() or (root / "artifact-manifest.json").exists():
        raise EvidenceError("output already contains a result; no overwrite/retry")
    if any(
        p.name.endswith(("-trace.json", "-mask.bin", "-logits.bin", "-events.jsonl"))
        for p in root.iterdir()
    ):
        raise EvidenceError(
            "output contains partial prior evidence; use a newly admitted output directory"
        )
    try:
        fixture_raw = safe_path(args.acceptance_file.parent, args.acceptance_file.name).read_bytes()
        if sha256(fixture_raw) != ACCEPTANCE_SHA256:
            raise CheckpointError("frozen acceptance bytes differ")
        fixture = json_object(fixture_raw)
        result["identities"].update(
            {
                "acceptance_sha256": ACCEPTANCE_SHA256,
                "prism_commit": _git("rev-parse", "HEAD"),
                "prism_tree": _git("rev-parse", "HEAD^{tree}"),
                "prism_tree_clean": not bool(_git("status", "--porcelain")),
                "oracle_revision": SOURCE_REVISION,
            }
        )
        if not result["identities"]["prism_tree_clean"]:
            raise EvidenceError("comparison requires exact clean reviewed PRISM revision")
        if args.run_context is None:
            raise EvidenceError(
                "private run context absent: acceptance revision, instrumentation review, ordered modules and authorized allocation unobserved"
            )
        context_raw = safe_path(args.run_context.parent, args.run_context.name).read_bytes()
        context = json_object(context_raw)
        required = (
            "acceptance_document_revision",
            "oracle_instrumentation_diff_and_review",
            "modules_in_order",
            "authorized_job_id",
            "execution_allocation",
            "imported_source_manifest",
            "imported_source_manifest_sha256",
        )
        missing = [k for k in required if not context.get(k)]
        if missing:
            raise EvidenceError(f"private run context missing: {missing}")
        result["run_context"] = {
            "sha256": sha256(context_raw),
            "source": str(args.run_context),
            "values": context,
        }
        result["identities"].update({k: context[k] for k in required[:2]})
        if (
            not isinstance(context["modules_in_order"], list)
            or not all(isinstance(item, str) and item for item in context["modules_in_order"])
            or not isinstance(context["execution_allocation"], dict)
        ):
            raise EvidenceError("invalid ordered module/allocation context")
        observed_job = os.environ.get("PBS_JOBID")
        observed_modules = os.environ.get("LOADEDMODULES", "").split(":")
        result["allocation_observation"] = {
            "job_id": observed_job,
            "source": "PBS_JOBID environment",
            "modules_in_order": observed_modules if observed_modules != [""] else None,
            "module_source": "LOADEDMODULES environment",
            "authority": "observations corroborate private context; they do not authorize execution",
        }
        if not observed_job or observed_modules == [""]:
            raise EvidenceError("observed PBS allocation/module evidence unavailable")
        if (
            observed_job != context["authorized_job_id"]
            or observed_modules != context["modules_in_order"]
        ):
            raise CheckpointError(
                "observed allocation/modules differ from admitted private context"
            )
        device = torch.device(args.device)
        if device.type != "xpu" or device.index is None:
            raise UnsupportedRuntimeError(
                "frozen Aurora comparison requires explicit indexed xpu:N"
            )
        result["runtime"] = {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "package_versions": {
                p: metadata.version(p) for p in ("torch", "transformers", "safetensors", "numpy")
            },
            "device_type": device.type,
            "device_index": device.index,
            "dtype": args.dtype,
            "modules_in_order": context["modules_in_order"],
            "authorized_job_id": context["authorized_job_id"],
            "execution_allocation": context["execution_allocation"],
            "attention_backend": "sdpa",
            "deterministic_settings": {
                "algorithms_enabled": torch.are_deterministic_algorithms_enabled()
            },
        }
        source = verify_manifest(
            args.source_root,
            Path(context["imported_source_manifest"]),
            expected_sha256=context["imported_source_manifest_sha256"],
            kind="source",
        )
        result["identities"]["oracle_six_file_manifest"] = SOURCE_HASHES
        # This explicit capability gate currently refuses XPU before shard access.
        require_import_runtime(device)
        backend = getattr(torch, device.type)
        if not backend.is_available() or device.index >= backend.device_count():
            raise UnsupportedRuntimeError("selected indexed device unavailable; no fallback")
        result["runtime"]["device_identity"] = str(backend.get_device_properties(device))
        files = verify_manifest(
            args.checkpoint_root,
            args.asset_manifest,
            expected_sha256=fixture["checkpoint"]["operator_asset_manifest_sha256"],
            kind="checkpoint",
            variant_prefix="nvidia/",
        )
        artifacts = {k: files.read(k) for k in TOKENIZER_ARTIFACT_NAMES if k in files.files}
        report = preflight_u2(
            args.source_root,
            args.checkpoint_root,
            U2PreflightRequirements(
                tokenizer_digest(artifacts),
                "llada2_moe",
                dtype=args.dtype,
                artifact_hashes={k: v["sha256"] for k, v in files.files.items()},
            ),
            source_manifest=args.source_manifest,
        )
        bound, generator = load_checkpoint(
            files, source, device=device, dtype=_DTYPES[args.dtype], preflight=report
        )
        result["identities"].update(
            {
                "loaded_model_binding_provenance": bound.provenance,
                "strict_load_key_shape_dtype_report": bound.provenance["strict_load"],
                "checkpoint_repository": files.identity["repository"],
                "checkpoint_revision": files.identity["revision"],
                "checkpoint_variant": files.identity["variant"],
                "checkpoint_shard_sha256s": {
                    k: v["sha256"] for k, v in files.files.items() if k.endswith(".safetensors")
                },
                "checkpoint_index_sha256": files.files["model.safetensors.index.json"]["sha256"],
                "checkpoint_config_sha256": files.files["config.json"]["sha256"],
                "checkpoint_bundled_code_sha256s": {
                    k: v["sha256"] for k, v in files.files.items() if k.endswith(".py")
                },
                "tokenizer_binding_provenance": dict(bound.tokenizer.construction_provenance),
                "tokenizer_digest": bound.tokenizer.tokenizer_digest,
                "tokenizer_artifact_sha256s": dict(bound.tokenizer.artifact_hashes),
                "special_token_ids": dict(bound.tokenizer.special_token_ids),
                "renderer_revision": bound.tokenizer.renderer_revision,
            }
        )
        result["cases"], missing_coverage = run_cases(bound, generator, report, fixture, root)
        bound.verify_binding(full=True)
        result["identities"]["tokenizer_mutation_check"] = "verified before/after every path"
        result["identities"]["oracle_imported_code_manifest"] = imported_code(source)
        result["runtime"]["seed_and_rng_state_hashes"] = [
            {"case": c["id"], "repetition": i, "path": path, **r[path]["rng"]}
            for c in result["cases"]
            for i, r in enumerate(c["repetitions"])
            for path in ("oracle", "prism")
        ]
        result["evidence_level"] = "measured"
        comparisons = [r["discrete_comparisons"] for c in result["cases"] for r in c["repetitions"]]
        failures = [f for c in comparisons for f in c["failures"]]
        missing = [m for c in comparisons for m in c["missing"]]
        missing.extend(
            f"identity: {key}"
            for key in fixture["required_identity_fields"]
            if key not in result["identities"]
        )
        missing.extend(
            f"runtime: {key}"
            for key in fixture["required_runtime_fields"]
            if key not in result["runtime"]
        )
        result["raw_comparison"] = {
            "complete": not missing and not missing_coverage,
            "status": "fail"
            if failures
            else "incomplete"
            if missing or missing_coverage
            else "matched",
            "failures": failures,
            "missing": missing,
            "missing_coverage": missing_coverage,
        }
        result["status"] = "fail" if failures else "incomplete"
        result["failure_reasons"] = (
            failures + missing + [f"missing coverage: {k}" for k in missing_coverage]
        )
        result["failure_reasons"].append("independent Evaluation acceptance pending")
    except BaseException as exc:
        result["failure_reasons"].append(f"{type(exc).__name__}: {exc}")
        if isinstance(exc, CheckpointError):
            result["status"] = "fail"
    for path in sorted(root.iterdir()):
        if path.name.endswith(("-trace.json", "-mask.bin", "-logits.bin", "-events.jsonl")):
            raw = safe_path(root, path.name).read_bytes()
            result["artifacts"].append(
                {"relative_path": path.name, "size_bytes": len(raw), "sha256": sha256(raw)}
            )
    write_json(root / "result.json", result)
    raw = (root / "result.json").read_bytes()
    write_json(
        root / "artifact-manifest.json",
        {
            "schema_version": 1,
            "files": [
                *result["artifacts"],
                {"relative_path": "result.json", "size_bytes": len(raw), "sha256": sha256(raw)},
            ],
        },
    )
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        result = execute(args)
    except Exception as exc:
        print(f"U2 comparison refused: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({"status": result["status"], "failure_reasons": result["failure_reasons"]}))
    return 1 if result["status"] == "fail" else 2


if __name__ == "__main__":
    raise SystemExit(main())
