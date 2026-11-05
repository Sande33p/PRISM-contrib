"""Private runner failures and actual pinned generator with synthetic CPU logits.

Set U2_PINNED_SOURCE_DIR to an already available read-only source checkout to
exercise the actual oracle. No fixture downloads or real model imports occur.
"""

import hashlib
import importlib
import json
import os
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from src.backends.text_contracts import RuntimeIdentity, TextRequest
from src.backends.u2_preflight import SOURCE_HASHES, SOURCE_REVISION, preflight_u2
from src.backends.u2_text_runtime import LocalModelSpec
from tools.parity import u2_text as runner

from test_u2_preflight import bundle as _bundle
from test_u2_text_renderer import SPECIALS

bundle = _bundle
FIXTURE = Path(__file__).parent / "fixtures/u2_aurora_text_parity_v1.json"


@pytest.fixture(scope="module", autouse=True)
def cpu_threads():
    before = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(before)


@pytest.fixture
def generator(monkeypatch):
    location = os.environ.get("U2_PINNED_SOURCE_DIR")
    if not location:
        pytest.skip("actual pinned source must be supplied locally; no download permitted")
    source = Path(location)
    for name, digest in SOURCE_HASHES.items():
        assert hashlib.sha256((source / name).read_bytes()).hexdigest() == digest
    previous = {k: v for k, v in sys.modules.items() if k.split(".")[0] in ("utils", "generators")}
    for name in previous:
        del sys.modules[name]
    monkeypatch.syspath_prepend(str(source))
    try:
        module = importlib.import_module("generators.text_understanding_generator")
        assert Path(module.__file__).resolve().is_relative_to(source.resolve())
        yield module
    finally:
        for name in list(sys.modules):
            if name.split(".")[0] in ("utils", "generators"):
                del sys.modules[name]
        sys.modules.update(previous)


@pytest.fixture
def report(bundle):
    source, checkpoint, requirements = bundle
    config = json.loads((checkpoint / "config.json").read_bytes())
    config["torch_dtype"] = "float32"
    config["vocab_size"] = 157184
    (checkpoint / "config.json").write_text(json.dumps(config))
    return preflight_u2(source, checkpoint, replace(requirements, dtype="float32"))


class SyntheticBound:
    def __init__(self, report, winner=101, fail_call=None):
        owner = self

        class SyntheticModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.zeros(1))

            def forward(self, x, types, **kwargs):
                return owner(x, types, **kwargs)

        self.model = SyntheticModule().eval()
        self.spec = LocalModelSpec(
            RuntimeIdentity(
                "synthetic-runner", SOURCE_REVISION, report.tokenizer_digest, 157184, 157166, 156895
            ),
            SPECIALS,
            torch.float32,
            torch.device("cpu"),
            report.max_position_embeddings,
        )
        self.tokenizer = SimpleNamespace(
            vocab_size=157177, tokenizer=object(), verify_binding=lambda: None
        )
        self.winner, self.fail_call, self.calls = winner, fail_call, 0

    def verify_binding(self, *, full=False):
        assert not self.model.training

    @property
    def training(self):
        return self.model.training

    def eval(self):
        self.model.eval()
        return self

    def __call__(self, x, types, **kwargs):
        self.calls += 1
        if self.calls == self.fail_call:
            raise RuntimeError("synthetic forward failure")
        n = int(((types == 3) & (kwargs["is_clean"] == 0)).sum())
        logits = torch.full((n, 157184), -20.0)
        logits[:, 100] = 0.0
        for row in range(n):
            logits[row, self.winner if row == 0 else 102 + row] = 8 - row
        return logits


def request(report, **kwargs):
    values = dict(
        request_id="synthetic",
        prompt_ids=(41, 42),
        tokenizer_digest=report.tokenizer_digest,
        max_new_tokens=5,
        block_size=3,
        steps_per_block=2,
        max_model_calls=32,
        seed=1729,
    )
    values.update(kwargs)
    return TextRequest(**values)


def run_pair(tmp_path, generator, report, **settings):
    bound = SyntheticBound(report)
    req = request(report, **settings)
    oracle = runner.run_path(bound, generator, report, req, "oracle", tmp_path, "oracle")
    prism = runner.run_path(bound, generator, report, req, "prism", tmp_path, "prism")
    return (
        oracle,
        prism,
        runner.compare_paths(
            oracle,
            prism,
            tmp_path,
            json.loads(FIXTURE.read_bytes())["numeric_tolerances"]["float32"],
        ),
    )


@pytest.mark.parametrize(
    "settings", [{}, {"max_new_tokens": 2, "block_size": 4, "steps_per_block": 5}]
)
def test_actual_pinned_generator_full_logits_masks_selection_and_zero_steps(
    tmp_path, generator, report, settings
):
    oracle, prism, comparison = run_pair(tmp_path, generator, report, **settings)
    assert comparison["status"] == "matched", comparison
    assert oracle["output_ids"] == prism["output_ids"]
    assert oracle["rng"] == prism["rng"]
    assert all(c["raw_logits"]["shape"][1] == 157184 for c in oracle["calls"])
    assert all(c["return_observed"] for c in prism["calls"])
    if settings:
        assert [c["transfer_quota"] for c in oracle["calls"]] == [1, 1, 0, 0, 0]
        assert all("diagnostic-only" in c["selection_observation"] for c in prism["calls"][2:])
        assert prism["observed_usage"]["forward_starts"] == 5
        assert prism["observed_usage"]["processed_positions"] == 25


@pytest.mark.parametrize("winner", [157166, 157183])
def test_eoa_or_unknown_extra_model_position_preserves_oracle(tmp_path, generator, report, winner):
    bound = SyntheticBound(report, winner=winner)
    req = request(report)
    paths = [
        runner.run_path(bound, generator, report, req, p, tmp_path, p) for p in ("oracle", "prism")
    ]
    assert paths[0]["output_ids"] == paths[1]["output_ids"]
    if winner == 157166:
        assert all(p["finish_reason"] == "eoa" and p["output_ids"] == [] for p in paths)
        assert all(p["blocks"][0]["completed_block_ids"][0] == 157166 for p in paths)
    else:
        assert all(p["output_ids"][0] == 157183 for p in paths)
        assert all("no serialized tokenizer ID" in p["exception"] for p in paths)
        assert all(not p["observed_usage"]["complete"] for p in paths)
        assert all(p["calls"][0]["raw_logits"]["shape"][-1] == 157184 for p in paths)


@pytest.mark.parametrize("path", ["oracle", "prism"])
def test_forward_exception_retains_started_call_and_partial_blocks(
    tmp_path, generator, report, path
):
    bound = SyntheticBound(report, fail_call=3)
    result = runner.run_path(bound, generator, report, request(report), path, tmp_path, path)
    assert result["observed_usage"] == {
        "forward_starts": 3,
        "forward_returns": 2,
        "processed_positions": 12,
        "complete": False,
    }
    assert len(result["blocks"]) == 1
    assert result["calls"][-1]["start_observed"]
    assert not result["calls"][-1]["return_observed"]
    assert "synthetic forward failure" in result["calls"][-1]["exception"]
    saved = json.loads((tmp_path / f"{path}-trace.json").read_bytes())
    assert saved["calls"] == result["calls"]


def test_discrete_divergence_never_excused_by_ties(tmp_path, generator, report):
    oracle, prism, _ = run_pair(tmp_path, generator, report)
    prism["calls"][0]["selected_absolute_indices"] = [5, 6]
    result = runner.compare_paths(
        oracle, prism, tmp_path, json.loads(FIXTURE.read_bytes())["numeric_tolerances"]["float32"]
    )
    assert result["status"] == "fail" and "call 0: reveal set" in result["failures"]


def test_missing_observation_and_coverage_cannot_match(tmp_path, generator, report):
    oracle, prism, _ = run_pair(tmp_path, generator, report)
    del prism["calls"][0]["raw_logits"]
    result = runner.compare_paths(
        oracle, prism, tmp_path, json.loads(FIXTURE.read_bytes())["numeric_tolerances"]["float32"]
    )
    assert result["status"] == "incomplete"
    coverage = runner.observed_coverage([oracle], 3, 5)
    assert "length_stop" in coverage and "eoa_stop" not in coverage


def test_sidecar_tamper_fails(tmp_path):
    record = runner.tensor_record(tmp_path, "tensor.bin", torch.tensor([1.0, 2.0]))
    (tmp_path / "tensor.bin").write_bytes(b"0" * record["size_bytes"])
    with pytest.raises(runner.EvidenceError, match="integrity"):
        runner.read_tensor(tmp_path, record)


def test_numeric_threshold_is_per_scalar_not_average():
    result = runner.numeric_summary(
        torch.ones(100),
        torch.cat((torch.ones(99), torch.tensor([1.1]))),
        {"atol": 1e-5, "rtol": 1e-5},
    )
    assert result["violation_count"] == 1 and result["first_violation"] == [99]


def test_frozen_file_and_exact_launcher_flags(tmp_path, monkeypatch):
    inputs = {
        k: tmp_path / k
        for k in ("source-root", "checkpoint-root", "asset-manifest", "source-manifest")
    }
    for path in inputs.values():
        if path.name.endswith("root"):
            path.mkdir()
        else:
            path.write_text("{}")
    argv = [arg for k, v in inputs.items() for arg in ("--" + k, str(v))]
    argv += [
        "--output-dir",
        str(tmp_path / "out"),
        "--acceptance-file",
        str(FIXTURE),
        "--device",
        "xpu:0",
        "--dtype",
        "bfloat16",
    ]
    args = runner.parser().parse_args(argv)
    monkeypatch.setattr(runner, "_git", lambda *a: "" if a[0] == "status" else "synthetic")
    monkeypatch.setattr(
        runner,
        "load_checkpoint",
        lambda *a, **k: pytest.fail("no context must stop before loading"),
    )
    result = runner.execute(args)
    assert result["status"] == "incomplete" and not result["complete"]
    assert result["evidence_level"] != "independently-validated"
    assert "run context absent" in result["failure_reasons"][0]
    assert json.loads((args.output_dir / "result.json").read_bytes()) == result
    with pytest.raises(runner.EvidenceError, match="no overwrite"):
        runner.execute(args)


def test_observer_refuses_unavailable_instrumentation(tmp_path):
    with pytest.raises(runner.EvidenceError, match="anchor"):
        runner.Observations(
            tmp_path, "x", "oracle", test_observer_refuses_unavailable_instrumentation, 2
        )


def test_both_frozen_orders_reset_rng_and_missing_eoa_remains_missing(tmp_path, generator, report):
    from src.backends.u2_text_renderer import RenderedTextPrompt

    bound = SyntheticBound(report)
    bound.tokenizer.tokenizer_digest = report.tokenizer_digest
    bound.tokenizer.render = lambda **kwargs: RenderedTextPrompt(
        (41, 42), report.tokenizer_digest, "synthetic"
    )
    fixture = json.loads(FIXTURE.read_bytes())
    cases, missing = runner.run_cases(bound, generator, report, fixture, tmp_path)
    assert [c["id"] for c in cases] == [c["id"] for c in fixture["cases"]]
    assert "eoa_stop" in missing
    assert len(cases) == 6
    for case in cases:
        assert len(case["repetitions"]) == 2
        for repeat in case["repetitions"]:
            assert repeat["discrete_comparisons"]["status"] == "matched"
            assert repeat["oracle"]["rng"] == repeat["prism"]["rng"]


@pytest.mark.parametrize("path", ["oracle", "prism"])
def test_budget_preserves_partial_committed_output_and_actual_counts(
    tmp_path, generator, report, path
):
    bound = SyntheticBound(report)
    result = runner.run_path(
        bound, generator, report, request(report, max_model_calls=3), path, tmp_path, path
    )
    assert result["output_ids"] == [101, 103, 104]
    assert result["observed_usage"]["forward_starts"] == 3
    assert result["observed_usage"]["processed_positions"] == 20
    assert not result["observed_usage"]["complete"]


def test_invalid_mask_block_is_observed_but_never_claimed_committed(tmp_path, generator, report):
    bound = SyntheticBound(report, winner=156895)
    result = runner.run_path(bound, generator, report, request(report), "prism", tmp_path, "mask")
    assert result["finish_reason"] == "error"
    assert result["output_ids"] == []
    assert result["blocks"][0]["completed_block_ids"][0] == 156895
    assert result["blocks"][0]["committed_output_ids"] == []
    assert result["blocks"][0]["committed_block_ends"] == []


def test_trace_events_keep_context_and_actual_selection_values(tmp_path, generator, report):
    bound = SyntheticBound(report)
    result = runner.run_path(
        bound,
        generator,
        report,
        request(report),
        "oracle",
        tmp_path,
        "events",
        case_id="synthetic-case",
        repetition=1,
    )
    for event in result["calls"]:
        assert event["case_id"] == "synthetic-case" and event["repetition"] == 1
        assert event["path"] == "oracle"
        assert event["candidate_ids"] == event["observed_candidate_ids"]
        assert event["selection_observation"] == "actual sampler locals"


@pytest.mark.parametrize("observed", ["absent", "job_mismatch", "modules_mismatch"])
def test_declared_context_cannot_substitute_for_observed_job(tmp_path, monkeypatch, observed):
    context = {
        "acceptance_document_revision": "synthetic",
        "oracle_instrumentation_diff_and_review": "synthetic",
        "modules_in_order": ["synthetic/module"],
        "authorized_job_id": "synthetic-job",
        "execution_allocation": {"synthetic": True},
        "imported_source_manifest": "not-opened",
        "imported_source_manifest_sha256": "0" * 64,
    }
    context_path = tmp_path / "context.json"
    context_path.write_text(json.dumps(context))
    args = runner.parser().parse_args(
        [
            "--source-root",
            str(tmp_path / "source"),
            "--checkpoint-root",
            str(tmp_path / "nvidia"),
            "--output-dir",
            str(tmp_path / "output"),
            "--acceptance-file",
            str(FIXTURE),
            "--asset-manifest",
            str(tmp_path / "asset.json"),
            "--source-manifest",
            str(tmp_path / "six.json"),
            "--device",
            "xpu:0",
            "--dtype",
            "bfloat16",
            "--run-context",
            str(context_path),
        ]
    )
    monkeypatch.setattr(runner, "_git", lambda *a: "" if a[0] == "status" else "synthetic")
    monkeypatch.delenv("PBS_JOBID", raising=False)
    monkeypatch.delenv("LOADEDMODULES", raising=False)
    if observed != "absent":
        monkeypatch.setenv(
            "PBS_JOBID", "different" if observed == "job_mismatch" else "synthetic-job"
        )
        monkeypatch.setenv(
            "LOADEDMODULES",
            "different/module" if observed == "modules_mismatch" else "synthetic/module",
        )
    monkeypatch.setattr(
        runner, "verify_manifest", lambda *a, **k: pytest.fail("must fail before inputs")
    )
    result = runner.execute(args)
    if observed == "absent":
        assert result["status"] == "incomplete"
        assert result["allocation_observation"]["job_id"] is None
        assert "observed PBS" in result["failure_reasons"][0]
    else:
        assert result["status"] == "fail"
        assert "differ from admitted" in result["failure_reasons"][0]


def test_missing_selection_in_both_paths_remains_incomplete(tmp_path, generator, report):
    oracle, prism, _ = run_pair(tmp_path, generator, report)
    del oracle["calls"][0]["selected_absolute_indices"]
    del prism["calls"][0]["selected_absolute_indices"]
    comparison = runner.compare_paths(
        oracle, prism, tmp_path, json.loads(FIXTURE.read_bytes())["numeric_tolerances"]["float32"]
    )
    assert comparison["status"] == "incomplete"
    assert any("selected_absolute_indices" in reason for reason in comparison["missing"])


@pytest.mark.parametrize("path", ["oracle", "prism"])
def test_interruption_saves_partial_trace_and_journal_before_propagating(
    tmp_path, generator, report, path
):
    bound = SyntheticBound(report)

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt("synthetic interruption")

    bound.model.forward = interrupt
    with pytest.raises(KeyboardInterrupt):
        runner.run_path(bound, generator, report, request(report), path, tmp_path, path)
    trace = json.loads((tmp_path / f"{path}-trace.json").read_bytes())
    assert trace["observed_usage"]["forward_starts"] == 1
    assert trace["observed_usage"]["forward_returns"] == 0
    assert trace["observed_usage"]["processed_positions"] == 0
    assert not trace["observed_usage"]["complete"]
    events = [
        json.loads(line) for line in (tmp_path / f"{path}-events.jsonl").read_text().splitlines()
    ]
    assert [event["event"] for event in events] == ["forward_start", "forward_exception"]
