"""CPU-only behavior contracts; no model, scheduler or network execution."""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import socket
import subprocess
import sys
from pathlib import Path

import pytest
from tools import launch_aurora_u2_parity as launcher

pytestmark = [pytest.mark.unit, pytest.mark.launcher]


@pytest.fixture
def setup(tmp_path: Path) -> dict:
    root = tmp_path.resolve()
    paths = {
        "prism-root": root / "prism ' checkout",
        "source-root": root / "source ; $(touch INJECTION) `touch INJECTION2`",
        "checkpoint-root": root / "checkpoint with spaces",
        "output-dir": root / "output ' ; $value",
    }
    for key, path in paths.items():
        if key != "output-dir":
            path.mkdir()
    metadata = {}
    for name in ("acceptance-file", "asset-manifest", "source-manifest"):
        path = root / (name + " ' .json")
        path.write_text(json.dumps({"synthetic": True, "identity": name}) + "\n")
        metadata[name] = path
    args = []
    for key, path in {**paths, **metadata}.items():
        args.extend(["--" + key, str(path)])
    for key, path in metadata.items():
        args.extend(["--" + key + "-sha256", hashlib.sha256(path.read_bytes()).hexdigest()])
    plan_dir = root / "plan"
    args.extend(
        [
            "--prism-revision",
            "a" * 40,
            "--plan-dir",
            str(plan_dir),
            "--account",
            "test-account",
            "--queue",
            "debug",
            "--nodes",
            "1",
            "--walltime",
            "00:05:00",
            "--filesystems",
            "test:storage",
            "--device",
            "xpu:0",
            "--dtype",
            "bfloat16",
            "--python",
            str(Path(sys.executable).resolve()),
            "--module-use",
            str(root / "module ' search ; $HOME"),
            "--module",
            "compiler/1",
            "--module",
            "mpi/2",
            "--module",
            "framework/3",
        ]
    )
    return {"root": root, "paths": paths, "metadata": metadata, "args": args, "plan": plan_dir}


def change(args: list[str], flag: str, value: str) -> list[str]:
    result = list(args)
    result[result.index(flag) + 1] = value
    return result


def plan(setup: dict, extra: list[str] | None = None) -> dict:
    assert launcher.main(setup["args"] + (extra or [])) == 0
    return json.loads((setup["plan"] / "launch-plan.json").read_text())


def fake_environment(
    setup: dict, *, code: int = 0, module_failure: bool = False, mpi_failure: bool = False
) -> dict[str, str]:
    """Prepare a synthetic runner, allocation and MPI stub; never use real MPI."""
    runner = setup["paths"]["prism-root"] / "tools/parity/u2_text.py"
    runner.parent.mkdir(parents=True)
    runner.write_text(
        "import json, os, pathlib, sys\n"
        "args = dict(zip(sys.argv[1::2], sys.argv[2::2]))\n"
        "out = pathlib.Path(args['--output-dir'])\n"
        "(out / 'argv.json').write_text(json.dumps(sys.argv[1:]))\n"
        "assert os.environ['HF_HUB_OFFLINE'] == '1'\n"
        "assert os.environ['TRANSFORMERS_OFFLINE'] == '1'\n"
        "assert not any(os.environ.get(k) for k in "
        "('HF_TOKEN', 'WANDB_API_KEY', 'PAPERCLIP_API_KEY', 'OPENAI_API_KEY'))\n"
        "print('synthetic stdout')\n"
        "print('synthetic stderr', file=sys.stderr)\n"
        "(out / 'result.json').write_text(json.dumps({'schema_version': 1, "
        "'complete': True, 'status': 'pass', 'evidence_level': 'synthetic'}))\n"
        f"raise SystemExit({code})\n"
    )
    startup = setup["root"] / "fixture-shell-init"
    order = setup["root"] / "module-order"
    startup.write_text(
        "module() {\n"
        + "    printf '%s\\n' \"$*\" >> "
        + shlex.quote(str(order))
        + "\n"
        + ("    return 23\n" if module_failure else "    return 0\n")
        + "}\n"
        + "mpiexec() {\n"
        + "    printf '%s\\0' \"$@\" > "
        + shlex.quote(str(setup["root"] / "mpi-argv"))
        + "\n"
        + '    [[ "$1" == -n && "$2" == 1 && "$3" == --ppn && "$4" == 1 ]] || return 99\n'
        + ("    return 42\n" if mpi_failure else "")
        + "    shift 4\n"
        + '    "$@"\n'
        + "}\n"
    )
    nodefile = setup["root"] / "fake-pbs-nodes"
    nodefile.write_text("synthetic-compute-node\n")
    env = dict(os.environ)
    env.update(
        {
            "BASH_ENV": str(startup),
            "HF_TOKEN": "synthetic-secret-hf",
            "WANDB_API_KEY": "synthetic-secret-wandb",
            "PAPERCLIP_API_KEY": "synthetic-secret-pc",
            "OPENAI_API_KEY": "synthetic-secret-api",
            "PBS_JOBID": "synthetic-allocation",
            "PBS_NODEFILE": str(nodefile),
        }
    )
    return env


def fake_run(
    setup: dict, *, code: int = 0, module_failure: bool = False
) -> subprocess.CompletedProcess:
    """Execute only a synthetic stdlib fixture through the generated shell script."""
    return subprocess.run(
        ["bash", str(setup["plan"] / "job.sh")],
        cwd=setup["root"],
        env=fake_environment(setup, code=code, module_failure=module_failure),
        capture_output=True,
        text=True,
        check=False,
        timeout=15,
    )


def test_dry_run_is_default_offline_and_does_not_need_runner(setup, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("generation attempted execution or network access")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(os, "system", forbidden)
    monkeypatch.setattr(socket, "socket", forbidden)
    monkeypatch.setenv("HF_TOKEN", "synthetic-not-for-plan")
    result = plan(setup)
    assert result["mode"] == "dry_run" and result["submitted"] is False
    assert result["job_id"] is None and result["execution_qualified"] is False
    assert result["runtime_existence"] == "unverified"
    assert result["runtime_path_safety"] == "unverified_until_execution_site_checks"
    assert result["prism_revision"]["verification"] == "declared_unverified"
    assert not setup["paths"]["output-dir"].exists()
    assert sorted(p.name for p in setup["plan"].iterdir()) == ["job.sh", "launch-plan.json"]
    script = (setup["plan"] / "job.sh").read_bytes()
    assert result["script"]["sha256"] == hashlib.sha256(script).hexdigest()
    assert b"synthetic-not-for-plan" not in script
    assert "synthetic-not-for-plan" not in json.dumps(result)
    for forbidden_word in (b"qsub ", b"ssh ", b"pip install", b"src.train", b"torchrun"):
        assert forbidden_word not in script
    assert result["argv"][1].endswith("/tools/parity/u2_text.py")
    assert (setup["plan"].stat().st_mode & 0o777) == 0o700
    assert ((setup["plan"] / "job.sh").stat().st_mode & 0o777) == 0o600


def test_declared_remote_inputs_and_local_hash_checks_are_distinct(setup):
    args = change(setup["args"], "--source-root", "/unavailable/source")
    assert launcher.main(args + ["--dry-run"]) == 0
    p = json.loads((setup["plan"] / "launch-plan.json").read_text())
    assert p["identities"]["source_manifest"]["verification"] == "declared_remote_unverified"


def test_validate_local_inputs_binds_exact_manifest_bytes(setup):
    result = plan(setup, ["--validate-local-inputs"])
    for value in result["identities"].values():
        assert value["verification"] == "local_file_hash_verified"
        assert value["sha256"] == hashlib.sha256(Path(value["path"]).read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--account", "bad\n#PBS -V"),
        ("--queue", "debug\n#PBS -V"),
        ("--filesystems", "a,b"),
        ("--filesystems", "test\n#PBS -V"),
        ("--account", "bad;cmd"),
        ("--nodes", "0"),
        ("--nodes", "3"),
        ("--walltime", "00:04:59"),
        ("--walltime", "01:00:01"),
        ("--walltime", "00:60:00"),
        ("--walltime", "5:00"),
        ("--device", "xpu"),
        ("--device", "xpu:-1"),
        ("--device", "cuda:0"),
        ("--module", "-bad"),
        ("--module", "x/../y"),
        ("--module", "x;cmd"),
        ("--module-use", "/tmp/../bad"),
        ("--source-root", "relative"),
        ("--source-root", "/a/../b"),
        ("--source-root", "/a\nb"),
        ("--output-dir", "/"),
        ("--source-root", "//a/b"),
        ("--acceptance-file-sha256", "f" * 63),
        ("--prism-revision", "main"),
    ],
)
def test_invalid_inputs_fail_before_writes(setup, flag, value):
    with pytest.raises(SystemExit) as exc:
        launcher.main(change(setup["args"], flag, value))
    assert exc.value.code == 2
    assert not setup["plan"].exists()


@pytest.mark.parametrize(
    "flag",
    [
        "--account",
        "--nodes",
        "--walltime",
        "--checkpoint-root",
        "--acceptance-file",
        "--asset-manifest",
        "--source-manifest",
        "--module-use",
        "--device",
        "--python",
    ],
)
def test_missing_required_argument_is_rejected(setup, flag):
    args = list(setup["args"])
    i = args.index(flag)
    del args[i : i + 2]
    with pytest.raises(SystemExit):
        launcher.main(args)
    assert not setup["plan"].exists()


@pytest.mark.parametrize("flag", ["--submit", "--execute", "--run-via-ssh", "--no-dry-run"])
def test_execution_modes_are_not_available(setup, flag):
    with pytest.raises(SystemExit):
        launcher.main(setup["args"] + [flag])
    assert not setup["plan"].exists()


def test_other_queues_require_explicit_bounds(setup):
    args = change(setup["args"], "--queue", "other")
    with pytest.raises(SystemExit):
        launcher.main(args)
    assert (
        launcher.main(
            args + ["--max-nodes", "1", "--min-walltime", "00:01:00", "--max-walltime", "00:10:00"]
        )
        == 0
    )


@pytest.mark.parametrize(
    "extra",
    [
        ["--max-nodes", "4"],
        ["--min-walltime", "00:00:01"],
        ["--max-walltime", "02:00:00"],
    ],
)
def test_debug_bounds_cannot_be_overridden(setup, extra):
    with pytest.raises(SystemExit):
        launcher.main(setup["args"] + extra)


@pytest.mark.parametrize("nodes,walltime", [("1", "00:05:00"), ("2", "01:00:00")])
def test_debug_endpoints_are_allowed(setup, nodes, walltime):
    args = change(change(setup["args"], "--nodes", nodes), "--walltime", walltime)
    assert launcher.main(args) == 0


@pytest.mark.parametrize("failure", ["missing-dir", "missing-file", "changed-hash", "non-json"])
def test_local_validation_failures(setup, failure):
    if failure == "missing-dir":
        setup["paths"]["checkpoint-root"].rmdir()
    elif failure == "missing-file":
        setup["metadata"]["asset-manifest"].unlink()
    elif failure == "changed-hash":
        setup["metadata"]["source-manifest"].write_text('{"changed": true}')
    else:
        setup["metadata"]["acceptance-file"].write_text("invalid JSON")
    with pytest.raises(SystemExit):
        launcher.main(setup["args"] + ["--validate-local-inputs"])
    assert not setup["plan"].exists()


def test_output_cannot_overlap_inputs_or_plan(setup):
    for path in (setup["paths"]["source-root"] / "outputs", setup["root"]):
        with pytest.raises(SystemExit):
            launcher.main(change(setup["args"], "--output-dir", str(path)))
    with pytest.raises(SystemExit):
        launcher.main(change(setup["args"], "--plan-dir", str(setup["paths"]["output-dir"])))


def test_existing_plan_is_never_overwritten(setup):
    plan(setup)
    old = (setup["plan"] / "launch-plan.json").read_bytes()
    with pytest.raises(SystemExit):
        launcher.main(setup["args"])
    assert (setup["plan"] / "launch-plan.json").read_bytes() == old


@pytest.mark.parametrize("code", [0, 7])
@pytest.mark.parametrize("nodes", ["1", "2"])
def test_generated_script_preserves_argv_modules_streams_and_exit_status(setup, code, nodes):
    setup["args"] = change(change(setup["args"], "--nodes", nodes), "--device", "xpu:3")
    p = plan(setup)
    result = fake_run(setup, code=code)
    assert result.returncode == code, result.stderr
    root = setup["paths"]["output-dir"]
    assert (root / "stdout.log").read_text() == "synthetic stdout\n"
    assert (root / "stderr.log").read_text() == "synthetic stderr\n"
    assert json.loads((root / "exit-status.json").read_text()) == {
        "schema_version": 1,
        "exit_code": code,
        "launch_id": p["launch_id"],
    }
    assert json.loads((root / "argv.json").read_text()) == p["argv"][2:]
    mpi_argv = (setup["root"] / "mpi-argv").read_bytes().decode().split("\0")[:-1]
    assert mpi_argv == ["-n", "1", "--ppn", "1", *p["argv"]]
    assert p["execution_argv"] == ["mpiexec", *mpi_argv]
    assert (setup["root"] / "module-order").read_text().splitlines() == [
        "purge",
        "use " + p["module_setup"]["use"][0],
        "load compiler/1",
        "load mpi/2",
        "load framework/3",
    ]
    assert not (setup["root"] / "INJECTION").exists()
    assert not (setup["root"] / "INJECTION2").exists()
    # A second execution cannot overwrite evidence from the first.
    before = (root / "exit-status.json").read_bytes()
    rerun = subprocess.run(
        ["bash", str(setup["plan"] / "job.sh")],
        env={
            **os.environ,
            "PBS_JOBID": "synthetic-allocation",
            "PBS_NODEFILE": str(setup["root"] / "fake-pbs-nodes"),
            "BASH_ENV": str(setup["root"] / "fixture-shell-init"),
        },
        capture_output=True,
        timeout=10,
    )
    assert rerun.returncode != 0
    assert (root / "exit-status.json").read_bytes() == before


@pytest.mark.parametrize(
    "invalid",
    [
        "job-id",
        "empty-job-id",
        "nodefile-env",
        "missing-nodefile",
        "empty-nodefile",
        "nodefile-directory",
    ],
)
def test_missing_allocation_refuses_execution_before_any_output(setup, invalid):
    plan(setup)
    env = fake_environment(setup)
    if invalid == "job-id":
        env.pop("PBS_JOBID")
    elif invalid == "empty-job-id":
        env["PBS_JOBID"] = ""
    elif invalid == "nodefile-env":
        env.pop("PBS_NODEFILE")
    elif invalid == "missing-nodefile":
        Path(env["PBS_NODEFILE"]).unlink()
    elif invalid == "nodefile-directory":
        env["PBS_NODEFILE"] = str(setup["root"])
    else:
        Path(env["PBS_NODEFILE"]).write_text("")
    result = subprocess.run(
        ["bash", str(setup["plan"] / "job.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "PBS allocation" in result.stderr
    assert not setup["paths"]["output-dir"].exists()
    assert not (setup["root"] / "mpi-argv").exists()
    assert not (setup["root"] / "module-order").exists()


@pytest.mark.parametrize("target", ["checkpoint-root", "source-root", "prism-root"])
@pytest.mark.parametrize("depth", [1, 2])
def test_output_symlink_ancestor_into_input_is_rejected_without_writes(setup, target, depth):
    env = fake_environment(setup)
    input_root = setup["paths"][target]
    if depth == 2:
        (input_root / "nested").mkdir()
    (input_root / "input-sentinel").write_bytes(b"unchanged synthetic input")
    before = {
        p.relative_to(input_root): p.read_bytes() for p in input_root.rglob("*") if p.is_file()
    }
    alias = setup["root"] / "alias-output-parent"
    output = alias / "results" if depth == 1 else alias / "nested/results"
    setup["args"] = change(setup["args"], "--output-dir", str(output))
    plan(setup)  # Remote safety cannot be established during plan generation.
    alias.symlink_to(input_root, target_is_directory=True)
    result = subprocess.run(
        ["bash", str(setup["plan"] / "job.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "symlink" in result.stderr
    assert not output.exists()
    assert not (input_root / "results").exists()
    after = {
        p.relative_to(input_root): p.read_bytes() for p in input_root.rglob("*") if p.is_file()
    }
    assert after == before
    assert not (setup["root"] / "module-order").exists()
    assert not (setup["root"] / "mpi-argv").exists()


@pytest.mark.parametrize(
    "input_name", [name for name in launcher.PATH_FLAGS if name != "output_dir"]
)
def test_symlinked_input_path_is_rejected_before_output_creation(setup, input_name):
    env = fake_environment(setup)
    flag = input_name.replace("_", "-")
    target = {**setup["paths"], **setup["metadata"]}[flag]
    alias = setup["root"] / "aliased-input"
    alias.symlink_to(target, target_is_directory=target.is_dir())
    setup["args"] = change(setup["args"], "--" + flag, str(alias))
    plan(setup)
    result = subprocess.run(
        ["bash", str(setup["plan"] / "job.sh")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert "symlink" in result.stderr
    assert not setup["paths"]["output-dir"].exists()
    assert not (setup["root"] / "module-order").exists()
    assert not (setup["root"] / "mpi-argv").exists()


def test_mpi_failure_keeps_exit_receipt_without_running_comparison(setup):
    p = plan(setup)
    result = subprocess.run(
        ["bash", str(setup["plan"] / "job.sh")],
        env=fake_environment(setup, mpi_failure=True),
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 42
    root = setup["paths"]["output-dir"]
    assert json.loads((root / "exit-status.json").read_text()) == {
        "schema_version": 1,
        "exit_code": 42,
        "launch_id": p["launch_id"],
    }
    assert not (root / "result.json").exists()
    assert not (root / "argv.json").exists()
    assert (root / "stdout.log").is_file()
    assert (root / "stderr.log").is_file()


def test_module_failure_has_exit_receipt_and_no_result(setup):
    plan(setup)
    result = fake_run(setup, module_failure=True)
    root = setup["paths"]["output-dir"]
    assert result.returncode == 23
    assert json.loads((root / "exit-status.json").read_text())["exit_code"] == 23
    assert not (root / "result.json").exists()


def test_changed_identity_file_fails_before_runner(setup):
    plan(setup)
    setup["metadata"]["asset-manifest"].write_text("{}\n")
    result = fake_run(setup)
    root = setup["paths"]["output-dir"]
    assert result.returncode != 0
    assert "SHA256 mismatch" in (root / "stderr.log").read_text()
    assert not (root / "result.json").exists()


def test_missing_runner_cannot_silently_succeed(setup):
    plan(setup)
    env = fake_environment(setup)
    (setup["paths"]["prism-root"] / "tools/parity/u2_text.py").unlink()
    result = subprocess.run(
        ["bash", str(setup["plan"] / "job.sh")],
        env=env,
        timeout=10,
    )
    root = setup["paths"]["output-dir"]
    assert result.returncode != 0
    assert "runner unavailable" in (root / "stderr.log").read_text()
    assert json.loads((root / "exit-status.json").read_text())["exit_code"] != 0


def collect_fixture(setup: dict, includes: list[str] | None = None) -> dict:
    return launcher.collect(
        setup["paths"]["output-dir"],
        setup["root"] / "collected",
        list(launcher.EVIDENCE_FILES) if includes is None else includes,
        setup["plan"] / "launch-plan.json",
    )


def test_collection_omits_all_private_content_and_hashes_actual_export_bytes(setup):
    p = plan(setup)
    fake_run(setup)
    root = setup["paths"]["output-dir"]
    secret = "synthetic-secret-arbitrary-string"
    (root / "stdout.log").write_text(secret + str(setup["root"]))
    (root / "stderr.log").write_text("Authorization: Bearer " + secret)
    (root / "result.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "complete": True,
                "status": "pass",
                "prompt": secret,
                "identities": {"path": str(setup["root"])},
            }
        )
    )
    (root / "model.safetensors").write_bytes(b"synthetic-do-not-copy")
    originals = {name: (root / name).read_bytes() for name in launcher.EVIDENCE_FILES}
    manifest = collect_fixture(setup)
    dest = setup["root"] / "collected"
    assert manifest["transport_complete"] is True
    assert manifest["parity_accepted"] is False
    assert manifest["launch_id"] == p["launch_id"]
    assert (
        manifest["launch_plan_sha256"]
        == hashlib.sha256((setup["plan"] / "launch-plan.json").read_bytes()).hexdigest()
    )
    assert [e["path"] for e in manifest["files"]] == sorted(launcher.EVIDENCE_FILES)
    for entry in manifest["files"]:
        data = (dest / entry["path"]).read_bytes()
        assert entry["sha256"] == hashlib.sha256(data).hexdigest()
        assert entry["bytes"] == len(data)
        assert entry["source_sha256"] == hashlib.sha256(originals[entry["path"]]).hexdigest()
        assert entry["source_bytes"] == len(originals[entry["path"]])
    for path in dest.iterdir():
        assert secret not in path.read_text()
        assert str(setup["root"]) not in path.read_text()
    assert not (dest / "model.safetensors").exists()
    public_result = json.loads((dest / "result.json").read_text())
    assert public_result["complete"] is False and public_result["status"] == "incomplete"
    assert public_result["reported_status"] == "pass"
    assert all((root / name).read_bytes() == data for name, data in originals.items())


@pytest.mark.parametrize("missing", ["exit-status.json", "result.json", "stdout.log"])
def test_missing_evidence_stays_incomplete(setup, missing):
    plan(setup)
    fake_run(setup)
    (setup["paths"]["output-dir"] / missing).unlink()
    manifest = collect_fixture(setup)
    assert manifest["transport_complete"] is False
    assert manifest["missing"] == [missing]
    assert not (setup["root"] / "collected" / missing).exists()


@pytest.mark.parametrize(
    "includes",
    [["../secret"], ["/etc/passwd"], ["stdout.log", "stdout.log"], ["tokenizer.json"], []],
)
def test_collection_rejects_non_allowlist_paths_and_collisions(setup, includes):
    plan(setup)
    fake_run(setup)
    with pytest.raises(ValueError):
        collect_fixture(setup, includes)
    assert not (setup["root"] / "collected").exists()


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "too-large", "wrong-launch"])
def test_unsafe_collection_inputs_fail_before_writes(setup, kind):
    plan(setup)
    fake_run(setup)
    root = setup["paths"]["output-dir"]
    path = root / "stdout.log"
    path.unlink()
    other = setup["root"] / "private"
    other.write_text("synthetic private input")
    if kind == "symlink":
        path.symlink_to(other)
    elif kind == "hardlink":
        os.link(other, path)
    elif kind == "fifo":
        os.mkfifo(path)
    elif kind == "too-large":
        with path.open("wb") as stream:
            stream.truncate(launcher.MAX_FILE_BYTES + 1)
    else:
        path.write_text("")
        (root / "exit-status.json").write_text(
            json.dumps({"schema_version": 1, "exit_code": 0, "launch_id": "0" * 64})
        )
    with pytest.raises(ValueError):
        collect_fixture(setup)
    assert not (setup["root"] / "collected").exists()


def test_collection_destination_symlink_overlap_and_collision_are_rejected(setup):
    plan(setup)
    fake_run(setup)
    root = setup["paths"]["output-dir"]
    dest = setup["root"] / "collected"
    dest.symlink_to(root, target_is_directory=True)
    with pytest.raises(ValueError):
        collect_fixture(setup)
    dest.unlink()
    with pytest.raises(ValueError):
        launcher.collect(
            root, root / "nested", list(launcher.EVIDENCE_FILES), setup["plan"] / "launch-plan.json"
        )
    collect_fixture(setup)
    before = (dest / "artifact-manifest.json").read_bytes()
    with pytest.raises(FileExistsError):
        collect_fixture(setup)
    assert (dest / "artifact-manifest.json").read_bytes() == before


def test_collect_cli_returns_incomplete_without_fabricating_result(setup):
    plan(setup)
    fake_run(setup, module_failure=True)
    args = [
        "collect",
        "--output-dir",
        str(setup["paths"]["output-dir"]),
        "--collection-dir",
        str(setup["root"] / "collected"),
        "--launch-plan",
        str(setup["plan"] / "launch-plan.json"),
    ]
    for name in launcher.EVIDENCE_FILES:
        args.extend(["--include", name])
    assert launcher.main(args) == 2
    assert not (setup["root"] / "collected/result.json").exists()


def test_changed_launch_plan_is_not_attributed_to_old_receipt(setup):
    plan(setup)
    fake_run(setup)
    path = setup["plan"] / "launch-plan.json"
    changed = json.loads(path.read_text())
    changed["identities"]["asset_manifest"]["sha256"] = "0" * 64
    path.write_text(json.dumps(changed))
    with pytest.raises(ValueError, match="does not match"):
        collect_fixture(setup)
    assert not (setup["root"] / "collected").exists()


@pytest.mark.parametrize("code", [True, -1, 256, "0"])
def test_malformed_exit_codes_are_rejected(setup, code):
    p = plan(setup)
    fake_run(setup)
    path = setup["paths"]["output-dir"] / "exit-status.json"
    path.write_text(
        json.dumps({"schema_version": 1, "exit_code": code, "launch_id": p["launch_id"]})
    )
    with pytest.raises(ValueError, match="exit receipt"):
        collect_fixture(setup)
    assert not (setup["root"] / "collected").exists()


@pytest.mark.parametrize("payload", ["[]", "{}", '{"value": NaN}', "not JSON"])
def test_malformed_results_cannot_be_transported_as_success(setup, payload):
    plan(setup)
    fake_run(setup)
    (setup["paths"]["output-dir"] / "result.json").write_text(payload)
    with pytest.raises(ValueError):
        collect_fixture(setup)
    assert not (setup["root"] / "collected").exists()


def test_nonzero_exit_can_have_complete_failure_evidence_without_acceptance(setup):
    plan(setup)
    assert fake_run(setup, code=7).returncode == 7
    manifest = collect_fixture(setup)
    assert manifest["transport_complete"] is True
    assert manifest["parity_accepted"] is False
    receipt = json.loads((setup["root"] / "collected/exit-status.json").read_text())
    assert receipt["exit_code"] == 7


def test_local_input_symlink_is_rejected(setup):
    path = setup["metadata"]["source-manifest"]
    old = path.read_bytes()
    path.unlink()
    other = setup["root"] / "manifest-target"
    other.write_bytes(old)
    path.symlink_to(other)
    with pytest.raises(SystemExit):
        launcher.main(setup["args"] + ["--validate-local-inputs"])
    assert not setup["plan"].exists()
