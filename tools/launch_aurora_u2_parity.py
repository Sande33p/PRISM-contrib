#!/usr/bin/env python3
"""Plan an offline U2 comparison and collect bounded, sanitized evidence.

This standard-library-only tool never submits or executes a job. The generated
script targets Integration's deferred tools/parity/u2_text.py entry point.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import stat
import sys
from pathlib import Path, PurePosixPath
from typing import Any

SOURCE_REVISION = "70f21c1df8fa26e514df26ba31bc061be8d3d018"
MAX_FILE_BYTES = 8 * 1024 * 1024
EVIDENCE_FILES = ("stdout.log", "stderr.log", "exit-status.json", "result.json")
IDENTITY_FLAGS = ("acceptance_file", "asset_manifest", "source_manifest")
PATH_FLAGS = ("prism_root", "source_root", "checkpoint_root", "output_dir", *IDENTITY_FLAGS)
RUNNER_FLAGS = (*PATH_FLAGS[1:], "device", "dtype")


def _json_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _match(value: str, pattern: str, label: str) -> str:
    if not re.fullmatch(pattern, value):
        raise ValueError(f"invalid {label}")
    return value


def _absolute(value: str) -> str:
    """Validate a literal POSIX path without expanding variables or resolving it."""
    if (
        not value.startswith("/")
        or value.startswith("//")
        or any(ord(c) < 32 or ord(c) == 127 for c in value)
        or any(p in {".", "..", ""} for p in value[1:].split("/"))
    ):
        raise ValueError(
            "paths must be absolute, non-root, and without traversal/control characters"
        )
    return value


def _seconds(value: str) -> int:
    _match(value, r"[0-9]{2}:[0-5][0-9]:[0-5][0-9]", "walltime (HH:MM:SS)")
    h, m, s = map(int, value.split(":"))
    return h * 3600 + m * 60 + s


def _overlap(a: str | Path, b: str | Path) -> bool:
    left, right = PurePosixPath(a), PurePosixPath(b)
    return left == right or left in right.parents or right in left.parents


def _read_file(path: Path) -> bytes:
    """Read one bounded regular file, rejecting symlinks, hardlinks and devices."""
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise ValueError("evidence/input must be a regular file with one link")
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        current = os.fstat(stream.fileno())
        if (info.st_dev, info.st_ino) != (current.st_dev, current.st_ino):
            raise ValueError("input changed while opening")
        data = stream.read(MAX_FILE_BYTES + 1)
    # Empty logs are valid; empty JSON is rejected by its parser.
    if len(data) > MAX_FILE_BYTES:
        raise ValueError("input exceeds the 8 MiB per-file bound")
    return data


def _object(data: bytes) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError("non-finite JSON is unsupported")

    result = json.loads(data, parse_constant=reject_constant)
    if not isinstance(result, dict):
        raise ValueError("expected a JSON object")
    return result


def _local_path(value: str) -> Path:
    """Reject symlinks in the supplied path, including ancestor components."""
    path = Path(_absolute(value))
    for component in (path, *path.parents):
        if component.is_symlink():
            raise ValueError("local paths must not contain symlinks; use the canonical path")
    return path


def _write_bundle(destination: Path, files: dict[str, bytes]) -> None:
    """Write only into a new private directory; never replace previous evidence."""
    destination = _local_path(str(destination))
    destination.mkdir(mode=0o700)  # The caller must provision its parent explicitly.
    for name, data in files.items():
        with (destination / name).open("xb") as stream:
            os.chmod(stream.fileno(), 0o600)
            stream.write(data)


def _generation_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    for name in PATH_FLAGS:
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    for name in IDENTITY_FLAGS:
        parser.add_argument("--" + name.replace("_", "-") + "-sha256", required=True)
    parser.add_argument("--prism-revision", required=True, help="declared exact PRISM commit")
    parser.add_argument("--plan-dir", required=True, help="new local directory for job.sh and plan")
    parser.add_argument("--account", required=True)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--walltime", required=True)
    parser.add_argument("--filesystems", required=True)
    parser.add_argument("--device", required=True, help="explicit indexed xpu:N")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), required=True)
    parser.add_argument("--python", required=True, help="absolute site Python executable")
    parser.add_argument(
        "--mpiexec", required=True, help="absolute site mpiexec; module purge drops PATH's"
    )
    parser.add_argument(
        "--module-use", action="append", required=True, help="module search directory"
    )
    parser.add_argument("--module", action="append", required=True, help="module, in load order")
    parser.add_argument("--max-nodes", type=int, help="required bound for non-debug queues")
    parser.add_argument("--min-walltime", help="required bound for non-debug queues")
    parser.add_argument("--max-walltime", help="required bound for non-debug queues")
    parser.add_argument("--validate-local-inputs", action="store_true")
    parser.add_argument(
        "--run-context", help="optional private runner evidence path; does not authorize execution"
    )
    parser.add_argument("--dry-run", action="store_true", default=True)
    return parser


def _configuration(args: argparse.Namespace) -> dict[str, Any]:
    paths = {name: _absolute(getattr(args, name)) for name in PATH_FLAGS}
    if args.run_context is not None:
        paths["run_context"] = _absolute(args.run_context)
    for name in paths:
        if name != "output_dir" and _overlap(paths["output_dir"], paths[name]):
            raise ValueError("runtime output must be disjoint from every input")
    _absolute(args.python)
    mpiexec = _absolute(args.mpiexec)
    _match(args.prism_revision, r"[0-9a-f]{40}", "PRISM revision")
    _match(args.device, r"xpu:(0|[1-9][0-9]*)", "indexed XPU device")
    for name in ("account", "queue"):
        _match(getattr(args, name), r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", name)
    _match(args.filesystems, r"[A-Za-z0-9_-]+(:[A-Za-z0-9_-]+)*", "filesystems")
    modules = [_match(m, r"[A-Za-z0-9][A-Za-z0-9_.+/-]{0,199}", "module") for m in args.module]
    if any(p in {"", ".", ".."} for m in modules for p in m.split("/")):
        raise ValueError("invalid module path")
    module_use = [_absolute(p) for p in args.module_use]
    seconds = _seconds(args.walltime)
    if args.queue == "debug":
        max_nodes, minimum, maximum = 2, 300, 3600
        if any(v is not None for v in (args.max_nodes, args.min_walltime, args.max_walltime)):
            raise ValueError("debug bounds are fixed and cannot be overridden")
    else:
        if any(v is None for v in (args.max_nodes, args.min_walltime, args.max_walltime)):
            raise ValueError("other queues require max-nodes, min-walltime and max-walltime")
        max_nodes = args.max_nodes
        minimum, maximum = _seconds(args.min_walltime), _seconds(args.max_walltime)
    if not (1 <= args.nodes <= max_nodes and 0 < minimum <= seconds <= maximum):
        raise ValueError("resource request exceeds declared bounds")
    identities = {}
    for name in IDENTITY_FLAGS:
        digest = _match(getattr(args, name + "_sha256"), r"[0-9a-f]{64}", name + " SHA256")
        state = "declared_remote_unverified"
        if args.validate_local_inputs:
            data = _read_file(_local_path(paths[name]))
            _object(data)
            if _sha(data) != digest:
                raise ValueError(name + " SHA256 mismatch")
            state = "local_file_hash_verified"
        identities[name] = {"path": paths[name], "sha256": digest, "verification": state}
    if args.validate_local_inputs:
        for name in ("prism_root", "source_root", "checkpoint_root"):
            if not _local_path(paths[name]).is_dir():
                raise ValueError(name + " must be an existing directory")
    context_identity = None
    if args.run_context is not None:
        context_identity = {
            "path": paths["run_context"],
            "verification": "declared_remote_unverified",
        }
        if args.validate_local_inputs:
            raw = _read_file(_local_path(paths["run_context"]))
            _object(raw)
            context_identity.update(sha256=_sha(raw), verification="local_file_hash_verified")
    argv = [args.python, paths["prism_root"] + "/tools/parity/u2_text.py"]
    for name in RUNNER_FLAGS:
        argv.extend(["--" + name.replace("_", "-"), getattr(args, name)])
    if args.run_context is not None:
        argv.extend(["--run-context", paths["run_context"]])
    return {
        "schema_version": 1,
        "mode": "dry_run",
        "submitted": False,
        "job_id": None,
        "resources": {
            "account": args.account,
            "queue": args.queue,
            "nodes": args.nodes,
            "walltime": args.walltime,
            "walltime_seconds": seconds,
            "filesystems": args.filesystems,
            "bounds": {"max_nodes": max_nodes, "min_seconds": minimum, "max_seconds": maximum},
            "allocation_authorized": False,
            "execution_processes": 1,
        },
        "module_setup": {"purge": True, "use": module_use, "load": modules},
        "argv": argv,
        "execution_argv": [mpiexec, "-n", "1", "--ppn", "1", *argv],
        "cwd": paths["prism_root"],
        "paths": paths,
        "identities": identities,
        "private_run_context": context_identity,
        "prism_revision": {"value": args.prism_revision, "verification": "declared_unverified"},
        "oracle_revision": SOURCE_REVISION,
        "runtime_existence": "unverified",
        "runtime_path_safety": "unverified_until_execution_site_checks",
        "execution_qualified": False,
    }


def _script(plan: dict[str, Any]) -> bytes:
    resources, paths = plan["resources"], plan["paths"]
    # PBS directives are not shell syntax. All variable directive values have
    # strict grammars; log directive names are fixed, never shell-quoted paths.
    lines = [
        "#!/bin/bash",
        "# Generated plan only. Execution needs separate operator admission.",
        "#PBS -N u2-parity",
        f"#PBS -A {resources['account']}",
        f"#PBS -q {resources['queue']}",
        f"#PBS -l select={resources['nodes']}",
        f"#PBS -l walltime={resources['walltime']}",
        f"#PBS -l filesystems={resources['filesystems']}",
        "#PBS -o u2-pbs.stdout.log",
        "#PBS -e u2-pbs.stderr.log",
        "set -euo pipefail",
        "umask 077",
        ': "${PBS_JOBID:?PBS allocation required}"',
        ': "${PBS_NODEFILE:?PBS allocation node file required}"',
        'if [[ ! -f "$PBS_NODEFILE" || ! -r "$PBS_NODEFILE" || ! -s "$PBS_NODEFILE" ]]; then',
        "    printf '%s\\n' 'PBS allocation node file must be readable and nonempty' >&2",
        "    exit 1",
        "fi",
        # Paths were validated lexically at generation. Reject every symlink
        # component on the execution filesystem before mkdir, redirects or
        # module setup. Parents must remain operator-controlled and stable.
        "check_path() {",
        "    local component=$1",
        '    while [[ -n "$component" ]]; do',
        '        if [[ -L "$component" ]]; then',
        "            printf '%s\\n' 'runtime paths must not contain symlinks' >&2",
        "            return 1",
        "        fi",
        "        component=${component%/*}",
        "    done",
        "}",
    ]
    lines.extend(shlex.join(["check_path", paths[name]]) for name in paths)
    lines.extend(
        [
            "out=" + shlex.quote(paths["output_dir"]),
            # A new output directory prevents stale results and stream clobbering.
            'mkdir -- "$out"',
            "finish() {",
            "    rc=$?",
            "    trap - EXIT",
            '    printf \'{"schema_version":1,"exit_code":%d,"launch_id":"'
            + plan["launch_id"]
            + '"}\\n\' "$rc" > "$out/exit-status.json"',
            '    exit "$rc"',
            "}",
            "trap finish EXIT",
            "trap 'exit 130' INT",
            "trap 'exit 143' TERM",
            'exec > "$out/stdout.log" 2> "$out/stderr.log"',
            "unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACEHUB_API_TOKEN WANDB_API_KEY",
            "unset OPENAI_API_KEY PAPERCLIP_API_KEY",
            "export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 WANDB_MODE=disabled",
            "export PYTHONDONTWRITEBYTECODE=1 PYTHONNOUSERSITE=1",
            # Lmod and site modulefiles may read unset variables, which nounset
            # makes fatal. Suspend it for module setup only; errexit stays on.
            "set +u",
            "module purge",
        ]
    )
    lines.extend(shlex.join(["module", "use", p]) for p in plan["module_setup"]["use"])
    lines.extend(shlex.join(["module", "load", m]) for m in plan["module_setup"]["load"])
    lines.append("set -u")
    lines.append(shlex.join(["cd", "--", plan["cwd"]]))
    # Recheck the small identity files at the execution site, without importing
    # Torch/tokenizers. Integration must verify their semantic contents/assets.
    preflight = "\n".join(
        [
            "import hashlib, pathlib, sys",
            "args = sys.argv[1:]",
            "for value in args[:3]:",
            "    if not pathlib.Path(value).is_dir(): raise SystemExit('missing input directory')",
            "if not pathlib.Path(args[3]).is_file(): raise SystemExit('comparison runner unavailable')",
            "for i in range(4, len(args), 2):",
            "    p = pathlib.Path(args[i])",
            f"    if p.is_symlink() or not p.is_file() or p.stat().st_size > {MAX_FILE_BYTES}:",
            "        raise SystemExit('invalid identity file')",
            "    if hashlib.sha256(p.read_bytes()).hexdigest() != args[i+1]:",
            "        raise SystemExit('identity file SHA256 mismatch')",
        ]
    )
    check = [plan["argv"][0], "-B", "-c", preflight]
    check.extend(paths[n] for n in ("prism_root", "source_root", "checkpoint_root"))
    check.append(plan["argv"][1])
    for name in IDENTITY_FLAGS:
        check.extend([paths[name], plan["identities"][name]["sha256"]])
    lines.append(shlex.join(check))
    if plan["private_run_context"] is not None:
        context_check = (
            "import hashlib,json,pathlib,sys\np=pathlib.Path(sys.argv[1])\n"
            f"if not p.is_file() or p.is_symlink() or p.stat().st_size > {MAX_FILE_BYTES}:\n"
            "    raise SystemExit('invalid private context')\n"
            f"with p.open('rb') as stream: data=stream.read({MAX_FILE_BYTES + 1})\n"
            f"if len(data) > {MAX_FILE_BYTES} or not isinstance(json.loads(data),dict):\n"
            "    raise SystemExit('private context must be a bounded object')\n"
            "if sys.argv[2] and hashlib.sha256(data).hexdigest()!=sys.argv[2]:\n"
            "    raise SystemExit('private context SHA256 mismatch')\n"
        )
        lines.append(
            shlex.join(
                [
                    plan["argv"][0],
                    "-B",
                    "-c",
                    context_check,
                    paths["run_context"],
                    plan["private_run_context"].get("sha256", ""),
                ]
            )
        )
    lines.extend([shlex.join(plan["execution_argv"]), ""])
    return "\n".join(lines).encode()


def generate(args: argparse.Namespace) -> dict[str, Any]:
    """Validate and save a dry-run plan; do not probe remote inputs or run commands."""
    plan = _configuration(args)
    destination = _local_path(args.plan_dir)
    if any(_overlap(destination, p) for p in plan["paths"].values()):
        raise ValueError("plan directory must be disjoint from runtime inputs and outputs")
    plan["launch_id"] = _sha(_json_bytes(plan))
    script = _script(plan)
    plan["script"] = {"path": "job.sh", "bytes": len(script), "sha256": _sha(script)}
    _write_bundle(destination, {"job.sh": script, "launch-plan.json": _json_bytes(plan)})
    return plan


def collect(root: Path, destination: Path, includes: list[str], plan_path: Path) -> dict[str, Any]:
    """Export a lossy public summary; keep all raw evidence private and unchanged.

    Free text is omitted wholesale, not heuristically scanned for secrets. This
    bundle cannot replace private evidence for an independent parity review.
    """
    root, destination = _local_path(str(root)), _local_path(str(destination))
    plan_path = _local_path(str(plan_path))
    if not root.is_dir() or _overlap(root, destination) or _overlap(destination, plan_path):
        raise ValueError("collection requires distinct input, plan and destination paths")
    if (
        not includes
        or len(set(includes)) != len(includes)
        or not set(includes) <= set(EVIDENCE_FILES)
    ):
        raise ValueError("collection requires unique names from the fixed evidence allowlist")
    plan_data = _read_file(plan_path)
    plan = _object(plan_data)
    launch_id = _match(str(plan.get("launch_id", "")), r"[0-9a-f]{64}", "launch identity")
    if plan.get("schema_version") != 1 or plan.get("mode") != "dry_run":
        raise ValueError("unsupported launch plan")
    config = {k: v for k, v in plan.items() if k not in {"launch_id", "script"}}
    if _sha(_json_bytes(config)) != launch_id:
        raise ValueError("launch plan content does not match its identity")
    files: dict[str, bytes] = {}
    entries = []
    missing = sorted(set(EVIDENCE_FILES) - set(includes))
    receipt_valid = False
    result_valid = False
    for name in sorted(includes):
        path = root / name
        try:
            raw = _read_file(path)
        except FileNotFoundError:
            missing.append(name)
            continue
        if name in {"stdout.log", "stderr.log"}:
            data = b"[Content omitted: raw logs require private review.]\n"
            transformation = "all_log_content_omitted"
        elif name == "exit-status.json":
            receipt = _object(raw)
            code = receipt.get("exit_code")
            receipt_valid = (
                receipt.get("schema_version") == 1
                and type(code) is int
                and 0 <= code <= 255
                and receipt.get("launch_id") == launch_id
            )
            if not receipt_valid:
                raise ValueError("invalid exit receipt or launch identity mismatch")
            data = _json_bytes({"schema_version": 1, "exit_code": code, "launch_id": launch_id})
            transformation = "only_exit_code_and_launch_identity_retained"
        else:
            result = _object(raw)
            reported = result.get("status")
            result_valid = (
                result.get("schema_version") == 1
                and reported in ("pass", "fail", "incomplete")
                and type(result.get("complete")) is bool
            )
            if not result_valid:
                raise ValueError("invalid result envelope")
            data = _json_bytes(
                {
                    "schema_version": 1,
                    "evidence_level": "sanitized_transport_summary",
                    "complete": False,
                    "status": "incomplete",
                    "reported_status": reported,
                    "failure_reasons": [
                        "raw evidence omitted; independent private review required"
                    ],
                }
            )
            transformation = "only_reported_status_retained_not_a_parity_verdict"
        files[name] = data
        entries.append(
            {
                "path": name,
                "bytes": len(data),
                "sha256": _sha(data),
                "source_bytes": len(raw),
                "source_sha256": _sha(raw),
                "transformation": transformation,
            }
        )
    manifest = {
        "schema_version": 1,
        "launch_id": launch_id,
        "launch_plan_sha256": _sha(plan_data),
        "transport_complete": not missing and receipt_valid and result_valid,
        "parity_accepted": False,
        "missing": sorted(missing),
        "files": entries,
        "policy": "fixed_allowlist_lossy_summary; all other files and free text omitted",
        "max_file_bytes": MAX_FILE_BYTES,
    }
    files["artifact-manifest.json"] = _json_bytes(manifest)
    _write_bundle(destination, files)
    return manifest


def main(argv: list[str] | None = None) -> int:
    """Generate by default, or collect a fixed allowlist with the collect subcommand."""
    argv = sys.argv[1:] if argv is None else argv
    if argv and argv[0] == "collect":
        parser = argparse.ArgumentParser(description=collect.__doc__, allow_abbrev=False)
        parser.add_argument("--output-dir", required=True)
        parser.add_argument("--collection-dir", required=True)
        parser.add_argument("--launch-plan", required=True)
        parser.add_argument("--include", choices=EVIDENCE_FILES, action="append", required=True)
        args = parser.parse_args(argv[1:])
        try:
            manifest = collect(
                Path(args.output_dir),
                Path(args.collection_dir),
                args.include,
                Path(args.launch_plan),
            )
        except (ValueError, OSError) as exc:
            parser.error(str(exc))
        print("Sanitized collection saved; no parity acceptance claimed.")
        return 0 if manifest["transport_complete"] else 2
    parser = _generation_parser()
    args = parser.parse_args(argv)
    try:
        generate(args)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print("Dry-run plan saved. No job submitted; remote execution remains unqualified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
