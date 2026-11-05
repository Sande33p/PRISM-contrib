# Aurora Operations Guide

**Last updated**: February 28, 2026

Operational knowledge for running PRISM training on Aurora HPC: environment setup, launcher usage, session debugging, and troubleshooting. For DAOS storage setup, see [daos_setup.md](daos_setup.md). For scaling results and distributed training strategies, see [scaling_study.md](../results/scaling_study.md).

---

## Table of Contents

1. [Quick Start: Interactive Session](#quick-start-interactive-session)
2. [Launcher Reference](#launcher-reference)
3. [Environment & Dependencies](#environment--dependencies)
4. [Distributed Training Setup](#distributed-training-setup)
5. [DAOS Operations (Runtime)](#daos-operations-runtime)
6. [Troubleshooting](#troubleshooting)
7. [Issues Fixed (Operational)](#issues-fixed-operational)
8. [Environment Variables Reference](#environment-variables-reference)
9. [U2 text parity preparation](#u2-text-parity-preparation)

---

## U2 text parity preparation

`tools/launch_aurora_u2_parity.py` generates an inference-only PBS script and
`launch-plan.json`. It does **not** execute, submit, connect to Aurora, install
packages, download assets, or import model libraries. `--dry-run` is the default
and only generation mode; execution/submission options are rejected. This path
is separate from the training recipes below. The deferred comparison entry point
is `tools/parity/u2_text.py`, owned by Integration and absent from the prerequisite
base. Generating a command for it does not qualify execution or activate global
U2 selectors.

Supply literal absolute paths, exact identity-file SHA256s, and an exact declared
PRISM commit. The example uses caller-provided site variables; none are defaults.
`PRISM_PLAN_DIR` must be a new directory with an existing parent on the machine
generating the plan. `PRISM_OUTPUT_DIR` is a different, new directory on the
eventual execution filesystem, with its parent provisioned by the operator.
Input, output and plan areas must be disjoint. On macOS, use canonical local
paths (for example, resolve the temporary directory's symlink first).
At execution, every input/output path, including all ancestor components, must
be free of symlinks. The script checks this before output creation, redirection
or module setup; rejection leaves no runtime output or exit receipt. Use stable,
operator-controlled parent directories: these checks do not protect against
concurrent replacement of ancestors by another writer.

```bash
python tools/launch_aurora_u2_parity.py --dry-run \
  --prism-root "${PRISM_REMOTE_ROOT:?}" --prism-revision "${PRISM_COMMIT:?}" \
  --source-root "${PRISM_U2_SOURCE_ROOT:?}" \
  --checkpoint-root "${PRISM_U2_CHECKPOINT_ROOT:?}" \
  --output-dir "${PRISM_OUTPUT_DIR:?}" --plan-dir "${PRISM_PLAN_DIR:?}" \
  --acceptance-file "${PRISM_ACCEPTANCE_FILE:?}" \
  --acceptance-file-sha256 "${PRISM_ACCEPTANCE_SHA256:?}" \
  --asset-manifest "${PRISM_ASSET_MANIFEST:?}" \
  --asset-manifest-sha256 "${PRISM_ASSET_MANIFEST_SHA256:?}" \
  --source-manifest "${PRISM_SOURCE_MANIFEST:?}" \
  --source-manifest-sha256 "${PRISM_SOURCE_MANIFEST_SHA256:?}" \
  --account "${PRISM_ACCOUNT:?}" --queue debug --nodes 1 --walltime 00:05:00 \
  --filesystems "${PRISM_FILESYSTEMS:?}" --device xpu:0 --dtype bfloat16 \
  --python "${PRISM_SITE_PYTHON:?}" --module-use "${PRISM_MODULE_PATH:?}" \
  --module oneapi/release/2025.3.1 \
  --module mpich/prd/5.0.0.aurora_test.51a9474 \
  --module frameworks/2025.3.1
```

The generated environment setup always runs `module purge`, every explicit
`--module-use` directory in order, then each `--module` in order. The supplied
operator readiness record observed the matching compiler/MPI/framework sequence
above; incomplete/default module hierarchies failed. The search directory and
Python path must come from the site's reviewed configuration. The metadata
readback observed Python 3.12.12, Torch 2.10.0a0+git449b176, Transformers 4.57.6
and Safetensors 0.7.0; it did not import Torch, run XPU, establish model fit or
qualify this comparison. Preserve the site's accelerator-aware Torch. This
launcher does no dependency resolution or environment installation.

The debug validation profile admits only 1–2 nodes and 00:05:00–01:00:00,
inclusive. Those bounds cannot be overridden. Another queue requires explicit
`--max-nodes`, `--min-walltime` and `--max-walltime`; these declarations are bounds,
not an allocation. `--device` requires `xpu:N` with a nonnegative index; physical
device existence is unverified. Dtypes are `float32`, `float16` and `bfloat16`.
The script requires `PBS_JOBID` and a readable, nonempty `PBS_NODEFILE` before
creating output. After module setup and input checks, it starts **one comparison
process** using `mpiexec -n 1 --ppn 1` for placement on allocated compute nodes.
The explicit `xpu:N` argument is preserved. Requesting two nodes still launches
only one rank; it does not implement sharding or establish memory fit. PBS
environment checks are refusal guards, not independent proof of a live job;
the operator must use a real allocation and retain observed placement evidence.

Dry-run permits Aurora-only paths that are absent locally and labels their
identity hashes `declared_remote_unverified`. Remote existence and path safety
remain unverified until execution-site checks. `--validate-local-inputs` additionally
checks the three input directories and bounded JSON identity files against the
supplied hashes. It does not inspect checkpoint shards, import upstream code,
validate tokenizer binding, or interpret Evaluation's criteria. PRISM revision
and the pinned oracle revision remain declarations until the future runner
independently verifies loaded source and assets. Keep the exact six-entry pinned
source manifest, full imported-code manifest and independent checkpoint identity
distinct. Sampler settings and prompts come solely from the acceptance file.

Pass optional `--run-context /absolute/private/context.json` to transport the
runner's reviewed acceptance/instrumentation identities, module order, allocation
evidence and full source-manifest identity. It receives the same path-containment
and quoting checks; `--validate-local-inputs` also validates its bounded JSON object
and binds its hash in the generated script. Keep this context and the generated
plan private. The runner corroborates `PBS_JOBID` and `LOADEDMODULES`; absent evidence
is incomplete and mismatches fail. This metadata never authorizes a submission.

Plans and scripts contain private paths and must stay in the admitted private
area. The plan records the exact runner argv/cwd and MPI execution argv, ordered
module setup, declared identities, resource bounds, deterministic launch identity, script bytes/SHA256,
`submitted=false` and `job_id=null`. It never creates runtime result receipts.
New plan files are mode 0600 inside a mode 0700 directory; collisions fail.

On separately authorized execution, the script requires a fresh output directory,
captures separate `stdout.log` and `stderr.log`, and traps ordinary command/module
failures to write `exit-status.json` with the actual exit code and launch identity.
It verifies identity-file hashes and the runner's presence before invoking it.
Known service credentials are unset, Hugging Face offline modes are enabled, and
no environment dump is saved. The runner must still enforce local-only loading.
Fixed PBS stdout/stderr filenames retain early setup errors separately. They do
not validate the eventual submission directory: the operator must verify a
private, canonical, symlink-free plan/submission location outside every input
and runtime output tree, with fresh spool filenames. PBS can open these files
before any script guard runs. Moving the plan requires this check again; no
submission feature or automatic safety certification is provided. Failures
before the output directory exists, hard scheduler kills, or storage failures can
prevent an exit receipt; missing evidence is incomplete, never success.

For sanitized collection, see [the parity transport guide](../../tools/parity/README.md#u2-text-parity-transport).
Raw outputs remain private and unchanged. The collector's allowlist and lossy
summaries prevent arbitrary logs, prompts, private paths, credentials or asset
bytes from becoming public artifacts. Sanitized summaries cannot establish parity.

Before future submission the operator must freeze the actual allocation,
concurrency, nodes/accelerators, walltime, total node/accelerator hours, storage
allowance, retry limit and permitted commands; verify staged source/assets and
the complete environment; obtain Integration implementation and independent
exact-revision Harness code review with Evaluation criteria frozen before execution.
Evaluation applies those criteria to actual Aurora evidence afterward. Mac-owned
SSH/PBS transport works independently of the unresolved remote Codex sandbox
limitation; it does not bypass or disable that sandbox. The operator
owns submission, job-ID recording, monitoring, retention and cleanup. Reusing an
output directory or automatically retrying an ambiguous submission is unsupported.
Rollback before submission is removal of only the operator's new plan directory;
after execution, retain raw evidence until independent review and delete only
that run's admitted output/collection areas according to the retention decision.

## Quick Start: Interactive Session

### Step 1: Request Interactive Job

```bash
# Without DAOS
qsub -I -l select=2 -l walltime=1:00:00 -q debug -A ModCon -l filesystems=flare:home

# With DAOS (required for DAOS data access)
qsub -I -l select=2 -l walltime=1:00:00 -q debug -A ModCon -l filesystems=flare:home:daos_user_fs
```

### Step 2: Note the Allocated Nodes

```bash
cat $PBS_NODEFILE
# Example output:
# x4418c4s4b0n0
# x4418c7s7b0n0
```

### Step 3: Run from UAN (Recommended)

Open a new terminal on the UAN and run:

```bash
cd /lus/flare/projects/<project>/<user>/PRISM

# Quick test (5 steps, 1B model):
python tools/launch_aurora_web.py \
    --id TEST-RUN \
    --design PRISM-OLMO3-DEBUG-2NODE \
    --nodes 2 \
    --hosts <node1>,<node2> \
    --run-via-ssh \
    --webdataset-dir /flare/ModCon/ngetty/data/zone_a/pixmo_cap_webdataset \
    model.backbone_id=allenai/OLMo-1B-0724-hf \
    training.max_steps=5
```

**Expected success indicators**:
1. Log shows: `Loaded N shards from local_manifest.json (node-staged)` or shard discovery messages
2. Training starts without "stream empty" errors
3. Throughput metrics appear (e.g., "Throughput: XXX samp/s")

### Step 4: Run DAOS Pipeline

```bash
# Mount DAOS on both nodes (if not auto-mounted)
for NODE in <node1> <node2>; do
    ssh $NODE "module load daos && mkdir -p /tmp/AuroraGPT/prism_training_data /tmp/AuroraGPT/prism_models && \
        dfuse --mountpoint=/tmp/AuroraGPT/prism_training_data --pool=AuroraGPT --container=prism_training_data --disable-wb-cache && \
        dfuse --mountpoint=/tmp/AuroraGPT/prism_models --pool=AuroraGPT --container=prism_models --disable-wb-cache"
done

# Run training
python tools/launch_aurora_daos.py \
    --id MY-TEST \
    --design PRISM-IMAGE-ONLY-7B \
    --nodes 2 \
    --hosts <node1>,<node2> \
    --run-via-ssh \
    --no-pil4dfs \
    --dataset-groups pixmo \
    training.max_steps=50
```

---

## Launcher Reference

### Unified Dispatcher (`tools/launch_aurora_unified.py`)

Single entry point that dispatches to the storage-specific launcher based on `--storage`. The chosen launcher is invoked exactly as if you'd called it directly — every other flag is forwarded verbatim.

```bash
python tools/launch_aurora_unified.py --storage daos \
    --id MY-RUN --design PRISM-OLMO3-E2E-PROD --nodes 2 --batch
# exec's launch_aurora_daos.py --id MY-RUN ...

python tools/launch_aurora_unified.py --storage webdataset-staged \
    --id WEB-RUN --design PRISM-IMAGE-ONLY-2N --nodes 2 \
    --webdataset-dir /flare/<proj>/path/to/shards
# exec's launch_aurora_web.py ...

python tools/launch_aurora_unified.py --storage lustre \
    --id PRISM-IMAGE-ONLY-1N --nodes 1
# exec's launch_aurora.py ...
```

Note: `launch_aurora.py` (lustre) uses `--id` as the experiment-design selector and has no separate `--design` flag — the daos and webdataset-staged launchers take `--id` (run tag) **and** `--design` (experiment) as distinct args.

To see the flags a specific backend accepts:
```bash
python tools/launch_aurora_unified.py --storage daos --help
```

The three direct launchers (`launch_aurora_daos.py`, `launch_aurora.py`, `launch_aurora_web.py`) remain supported entry points — the dispatcher just gives you one name to remember and one place to discover the storage options.

### WebDataset Launcher (`tools/launch_aurora_web.py`)

Stages shards from Lustre to `/tmp` on each node, then trains from local NVMe.

```bash
python tools/launch_aurora_web.py \
    --id <run-id> \
    --design <experiment-design> \
    --nodes <N> \
    --batch                      # Submit as batch job (vs interactive)
    --queue debug                # PBS queue
    --walltime 00:30:00 \
    --webdataset-dir /flare/ModCon/ngetty/data/zone_a/pixmo_cap_webdataset \
    --dist-strategy ddp          # ddp, fsdp, hsdp
    --ddp-bucket-mb 50 \
    [hydra overrides...]
```

**Key flags**:
- `--hosts <h1>,<h2>` -- explicit hostnames (for interactive jobs)
- `--run-via-ssh` -- execute via SSH to first host (for UAN execution)

### DAOS Launcher (`tools/launch_aurora_daos.py`)

Reads data directly from DAOS mount, no staging.

```bash
python tools/launch_aurora_daos.py \
    --id <run-id> \
    --design <experiment-design> \
    --nodes <N> \
    --batch --queue debug --walltime 00:30:00 \
    --no-pil4dfs                 # CRITICAL: prevents FSDP hangs
    --dataset-groups pixmo       # or: all, projector, quick
    --use-bucketing \
    --bucket-buffer-size 5000 \
    --max-seq-length 1024 \
    --find-unused-params         # Only if model has unused modalities
    --composite                  # Use COMPOSITE GPU hierarchy (6 ranks/node)
    --dist-strategy fsdp \
    --fsdp-sharding full_shard \
    --fsdp-production-mode \
    --grad-ckpt-freq 2 \
    --per-rank-timing \
    --viz-interval 100 \
    [hydra overrides...]
```

### Key Experiment Designs

| Design | Model | Purpose |
|--------|-------|---------|
| `PRISM-IMAGE-ONLY-2N` | OLMo-1B | 2-node projector training |
| `PRISM-IMAGE-ONLY-7B` | OLMo-7B | Projector training |
| `PRISM-OLMO3-E2E-PROD` | OLMo-7B | E2E training (all params) |
| `PRISM-OLMO3-E2E-COMPOSITE-1NODE` | OLMo-7B | E2E on COMPOSITE (128 GB) |
| `PRISM-AGPT2B-PROJ` | AuroraGPT-2B | AuroraGPT projector training |
| `PRISM-AGPT2B-PROJ-DEBUG` | AuroraGPT-2B | Quick debug (100 steps) |

---

## Environment & Dependencies

### Frameworks module: PRISM pins 2025.3.1

**Load `frameworks/2025.3.1` explicitly. Do not load bare `frameworks`.**

```bash
module load frameworks/2025.3.1
```

`2025.3.1` is the validated stack: it is what the Aurora lockfiles in
`requirements/` were resolved against, what the launchers and
`tools/build_aurora_env.sh` load, and what every throughput number in
[../results/scaling_study.md](../results/scaling_study.md) was measured on. It
also carries vLLM `0.15.0+xpu`, which the in-tree plugin needs — the older
`2025.2.0` shipped `0.10.1rc2` and lacks the required surface (see
[../evaluation/inference_vllm.md](../evaluation/inference_vllm.md)).

**Aurora's default has since moved to `2026.1.0`.** A bare `module load
frameworks` now gives you 2026.1.0, not the stack PRISM is validated on, and
`2025.3.1` is reachable only under the older `/opt/aurora/26.26.0` tree. Every
PRISM doc and script therefore names the version explicitly.

Support for 2026.1.0 is planned but **not yet validated** — nothing in this
repository has been run against it, and the lockfiles have not been
regenerated for it. Until that work lands, treat a bare `module load
frameworks` as a configuration error: the symptom is usually an import or ABI
mismatch between the module's torch and the venv built against the pinned one.

Anywhere else in the docs that shows `module load frameworks/2025.3.1`, this is
the reason; those pages do not restate it.

### Module Loading Order (Critical)

```bash
# MUST be in this order:
module load frameworks/2025.3.1       # Provides PyTorch, IPEX, torch_geometric
source .venv-deepspeed/bin/activate   # Provides webdataset, custom packages
```

Reversing this order causes `torch_geometric` import failure. The `frameworks` module provides base Python; custom packages are in the venv.

### One-shot venv build

`tools/setup_deepspeed_env.sh` automates the venv creation:

```bash
./tools/setup_deepspeed_env.sh
```

It loads the pinned frameworks module, creates `.venv-deepspeed` with
`--system-site-packages`, and installs `deepspeed`, `torch_geometric`, and
the other custom packages — handling `timm`/`walrus` install quirks.

Both env builders read `FRAMEWORKS_MODULE`, defaulting to
`frameworks/2025.3.1`. That is the override point when 2026.1.0 is validated:

```bash
FRAMEWORKS_MODULE=frameworks/2026.1.0 bash tools/build_aurora_env.sh
```

Expect to regenerate the lockfiles alongside it — they are resolved as a delta
over the module, so a venv built against one module and a lockfile resolved
against another is the mismatch `setup_deepspeed_env.sh` refuses with
"was built against a different frameworks python".

### transformers Version

Aurora's system `transformers` can lag the model code PRISM uses, and two of
PRISM's requirements cannot be satisfied at the same time:

| | needs | why |
|---|---|---|
| OLMo-3 backbone | `transformers>=4.57.0` | model type not in older releases |
| Intern-S2 Preview encoder | `transformers>=5.2.0` | its vendored config imports `RopeParameters` from `transformers.modeling_rope_utils`, absent on 4.57.6 |
| vLLM 0.15.0+xpu (system) | `transformers<5,>=4.56.0` | declared in the wheel bundled with frameworks/2025.3.1 |

`>=5.2.0` and `<5` have an empty intersection, so the build picks a side.

**Two build variants**, mutually exclusive, on separate `VENV_PATH`s:

```bash
# default — vLLM usable, intern_s2 encoders unavailable
VENV_PATH=/flare/ModCon/$USER/prism-envs/py3.12 bash tools/build_aurora_env.sh

# intern-s2 — encoders usable, vLLM outside its declared range
VENV_PATH=/flare/ModCon/$USER/prism-envs/py3.12-interns2 bash tools/build_aurora_env.sh --intern-s2
```

`PRISM_INTERN_S2=1` in the environment is equivalent to the flag, for job
scripts that cannot pass one. The variant is recorded in the venv's
`PRISM_BUILD_INFO` (`variant:` and `intern_s2_encoders:`), so you can tell
which one you are sourcing without re-deriving it from `pip freeze`.

**Default variant** installs no transformers overlay at all — it keeps the
system 4.57.6, which clears OLMo-3's floor and stays inside vLLM's range.
Setting `ts_projector: intern_s2` or `intern_s2_397b` against this venv raises
a `RuntimeError` from `src/encoders/time_series.py` naming the transformers
version it found. That is the intended failure: it is loud, at construction
time, and tells you to rebuild with the flag.

**`--intern-s2` variant** additionally installs
`requirements/aurora-py3.12.intern-s2.nodeps.txt`, which pins three coupled
packages. They are validated together, not independent minimums — bump one
only after re-validating the others:

```bash
pip install --no-deps 'transformers==5.2.0' 'huggingface-hub==1.32.0' 'hf_xet==1.6.0'
```

Use `--no-deps` to avoid pulling in a conflicting `torch` version.

*Why all three*: transformers 5.2.0 imports `is_offline_mode` from
`huggingface_hub`, which does not exist before the huggingface-hub 1.x line
(Aurora's system version is 0.36.2) — without it you hit `ImportError: cannot
import name 'is_offline_mode' from 'huggingface_hub'`. Bumping to
huggingface-hub 1.32.0 in turn needs a newer `hf_xet` than Aurora ships
(1.2.0-ish); with the old one you see a misleading `OSError: To use optimized
download using Xet storage, you need to install the hf_xet package` even
though `hf_xet` **is** installed — it is just too old.

**vLLM in the `--intern-s2` variant (unverified — do not assume it's fine)**:
`pip check` reports vLLM as broken once transformers 5.2.0 is installed, and
`tools/build_aurora_env.sh` prints that as a non-fatal warning rather than
failing the build — you asked for the trade by passing the flag. `import vllm`
and constructing `vllm.LLM(...)` have been confirmed to still succeed under
transformers 5.2.0 in ad hoc testing on the login node, but that is not the
same as verified serving. Model loading, worker startup, and actual generation
under transformers 5.2.0 have **not** been tested end-to-end. Use a
default-variant venv for vLLM work.

If the same warning appears in a **default** build, something unexpected
installed transformers 5.x — the build says so explicitly rather than reusing
the Intern-S2 wording. Check the lockfile and nodeps manifests before using
that venv.

**CI takes the default side of the same split.** `requirements/ci.txt` pins
`transformers==4.57.6` — matching what frameworks/2025.3.1 ships, so `mypy src`
type-checks against the API surface the cluster actually runs. It is the only
pinned runtime package in that file; CI floats the rest. Tests that need
Intern-S2 are marked `aurora` and are deselected by every CI job.

For `launch_aurora_web.py`, remember that the default path uses `deepspeed_env.tar.gz`, not the repo checkout's live `.venv-deepspeed`. After updating `.venv-deepspeed`, either repack it:

```bash
tar -czf deepspeed_env.tar.gz .venv-deepspeed
```

or launch with `--use-shared-venv` and `VENV_PATH` pointing at the updated venv.

**Failed approach**: Setting `PYTHONPATH` to user site-packages caused `std::bad_alloc` crashes from `torch` ABI mismatch.

### Triton Backend

The packed venv may contain `triton-3.6.0` with only AMD/NVIDIA backends, shadowing the system `pytorch_triton_xpu-3.4.0` which has the Intel backend.

**Fix**: Remove `triton` and `triton-3.6.0.dist-info` from the packed venv. Python falls through to system Triton 3.4.0.

### Packed Venv (deepspeed_env.tar.gz)

For batch jobs on many nodes, the venv is packed as a tarball and extracted to `/tmp`:
```bash
tar -xzf deepspeed_env.tar.gz -C /tmp/deepspeed_env
```

---

## Distributed Training Setup

### Backend Configuration

```python
# MANDATORY on Aurora:
torch.distributed.init_process_group(backend="xccl")
```

### Environment Variable Setup (Native DDP)

Bypass mpi4py to avoid MPI re-initialization conflicts:

```python
def _setup_distributed_env_only():
    rank = get_env_int(["RANK", "PALS_RANKID"], 0)
    world_size = get_env_int(["WORLD_SIZE", "PALS_SIZE", "PALS_LOCAL_SIZE"], 1)
    local_rank = get_env_int(["LOCAL_RANK", "PALS_LOCAL_RANKID"], 0)

    # When ZE_AFFINITY_MASK is set, each rank sees only its own GPU
    if "ZE_AFFINITY_MASK" in os.environ:
        device = "xpu:0"
    else:
        device = f"xpu:{local_rank}"
```

### Multi-Node WORLD_SIZE

On multi-node jobs, `PALS_SIZE` may not be set correctly. Compute:
```bash
export WORLD_SIZE=$((NUM_NODES * LOCAL_WORLD_SIZE))
```

### CCL Settings (Critical)

```bash
export CCL_PROCESS_LAUNCHER=none   # NOT pmix
export CCL_ATL_TRANSPORT=ofi       # NOT mpi
```

Without this, CCL tries to initialize MPI inside mpiexec causing "Fatal error in internal_Init_thread".

**After `module load frameworks`**: Re-export CCL variables because `frameworks` overrides them:
```bash
export CCL_PROCESS_LAUNCHER=none
export CCL_ATL_TRANSPORT=ofi
export CCL_OP_SYNC=1
export CCL_OFI_ENABLE_HOSTNAME_SHARING=0
```

### Barrier Pattern

```python
# CORRECT: All ranks call the same barrier
if local_rank == 0:
    model = load_model()     # Rank 0 does work first
dist.barrier()               # ALL ranks call this
if local_rank != 0:
    model = load_model()     # Others load from cache
dist.barrier()               # ALL ranks call this
```

**Anti-pattern** (causes deadlock):
```python
# WRONG: Different ranks call different barriers
if local_rank != 0:
    dist.barrier()
if local_rank == 0:
    dist.barrier()
```

### HOSTNAME Resolution

PBS_NODEFILE contains FQDNs. Strip before adding HSN suffix:
```bash
MASTER_HOST=${MASTER_HOST%.hsn.cm.aurora.alcf.anl.gov}
export MASTER_ADDR="${MASTER_HOST}.hsn.cm.aurora.alcf.anl.gov"
```

---

## DAOS Operations (Runtime)

For initial DAOS setup (container creation, data upload), see [daos_setup.md](daos_setup.md).

### DAOS Container Structure

```
AuroraGPT pool
├── prism_training_data     # Training data
│   ├── pixmo/
│   ├── s1mmalign/
│   ├── cosyn/
│   └── nemotron/
└── prism_models            # HuggingFace model weights
    └── hub/
        ├── models--allenai--OLMo-1B-0724-hf/
        ├── models--allenai--OLMo-7B-0724-hf/
        ├── models--google--siglip2-base-patch16-224/
        └── ...
```

### Mount Paths

| Container | Mount Path |
|-----------|------------|
| Data | `/tmp/${USER}/AuroraGPT/prism_training_data` |
| Models | `/tmp/${USER}/AuroraGPT/prism_models` |
| Models Hub | `/tmp/${USER}/AuroraGPT/prism_models/hub` |

### Model Loading Strategy

The launcher uses this priority:
1. **DAOS symlink** (instant) -- symlinks from `/tmp/huggingface/hub/` to DAOS models hub
2. **Lustre copy** (slow, 4.5 min for 7B) -- copies from `/flare/ModCon/ngetty/huggingface/hub/`

### Key Fixes Applied

| Issue | Fix |
|-------|-----|
| Rank-0 coordinated shard discovery | Only rank 0 reads manifest, broadcasts via `dist.broadcast()` (0.6s vs 30+ min timeout) |
| `glob.glob()` hangs on dfuse | Changed to `os.listdir()` for validation shards |
| `libpil4dfs.so` hangs Python | Use `--no-pil4dfs` flag (DAOS-17499) |
| Case-sensitive model names | Added `find -iname` fallback in launcher |

---

## Troubleshooting

### "Stream persistently empty" Error

**Check 1**: Verify local_manifest.json exists
```bash
ssh <node> 'cat /tmp/webdataset/local_manifest.json | head -20'
```

**Check 2**: Verify shards are staged
```bash
ssh <node> 'ls /tmp/webdataset/*.tar | wc -l'  # Should be ~307
```

**Check 3**: Verify code fix is applied
```bash
grep -n "local_manifest.json" src/data/multimodal.py | head -5
```

### SSH Connection Fails

Use short hostname (not FQDN):
```bash
# Good: x4418c4s4b0n0
# Bad:  x4418c4s4b0n0.hsn.cm.aurora.alcf.anl.gov
```

### mpiexec Hangs

**Check 1**: Kill stale processes
```bash
for node in <node1> <node2>; do
    ssh $node 'pkill -u $USER -f "python src/train.py"'
done
```

**Check 2**: Verify HSN connectivity
```bash
ssh <node1> 'ping -c 1 <node2>.hsn.cm.aurora.alcf.anl.gov'
```

### DAOS Issues

**"DAOS agent not found"**:
```
Failed to connect to /var/run/daos_agent/daos_agent.sock
```
Job was not submitted with `-l filesystems=daos_user_fs`. Re-submit.

**"Data mount empty"**: Run `./scripts/setup_daos_container.sh mount` first.

**Slow model loading (4+ min)**: Models container not mounted. Run `./scripts/setup_daos_models.sh mount`.

**NA_HOSTUNREACH errors**: Missing `--no-vni` flag in mpiexec (required for DAOS).

### DDP Crashes

**"Empty bucket specified" or "Your training graph has changed"**:
- Model has trainable parameters for modalities not in the data
- Fix: Use `--design PRISM-IMAGE-ONLY-*` which sets `model.modalities=[text,image]`

**"Fatal error in internal_Init_thread"**:
- CCL trying to initialize MPI inside mpiexec
- Fix: `CCL_PROCESS_LAUNCHER=none` and `CCL_ATL_TRANSPORT=ofi`

### OOM Errors

| Error Location | Likely Cause | Fix |
|----------------|-------------|-----|
| `cross_entropy_loss` | Batch size too large | Reduce BS (7B max BS=2 for DDP, BS=16 for FSDP) |
| `logits.float()` | 256K vocab FP32 upcast | Use manual BF16 cross-entropy (see [auroragpt_vlm.md](../models/auroragpt_vlm.md)) |
| `optimizer.step()` | DDP E2E (88 GB needed) | Switch to FSDP |

### XCCL / UR_RESULT_ERROR_OUT_OF_RESOURCES

**In COMPOSITE mode**: Usually stale state from killed processes. `pkill -9 python3; sleep 10` then retry.

**With AuroraGPT-2B**: Usually from `logits.float()` upcast with 256K vocab. Use manual BF16 cross-entropy. See [auroragpt_vlm.md](../models/auroragpt_vlm.md) Issues 7-9.

### `xpu-smi` Symbol Errors

You may see errors like `xpu-smi: symbol lookup error ... undefined symbol: spdlog...`. This is a conflict between system libraries and the python environment. **Performance monitoring tools may fail, but training will still run** — safe to ignore.

### HDF5 / Walrus Import Errors

If you see errors related to `libhdf5` or `walrus` imports, ensure `module load hdf5` is in your launch script (the launchers add it automatically). Re-run the launcher generator if you are using an old generated script.

---

## Issues Fixed (Operational)

### Multi-Node Interactive Launcher (Feb 16)

**Problem**: Launcher couldn't run multi-node interactive jobs from UAN because `PBS_NODEFILE` is only available inside the PBS job shell.

**Fix**: Added `--hosts <host1>,<host2>` and `--run-via-ssh` arguments to launchers.

### WebDataset Local Manifest (Feb 16)

**Problem**: `stage_shards.py` stages only ~307 shards per node and creates `local_manifest.json`, but the data loader read `manifest.json` (full 614 shards), causing "stream empty" errors.

**Fix**: Updated `src/data/multimodal.py` to prefer `local_manifest.json` when present.

### WebDataset Launcher List Handling (Feb 17)

**Problem**: Hydra list overrides like `model.modalities=[text,image]` were broken.

**Fix**: Added proper list serialization in both `launch_aurora_web.py` and `launch_aurora_daos.py`.

### Hydra Config Override Syntax

- Use `+data.active_zone=zone_a` (with `+`) to ADD new config keys
- Use `data.active_zone=zone_a` (without `+`) only for EXISTING keys

### Empty Env Var Handling

MPI environment variables may be empty strings. Use defensive parsing:
```python
def get_env_int(keys, default):
    for key in keys:
        val = os.environ.get(key, "")
        if val and val.strip():
            try:
                return int(val)
            except ValueError:
                continue
    return default
```

---

## Environment Variables Reference

### Training Control

| Variable | Default | Purpose |
|----------|---------|---------|
| `USE_NATIVE_DDP` | `1` | Use env-var-based distributed setup (bypass mpi4py) |
| `USE_MULTI_DATASET` | `1` | Use MultiWebDataset for multi-source loading |
| `MAX_SEQ_LENGTH` | `2048` | Max token length (use 1024 for projector training) |
| `GRAD_CKPT_FREQ` | `1` | Gradient checkpoint frequency (0=off, 2=every other layer) |
| `FSDP_PRODUCTION_MODE` | `0` | Skip non-essential XPU syncs in `train.py` (+6.6% throughput) |
| `PRISM_PRODUCTION_MODE` | `0` | Skip ZoneATrainer per-microbatch syncs and barriers; pairs with `FSDP_PRODUCTION_MODE`. Both are set together by `--benchmark-mode` and by `--fsdp-production-mode`. |
| `TORCH_COMPILE` | `0` | Enable torch.compile (not viable on Aurora) |
| `PRISM_CACHE_CLEAR_INTERVAL` | `0` | Periodic gc.collect + empty_cache (0=off) |

### DDP Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `DDP_BUCKET_CAP_MB` | `25` | DDP bucket size |
| `PRISM_DDP_FIND_UNUSED` | auto | Force `find_unused_parameters`. Auto-detects **(a)** COMPOSITE mode and **(b)** multi-modality projector-only (`trainable < 1GB` AND `len(model.modalities) > 1`) — the second auto-detect was added to prevent the empty-bucket DDP crash when ranks see heterogeneous batches (see note below). Set explicitly to override either auto-detect. |
| `PRISM_DDP_GRAD_BUCKET_VIEW` | `1` | Use `gradient_as_bucket_view` |
| `DDP_DEBUG` | `0` | Per-microbatch progress logging |
| `DDP_DEBUG_STEPS` | `3` | Steps to log when DDP_DEBUG=1 |
| `DEBUG_SYNC` | `0` | Verbose sync logging |

> **Multi-modality projector-only DDP:** when the LLM is frozen and
> `len(model.modalities) > 1`, some ranks see batches that skip a modality,
> so the corresponding projector gets no gradient flow and DDP's bucket
> rebuild fails at the first forward with `RuntimeError: Empty bucket specified`.
> **Auto-fixed:** `_wrap_ddp` detects this regime (trainable < 1 GB AND
> multiple modalities) and sets `find_unused_parameters=True` automatically.
> Set `PRISM_DDP_FIND_UNUSED=0` to override the auto-detect.

### Data Pipeline

| Variable | Default | Purpose |
|----------|---------|---------|
| `DAOS_MOUNT` | `/tmp/${USER}/AuroraGPT/prism_training_data` | DAOS data mount |
| `DATASET_CONFIG` | `src/conf/data/daos_datasets.yaml` | Config file |
| `DATASET_GROUPS` | `all` | Groups/preset to load |
| `USE_BUCKETING` | `false` | Use BucketedMultiWebDatasetWrapper |
| `BUCKET_BUFFER_SIZE` | `2000` | Buffer size for bucketing |
| `BUCKET_NUM_BUCKETS` | `8` | Number of length sub-buckets |
| `ENABLE_ALL_MODALITIES` | `0` | Activate every modality in `model.modalities`, falling back to dummy tensors when shards are missing — see note below. (Multi-modality projector DDP runs no longer need a manual `PRISM_DDP_FIND_UNUSED=1` — `_wrap_ddp` auto-detects.) |
| `PER_RANK_TIMING` | `0` | Straggler detection every 50 steps |
| `ENABLE_PROFILER` | `0` | PyTorch profiler |
| `LOG_EVERY_N_STEPS` | `10` | Logging frequency |

> **`ENABLE_ALL_MODALITIES=1` details:** passes `allow_dummy_data=True` to
> `StreamingMultimodalDataset`, activates every `skip=true` entry in
> `datasets_config.json` whose modality is in `model.modalities`, and marks
> them `fallback_dummy=true` at runtime. Missing local shards then degrade to
> modality-correct dummy tensors (`src/modalities.py:make_dummy_batch`)
> instead of crashing.

### CCL/Network

| Variable | Value | Purpose |
|----------|-------|---------|
| `CCL_PROCESS_LAUNCHER` | `none` | Avoid MPI re-init |
| `CCL_ATL_TRANSPORT` | `ofi` | Use OFI transport |
| `CCL_WORKER_COUNT` | `4` | CCL worker threads |
| `CCL_ALLREDUCE` | `ring` | AllReduce algorithm |
| `CCL_OP_SYNC` | `1` | Synchronous ops |
| `FI_CXI_RX_MATCH_MODE` | `hybrid` | Slingshot optimization |

### GPU/Device

| Variable | Value | Purpose |
|----------|-------|---------|
| `ZE_FLAT_DEVICE_HIERARCHY` | `FLAT` | 12 tiles/node (default) |
| `ZE_FLAT_DEVICE_HIERARCHY` | `COMPOSITE` | 6 cards/node (128 GB each) |
| `ZE_AFFINITY_MASK` | `$LOCAL_RANK` | Restrict each rank to one tile |
| `ZE_ENABLE_PCI_ID_DEVICE_ORDER` | `1` | Consistent GPU ordering |
| `HF_HUB_OFFLINE` | `1` | Prevent HF network calls |
| `TRANSFORMERS_OFFLINE` | `1` | Prevent transformers network calls |

### Key Files

| File | Purpose |
|------|---------|
| `tools/launch_aurora_daos.py` | DAOS launcher |
| `tools/launch_aurora_web.py` | WebDataset staging launcher |
| `tools/run_composite_interactive.sh` | COMPOSITE mode interactive |
| `tools/run_agpt2b_interactive.sh` | AuroraGPT-2B interactive |
| `scripts/daos_mount_helper.sh` | Unified DAOS mount helper |
| `scripts/setup_daos_container.sh` | Data container setup |
| `scripts/setup_daos_models.sh` | Models container setup |
