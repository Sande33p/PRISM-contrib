# Parity harness

Smoke and dry-run scripts originally created as gates by the
`aurora-scaling-vlm` → `main` migration, completed 2026-05-15. The
scripts here are
skeletons; the real validated invocations are documented inline in
this README and were used to validate phase 2 PRs on 2026-05-15.

The historical training scripts below assume:

- `module load frameworks/2025.3.1` has been run (or is run inside the script).
- The caller is on a UAN. Any compute-node work goes through
  `tools/launch_aurora_daos.py` (typically `--batch` for hands-off
  validation or `--run-via-ssh` for interactive iteration on a hold).

## U2 text parity transport

The independent U2 prerequisite uses `tools/launch_aurora_u2_parity.py` and the
deferred Integration entry point `tools/parity/u2_text.py`. It does not use the
training scripts or generic token-embedding path. See the
[Aurora preparation guide](../../docs/platforms/aurora_operations.md#u2-text-parity-preparation)
for required launcher options, module purge/use/load ordering and execution gates.
The launcher itself has no submission or execution mode. CPU fixtures prove
packaging/failure behavior only, never model/XPU/checkpoint parity.

Generation creates only `job.sh` and `launch-plan.json` in a new private plan
directory. Stage those unchanged bytes through the separately admitted operator
transport, verify their hashes, and retain the plan with private runtime evidence.
The eventual plan/submission directory must be private, canonical, symlink-free
and disjoint from inputs and runtime outputs, with unused scheduler-spool names.
Fixed relative PBS stdout/stderr names do not establish that safety; PBS may
open them before script checks. The operator must verify the execution-site
location after transport. Remote dry-run existence and path safety remain
unverified.
The generated runner argv passes source-root, checkpoint-root, output-dir,
acceptance-file, asset-manifest, source-manifest, device and dtype unchanged. The
PRISM root is its working directory. Exact manifest hashes are checked again by
the script before the runner; verifying their semantics and actual source/model
binding remains Integration's responsibility. The launcher never invents sampler
defaults, expected outputs or Evaluation's numerical thresholds.

The generated script refuses execution without PBS allocation metadata and a
readable, nonempty node file. It rejects symlink components in all input/output
paths before output creation, redirects or module setup; keep their parent
directories stable and operator-controlled throughout execution. Early rejection
has no runtime receipt and remains incomplete. After the checks and ordered
module setup, `mpiexec -n 1 --ppn 1` launches the unchanged runner argv with its
explicit indexed XPU. The plan records this separate execution argv. One rank
does not establish memory fit or distributed execution. CPU fake allocation/MPI
fixtures test these contracts; actual scheduler placement still needs separately
authorized compute-node evidence and independent review.

After an authorized run and a separately authorized copy of its private evidence,
collect a sanitized summary locally with an explicit filename allowlist:

```bash
python tools/launch_aurora_u2_parity.py collect \
  --output-dir "${PRISM_PRIVATE_RESULT_DIR:?}" \
  --collection-dir "${PRISM_NEW_COLLECTION_DIR:?}" \
  --launch-plan "${PRISM_PLAN_DIR:?}/launch-plan.json" \
  --include stdout.log --include stderr.log \
  --include exit-status.json --include result.json
```

Both local input and destination paths must be canonical and free of symlinks;
the destination must be new, with an existing parent and outside the input area.
There is no recursive copy. Only the four names above are accepted; duplicate
names, path traversal, symlinks, hardlinks, non-regular files and files over 8 MiB
are rejected before writing. Thus selected raw evidence is bounded to 32 MiB
(plus an 8 MiB launch plan); oversized traces stay in private storage for separate
review. Files outside the allowlist, including model/tokenizer bytes, are never
read. An existing collection directory is never overwritten.

`stdout.log` and `stderr.log` remain separate but contain only omission notices.
`exit-status.json` retains the validated exit code and matching launch identity.
`result.json` retains only a fixed-enum `reported_status` and an explicit
incomplete summary: free-text failures, prompts, identities, paths and traces are
omitted wholesale. This intentionally avoids claiming that pattern matching can
find every credential. Unknown exit/result fields are never copied. Raw files
are not modified, and these sanitized summaries do not replace raw evidence.

`artifact-manifest.json` has schema version 1, sorted relative file entries,
exported byte lengths and SHA256s, separate source byte lengths/SHA256s, explicit
transformation labels, the launch-plan hash, missing required names and
`parity_accepted=false`. It does not hash itself. `transport_complete` means all
four files and valid envelope/exit binding were collected; even a nonzero exit
can have complete failure evidence. It is **not** a successful execution or
parity verdict. Missing files are not manufactured; an incomplete collection is
saved with CLI exit code 2. Invalid receipts/inputs fail without a collection.
Generation and complete collection return 0; usage/validation errors return 2.

Independent Evaluation must inspect the private, exact-revision source/asset
identities, observed runtime/job identity, traces, numerical comparisons and
terminal semantics against its previously frozen criteria. No successful verdict
can be inferred from an author-supplied status or the public transport summary.

## Scripts

| Script | Purpose | Status |
|---|---|---|
| `launcher_dry_run.sh` | Diff `launch_aurora_daos.py --dry-run` between `main` and a candidate branch. | Ready |
| `smoke_ddp_projector.sh` | 1-node DDP projector-only, ~50 steps (DAOS). | Ready (PR #48) |
| `smoke_ddp_projector_web.sh` | 1-node DDP projector-only, ~50 steps (Lustre/WebDataset fallback). | Ready (PR #48) |
| `smoke_fsdp_e2e.sh` | 2-node FSDP E2E, ~30 steps (DAOS, BS=4 to fit 2N HBM). | Ready (PR #48) |
| `smoke_fsdp_e2e_web.sh` | 2-node FSDP E2E, ~30 steps (Lustre/WebDataset fallback, BS=4). | Ready (PR #48) |
| `smoke_vla.sh` | 20-step `CalvinVLADataset` run via `ZoneAVLATrainer`. | Ready (PR #48; DAOS-only, no Lustre variant) |
| `smoke_zero2.sh` | 1-node ZeRO-2 30-step. | Skeleton — see invocations below |
| `smoke_zero3.sh` | 1-node ZeRO-3 30-step. | Skeleton — see invocations below |

### Lustre/WebDataset variants (`*_web.sh`)

Use when DAOS is unavailable (e.g. `daos_user_fs` PBS resource off). These call
`tools/launch_aurora_web.py` instead of `_daos.py` and stage shards from
`/flare/.../pixmo_cap_webdataset` to `/tmp` at job start (~10–15 min overhead
on top of normal job time). Pass `WEBDATASET_DIR=<path>` and
`SHARED_HF_HOME=<hf-hub-on-flare>` env vars to override defaults.

### Throughput pass criterion

The trainer logs `[THROUGHPUT] N samples/sec` every 10 steps plus a final
`Throughput: N samp/s` summary at end. Scripts grep for both patterns; defaults:

- DDP (50 steps): ≥ 4 of ~5 throughput lines
- FSDP (30 steps): ≥ 2 of ~3 throughput lines
- VLA (20 steps): ≥ 1 of ~2 throughput lines

This validates that training reached at least the second 10-step checkpoint
without crashing. The pass criterion is "completes without OOM/error";
throughput numbers and loss trajectory are reviewed manually.

A pytest gate (`pytest tests/ --timeout=60 -q`) must remain green on
`origin/main` throughout.

vLLM token-level parity (`tools/vllm_parity.py`) target: bitwise
match on top-1 tokens (`[demo] >>>` and `[vllm] >>>` lines should
print identical strings). Achieved on PR 10.

---

## Validated smoke invocations (2026-05-15)

These are the actual commands that successfully validated each phase 2
PR end-to-end. Use them as templates; substitute your own `--id` and
node names.

### Pre-flight

```bash
module load frameworks/2025.3.1
# Confirm the launcher's mpiexec body has no stray apostrophes
# (apostrophes inside `bash -lc '...'` silently kill 11/12 ranks):
python tools/launch_aurora_daos.py --id Q --design PRISM-PROJ-ABLATION-LAYERNORM \
    --batch --dry-run --nodes 1 --queue debug
awk '/^mpiexec/,/^.{0,2}'\''$/' jobs/run_aurora_daos_Q_*_batch.sh | grep -c "'"
# Should report exactly 2 (the opener and closer of bash -lc '...').
```

### 1-node DDP projector smoke (~5 min including setup)

Native DDP path. **Do NOT add `--use-accelerate`** — Accelerate's DDP
wrapper raises `IndexError` on Aurora XPU (see DeepSpeed gotchas below).

```bash
python tools/launch_aurora_daos.py \
    --id PR3-DDP \
    --design PRISM-PROJ-ABLATION-LAYERNORM \
    --batch --nodes 1 --queue debug \
    --dist-strategy ddp \
    --max-seq-length 1024 --use-bucketing --no-pil4dfs \
    training.max_steps=50 \
    training.eval_every_n_steps=10000 training.save_every_n_steps=10000
```

Expected: 50 steps complete, ~190–250 samp/s (variance is environment
noise — compile-cache state, rank allocation), loss decreasing.

### 2-node FSDP smoke (~10 min)

```bash
# Submit batch:
python tools/launch_aurora_daos.py \
    --id FSDP-2N \
    --design PRISM-PROJ-ABLATION-LAYERNORM \
    --batch --nodes 2 --queue debug-scaling \
    --dist-strategy fsdp --fsdp-sharding full_shard --fsdp-production-mode \
    --max-seq-length 1024 --use-bucketing --no-pil4dfs \
    training.max_steps=20 training.batch_size=4 \
    training.eval_every_n_steps=10000 training.save_every_n_steps=10000
```

Or interactively on a hold:

```bash
# After PBS_JOBID is set in env (qsub -I or hold script writes nodefile):
python tools/launch_aurora_daos.py \
    --id FSDP-2N \
    --design PRISM-PROJ-ABLATION-LAYERNORM \
    --nodes 2 --hosts <node1>,<node2> --run-via-ssh \
    --pbs-jobid $PBS_JOBID \
    --dist-strategy fsdp --fsdp-sharding full_shard --fsdp-production-mode \
    --max-seq-length 1024 --use-bucketing --no-pil4dfs \
    training.max_steps=20 training.batch_size=4 \
    training.eval_every_n_steps=10000 training.save_every_n_steps=10000
```

Expected: 20 steps complete, ~28 samp/s on OLMo-1B (this design is
projector-only smoke — production OLMo-3 7B HSDP+compile gets 122 samp/s,
see scaling_study.md). Loss decreasing.

### DeepSpeed ZeRO-2 / ZeRO-3 smokes (~5–8 min each)

Requires PR 6a + PR 6b + PR 6c on the working branch.

```bash
# ZeRO-2:
python tools/launch_aurora_daos.py \
    --id Z2-SMOKE \
    --design PRISM-PROJ-ABLATION-LAYERNORM \
    --batch --nodes 1 --queue debug-scaling \
    --deepspeed 2 \
    --max-seq-length 1024 --use-bucketing --no-pil4dfs \
    training.max_steps=20 \
    training.eval_every_n_steps=10000 training.save_every_n_steps=10000

# ZeRO-3: same with `--deepspeed 3`
```

Expected: 20 steps complete, ZeRO-2 ~170 samp/s, ZeRO-3 ~155 samp/s.
Loss should decrease (e.g. 11.06 → 10.19 in 10 steps for OLMo-1B from
scratch; converged checkpoints will start lower).

### vLLM parity (~3 min)

Requires PR 10. Re-export each new checkpoint with
`--language-model-arch <ArchName>` to avoid vLLM RecursionError.

```bash
# 1) Export (one-time per checkpoint):
PY=/opt/aurora/26.26.0/frameworks/aurora_frameworks-2025.3.1/bin/python
$PY -m src.vllm_plugin.checkpoint_export \
    --checkpoint outputs/<RUN>/<DATE>/checkpoints/step_<N>/model.safetensors \
    --backbone allenai/OLMo-1B-0724-hf \
    --image-encoder google/siglip2-base-patch16-224 \
    --language-model-arch OlmoForCausalLM \
    --out exported/<NAME>

# 2) Parity (HF demo vs vLLM, both runs):
bash tools/_vllm_parity_runner.sh \
    --checkpoint outputs/<RUN>/<DATE>/checkpoints/step_<N> \
    --vllm-model exported/<NAME> \
    --image test_images/cat.jpg \
    --mode both
```

Pass criteria:
- `[demo] missing=0 unexpected=0 projector_missing=0`
- `[demo] >>> '...'` and `[vllm] >>> '...'` print **identical** strings.

### PR 9 API server (manual, ~2 min after server health)

Requires PR 9.

```bash
# On a compute node, after env unpack + DAOS mount + venv activate:
export PRISM_CHECKPOINT=outputs/<RUN>/<DATE>/checkpoints/step_<N>
export PORT=8765
python -m src.api.server &
SERVER_PID=$!

# Wait for health:
for i in $(seq 1 60); do
    curl -sf http://localhost:$PORT/v1/models > /dev/null && break
    sleep 5
done

# Text-only:
curl -X POST http://localhost:$PORT/v1/chat/completions \
    -H 'Content-Type: application/json' \
    -d '{"model":"prism-mm-v1","messages":[{"role":"user","content":"What is 2+2?"}],"max_tokens":20}'

# Image:
B64=$(base64 -w0 test_images/cat.jpg)
cat > /tmp/req.json <<EOF
{"model":"prism-mm-v1","messages":[{"role":"user","content":[
  {"type":"text","text":"Describe this:"},
  {"type":"image_url","image_url":{"url":"data:image/jpeg;base64,$B64"}}
]}],"max_tokens":30}
EOF
curl -X POST http://localhost:$PORT/v1/chat/completions \
    -H 'Content-Type: application/json' -d @/tmp/req.json

kill $SERVER_PID
```

Pass: both endpoints return OpenAI-shaped JSON with non-empty `content`.

---

## Throughput history

Per-PR throughput numbers, validated production configs, and the
DeepSpeed ZeRO-2/3 results live in
[`docs/results/scaling_study.md`](../../docs/results/scaling_study.md). Phase 2 PR
results were recorded in the 2026-05-14 migration plan (§7), since retired;
see git history.
