# Text-reference protocol (version 1)

`src.backends.text_contracts` and `src.backends.u2_reference` define an
injected-runtime boundary for text-reference experiments. The implementation
validates token requests, collects committed output and instrumented work,
and retains terminal receipts for replay within one adapter instance.

`src.backends.u2_text_runtime.U2TextRuntime` now implements the pinned greedy
text loop with an explicitly supplied local model. The renderer and metadata
preflight are separate, dependency-free modules. A strict offline loader is
available for explicit use; weights are not provided, and checkpoint-backed
execution remains unqualified. The global
`u2_reference` and `u2_native` selectors remain planned and raise
`NotImplementedError`; selecting them in `ModelConfig` does not activate this
adapter. Existing local/HF model construction, checkpoint keys and chat
behavior are unchanged. See the [backend selection contract](model.md).

## Request and identity

`TextRequest` is an immutable version-1 request with:

| Field | Meaning |
|---|---|
| `request_id` | Nonempty identifier, at most 128 characters; the replay key within one adapter instance |
| `prompt_ids` | Nonempty, immutable sequence of pretokenized prompt IDs; at most 4096 entries |
| `tokenizer_digest` | Must match the injected runtime identity |
| `max_new_tokens` | Committed output limit, default 128 |
| `max_model_calls` | Instrumented call-start limit, default 128 |
| `block_size`, `steps_per_block` | Driver settings, each defaulting to 32 |
| `seed` | Recorded unsigned 64-bit seed; no global RNG is modified by the adapter |
| `contract_version` | Integer `1` |

Output/call/block/step limits must each be positive integers no greater than
128. Booleans, numeric strings, unsupported versions and malformed IDs are
rejected rather than coerced. These admission limits are not a model context
window, hardware-fit measurement or speed claim. The protocol passes sampler
settings to the driver; the concrete U2 loop below defines their semantics.

`RuntimeIdentity` declares `runtime_id`, `source_revision`, `tokenizer_digest`,
`vocab_size`, `eoa_token_id` and `mask_token_id`. The adapter snapshots this
identity and validates request token ranges against it before dispatch.
An identity supplied by a driver is not proof that its artifacts were
downloaded, hashed, licensed or qualified. Model-backed use still requires
independent artifact validation and direct-upstream parity.

## Driver protocol and committed output

A driver implements `generate(request, control)` and returns an explicit
terminal reason. Every forward must pass through
`control.run_model_call(processed_positions, operation)`; stable output-only
blocks pass through `control.commit(token_ids)`. The adapter does not inspect
or return tentative predictions. It never assumes that a runtime result is
an HF prompt-plus-output sequence or slices tokens according to that guess.

The control checks the call budget and cooperative cancellation around each
instrumented operation. It counts starts, records supplied position counts
for operations that return, and rejects invalid commits. Output IDs must be
in vocabulary, must not contain unresolved mask IDs, and must fit the output
limit. Committed prefixes cannot be revised. A final EOA token is observed
before stripping it from the returned IDs; output after EOA is invalid.

The injected driver is trusted to route all work through this interface and
report its position counts faithfully. These controls are not a sandbox for
arbitrary Python, do not instrument hidden driver work, and do not establish
actual U2 sampling or accelerator performance.

`TextResult` includes the request ID, identity, seed, contract version,
`output_ids`, cumulative `committed_block_ends`, `eoa_observed`, a usage receipt
and the terminal reason:

| `finish_reason` | Required evidence |
|---|---|
| `eoa` | EOA was observed in a validated commit before stripping |
| `length` | The committed output reached `max_new_tokens`; this is truncation, not an inferred stop |
| `budget` | Control acknowledged the call limit |
| `cancelled` | Control acknowledged a cancellation request |
| `error` | Runtime exception, invalid event or missing/inconsistent terminal evidence |

Earlier committed output is retained on failure. Errors have stable public
codes; raw runtime exception messages are not returned. Missing terminal
evidence never becomes a successful `stop`, and the adapter performs no
automatic retry or fallback.

## Usage and cancellation

The usage receipt reports prompt and committed completion counts plus
`observed_model_calls` and `observed_processed_positions`. A zero-reveal
forward still counts as a call; repeated processing still counts positions.
Counts are not inferred from requested steps or generated token length.

`control.check_can_continue()` checks live state, cancellation and the call
budget before the runtime allocates each block's tensors and attention mask.
It consumes no calls or positions and reserves no future operation;
`run_model_call` still validates and rechecks admission for every actual forward.
Zero-quota steps execute and validate their forward but skip sampling, including
the float64 softmax and confidence ranking.

`model_calls` and `processed_positions` are aggregate totals only when
`complete` is true. Runtime/protocol errors or interrupted operations leave
the totals unknown (`null` in JSON), retaining the observed lower bounds.
No cost, device-time or throughput estimate is fabricated.

`adapter.cancel(request_id)` records a cooperative signal for a live request.
Its return value acknowledges the signal, not runtime termination. Control
prevents subsequent work and observes cancellation after an in-flight call
returns; it cannot interrupt that call or prove device resources were
released. A signal recorded before atomic receipt finalization wins, including
after an EOA commit; the cancelled receipt retains committed output and EOA
evidence. A completed receipt remains completed, and cancellation after
finalization returns false. An HTTP waiter's state does not determine this
outcome.

## Replay scope

The adapter records admission before dispatch and retains a terminal receipt
before returning it. Runtime execution is serialized within the instance.

- Same request ID and identical payload: return the saved receipt without
  calling the runtime again, including after failures.
- Same ID with a changed payload: `RequestConflictError`.
- Identical ID still in progress: `RequestInProgressError`.
- Full ledger: `LedgerCapacityError`; no silent eviction or redispatch.

`max_entries` defaults to 128. The ledger is bounded and process-local, with
no persistent storage, automatic expiry or retry controller. Restarting or
creating another adapter instance loses this replay guarantee. It is not
durable exactly-once execution across processes or hosts.

## Versioned HTTP route

`POST /v1/backends/u2_reference/generate` accepts the request fields above,
with `prompt_ids` represented as a JSON array. Extra fields and unsupported
types are rejected. There is no conversation renderer, text decoding, media,
tool execution, streaming or constrained-JSON generation in this route.

The hosting application must explicitly provide an adapter:

```python
from src.api.server import create_app
from src.backends.u2_reference import U2ReferenceAdapter

# runtime is a separately supplied and qualified driver implementing the protocol.
adapter = U2ReferenceAdapter(runtime)
app = create_app(reference_adapter=adapter, load_legacy_model_on_startup=False)
```

The snippet illustrates dependency injection; `runtime` can be a separately
constructed `U2TextRuntime` as described below. `create_app()` and the default
module-level application retain legacy model startup. Offline tests disable that startup explicitly
and inject deterministic fixtures. The adapter modules need only the Python
standard library; importing the existing API server still needs its normal
serving/platform dependencies.

An unconfigured reference route returns HTTP 503 with
`reference_adapter_unconfigured`, without choosing or loading another model.
Validation errors use 422, conflicting/in-progress IDs use 409, and ledger
capacity uses 503. Accepted terminal outcomes return the canonical receipt,
including cancellation, budget and error outcomes, without converting their
finish reasons to OpenAI `stop`.

This HTTP route does not signal cancellation on client disconnect or timeout.
The synchronous handler runs in the server threadpool; cancellation is an
explicit adapter operation in this slice. A disconnected client must not
assume computation stopped. A later request with the same ID/payload can
retrieve a retained terminal receipt from the same adapter instance.

## Renderer and tokenizer artifact identity

`render_text_prompt(tokenizer, system_prompt=..., user_text=...,
tokenizer_artifacts=...)` renders exactly
`<system>{system_prompt}</system><user>{user_text}</user>`. It preserves
whitespace, Unicode, explicit empty text and the pinned tokenizer call:
`truncation=True, max_length=4096, padding=False`. Truncation can remove the
closing user tag. Reserved markers, media/history inputs, malformed IDs and
incompatible special-token encodings fail. There is no implicit HF chat
template, default system text, answer prefix or invented think-token mapping.
The immutable result supplies `prompt_ids`, `tokenizer_digest` and
`renderer_revision`. Pass its prefix and digest directly to `TextRequest`;
the prefix excludes BOA. Text decoding is outside the driver.

Both renderer and preflight use `u2_text_renderer.tokenizer_digest`:

1. Supply nonempty exact bytes for `tokenizer.json` and
   `tokenizer_config.json`, plus any present `special_tokens_map.json` and
   `added_tokens.json`. These four basenames are the admitted set; the renderer
   rejects unknown names instead of omitting supplied bytes.
2. Compute each raw file's lowercase bare SHA256, sort `(filename, file_hash)`
   pairs, then serialize with `json.dumps(pairs, ensure_ascii=True,
   separators=(",", ":")).encode()`.
3. Return `"sha256:" + SHA256(b"u2-tokenizer-artifacts-v1\0" + serialized).hexdigest()`.

Ordering is irrelevant; file names, contents, configuration and optional files
affect identity. Preflight's earlier bare aggregate identity is incompatible:
reinspect the same snapshot and use the prefixed identity in all three
components. Per-file `FileEvidence.sha256` and requirement `artifact_hashes`
remain bare 64-hex SHA256. Both boundaries validate the complete pinned
17-token contract. The renderer also checks atomic encoding.

`preflight_u2(source_root, checkpoint_root, requirements)` inspects local
metadata and hashes without importing checkpoint Python or loading tensors.
Unsafe paths/escaping symlinks, empty files, malformed JSON, incorrect source
hashes, missing shards and incompatible token/config/layout identities fail
metadata admission. Tokenizer bytes are bounded to 64 MiB per file and checked
against their just-observed file hash. Use a stable local snapshot; concurrent
file or object mutation is not supported.

### Prepare the source manifest

A clean checkout of the pinned upstream revision does **not** include
`source-manifest.json`. Before calling preflight, prepare
`source_root/source-manifest.json` locally. It must be a JSON list of exactly
six entries, one for each unique relative path in `SOURCE_HASHES` from
[`src/backends/u2_preflight.py`](../../src/backends/u2_preflight.py):
`infer_mmu_sft.py`, `generators/text_understanding_generator.py`,
`utils/generation_utils.py`, `utils/tokenizer_contract.py`, `LICENSE`, and
`README.md`.

Each entry has `path`, `sha256`, and `revision` fields. Its `path` must match
the corresponding `SOURCE_HASHES` key exactly, `sha256` must equal that key's
lowercase 64-character SHA256 value, and `revision` must be the full
`SOURCE_REVISION`, `70f21c1df8fa26e514df26ba31bc061be8d3d018`.
Do not add, omit or duplicate paths. Preflight verifies both this mapping and
the actual source file bytes; declaring a revision alone is not verification.

From the PRISM repository root, this example verifies all six local files
before writing the manifest (replace the source directory argument):

```bash
python - /path/to/local/InternLumina-U2 <<'PY'
import hashlib
import json
import sys
from pathlib import Path

from src.backends.u2_preflight import SOURCE_HASHES, SOURCE_REVISION

source_root = Path(sys.argv[1]).resolve(strict=True)
entries = []
for name, expected in sorted(SOURCE_HASHES.items()):
    path = (source_root / name).resolve(strict=True)
    if not path.is_relative_to(source_root):
        raise SystemExit(f"Source path escapes root: {name}")
    observed = hashlib.sha256(path.read_bytes()).hexdigest()
    if observed != expected:
        raise SystemExit(f"Source SHA256 mismatch: {name}")
    entries.append({"path": name, "sha256": observed, "revision": SOURCE_REVISION})
with (source_root / "source-manifest.json").open("x", encoding="utf-8") as manifest:
    json.dump(entries, manifest, indent=2)
    manifest.write("\n")
PY
```

The example refuses to overwrite an existing manifest. Supplemental provenance
records for additional source files are separate, optional evidence; preflight
does not consume them. Do not append supplemental entries to this six-entry
manifest.

`report.require_metadata()` does not qualify model execution:
`execution_qualified` remains `False`, and `split_candidate` describes index
keys only. A matching digest does not prove that an injected tokenizer or
model was built from those bytes. The caller owns that binding, strict tensor
key/shape/dtype validation, split-layout verification and later model parity.

## Explicit local model and source-backed loop

The source pin is InternLM/InternLumina-U2
`70f21c1df8fa26e514df26ba31bc061be8d3d018`. The loop and text metadata mask are
adapted from its Apache-2.0
[text generator](https://github.com/InternLM/InternLumina-U2/blob/70f21c1df8fa26e514df26ba31bc061be8d3d018/generators/text_understanding_generator.py)
and [sampling helpers](https://github.com/InternLM/InternLumina-U2/blob/70f21c1df8fa26e514df26ba31bc061be8d3d018/utils/generation_utils.py).
The pinned media-layout helper defines eight replicated channels; the model's
`infer_mmu` branch returns a tensor for noisy type-3 rows directly.

```python
from src.backends.u2_reference import U2ReferenceAdapter
from src.backends.u2_text_runtime import U2TextRuntime

# Explicit, caller-supplied LocalTextModel and inspected U2PreflightReport.
runtime = U2TextRuntime(local_model, preflight=preflight_report)
adapter = U2ReferenceAdapter(runtime)
result = adapter.generate(request)
```

The local model implements `LocalTextModel`: an immutable `LocalModelSpec`,
`training`, `eval()` and a callable matching the pinned forward signature.
`LocalModelSpec` declares `RuntimeIdentity`, the full special-token map,
`torch.dtype`, `torch.device`, `max_position_embeddings`, `num_codebooks=8`
and `layout="split"`. Construction checks these declarations against the
preflight report and calls `eval()`; every forward runs in Torch inference
mode. Subsequent declaration changes or training mode fail. These checks
cannot detect a dishonest binding or all internal model mutations. Synthetic
models must be explicitly identified in `runtime_id`; they are never a fallback.

Supported declared dtypes are float32, float16 and bfloat16. CPU devices must
be unindexed; CUDA and XPU declarations must specify an explicit index (`cuda:N`
or `xpu:N`). Other devices are rejected. This validates metadata without probing
availability or falling back to another device. Only synthetic CPU execution has
been tested; allowing an accelerator
declaration is not accelerator qualification or execution authorization.

Each request reserves `P + 1 + max_new_tokens` against inspected capacity.
The extra position is exactly one BOA `157165`; EOA is `157166` and mask is
`156895`. A token-level prefix already containing BOA, EOA or mask fails
before a forward. Every generated ID other than terminal EOA is preserved,
including other specials; an unresolved mask rejects the entire pending block.

For current sequence length L, each actual forward is independently invoked
through `control.run_model_call(L, operation)`:

```python
model(input_ids, type_position_ids, attention_mask=mask, infer_mmu=True,
      is_clean=is_clean, position_ids=positions)
```

Inputs are integer `[1,L,8]`, with each token replicated across channels.
Types, clean flags and positions are integer `[1,L]`. The additive attention
mask is `[1,1,L,L]` in the model dtype: zero permits visibility and
`torch.finfo(dtype).min` denies it. Context is type 1/turn 0/block -1/clean 1;
BOA is type 1/turn 1/block 0/clean 0. Generated positions have type 3/turn 1,
block `j // block_size`, and positions `P+1+j`. Previous blocks become clean;
the entire current block stays noisy even after individual reveals. BOA stays
noisy. No future block or KV-cache shortcut is used.

The model must return dense finite logits `[current_block_length,vocab_size]`
for all noisy type-3 rows, including revealed rows, on the declared device and
dtype. HF objects, tuples, wrong shapes/devices/dtypes and nonfinite values
fail explicitly. Validation follows the instrumented return, so invalid
returned logits still count that operation's L positions. A raising operation
counts its start but contributes no completed-operation positions; errors
leave aggregate totals unknown.

Sampling is fixed at temperature zero, cfg zero and low-confidence revealing.
Candidate IDs use raw-logit argmax; reveal confidence is gathered softmax
probability computed in float64. Revealed tokens never change. Ties use Torch
argmax/topk on the executing runtime, with no portable tied-index ordering
promise. For block length n and t steps, quotas are `n//t + 1` for the first
`n%t` steps and `n//t` thereafter. All t forwards execute, even with zero quota.
The last short block uses its actual length. The recorded seed does not cause
random draws or global RNG changes in this greedy loop.

EOA is checked only after the complete block schedule. The driver commits the
prefix through the first EOA and returns `eoa`; control observes and strips
it. Otherwise it commits the whole block. Budget/cancellation drops the
pending block and keeps previous commits. Short output never substitutes for
EOA evidence. The existing control owns stop/error precedence and replay.

## Verification boundary

`tests/test_u2_text_runtime.py` runs this actual loop through
`U2ReferenceAdapter`, with synthetic logits and literal masks, trajectories,
quotas, commits, EOA and usage expectations frozen before implementation.
It covers cancellation with synchronization events, invalid outputs, call
caps, request admission, replay and bounded ledger behavior. Renderer and
preflight tests use tiny local artifact snapshots, including a cross-component
identity regression. Existing adapter/API/global-selector tests remain
regression checks for the unchanged interfaces.

This is deterministic loop/contract evidence. No checkpoint or actual tokenizer
assets were loaded, and no real-model parity, quality, performance, memory-fit
or accelerator result is established. Exact artifact-to-object binding,
identical-checkpoint direct-upstream parity, durable receipts, transport
cancellation, mid-kernel interruption and device-release guarantees remain
separate qualification work. No global backend is enabled by these tests.


## Strict local loading and private comparison

`src.backends.u2_checkpoint` verifies independently admitted immutable manifests
before binding model objects. Checkpoint entries map only through an explicit
`nvidia/` prefix to a checkpoint root named `nvidia`. It checks original Git-blob
SHA1 identities for small files and LFS SHA256 identities for large files, and
records a fresh SHA256 for every consumed file. A prior verification receipt is
additional evidence, not a replacement trust input. Missing, extra, duplicate,
unsafe and symlink inputs fail closed.

The loader validates the complete index and safetensors headers, split experts,
eight channels, every constructed state key, shape and dtype, and read-back tensor
bytes. Each shard is read into a bounded owned byte buffer, verified against its
admitted size/SHA256, then deserialized from that same immutable buffer. A pathname
swap cannot substitute unverified parameter bytes. Peak host storage includes the
model, the largest serialized shard (approximately 5 GB), deserializer tensors and
per-tensor hashing copies; production memory fit is unmeasured. Allocation failures
propagate without returning a partial model. No implicit dtype conversion is admitted.
Checkpoint-bundled Python is verified as an artifact but is never imported as the
pinned Git oracle. The six-entry preflight manifest remains separate from the
complete source/imported-module manifest. The optional `source_manifest` keyword
on `preflight_u2` permits a separate six-entry file without copying it into the
source checkout. Inputs and their ancestors must be operator-controlled during a
run; integrity checks are not a filesystem sandbox.

The bound model checks parameters, all buffers (including nonpersistent rotary
state), module registration, execution attributes, callable identities/code/defaults,
partial arguments, class helpers, hooks and configuration. Full byte checks bracket
comparison paths. Ordinary mutation fails before another forward; arbitrary hostile
in-process Python and concurrent mutation are outside this boundary.

`U2PreflightReport.vocab_size` and `RuntimeIdentity.vocab_size` describe the model
logit domain. `tokenizer_vocab_size` records serialized tokenizer cardinality;
`BoundTextTokenizer.vocab_size` records the actual constructed tokenizer. Prepared
metadata reports 157177 serialized IDs versus 157184 model positions. The seven
additional positions are not established padding. Strict loading and observed
output shapes are separate qualification steps. All logits and selected IDs are
preserved. Selecting a position absent from the tokenizer produces an explicit
incomplete decoding result, never invented IDs or hidden logit masking.
The strict loader sets `LocalModelSpec.prompt_vocab_size` from its actual bound
tokenizer and checks it against preflight. External prompt IDs must be below that
limit before the first forward. This optional field defaults to `None` for existing
injected CPU/CUDA fixtures, preserving their declared model-domain admission.
It never restricts generated IDs or the 157184-column output domain.

`tools/parity/u2_text.py` accepts the inherited launcher's `--source-root`,
`--checkpoint-root`, `--output-dir`, `--acceptance-file`, `--asset-manifest`,
`--source-manifest`, `--device` and `--dtype`. Its optional `--run-context` names
private JSON containing `acceptance_document_revision`,
`oracle_instrumentation_diff_and_review`, `modules_in_order`, `authorized_job_id`,
`execution_allocation`, `imported_source_manifest` and
`imported_source_manifest_sha256`. Supply reviewed evidence and original trust
identities; these fields do not grant execution authority. The runner corroborates
job identity and ordered loaded modules with `PBS_JOBID` and `LOADEDMODULES`.
Missing context or observations stay incomplete; mismatches fail. The dry-run
launcher forwards the optional `--run-context` path with the same shell quoting and
input/output separation checks as other inputs. Local validation records its hash;
the emitted script checks that hash when available. Raw context and paths stay in
private plans/results and are omitted from sanitized collection.

The comparison invokes actual pinned `generate_text_bd`, sharing the bound model,
tokenizer, device/dtype and frozen settings with PRISM. Read-only tracing observes
sampler locals and every forward entry/return, including zero-transfer steps.
Full logits and masks remain in private binary sidecars with shapes, dtypes and
hashes. Both frozen path orders independently reset RNG. Partial failures and
missing EOA/length coverage remain visible. Numeric summaries use the frozen
per-scalar thresholds. Raw comparison status is distinct from independent
Evaluation acceptance; the driver never emits `independently-validated`.

**Restricted import profile:** `split-only-v1` replaces exactly the two pinned
imports of `fused_moe_forward` and `get_parallel_state` with explicit unsupported
guards. Only the excluded packed expert branch calls these symbols. The original
model bytes and every computation body remain unchanged; original and compiled
SHA256s plus the exact import replacement are recorded. The actual split/eight-channel
configuration and constructed experts are checked. Genuine VeOmni utilities,
logging, Torch, Transformers and tokenbridge/timm components remain in use. Optional
`veomni_patch` and Liger packages on the effective path or already loaded are rejected
before relevant imports; stale pyc files never supply pinned code.

A tiny pinned-source synthetic-weight CPU test exercises genuine model construction,
strict loading and BF16 forward using an existing timm 1.0.22 environment. This is
source-path evidence only. Genuine timm is missing from the recorded Aurora runtime;
dependency preparation is separately authorized work, and no substitute is supplied.
Actual XPU execution, real tokenizer/checkpoint binding, memory fit and independent
exact-revision adaptation review remain pending.
The pinned loader's BF16/SDPA semantics are retained; optional diagnostic dtype
conversion is not implemented. Global U2 selectors remain unavailable.
