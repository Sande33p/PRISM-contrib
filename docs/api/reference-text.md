# Text-reference protocol (version 1)

`src.backends.text_contracts` and `src.backends.u2_reference` define an
injected-runtime boundary for text-reference experiments. The implementation
validates token requests, collects committed output and instrumented work,
and retains terminal receipts for replay within one adapter instance.

This is an offline-tested protocol, not a production U2 driver. It contains
no model loader, tokenizer renderer, weights or upstream sampler. The global
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
settings to the driver; it does not implement their numerical semantics.

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

The snippet illustrates dependency injection; this repository does not ship
the `runtime` driver. `create_app()` and the default module-level application
retain legacy model startup. Offline tests disable that startup explicitly
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

## Verification boundary

Offline tests use fixture runtimes to exercise admission, commits, terminal
evidence, actual instrumented operation counts, failure accounting,
cancellation, concurrent duplicate requests, replay and legacy-route
retention. Fixture identities and simulated operations are not model
execution or artifact verification. A real U2 driver, exact tokenizer/prompt
rendering, direct-upstream parity, durable receipts and transport cancellation
remain separate implementation and qualification work.
