"""Injected text-runtime protocol with committed output and bounded replay.

No pretrained driver, tokenizer rendering, model loading, or global backend
activation is supplied. A compliant driver instruments every forward through
ExecutionControl; arbitrary injected Python is not a security boundary.
"""

import threading
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Protocol, TypeVar, cast

from .text_contracts import (
    FinishReason,
    RuntimeIdentity,
    TextRequest,
    TextResult,
    UsageReceipt,
    _integer,
    _text,
    _token_ids,
)

_T = TypeVar("_T")


class RequestConflictError(ValueError):
    """The request ID is already bound to a different immutable payload."""


class RequestInProgressError(RuntimeError):
    """The identical request is already admitted and has no terminal receipt."""


class LedgerCapacityError(RuntimeError):
    """The process-local ledger is full; no request IDs are silently evicted."""


class _Stopped(Exception):
    pass


class _ProtocolError(Exception):
    pass


class ExecutionControl:
    """Cooperative call budget, cancellation, committed blocks and counters.

    A model call is counted when its operation starts; processed positions are
    counted only after it returns. Cancellation cannot interrupt that operation
    or prove device release. Commits contain output-only IDs, never prompt IDs
    or unstable previews. The driver must honor this protocol for every call.
    """

    def __init__(
        self, request: TextRequest, identity: RuntimeIdentity, cancelled: threading.Event
    ) -> None:
        self._request = request
        self._identity = identity
        self._cancelled = cancelled
        self._lock = threading.RLock()
        self._calls = 0
        self._positions = 0
        self._output: list[int] = []
        self._block_ends: list[int] = []
        self._eoa = False
        self._stop: str | None = None
        self._error: str | None = None
        self._in_flight = False
        self._finalized = False

    @property
    def cancelled(self) -> bool:
        """Return the cooperative signal, not proof that execution has stopped."""
        return self._cancelled.is_set()

    def _fail(self, code: str) -> None:
        with self._lock:
            if self._error is None:
                self._error = code

    def _check_live(self) -> None:
        if self._finalized:
            raise _ProtocolError("already_finalized")
        if self._error is not None:
            raise _ProtocolError(self._error)
        if self._stop is not None:
            raise _Stopped(self._stop)
        if self.cancelled:
            self._stop = "cancelled"
            raise _Stopped("cancelled")

    def _check_budget(self) -> None:
        if self._calls >= self._request.max_model_calls:
            self._stop = "budget"
            raise _Stopped("budget")

    def check_can_continue(self) -> None:
        """Check live state and call budget before preparing another block.

        No call or positions are consumed or reserved. A successful check does
        not guarantee later admission: run_model_call still validates arguments
        and rechecks live state and budget under the lock for every operation.
        """
        with self._lock:
            self._check_live()
            self._check_budget()

    def run_model_call(self, processed_positions: int, operation: Callable[[], _T]) -> _T:
        """Execute and count one actual driver operation, enforcing admission.

        Exceptions leave aggregate totals unknown. A call returning no revealed
        tokens still counts. Future calls stop after cancellation or the budget.
        """
        with self._lock:
            self._check_live()
            try:
                _integer(processed_positions, "processed_positions", 1)
                if not callable(operation) or self._in_flight or self._eoa:
                    raise ValueError("invalid model call")
            except ValueError as exc:
                self._fail("invalid_model_call")
                raise _ProtocolError("invalid_model_call") from exc
            self._check_budget()
            self._calls += 1
            self._in_flight = True
        try:
            result = operation()
        except BaseException:
            with self._lock:
                self._in_flight = False
                self._fail("model_call_failed")
            raise
        with self._lock:
            self._in_flight = False
            self._positions += processed_positions
            self._check_live()
        return result

    def commit(self, token_ids: Iterable[int]) -> None:
        """Append a stable output block, observing and stripping only final EOA.

        Empty commits are harmless. Nonempty output adds a cumulative block end;
        EOA-only commits do not add an empty block. Invalid events are sticky.
        """
        with self._lock:
            self._check_live()
            try:
                ids = _token_ids(token_ids, "committed IDs", 129)
                # Iteration may execute driver code that stops or cancels work.
                self._check_live()
                if self._in_flight or self._eoa:
                    raise ValueError("commit after EOA or during model call")
                if any(token >= self._identity.vocab_size for token in ids):
                    raise ValueError("out-of-vocabulary committed ID")
                if self._identity.mask_token_id in ids:
                    raise ValueError("mask ID cannot be committed")
                terminal = bool(ids and ids[-1] == self._identity.eoa_token_id)
                output = ids[:-1] if terminal else ids
                if self._identity.eoa_token_id in output:
                    raise ValueError("EOA must occur once at the end of a commit")
                if len(self._output) + len(output) > self._request.max_new_tokens:
                    raise ValueError("output limit exceeded")
            except _Stopped:
                raise
            except Exception as exc:
                self._fail("invalid_commit")
                raise _ProtocolError("invalid_commit") from exc
            except BaseException:
                self._fail("invalid_commit")
                raise
            self._output.extend(output)
            if output:
                self._block_ends.append(len(self._output))
            self._eoa = terminal

    def _finish(self, terminal: object) -> TextResult:
        with self._lock:
            if self._in_flight:
                self._fail("unfinished_model_call")
            if self._error is None:
                if self.cancelled:
                    self._stop = "cancelled"
                if self._stop is not None:
                    reason = self._stop
                elif type(terminal) is not str:
                    self._fail("invalid_terminal_reason")
                    reason = "error"
                elif terminal == "eoa" and self._eoa:
                    reason = "eoa"
                elif (
                    terminal == "length"
                    and len(self._output) == self._request.max_new_tokens
                    and not self._eoa
                ):
                    reason = "length"
                elif (
                    terminal == "budget"
                    and self._calls == self._request.max_model_calls
                    and not self._eoa
                ):
                    reason = "budget"
                elif terminal == "error":
                    self._fail("runtime_reported_error")
                    reason = "error"
                else:
                    self._fail("invalid_terminal_reason")
                    reason = "error"
            else:
                reason = "error"
            self._finalized = True
            complete = self._error is None
            return TextResult(
                contract_version=self._request.contract_version,
                request_id=self._request.request_id,
                identity=self._identity,
                output_ids=tuple(self._output),
                committed_block_ends=tuple(self._block_ends),
                finish_reason=cast(FinishReason, reason),
                usage=UsageReceipt(
                    prompt_tokens=len(self._request.prompt_ids),
                    completion_tokens=len(self._output),
                    observed_model_calls=self._calls,
                    observed_processed_positions=self._positions,
                    model_calls=self._calls if complete else None,
                    processed_positions=self._positions if complete else None,
                    complete=complete,
                ),
                seed=self._request.seed,
                eoa_observed=self._eoa,
                error_code=self._error,
            )


class ReferenceRuntime(Protocol):
    """Driver contract only; implementors supply and qualify their own runtime."""

    identity: RuntimeIdentity

    def generate(self, request: TextRequest, control: ExecutionControl) -> str:
        """Use control for every model call/commit and return an explicit terminal."""
        ...


@dataclass
class _Record:
    request: TextRequest
    cancelled: threading.Event
    result: TextResult | None = None


class U2ReferenceAdapter:
    """Serialize an injected runtime and cache bounded process-local receipts.

    Identical accepted requests replay exact receipts, including failures. IDs
    are never evicted: capacity exhaustion rejects new work. This is nondurable
    and does not deduplicate across restarts or independent adapter instances.
    """

    def __init__(self, runtime: ReferenceRuntime, *, max_entries: int = 128) -> None:
        _integer(max_entries, "max_entries", 1)
        identity = runtime.identity
        if not isinstance(identity, RuntimeIdentity) or not callable(runtime.generate):
            raise ValueError("runtime must provide canonical identity and generate")
        self._identity = identity
        self._runtime = runtime
        self._max_entries = max_entries
        self._records: dict[str, _Record] = {}
        self._state_lock = threading.Lock()
        self._runtime_lock = threading.Lock()

    @property
    def identity(self) -> RuntimeIdentity:
        """Return the immutable driver identity captured once at construction."""
        return self._identity

    def generate(self, request: TextRequest) -> TextResult:
        """Admit one immutable request or replay its cached terminal receipt."""
        if not isinstance(request, TextRequest):
            raise ValueError("request must be a TextRequest")
        with self._state_lock:
            previous = self._records.get(request.request_id)
            if previous is not None:
                if previous.request != request:
                    raise RequestConflictError("request ID already has a different payload")
                if previous.result is None:
                    raise RequestInProgressError("request is already in progress")
                return previous.result
            if request.tokenizer_digest != self.identity.tokenizer_digest:
                raise ValueError("request tokenizer_digest does not match runtime identity")
            if any(token >= self.identity.vocab_size for token in request.prompt_ids):
                raise ValueError("prompt_ids contain out-of-vocabulary tokens")
            if len(self._records) >= self._max_entries:
                raise LedgerCapacityError("reference request ledger is full")
            record = _Record(request, threading.Event())
            self._records[request.request_id] = record
        control = ExecutionControl(request, self.identity, record.cancelled)
        terminal: object = None
        try:
            with self._runtime_lock:
                if not record.cancelled.is_set():
                    terminal = self._runtime.generate(request, control)
        except _Stopped:
            pass
        except _ProtocolError:
            # The failing control operation already recorded a sticky code.
            pass
        except Exception:
            control._fail("runtime_error")
        except BaseException:
            control._fail("runtime_interrupted")
            with self._state_lock:
                record.result = control._finish(terminal)
            raise
        with self._state_lock:
            record.result = control._finish(terminal)
            return record.result

    def cancel(self, request_id: str) -> bool:
        """Signal a live request once; false means no new signal was recorded.

        A true result is not acknowledgement of termination or device release.
        Receipt finalization and cancellation are atomic with respect to each
        other, including requests waiting for the serialized runtime.
        """
        _text(request_id, "request_id", 128)
        with self._state_lock:
            record = self._records.get(request_id)
            if record is None or record.result is not None or record.cancelled.is_set():
                return False
            record.cancelled.set()
            return True
