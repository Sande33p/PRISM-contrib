"""Offline acceptance for the injected text protocol, not pretrained execution."""

import dataclasses
import subprocess
import sys
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from src.backends import describe_backend, resolve_backend
from src.backends.text_contracts import RuntimeIdentity, TextRequest, UsageReceipt
from src.backends.u2_reference import (
    LedgerCapacityError,
    RequestConflictError,
    RequestInProgressError,
    U2ReferenceAdapter,
)


def identity(**changes):
    fields = dict(
        runtime_id="offline-fixture",
        source_revision="fixture-revision",
        tokenizer_digest="fixture-tokenizer",
        vocab_size=100,
        eoa_token_id=99,
        mask_token_id=98,
    )
    fields.update(changes)
    return RuntimeIdentity(**fields)


def request(**changes):
    fields = dict(
        request_id="request-one", prompt_ids=(1, 2, 3), tokenizer_digest="fixture-tokenizer"
    )
    fields.update(changes)
    return TextRequest(**fields)


class Runtime:
    def __init__(self, generate=None):
        self.identity = identity()
        self.calls = []
        self.controls = []
        self.operation = generate or self.success

    def generate(self, value, control):
        self.calls.append(value)
        self.controls.append(control)
        return self.operation(value, control)

    @staticmethod
    def success(value, control):
        output = control.run_model_call(len(value.prompt_ids), lambda: (7, 8))
        control.commit(output)
        control.commit((99,))
        return "eoa"


class ContractValidationTests(unittest.TestCase):
    def test_identity_rejects_invalid_declarations(self):
        for changes in (
            {"runtime_id": ""},
            {"source_revision": " "},
            {"tokenizer_digest": "a" * 257},
            {"vocab_size": True},
            {"vocab_size": 1},
            {"eoa_token_id": False},
            {"eoa_token_id": 100},
            {"mask_token_id": -1},
            {"mask_token_id": 99},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                identity(**changes)

    def test_request_copies_caller_tokens_and_is_frozen(self):
        tokens = [1, 2]
        value = request(prompt_ids=tokens)
        tokens.append(3)
        self.assertEqual(value.prompt_ids, (1, 2))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            value.seed = 4
        self.assertEqual(request(seed=2**64 - 1).seed, 2**64 - 1)

    def test_request_rejects_invalid_values_without_coercion(self):
        cases = [
            {"request_id": ""},
            {"request_id": "x" * 129},
            {"tokenizer_digest": "x" * 257},
            {"prompt_ids": ()},
            {"prompt_ids": "1"},
            {"prompt_ids": {1: 2}},
            {"prompt_ids": None},
            {"prompt_ids": (True,)},
            {"prompt_ids": (1.0,)},
            {"prompt_ids": (-1,)},
            {"prompt_ids": (1,) * 4097},
            {"seed": -1},
            {"seed": 2**64},
            {"seed": True},
            {"contract_version": True},
            {"contract_version": "1"},
            {"contract_version": 2},
        ]
        for field in ("max_new_tokens", "max_model_calls", "block_size", "steps_per_block"):
            cases.extend({field: invalid} for invalid in (False, "1", 1.0, 0, 129))
        for changes in cases:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                request(**changes)

    def test_prompt_iterable_is_bounded_during_materialization(self):
        seen = []

        def tokens():
            for index in range(5000):
                seen.append(index)
                yield 1

        with self.assertRaises(ValueError):
            request(prompt_ids=tokens())
        self.assertEqual(len(seen), 4097)

    def test_usage_receipt_never_labels_unknown_aggregates_complete(self):
        good = dict(
            prompt_tokens=3,
            completion_tokens=0,
            observed_model_calls=2,
            observed_processed_positions=4,
            model_calls=None,
            processed_positions=None,
            complete=False,
        )
        self.assertIsNone(UsageReceipt(**good).model_calls)
        for changes in (
            {"complete": True},
            {"complete": 0},
            {"model_calls": 2},
            {"observed_model_calls": True},
            {"observed_processed_positions": -1},
            {"complete": True, "model_calls": 3, "processed_positions": 4},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                UsageReceipt(**{**good, **changes})

    def test_result_validates_stable_output_and_boundary_consistency(self):
        result = U2ReferenceAdapter(Runtime()).generate(request())
        for changes in (
            {"output_ids": (98, 8)},
            {"output_ids": (99, 8)},
            {"output_ids": (100, 8)},
            {"committed_block_ends": (1,)},
            {"committed_block_ends": (2, 2)},
            {"finish_reason": "stop"},
            {"eoa_observed": False},
            {"error_code": "spurious_error"},
            {"contract_version": True},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                dataclasses.replace(result, **changes)


class AdapterAcceptanceTests(unittest.TestCase):
    def test_eoa_returns_only_committed_output_with_observed_usage(self):
        def generate(value, control):
            control.run_model_call(11, lambda: (40, 41))  # Unstable preview is never committed.
            control.commit((7,))
            control.run_model_call(15, lambda: ())  # Zero-reveal forward still costs a call.
            control.commit(())
            control.run_model_call(17, lambda: (8, 9))
            control.commit((8, 9, 99))
            return "eoa"

        result = U2ReferenceAdapter(Runtime(generate)).generate(request(steps_per_block=32))
        self.assertEqual(result.output_ids, (7, 8, 9))
        self.assertEqual(result.committed_block_ends, (1, 3))
        self.assertEqual(result.finish_reason, "eoa")
        self.assertTrue(result.eoa_observed)
        self.assertEqual(result.usage, UsageReceipt(3, 3, 3, 43, 3, 43, True))
        self.assertNotIn(1, result.output_ids)

    def test_eoa_only_commit_has_no_empty_output_block(self):
        def generate(value, control):
            control.run_model_call(3, lambda: None)
            control.commit((99,))
            return "eoa"

        result = U2ReferenceAdapter(Runtime(generate)).generate(request())
        self.assertEqual(result.output_ids, ())
        self.assertEqual(result.committed_block_ends, ())
        self.assertEqual(result.usage.completion_tokens, 0)
        self.assertTrue(result.eoa_observed)

    def test_length_requires_exact_limit_without_inferred_eoa(self):
        def generate(value, control):
            control.run_model_call(3, lambda: None)
            control.commit((7, 8))
            return "length"

        result = U2ReferenceAdapter(Runtime(generate)).generate(request(max_new_tokens=2))
        self.assertEqual(result.finish_reason, "length")
        self.assertFalse(result.eoa_observed)
        self.assertTrue(result.usage.complete)

    def test_request_binding_fails_before_runtime_or_ledger_admission(self):
        driver = Runtime()
        adapter = U2ReferenceAdapter(driver, max_entries=1)
        for value in (
            request(tokenizer_digest="wrong-tokenizer"),
            request(prompt_ids=(100,)),
            object(),
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                adapter.generate(value)
        self.assertEqual(driver.calls, [])
        self.assertEqual(adapter.generate(request()).finish_reason, "eoa")

    def test_runtime_identity_is_captured_once_for_results_and_replay(self):
        driver = Runtime()
        original = driver.identity
        adapter = U2ReferenceAdapter(driver)
        driver.identity = identity(
            runtime_id="changed-driver-declaration", tokenizer_digest="other"
        )
        result = adapter.generate(request())
        self.assertIs(result.identity, original)
        self.assertIs(adapter.identity, original)
        self.assertIs(adapter.generate(request()), result)
        with self.assertRaises(AttributeError):
            adapter.identity = driver.identity

    def test_invalid_terminal_and_missing_eoa_leave_aggregates_unknown(self):
        for terminal in (None, "stop", "eoa", "length", "budget", "cancelled", 1, [], object()):
            with self.subTest(terminal=terminal):
                driver = Runtime(lambda value, control, terminal=terminal: terminal)
                result = U2ReferenceAdapter(driver).generate(request())
                self.assertEqual(result.finish_reason, "error")
                self.assertEqual(result.error_code, "invalid_terminal_reason")
                self.assertIsNone(result.usage.model_calls)
                self.assertFalse(result.usage.complete)

    def test_runtime_reported_error_is_explicit_not_success(self):
        result = U2ReferenceAdapter(Runtime(lambda value, control: "error")).generate(request())
        self.assertEqual(result.finish_reason, "error")
        self.assertEqual(result.error_code, "runtime_reported_error")
        self.assertFalse(result.usage.complete)

    def test_invalid_commits_are_sticky_even_when_driver_swallows_exceptions(self):
        for invalid in ((98,), (100,), (-1,), (True,), (99, 7), (99, 99), (7, 8, 9), "7"):
            with self.subTest(invalid=invalid):

                def generate(value, control, invalid=invalid):
                    control.run_model_call(3, lambda: None)
                    control.commit((6,))
                    try:
                        control.commit(invalid)
                    except Exception:
                        pass
                    return "length"

                result = U2ReferenceAdapter(Runtime(generate)).generate(request(max_new_tokens=2))
                self.assertEqual(result.output_ids, (6,))
                self.assertEqual(result.committed_block_ends, (1,))
                self.assertEqual(result.error_code, "invalid_commit")
                self.assertIsNone(result.usage.processed_positions)

    def test_failing_commit_iterator_is_sticky(self):
        def bad_tokens():
            yield 7
            raise RuntimeError("private iterator detail")

        def generate(value, control):
            try:
                control.commit(bad_tokens())
            except Exception:
                pass
            return "eoa"

        result = U2ReferenceAdapter(Runtime(generate)).generate(request())
        self.assertEqual(result.output_ids, ())
        self.assertEqual(result.error_code, "invalid_commit")

    def test_lazy_commit_preserves_budget_and_cancellation_receipts(self):
        def check_stop(reason, swallow_stop):
            invoked = []
            cancellation_signals = []

            def generate(value, control):
                control.run_model_call(3, lambda: invoked.append("prefix"))
                control.commit((6,))

                def operation():
                    invoked.append("lazy")
                    cancellation_signals.append(adapter.cancel(value.request_id))
                    return 8

                def tokens():
                    yield 7
                    try:
                        yield control.run_model_call(5, operation)
                    except Exception:
                        if not swallow_stop:
                            raise
                    yield 9

                control.commit(tokens())
                return "eoa"

            driver = Runtime(generate)
            adapter = U2ReferenceAdapter(driver)
            value = request(max_model_calls=1 if reason == "budget" else 2)
            result = adapter.generate(value)
            calls, positions = (1, 3) if reason == "budget" else (2, 8)
            self.assertEqual(result.finish_reason, reason)
            self.assertIsNone(result.error_code)
            self.assertFalse(result.eoa_observed)
            self.assertEqual(result.output_ids, (6,))
            self.assertEqual(result.committed_block_ends, (1,))
            self.assertEqual(
                result.usage,
                UsageReceipt(3, 1, calls, positions, calls, positions, True),
            )
            self.assertIs(adapter.generate(value), result)
            self.assertEqual(len(driver.calls), 1)
            self.assertEqual(invoked, ["prefix"] if reason == "budget" else ["prefix", "lazy"])
            self.assertEqual(cancellation_signals, [] if reason == "budget" else [True])

        for reason in ("budget", "cancelled"):
            for swallow_stop in (False, True):
                with self.subTest(reason=reason, swallow_stop=swallow_stop):
                    check_stop(reason, swallow_stop)

    def test_cancellation_during_lazy_commit_without_a_call_discards_pending_block(self):
        cancellation_signals = []

        def generate(value, control):
            control.run_model_call(3, lambda: None)
            control.commit((6,))

            def tokens():
                yield 7
                cancellation_signals.append(adapter.cancel(value.request_id))
                yield 8

            control.commit(tokens())
            return "eoa"

        driver = Runtime(generate)
        adapter = U2ReferenceAdapter(driver)
        result = adapter.generate(request())
        self.assertEqual(cancellation_signals, [True])
        self.assertEqual(result.finish_reason, "cancelled")
        self.assertIsNone(result.error_code)
        self.assertEqual(result.output_ids, (6,))
        self.assertEqual(result.committed_block_ends, (1,))
        self.assertEqual(result.usage, UsageReceipt(3, 1, 1, 3, 1, 3, True))
        self.assertIs(adapter.generate(request()), result)
        self.assertEqual(len(driver.calls), 1)

    def test_call_or_commit_after_eoa_is_rejected_without_losing_prior_output(self):
        for event in ("call", "commit"):
            with self.subTest(event=event):
                invoked = []

                def generate(value, control, event=event, invoked=invoked):
                    control.run_model_call(3, lambda: None)
                    control.commit((7, 99))
                    try:
                        if event == "call":
                            control.run_model_call(3, lambda: invoked.append(True))
                        else:
                            control.commit((8,))
                    except Exception:
                        pass
                    return "eoa"

                result = U2ReferenceAdapter(Runtime(generate)).generate(request())
                self.assertEqual(result.finish_reason, "error")
                self.assertEqual(result.output_ids, (7,))
                self.assertTrue(result.eoa_observed)
                self.assertEqual(invoked, [])
                self.assertEqual(result.usage.observed_model_calls, 1)

    def test_runtime_failure_preserves_output_and_sanitizes_exception(self):
        def generate(value, control):
            control.run_model_call(5, lambda: None)
            control.commit((7,))
            raise RuntimeError("private runtime detail must not escape")

        driver = Runtime(generate)
        adapter = U2ReferenceAdapter(driver)
        result = adapter.generate(request())
        self.assertEqual(result.output_ids, (7,))
        self.assertEqual(result.error_code, "runtime_error")
        self.assertEqual(result.usage, UsageReceipt(3, 1, 1, 5, None, None, False))
        self.assertNotIn("private", repr(result))
        self.assertIs(adapter.generate(request()), result)
        self.assertEqual(len(driver.calls), 1)

    def test_swallowed_operation_failure_is_sticky_and_positions_are_lower_bound(self):
        def fail():
            raise RuntimeError("operation failure")

        def generate(value, control):
            control.run_model_call(5, lambda: None)
            control.commit((7,))
            try:
                control.run_model_call(999, fail)
            except RuntimeError:
                pass
            return "eoa"

        result = U2ReferenceAdapter(Runtime(generate)).generate(request())
        self.assertEqual(result.error_code, "model_call_failed")
        self.assertEqual(result.usage, UsageReceipt(3, 1, 2, 5, None, None, False))

    def test_invalid_call_never_invokes_operation_even_if_error_swallowed(self):
        for positions in (True, "3", 0, -1):
            with self.subTest(positions=positions):
                invoked = []

                def generate(value, control, positions=positions, invoked=invoked):
                    try:
                        control.run_model_call(positions, lambda: invoked.append(True))
                    except Exception:
                        pass
                    return "eoa"

                result = U2ReferenceAdapter(Runtime(generate)).generate(request())
                self.assertEqual(result.error_code, "invalid_model_call")
                self.assertEqual(invoked, [])
                self.assertEqual(result.usage.observed_model_calls, 0)

    def test_budget_is_enforced_before_extra_call_and_cannot_be_swallowed(self):
        invoked = []

        def generate(value, control):
            for _ in range(3):
                try:
                    control.run_model_call(3, lambda: invoked.append(True))
                except Exception:
                    pass
            return "eoa"

        result = U2ReferenceAdapter(Runtime(generate)).generate(request(max_model_calls=2))
        self.assertEqual(result.finish_reason, "budget")
        self.assertEqual(len(invoked), 2)
        self.assertEqual(result.usage, UsageReceipt(3, 0, 2, 6, 2, 6, True))

    def test_runtime_can_acknowledge_exact_call_budget_without_extra_attempt(self):
        def generate(value, control):
            control.run_model_call(3, lambda: None)
            return "budget"

        result = U2ReferenceAdapter(Runtime(generate)).generate(request(max_model_calls=1))
        self.assertEqual(result.finish_reason, "budget")
        self.assertTrue(result.usage.complete)

    def test_identical_replay_conflicting_payload_and_capacity_do_not_rerun(self):
        driver = Runtime()
        adapter = U2ReferenceAdapter(driver, max_entries=1)
        result = adapter.generate(request())
        self.assertIs(adapter.generate(request()), result)
        for value in (
            request(seed=1),
            request(tokenizer_digest="different"),
            request(prompt_ids=(100,)),
        ):
            with self.subTest(value=value), self.assertRaises(RequestConflictError):
                adapter.generate(value)
        with self.assertRaises(LedgerCapacityError):
            adapter.generate(request(request_id="new-request"))
        self.assertIs(adapter.generate(request()), result)
        self.assertEqual(len(driver.calls), 1)

    def test_max_entries_is_a_positive_integer(self):
        for invalid in (True, 0, -1, 1.0, "1"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                U2ReferenceAdapter(Runtime(), max_entries=invalid)

    def test_base_exception_leaves_cached_failure_and_releases_execution_lock(self):
        def generate(value, control):
            if value.request_id == "request-one":
                control.run_model_call(3, lambda: (_ for _ in ()).throw(KeyboardInterrupt()))
            return Runtime.success(value, control)

        driver = Runtime(generate)
        adapter = U2ReferenceAdapter(driver)
        with self.assertRaises(KeyboardInterrupt):
            adapter.generate(request())
        result = adapter.generate(request())
        self.assertEqual(result.finish_reason, "error")
        self.assertEqual(result.error_code, "model_call_failed")
        self.assertEqual(result.usage.observed_model_calls, 1)
        self.assertIsNone(result.usage.model_calls)
        self.assertEqual(adapter.generate(request(request_id="second")).finish_reason, "eoa")
        self.assertEqual(len(driver.calls), 2)

    def test_retained_control_cannot_call_or_commit_after_finalization(self):
        driver = Runtime()
        adapter = U2ReferenceAdapter(driver)
        result = adapter.generate(request())
        invoked = []
        with self.assertRaisesRegex(Exception, "^already_finalized$"):
            driver.controls[0].run_model_call(3, lambda: invoked.append(True))
        with self.assertRaisesRegex(Exception, "^already_finalized$"):
            driver.controls[0].commit((7,))
        self.assertEqual(invoked, [])
        self.assertIs(adapter.generate(request()), result)
        self.assertEqual(result.output_ids, (7, 8))


class ConcurrencyAcceptanceTests(unittest.TestCase):
    def test_in_progress_duplicate_and_changed_payload_never_dispatch_twice(self):
        entered = threading.Event()
        release = threading.Event()

        def generate(value, control):
            entered.set()
            self.assertTrue(release.wait(3))
            return Runtime.success(value, control)

        driver = Runtime(generate)
        adapter = U2ReferenceAdapter(driver)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(adapter.generate, request())
            try:
                self.assertTrue(entered.wait(3))
                with self.assertRaises(RequestInProgressError):
                    adapter.generate(request())
                with self.assertRaises(RequestConflictError):
                    adapter.generate(request(seed=1))
            finally:
                release.set()
            result = first.result(timeout=3)
        self.assertIs(adapter.generate(request()), result)
        self.assertEqual(len(driver.calls), 1)

    def test_cancel_in_flight_waits_for_operation_and_counts_completed_positions(self):
        entered = threading.Event()
        release = threading.Event()
        invoked = []

        def generate(value, control):
            control.run_model_call(4, lambda: None)
            control.commit((7,))

            def operation():
                entered.set()
                self.assertTrue(release.wait(3))

            try:
                control.run_model_call(6, operation)
            except Exception:
                pass
            try:
                control.run_model_call(99, lambda: invoked.append(True))
            except Exception:
                pass
            return "eoa"

        adapter = U2ReferenceAdapter(Runtime(generate))
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(adapter.generate, request())
            try:
                self.assertTrue(entered.wait(3))
                self.assertTrue(adapter.cancel("request-one"))
                self.assertFalse(adapter.cancel("request-one"))
                self.assertFalse(future.done())
            finally:
                release.set()
            result = future.result(timeout=3)
        self.assertEqual(result.finish_reason, "cancelled")
        self.assertEqual(result.output_ids, (7,))
        self.assertEqual(result.usage, UsageReceipt(3, 1, 2, 10, 2, 10, True))
        self.assertEqual(invoked, [])
        self.assertFalse(adapter.cancel("request-one"))

    def test_queued_cancellation_skips_driver_and_returns_complete_zero_call_receipt(self):
        entered = threading.Event()
        release = threading.Event()

        def generate(value, control):
            entered.set()
            self.assertTrue(release.wait(3))
            return Runtime.success(value, control)

        driver = Runtime(generate)
        adapter = U2ReferenceAdapter(driver)
        with ThreadPoolExecutor(max_workers=2) as pool:
            first = pool.submit(adapter.generate, request())
            try:
                self.assertTrue(entered.wait(3))
                queued = pool.submit(adapter.generate, request(request_id="queued"))
                deadline = time.monotonic() + 3
                accepted = False
                pause = threading.Event()
                while time.monotonic() < deadline:
                    if adapter.cancel("queued"):
                        accepted = True
                        break
                    pause.wait(0.001)
                self.assertTrue(accepted)
                self.assertFalse(queued.done())
            finally:
                release.set()
            first.result(timeout=3)
            result = queued.result(timeout=3)
        self.assertEqual(result.finish_reason, "cancelled")
        self.assertEqual(result.usage, UsageReceipt(3, 0, 0, 0, 0, 0, True))
        self.assertEqual(len(driver.calls), 1)

    def test_cancellation_before_finalization_wins_even_after_eoa_commit(self):
        committed = threading.Event()
        release = threading.Event()

        def generate(value, control):
            Runtime.success(value, control)
            committed.set()
            self.assertTrue(release.wait(3))
            return "eoa"

        adapter = U2ReferenceAdapter(Runtime(generate))
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(adapter.generate, request())
            try:
                self.assertTrue(committed.wait(3))
                self.assertTrue(adapter.cancel("request-one"))
            finally:
                release.set()
            result = future.result(timeout=3)
        self.assertEqual(result.finish_reason, "cancelled")
        self.assertTrue(result.eoa_observed)
        self.assertEqual(result.output_ids, (7, 8))
        self.assertFalse(adapter.cancel("request-one"))
        self.assertIs(adapter.generate(request()), result)

    def test_completed_or_unknown_request_cannot_accept_cancel(self):
        adapter = U2ReferenceAdapter(Runtime())
        self.assertFalse(adapter.cancel("unknown"))
        result = adapter.generate(request())
        self.assertFalse(adapter.cancel("request-one"))
        self.assertEqual(result.finish_reason, "eoa")


class ImportBoundaryTests(unittest.TestCase):
    def test_protocol_imports_without_site_packages_or_model_libraries(self):
        root = Path(__file__).resolve().parents[1]
        code = (
            "import sys; "
            "from src.backends.text_contracts import RuntimeIdentity, TextRequest; "
            "from src.backends.u2_reference import U2ReferenceAdapter; "
            "assert 'torch' not in sys.modules; "
            "assert 'transformers' not in sys.modules; "
            "assert 'fastapi' not in sys.modules"
        )
        completed = subprocess.run(
            [sys.executable, "-S", "-B", "-c", code],
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)

    def test_global_u2_registry_remains_planned_and_unavailable(self):
        for backend in ("u2_reference", "u2_native"):
            self.assertEqual(describe_backend(backend).status, "planned")
            with self.assertRaises(NotImplementedError):
                resolve_backend(backend=backend)


if __name__ == "__main__":
    unittest.main()
