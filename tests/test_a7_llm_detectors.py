"""Phase A7 — tests for LLM-backed signal-shaped detectors.

Uses mock anthropic clients to avoid real API calls. Verifies:
  - Signal shape correctness (fired/confidence/signature/verdict_action)
  - Async dispatch via DetectorRegistry.scan_for_phase_async
  - T4-only gating for fact-check
  - Structured-output parsing (valid JSON, malformed JSON, empty)
  - Error handling on API exceptions
  - Sync `scan` raises NotImplementedError (must use async_scan)
"""
from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import conftest  # noqa: F401  — sets sys.path

from pipeline.detectors import (
    PHASE_AUDIT_LLM,
    VERDICT_ADVISORY,
)
from pipeline.detectors.llm_ambiguity import LlmAmbiguityDetector
from pipeline.detectors.llm_fact_check import LlmFactCheckDetector
from pipeline.detectors.registry import create_detector_registry


def run_async(coro):
    return asyncio.run(coro)


class _DummySemaphore:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _mock_response(text: str, in_toks: int = 100, out_toks: int = 50):
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        usage=SimpleNamespace(input_tokens=in_toks, output_tokens=out_toks),
    )


def _mock_client(response_text: str):
    """Build a mock anthropic client that returns `response_text` when
    messages.create is called."""
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=_mock_response(response_text))
    return client


# ── Sync scan must raise ─────────────────────────────────────

class TestSyncScanForbidden(unittest.TestCase):
    def test_ambiguity_sync_scan_raises(self):
        det = LlmAmbiguityDetector()
        with self.assertRaises(NotImplementedError):
            det.scan({}, None)

    def test_fact_check_sync_scan_raises(self):
        det = LlmFactCheckDetector()
        with self.assertRaises(NotImplementedError):
            det.scan({}, None)


# ── llm_ambiguity ────────────────────────────────────────────

class TestLlmAmbiguityDetector(unittest.TestCase):
    def setUp(self):
        self.det = LlmAmbiguityDetector()

    def _mk_question(self):
        return {
            "domain_code": "CASS",
            "difficulty_tier": 4,
            "question_stem": "A 16-year-old patient is referred for cognitive assessment.",
            "options": [
                {"letter": "A", "is_correct": False, "text": "WAIS-IV is appropriate at age 16."},
                {"letter": "B", "is_correct": True, "text": "WISC-V is appropriate at age 16."},
                {"letter": "C", "is_correct": False, "text": "Halstead-Reitan."},
                {"letter": "D", "is_correct": False, "text": "Clinical interview only."},
            ],
        }

    def test_no_defensible_alternatives(self):
        client = _mock_client('{"defensible_alternatives": []}')
        sigs = run_async(self.det.async_scan(
            self._mk_question(),
            context={"client": client, "semaphore": _DummySemaphore()},
        ))
        self.assertEqual(len(sigs), 1)
        self.assertFalse(sigs[0].fired)
        self.assertEqual(sigs[0].verdict_action, VERDICT_ADVISORY)

    def test_one_defensible_alternative_fires(self):
        response = json.dumps({
            "defensible_alternatives": [
                {"letter": "A", "argument": "WAIS-IV is also valid at age 16:0 — overlap zone with WISC-V."},
            ],
        })
        client = _mock_client(response)
        sigs = run_async(self.det.async_scan(
            self._mk_question(),
            context={"client": client, "semaphore": _DummySemaphore()},
        ))
        # One fired signal for the defensible alternative.
        fired = [s for s in sigs if s.fired]
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0].letter, "A")
        self.assertEqual(fired[0].signature, "llm_ambiguity")
        self.assertEqual(fired[0].verdict_action, VERDICT_ADVISORY,
                         "A7 always advisory until calibration data ships")
        self.assertIn("overlap zone", fired[0].reason.lower())

    def test_multiple_defensible_alternatives(self):
        response = json.dumps({
            "defensible_alternatives": [
                {"letter": "A", "argument": "Argument A"},
                {"letter": "C", "argument": "Argument C"},
            ],
        })
        client = _mock_client(response)
        sigs = run_async(self.det.async_scan(
            self._mk_question(),
            context={"client": client, "semaphore": _DummySemaphore()},
        ))
        fired = [s for s in sigs if s.fired]
        self.assertEqual(len(fired), 2)
        letters = {s.letter for s in fired}
        self.assertEqual(letters, {"A", "C"})

    def test_malformed_json_response(self):
        client = _mock_client("not valid json {broken")
        sigs = run_async(self.det.async_scan(
            self._mk_question(),
            context={"client": client, "semaphore": _DummySemaphore()},
        ))
        self.assertEqual(len(sigs), 1)
        self.assertFalse(sigs[0].fired)
        self.assertIn("parse_failed", sigs[0].reason)

    def test_no_client_in_context(self):
        sigs = run_async(self.det.async_scan(
            self._mk_question(),
            context={},
        ))
        self.assertEqual(len(sigs), 1)
        self.assertFalse(sigs[0].fired)
        self.assertIn("no client", sigs[0].reason)

    def test_api_exception_handled(self):
        client = MagicMock()
        client.messages = MagicMock()
        client.messages.create = AsyncMock(side_effect=RuntimeError("network fail"))
        sigs = run_async(self.det.async_scan(
            self._mk_question(),
            context={"client": client, "semaphore": _DummySemaphore()},
        ))
        self.assertEqual(len(sigs), 1)
        self.assertFalse(sigs[0].fired)
        self.assertIn("api_error", sigs[0].reason)


# ── llm_fact_check ──────────────────────────────────────────

class TestLlmFactCheckDetector(unittest.TestCase):
    def setUp(self):
        self.det = LlmFactCheckDetector()

    def _mk_t4_question(self):
        return {
            "domain_code": "CASS",
            "difficulty_tier": 4,
            "question_stem": "Patient evaluation at age 16:0...",
            "options": [
                {"letter": "A", "is_correct": False, "text": "WAIS-IV norms cover 16:0–90:11."},
                {"letter": "B", "is_correct": True, "text": "WISC-V norms cover 6:0–16:11."},
            ],
        }

    def _mk_t1_question(self):
        return {
            "domain_code": "BPSY",
            "difficulty_tier": 1,
            "question_stem": "Which neurotransmitter is in reward?",
            "options": [
                {"letter": "A", "is_correct": True, "text": "Dopamine"},
                {"letter": "B", "is_correct": False, "text": "Serotonin"},
            ],
        }

    def test_t1_returns_no_fire_no_api_call(self):
        """T1 question short-circuits before any API call."""
        client = MagicMock()
        client.messages = MagicMock()
        client.messages.create = AsyncMock(return_value=_mock_response("{}"))

        sigs = run_async(self.det.async_scan(
            self._mk_t1_question(),
            context={"client": client, "semaphore": _DummySemaphore()},
        ))
        self.assertEqual(len(sigs), 1)
        self.assertFalse(sigs[0].fired)
        self.assertIn("T4 only", sigs[0].reason)
        # No API call made.
        client.messages.create.assert_not_called()

    def test_t4_no_factual_errors(self):
        client = _mock_client('{"factual_errors": []}')
        sigs = run_async(self.det.async_scan(
            self._mk_t4_question(),
            context={"client": client, "semaphore": _DummySemaphore()},
        ))
        self.assertEqual(len(sigs), 1)
        self.assertFalse(sigs[0].fired)

    def test_t4_factual_error_fires(self):
        response = json.dumps({
            "factual_errors": [
                {"letter": "A", "claim": "WAIS-IV norms cover 16:0–90:11",
                 "correction": "WAIS-IV norms cover 16:0–90:11 — claim is correct, but..."},
            ],
        })
        client = _mock_client(response)
        sigs = run_async(self.det.async_scan(
            self._mk_t4_question(),
            context={"client": client, "semaphore": _DummySemaphore()},
        ))
        fired = [s for s in sigs if s.fired]
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0].letter, "A")
        self.assertEqual(fired[0].signature, "llm_fact_check")
        self.assertEqual(fired[0].verdict_action, VERDICT_ADVISORY)


# ── Registry integration ────────────────────────────────────

class TestA7DetectorsInRegistry(unittest.TestCase):
    def setUp(self):
        self.registry = create_detector_registry()

    def test_a7_detectors_registered(self):
        ids = {d.detector_id for d in self.registry.all_detectors()}
        self.assertIn("llm_ambiguity", ids)
        self.assertIn("llm_fact_check", ids)

    def test_a7_detectors_at_audit_llm_phase(self):
        ids = {d.detector_id for d in self.registry.detectors_for_phase(PHASE_AUDIT_LLM)}
        self.assertIn("llm_ambiguity", ids)
        self.assertIn("llm_fact_check", ids)

    def test_async_scan_for_phase_runs_a7_detectors(self):
        """The registry's async path correctly awaits async_scan on
        A7 detectors."""
        question = {
            "domain_code": "BPSY",
            "difficulty_tier": 1,
            "question_stem": "Which NT is in reward?",
            "options": [
                {"letter": "A", "is_correct": True, "text": "Dopamine"},
                {"letter": "B", "is_correct": False, "text": "Serotonin"},
            ],
        }
        # Both detectors will run; ambiguity will hit the no-client
        # path; fact_check will short-circuit on T1.
        sigs = run_async(self.registry.scan_for_phase_async(
            PHASE_AUDIT_LLM, question, context={},
        ))
        # Each detector emits at least one signal even when not firing.
        ids = {s.detector_id for s in sigs}
        self.assertIn("llm_ambiguity", ids)
        self.assertIn("llm_fact_check", ids)


if __name__ == "__main__":
    unittest.main()
