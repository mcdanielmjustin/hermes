"""Phase A7+ — ambiguity fixer tests.

Mocks the Sonnet client. Verifies:
  - handles_signatures includes 'llm_ambiguity'
  - Fixer routing via FixerRegistry
  - Invariant preservation (correct option, all 4 options, no UQ in rewrite)
  - Graceful failure on malformed LLM response, missing argument, etc.
"""
from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import conftest  # noqa: F401  — sets sys.path

from pipeline.detectors import DetectorSignal, VERDICT_ADVISORY
from pipeline.fixers import create_fixer_registry
from pipeline.fixers.ambiguity_fixer import AmbiguityFixer


def run_async(coro):
    return asyncio.run(coro)


class _DummySemaphore:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def _mock_response(text: str):
    return SimpleNamespace(
        content=[SimpleNamespace(text=text)],
        usage=SimpleNamespace(input_tokens=100, output_tokens=50),
    )


def _mock_client(response_text: str):
    client = MagicMock()
    client.messages = MagicMock()
    client.messages.create = AsyncMock(return_value=_mock_response(response_text))
    return client


class TestRegistration(unittest.TestCase):
    def test_ambiguity_fixer_registered_in_default_registry(self):
        r = create_fixer_registry()
        fixer = r.fixer_for_signature("llm_ambiguity")
        self.assertIsNotNone(fixer)
        self.assertEqual(fixer.fixer_id, "ambiguity_fixer")

    def test_handles_signatures(self):
        f = AmbiguityFixer()
        self.assertIn("llm_ambiguity", f.handles_signatures)


class TestAmbiguityFixerBehavior(unittest.TestCase):
    def setUp(self):
        self.fixer = AmbiguityFixer()

    def _mk_question(self):
        return {
            "question_id": "TEST",
            "difficulty_tier": 4,
            "domain_code": "CASS",
            "question_stem": "A 16-year-old patient is referred for cognitive assessment after TBI.",
            "options": [
                {"letter": "A", "is_correct": False, "text": "Use the WAIS-IV; the patient meets the age floor."},
                {"letter": "B", "is_correct": True, "text": "Use the WISC-V; the patient is within the upper band."},
                {"letter": "C", "is_correct": False, "text": "Halstead-Reitan."},
                {"letter": "D", "is_correct": False, "text": "Clinical interview only."},
            ],
        }

    def _mk_signal(self, argument="WAIS-IV is also valid at age 16:0 due to overlap zone."):
        return DetectorSignal(
            detector_id="llm_ambiguity",
            letter="A",
            fired=True,
            confidence=0.6,
            signature="llm_ambiguity",
            verdict_action=VERDICT_ADVISORY,
            reason=argument,
            extra={"argument": argument},
        )

    def test_rewrite_applied_on_valid_response(self):
        response = json.dumps({
            "letter": "A",
            "new_text": "Use the WAIS-IV; recent TBI invalidates standard age cutoffs.",
            "new_explanation": "Wrong because the WAIS-IV's age floor is firm at 16:0.",
        })
        client = _mock_client(response)
        patched = run_async(self.fixer.fix(
            client, self._mk_question(), self._mk_signal(), _DummySemaphore(),
        ))
        a_opt = next(o for o in patched["options"] if o["letter"] == "A")
        self.assertNotEqual(a_opt["text"], "Use the WAIS-IV; the patient meets the age floor.")
        self.assertIn("recent TBI", a_opt["text"])
        self.assertEqual(a_opt["_routed_fixer"], "ambiguity:llm_rewrite")
        # Correct option preserved.
        b_opt = next(o for o in patched["options"] if o["letter"] == "B")
        self.assertTrue(b_opt["is_correct"])

    def test_rewrite_with_universal_quantifier_rejected(self):
        """Invariant guard: rewrite that introduces a UQ is rejected."""
        response = json.dumps({
            "letter": "A",
            "new_text": "WAIS-IV invalidates ALL standard age cutoffs.",
            "new_explanation": "Wrong because of UQ.",
        })
        client = _mock_client(response)
        patched = run_async(self.fixer.fix(
            client, self._mk_question(), self._mk_signal(), _DummySemaphore(),
        ))
        # Returned unchanged (UQ guard rejected the rewrite).
        a_opt = next(o for o in patched["options"] if o["letter"] == "A")
        self.assertEqual(a_opt["text"], "Use the WAIS-IV; the patient meets the age floor.")

    def test_correct_option_never_modified(self):
        signal = DetectorSignal(
            detector_id="llm_ambiguity", letter="B", fired=True,
            confidence=0.6, signature="llm_ambiguity",
            verdict_action=VERDICT_ADVISORY,
            extra={"argument": "B is also defensible."},
        )
        # Even though signal targets B, the fixer should refuse to modify
        # the correct option.
        client = _mock_client('{"letter": "B", "new_text": "modified", "new_explanation": "x"}')
        patched = run_async(self.fixer.fix(
            client, self._mk_question(), signal, _DummySemaphore(),
        ))
        # No call should have been made; question unchanged.
        b_opt = next(o for o in patched["options"] if o["letter"] == "B")
        self.assertEqual(b_opt["text"], "Use the WISC-V; the patient is within the upper band.")
        self.assertTrue(b_opt["is_correct"])

    def test_no_argument_in_signal_returns_unchanged(self):
        sig = DetectorSignal(
            detector_id="llm_ambiguity", letter="A", fired=True,
            confidence=0.6, signature="llm_ambiguity",
            verdict_action=VERDICT_ADVISORY,
            reason="",  # no reason
            extra={},   # no argument
        )
        patched = run_async(self.fixer.fix(
            None, self._mk_question(), sig, _DummySemaphore(),
        ))
        # Returned unchanged — no LLM call needed.
        self.assertEqual(patched, self._mk_question())

    def test_malformed_json_returns_unchanged(self):
        client = _mock_client("not valid json {broken")
        patched = run_async(self.fixer.fix(
            client, self._mk_question(), self._mk_signal(), _DummySemaphore(),
        ))
        # Returned unchanged.
        a_opt = next(o for o in patched["options"] if o["letter"] == "A")
        self.assertEqual(a_opt["text"], "Use the WAIS-IV; the patient meets the age floor.")

    def test_api_exception_returns_unchanged(self):
        client = MagicMock()
        client.messages = MagicMock()
        client.messages.create = AsyncMock(side_effect=RuntimeError("network fail"))
        patched = run_async(self.fixer.fix(
            client, self._mk_question(), self._mk_signal(), _DummySemaphore(),
        ))
        a_opt = next(o for o in patched["options"] if o["letter"] == "A")
        self.assertEqual(a_opt["text"], "Use the WAIS-IV; the patient meets the age floor.")

    def test_all_four_options_preserved(self):
        response = json.dumps({
            "letter": "A",
            "new_text": "Refined distractor text.",
            "new_explanation": "Refined.",
        })
        client = _mock_client(response)
        patched = run_async(self.fixer.fix(
            client, self._mk_question(), self._mk_signal(), _DummySemaphore(),
        ))
        self.assertEqual(len(patched["options"]), 4)
        letters = {o["letter"] for o in patched["options"]}
        self.assertEqual(letters, {"A", "B", "C", "D"})


if __name__ == "__main__":
    unittest.main()
