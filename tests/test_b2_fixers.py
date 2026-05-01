"""Phase B2 — tests for numeric_ratio + stage_timing fixers.

Each fixer gets ~5 tests covering registration, fire/no-fire cases,
invariant preservation, and graceful failure modes.

numeric_ratio_fixer is LLM-backed (Sonnet) — uses mock client.
stage_timing_fixer is deterministic — no LLM needed.
"""
from __future__ import annotations

import asyncio
import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import conftest  # noqa: F401  — sets sys.path

from pipeline.detectors import DetectorSignal, VERDICT_OVERRIDE_TO
from pipeline.fixers import create_fixer_registry
from pipeline.fixers.numeric_ratio_fixer import NumericRatioFixer
from pipeline.fixers.stage_timing_fixer import StageTimingFixer


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


# ── Registry ────────────────────────────────────────────────

class TestB2Registration(unittest.TestCase):
    def test_numeric_ratio_fixer_registered(self):
        r = create_fixer_registry()
        f = r.fixer_for_signature("numeric_ratio")
        self.assertIsNotNone(f)
        self.assertEqual(f.fixer_id, "numeric_ratio_fixer")

    def test_stage_timing_fixer_registered(self):
        r = create_fixer_registry()
        f = r.fixer_for_signature("stage_timing")
        self.assertIsNotNone(f)
        self.assertEqual(f.fixer_id, "stage_timing_fixer")


# ── numeric_ratio_fixer ────────────────────────────────────

class TestNumericRatioFixer(unittest.TestCase):
    def setUp(self):
        self.fixer = NumericRatioFixer()

    def _mk_question(self):
        return {
            "question_id": "TEST-RATIO",
            "difficulty_tier": 2,
            "question_stem": (
                "By adulthood, the female-to-male ratio for major depressive "
                "disorder is approximately 2:1."
            ),
            "options": [
                {"letter": "A", "is_correct": True, "text": "Pubertal divergence"},
                {"letter": "B", "is_correct": False, "text": (
                    "The ratio is 3:1 across the lifespan from childhood onward."
                )},
                {"letter": "C", "is_correct": False, "text": "Equivalent rates persist."},
                {"letter": "D", "is_correct": False, "text": "Male-predominant in childhood."},
            ],
        }

    def _mk_signal(self):
        return DetectorSignal(
            detector_id="english_gap_scanner",
            letter="B",
            fired=True,
            confidence=0.80,
            signature="numeric_ratio",
            verdict_action=VERDICT_OVERRIDE_TO,
            proposed_class="english_gap",
            reason="ratio_mismatch:stem='2:1'_vs_dist='3:1'",
        )

    def test_rewrite_drops_ratio(self):
        response = json.dumps({
            "letter": "B",
            "new_text": "Reverse the developmental pattern by extrapolating childhood symmetry.",
            "new_explanation": "Wrong because the pattern emerges at puberty, not after.",
        })
        client = _mock_client(response)
        patched = run_async(self.fixer.fix(
            client, self._mk_question(), self._mk_signal(), _DummySemaphore(),
        ))
        b_opt = next(o for o in patched["options"] if o["letter"] == "B")
        # New text doesn't contain a ratio
        self.assertNotIn(":", b_opt["text"])
        self.assertEqual(b_opt["_routed_fixer"], "numeric_ratio:llm_rewrite")

    def test_rewrite_with_universal_quantifier_rejected(self):
        response = json.dumps({
            "letter": "B",
            "new_text": "All depression patterns reverse at puberty.",
            "new_explanation": "Wrong.",
        })
        client = _mock_client(response)
        patched = run_async(self.fixer.fix(
            client, self._mk_question(), self._mk_signal(), _DummySemaphore(),
        ))
        # Returned unchanged (UQ guard rejected)
        b_opt = next(o for o in patched["options"] if o["letter"] == "B")
        self.assertIn("3:1", b_opt["text"])

    def test_rewrite_that_keeps_ratio_rejected(self):
        """If the LLM substitutes another ratio, reject the rewrite."""
        response = json.dumps({
            "letter": "B",
            "new_text": "The ratio is 4:1 in some age groups.",
            "new_explanation": "Wrong.",
        })
        client = _mock_client(response)
        patched = run_async(self.fixer.fix(
            client, self._mk_question(), self._mk_signal(), _DummySemaphore(),
        ))
        b_opt = next(o for o in patched["options"] if o["letter"] == "B")
        # Original text preserved (rewrite contained a ratio, rejected).
        self.assertIn("3:1", b_opt["text"])

    def test_correct_option_never_modified(self):
        sig = DetectorSignal(
            detector_id="english_gap_scanner", letter="A", fired=True,
            confidence=0.80, signature="numeric_ratio",
            verdict_action=VERDICT_OVERRIDE_TO, proposed_class="english_gap",
        )
        client = _mock_client('{"letter":"A","new_text":"modified","new_explanation":"x"}')
        patched = run_async(self.fixer.fix(
            client, self._mk_question(), sig, _DummySemaphore(),
        ))
        a_opt = next(o for o in patched["options"] if o["letter"] == "A")
        self.assertEqual(a_opt["text"], "Pubertal divergence")

    def test_no_ratio_in_stem_returns_unchanged(self):
        q = {
            "question_stem": "Which neurotransmitter is in reward?",
            "options": [
                {"letter": "A", "is_correct": True, "text": "Dopamine"},
                {"letter": "B", "is_correct": False, "text": "The ratio is 3:1."},
            ],
        }
        sig = self._mk_signal()
        patched = run_async(self.fixer.fix(
            None, q, sig, _DummySemaphore(),
        ))
        # No ratio in stem → no fix applied (defensive return).
        b_opt = next(o for o in patched["options"] if o["letter"] == "B")
        self.assertIn("3:1", b_opt["text"])


# ── stage_timing_fixer ────────────────────────────────────

class TestStageTimingFixer(unittest.TestCase):
    def setUp(self):
        self.fixer = StageTimingFixer()

    def test_childhood_adulthood_flip(self):
        q = {
            "question_id": "TEST-STAGE",
            "difficulty_tier": 1,
            "question_stem": "The pattern emerges during childhood and is established before school age.",
            "options": [
                {"letter": "A", "is_correct": True, "text": "Onset before age 6."},
                {"letter": "B", "is_correct": False, "text": (
                    "Adulthood is when the developmental shift occurs."
                ), "explanation": "Wrong because adulthood is too late."},
            ],
        }
        sig = DetectorSignal(
            detector_id="english_gap_scanner", letter="B", fired=True,
            confidence=0.65, signature="stage_timing",
            verdict_action=VERDICT_OVERRIDE_TO, proposed_class="english_gap",
        )
        patched = run_async(self.fixer.fix(None, q, sig, _DummySemaphore()))
        b_opt = next(o for o in patched["options"] if o["letter"] == "B")
        self.assertNotIn("adulthood", b_opt["text"].lower())
        self.assertIn("childhood", b_opt["text"].lower())
        self.assertEqual(b_opt["_routed_fixer"], "stage_timing:deterministic_flip")
        # Explanation also flipped
        self.assertNotIn("adulthood", b_opt.get("explanation", "").lower())

    def test_acute_chronic_flip(self):
        q = {
            "question_stem": "The acute episode begins suddenly with hallmark psychomotor agitation.",
            "options": [
                {"letter": "A", "is_correct": True, "text": "Sudden onset."},
                {"letter": "B", "is_correct": False, "text": "Chronic course is the defining feature."},
            ],
        }
        sig = DetectorSignal(
            detector_id="english_gap_scanner", letter="B", fired=True,
            confidence=0.65, signature="stage_timing",
            verdict_action=VERDICT_OVERRIDE_TO, proposed_class="english_gap",
        )
        patched = run_async(self.fixer.fix(None, q, sig, _DummySemaphore()))
        b_opt = next(o for o in patched["options"] if o["letter"] == "B")
        self.assertNotIn("chronic", b_opt["text"].lower())
        self.assertIn("acute", b_opt["text"].lower())

    def test_no_exclusive_stage_in_stem_no_op(self):
        """Stem has BOTH childhood and adulthood — can't flip safely."""
        q = {
            "question_stem": "Comparing childhood-onset to adulthood-onset trajectories.",
            "options": [
                {"letter": "A", "is_correct": True, "text": "Both differ in prognosis."},
                {"letter": "B", "is_correct": False, "text": "Adulthood-onset has earlier symptoms."},
            ],
        }
        sig = DetectorSignal(
            detector_id="english_gap_scanner", letter="B", fired=True,
            confidence=0.65, signature="stage_timing",
            verdict_action=VERDICT_OVERRIDE_TO, proposed_class="english_gap",
        )
        patched = run_async(self.fixer.fix(None, q, sig, _DummySemaphore()))
        # Returned unchanged
        self.assertEqual(patched, q)

    def test_correct_option_never_modified(self):
        q = {
            "question_stem": "Childhood onset is the rule.",
            "options": [
                {"letter": "A", "is_correct": True, "text": "Adulthood onset is unusual."},
                {"letter": "B", "is_correct": False, "text": "Adolescence is critical."},
            ],
        }
        sig = DetectorSignal(
            detector_id="english_gap_scanner", letter="A", fired=True,
            confidence=0.65, signature="stage_timing",
            verdict_action=VERDICT_OVERRIDE_TO, proposed_class="english_gap",
        )
        patched = run_async(self.fixer.fix(None, q, sig, _DummySemaphore()))
        # Correct option unchanged
        a_opt = next(o for o in patched["options"] if o["letter"] == "A")
        self.assertEqual(a_opt["text"], "Adulthood onset is unusual.")

    def test_distractor_without_opposite_stage_no_op(self):
        """Distractor doesn't contain the expected opposite stage —
        defensive return."""
        q = {
            "question_stem": "Childhood-onset case.",
            "options": [
                {"letter": "A", "is_correct": True, "text": "Yes"},
                {"letter": "B", "is_correct": False, "text": "Adolescence is the key timing"},
            ],
        }
        sig = DetectorSignal(
            detector_id="english_gap_scanner", letter="B", fired=True,
            confidence=0.65, signature="stage_timing",
            verdict_action=VERDICT_OVERRIDE_TO, proposed_class="english_gap",
        )
        patched = run_async(self.fixer.fix(None, q, sig, _DummySemaphore()))
        self.assertEqual(patched, q)


if __name__ == "__main__":
    unittest.main()
