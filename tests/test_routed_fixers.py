"""Phase A6 — routed-fixer tests.

Each fixer gets one signature-specific positive case + invariant tests
that verify the universal preservation rules (correct-option-stays-
correct, all-4-options-preserved, only the flagged option modified
unless the fixer is authorized to touch the stem).

Fixers are async; tests use asyncio. The deterministic fixers don't
need an actual LLM client (they short-circuit on the no-LLM path);
universal_quantifier_fixer's LLM path is exercised separately with a
fake client.
"""
from __future__ import annotations

import asyncio
import unittest
from types import SimpleNamespace

import conftest  # noqa: F401  — sets sys.path

from pipeline.detectors import (
    DetectorSignal,
    VERDICT_OVERRIDE_TO,
    VERDICT_BLOCK,
)
from pipeline.fixers import Fixer, FixerRegistry, create_fixer_registry
from pipeline.fixers.universal_quantifier_fixer import (
    UniversalQuantifierFixer,
    _drop_uq_word,
)
from pipeline.fixers.laterality_fixer import LateralityFixer
from pipeline.fixers.schema_labeling_fixer import SchemaLabelingFixer
from pipeline.fixers.numeric_overlap_fixer import NumericOverlapFixer


# Lightweight semaphore used everywhere we don't actually call an LLM.
class _DummySemaphore:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


def run_async(coro):
    """Run an async coroutine in a fresh event loop for testing."""
    return asyncio.run(coro)


# ── Registry ────────────────────────────────────────────────

class TestFixerRegistry(unittest.TestCase):
    def test_create_fixer_registry_has_expected_fixers(self):
        r = create_fixer_registry()
        ids = {f.fixer_id for f in r.all_fixers()}
        self.assertIn("universal_quantifier_fixer", ids)
        self.assertIn("laterality_fixer", ids)
        self.assertIn("schema_labeling_fixer", ids)
        self.assertIn("numeric_overlap_fixer", ids)

    def test_signature_routing(self):
        r = create_fixer_registry()
        self.assertIsNotNone(r.fixer_for_signature("universal_quantifier"))
        self.assertIsNotNone(r.fixer_for_signature("laterality"))
        self.assertIsNotNone(r.fixer_for_signature("tier_a_brief"))
        self.assertIsNotNone(r.fixer_for_signature("tier_b_lexical"))
        self.assertIsNotNone(r.fixer_for_signature("numeric_overlap"))
        # Unhandled signatures return None.
        self.assertIsNone(r.fixer_for_signature("imperative_lead"))
        self.assertIsNone(r.fixer_for_signature("meta_evaluative"))
        self.assertIsNone(r.fixer_for_signature("nonexistent"))


# ── universal_quantifier_fixer ──────────────────────────────

class TestUniversalQuantifierFixer(unittest.TestCase):
    def setUp(self):
        self.fixer = UniversalQuantifierFixer()

    def test_drop_uq_word_helper(self):
        new_text, dropped = _drop_uq_word("Retrograde amnesia erases ALL pre-injury memories.")
        self.assertEqual(dropped.lower(), "all")
        self.assertNotIn("ALL ", new_text)
        self.assertNotIn(" all ", new_text.lower())

    def test_drop_uq_word_no_uq(self):
        new_text, dropped = _drop_uq_word("Hippocampal damage causes anterograde amnesia.")
        self.assertIsNone(dropped)
        self.assertEqual(new_text, "Hippocampal damage causes anterograde amnesia.")

    def test_deterministic_drop_when_text_long_enough(self):
        """Long distractor → just drop the UQ; no LLM call."""
        q = {
            "question_id": "TEST",
            "difficulty_tier": 1,
            "question_stem": "After bilateral hippocampal damage, Dr. Smith still recalls his wedding from a decade earlier.",
            "options": [
                {"letter": "A", "is_correct": False, "text": (
                    "Retrograde amnesia erases ALL pre-injury memories regardless "
                    "of when they were consolidated."
                ), "explanation": "Wrong because amnesia is selective."},
                {"letter": "B", "is_correct": True, "text": "Hippocampal damage causes anterograde amnesia."},
                {"letter": "C", "is_correct": False, "text": "Encoding is preserved while retrieval is selectively impaired."},
                {"letter": "D", "is_correct": False, "text": "Long-term storage is reorganized over time."},
            ],
        }
        sig = DetectorSignal(
            detector_id="english_gap_scanner", letter="A", fired=True,
            confidence=0.85, signature="universal_quantifier",
            verdict_action=VERDICT_OVERRIDE_TO, proposed_class="english_gap",
            reason="universal_quantifier:'ALL'",
        )
        # Pass a None client — should never be called for deterministic path.
        patched = run_async(self.fixer.fix(None, q, sig, _DummySemaphore()))
        self.assertEqual(len(patched["options"]), 4)
        a_opt = next(o for o in patched["options"] if o["letter"] == "A")
        self.assertNotIn("ALL", a_opt["text"])
        self.assertNotIn("all ", a_opt["text"].lower())
        self.assertEqual(a_opt["_routed_fixer"], "universal_quantifier:deterministic_drop")
        # Correct option must stay correct.
        b_opt = next(o for o in patched["options"] if o["letter"] == "B")
        self.assertTrue(b_opt["is_correct"])
        # Other distractors untouched.
        c_opt = next(o for o in patched["options"] if o["letter"] == "C")
        d_opt = next(o for o in patched["options"] if o["letter"] == "D")
        self.assertNotIn("_routed_fixer", c_opt)
        self.assertNotIn("_routed_fixer", d_opt)

    def test_no_uq_word_returns_unchanged(self):
        q = {
            "options": [
                {"letter": "A", "is_correct": False, "text": "Distractor without universal quantifier."},
                {"letter": "B", "is_correct": True, "text": "Correct."},
            ],
        }
        sig = DetectorSignal(
            detector_id="english_gap_scanner", letter="A", fired=True,
            confidence=0.85, signature="universal_quantifier",
            verdict_action=VERDICT_OVERRIDE_TO, proposed_class="english_gap",
        )
        patched = run_async(self.fixer.fix(None, q, sig, _DummySemaphore()))
        # Returned unchanged (the fixer couldn't find a UQ to drop).
        self.assertEqual(patched, q)

    def test_correct_option_never_modified(self):
        """Even if the signal targets the correct option (anomalous),
        fixer must not modify it."""
        q = {
            "options": [
                {"letter": "A", "is_correct": True, "text": "ALL pre-injury memories preserved."},
                {"letter": "B", "is_correct": False, "text": "Other."},
            ],
        }
        sig = DetectorSignal(
            detector_id="english_gap_scanner", letter="A", fired=True,
            confidence=0.85, signature="universal_quantifier",
            verdict_action=VERDICT_OVERRIDE_TO, proposed_class="english_gap",
        )
        patched = run_async(self.fixer.fix(None, q, sig, _DummySemaphore()))
        # Returned unchanged.
        self.assertEqual(patched, q)


# ── laterality_fixer ────────────────────────────────────────

class TestLateralityFixer(unittest.TestCase):
    def setUp(self):
        self.fixer = LateralityFixer()

    def test_bilateral_unilateral_flip(self):
        q = {
            "question_id": "TEST",
            "difficulty_tier": 2,
            "question_stem": "After bilateral hippocampal damage, the patient shows declarative memory deficits.",
            "options": [
                {"letter": "A", "is_correct": True, "text": "Anterograde amnesia for new declarative content."},
                {"letter": "B", "is_correct": False, "text": "Procedural learning is preserved."},
                {"letter": "C", "is_correct": False, "text": (
                    "The injury produces unilateral cognitive impairment."
                ), "explanation": "Unilateral damage would produce different effects."},
                {"letter": "D", "is_correct": False, "text": "Working memory unaffected."},
            ],
        }
        sig = DetectorSignal(
            detector_id="english_gap_scanner", letter="C", fired=True,
            confidence=0.75, signature="laterality",
            verdict_action=VERDICT_OVERRIDE_TO, proposed_class="english_gap",
        )
        patched = run_async(self.fixer.fix(None, q, sig, _DummySemaphore()))
        c_opt = next(o for o in patched["options"] if o["letter"] == "C")
        self.assertNotIn("unilateral", c_opt["text"].lower())
        self.assertIn("bilateral", c_opt["text"].lower())
        self.assertEqual(c_opt["_routed_fixer"], "laterality:deterministic_flip")
        # Explanation also flipped.
        self.assertNotIn("unilateral", c_opt.get("explanation", "").lower())

    def test_no_exclusive_laterality_in_stem_no_op(self):
        """Stem mentions BOTH bilateral and unilateral → no exclusive
        laterality to flip toward."""
        q = {
            "question_stem": "Comparing bilateral to unilateral lesions in cognitive deficits.",
            "options": [
                {"letter": "A", "is_correct": True, "text": "Differential effects."},
                {"letter": "B", "is_correct": False, "text": "Unilateral lesions only."},
            ],
        }
        sig = DetectorSignal(
            detector_id="english_gap_scanner", letter="B", fired=True,
            confidence=0.75, signature="laterality",
            verdict_action=VERDICT_OVERRIDE_TO, proposed_class="english_gap",
        )
        patched = run_async(self.fixer.fix(None, q, sig, _DummySemaphore()))
        self.assertEqual(patched, q)


# ── schema_labeling_fixer ───────────────────────────────────

class TestSchemaLabelingFixer(unittest.TestCase):
    def setUp(self):
        self.fixer = SchemaLabelingFixer()

    def test_pair_swap_single_sided(self):
        """Distractor has ONLY one member of the pair → fixer swaps it
        to the other. (Both-sided distractors are skipped because the
        swap is ambiguous.)"""
        q = {
            "question_stem": "In Park's experiment, encoding is varied while retrieval is held constant.",
            "options": [
                {"letter": "A", "is_correct": True, "text": "Variation in encoding affects later test performance."},
                {"letter": "B", "is_correct": False, "text": "Retrieval drives the experimental manipulation."},
            ],
        }
        sig = DetectorSignal(
            detector_id="schema_labeling", letter="B", fired=True,
            confidence=0.5, signature="tier_b_lexical",
            verdict_action=VERDICT_OVERRIDE_TO, proposed_class="content_gap",
            extra={"pair_matched": ("encoding", "retrieval"), "brief_boosted": False,
                   "universal_quantifier_blocked": False},
        )
        patched = run_async(self.fixer.fix(None, q, sig, _DummySemaphore()))
        b_opt = next(o for o in patched["options"] if o["letter"] == "B")
        # "Retrieval drives ..." → "Encoding drives ..."
        self.assertNotIn("retrieval", b_opt["text"].lower())
        self.assertIn("encoding", b_opt["text"].lower())
        self.assertEqual(b_opt["_routed_fixer"], "schema_labeling:deterministic_swap")

    def test_pair_swap_both_present_skipped(self):
        """Distractor has BOTH members of the pair → fixer skips
        (the swap target is ambiguous). Returns unchanged."""
        q = {
            "options": [
                {"letter": "A", "is_correct": True, "text": "Encoding affects retrieval."},
                {"letter": "B", "is_correct": False, "text": "Retrieval is the encoding."},
            ],
        }
        sig = DetectorSignal(
            detector_id="schema_labeling", letter="B", fired=True,
            confidence=0.5, signature="tier_b_lexical",
            verdict_action=VERDICT_OVERRIDE_TO, proposed_class="content_gap",
            extra={"pair_matched": ("encoding", "retrieval")},
        )
        patched = run_async(self.fixer.fix(None, q, sig, _DummySemaphore()))
        self.assertEqual(patched, q,
                         "ambiguous swap (both sides present) must be skipped")


# ── numeric_overlap_fixer ──────────────────────────────────

class TestNumericOverlapFixer(unittest.TestCase):
    def setUp(self):
        self.fixer = NumericOverlapFixer()

    def test_t4_returns_unchanged(self):
        q = {
            "difficulty_tier": 4,
            "question_stem": "A 16-year-old patient.",
            "options": [
                {"letter": "A", "is_correct": True, "text": "Use the WISC-V."},
            ],
        }
        sig = DetectorSignal(
            detector_id="numeric_overlap", letter=None, fired=True,
            confidence=0.7, signature="numeric_overlap",
            verdict_action=VERDICT_BLOCK,
        )
        patched = run_async(self.fixer.fix(None, q, sig, _DummySemaphore()))
        # Returned unchanged at T4.
        self.assertEqual(patched, q)

    def test_t2_age_replaced_to_mid_range(self):
        q = {
            "difficulty_tier": 2,
            "question_stem": "A 16-year-old patient is referred for cognitive assessment.",
            "options": [
                {"letter": "A", "is_correct": True, "text": "Use the WISC-V; the patient is in the upper band."},
            ],
        }
        sig = DetectorSignal(
            detector_id="numeric_overlap", letter=None, fired=True,
            confidence=0.7, signature="numeric_overlap",
            verdict_action=VERDICT_BLOCK,
        )
        patched = run_async(self.fixer.fix(None, q, sig, _DummySemaphore()))
        self.assertNotEqual(patched["question_stem"], q["question_stem"],
                            "stem should be modified at T2")
        # The new age should be unambiguously in WISC-V range (6.0–16.92);
        # the fixer picks mid-range (~11), avoiding 16:0–16:11 overlap zone.
        self.assertNotIn("16-year-old", patched["question_stem"])
        self.assertIn("_routed_fixer", patched)
        self.assertTrue(patched["_routed_fixer"].startswith("numeric_overlap:"))


if __name__ == "__main__":
    unittest.main()
