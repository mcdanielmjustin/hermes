"""Phase A6 wiring — unit tests for _routed_fix_chapter helper.

Verifies the ship_readiness fix-dispatch wiring: given audit_results
with scanner_signals, the helper correctly dispatches to routed fixers
based on signature, falls through cleanly when no signatures match,
and reports the right summary.

Avoids LLM calls by feeding scanner_signals that match deterministic
fixers only (laterality, schema_labeling) plus universal_quantifier
where the deterministic-drop path applies.
"""
from __future__ import annotations

import asyncio
import sys
import pathlib
import unittest

import conftest  # noqa: F401  — sets sys.path

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def run_async(coro):
    return asyncio.run(coro)


class _DummySemaphore:
    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False


class TestRoutedFixChapter(unittest.TestCase):
    def setUp(self):
        # Lazy import inside setUp so each test gets a fresh module
        # state; the helper is async and uses fixer_registry at call time.
        from ship_readiness import _routed_fix_chapter
        self._routed_fix_chapter = _routed_fix_chapter

    def test_no_scanner_signals_passthrough(self):
        """Audit results with no scanner_signals → questions pass
        through unchanged; no fixers attempted."""
        questions = [
            {
                "question_id": "Q1",
                "question_stem": "Stem with no patterns.",
                "options": [
                    {"letter": "A", "is_correct": True, "text": "Correct."},
                    {"letter": "B", "is_correct": False, "text": "Distractor."},
                ],
            },
        ]
        audit_results = [{"flagged_distractors": [], "scanner_signals": {}}]

        patched, cost, summary = run_async(self._routed_fix_chapter(
            None, questions, audit_results, _DummySemaphore(),
        ))
        self.assertEqual(patched, questions)
        self.assertEqual(cost, 0.0)
        self.assertEqual(summary["routed_fixes_attempted"], 0)
        self.assertEqual(summary["routed_fixes_applied"], 0)

    def test_universal_quantifier_signature_dispatches_to_fixer(self):
        """A scanner_signal carrying signature='universal_quantifier' on
        letter A → universal_quantifier_fixer applies (deterministic
        drop path; no LLM)."""
        questions = [
            {
                "question_id": "Q1",
                "difficulty_tier": 1,
                "question_stem": "After bilateral hippocampal damage, Dr. Smith still recalls his wedding from a decade earlier.",
                "options": [
                    {"letter": "A", "is_correct": False, "text": (
                        "Retrograde amnesia erases ALL pre-injury memories regardless "
                        "of when they were consolidated."
                    )},
                    {"letter": "B", "is_correct": True, "text": "Hippocampal damage causes anterograde amnesia."},
                    {"letter": "C", "is_correct": False, "text": "Encoding is preserved while retrieval is impaired."},
                    {"letter": "D", "is_correct": False, "text": "Long-term storage reorganizes."},
                ],
            },
        ]
        audit_results = [{
            "flagged_distractors": [{"letter": "A"}],
            "scanner_signals": {
                "A": {"fired": True, "confidence": 0.85,
                      "signature": "universal_quantifier",
                      "reason": "universal_quantifier:'ALL'"},
            },
        }]

        patched, cost, summary = run_async(self._routed_fix_chapter(
            None, questions, audit_results, _DummySemaphore(),
        ))
        self.assertEqual(summary["routed_fixes_attempted"], 1)
        self.assertEqual(summary["routed_fixes_applied"], 1)
        self.assertEqual(summary["by_fixer"], {"universal_quantifier_fixer": 1})

        a_opt = next(o for o in patched[0]["options"] if o["letter"] == "A")
        # Universal quantifier dropped; "_routed_fixer" trace stamped.
        self.assertNotIn("ALL", a_opt["text"])
        self.assertIn("_routed_fixer", a_opt)
        self.assertEqual(a_opt["_routed_fixer"], "universal_quantifier:deterministic_drop")
        # Correct option preserved.
        b_opt = next(o for o in patched[0]["options"] if o["letter"] == "B")
        self.assertTrue(b_opt["is_correct"])

    def test_laterality_signature_dispatches_to_laterality_fixer(self):
        """Scanner signal with signature='laterality' → laterality_fixer
        deterministically flips the laterality."""
        questions = [
            {
                "question_id": "Q2",
                "difficulty_tier": 2,
                "question_stem": "After bilateral hippocampal damage, the patient shows declarative memory deficits.",
                "options": [
                    {"letter": "A", "is_correct": True, "text": "Anterograde amnesia."},
                    {"letter": "B", "is_correct": False, "text": (
                        "The injury produces unilateral cognitive impairment."
                    )},
                ],
            },
        ]
        audit_results = [{
            "flagged_distractors": [{"letter": "B"}],
            "scanner_signals": {
                "B": {"fired": True, "confidence": 0.75,
                      "signature": "laterality",
                      "reason": "laterality:'bilateral'(stem)_vs_'unilateral'(distractor)"},
            },
        }]

        patched, cost, summary = run_async(self._routed_fix_chapter(
            None, questions, audit_results, _DummySemaphore(),
        ))
        self.assertEqual(summary["by_fixer"], {"laterality_fixer": 1})
        b_opt = next(o for o in patched[0]["options"] if o["letter"] == "B")
        self.assertNotIn("unilateral", b_opt["text"].lower())
        self.assertIn("bilateral", b_opt["text"].lower())

    def test_unsignatured_signal_falls_through(self):
        """A scanner signal with an unhandled signature (e.g.
        a hypothetical 'novel_pattern' not yet covered by any fixer)
        → no fixer dispatched; residual_flagged increments.

        Phase B2 added stage_timing + numeric_ratio fixers, so this
        test now uses a synthetic unhandled signature to keep the
        fall-through semantics verified."""
        questions = [
            {
                "question_id": "Q3",
                "difficulty_tier": 1,
                "question_stem": "Some stem.",
                "options": [
                    {"letter": "A", "is_correct": True, "text": "Correct."},
                    {"letter": "B", "is_correct": False, "text": "Wrong."},
                ],
            },
        ]
        audit_results = [{
            "flagged_distractors": [{"letter": "B"}],
            "scanner_signals": {
                "B": {"fired": True, "confidence": 0.65,
                      "signature": "novel_unhandled_pattern",
                      "reason": "synthetic test signature"},
            },
        }]

        patched, cost, summary = run_async(self._routed_fix_chapter(
            None, questions, audit_results, _DummySemaphore(),
        ))
        # No fixer for the synthetic novel signature.
        self.assertEqual(summary["routed_fixes_attempted"], 0)
        self.assertEqual(summary["routed_fixes_applied"], 0)
        self.assertEqual(summary["residual_flagged"], 1)
        # Question unchanged.
        self.assertEqual(patched, questions)

    def test_stage_timing_signature_dispatches_to_stage_timing_fixer(self):
        """B2: stage_timing signal → stage_timing_fixer flips the
        distractor's stage to match the stem's stage."""
        questions = [
            {
                "question_id": "Q-STAGE",
                "difficulty_tier": 1,
                "question_stem": "The pattern emerges during childhood and is established before school age.",
                "options": [
                    {"letter": "A", "is_correct": True, "text": "Onset before age 6."},
                    {"letter": "B", "is_correct": False, "text": (
                        "Adulthood is when the developmental shift occurs."
                    )},
                ],
            },
        ]
        audit_results = [{
            "flagged_distractors": [{"letter": "B"}],
            "scanner_signals": {
                "B": {"fired": True, "confidence": 0.65,
                      "signature": "stage_timing",
                      "reason": "stage_timing:'childhood'(stem)_vs_'adulthood'(distractor)"},
            },
        }]

        patched, cost, summary = run_async(self._routed_fix_chapter(
            None, questions, audit_results, _DummySemaphore(),
        ))
        self.assertEqual(summary["by_fixer"], {"stage_timing_fixer": 1})
        b_opt = next(o for o in patched[0]["options"] if o["letter"] == "B")
        self.assertNotIn("adulthood", b_opt["text"].lower())
        self.assertIn("childhood", b_opt["text"].lower())

    def test_non_fired_signal_skipped(self):
        """Scanner signal with fired=False → not dispatched."""
        questions = [{"question_id": "Q4", "options": [
            {"letter": "A", "is_correct": True, "text": "X"},
        ]}]
        audit_results = [{
            "flagged_distractors": [],
            "scanner_signals": {
                "A": {"fired": False, "signature": "universal_quantifier",
                      "reason": "no_signal", "confidence": 0.0},
            },
        }]
        patched, cost, summary = run_async(self._routed_fix_chapter(
            None, questions, audit_results, _DummySemaphore(),
        ))
        self.assertEqual(summary["routed_fixes_attempted"], 0)
        self.assertEqual(summary["routed_fixes_applied"], 0)


if __name__ == "__main__":
    unittest.main()
