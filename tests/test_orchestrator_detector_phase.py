"""Phase A3 — orchestrator gen-time detector pass tests.

Verifies:
  - english_gap detector now declares PHASE_GENERATION in its phases tuple
  - registry.scan_for_phase("generation", q) returns english_gap signals
  - At T1/T2 with override-eligible signature: signal carries OVERRIDE_TO
    (the orchestrator should treat as gate failure)
  - At T3 with universal_quantifier: still OVERRIDE_TO at gen-time
    (post A2.5 + S2)
  - At T4: signature stays advisory at gen-time (threshold 0.95 not met)
  - build_correction_prompt routes "detector:*" failures to signature
    guidance
  - Unknown detector signature falls back to generic guidance
"""
from __future__ import annotations

import unittest

import conftest  # noqa: F401  — sets sys.path

from pipeline.detectors import (
    PHASE_AUDIT,
    PHASE_GENERATION,
    VERDICT_ADVISORY,
    VERDICT_OVERRIDE_TO,
)
from pipeline.detectors.english_gap import EnglishGapDetector
from pipeline.detectors.registry import create_detector_registry
from pipeline.prompts import build_correction_prompt


def _q_uq(tier: int) -> dict:
    return {
        "question_id": f"TEST-T{tier}",
        "difficulty_tier": tier,
        "question_stem": (
            "After bilateral hippocampal damage, Dr. Smith, age 56, "
            "still recalls his wedding from a decade earlier."
        ),
        "options": [
            {"letter": "A", "is_correct": False, "text": (
                "Retrograde amnesia erases ALL pre-injury memories regardless "
                "of when they were consolidated."
            )},
            {"letter": "B", "is_correct": True, "text": (
                "Hippocampal damage causes anterograde amnesia."
            )},
            {"letter": "C", "is_correct": False, "text": (
                "Encoding is preserved while retrieval is selectively impaired."
            )},
            {"letter": "D", "is_correct": False, "text": (
                "Long-term storage is reorganized over time."
            )},
        ],
    }


def _q_clean(tier: int) -> dict:
    return {
        "question_id": f"TEST-CLEAN-T{tier}",
        "difficulty_tier": tier,
        "question_stem": "Which neurotransmitter is implicated in reward?",
        "options": [
            {"letter": "A", "is_correct": True, "text": "Dopamine"},
            {"letter": "B", "is_correct": False, "text": "Serotonin"},
            {"letter": "C", "is_correct": False, "text": "Norepinephrine"},
            {"letter": "D", "is_correct": False, "text": "GABA"},
        ],
    }


class TestEnglishGapDetectorAtGenerationPhase(unittest.TestCase):
    """A3: english_gap detector now runs at PHASE_GENERATION too."""

    def setUp(self):
        self.registry = create_detector_registry()

    def test_english_gap_detector_declares_generation_phase(self):
        det = self.registry.get("english_gap_scanner")
        self.assertIsNotNone(det)
        self.assertIn(PHASE_GENERATION, det.phases)
        self.assertIn(PHASE_AUDIT, det.phases)

    def test_t1_uq_at_gen_phase_emits_override_to(self):
        """T1 universal_quantifier signal at gen-time carries OVERRIDE_TO
        (orchestrator interprets as gate failure)."""
        signals = self.registry.scan_for_phase(PHASE_GENERATION, _q_uq(1))
        eg = [s for s in signals if s.detector_id == "english_gap_scanner"]
        # Letter A has the universal-quantifier "ALL".
        a_sig = next((s for s in eg if s.letter == "A"), None)
        self.assertIsNotNone(a_sig)
        self.assertTrue(a_sig.fired)
        self.assertEqual(a_sig.verdict_action, VERDICT_OVERRIDE_TO)
        self.assertEqual(a_sig.proposed_class, "english_gap")
        self.assertEqual(a_sig.signature, "universal_quantifier")

    def test_t3_uq_at_gen_phase_still_overrides(self):
        """T3 still overrides post-A2.5 (threshold 0.85 met by UQ 0.85)."""
        signals = self.registry.scan_for_phase(PHASE_GENERATION, _q_uq(3))
        eg = [s for s in signals if s.detector_id == "english_gap_scanner"]
        a_sig = next((s for s in eg if s.letter == "A"), None)
        self.assertIsNotNone(a_sig)
        self.assertTrue(a_sig.fired)
        self.assertEqual(a_sig.verdict_action, VERDICT_OVERRIDE_TO)

    def test_t4_uq_at_gen_phase_stays_advisory(self):
        """T4 threshold 0.95 not met by UQ 0.85 → advisory at gen-time
        too (consistent with audit-time)."""
        signals = self.registry.scan_for_phase(PHASE_GENERATION, _q_uq(4))
        eg = [s for s in signals if s.detector_id == "english_gap_scanner"]
        a_sig = next((s for s in eg if s.letter == "A"), None)
        self.assertIsNotNone(a_sig)
        self.assertTrue(a_sig.fired)
        self.assertEqual(a_sig.verdict_action, VERDICT_ADVISORY,
                         "T4 single-signature override below 0.95 threshold")

    def test_clean_question_no_gen_signals_fire(self):
        signals = self.registry.scan_for_phase(PHASE_GENERATION, _q_clean(1))
        eg_fired = [
            s for s in signals
            if s.detector_id == "english_gap_scanner" and s.fired
        ]
        self.assertEqual(len(eg_fired), 0)


class TestDetectorCorrectionPrompt(unittest.TestCase):
    """A3: build_correction_prompt recognizes detector failures and emits
    signature-targeted guidance."""

    def test_universal_quantifier_failure_uses_uq_guidance(self):
        failures = [(
            "detector:english_gap_scanner",
            "universal_quantifier on letter A: universal_quantifier:'ALL'+specific_stem:named_subject",
        )]
        prompt = build_correction_prompt("Original prompt", failures, tier=1)
        # Guidance specific to universal_quantifier should be in the output.
        self.assertIn("universal quantifier", prompt.lower())
        self.assertIn("detector:english_gap_scanner", prompt)
        # Should NOT use generic "fix this issue" fallback.
        self.assertNotIn("Fix this issue:", prompt)

    def test_laterality_failure_uses_laterality_guidance(self):
        failures = [(
            "detector:english_gap_scanner",
            "laterality on letter C: laterality:'bilateral'(stem)_vs_'unilateral'(distractor)",
        )]
        prompt = build_correction_prompt("Original prompt", failures, tier=2)
        self.assertIn("laterality", prompt.lower())
        # The guidance should mention reading the stem alone (no concept knowledge).
        self.assertIn("stem alone", prompt.lower())

    def test_unknown_detector_signature_falls_back(self):
        """If the signature isn't in _DETECTOR_SIGNATURE_GUIDANCE, the
        prompt still emits something useful (the reason text)."""
        failures = [(
            "detector:english_gap_scanner",
            "mystery_signature on letter B: something happened",
        )]
        prompt = build_correction_prompt("Original prompt", failures, tier=1)
        self.assertIn("Fix this detector finding:", prompt)
        self.assertIn("mystery_signature", prompt)

    def test_mixed_gate_and_detector_failures(self):
        """Both classic gate failures and detector failures are formatted
        in the same prompt."""
        failures = [
            ("structure", "missing flashcard_seeds"),
            (
                "detector:english_gap_scanner",
                "universal_quantifier on letter A: universal_quantifier:'all'+specific_stem:named_subject",
            ),
        ]
        prompt = build_correction_prompt("Original prompt", failures, tier=1)
        self.assertIn("structure", prompt)
        self.assertIn("detector:english_gap_scanner", prompt)
        # Each failure section is a heading.
        self.assertIn("### structure", prompt)
        self.assertIn("### detector:english_gap_scanner", prompt)


if __name__ == "__main__":
    unittest.main()
