"""Phase A2 — english_gap detector override tests.

Verifies:
  - High-confidence signatures (universal_quantifier, laterality,
    numeric_ratio) PROMOTE to OVERRIDE_TO on T1/T2.
  - Same signatures stay ADVISORY on T3/T4 (deferred to A2.5).
  - Low-confidence signature (stage_timing, conf 0.65) stays
    ADVISORY at every tier.
  - apply_english_gap_override flips classifications correctly with
    full tracing metadata.
  - The schema_labeling × english_gap interaction case: when both
    classifiers fire on the same letter, english_gap wins.

A2's promotion criteria:
  - signature in OVERRIDE_ELIGIBLE_SIGNATURES, AND
  - tier in OVERRIDE_ELIGIBLE_TIERS (T1, T2)
  - signal.fired
Otherwise: ADVISORY.
"""
from __future__ import annotations

import unittest

import conftest  # noqa: F401  — sets sys.path

from pipeline.detectors import (
    PHASE_AUDIT,
    VERDICT_ADVISORY,
    VERDICT_OVERRIDE_TO,
)
from pipeline.detectors.english_gap import (
    EnglishGapDetector,
    OVERRIDE_ELIGIBLE_SIGNATURES,
    OVERRIDE_ELIGIBLE_TIERS,
)
from pipeline.detectors.registry import create_detector_registry
from pipeline.english_gap_scanner import apply_english_gap_override


# ── Fixtures ────────────────────────────────────────────────

def _q_universal_quantifier(tier: int) -> dict:
    """Lester-pattern: stem has named subject + preservation marker;
    distractor A uses universal-quantifier denial."""
    return {
        "question_id": f"TEST-UQ-T{tier}",
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
                "Long-term storage in cortex is reorganized over time."
            )},
        ],
    }


def _q_laterality(tier: int) -> dict:
    """Stem says bilateral; distractor C says unilateral."""
    return {
        "question_id": f"TEST-LAT-T{tier}",
        "difficulty_tier": tier,
        "question_stem": (
            "After bilateral hippocampal damage, the patient shows clear "
            "deficits in declarative memory consolidation."
        ),
        "options": [
            {"letter": "A", "is_correct": True, "text": (
                "Anterograde amnesia for new declarative content."
            )},
            {"letter": "B", "is_correct": False, "text": (
                "Procedural learning is preserved across sessions."
            )},
            {"letter": "C", "is_correct": False, "text": (
                "The injury produces unilateral cognitive impairment."
            )},
            {"letter": "D", "is_correct": False, "text": (
                "Working-memory capacity is reduced for verbal material."
            )},
        ],
    }


def _q_numeric_ratio(tier: int) -> dict:
    """Stem says 2:1; distractor says 3:1."""
    return {
        "question_id": f"TEST-NR-T{tier}",
        "difficulty_tier": tier,
        "question_stem": (
            "By adulthood, the female-to-male ratio for major depressive "
            "disorder is approximately 2:1."
        ),
        "options": [
            {"letter": "A", "is_correct": True, "text": "Pubertal divergence"},
            {"letter": "B", "is_correct": False, "text": (
                "The ratio is 3:1 across the lifespan from childhood onward."
            )},
            {"letter": "C", "is_correct": False, "text": (
                "Equivalent rates persist across all age cohorts."
            )},
            {"letter": "D", "is_correct": False, "text": (
                "Male-predominant in childhood, female-predominant by adulthood."
            )},
        ],
    }


def _q_stage_timing(tier: int) -> dict:
    """Stem mentions childhood; distractor mentions adulthood."""
    return {
        "question_id": f"TEST-ST-T{tier}",
        "difficulty_tier": tier,
        "question_stem": (
            "The pattern emerges during childhood and is established before "
            "school age."
        ),
        "options": [
            {"letter": "A", "is_correct": True, "text": (
                "Onset before age 6 with stable expression thereafter."
            )},
            {"letter": "B", "is_correct": False, "text": (
                "Adulthood is when the developmental shift occurs."
            )},
            {"letter": "C", "is_correct": False, "text": (
                "Genetic loading accounts for most variance in onset."
            )},
            {"letter": "D", "is_correct": False, "text": (
                "Environmental triggers determine first manifestation."
            )},
        ],
    }


def _q_clean(tier: int) -> dict:
    return {
        "question_id": f"TEST-CLEAN-T{tier}",
        "difficulty_tier": tier,
        "question_stem": (
            "Which neurotransmitter is most directly implicated in the "
            "reward pathway?"
        ),
        "options": [
            {"letter": "A", "is_correct": True, "text": "Dopamine"},
            {"letter": "B", "is_correct": False, "text": "Serotonin"},
            {"letter": "C", "is_correct": False, "text": "Norepinephrine"},
            {"letter": "D", "is_correct": False, "text": "GABA"},
        ],
    }


# ── Detector promotion tests ─────────────────────────────────

class TestEnglishGapDetectorPromotion(unittest.TestCase):
    """The EnglishGapDetector emits OVERRIDE_TO on T1/T2 with override-
    eligible signatures; ADVISORY otherwise."""

    def setUp(self):
        self.detector = EnglishGapDetector()

    def _eg_signal_for(self, q, letter):
        signals = self.detector.scan(q)
        for s in signals:
            if s.letter == letter:
                return s
        self.fail(f"no english_gap signal for letter {letter}")

    # T1/T2 with override-eligible signatures: should fire OVERRIDE_TO
    # (T1/T2 cell threshold = 0.75; UQ 0.85, laterality 0.75, num_ratio
    # 0.80 all reach it)

    def test_t1_universal_quantifier_overrides(self):
        sig = self._eg_signal_for(_q_universal_quantifier(1), "A")
        self.assertTrue(sig.fired)
        self.assertEqual(sig.signature, "universal_quantifier")
        self.assertEqual(sig.verdict_action, VERDICT_OVERRIDE_TO)
        self.assertEqual(sig.proposed_class, "english_gap")
        # A2.5: detector stamps cell_threshold and tier in extra.
        self.assertEqual(sig.extra.get("tier"), 1)
        self.assertAlmostEqual(sig.extra.get("cell_threshold"), 0.75, places=3)

    def test_t2_laterality_overrides(self):
        sig = self._eg_signal_for(_q_laterality(2), "C")
        self.assertTrue(sig.fired)
        self.assertEqual(sig.signature, "laterality")
        self.assertEqual(sig.verdict_action, VERDICT_OVERRIDE_TO)
        self.assertEqual(sig.proposed_class, "english_gap")

    def test_t1_numeric_ratio_overrides(self):
        sig = self._eg_signal_for(_q_numeric_ratio(1), "B")
        self.assertTrue(sig.fired)
        self.assertEqual(sig.signature, "numeric_ratio")
        self.assertEqual(sig.verdict_action, VERDICT_OVERRIDE_TO)

    # T3 cell threshold = 0.85.
    # universal_quantifier (0.85) reaches it → override.
    # laterality (0.75) and numeric_ratio (0.80) below → advisory.

    def test_t3_universal_quantifier_overrides_at_threshold_0_85(self):
        sig = self._eg_signal_for(_q_universal_quantifier(3), "A")
        self.assertTrue(sig.fired)
        self.assertEqual(sig.verdict_action, VERDICT_OVERRIDE_TO,
                         "A2.5: T3 universal_quantifier (0.85) meets T3 "
                         "threshold (0.85)")
        self.assertAlmostEqual(sig.extra.get("cell_threshold"), 0.85, places=3)

    def test_t3_numeric_ratio_below_threshold_advisory(self):
        sig = self._eg_signal_for(_q_numeric_ratio(3), "B")
        self.assertTrue(sig.fired)
        self.assertEqual(sig.verdict_action, VERDICT_ADVISORY,
                         "T3 numeric_ratio (0.80) below threshold (0.85)")

    def test_t3_laterality_below_threshold_advisory(self):
        sig = self._eg_signal_for(_q_laterality(3), "C")
        self.assertTrue(sig.fired)
        self.assertEqual(sig.verdict_action, VERDICT_ADVISORY,
                         "T3 laterality (0.75) below threshold (0.85)")

    # T4 cell threshold = 0.95.
    # No single english_gap_scanner signature reaches 0.95 → all advisory.

    def test_t4_numeric_ratio_advisory(self):
        sig = self._eg_signal_for(_q_numeric_ratio(4), "B")
        self.assertTrue(sig.fired)
        self.assertEqual(sig.verdict_action, VERDICT_ADVISORY,
                         "T4: numeric_ratio (0.80) below threshold (0.95)")

    def test_t4_laterality_advisory(self):
        sig = self._eg_signal_for(_q_laterality(4), "C")
        self.assertTrue(sig.fired)
        self.assertEqual(sig.verdict_action, VERDICT_ADVISORY,
                         "T4: laterality (0.75) below threshold (0.95)")

    def test_t4_universal_quantifier_advisory(self):
        sig = self._eg_signal_for(_q_universal_quantifier(4), "A")
        self.assertTrue(sig.fired)
        self.assertEqual(sig.verdict_action, VERDICT_ADVISORY,
                         "T4: universal_quantifier (0.85) below threshold (0.95)")

    # Low-confidence signature stage_timing: always ADVISORY

    def test_t1_stage_timing_advisory(self):
        sig = self._eg_signal_for(_q_stage_timing(1), "B")
        # Note: stage_timing fires only when stem has one stage and
        # distractor the other AND stem doesn't have both. Verify fired
        # before checking advisory.
        if sig.fired:
            self.assertEqual(sig.verdict_action, VERDICT_ADVISORY,
                             "stage_timing (conf 0.65) is below override "
                             "threshold; should always be advisory")
            self.assertEqual(sig.signature, "stage_timing")

    def test_t2_stage_timing_advisory(self):
        sig = self._eg_signal_for(_q_stage_timing(2), "B")
        if sig.fired:
            self.assertEqual(sig.verdict_action, VERDICT_ADVISORY)

    # Clean question: no fire, advisory

    def test_clean_t1_no_fire(self):
        for letter in ("B", "C", "D"):
            sig = self._eg_signal_for(_q_clean(1), letter)
            self.assertFalse(sig.fired)
            self.assertEqual(sig.verdict_action, VERDICT_ADVISORY)

    # Constants are well-formed

    def test_override_eligible_constants(self):
        self.assertIn("universal_quantifier", OVERRIDE_ELIGIBLE_SIGNATURES)
        self.assertIn("laterality", OVERRIDE_ELIGIBLE_SIGNATURES)
        self.assertIn("numeric_ratio", OVERRIDE_ELIGIBLE_SIGNATURES)
        self.assertNotIn("stage_timing", OVERRIDE_ELIGIBLE_SIGNATURES,
                         "stage_timing must NOT be override-eligible")
        self.assertEqual(OVERRIDE_ELIGIBLE_TIERS, frozenset({1, 2}))


# ── Override application tests ───────────────────────────────

class TestApplyEnglishGapOverride(unittest.TestCase):
    """apply_english_gap_override flips classifications correctly with
    full tracing metadata."""

    def setUp(self):
        self.registry = create_detector_registry()

    def _eg_signals(self, q):
        all_sig = self.registry.scan_for_phase(PHASE_AUDIT, q)
        return [s for s in all_sig if s.detector_id == "english_gap_scanner"]

    def test_t1_uq_clean_classification_flips(self):
        """Audit said 'clean'; scanner fires UQ at T1 → flips to english_gap."""
        q = _q_universal_quantifier(1)
        eg = self._eg_signals(q)
        # Simulate LLM verdict: A=clean (the audit missed the universal_quantifier)
        classifications = [
            {"letter": "A", "class": "clean", "distractor_text": q["options"][0]["text"]},
            {"letter": "C", "class": "clean", "distractor_text": q["options"][2]["text"]},
            {"letter": "D", "class": "clean", "distractor_text": q["options"][3]["text"]},
        ]
        new_cls, count = apply_english_gap_override(q, classifications, eg)
        self.assertEqual(count, 1)
        a_entry = next(c for c in new_cls if c["letter"] == "A")
        self.assertEqual(a_entry["class"], "english_gap")
        self.assertEqual(a_entry["original_class"], "clean")
        self.assertEqual(a_entry["structural_override"], "english_gap_scanner")
        self.assertEqual(a_entry["structural_override_signature"], "universal_quantifier")
        # A2.5: stamps tier + cell_threshold (replaces A2's tier_gate).
        self.assertEqual(a_entry["structural_override_tier"], 1)
        self.assertAlmostEqual(
            a_entry["structural_override_cell_threshold"], 0.75, places=3
        )
        self.assertGreaterEqual(a_entry["structural_override_confidence"], 0.85)

    def test_t1_uq_already_eg_no_op(self):
        """Audit ALREADY classified english_gap → override is no-op."""
        q = _q_universal_quantifier(1)
        eg = self._eg_signals(q)
        classifications = [
            {"letter": "A", "class": "english_gap", "distractor_text": q["options"][0]["text"]},
        ]
        new_cls, count = apply_english_gap_override(q, classifications, eg)
        self.assertEqual(count, 0, "no-op when class is already english_gap")
        self.assertEqual(new_cls[0]["class"], "english_gap")
        self.assertNotIn("structural_override", new_cls[0],
                         "no override stamping when already eg")

    def test_t3_uq_overrides_at_a2_5_threshold(self):
        """A2.5: T3 question with universal_quantifier (0.85) meets the
        T3 cell threshold (0.85) → DOES trigger override."""
        q = _q_universal_quantifier(3)
        eg = self._eg_signals(q)
        classifications = [
            {"letter": "A", "class": "clean", "distractor_text": q["options"][0]["text"]},
        ]
        new_cls, count = apply_english_gap_override(q, classifications, eg)
        self.assertEqual(count, 1, "A2.5: T3 UQ at threshold 0.85 triggers override")
        self.assertEqual(new_cls[0]["class"], "english_gap")
        self.assertEqual(new_cls[0]["structural_override_tier"], 3)
        self.assertAlmostEqual(
            new_cls[0]["structural_override_cell_threshold"], 0.85, places=3
        )

    def test_t3_laterality_no_override(self):
        """A2.5: T3 laterality (0.75) is BELOW the T3 cell threshold (0.85)
        → NO override (advisory only)."""
        q = _q_laterality(3)
        eg = self._eg_signals(q)
        classifications = [
            {"letter": "C", "class": "clean", "distractor_text": q["options"][2]["text"]},
        ]
        new_cls, count = apply_english_gap_override(q, classifications, eg)
        self.assertEqual(count, 0, "T3 laterality (0.75) below threshold (0.85)")

    def test_t4_no_override(self):
        """T4 question, scanner fires laterality → NO override."""
        q = _q_laterality(4)
        eg = self._eg_signals(q)
        classifications = [
            {"letter": "C", "class": "clean", "distractor_text": q["options"][2]["text"]},
        ]
        new_cls, count = apply_english_gap_override(q, classifications, eg)
        self.assertEqual(count, 0, "T4 laterality must NOT trigger override")

    def test_no_signals_no_override(self):
        """Clean question, no signals → no overrides."""
        q = _q_clean(1)
        eg = self._eg_signals(q)
        classifications = [
            {"letter": "B", "class": "clean", "distractor_text": q["options"][1]["text"]},
        ]
        new_cls, count = apply_english_gap_override(q, classifications, eg)
        self.assertEqual(count, 0)


# ── Schema_labeling × english_gap interaction ───────────────

class TestSchemaLabelingEnglishGapInteraction(unittest.TestCase):
    """When both classifiers could fire on the same letter, english_gap
    wins (the scanner is more specific). Per A2 plan: the
    universal-quantifier guard inside classify_distractor already prevents
    schema_labeling from firing on UQ cases, so the only conflict is on
    Tier-B lexical pairs that happen to also have a scanner signature.
    A2 lets english_gap re-promote in that case (it's the higher-precision
    signal)."""

    def setUp(self):
        self.registry = create_detector_registry()

    def test_universal_quantifier_blocks_schema_labeling_no_conflict(self):
        """If a distractor has a universal quantifier, schema_labeling
        does NOT fire (universal-quantifier guard); english_gap scanner
        does fire. So no conflict — english_gap override applies cleanly.
        """
        # Construct a question where both schema_labeling pair AND a
        # universal quantifier are present on the same distractor.
        q = {
            "question_id": "TEST-INT-T1",
            "difficulty_tier": 1,
            "question_stem": (
                "In Dr. Park's experiment, age 30 participants compare "
                "encoding vs retrieval performance under stress."
            ),
            "options": [
                {"letter": "A", "is_correct": True, "text": (
                    "Encoding decrements under acute stress."
                )},
                {"letter": "B", "is_correct": False, "text": (
                    "Retrieval ALL pre-encoded items perfectly under stress."
                )},
                {"letter": "C", "is_correct": False, "text": (
                    "Working memory shows no stress effect."
                )},
                {"letter": "D", "is_correct": False, "text": (
                    "Procedural learning interferes with encoding."
                )},
            ],
        }

        all_sig = self.registry.scan_for_phase(PHASE_AUDIT, q)
        sl = [s for s in all_sig if s.detector_id == "schema_labeling" and s.letter == "B"]
        eg = [s for s in all_sig if s.detector_id == "english_gap_scanner" and s.letter == "B"]

        # Schema_labeling either doesn't fire (UQ guard) or fires with
        # blocked status; english_gap fires.
        self.assertEqual(len(sl), 1)
        self.assertFalse(sl[0].fired,
                         "schema_labeling must be blocked by universal-quantifier guard")
        self.assertEqual(len(eg), 1)
        self.assertTrue(eg[0].fired)
        self.assertEqual(eg[0].verdict_action, VERDICT_OVERRIDE_TO)


if __name__ == "__main__":
    unittest.main()
