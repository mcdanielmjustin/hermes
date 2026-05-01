"""Phase A1 parity test — the detector registry must produce signals
that, projected back through the existing manifest shape, give bit-
identical output to the pre-A1 direct calls.

This is the safety net for A1's "no behavior change" invariant.

We test:

1. English_gap parity — `EnglishGapDetector` (via the registry) emits
   the same fired/confidence/reason/signature per letter as a direct
   call to `english_gap_scanner.scan_question`.

2. Schema_labeling parity — `SchemaLabelingDetector` (via the registry)
   produces fired/confidence values for each distractor that match a
   direct call to `classify_distractor` on the same (stem, distractor,
   discriminators) triple.

3. Laterality / UniversalDenial parity — the gate wrappers' signals
   carry the same ok/fail outcome as a direct `gate.check()` call.

4. Negative parity — `fired=False` is preserved; the registry doesn't
   silently drop no-fire results.

Test fixtures cover the canonical patterns called out in the wrapped
modules' docstrings.
"""
from __future__ import annotations

import unittest

import conftest  # noqa: F401  — sets sys.path

from pipeline.detectors import (
    PHASE_AUDIT,
    PHASE_GENERATION,
    VERDICT_ADVISORY,
    VERDICT_BLOCK,
    VERDICT_OVERRIDE_TO,
)
from pipeline.detectors.registry import create_detector_registry
from pipeline.english_gap_scanner import scan_question as direct_eg_scan
from pipeline.gates import (
    LateralityIntegrityGate,
    UniversalDenialGate,
)
from pipeline.schema_labeling_classifier import classify_distractor


def _q_lester() -> dict:
    """Canonical Lester/wedding pattern: stem has bilateral + named
    subject + preservation marker; distractor A uses universal-quantifier
    denial; distractor C inverts laterality.
    """
    return {
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
                "The injury produces unilateral cognitive impairment in "
                "temporal regions."
            )},
            {"letter": "D", "is_correct": False, "text": (
                "Encoding is preserved while retrieval is selectively impaired."
            )},
        ],
    }


def _q_iv_dv() -> dict:
    """Schema-labeling case: IV/DV swap pattern."""
    return {
        "question_stem": (
            "In a study where caffeine dose is the IV and reaction time is "
            "the DV, which option correctly identifies the IV?"
        ),
        "options": [
            {"letter": "A", "is_correct": True, "text": "Caffeine dose"},
            {"letter": "B", "is_correct": False, "text": "Reaction time as IV"},
            {"letter": "C", "is_correct": False, "text": "DV is the caffeine dose"},
            {"letter": "D", "is_correct": False, "text": "Random assignment"},
        ],
    }


def _q_clean() -> dict:
    """Question with no flagged patterns — every detector should return
    fired=False signals."""
    return {
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


class TestEnglishGapParity(unittest.TestCase):
    """Registry-emitted english_gap signals match direct scan_question."""

    def setUp(self):
        self.registry = create_detector_registry()

    def _eg_signals(self, q):
        all_signals = self.registry.scan_for_phase(PHASE_AUDIT, q)
        return {
            s.letter: s for s in all_signals
            if s.detector_id == "english_gap_scanner" and s.letter is not None
        }

    def _assert_parity(self, q):
        registry_signals = self._eg_signals(q)
        direct_signals = direct_eg_scan(q)

        # Every distractor letter the direct call covers must appear in
        # the registry output, and vice versa.
        self.assertEqual(set(registry_signals.keys()), set(direct_signals.keys()))

        for letter, direct in direct_signals.items():
            reg = registry_signals[letter]
            self.assertEqual(reg.fired, direct.fired,
                             f"fired mismatch on {letter}")
            self.assertAlmostEqual(reg.confidence, direct.confidence, places=3,
                                   msg=f"confidence mismatch on {letter}")
            self.assertEqual(reg.signature, direct.signature,
                             f"signature mismatch on {letter}")
            self.assertEqual(reg.reason, direct.reason,
                             f"reason mismatch on {letter}")
            # A1: english_gap signals are advisory only.
            self.assertEqual(reg.verdict_action, VERDICT_ADVISORY,
                             f"english_gap should be advisory in A1, got "
                             f"{reg.verdict_action} for {letter}")

    def test_lester_universal_quantifier(self):
        self._assert_parity(_q_lester())

    def test_clean_question(self):
        self._assert_parity(_q_clean())


class TestSchemaLabelingParity(unittest.TestCase):
    """Registry-emitted schema_labeling fired flag matches the direct
    classify_distractor call per distractor."""

    def setUp(self):
        self.registry = create_detector_registry()

    def _schema_signals(self, q):
        all_signals = self.registry.scan_for_phase(PHASE_AUDIT, q)
        return {
            s.letter: s for s in all_signals
            if s.detector_id == "schema_labeling" and s.letter is not None
        }

    def test_iv_dv_pattern_fires_on_swapped_distractor(self):
        q = _q_iv_dv()
        registry_signals = self._schema_signals(q)

        # Direct classify per distractor.
        stem = q["question_stem"]
        for opt in q["options"]:
            if opt.get("is_correct"):
                continue
            letter = opt["letter"]
            direct = classify_distractor(
                stem=stem,
                distractor_text=opt["text"],
                discriminators=None,
            )
            reg = registry_signals[letter]
            self.assertEqual(
                reg.fired, direct.fired,
                f"fired mismatch on {letter}: reg={reg.fired} direct={direct.fired}",
            )
            self.assertAlmostEqual(
                reg.confidence, direct.confidence, places=3,
                msg=f"confidence mismatch on {letter}",
            )
            if direct.fired:
                self.assertEqual(reg.verdict_action, VERDICT_OVERRIDE_TO,
                                 f"fired schema signal must be override_to "
                                 f"on {letter}")
                self.assertEqual(reg.proposed_class, "content_gap",
                                 f"schema_labeling overrides to content_gap")
            else:
                self.assertEqual(reg.verdict_action, VERDICT_ADVISORY,
                                 f"non-fired schema signal must be advisory")

    def test_clean_question_no_fire(self):
        q = _q_clean()
        registry_signals = self._schema_signals(q)
        # Every distractor's schema signal should be fired=False.
        for letter, sig in registry_signals.items():
            self.assertFalse(sig.fired,
                             f"schema_labeling should not fire on clean "
                             f"question option {letter}")
            self.assertEqual(sig.verdict_action, VERDICT_ADVISORY)


class TestLateralityParity(unittest.TestCase):
    """LateralityDetector signal matches LateralityIntegrityGate.check."""

    def setUp(self):
        self.registry = create_detector_registry()
        self.gate = LateralityIntegrityGate()

    def _lat_signals(self, q):
        all_signals = self.registry.scan_for_phase(PHASE_GENERATION, q)
        return [s for s in all_signals if s.detector_id == "laterality_integrity"]

    def test_lester_inversion_blocks(self):
        q = _q_lester()
        ok, reason = self.gate.check(q)
        signals = self._lat_signals(q)
        self.assertEqual(len(signals), 1, "expected exactly one laterality signal")
        sig = signals[0]
        self.assertNotEqual(ok, sig.fired,
                            "fired=True iff gate ok=False")
        if not ok:
            self.assertEqual(sig.verdict_action, VERDICT_BLOCK)
            self.assertEqual(sig.reason, reason)
            # The gate's reason should name the offending letter (C in
            # the Lester fixture).
            self.assertEqual(sig.letter, "C")

    def test_clean_question_passes(self):
        q = _q_clean()
        ok, reason = self.gate.check(q)
        self.assertTrue(ok, "clean question should pass laterality gate")
        signals = self._lat_signals(q)
        self.assertEqual(len(signals), 1)
        self.assertFalse(signals[0].fired)
        self.assertEqual(signals[0].verdict_action, VERDICT_ADVISORY)


class TestUniversalDenialParity(unittest.TestCase):
    """UniversalDenialDetector signal matches UniversalDenialGate.check."""

    def setUp(self):
        self.registry = create_detector_registry()
        self.gate = UniversalDenialGate()

    def _ud_signals(self, q):
        all_signals = self.registry.scan_for_phase(PHASE_GENERATION, q)
        return [s for s in all_signals if s.detector_id == "universal_denial"]

    def test_clean_question_passes(self):
        q = _q_clean()
        ok, reason = self.gate.check(q)
        self.assertTrue(ok, "clean question should pass universal_denial gate")
        signals = self._ud_signals(q)
        self.assertEqual(len(signals), 1)
        self.assertFalse(signals[0].fired)

    def test_gate_outcome_mirrored_on_lester(self):
        # Lester case may or may not fire universal_denial depending on
        # the shared-content-word check; the test asserts the registry
        # mirrors whatever the gate decides.
        q = _q_lester()
        ok, reason = self.gate.check(q)
        signals = self._ud_signals(q)
        self.assertEqual(len(signals), 1)
        sig = signals[0]
        self.assertNotEqual(ok, sig.fired,
                            f"fired=True iff gate ok=False; "
                            f"gate ok={ok}, signal fired={sig.fired}")
        if not ok:
            self.assertEqual(sig.verdict_action, VERDICT_BLOCK)
            self.assertEqual(sig.reason, reason)


class TestRegistryStructure(unittest.TestCase):
    """The registry's structure is well-formed."""

    def setUp(self):
        self.registry = create_detector_registry()

    def test_expected_detectors_registered(self):
        ids = {d.detector_id for d in self.registry.all_detectors()}
        self.assertIn("english_gap_scanner", ids)
        self.assertIn("schema_labeling", ids)
        self.assertIn("laterality_integrity", ids)
        self.assertIn("universal_denial", ids)

    def test_audit_phase_has_audit_detectors(self):
        ids = [d.detector_id for d in self.registry.detectors_for_phase(PHASE_AUDIT)]
        self.assertIn("english_gap_scanner", ids)
        self.assertIn("schema_labeling", ids)
        # Generation-only detectors must NOT be in audit phase.
        self.assertNotIn("laterality_integrity", ids)
        self.assertNotIn("universal_denial", ids)

    def test_generation_phase_has_gen_detectors(self):
        ids = [d.detector_id for d in self.registry.detectors_for_phase(PHASE_GENERATION)]
        self.assertIn("laterality_integrity", ids)
        self.assertIn("universal_denial", ids)
        # A3: english_gap_scanner now also runs at generation time
        # (when GOLIATH_DETECTORS_AT_GEN env flag is set).
        self.assertIn("english_gap_scanner", ids)
        # schema_labeling stays audit-only — at gen-time we don't know
        # whether a paired-name swap is the question's intended pattern
        # (legitimate) or accidental (illegitimate).
        self.assertNotIn("schema_labeling", ids)


if __name__ == "__main__":
    unittest.main()
