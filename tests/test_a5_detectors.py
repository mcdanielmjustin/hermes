"""Phase A5 — tests for the five new tier-aware structural classifiers.

Each detector ships with: 3 should-fire cases + 2 should-not-fire cases
(plus tier-conditional cases where relevant).

Detectors under test:
  - NumericOverlapDetector   (A5a — WISC age-band overlap, T2/T3 BLOCK
                              vs T4 ADVISORY)
  - ImperativeLeadDetector   (A5b — distractor leads with "Identify"/etc)
  - MetaEvaluativeDetector   (A5c — stem contains "best"/"most"/etc)
  - LeadFormParallelismDetector (A5d — options diverge in grammatical form)
  - DefensibleAlternativeDetector (A5e — T4 only; advisory stub)
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
from pipeline.detectors.numeric_overlap import NumericOverlapDetector
from pipeline.detectors.imperative_lead import ImperativeLeadDetector
from pipeline.detectors.meta_evaluative import MetaEvaluativeDetector
from pipeline.detectors.lead_form_parallelism import LeadFormParallelismDetector
from pipeline.detectors.defensible_alternative import DefensibleAlternativeDetector
from pipeline.detectors.registry import create_detector_registry


# ── A5a — numeric_overlap ────────────────────────────────────

class TestNumericOverlapDetector(unittest.TestCase):
    def setUp(self):
        self.det = NumericOverlapDetector()

    def _wisc_overlap_q(self, tier: int) -> dict:
        """Canonical D8 case: 16-year-old with WISC-V and WAIS-IV
        both supporting the age band."""
        return {
            "difficulty_tier": tier,
            "question_stem": (
                "A 16-year-old patient is referred for cognitive "
                "assessment after a traumatic brain injury."
            ),
            "options": [
                {"letter": "A", "is_correct": False, "text": (
                    "Use the WAIS-IV; the patient meets the floor age."
                )},
                {"letter": "B", "is_correct": True, "text": (
                    "Use the WISC-V; the patient is within the upper band."
                )},
                {"letter": "C", "is_correct": False, "text": (
                    "Use the Halstead-Reitan battery."
                )},
                {"letter": "D", "is_correct": False, "text": (
                    "Refer for clinical interview without instruments."
                )},
            ],
        }

    def test_t2_overlap_blocks(self):
        sigs = self.det.scan(self._wisc_overlap_q(2))
        fired = [s for s in sigs if s.fired]
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0].verdict_action, VERDICT_BLOCK)
        self.assertEqual(fired[0].signature, "numeric_overlap")

    def test_t4_overlap_advisory(self):
        sigs = self.det.scan(self._wisc_overlap_q(4))
        fired = [s for s in sigs if s.fired]
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0].verdict_action, VERDICT_ADVISORY,
                         "T4 overlap-zones can be the legitimate test")

    def test_no_age_in_stem_no_fire(self):
        q = {
            "difficulty_tier": 2,
            "question_stem": "Which neurotransmitter is implicated in reward?",
            "options": [
                {"letter": "A", "is_correct": True, "text": "Dopamine"},
                {"letter": "B", "is_correct": False, "text": "Serotonin"},
                {"letter": "C", "is_correct": False, "text": "Norepinephrine"},
                {"letter": "D", "is_correct": False, "text": "GABA"},
            ],
        }
        sigs = self.det.scan(q)
        fired = [s for s in sigs if s.fired]
        self.assertEqual(len(fired), 0)

    def test_only_one_option_supports_age_no_fire(self):
        """Only one option references an age-supporting instrument →
        no overlap ambiguity."""
        q = {
            "difficulty_tier": 2,
            "question_stem": "A 16-year-old patient presents with TBI.",
            "options": [
                {"letter": "A", "is_correct": True, "text": "Use the WISC-V."},
                {"letter": "B", "is_correct": False, "text": "Refer for interview."},
                {"letter": "C", "is_correct": False, "text": "Order MRI."},
                {"letter": "D", "is_correct": False, "text": "Begin therapy."},
            ],
        }
        sigs = self.det.scan(q)
        fired = [s for s in sigs if s.fired]
        self.assertEqual(len(fired), 0)


# ── A5b — imperative_lead ────────────────────────────────────

class TestImperativeLeadDetector(unittest.TestCase):
    def setUp(self):
        self.det = ImperativeLeadDetector()

    def test_identify_lead_fires(self):
        q = {
            "options": [
                {"letter": "A", "is_correct": True, "text": "Dopamine"},
                {"letter": "B", "is_correct": False, "text": "Identify the corticospinal tract."},
                {"letter": "C", "is_correct": False, "text": "Serotonin"},
                {"letter": "D", "is_correct": False, "text": "GABA"},
            ],
        }
        sigs = self.det.scan(q)
        fired = [s for s in sigs if s.fired]
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0].letter, "B")
        self.assertEqual(fired[0].verdict_action, VERDICT_BLOCK)
        self.assertEqual(fired[0].extra.get("imperative"), "identify")

    def test_predict_lead_fires(self):
        q = {
            "options": [
                {"letter": "A", "is_correct": True, "text": "Dopamine"},
                {"letter": "B", "is_correct": False, "text": "Predict bilateral hemiplegia."},
            ],
        }
        sigs = self.det.scan(q)
        fired = [s for s in sigs if s.fired]
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0].extra.get("imperative"), "predict")

    def test_classify_lead_fires(self):
        q = {
            "options": [
                {"letter": "A", "is_correct": True, "text": "The reward pathway."},
                {"letter": "B", "is_correct": False, "text": "Classify the lesion."},
            ],
        }
        sigs = self.det.scan(q)
        fired = [s for s in sigs if s.fired]
        self.assertEqual(len(fired), 1)

    def test_correct_option_imperative_not_fired(self):
        """Imperative on the CORRECT option is skipped (we only check
        distractors)."""
        q = {
            "options": [
                {"letter": "A", "is_correct": True, "text": "Identify the lesion site."},
                {"letter": "B", "is_correct": False, "text": "Dopamine"},
            ],
        }
        sigs = self.det.scan(q)
        fired = [s for s in sigs if s.fired]
        self.assertEqual(len(fired), 0)

    def test_noun_phrase_lead_no_fire(self):
        q = {
            "options": [
                {"letter": "A", "is_correct": True, "text": "Dopamine"},
                {"letter": "B", "is_correct": False, "text": "Serotonin"},
                {"letter": "C", "is_correct": False, "text": "Norepinephrine"},
            ],
        }
        sigs = self.det.scan(q)
        fired = [s for s in sigs if s.fired]
        self.assertEqual(len(fired), 0)


# ── A5c — meta_evaluative ────────────────────────────────────

class TestMetaEvaluativeDetector(unittest.TestCase):
    def setUp(self):
        self.det = MetaEvaluativeDetector()

    def test_stem_with_best_fires(self):
        q = {"question_stem": "Which option best describes the reward pathway?"}
        sigs = self.det.scan(q)
        fired = [s for s in sigs if s.fired]
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0].verdict_action, VERDICT_BLOCK)
        # "which option" is detected as the multi-word phrase first
        self.assertIn(fired[0].extra.get("token"), {"best", "best describes", "which option"})

    def test_stem_with_most_accurately_fires(self):
        q = {"question_stem": "Which most accurately describes the lesion?"}
        sigs = self.det.scan(q)
        fired = [s for s in sigs if s.fired]
        self.assertEqual(len(fired), 1)

    def test_stem_with_correctly_fires(self):
        q = {"question_stem": "Which option correctly identifies the structure?"}
        sigs = self.det.scan(q)
        fired = [s for s in sigs if s.fired]
        self.assertEqual(len(fired), 1)

    def test_clean_stem_no_fire(self):
        q = {"question_stem": "Which neurotransmitter is implicated in reward?"}
        sigs = self.det.scan(q)
        fired = [s for s in sigs if s.fired]
        self.assertEqual(len(fired), 0)

    def test_empty_stem_no_fire(self):
        q = {"question_stem": ""}
        sigs = self.det.scan(q)
        fired = [s for s in sigs if s.fired]
        self.assertEqual(len(fired), 0)


# ── A5d — lead_form_parallelism ─────────────────────────────

class TestLeadFormParallelismDetector(unittest.TestCase):
    def setUp(self):
        self.det = LeadFormParallelismDetector()

    def test_all_proper_nouns_no_fire(self):
        """All four options share the PROPER_NP shape."""
        q = {
            "options": [
                {"letter": "A", "text": "Dopamine"},
                {"letter": "B", "text": "Serotonin"},
                {"letter": "C", "text": "Norepinephrine"},
                {"letter": "D", "text": "GABA"},
            ],
        }
        sigs = self.det.scan(q)
        fired = [s for s in sigs if s.fired]
        self.assertEqual(len(fired), 0)

    def test_mixed_lead_shapes_fires(self):
        """One PROPER_NP + one DEF_ART_NP + one VERB_ING + one WHEN_IF
        → 4 different shapes → fire."""
        q = {
            "options": [
                {"letter": "A", "text": "Dopamine release in the nucleus accumbens"},
                {"letter": "B", "text": "The reward circuit responds to..."},
                {"letter": "C", "text": "Activating the mesolimbic pathway"},
                {"letter": "D", "text": "When dopamine receptors are stimulated"},
            ],
        }
        sigs = self.det.scan(q)
        fired = [s for s in sigs if s.fired]
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0].verdict_action, VERDICT_ADVISORY)

    def test_two_shapes_only_fires(self):
        """3 NP + 1 DEF_ART_NP — 2 shapes → fires (advisory)."""
        q = {
            "options": [
                {"letter": "A", "text": "Dopamine"},
                {"letter": "B", "text": "Serotonin"},
                {"letter": "C", "text": "Norepinephrine"},
                {"letter": "D", "text": "The GABAergic inhibition response"},
            ],
        }
        sigs = self.det.scan(q)
        fired = [s for s in sigs if s.fired]
        self.assertEqual(len(fired), 1)

    def test_too_few_options_no_fire(self):
        q = {"options": [{"letter": "A", "text": "Dopamine"}]}
        sigs = self.det.scan(q)
        fired = [s for s in sigs if s.fired]
        self.assertEqual(len(fired), 0)


# ── A5e — defensible_alternative ────────────────────────────

class TestDefensibleAlternativeDetector(unittest.TestCase):
    def setUp(self):
        self.det = DefensibleAlternativeDetector()

    def test_t1_returns_no_fire(self):
        q = {
            "difficulty_tier": 1,
            "testable_fact": "Schizophrenia involves dopamine dysregulation in mesolimbic pathways.",
            "options": [
                {"letter": "A", "is_correct": True, "text": "Dopamine dysregulation in mesolimbic pathways."},
                {"letter": "B", "is_correct": False, "text": "Dopamine dysregulation in cortical pathways."},
            ],
        }
        sigs = self.det.scan(q)
        fired = [s for s in sigs if s.fired]
        self.assertEqual(len(fired), 0,
                         "detector applies to T4 only")

    def test_t4_with_2_distractors_sharing_content_fires(self):
        q = {
            "difficulty_tier": 4,
            "testable_fact": (
                "Schizophrenia involves dopamine dysregulation in mesolimbic "
                "pathways with corresponding behavioral activation deficits "
                "and prefrontal hypofunction symptoms."
            ),
            "options": [
                {"letter": "A", "is_correct": True, "text": (
                    "Dopamine dysregulation produces mesolimbic "
                    "behavioral activation symptoms."
                )},
                {"letter": "B", "is_correct": False, "text": (
                    "Mesolimbic dopamine activation drives "
                    "behavioral prefrontal hypofunction."
                )},
                {"letter": "C", "is_correct": False, "text": (
                    "Cortical dopamine activation produces "
                    "behavioral mesolimbic prefrontal symptoms."
                )},
                {"letter": "D", "is_correct": False, "text": "Serotonin only."},
            ],
        }
        sigs = self.det.scan(q)
        fired = [s for s in sigs if s.fired]
        self.assertEqual(len(fired), 1)
        self.assertEqual(fired[0].verdict_action, VERDICT_ADVISORY,
                         "always advisory — heuristic stub")

    def test_t4_low_overlap_no_fire(self):
        """T4 question where distractors are clearly off-topic."""
        q = {
            "difficulty_tier": 4,
            "testable_fact": "Schizophrenia involves dopamine dysregulation in mesolimbic pathways.",
            "options": [
                {"letter": "A", "is_correct": True, "text": (
                    "Dopamine dysregulation in mesolimbic pathways."
                )},
                {"letter": "B", "is_correct": False, "text": "Sodium pumps."},
                {"letter": "C", "is_correct": False, "text": "Liver enzymes."},
                {"letter": "D", "is_correct": False, "text": "Heart valves."},
            ],
        }
        sigs = self.det.scan(q)
        fired = [s for s in sigs if s.fired]
        self.assertEqual(len(fired), 0)


# ── Registry integration ────────────────────────────────────

class TestA5DetectorsRegistered(unittest.TestCase):
    """Verify the new detectors are present in the registry and run
    at the expected phases."""

    def setUp(self):
        self.registry = create_detector_registry()

    def test_all_a5_detectors_present(self):
        ids = {d.detector_id for d in self.registry.all_detectors()}
        self.assertIn("numeric_overlap", ids)
        self.assertIn("imperative_lead", ids)
        self.assertIn("meta_evaluative", ids)
        self.assertIn("lead_form_parallelism", ids)
        self.assertIn("defensible_alternative", ids)

    def test_block_detectors_in_generation_phase(self):
        """The BLOCK-emitting A5 detectors must run at PHASE_GENERATION
        (so the orchestrator can gate on them)."""
        gen_ids = {d.detector_id for d in self.registry.detectors_for_phase(PHASE_GENERATION)}
        self.assertIn("numeric_overlap", gen_ids)
        self.assertIn("imperative_lead", gen_ids)
        self.assertIn("meta_evaluative", gen_ids)

    def test_advisory_only_detectors_audit_phase(self):
        """defensible_alternative and lead_form_parallelism are advisory-
        oriented; both run at audit. defensible is audit-only (T4-specific
        heuristic)."""
        audit_ids = {d.detector_id for d in self.registry.detectors_for_phase(PHASE_AUDIT)}
        self.assertIn("lead_form_parallelism", audit_ids)
        self.assertIn("defensible_alternative", audit_ids)


if __name__ == "__main__":
    unittest.main()
