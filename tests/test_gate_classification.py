"""Tests for gate prerequisite vs content classification.

Gates are split into two categories via the is_prerequisite class attr:
  • Prerequisites: short-circuit on first failure (Structure, ContentQuality,
    Consistency, Uniqueness)
  • Content gates: collected so a single retry can address all of them
    (AnchorGrounding, Attribution, OptionLengthBalance, DistractorMix)

If someone adds a new gate or reclassifies an existing one, these tests
catch the change. They also document the intended categorization.
"""
import unittest

import conftest  # noqa: F401  — sets sys.path

from pipeline import BaseGate
from pipeline.gates import (
    StructureGate, ContentQualityGate, ConsistencyGate, UniquenessGate,
    AnchorGroundingGate, AttributionGate, OptionLengthBalanceGate,
    DistractorMixGate,
    BloomsCognitiveLevelGate, DomainExpertiseGate,
    ScopeMatchGate, OptionClaimGate, OriginalityGate,
    KeywordDistributionGate, StemKeywordDistributionGate,
    TopicRealmGate, ApplyIdentityGate,
    RememberIdentityGate, UnderstandIdentityGate, EvaluateIdentityGate,
    LateralityIntegrityGate, UniversalDenialGate,
    create_gate_pipeline,
)


PREREQUISITES = {
    StructureGate, ContentQualityGate, ConsistencyGate, UniquenessGate,
}

CONTENT_GATES = {
    AnchorGroundingGate, AttributionGate, OptionLengthBalanceGate,
    DistractorMixGate,
    BloomsCognitiveLevelGate, DomainExpertiseGate,
    ScopeMatchGate, OptionClaimGate, OriginalityGate,
    # KeywordDistributionGate is kept for reference but no longer
    # registered in the pipeline (replaced by TopicRealmGate).
    KeywordDistributionGate, StemKeywordDistributionGate,
    TopicRealmGate, ApplyIdentityGate,
    # Bloom's-tier identity layer (one gate per tier; see each gate's
    # docstring for the curated structural enforcement shape).
    RememberIdentityGate, UnderstandIdentityGate, EvaluateIdentityGate,
    # Layer B (Phase 9) — narrow heuristics for the two highest-confidence
    # stem-eliminable distractor patterns. Sonnet audit handles the rest.
    LateralityIntegrityGate, UniversalDenialGate,
}


class TestGateClassification(unittest.TestCase):
    def test_base_gate_default_is_prerequisite(self):
        # Default protects new gates from accidentally becoming non-blocking.
        self.assertTrue(BaseGate.is_prerequisite)

    def test_prerequisite_gates(self):
        for gate_cls in PREREQUISITES:
            self.assertTrue(
                gate_cls.is_prerequisite,
                msg=f"{gate_cls.__name__} must be is_prerequisite=True",
            )

    def test_content_gates(self):
        for gate_cls in CONTENT_GATES:
            self.assertFalse(
                gate_cls.is_prerequisite,
                msg=f"{gate_cls.__name__} must be is_prerequisite=False",
            )

    def test_every_gate_in_pipeline_is_classified(self):
        all_classified = PREREQUISITES | CONTENT_GATES
        for gate in create_gate_pipeline():
            self.assertIn(
                type(gate), all_classified,
                msg=f"{type(gate).__name__} is in the pipeline but not in "
                    f"PREREQUISITES or CONTENT_GATES — update this test "
                    f"and the docstring of BaseGate.is_prerequisite.",
            )


if __name__ == "__main__":
    unittest.main()
