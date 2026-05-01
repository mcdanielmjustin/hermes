"""Tests for ConceptIntegrationPlannerAgent.

The agent is the Layer-3 (pre-LLM scaffolding) for the Tier 4 two-concept
integration requirement. It mirrors DistractorPlannerAgent's strategy:
collapse the LLM's "decide which concepts to integrate" decision so the
prompt carries an explicit constraint instead of a hopeful instruction.
"""
import unittest

import conftest  # noqa: F401  — sets sys.path

from pipeline.agents import ConceptIntegrationPlannerAgent


CONCEPTS = [
    {"concept_id": "agonist", "label": "Agonist"},
    {"concept_id": "antagonist", "label": "Antagonist"},
    {"concept_id": "receptor-blockade", "label": "Receptor Blockade"},
    {"concept_id": "intrinsic-activity", "label": "Intrinsic Activity"},
    {"concept_id": "synthesis-inhibition", "label": "Synthesis Inhibition"},
]


class TestTier4Activation(unittest.TestCase):
    def setUp(self):
        self.agent = ConceptIntegrationPlannerAgent()

    def test_t4_returns_pair(self):
        out = self.agent.execute({"tier": 4, "variant": 1, "concepts": CONCEPTS})
        self.assertTrue(out["requires_integration"])
        self.assertIn("primary_concept_id", out)
        self.assertIn("secondary_concept_id", out)

    def test_primary_and_secondary_differ(self):
        for variant in (1, 2, 3, 4, 5):
            out = self.agent.execute({"tier": 4, "variant": variant,
                                      "concepts": CONCEPTS})
            self.assertNotEqual(
                out["primary_concept_id"],
                out["secondary_concept_id"],
                msg=f"variant {variant}: primary == secondary",
            )

    def test_returns_concept_labels(self):
        # Rotation: variant=1, tier=4 → primary_idx = (0*4 + 3) % 5 = 3
        # → primary = CONCEPTS[3] = intrinsic-activity. The exact pick
        # depends on the rotation formula; just verify both labels come
        # from the input set.
        out = self.agent.execute({"tier": 4, "variant": 1, "concepts": CONCEPTS})
        labels = {c["label"] for c in CONCEPTS}
        self.assertIn(out["primary_concept_label"], labels)
        self.assertIn(out["secondary_concept_label"], labels)


class TestTierActivationGuard(unittest.TestCase):
    def setUp(self):
        self.agent = ConceptIntegrationPlannerAgent()

    def test_t1_no_integration(self):
        out = self.agent.execute({"tier": 1, "variant": 1, "concepts": CONCEPTS})
        self.assertFalse(out["requires_integration"])

    def test_t2_no_integration(self):
        out = self.agent.execute({"tier": 2, "variant": 1, "concepts": CONCEPTS})
        self.assertFalse(out["requires_integration"])

    def test_t3_no_integration(self):
        # T3 requires apply-level cognitive demand but NOT 2-concept
        # integration (that's a T4-specific Bloom's anti-pattern fix).
        out = self.agent.execute({"tier": 3, "variant": 1, "concepts": CONCEPTS})
        self.assertFalse(out["requires_integration"])


class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.agent = ConceptIntegrationPlannerAgent()

    def test_empty_concepts_returns_no_integration(self):
        out = self.agent.execute({"tier": 4, "variant": 1, "concepts": []})
        self.assertFalse(out["requires_integration"])

    def test_single_concept_returns_no_integration(self):
        out = self.agent.execute({"tier": 4, "variant": 1,
                                  "concepts": [CONCEPTS[0]]})
        self.assertFalse(out["requires_integration"])

    def test_two_concepts_works(self):
        out = self.agent.execute({"tier": 4, "variant": 1,
                                  "concepts": CONCEPTS[:2]})
        self.assertTrue(out["requires_integration"])
        self.assertNotEqual(out["primary_concept_id"], out["secondary_concept_id"])


class TestRotationDeterminism(unittest.TestCase):
    """Same (tier, variant, concepts) inputs produce same output —
    ensures resume-safety and deterministic question generation."""

    def test_idempotent(self):
        agent = ConceptIntegrationPlannerAgent()
        out1 = agent.execute({"tier": 4, "variant": 3, "concepts": CONCEPTS})
        out2 = agent.execute({"tier": 4, "variant": 3, "concepts": CONCEPTS})
        self.assertEqual(out1, out2)

    def test_different_variants_pick_differently(self):
        agent = ConceptIntegrationPlannerAgent()
        seen_pairs = set()
        for variant in range(1, 6):
            out = agent.execute({"tier": 4, "variant": variant,
                                 "concepts": CONCEPTS})
            pair = (out["primary_concept_id"], out["secondary_concept_id"])
            seen_pairs.add(pair)
        # Across 5 variants × 5 concepts, expect at least 3 distinct pairs
        # (some rotation, but not necessarily 5 unique pairs since the
        # secondary index uses (primary+1+variant) % len modulo).
        self.assertGreaterEqual(len(seen_pairs), 3)


if __name__ == "__main__":
    unittest.main()
