"""Tests for brief_pool_adequacy validator.

The validator catches structural problems in a brief's misconception pool
that would degrade DistractorPlanner output even though the pool passes
basic shape validation. False negatives are expected (heuristic checks);
false positives should be rare (warnings, not failures).
"""
import unittest

import conftest  # noqa: F401  — sets sys.path

from pipeline.brief_pool_adequacy import validate_pool_adequacy


def _well_formed_brief():
    """Reasonable brief that should pass all adequacy checks."""
    return {
        "concepts": [
            {"concept_id": "agonist", "label": "Agonist"},
            {"concept_id": "antagonist", "label": "Antagonist"},
            {"concept_id": "receptor-blockade", "label": "Receptor Blockade"},
        ],
        "misconceptions": [
            {"misconception_id": "agonist-vs-antagonist",
             "label": "Confusing agonist and antagonist",
             "type": "opposite_direction",
             "concepts_involved": ["agonist", "antagonist"]},
            {"misconception_id": "antagonist-vs-blockade",
             "label": "Conflating antagonism with blockade",
             "type": "similar_property",
             "concepts_involved": ["antagonist", "receptor-blockade"]},
            {"misconception_id": "agonist-vs-blockade",
             "label": "Believing agonists block",
             "type": "similar_name",
             "concepts_involved": ["agonist", "receptor-blockade"]},
            {"misconception_id": "blockade-mechanism",
             "label": "Mistaking the blockade mechanism",
             "type": "partial_understanding",
             "concepts_involved": ["receptor-blockade", "agonist"]},
        ],
    }


class TestWellFormedBrief(unittest.TestCase):
    def test_clean_brief_returns_no_issues(self):
        issues = validate_pool_adequacy(_well_formed_brief())
        self.assertEqual(issues, [])


class TestPoolSize(unittest.TestCase):
    def test_pool_under_three_flagged(self):
        brief = _well_formed_brief()
        brief["misconceptions"] = brief["misconceptions"][:2]
        issues = validate_pool_adequacy(brief)
        self.assertTrue(any(i["type"] == "pool_too_small" for i in issues))

    def test_pool_at_three_passes(self):
        brief = _well_formed_brief()
        brief["misconceptions"] = brief["misconceptions"][:3]
        issues = validate_pool_adequacy(brief)
        self.assertFalse(any(i["type"] == "pool_too_small" for i in issues))

    def test_empty_pool_flagged(self):
        brief = {"concepts": [], "misconceptions": []}
        issues = validate_pool_adequacy(brief)
        self.assertTrue(any(i["type"] == "pool_too_small" for i in issues))


class TestPrimaryMisconceptionCoverage(unittest.TestCase):
    def test_concept_with_zero_primary_flagged(self):
        # 'agonist' is in concepts but no misconception references it
        brief = {
            "concepts": [
                {"concept_id": "agonist", "label": "Agonist"},
                {"concept_id": "antagonist", "label": "Antagonist"},
                {"concept_id": "receptor-blockade", "label": "Receptor Blockade"},
            ],
            "misconceptions": [
                {"misconception_id": "ant-vs-block",
                 "label": "...", "type": "similar_property",
                 "concepts_involved": ["antagonist", "receptor-blockade"]},
                {"misconception_id": "ant-vs-block-2",
                 "label": "...", "type": "partial_understanding",
                 "concepts_involved": ["antagonist", "receptor-blockade"]},
                {"misconception_id": "ant-vs-block-3",
                 "label": "...", "type": "opposite_direction",
                 "concepts_involved": ["antagonist", "receptor-blockade"]},
            ],
        }
        issues = validate_pool_adequacy(brief)
        no_primary = [i for i in issues if i["type"] == "no_primary_misconceptions"]
        self.assertEqual(len(no_primary), 1)
        self.assertEqual(no_primary[0]["concept_id"], "agonist")

    def test_all_concepts_covered_passes(self):
        issues = validate_pool_adequacy(_well_formed_brief())
        self.assertFalse(
            any(i["type"] == "no_primary_misconceptions" for i in issues)
        )


class TestOrphanReferences(unittest.TestCase):
    def test_misconception_referencing_unknown_concept_flagged(self):
        brief = _well_formed_brief()
        # Add a misconception referencing a concept_id that doesn't exist
        brief["misconceptions"].append({
            "misconception_id": "bogus",
            "label": "...",
            "type": "similar_name",
            "concepts_involved": ["agonist", "ghost-concept-not-in-brief"],
        })
        issues = validate_pool_adequacy(brief)
        orphans = [i for i in issues if i["type"] == "orphan_concept_references"]
        self.assertEqual(len(orphans), 1)
        self.assertIn("ghost-concept-not-in-brief", orphans[0]["concepts"])

    def test_no_orphans_when_clean(self):
        issues = validate_pool_adequacy(_well_formed_brief())
        self.assertFalse(
            any(i["type"] == "orphan_concept_references" for i in issues)
        )


class TestTypeDiversity(unittest.TestCase):
    def test_pool_of_4_with_one_type_flagged(self):
        brief = _well_formed_brief()
        for m in brief["misconceptions"]:
            m["type"] = "similar_property"
        issues = validate_pool_adequacy(brief)
        self.assertTrue(any(i["type"] == "low_type_diversity" for i in issues))

    def test_pool_of_3_with_one_type_not_flagged(self):
        # The diversity check is skipped for small pools.
        brief = _well_formed_brief()
        brief["misconceptions"] = brief["misconceptions"][:3]
        for m in brief["misconceptions"]:
            m["type"] = "similar_property"
        issues = validate_pool_adequacy(brief)
        self.assertFalse(any(i["type"] == "low_type_diversity" for i in issues))

    def test_diverse_pool_passes(self):
        issues = validate_pool_adequacy(_well_formed_brief())
        self.assertFalse(
            any(i["type"] == "low_type_diversity" for i in issues)
        )


class TestRealCorpusBrief(unittest.TestCase):
    """Verify the existing committed briefs produce no issues."""

    def test_squire_memory_brief_clean(self):
        # Mirrors the fields in data/anchor_briefs/BPSY/D7-PHY-021-b323a513.json
        brief = {
            "concepts": [
                {"concept_id": "nondeclarative-memory-system", "label": "..."},
                {"concept_id": "basal-ganglia-procedural", "label": "..."},
                {"concept_id": "cerebellum-classical-conditioning", "label": "..."},
                {"concept_id": "hippocampal-declarative-system", "label": "..."},
                {"concept_id": "amnesia-preserved-implicit", "label": "..."},
            ],
            "misconceptions": [
                {"misconception_id": "m1", "type": "similar_store",
                 "concepts_involved": ["hippocampal-declarative-system",
                                       "basal-ganglia-procedural"]},
                {"misconception_id": "m2", "type": "similar_property",
                 "concepts_involved": ["cerebellum-classical-conditioning",
                                       "basal-ganglia-procedural"]},
                {"misconception_id": "m3", "type": "overgeneralization",
                 "concepts_involved": ["amnesia-preserved-implicit",
                                       "nondeclarative-memory-system"]},
                {"misconception_id": "m4", "type": "similar_store",
                 "concepts_involved": ["cerebellum-classical-conditioning",
                                       "hippocampal-declarative-system"]},
                {"misconception_id": "m5", "type": "partial_understanding",
                 "concepts_involved": ["nondeclarative-memory-system",
                                       "hippocampal-declarative-system"]},
                {"misconception_id": "m6", "type": "opposite_direction",
                 "concepts_involved": ["amnesia-preserved-implicit"]},
                {"misconception_id": "m7", "type": "similar_property",
                 "concepts_involved": ["basal-ganglia-procedural",
                                       "hippocampal-declarative-system"]},
                {"misconception_id": "m8", "type": "overgeneralization",
                 "concepts_involved": ["nondeclarative-memory-system",
                                       "hippocampal-declarative-system"]},
            ],
        }
        issues = validate_pool_adequacy(brief)
        self.assertEqual(issues, [], f"Real corpus brief flagged: {issues}")


if __name__ == "__main__":
    unittest.main()
