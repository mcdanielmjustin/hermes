"""Tests for brief grounding validator.

The validator catches concepts whose key terms don't appear in the anchor's
source text — a heuristic for LLM hallucination during brief generation.
False negatives are acceptable (legitimate abstraction); false positives
should be rare with the >=50% coverage threshold.
"""
import unittest

import conftest  # noqa: F401  — sets sys.path

from pipeline.brief_grounding import (
    validate_concept_grounding, validate_brief_grounding,
)


class TestConceptGrounding(unittest.TestCase):
    def test_well_grounded_concept_passes(self):
        concept = {
            "concept_id": "antagonist",
            "label": "Antagonist",
            "description": "A ligand that produces no intrinsic effect...",
        }
        source = ("An antagonist is a ligand that binds receptors but "
                  "produces no intrinsic biological effect on its own.")
        ok, coverage, missing = validate_concept_grounding(concept, source)
        self.assertTrue(ok)
        self.assertEqual(coverage, 1.0)
        self.assertEqual(missing, [])

    def test_ungrounded_concept_fails(self):
        concept = {
            "concept_id": "ligand-binding-affinity",
            "label": "Ligand Binding Affinity",
            "description": "The strength of binding...",
        }
        # Source is about something completely different.
        source = "The patient was treated with cognitive behavioral therapy."
        ok, coverage, missing = validate_concept_grounding(concept, source)
        self.assertFalse(ok)
        self.assertLess(coverage, 0.5)
        self.assertIn("ligand", missing)

    def test_partial_grounding_at_threshold(self):
        concept = {
            "concept_id": "neurotransmitter-synthesis-inhibition",
            "label": "Synthesis Inhibition",
            "description": "...",
        }
        # Source mentions "neurotransmitter" and "synthesis" but not "inhibition"
        source = "Neurotransmitter synthesis happens in the cell body."
        ok, coverage, missing = validate_concept_grounding(concept, source)
        # neurotransmitter, synthesis found; inhibition missing → 2/3 ≈ 0.67
        self.assertTrue(ok)
        self.assertGreater(coverage, 0.5)

    def test_kebab_id_words_extracted(self):
        # Significant words from kebab-case ID should be checked.
        # Note: validator does strict substring match — no stemming. So the
        # source text must contain the exact form of the keyword. Real
        # corpus may have morphological variants ("hippocampus" vs
        # "hippocampal") that produce false negatives; the threshold
        # absorbs those without flagging the whole concept.
        concept = {
            "concept_id": "hippocampal-declarative-memory-system",
            "label": "Memory",
            "description": "...",
        }
        source = "Hippocampal lesions impair declarative memory."
        ok, coverage, missing = validate_concept_grounding(concept, source)
        # hippocampal, declarative, memory found (3); system filtered (stop)
        self.assertTrue(ok)
        self.assertEqual(missing, [])

    def test_empty_concept_passes(self):
        concept = {"concept_id": "", "label": "", "description": ""}
        ok, _, _ = validate_concept_grounding(concept, "any source")
        self.assertTrue(ok)

    def test_short_kebab_segments_ignored(self):
        # 3-letter words don't count as keywords (would catch "the", "and"
        # if they appeared in concept_ids).
        concept = {
            "concept_id": "x-y-z",  # all <4 chars
            "label": "",
            "description": "",
        }
        ok, coverage, _ = validate_concept_grounding(concept, "")
        self.assertTrue(ok)
        self.assertEqual(coverage, 1.0)


class TestBriefGrounding(unittest.TestCase):
    def test_clean_brief_returns_no_issues(self):
        brief = {
            "concepts": [
                {"concept_id": "agonist", "label": "Agonist",
                 "description": "..."},
                {"concept_id": "antagonist", "label": "Antagonist",
                 "description": "..."},
            ]
        }
        source = "An agonist binds receptors. An antagonist blocks them."
        issues = validate_brief_grounding(brief, source)
        self.assertEqual(issues, [])

    def test_drift_concept_is_flagged(self):
        brief = {
            "concepts": [
                {"concept_id": "agonist", "label": "Agonist", "description": "..."},
                {"concept_id": "psychotherapy-efficacy",
                 "label": "Psychotherapy Efficacy", "description": "..."},
            ]
        }
        # Source is only about agonists, not psychotherapy.
        source = "An agonist is a ligand that activates receptors."
        issues = validate_brief_grounding(brief, source)
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0]["concept_id"], "psychotherapy-efficacy")

    def test_threshold_is_configurable(self):
        brief = {
            "concepts": [
                {"concept_id": "agonist-receptor-binding",
                 "label": "Binding", "description": "..."},
            ]
        }
        source = "An agonist activates receptors."
        # At default threshold (0.5): agonist + receptor found; binding missing
        # → 2/3 ≈ 0.67 → ok
        issues_loose = validate_brief_grounding(brief, source, threshold=0.5)
        self.assertEqual(issues_loose, [])
        # At strict threshold (0.9): 0.67 < 0.9 → flag
        issues_strict = validate_brief_grounding(brief, source, threshold=0.9)
        self.assertEqual(len(issues_strict), 1)


if __name__ == "__main__":
    unittest.main()
