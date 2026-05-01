"""Tests for BloomsCognitiveLevelGate.

The gate codifies the Bloom's anti-patterns from `_blooms_stem_enforcement()`
as post-generation enforcement. Empirical audit found 50% T3/T4 violation
rate when only the prompt rule existed; this gate forces compliance.

T3 path: correct answer must contain an application/analysis indicator.
T4 path: stem + correct answer must reference ≥2 brief concepts.
"""
import unittest

import conftest  # noqa: F401  — sets sys.path

from pipeline.gates import BloomsCognitiveLevelGate


def _q(*, tier=3, stem="A clean stem.", correct_text="An applied answer.",
       distractors=None, correct_letter="A"):
    """Build a minimal question dict for gate testing."""
    distractors = distractors or ["d1", "d2", "d3"]
    options = []
    di = iter(distractors)
    for L in ("A", "B", "C", "D"):
        if L == correct_letter:
            options.append({"letter": L, "text": correct_text,
                            "is_correct": True, "explanation": ""})
        else:
            options.append({"letter": L, "text": next(di),
                            "is_correct": False, "explanation": ""})
    return {
        "difficulty_tier": tier,
        "question_stem": stem,
        "options": options,
    }


def _t4_context(concept_ids, labels=None):
    """Mimic the gate_context dict the orchestrator passes."""
    return {
        "anchor_concept_ids": concept_ids,
        "anchor_concept_labels": labels or {cid: cid.replace("-", " ")
                                            for cid in concept_ids},
    }


class TestT3CognitiveLevel(unittest.TestCase):
    def setUp(self):
        self.gate = BloomsCognitiveLevelGate()

    def test_t3_with_apply_verb_passes(self):
        q = _q(tier=3, correct_text="The clinician should evaluate the patient's "
                                    "responses against the standard.")
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_t3_with_predict_passes(self):
        q = _q(tier=3, correct_text="The model predicts a stronger response after "
                                    "repeated exposure.")
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_t3_with_distinguish_passes(self):
        q = _q(tier=3, correct_text="The patient distinguishes objects by texture "
                                    "rather than shape.")
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_t3_with_comparative_connective_passes(self):
        q = _q(tier=3, correct_text="The hippocampus handles declarative memory, "
                                    "whereas the basal ganglia mediate procedural learning.")
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_t3_bare_definition_fails(self):
        # No application/analysis verbs, no comparison structure — bare label.
        q = _q(tier=3, correct_text="A receptor antagonist with no intrinsic activity.")
        ok, reason = self.gate.check(q)
        self.assertFalse(ok)
        self.assertIn("definitional", reason.lower())

    def test_t3_bare_term_with_explanation_words_passes(self):
        # "produces" is in the indicator list — outcome-style language counts
        q = _q(tier=3, correct_text="The drug produces a reduced response when "
                                    "co-administered with the agonist.")
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)


class TestT4Integration(unittest.TestCase):
    def setUp(self):
        self.gate = BloomsCognitiveLevelGate()

    def test_t4_with_two_concepts_in_stem_and_answer_passes(self):
        ctx = _t4_context(["agonist", "antagonist", "receptor-blockade"])
        q = _q(
            tier=4,
            stem="A drug binds receptors but produces no biological response. "
                 "Compared with an agonist, what would you expect?",
            correct_text="The drug acts as an antagonist by competing with "
                         "endogenous agonists at the receptor site.",
        )
        ok, _ = self.gate.check(q, ctx)
        self.assertTrue(ok)

    def test_t4_single_concept_sufficient_fails(self):
        ctx = _t4_context(["agonist", "antagonist", "receptor-blockade"])
        # Stem and answer reference only "agonist" — single-concept-sufficient
        q = _q(
            tier=4,
            stem="What is the defining feature of an agonist?",
            correct_text="An agonist binds and produces a response.",
        )
        ok, reason = self.gate.check(q, ctx)
        self.assertFalse(ok)
        self.assertIn("integrates only", reason)

    def test_t4_no_concepts_referenced_fails(self):
        ctx = _t4_context(["agonist", "antagonist", "receptor-blockade"])
        q = _q(
            tier=4,
            stem="The patient walked into the office.",
            correct_text="Something happened that day.",
        )
        ok, _ = self.gate.check(q, ctx)
        self.assertFalse(ok)

    def test_t4_without_context_passes(self):
        # If the gate has no brief vocabulary to verify against, it can't fail.
        q = _q(tier=4)
        ok, _ = self.gate.check(q, None)
        self.assertTrue(ok)


class TestNonGatedTiers(unittest.TestCase):
    def setUp(self):
        self.gate = BloomsCognitiveLevelGate()

    def test_t1_not_gated(self):
        q = _q(tier=1, correct_text="A bare term.")
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_t2_not_gated(self):
        q = _q(tier=2, correct_text="A bare term.")
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)


class TestStructuralPassthrough(unittest.TestCase):
    def setUp(self):
        self.gate = BloomsCognitiveLevelGate()

    def test_skips_when_not_four_options(self):
        q = {"difficulty_tier": 3, "options": [{"letter": "A", "text": "x",
                                                "is_correct": True}]}
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
