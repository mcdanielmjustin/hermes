"""Tests for DomainExpertiseGate.

The gate catches lay-person solvability creep at T2+. Empirical audit found
60-80% of T2+ questions answerable with general knowledge despite the
prompt requiring domain expertise. The gate enforces a concrete heuristic:
the correct answer + explanation must contain ≥2 technical terms from the
brief's vocabulary.

Hard fail at T3+. Soft warn at T2 (still passes). Not gated at T1.
"""
import unittest

import conftest  # noqa: F401  — sets sys.path

from pipeline.gates import DomainExpertiseGate


def _q(*, tier=3, correct_text="Generic answer.", correct_explanation=""):
    return {
        "difficulty_tier": tier,
        "options": [
            {"letter": "A", "text": correct_text,
             "explanation": correct_explanation, "is_correct": True},
            {"letter": "B", "text": "d1", "explanation": "", "is_correct": False},
            {"letter": "C", "text": "d2", "explanation": "", "is_correct": False},
            {"letter": "D", "text": "d3", "explanation": "", "is_correct": False},
        ],
    }


def _ctx(labels_dict, testable_fact=""):
    return {
        "anchor_concept_labels": labels_dict,
        "anchor_testable_fact": testable_fact,
    }


class TestT3PlusEnforcement(unittest.TestCase):
    def setUp(self):
        self.gate = DomainExpertiseGate()

    def test_t3_with_two_technical_terms_passes(self):
        ctx = _ctx({"agonist": "Agonist Receptor Binding",
                    "antagonist": "Antagonist Mechanism"})
        q = _q(tier=3, correct_text="An agonist binds the antagonist's site.",
               correct_explanation="Receptor binding produces response.")
        ok, _ = self.gate.check(q, ctx)
        self.assertTrue(ok)

    def test_t3_with_single_technical_term_fails(self):
        ctx = _ctx({"agonist": "Agonist", "antagonist": "Antagonist"})
        q = _q(tier=3, correct_text="The patient improved over time.",
               correct_explanation="Recovery is common.")
        ok, reason = self.gate.check(q, ctx)
        self.assertFalse(ok)
        self.assertIn("technical term", reason.lower())

    def test_t4_same_threshold(self):
        ctx = _ctx({"agonist": "Agonist", "antagonist": "Antagonist"})
        q = _q(tier=4, correct_text="A simple answer.")
        ok, _ = self.gate.check(q, ctx)
        self.assertFalse(ok)

    def test_terms_from_testable_fact_count(self):
        ctx = _ctx({}, testable_fact="The hippocampus mediates declarative "
                                     "memory consolidation.")
        q = _q(tier=3, correct_text="The hippocampus shows declarative deficits.")
        ok, _ = self.gate.check(q, ctx)
        self.assertTrue(ok)


class TestT2SoftWarn(unittest.TestCase):
    def setUp(self):
        self.gate = DomainExpertiseGate()

    def test_t2_with_only_one_term_warns_but_passes(self):
        ctx = _ctx({"agonist": "Agonist Activity"})
        q = _q(tier=2, correct_text="Common-sense response.")
        ok, _ = self.gate.check(q, ctx)
        # T2 soft-warns and still passes
        self.assertTrue(ok)


class TestT1NotGated(unittest.TestCase):
    def test_t1_passes_regardless(self):
        gate = DomainExpertiseGate()
        ctx = _ctx({"agonist": "Agonist"})
        q = _q(tier=1, correct_text="A bare term.")
        ok, _ = gate.check(q, ctx)
        self.assertTrue(ok)


class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.gate = DomainExpertiseGate()

    def test_no_context_passes(self):
        # Without brief vocabulary, the gate can't check.
        q = _q(tier=3)
        ok, _ = self.gate.check(q, None)
        self.assertTrue(ok)

    def test_empty_vocabulary_passes(self):
        q = _q(tier=3)
        ok, _ = self.gate.check(q, _ctx({}))
        self.assertTrue(ok)

    def test_common_words_filtered(self):
        # "system", "process", "model", "theory" are stripped from the
        # technical-term set even though they're 5+ chars. With ALL labels
        # filtered to nothing, the gate has empty vocabulary and passes
        # (it can't gate without a vocabulary baseline).
        ctx = _ctx({"a": "system process",
                    "b": "model theory"})
        q = _q(tier=3, correct_text="The system process is a theory model.")
        ok, _ = self.gate.check(q, ctx)
        # Gate returns True when no vocabulary is available.
        self.assertTrue(ok)

    def test_real_terms_alongside_common_words(self):
        # When some labels survive filtering and others don't, the gate
        # should still correctly count using the non-common terms.
        ctx = _ctx({"a": "Hippocampal Declarative Memory",
                    "b": "system process"})  # 'b' fully filtered
        q = _q(tier=3, correct_text="The hippocampal region affects declarative recall.")
        ok, _ = self.gate.check(q, ctx)
        # 'hippocampal' + 'declarative' both present → ≥2 technical terms
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
