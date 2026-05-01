"""Tests for ScopeMatchGate.

Catches the asymmetric-scope problem on comparison/contrast/best-answer
stems: when the correct answer addresses 2+ concepts but a distractor
mentions only one, students rule out the under-scoped distractor without
reasoning about the actual concepts.
"""
import unittest

import conftest  # noqa: F401  — sets sys.path

from pipeline.gates import ScopeMatchGate


def _q(*, stem_pattern="comparison", correct_text="X and Y differ.",
       distractor_texts=None, correct_letter="A"):
    distractor_texts = distractor_texts or ["d1", "d2", "d3"]
    options = []
    di = iter(distractor_texts)
    for L in ("A", "B", "C", "D"):
        if L == correct_letter:
            options.append({"letter": L, "text": correct_text,
                            "is_correct": True})
        else:
            options.append({"letter": L, "text": next(di),
                            "is_correct": False})
    return {"stem_pattern": stem_pattern, "options": options}


def _ctx_two_concepts():
    return {
        "anchor_concept_labels": {
            "agonist": "Agonist",
            "antagonist": "Antagonist",
        },
    }


class TestScopeMatchPositive(unittest.TestCase):
    def setUp(self):
        self.gate = ScopeMatchGate()

    def test_symmetric_distractors_pass(self):
        # Correct mentions both; all distractors mention both
        q = _q(
            correct_text="An agonist activates; an antagonist blocks.",
            distractor_texts=[
                "An agonist blocks; an antagonist activates.",
                "Both agonist and antagonist activate the receptor.",
                "Both agonist and antagonist block the receptor.",
            ],
        )
        ok, _ = self.gate.check(q, _ctx_two_concepts())
        self.assertTrue(ok)

    def test_one_sided_distractor_fails(self):
        q = _q(
            correct_text="An agonist activates; an antagonist blocks.",
            distractor_texts=[
                "An agonist increases neurotransmitter availability.",  # only agonist
                "Both agonist and antagonist activate the receptor.",
                "Both agonist and antagonist block the receptor.",
            ],
        )
        ok, reason = self.gate.check(q, _ctx_two_concepts())
        self.assertFalse(ok)
        self.assertIn("scope", reason.lower())

    def test_pattern_exemption_for_non_comparison(self):
        # direct_definition stem doesn't require symmetric scope
        q = _q(
            stem_pattern="direct_definition",
            correct_text="An agonist activates; an antagonist blocks.",
            distractor_texts=["agonist alone", "wrong claim 1", "wrong claim 2"],
        )
        ok, _ = self.gate.check(q, _ctx_two_concepts())
        self.assertTrue(ok)


class TestScopeMatchEdgeCases(unittest.TestCase):
    def setUp(self):
        self.gate = ScopeMatchGate()

    def test_correct_with_single_concept_passes(self):
        # If the correct answer itself only mentions 1 concept, the gate
        # has no asymmetry to enforce.
        q = _q(
            correct_text="An agonist activates the receptor.",
            distractor_texts=["d1", "d2", "d3"],
        )
        ok, _ = self.gate.check(q, _ctx_two_concepts())
        self.assertTrue(ok)

    def test_no_context_passes(self):
        q = _q()
        ok, _ = self.gate.check(q, None)
        self.assertTrue(ok)

    def test_no_concept_labels_passes(self):
        q = _q()
        ok, _ = self.gate.check(q, {"anchor_concept_labels": {}})
        self.assertTrue(ok)

    def test_best_answer_pattern_also_gated(self):
        q = _q(
            stem_pattern="best_answer",
            correct_text="An agonist activates; an antagonist blocks.",
            distractor_texts=["only agonist talk", "both", "both"],
        )
        ok, _ = self.gate.check(q, _ctx_two_concepts())
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
