"""Tests for OptionClaimGate.

Catches options whose `text` field contains reasoning markers ("because",
"since", "due to") that turn the answer into a self-justifying claim,
spoon-feeding the student a matching cue.
"""
import unittest

import conftest  # noqa: F401  — sets sys.path

from pipeline.gates import OptionClaimGate


def _q(*, options_texts):
    """options_texts: list of (letter, text, is_correct) tuples."""
    return {
        "options": [
            {"letter": l, "text": t, "is_correct": c, "explanation": ""}
            for l, t, c in options_texts
        ],
    }


class TestOptionClaimViolations(unittest.TestCase):
    def setUp(self):
        self.gate = OptionClaimGate()

    def test_because_in_correct_fails(self):
        q = _q(options_texts=[
            ("A", "It is acting as an agonist because it binds the receptor.", True),
            ("B", "It is an antagonist that blocks the site.", False),
            ("C", "Reuptake inhibitor.", False),
            ("D", "Inverse agonist.", False),
        ])
        ok, reason = self.gate.check(q)
        self.assertFalse(ok)
        self.assertIn("because", reason.lower())

    def test_because_in_distractor_fails(self):
        q = _q(options_texts=[
            ("A", "An agonist that activates the receptor.", True),
            ("B", "An antagonist because it blocks neurotransmitter access.", False),
            ("C", "Reuptake inhibitor.", False),
            ("D", "Inverse agonist.", False),
        ])
        ok, reason = self.gate.check(q)
        self.assertFalse(ok)

    def test_due_to_fails(self):
        q = _q(options_texts=[
            ("A", "Antagonist due to lack of intrinsic activity.", True),
            ("B", "Agonist effect.", False),
            ("C", "Reuptake inhibitor.", False),
            ("D", "Inverse agonist.", False),
        ])
        ok, _ = self.gate.check(q)
        self.assertFalse(ok)

    def test_since_fails(self):
        q = _q(options_texts=[
            ("A", "An antagonist since it produces no biological effect.", True),
            ("B", "An agonist.", False),
            ("C", "Inhibitor.", False),
            ("D", "Inverse agonist.", False),
        ])
        ok, _ = self.gate.check(q)
        self.assertFalse(ok)


class TestOptionClaimCleanCases(unittest.TestCase):
    def setUp(self):
        self.gate = OptionClaimGate()

    def test_pure_claim_passes(self):
        q = _q(options_texts=[
            ("A", "An antagonist that blocks dopamine binding.", True),
            ("B", "An agonist that activates dopamine receptors.", False),
            ("C", "A reuptake inhibitor.", False),
            ("D", "An inverse agonist.", False),
        ])
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_comparative_connectives_allowed(self):
        # 'whereas' and 'but' are comparative, not causal — allowed
        q = _q(options_texts=[
            ("A", "An agonist activates, whereas an antagonist blocks.", True),
            ("B", "An agonist blocks, whereas an antagonist activates.", False),
            ("C", "Both activate.", False),
            ("D", "Both block.", False),
        ])
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_however_allowed(self):
        q = _q(options_texts=[
            ("A", "Procedural memory is preserved; however, declarative is impaired.", True),
            ("B", "Both preserved.", False),
            ("C", "Both impaired.", False),
            ("D", "Reverse pattern.", False),
        ])
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)


class TestOptionClaimEdgeCases(unittest.TestCase):
    def setUp(self):
        self.gate = OptionClaimGate()

    def test_empty_text_does_not_crash(self):
        q = _q(options_texts=[
            ("A", "", True), ("B", "", False), ("C", "", False), ("D", "", False),
        ])
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_word_boundary_protects_substrings(self):
        # "bezier" contains "be" but no reasoning marker; should pass
        q = _q(options_texts=[
            ("A", "Bezier curves describe the activity.", True),
            ("B", "Linear functions describe.", False),
            ("C", "Quadratic functions.", False),
            ("D", "Cubic functions.", False),
        ])
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
