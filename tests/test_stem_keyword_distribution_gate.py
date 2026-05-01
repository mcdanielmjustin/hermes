"""Tests for StemKeywordDistributionGate.

Catches the stem-keyword-in-correct-only tell — words that appear in
both the stem AND the correct option but in NO distractor. Students
match the stem's keywords to the option containing them.

Tighter than KeywordDistributionGate (threshold 2 vs 3) because stem-
keyword leakage is a more direct match-the-words tell.
"""
import unittest

import conftest  # noqa: F401  — sets sys.path

from pipeline.gates import StemKeywordDistributionGate


def _q(*, stem, correct_text, distractor_texts, stem_pattern="comparison",
       difficulty_tier=None):
    options = [{"letter": "A", "text": correct_text, "is_correct": True,
                "explanation": ""}]
    for L, t in zip(("B", "C", "D"), distractor_texts):
        options.append({"letter": L, "text": t, "is_correct": False,
                        "explanation": ""})
    q = {
        "question_stem": stem,
        "options": options,
        "stem_pattern": stem_pattern,
    }
    if difficulty_tier is not None:
        q["difficulty_tier"] = difficulty_tier
    return q


class TestStemKeywordTellViolations(unittest.TestCase):
    def setUp(self):
        self.gate = StemKeywordDistributionGate()

    def test_calibration_pattern_fails(self):
        # D7-PHY-209 X-03: stem mentions ipsilateral / lesion / loss; only
        # correct repeats them.
        q = _q(
            stem=("Dr. Maldonado is reviewing a case with sudden complete "
                  "ipsilateral motor loss following a contralateral lesion."),
            correct_text=("Embolic hemiplegia reflects voluntary motor loss "
                          "contralateral to the lesion, not ipsilateral"),
            distractor_texts=[
                "Right-hemisphere embolism produces right-sided contralateral hemiplegia",
                "Embolic hemiplegia is bilateral voluntary motor disruption",
                "Decussation is irrelevant; thrombosis alone explains the deficit",
            ],
        )
        ok, reason = self.gate.check(q)
        self.assertFalse(ok, msg=f"Should fail; reason was: {reason}")
        self.assertIn("stem keyword", reason.lower())

    def test_two_leaked_keywords_fails(self):
        q = _q(
            stem="The pyramidal pathway projects motor signals via decussation at the medulla.",
            correct_text="Pyramidal fibers decussate at the medulla before descending.",
            distractor_texts=[
                "Motor signals travel directly without crossing.",
                "The pathway is bilateral and synchronous to both sides.",
                "Direct cortical projection to spinal motor neurons.",
            ],
        )
        ok, reason = self.gate.check(q)
        self.assertFalse(ok, msg=f"Should fail; reason was: {reason}")


class TestCleanCases(unittest.TestCase):
    def setUp(self):
        self.gate = StemKeywordDistributionGate()

    def test_keywords_distributed_passes(self):
        q = _q(
            stem="What does receptor antagonism do to the signal?",
            correct_text="Receptor antagonism blocks the signal at the binding site.",
            distractor_texts=[
                "Receptor antagonism enhances the signal pathway.",
                "Receptor antagonism produces an identical signal as the agonist.",
                "Receptor antagonism causes irreversible signal damage.",
            ],
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_one_leaked_keyword_passes(self):
        # Threshold is 2; a single shared-stem-and-correct keyword is OK.
        q = _q(
            stem="Which compound demonstrates antagonist binding behavior?",
            correct_text="Compound X demonstrates antagonist activity in the assay.",
            distractor_texts=[
                "Compound Y produces an agonist response in the assay.",
                "Compound Z behaves as a partial agonist at the receptor.",
                "Compound W shows reuptake inhibition not antagonist activity.",
            ],
        )
        # 'antagonist' appears in stem, correct, and distractor C+D — distributed
        # 'demonstrates' might be unique but it's only 1 word
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)


class TestExceptExemption(unittest.TestCase):
    def setUp(self):
        self.gate = StemKeywordDistributionGate()

    def test_feature_listing_exempt(self):
        q = _q(
            stem="Each of the following is true of hemiplegia EXCEPT:",
            correct_text="Hemiplegia commonly results from acute closed head trauma.",
            distractor_texts=[
                "Left hemisphere lesion produces right-sided paralysis.",
                "Hemiplegia involves voluntary movement loss on one side.",
                "Caused by thrombosis or embolism of cerebral vessels.",
            ],
            stem_pattern="feature_listing",
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok, "feature_listing pattern must be exempt")

    def test_except_in_stem_exempt(self):
        q = _q(
            stem="All of the following describe hemiplegia except:",
            correct_text="Hemiplegia is caused by acute closed head trauma.",
            distractor_texts=[
                "Hemiplegia produces contralateral motor deficits.",
                "Hemiplegia results from cerebrovascular events.",
                "Hemiplegia involves voluntary motor loss.",
            ],
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok, "stem-text EXCEPT must trigger exemption")


class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.gate = StemKeywordDistributionGate()

    def test_no_options_passes(self):
        q = {"question_stem": "stem", "options": [], "stem_pattern": "comparison"}
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_no_correct_passes(self):
        q = {
            "question_stem": "stem",
            "options": [
                {"letter": "A", "text": "no", "is_correct": False, "explanation": ""},
            ],
            "stem_pattern": "comparison",
        }
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_empty_stem_passes(self):
        q = _q(
            stem="",
            correct_text="Compound X demonstrates antagonist activity.",
            distractor_texts=["dist1", "dist2", "dist3"],
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_no_shared_keywords_passes(self):
        # If the correct doesn't repeat any stem words, the gate has
        # nothing to flag.
        q = _q(
            stem="Which option describes the receptor mechanism?",
            correct_text="An agonist activates after binding.",
            distractor_texts=[
                "An agonist blocks completely.",
                "An agonist prevents response.",
                "An agonist inhibits indefinitely.",
            ],
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)


class TestTierKeyedThresholds(unittest.TestCase):
    """Phase 6: Bloom's-identity-respecting threshold inversion.

    T1/T2 — vocabulary echo of stem keywords is the recognition act,
    threshold loosened to 3. T3/T4 — stem-keyword echo is the substituting-
    recognition-for-application tell, strict threshold 2 preserved.
    """

    # Exactly 2 leaked stem keywords: pyramidal, decussation. Distractors
    # avoid both as substrings.
    _STEM_2 = "The pyramidal tract conducts motor decussation in this case."
    _CORRECT_2 = "Pyramidal fibers undergo decussation at the inferior brainstem region."
    _DISTRACTORS_2 = [
        "Motor signals propagate directly without crossing.",
        "The bilateral pathway operates synchronous to either limb.",
        "Direct cortical projection delivers messages straight.",
    ]

    def setUp(self):
        self.gate = StemKeywordDistributionGate()

    def test_t1_two_leaked_passes(self):
        # T1 threshold is 3; 2 leaked keywords is below.
        q = _q(stem=self._STEM_2, correct_text=self._CORRECT_2,
               distractor_texts=self._DISTRACTORS_2, difficulty_tier=1)
        ok, reason = self.gate.check(q)
        self.assertTrue(
            ok, f"T1 with 2 leaked keywords should pass (loose threshold): {reason}"
        )

    def test_t1_three_leaked_fails(self):
        # T1 threshold is 3; 3 leaked keywords (pyramidal, decussation,
        # medulla) triggers it.
        stem = "The pyramidal pathway crosses via decussation through the medulla region."
        correct = "Pyramidal fibers cross via decussation in the medulla descending."
        distractors = [
            "Motor signals travel directly without crossing.",
            "The pathway is bilateral and synchronous to both sides.",
            "Direct cortical projection to spinal motor neurons.",
        ]
        q = _q(stem=stem, correct_text=correct,
               distractor_texts=distractors, difficulty_tier=1)
        ok, _ = self.gate.check(q)
        self.assertFalse(ok, "T1 with 3 leaked keywords should fail at threshold")

    def test_t2_two_leaked_passes(self):
        # T2 threshold is 3; 2 leaked keywords is below.
        q = _q(stem=self._STEM_2, correct_text=self._CORRECT_2,
               distractor_texts=self._DISTRACTORS_2, difficulty_tier=2)
        ok, reason = self.gate.check(q)
        self.assertTrue(
            ok, f"T2 with 2 leaked keywords should pass (loose threshold): {reason}"
        )

    def test_t3_two_leaked_fails(self):
        # T3 threshold is 2 (strict); 2 leaked keywords triggers it.
        q = _q(stem=self._STEM_2, correct_text=self._CORRECT_2,
               distractor_texts=self._DISTRACTORS_2, difficulty_tier=3)
        ok, _ = self.gate.check(q)
        self.assertFalse(
            ok, "T3 with 2 leaked keywords MUST fail (strict — Bloom's identity)"
        )

    def test_t4_two_leaked_fails(self):
        # T4 threshold is 2 (strict).
        q = _q(stem=self._STEM_2, correct_text=self._CORRECT_2,
               distractor_texts=self._DISTRACTORS_2, difficulty_tier=4)
        ok, _ = self.gate.check(q)
        self.assertFalse(
            ok, "T4 with 2 leaked keywords MUST fail (strict — Bloom's identity)"
        )

    def test_missing_tier_uses_strict_default(self):
        # Defensive: malformed Q without difficulty_tier gets strict default.
        q = _q(stem=self._STEM_2, correct_text=self._CORRECT_2,
               distractor_texts=self._DISTRACTORS_2)  # no tier
        ok, _ = self.gate.check(q)
        self.assertFalse(
            ok, "missing tier must default to strict (2) so malformed Q "
                "cannot bypass the gate"
        )


if __name__ == "__main__":
    unittest.main()
