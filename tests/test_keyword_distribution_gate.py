"""Tests for KeywordDistributionGate.

Catches the synonym-uniqueness testwise tell — content vocabulary that
appears only in the correct option, not in the stem or any distractor.
Empirical motivation: 29% of D7-PHY-209 non-EXCEPT survivors had this
pattern (e.g., E-02 stem says "one entire side / opposite cerebral
hemisphere" but only correct uses "unilateral / contralateral /
hemiplegia" — recognizable as the answer without engaging the concept).
"""
import unittest

import conftest  # noqa: F401  — sets sys.path

from pipeline.gates import KeywordDistributionGate


def _q(*, stem, correct_text, distractor_texts, stem_pattern="comparison",
       difficulty_tier=None):
    """Build a minimal question dict for gate testing."""
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


class TestSynonymUniquenessViolations(unittest.TestCase):
    def setUp(self):
        self.gate = KeywordDistributionGate()

    def test_calibration_pattern_fails(self):
        # Direct from D7-PHY-209 E-02: stem uses "one entire side / opposite
        # cerebral hemisphere"; correct uses technical synonyms
        # (unilateral, contralateral, hemiplegia) that distractors lack.
        q = _q(
            stem=("Complete loss of voluntary movement on one entire side of "
                  "the body, typically caused by a vascular lesion in the "
                  "opposite cerebral hemisphere, is known as which of the "
                  "following?"),
            correct_text=("Hemiplegia, complete unilateral paralysis from a "
                          "contralateral hemisphere lesion"),
            distractor_texts=[
                "Embolic infarction, identical in mechanism to thrombotic vessel occlusion",
                "Ipsilateral hemiparesis, paralysis on the same side as the cerebral lesion",
                "Cortical motor lesion producing direct, uncrossed weakness of the opposite limb",
            ],
        )
        ok, reason = self.gate.check(q)
        self.assertFalse(ok, msg=f"Should fail; reason was: {reason}")
        # The reason should name some unique words
        for term in ("hemiplegia", "unilateral", "contralateral"):
            self.assertIn(term, reason.lower(),
                          f"reason should mention {term} as unique-to-correct")

    def test_three_unique_words_fails(self):
        q = _q(
            stem="Which option correctly describes the mechanism?",
            correct_text="Decussation occurs at the medullary pyramid via crossing fibers",
            distractor_texts=[
                "The signal travels straight without crossing.",
                "The pathway is bilateral and synchronous.",
                "Direct cortical projection to the muscle.",
            ],
        )
        ok, reason = self.gate.check(q)
        self.assertFalse(ok, msg=f"Should fail; reason was: {reason}")


class TestCleanCases(unittest.TestCase):
    def setUp(self):
        self.gate = KeywordDistributionGate()

    def test_shared_vocabulary_passes(self):
        # All four options use the same technical vocabulary, just
        # arranged into different (right vs wrong) claims.
        q = _q(
            stem="Which describes how an antagonist affects the receptor?",
            correct_text="An antagonist binds the receptor without producing intrinsic activity",
            distractor_texts=[
                "An antagonist produces intrinsic activity at the receptor",
                "An antagonist blocks intrinsic activity at the receptor only when bound elsewhere",
                "An antagonist prevents binding without affecting intrinsic activity directly",
            ],
        )
        ok, reason = self.gate.check(q)
        self.assertTrue(ok, msg=f"Should pass; reason was: {reason}")

    def test_two_unique_words_passes(self):
        # Below the threshold of 3 — natural variation, not a tell.
        q = _q(
            stem="Which option describes the receptor mechanism?",
            correct_text="The receptor activates after agonist binding",
            distractor_texts=[
                "The receptor blocks agonist binding completely",
                "The receptor prevents agonist response permanently",
                "The receptor inhibits agonist signal indefinitely",
            ],
        )
        ok, reason = self.gate.check(q)
        self.assertTrue(ok, msg=f"Should pass; reason was: {reason}")


class TestExceptPatternExemption(unittest.TestCase):
    def setUp(self):
        self.gate = KeywordDistributionGate()

    def test_feature_listing_pattern_exempt(self):
        # EXCEPT-style: feature_listing pattern. Vocabulary divergence is
        # structural — the correct is the off-topic option by design.
        # D7-PHY-209 E-05 had 8 unique-to-correct words but is a valid
        # EXCEPT question.
        q = _q(
            stem="Each of the following is true of hemiplegia EXCEPT which one?",
            correct_text=("It most commonly results from acute closed head "
                          "trauma rather than vascular causes"),
            distractor_texts=[
                "A left hemisphere lesion typically produces paralysis on the right side of the body.",
                "It involves complete loss of voluntary movement on only one side of the body.",
                "It can be caused by thrombosis or embolism affecting cerebral blood vessels.",
            ],
            stem_pattern="feature_listing",
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok, "feature_listing pattern must be exempt")

    def test_except_in_stem_text_exempt(self):
        # Even with comparison stem_pattern, if stem text contains EXCEPT
        # the gate exempts (defensive — pattern label might disagree with
        # actual stem wording).
        q = _q(
            stem="All of the following are true of hemiplegia except:",
            correct_text="It is caused by acute closed head trauma in most cases",
            distractor_texts=[
                "It causes contralateral paralysis.",
                "It results from cerebral vascular events.",
                "It involves voluntary movement loss.",
            ],
            stem_pattern="comparison",  # mislabeled, but stem says EXCEPT
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok, "stem-text EXCEPT must trigger exemption")

    def test_normal_stem_not_exempt(self):
        q = _q(
            stem="Which option describes hemiplegia?",
            correct_text="Hemiplegia is unilateral paralysis from contralateral lesion of decussating fibers",
            distractor_texts=[
                "It affects bilateral motor function.",
                "It reduces sensation on the same side.",
                "It causes peripheral nerve damage.",
            ],
        )
        ok, _ = self.gate.check(q)
        self.assertFalse(ok, "non-EXCEPT pattern must NOT be exempt")


class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.gate = KeywordDistributionGate()

    def test_no_options_passes(self):
        q = {"question_stem": "test", "options": [], "stem_pattern": "comparison"}
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_no_correct_passes(self):
        q = {
            "question_stem": "test",
            "options": [
                {"letter": "A", "text": "no", "is_correct": False, "explanation": ""},
            ],
            "stem_pattern": "comparison",
        }
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_no_distractors_passes(self):
        q = {
            "question_stem": "test",
            "options": [
                {"letter": "A", "text": "answer", "is_correct": True, "explanation": ""},
            ],
            "stem_pattern": "comparison",
        }
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_empty_correct_text_passes(self):
        q = _q(
            stem="test stem",
            correct_text="",
            distractor_texts=["dist1", "dist2", "dist3"],
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_substring_match_handles_plurals(self):
        # Direct test of substring-matching at the helper level: singular
        # in correct, plural in distractors. "antagonist" is a substring
        # of "antagonists", so it should NOT count as unique-to-correct.
        gate = self.gate
        # Use the helper directly to isolate the substring-match behavior
        # from the threshold/aggregate logic.
        self.assertTrue(
            gate._word_appears_in("antagonist",
                                  ["The antagonists block the pathway"]),
            "singular form must match in plural-containing text",
        )
        self.assertFalse(
            gate._word_appears_in("antagonists", ["The antagonist binds"]),
            "plural form does NOT match singular (asymmetric — known)",
        )

    def test_stem_words_count_as_shared(self):
        # If a word appears in the stem, it can't be "unique to correct."
        q = _q(
            stem="Decussation describes the crossing of motor fibers at the medullary level",
            correct_text="Decussation at the medullary pyramid via crossing fibers",
            distractor_texts=[
                "No crossing of fibers in this case.",
                "Bilateral synchronous travel.",
                "Direct projection to the muscle.",
            ],
        )
        # decussation, medullary, pyramid, crossing, fibers all in stem.
        # After stem-subtraction, correct shouldn't have many unique words.
        ok, _ = self.gate.check(q)
        self.assertTrue(ok, "words in stem don't count as unique-to-correct")


class TestThresholdBoundary(unittest.TestCase):
    def test_exactly_at_threshold_fails(self):
        gate = KeywordDistributionGate()
        # Construct a case with exactly 3 unique-to-correct words
        q = _q(
            stem="Which describes the result?",
            correct_text="Decussation crossing pyramidal mechanism activates",
            # decussation, crossing, pyramidal, mechanism = 4 unique words
            distractor_texts=[
                "Activates differently in this scenario only briefly.",
                "Activates with delay over time gradually.",
                "Activates the receptor directly without intermediates.",
            ],
        )
        ok, _ = gate.check(q)
        self.assertFalse(ok)


class TestTierKeyedThresholds(unittest.TestCase):
    """Phase 6: Bloom's-identity-respecting threshold inversion.

    T1/T2 (recognition / understand) — vocab IS the test, looser thresholds
    (5 / 4). T3/T4 (apply / analyze-evaluate) — vocab divergence IS a tell,
    strict thresholds (3 / 3, matching the prior uniform default).
    """

    # 4 unique-to-correct content words: decussation, crossing, pyramidal,
    # mechanism. (activates is shared with all distractors.)
    _STEM_4 = "Which option describes the result?"
    _CORRECT_4 = "Decussation crossing pyramidal mechanism activates."
    _DISTRACTORS_4 = [
        "Activates differently in this scenario only briefly.",
        "Activates with delay over time gradually.",
        "Activates the receptor directly without intermediates.",
    ]

    # 3 unique-to-correct content words: pyramidal, decussation, crossing.
    _STEM_3 = "Which option describes the activated mechanism?"
    _CORRECT_3 = "Pyramidal decussation crossing activated mechanism."
    _DISTRACTORS_3 = [
        "Receptor blocks the activated mechanism completely.",
        "Antagonist prevents activated mechanism response permanently.",
        "Inhibitor delays activated mechanism signal indefinitely.",
    ]

    def setUp(self):
        self.gate = KeywordDistributionGate()

    def test_t1_below_threshold_passes(self):
        # T1 threshold is 5; 4 unique words is below.
        q = _q(stem=self._STEM_4, correct_text=self._CORRECT_4,
               distractor_texts=self._DISTRACTORS_4, difficulty_tier=1)
        ok, reason = self.gate.check(q)
        self.assertTrue(
            ok, f"T1 with 4 unique words should pass (loose threshold): {reason}"
        )

    def test_t1_at_threshold_fails(self):
        # T1 threshold is 5; need 5 unique. Add "hemiplegia" → 5 uniques.
        correct = "Decussation crossing pyramidal mechanism hemiplegia activates."
        q = _q(stem=self._STEM_4, correct_text=correct,
               distractor_texts=self._DISTRACTORS_4, difficulty_tier=1)
        ok, _ = self.gate.check(q)
        self.assertFalse(ok, "T1 with 5 unique words should fail at threshold")

    def test_t2_below_threshold_passes(self):
        # T2 threshold is 4; 3 unique words is below.
        q = _q(stem=self._STEM_3, correct_text=self._CORRECT_3,
               distractor_texts=self._DISTRACTORS_3, difficulty_tier=2)
        ok, reason = self.gate.check(q)
        self.assertTrue(
            ok, f"T2 with 3 unique words should pass (loose threshold): {reason}"
        )

    def test_t2_at_threshold_fails(self):
        # T2 threshold is 4; 4 unique words triggers it.
        q = _q(stem=self._STEM_4, correct_text=self._CORRECT_4,
               distractor_texts=self._DISTRACTORS_4, difficulty_tier=2)
        ok, _ = self.gate.check(q)
        self.assertFalse(ok, "T2 with 4 unique words should fail at threshold")

    def test_t3_at_strict_threshold_fails(self):
        # T3 threshold is 3 (strict); 3 unique words triggers it.
        q = _q(stem=self._STEM_3, correct_text=self._CORRECT_3,
               distractor_texts=self._DISTRACTORS_3, difficulty_tier=3)
        ok, _ = self.gate.check(q)
        self.assertFalse(
            ok, "T3 with 3 unique words MUST fail (strict — Bloom's identity)"
        )

    def test_t4_at_strict_threshold_fails(self):
        # T4 threshold is 3 (strict); 3 unique words triggers it.
        q = _q(stem=self._STEM_3, correct_text=self._CORRECT_3,
               distractor_texts=self._DISTRACTORS_3, difficulty_tier=4)
        ok, _ = self.gate.check(q)
        self.assertFalse(
            ok, "T4 with 3 unique words MUST fail (strict — Bloom's identity)"
        )

    def test_missing_tier_uses_strict_default(self):
        # Defensive: a malformed question without difficulty_tier still gets
        # the strict threshold so it can never bypass the gate.
        q = _q(stem=self._STEM_3, correct_text=self._CORRECT_3,
               distractor_texts=self._DISTRACTORS_3)  # no tier
        ok, _ = self.gate.check(q)
        self.assertFalse(
            ok, "missing tier must default to strict (3) so malformed Q "
                "cannot bypass the gate"
        )


class TestT3GenericEnglishExclusion(unittest.TestCase):
    """T3+ filters generic English from the unique-word count so the
    gate catches technical-synonym tells, not descriptive-English
    asymmetry between options."""

    def setUp(self):
        self.gate = KeywordDistributionGate()

    def test_t3_drops_generic_descriptors(self):
        # 3 unique-to-correct content words at min_len 7: 'fluctuations'
        # (12, generic), 'reflecting' (10, generic), 'irritability' (12,
        # canonical psych). With T3 generic-English filter, only the
        # canonical term remains (1 < 3) → passes.
        # Pre-T3-filter the gate would fire (3 unique ≥ threshold 3).
        q = _q(
            stem="Following frontal lobe injury, the clinical pattern develops.",
            correct_text=(
                "Behavioral disturbance with fluctuations reflecting "
                "irritability across situations."
            ),
            distractor_texts=[
                "Behavioral disturbance worsens during specific clinical situations.",
                "Behavioral disturbance shifts gradually across clinical situations.",
                "Behavioral disturbance localizes to cerebellum across situations.",
            ],
            difficulty_tier=3,
        )
        ok, reason = self.gate.check(q)
        self.assertTrue(
            ok,
            "T3 should pass when 2 of 3 unique words are generic English "
            f"(fluctuations, reflecting). Reason: {reason}",
        )

    def test_t1_keeps_generic_descriptors(self):
        # Same content at T1 — generic English IS countable at recognition
        # tier (vocabulary IS the test). 3 unique words but T1 threshold
        # is 5 → still passes anyway. The point: generic filter is gated
        # to T3+ only.
        q = _q(
            stem="What describes the patient's condition?",
            correct_text="Apathy and rapid reflecting episodes characterize this.",
            distractor_texts=[
                "Distractor one describes a different pattern.",
                "Distractor two describes a different pattern.",
                "Distractor three describes a different pattern.",
            ],
            difficulty_tier=1,
        )
        # T1 threshold is 5; below it. Should pass.
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)


class TestT3CanonicalVocabFilter(unittest.TestCase):
    """T3+ restricts the unique-word count to canonical technical
    vocabulary (curated domain pool + brief concept terms). Generic
    English asymmetry between options is question content, not a tell."""

    def setUp(self):
        self.gate = KeywordDistributionGate()

    def test_t3_with_canonical_context_filters_to_canonical_only(self):
        # 4 unique-to-correct content words at 7+ chars: 'apathy' won't
        # land (6 chars; min_len 7 at T3), but 'fluctuations' (12),
        # 'lability' (8), 'episodic' (8), 'consolidated' (12) all land.
        # Canonical = {'lability', 'consolidated'} via context. Other 2
        # are filtered. Below threshold 3 → passes.
        q = _q(
            stem="In the patient's presentation, which describes the syndrome?",
            correct_text=(
                "Episodic fluctuations with lability constitute the consolidated "
                "syndrome's signature presentation."
            ),
            distractor_texts=[
                "The presentation features ipsilateral motor deficit only.",
                "The presentation involves cerebellar tremor with intact cognition.",
                "The presentation includes sensorimotor numbness without affect.",
            ],
            difficulty_tier=3,
        )
        # Without context: 4 unique words → fails at T3 threshold 3
        ok_no_ctx, _ = self.gate.check(q)
        self.assertFalse(ok_no_ctx, "without context, 4 unique words triggers")

        # With context restricting to canonical {lability, consolidated}:
        # only 2 of the 4 survive → below threshold → passes
        ctx = {
            "domain_vocab": ["lability", "consolidated"],
            "brief_concept_terms": [],
        }
        ok_with_ctx, _ = self.gate.check(q, ctx)
        self.assertTrue(
            ok_with_ctx,
            "with canonical context, only canonical-vocab words count",
        )

    def test_canonical_filter_skipped_at_t1_t2(self):
        # T1/T2 ignore the canonical filter — vocab IS the test there.
        # Construct 5 unique-to-correct words at 5+ chars; none in the
        # canonical-pool we'll pass. T1 threshold is 5 → fails.
        q = _q(
            stem="Which condition is described?",
            correct_text=(
                "Hemiplegia, paralysis, contralateral, decussation, fibers."
            ),
            distractor_texts=[
                "Other distractor option text.",
                "Yet another distractor option.",
                "Final distractor option text.",
            ],
            difficulty_tier=1,
        )
        ctx = {
            "domain_vocab": ["zzzzzzzzzz"],  # nothing canonical matches
            "brief_concept_terms": [],
        }
        # T1 threshold 5; 5+ unique words all 6+ chars → fails
        # Canonical filter is skipped at T1, so all unique words count.
        ok, _ = self.gate.check(q, ctx)
        self.assertFalse(
            ok,
            "T1 ignores canonical filter — vocab IS the test at recognition",
        )


class TestT3MinWordLength(unittest.TestCase):
    """T3+ raises min word length from 5 to 7 to filter short generic
    English (rapid, triad, etc.) from the unique-word count."""

    def setUp(self):
        self.gate = KeywordDistributionGate()

    def test_t3_drops_5_6_char_words(self):
        # Correct uses 5-6 char common words ('triad', 'rapid', 'severe').
        # At T3 these are below min_len 7 → not counted → passes.
        q = _q(
            stem="What describes the patient's pattern?",
            correct_text="Rapid severe triad with consistent characteristics overall.",
            distractor_texts=[
                "The pattern emerges gradually with mild characteristics.",
                "The pattern follows reversible course without consistent features.",
                "The pattern predicts cerebellar pathology with characteristics.",
            ],
            difficulty_tier=3,
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(
            ok,
            "T3 should ignore 5-6 char generic English in unique-word count",
        )

    def test_t1_keeps_5_6_char_words(self):
        # T1 keeps 5-char floor — 'rapid', 'triad' count as content words
        # there. Threshold 5 still applies though.
        q = _q(
            stem="Pattern?",
            correct_text="Rapid severe triad consistent reflecting pattern overall.",
            distractor_texts=[
                "Different word one only.",
                "Different word two only.",
                "Different word three only.",
            ],
            difficulty_tier=1,
        )
        # 5 unique words ≥ 5 char + T1 threshold 5 → fails
        ok, _ = self.gate.check(q)
        self.assertFalse(ok, "T1 keeps 5-char floor")


if __name__ == "__main__":
    unittest.main()
