"""Tests for ApplyIdentityGate.

Bloom's Apply tier identity check (T3 only). Two structural requirements:
  (a) Stem contains a novel scenario (named subject, clinical context,
      "patient" / "client" framing, or "given X" / "after Y" setup).
  (b) Correct option's application verb is followed by ≥4 content words
      (a meaningful prediction, not a verb tagged onto a bare label).

Other tiers (T1/T2/T4) skip the gate. T1/T2 are recognition/description;
T4 has its own integration check via BloomsCognitiveLevelGate.
"""
import unittest

import conftest  # noqa: F401  — sets sys.path

from pipeline.gates import ApplyIdentityGate


def _q(*, stem, correct_text, distractor_texts=None, difficulty_tier=3,
       stem_pattern="case_analysis"):
    distractor_texts = distractor_texts or [
        "Predict bilateral hemiplegia from corticospinal projections crossing "
        "to both hemispheres equally during voluntary motor output.",
        "Predict ipsilateral hemiplegia from pyramidal fibers projecting to "
        "the same-side hemisphere despite cortical decussation.",
        "Predict transient unilateral weakness from cortical lesion without "
        "any pyramidal-tract involvement at all.",
    ]
    options = [{"letter": "A", "text": correct_text, "is_correct": True,
                "explanation": ""}]
    for L, t in zip(("B", "C", "D"), distractor_texts):
        options.append({"letter": L, "text": t, "is_correct": False,
                        "explanation": ""})
    return {
        "question_stem": stem,
        "options": options,
        "stem_pattern": stem_pattern,
        "difficulty_tier": difficulty_tier,
    }


class TestSceneNoveltyCheck(unittest.TestCase):
    """The stem must contain a scenario indicator at T3."""

    def setUp(self):
        self.gate = ApplyIdentityGate()

    def test_named_doctor_passes(self):
        q = _q(
            stem=(
                "Dr. Mahlangu reviews imaging for a patient with sudden "
                "right-sided motor loss after an embolic stroke. Predict "
                "the lesion site."
            ),
            correct_text=(
                "Predict left-hemisphere lesion from corticospinal "
                "decussation in the medullary pyramids producing "
                "contralateral hemiplegia."
            ),
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_age_marker_passes(self):
        q = _q(
            stem=(
                "A 67-year-old man arrives with sudden complete paralysis "
                "of the left arm and leg following an ischemic event."
            ),
            correct_text=(
                "Predict right-hemisphere lesion from corticospinal "
                "decussation producing left-sided hemiplegia in this "
                "presentation."
            ),
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_patient_keyword_passes(self):
        q = _q(
            stem=(
                "The patient presents with imaging confirming an acute "
                "right-hemisphere infarct. Predict motor consequences."
            ),
            correct_text=(
                "Predict left-sided hemiplegia from contralateral "
                "corticospinal decussation in the medullary pyramids."
            ),
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_definitional_stem_fails(self):
        # No scenario markers — reads as recall.
        q = _q(
            stem="What is the relationship between cerebral hemispheres and motor control?",
            correct_text=(
                "Predict that each hemisphere controls voluntary motor "
                "output on the contralateral side via corticospinal "
                "decussation in the medulla."
            ),
        )
        ok, reason = self.gate.check(q)
        self.assertFalse(
            ok, f"definitional stem should fail; reason: {reason}"
        )
        self.assertIn("scenario", reason.lower())

    def test_principle_application_passes(self):
        # "Applying the principle of X" is a recognized scenario framing.
        q = _q(
            stem=(
                "Following an acute stroke, applying the principle of "
                "corticospinal decussation, predict the motor outcome."
            ),
            correct_text=(
                "Predict contralateral hemiplegia from corticospinal "
                "decussation in the medullary pyramids contralateral "
                "to the lesion."
            ),
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)


class TestApplicationSubstance(unittest.TestCase):
    """The correct option must have verb + ≥4 content words."""

    _SCENARIO_STEM = (
        "Dr. Lane evaluates a 62-year-old patient with sudden right-sided "
        "motor loss after an ischemic stroke. Predict the lesion location."
    )

    def setUp(self):
        self.gate = ApplyIdentityGate()

    def test_rich_correct_passes(self):
        q = _q(
            stem=self._SCENARIO_STEM,
            correct_text=(
                "Predict left-hemisphere lesion from corticospinal "
                "decussation in the medullary pyramids producing "
                "contralateral hemiplegia."
            ),
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_bare_label_fails(self):
        q = _q(
            stem=self._SCENARIO_STEM,
            correct_text="Predict hemiplegia.",  # only 1 content word after verb
        )
        ok, reason = self.gate.check(q)
        self.assertFalse(
            ok, f"bare-label answer should fail; reason: {reason}"
        )
        self.assertIn("application substance", reason.lower())

    def test_verb_with_3_words_fails(self):
        # 3 content words after verb — below the threshold of 4.
        q = _q(
            stem=self._SCENARIO_STEM,
            correct_text="Predict left-sided hemiplegia from lesion.",  # 3 content words
        )
        ok, _ = self.gate.check(q)
        # "left-sided" tokenizes as 'sided' (5+ chars), 'hemiplegia',
        # 'lesion' = 3 content words — should fail
        self.assertFalse(ok, "≤3 content words after verb should fail")

    def test_no_application_verb_fails(self):
        q = _q(
            stem=self._SCENARIO_STEM,
            correct_text=(
                "Hemiplegia denotes complete unilateral loss of voluntary "
                "movement contralateral to the lesion in the medulla."
            ),
        )
        ok, _ = self.gate.check(q)
        self.assertFalse(
            ok, "no application verb in correct option should fail"
        )


class TestApplyIdentity_MechanismMarker(unittest.TestCase):
    """T3 correct option must contain a mechanism/causal marker, not
    just a triad-component label. Empirical: 3/5 T3 questions in the
    D7-PHY-058 Bloom's-identity validation batch drifted into the
    labeling-as-prediction pattern despite passing word-count checks."""

    _SCENARIO_STEM = (
        "Avi Bates, age 58, sustained a frontal lobe injury six weeks ago "
        "and now presents to Dr. Dodge with the classic post-injury triad."
    )

    def setUp(self):
        self.gate = ApplyIdentityGate()

    def test_labeling_as_prediction_fails(self):
        # "Predict X as part of Y" — no mechanism marker → drift.
        q = _q(
            stem=self._SCENARIO_STEM,
            correct_text=(
                "Predict ongoing mood fluctuations as part of the frontal "
                "lobe injury triad."
            ),
        )
        ok, reason = self.gate.check(q)
        self.assertFalse(
            ok, f"labeling-as-prediction should fail; reason: {reason}"
        )
        self.assertIn("mechanism", reason.lower())

    def test_lability_component_labeling_fails(self):
        # "Predict X as the Y component of Z" — labeling.
        q = _q(
            stem=self._SCENARIO_STEM,
            correct_text=(
                "Predict ongoing rapid mood fluctuations as the lability "
                "component of the frontal triad."
            ),
        )
        ok, _ = self.gate.check(q)
        self.assertFalse(ok)

    def test_alongside_labeling_fails(self):
        q = _q(
            stem=self._SCENARIO_STEM,
            correct_text=(
                "Predict rapid mood fluctuations alongside the rest of the "
                "frontal injury triad."
            ),
        )
        ok, _ = self.gate.check(q)
        self.assertFalse(ok)

    def test_reflecting_marker_passes(self):
        # "reflecting [mechanism]" — clean apply.
        q = _q(
            stem=self._SCENARIO_STEM,
            correct_text=(
                "Apply emotional lability within the frontal lobe injury "
                "triad, reflecting lost regulation of limbic output."
            ),
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_with_poor_regulation_passes(self):
        # "with poor [feature]" — clean apply.
        q = _q(
            stem=self._SCENARIO_STEM,
            correct_text=(
                "Select rapid mood fluctuations with poor regulation as "
                "the triad's emotional sign."
            ),
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_from_mechanism_passes(self):
        # "from [mechanism]" — clean apply.
        q = _q(
            stem=self._SCENARIO_STEM,
            correct_text=(
                "Predict left-sided hemiplegia from corticospinal "
                "decussation in the medullary pyramids."
            ),
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_via_mechanism_passes(self):
        q = _q(
            stem=self._SCENARIO_STEM,
            correct_text=(
                "Predict contralateral hemiplegia via pyramidal "
                "decussation affecting voluntary motor output."
            ),
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_by_verb_ing_passes(self):
        # "by [verb-ing]" — clean apply.
        q = _q(
            stem=self._SCENARIO_STEM,
            correct_text=(
                "Predict bilateral motor disruption by interrupting "
                "corticospinal projections at the medulla."
            ),
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)


class TestApplyIdentity_CriterionApplication(unittest.TestCase):
    """Phase 10 — criterion-application markers count as causal anchors
    at T3 alongside mechanism markers. Motivated by CPAT D3-PPA-034
    cross-anchor regression: ADHD/autism diagnostic criteria are
    fundamentally threshold-checking, not mechanism-application; the
    T3 cognition is 'apply this criterion to this case', and the
    criterion IS the causal anchor for the determination.

    Without these markers, the gate over-fires on legitimate criteria-
    driven T3 questions, costing 5/5 T3 questions in the CPAT batch.
    """

    _CRITERIA_STEM = (
        "Dr. Quinn evaluates Marcus, age 9, who presents with "
        "inattentive symptoms of 7-month duration across home and "
        "school. Determine whether ADHD criteria are met."
    )

    def setUp(self):
        self.gate = ApplyIdentityGate()

    def test_based_on_marker_passes(self):
        q = _q(
            stem=self._CRITERIA_STEM,
            correct_text=(
                "Determine ADHD criteria are met based on age-9 onset "
                "and pervasive two-setting presentation."
            ),
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_given_that_marker_passes(self):
        q = _q(
            stem=self._CRITERIA_STEM,
            correct_text=(
                "Determine ADHD criteria are met given that age-9 "
                "onset precedes the age-12 cutoff."
            ),
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_satisfying_threshold_marker_passes(self):
        q = _q(
            stem=self._CRITERIA_STEM,
            correct_text=(
                "Determine ADHD criteria are met, satisfying the "
                "age-12 onset threshold and two-setting pervasiveness."
            ),
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_failing_criterion_marker_passes(self):
        q = _q(
            stem=self._CRITERIA_STEM,
            correct_text=(
                "Determine ADHD criteria are unmet, failing the "
                "criterion of 6-month symptom persistence."
            ),
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_for_meeting_criterion_marker_passes(self):
        q = _q(
            stem=self._CRITERIA_STEM,
            correct_text=(
                "Determine ADHD criteria are satisfied for meeting "
                "the age-12 onset and two-setting requirements."
            ),
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_exceeds_cutoff_marker_passes(self):
        q = _q(
            stem=self._CRITERIA_STEM,
            correct_text=(
                "Determine ADHD criteria are unmet, exceeding the "
                "age-12 cutoff for symptom onset."
            ),
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_bare_label_decision_still_fails(self):
        # A determination without ANY anchor — neither mechanism nor
        # criterion — should still fail. The criterion path doesn't
        # weaken the gate; it broadens what counts as an anchor.
        q = _q(
            stem=self._CRITERIA_STEM,
            correct_text=(
                "Determine that ADHD criteria are met across the "
                "presented domains of childhood symptom onset."
            ),
        )
        ok, reason = self.gate.check(q)
        self.assertFalse(
            ok,
            "no mechanism OR criterion anchor → still labeling cognition; "
            f"reason: {reason}",
        )
        self.assertIn("criterion", reason.lower())


class TestTierExemptions(unittest.TestCase):
    """T1/T2/T4 are exempt from this gate."""

    def setUp(self):
        self.gate = ApplyIdentityGate()

    def test_t1_exempt_even_with_definitional_stem(self):
        q = _q(
            stem="What is hemiplegia?",  # definitional — would fail at T3
            correct_text="Hemiplegia is unilateral motor loss.",
            difficulty_tier=1,
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok, "T1 must bypass — recognition stems are valid")

    def test_t2_exempt(self):
        q = _q(
            stem="Which describes contralateral motor control?",
            correct_text="Each hemisphere controls the opposite side.",
            difficulty_tier=2,
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok, "T2 must bypass")

    def test_t4_exempt(self):
        # T4 has its own integration check, not Apply identity
        q = _q(
            stem="Synthesize the integration question",
            correct_text="Some long synthesis claim.",
            difficulty_tier=4,
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok, "T4 must bypass — has its own integration check")


class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.gate = ApplyIdentityGate()

    def test_no_options_passes(self):
        q = {"question_stem": "Dr. X examines patient", "options": [],
             "stem_pattern": "case_analysis", "difficulty_tier": 3}
        ok, _ = self.gate.check(q)
        # No correct option — bypass after stem-novelty check passes
        self.assertTrue(ok)

    def test_no_correct_passes(self):
        q = {
            "question_stem": "Dr. X examines patient",
            "options": [
                {"letter": "A", "text": "no", "is_correct": False,
                 "explanation": ""},
            ],
            "stem_pattern": "case_analysis",
            "difficulty_tier": 3,
        }
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_missing_difficulty_tier_passes(self):
        # No tier → not T3 → bypass (defensive, doesn't false-positive)
        q = _q(stem="Some stem", correct_text="Predict X")
        del q["difficulty_tier"]
        ok, _ = self.gate.check(q)
        self.assertTrue(ok, "missing tier must bypass")


class TestRegistration(unittest.TestCase):
    def test_apply_identity_in_pipeline(self):
        from pipeline.gates import create_gate_pipeline
        names = [g.name for g in create_gate_pipeline()]
        self.assertIn("apply_identity", names)


if __name__ == "__main__":
    unittest.main()
