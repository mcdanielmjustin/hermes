"""Tests for the Bloom's tier-identity layer.

Three gates with curated structural enforcement, one per tier (T3 has
its own dedicated test file in test_apply_identity_gate.py):

  • RememberIdentityGate (T1) — negative scenario check + static-form correct
  • UnderstandIdentityGate (T2) — brevity cap + non-evaluative framing
  • EvaluateIdentityGate (T4) — complex stimulus + defensible distractors

Each gate is curated to its tier's cognitive identity. Same-shape
boilerplate would not work — T1's check is the inverse of T3's;
T2's check is about complexity ceiling; T4's check is about complexity
floor + distractor quality.
"""
import unittest

import conftest  # noqa: F401  — sets sys.path

from pipeline.gates import (
    RememberIdentityGate, UnderstandIdentityGate, EvaluateIdentityGate,
)


def _q(*, tier, stem, correct_text, distractor_texts=None):
    distractor_texts = distractor_texts or [
        "Distractor option one with substantial content reasonably similar to correct.",
        "Distractor option two with substantial content reasonably similar to correct.",
        "Distractor option three with substantial content reasonably similar to correct.",
    ]
    options = [{"letter": "A", "text": correct_text, "is_correct": True,
                "explanation": ""}]
    for L, t in zip(("B", "C", "D"), distractor_texts):
        options.append({"letter": L, "text": t, "is_correct": False,
                        "explanation": ""})
    return {
        "question_stem": stem,
        "options": options,
        "stem_pattern": "x",
        "difficulty_tier": tier,
    }


# ── RememberIdentityGate (T1) ──────────────────────────────────


class TestRememberIdentity_StemDirectness(unittest.TestCase):
    """T1 stems must NOT contain specific scenario indicators."""

    def setUp(self):
        self.gate = RememberIdentityGate()

    def test_direct_definitional_stem_passes(self):
        q = _q(
            tier=1,
            stem="Which of the following defines hemiplegia?",
            correct_text="Complete unilateral loss of voluntary movement.",
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_named_doctor_fails(self):
        q = _q(
            tier=1,
            stem="Dr. Mahlangu reviews imaging for a stroke patient. What is hemiplegia?",
            correct_text="Complete unilateral loss of voluntary movement.",
        )
        ok, reason = self.gate.check(q)
        self.assertFalse(ok, f"named doctor should fail T1; reason: {reason}")
        self.assertIn("scenario", reason.lower())

    def test_age_marker_fails(self):
        q = _q(
            tier=1,
            stem="A 67-year-old man asks: what is hemiplegia?",
            correct_text="Complete unilateral loss of voluntary movement.",
        )
        ok, _ = self.gate.check(q)
        self.assertFalse(ok)

    def test_the_patient_fails(self):
        q = _q(
            tier=1,
            stem="The patient asks the clinician to define hemiplegia.",
            correct_text="Complete unilateral loss of voluntary movement.",
        )
        ok, _ = self.gate.check(q)
        self.assertFalse(ok)

    def test_generic_patients_passes(self):
        # "patients" plural + generic context — NOT a specific scenario.
        # T1 should allow this.
        q = _q(
            tier=1,
            stem="In stroke patients, what does the term hemiplegia denote?",
            correct_text="Complete unilateral loss of voluntary movement.",
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok, "generic 'patients' (plural) should not trigger")

    def test_in_clinical_practice_passes(self):
        q = _q(
            tier=1,
            stem="In clinical practice, hemiplegia denotes which of the following?",
            correct_text="Complete unilateral loss of voluntary movement.",
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)


class TestRememberIdentity_StaticForm(unittest.TestCase):
    """T1 correct option must NOT start with forward-action verbs."""

    def setUp(self):
        self.gate = RememberIdentityGate()

    def test_definitional_correct_passes(self):
        q = _q(
            tier=1,
            stem="What is hemiplegia?",
            correct_text="Complete unilateral loss of voluntary movement.",
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_identify_verb_passes(self):
        # T1's own verb — allowed.
        q = _q(
            tier=1,
            stem="Which option correctly defines hemiplegia?",
            correct_text="Identify hemiplegia as complete unilateral paralysis.",
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok, "T1 'Identify' verb is allowed at recognition tier")

    def test_predict_verb_fails(self):
        q = _q(
            tier=1,
            stem="What is hemiplegia?",
            correct_text="Predict bilateral hemiplegia from the cortical lesion.",
        )
        ok, reason = self.gate.check(q)
        self.assertFalse(ok)
        self.assertIn("forward-action", reason.lower())

    def test_determine_verb_fails(self):
        q = _q(
            tier=1,
            stem="What is hemiplegia?",
            correct_text="Determine the lesion location from the symptom side.",
        )
        ok, _ = self.gate.check(q)
        self.assertFalse(ok)


class TestRememberIdentity_Exemptions(unittest.TestCase):
    def setUp(self):
        self.gate = RememberIdentityGate()

    def test_t2_exempt(self):
        # Same content but T2 — gate must bypass.
        q = _q(
            tier=2,
            stem="The patient asks: what is hemiplegia?",
            correct_text="Predict bilateral...",
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_t3_exempt(self):
        q = _q(tier=3, stem="x", correct_text="x")
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_t4_exempt(self):
        q = _q(tier=4, stem="x", correct_text="x")
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)


# ── UnderstandIdentityGate (T2) ────────────────────────────────


class TestUnderstandIdentity_Brevity(unittest.TestCase):
    """T2 stems must be ≤ 280 chars AND ≤ 40 words."""

    def setUp(self):
        self.gate = UnderstandIdentityGate()

    def test_brief_stem_passes(self):
        q = _q(
            tier=2,
            stem="What key distinction separates contralateral motor control from ipsilateral motor organization?",
            correct_text="Each cerebral hemisphere directs voluntary movement on the opposite side.",
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_long_vignette_stem_fails(self):
        long_stem = (
            "Mrs. Alvarez, a 67-year-old woman, was admitted to the emergency "
            "department after suddenly losing voluntary movement on the right "
            "side of her body following an episode at home. Imaging confirms an "
            "ischemic infarct in the left primary motor cortex with intact "
            "brainstem and spinal pathways below the medullary pyramids. "
            "Her family reports no prior history of stroke or transient ischemic attacks. "
            "Which of the following describes hemiplegia?"
        )
        q = _q(
            tier=2,
            stem=long_stem,
            correct_text="Complete unilateral loss of voluntary movement.",
        )
        ok, reason = self.gate.check(q)
        self.assertFalse(ok)
        self.assertIn("too long", reason.lower())


class TestUnderstandIdentity_NonEvaluativeFraming(unittest.TestCase):
    """T2 must not use T4-specific evaluative framings."""

    def setUp(self):
        self.gate = UnderstandIdentityGate()

    def test_most_appropriate_framing_fails(self):
        q = _q(
            tier=2,
            stem="Which is the MOST appropriate intervention for stroke recovery?",
            correct_text="Early-onset structured physical therapy.",
        )
        ok, reason = self.gate.check(q)
        self.assertFalse(ok)
        self.assertIn("evaluative framing", reason.lower())

    def test_critique_framing_fails(self):
        q = _q(
            tier=2,
            stem="Critique the resident's reasoning about hemiplegia mechanisms.",
            correct_text="The reasoning conflates ipsilateral with contralateral.",
        )
        ok, _ = self.gate.check(q)
        self.assertFalse(ok)

    def test_best_illustrates_passes(self):
        # Project's example_recognition stem pattern — legitimate T2.
        q = _q(
            tier=2,
            stem="Which scenario best illustrates the principle of contralateral motor control?",
            correct_text="A left-hemisphere stroke producing right-sided motor loss.",
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok, "'best illustrates' is a valid T2 framing")

    def test_most_accurate_restatement_passes(self):
        # Project's paraphrase stem pattern.
        q = _q(
            tier=2,
            stem="Which is the most accurate restatement of contralateral motor control?",
            correct_text="Each hemisphere governs voluntary movement on the opposite side.",
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(
            ok, "'most accurate restatement' is a valid T2 paraphrase framing",
        )


class TestUnderstandIdentity_Exemptions(unittest.TestCase):
    def setUp(self):
        self.gate = UnderstandIdentityGate()

    def test_t1_exempt(self):
        long_stem = "x" * 500
        q = _q(tier=1, stem=long_stem, correct_text="x")
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_t3_exempt(self):
        q = _q(tier=3, stem="MOST appropriate intervention?", correct_text="x")
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_t4_exempt(self):
        q = _q(tier=4, stem="MOST defensible position", correct_text="x")
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)


# ── EvaluateIdentityGate (T4) ──────────────────────────────────


class TestEvaluateIdentity_ComplexStimulus(unittest.TestCase):
    """T4 stems require complex-stimulus markers."""

    def setUp(self):
        self.gate = EvaluateIdentityGate()

    def _common_distractors(self):
        # Defensible distractors (high Jaccard with correct: each shares
        # `integrate`, `hemisphere`, `lesion`, `corticospinal`, `decussation`,
        # `hemiplegia` vocabulary with the correct option).
        return [
            (
                "Integrate right-hemisphere lesion with contralateral "
                "corticospinal decussation producing left-sided hemiplegia."
            ),
            (
                "Integrate left-hemisphere lesion with ipsilateral "
                "corticospinal projections producing right-sided hemiplegia."
            ),
            (
                "Integrate bilateral hemisphere lesion with corticospinal "
                "decussation producing bilateral hemiplegia patterns."
            ),
        ]

    def test_multi_sentence_stem_passes(self):
        q = _q(
            tier=4,
            stem=(
                "A patient presents with sudden right-sided hemiplegia "
                "following an embolic stroke. The resident argues the "
                "lesion must be in the left hemisphere. Evaluate."
            ),
            correct_text=(
                "Integrate left-hemisphere lesion with contralateral "
                "corticospinal decussation producing right-sided hemiplegia."
            ),
            distractor_texts=self._common_distractors(),
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok, "≥2 sentences = complex stimulus")

    def test_conjunctive_complexity_passes(self):
        q = _q(
            tier=4,
            stem=(
                "Predict the motor outcome whereas the resident argued "
                "ipsilateral involvement based on the embolic mechanism."
            ),
            correct_text=(
                "Integrate left-hemisphere lesion with contralateral "
                "corticospinal decussation producing right-sided hemiplegia."
            ),
            distractor_texts=self._common_distractors(),
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok, "'whereas' = conjunctive complexity")

    def test_competing_claim_passes(self):
        q = _q(
            tier=4,
            stem=(
                "The resident argues that pyramidal decussation explains "
                "the contralateral pattern in this hemiplegia presentation."
            ),
            correct_text=(
                "Integrate left-hemisphere lesion with contralateral "
                "corticospinal decussation producing right-sided hemiplegia."
            ),
            distractor_texts=self._common_distractors(),
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok, "'argues' = competing-claim marker")

    def test_simple_one_sentence_stem_fails(self):
        q = _q(
            tier=4,
            stem="Which describes contralateral motor control",
            correct_text=(
                "Integrate left-hemisphere lesion with contralateral "
                "corticospinal decussation producing right-sided hemiplegia."
            ),
            distractor_texts=self._common_distractors(),
        )
        ok, reason = self.gate.check(q)
        self.assertFalse(ok)
        self.assertIn("complex-stimulus", reason.lower())


class TestEvaluateIdentity_DistractorCheckRemoved(unittest.TestCase):
    """An earlier draft enforced "defensible distractors" via direct
    Jaccard overlap with correct. That check was removed because
    misconception-based distractors at T4 legitimately use the
    misconception's vocab (lowering direct overlap with correct
    even though the distractor IS engaging the stem's realm).
    TopicRealmGate already catches truly off-topic distractors.
    """

    def setUp(self):
        self.gate = EvaluateIdentityGate()

    _COMPLEX_STEM = (
        "A patient presents with sudden right-sided hemiplegia following "
        "an embolic stroke. Evaluate the resident's reasoning about the "
        "lesion site. Which integration is most defensible?"
    )

    def test_misconception_vocab_distractors_pass(self):
        # Misconception-based distractors (using their own vocab) are
        # legitimate at T4 — TopicRealmGate handles off-topic detection.
        q = _q(
            tier=4,
            stem=self._COMPLEX_STEM,
            correct_text=(
                "Integrate left-hemisphere lesion with contralateral "
                "corticospinal decussation producing right-sided hemiplegia."
            ),
            distractor_texts=[
                "Reframe as cerebellar dysfunction producing intention tremor.",
                "Reframe as bilateral motor pathway disruption.",
                "Reframe as Pavlovian conditioning of the motor response.",
            ],
        )
        # No defensible-distractor check anymore; only complex-stimulus.
        # Stem has multiple sentences and "Evaluate" marker → passes.
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)


class TestEvaluateIdentity_Exemptions(unittest.TestCase):
    def setUp(self):
        self.gate = EvaluateIdentityGate()

    def test_t1_t2_t3_exempt(self):
        for tier in (1, 2, 3):
            q = _q(tier=tier, stem="simple", correct_text="x")
            ok, _ = self.gate.check(q)
            self.assertTrue(ok, f"T{tier} must bypass EvaluateIdentityGate")


# ── Pipeline registration ──────────────────────────────────────


class TestPipelineRegistration(unittest.TestCase):
    """All four identity gates must be in create_gate_pipeline()."""

    def test_all_four_identity_gates_registered(self):
        from pipeline.gates import create_gate_pipeline
        names = [g.name for g in create_gate_pipeline()]
        self.assertIn("remember_identity", names)
        self.assertIn("understand_identity", names)
        self.assertIn("apply_identity", names)
        self.assertIn("evaluate_identity", names)


if __name__ == "__main__":
    unittest.main()
