"""Tests for AttributionGate and OptionLengthBalanceGate.

These gates were added to defeat three testwise heuristics:
  - Researcher attribution leakage (catches names the sanitizer missed)
  - Option-length tells (correct answer often longest)
  - Elaboration tells (parens/semicolons cluster on correct) — prompt-only

Gate API: .check(question, context=None) -> (ok: bool, reason: str)
"""
import unittest

import conftest  # noqa: F401  — sets sys.path

from pipeline.gates import AttributionGate, OptionLengthBalanceGate


def make_question(*, stem="A clean stem.", letters_lengths=None,
                  correct="A", explanations=None):
    """Build a minimal question dict for gate testing.

    letters_lengths: dict like {'A': 100, 'B': 100, 'C': 100, 'D': 100}.
                     If None, all options are length 100.
    explanations:    optional dict {'A': 'expl', ...} for explanation text.
    """
    if letters_lengths is None:
        letters_lengths = {"A": 100, "B": 100, "C": 100, "D": 100}
    explanations = explanations or {}
    return {
        "question_stem": stem,
        "tested_concept": {"knowledge_tested": ""},
        "options": [
            {
                "letter": L,
                "text": L.lower() * letters_lengths[L],
                "explanation": explanations.get(L, ""),
                "is_correct": (L == correct),
            }
            for L in ("A", "B", "C", "D")
        ],
    }


# ── AttributionGate ─────────────────────────────────────────────

class TestAttributionGateYearCitation(unittest.TestCase):
    def setUp(self):
        self.gate = AttributionGate()

    def test_flags_non_whitelisted_year_citation(self):
        q = make_question(stem="Squire (2004) found implicit memory preserved.")
        ok, reason = self.gate.check(q)
        self.assertFalse(ok)
        self.assertIn("Squire (2004)", reason)

    def test_passes_whitelisted_year_citation(self):
        q = make_question(stem="Piaget (1936) described sensorimotor stages.")
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_passes_year_letter_suffix_when_whitelisted(self):
        q = make_question(stem="Bandura (1977a) introduced self-efficacy.")
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_flags_year_in_explanation(self):
        q = make_question(
            stem="Clean stem.",
            explanations={"A": "Smith (1985) demonstrated this."},
        )
        ok, reason = self.gate.check(q)
        self.assertFalse(ok)
        self.assertIn("opt_A_explanation", reason)


class TestAttributionGateAccordingTo(unittest.TestCase):
    def setUp(self):
        self.gate = AttributionGate()

    def test_flags_according_to_non_whitelisted(self):
        q = make_question(stem="According to Smith, the model holds.")
        ok, _ = self.gate.check(q)
        self.assertFalse(ok)

    def test_passes_according_to_whitelisted(self):
        q = make_question(stem="According to Pavlov, conditioning extinguishes.")
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)


class TestAttributionGateEtAl(unittest.TestCase):
    def setUp(self):
        self.gate = AttributionGate()

    def test_flags_non_whitelisted_et_al(self):
        q = make_question(stem="Smith et al. demonstrated the effect.")
        ok, _ = self.gate.check(q)
        self.assertFalse(ok)

    def test_passes_whitelisted_et_al_bandura(self):
        # Regression: was failing before \band\b fix because "Bandura"
        # got split into ["B", "ura"] by the alternation.
        q = make_question(stem="Bandura et al. observed reciprocal determinism.")
        ok, _ = self.gate.check(q)
        self.assertTrue(ok, msg="Bandura must pass — \\band\\b fix regression check")


class TestAttributionGatePossessive(unittest.TestCase):
    def setUp(self):
        self.gate = AttributionGate()

    def test_flags_non_whitelisted_possessive(self):
        q = make_question(stem="Squire's framework explains amnesia.")
        ok, _ = self.gate.check(q)
        self.assertFalse(ok)

    def test_passes_whitelisted_possessive(self):
        q = make_question(stem="Piaget's stages map to age ranges.")
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_skips_unrelated_possessive_nouns(self):
        # "Squire's lab" — "lab" isn't in the research-noun list, so the
        # possessive regex doesn't match.
        q = make_question(stem="Squire's lab is in the basement.")
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)


class TestAttributionGateCleanText(unittest.TestCase):
    def setUp(self):
        self.gate = AttributionGate()

    def test_passes_clean_clinical_vignette(self):
        q = make_question(
            stem="Dr. Harding evaluates a 54-year-old client with hippocampal damage."
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_passes_eponymous_adjective_form(self):
        q = make_question(
            stem="The student must apply Pavlovian conditioning to a novel scenario."
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)


class TestAttributionGateInstitutionalDocuments(unittest.TestCase):
    """Document/organization names whitelisted to prevent false positives.

    Source data contains legitimate citations like "Forensic Psychology
    (2013)" referring to APA Specialty Guidelines. The regex captures
    "Psychology" as a name-shaped token; whitelist exempts it.
    """
    def setUp(self):
        self.gate = AttributionGate()

    def test_passes_psychology_document_citation(self):
        q = make_question(stem="The APA Guidelines for Forensic Psychology (2013) state...")
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)


# ── OptionLengthBalanceGate ─────────────────────────────────────

class TestOptionLengthBalanceGate(unittest.TestCase):
    def setUp(self):
        self.gate = OptionLengthBalanceGate()

    def test_passes_when_all_options_equal(self):
        q = make_question(letters_lengths={"A": 100, "B": 100, "C": 100, "D": 100})
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_passes_small_variance(self):
        q = make_question(letters_lengths={"A": 110, "B": 100, "C": 95, "D": 105})
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_fails_when_correct_is_2x_distractors(self):
        q = make_question(
            letters_lengths={"A": 200, "B": 100, "C": 100, "D": 100},
            correct="A",
        )
        ok, reason = self.gate.check(q)
        self.assertFalse(ok)
        self.assertIn("ratio", reason.lower())

    def test_fails_correct_more_than_20pct_longer_than_max_distractor(self):
        q = make_question(
            letters_lengths={"A": 130, "B": 100, "C": 100, "D": 100},
            correct="A",
        )
        ok, reason = self.gate.check(q)
        self.assertFalse(ok)
        self.assertIn("correct option", reason.lower())

    def test_fails_when_correct_is_D(self):
        # The 'tell' check must be position-agnostic — correct can be any letter.
        q = make_question(
            letters_lengths={"A": 100, "B": 100, "C": 100, "D": 130},
            correct="D",
        )
        ok, _ = self.gate.check(q)
        self.assertFalse(ok)

    def test_passes_correct_5pct_longer(self):
        q = make_question(
            letters_lengths={"A": 105, "B": 100, "C": 100, "D": 100},
            correct="A",
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_fails_high_spread_even_if_correct_isnt_longest(self):
        q = make_question(
            letters_lengths={"A": 50, "B": 100, "C": 100, "D": 50},
            correct="A",
        )
        ok, _ = self.gate.check(q)
        self.assertFalse(ok)

    def test_skips_check_if_not_four_options(self):
        # StructureGate owns this — length gate should pass-through.
        q = {"options": [{"letter": "A", "text": "x" * 100, "is_correct": True}]}
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_custom_thresholds(self):
        # Tighter gate: tell_margin=1.05 (5%), ratio_max=1.3.
        tight = OptionLengthBalanceGate(ratio_max=1.3, tell_margin=1.05)
        q = make_question(
            letters_lengths={"A": 110, "B": 100, "C": 100, "D": 100},
            correct="A",
        )
        ok, _ = tight.check(q)
        self.assertFalse(ok, "Tighter gate should fail at 10% margin")


if __name__ == "__main__":
    unittest.main()
