"""Tests for Layer B stem-integrity gates (Phase 9).

Covers:
  - LateralityIntegrityGate: stem 'bilateral' → distractor 'unilateral'
    (and vice versa) is a stem-eliminable contradiction.
  - UniversalDenialGate: stem cites a preserved counterexample +
    distractor uses universal-quantifier denial for the same category
    is a stem-eliminable contradiction.

Both gates are narrow heuristics. The Sonnet audit
(scripts/audit_stem_contradictions.py) catches the broader semantic
versions these regexes miss.
"""
import unittest

import conftest  # noqa: F401  — sets sys.path

from pipeline.gates import (
    LateralityIntegrityGate,
    UniversalDenialGate,
    create_gate_pipeline,
)


def _q(*, stem, options, tier=4, stem_pattern="case_synthesis"):
    return {
        "question_stem": stem,
        "options": options,
        "stem_pattern": stem_pattern,
        "difficulty_tier": tier,
    }


def _opt(letter, text, *, correct=False):
    return {
        "letter": letter,
        "text": text,
        "is_correct": correct,
        "explanation": "",
    }


class TestLateralityIntegrity(unittest.TestCase):
    def setUp(self):
        self.gate = LateralityIntegrityGate()

    # ─ Canonical fail cases (from D7-PHY-076 audit) ────────────────

    def test_bilateral_stem_unilateral_distractor_fails(self):
        # Direct from audit: X-02 distractor A
        q = _q(
            stem=(
                "Nadira Buckley sustained bilateral hippocampal damage "
                "in a hypoxic event eight months ago."
            ),
            options=[
                _opt("A", "Synthesize unilateral hippocampal injury "
                          "with anterograde declarative loss",
                     correct=False),
                _opt("B", "Synthesize bilateral hippocampal damage "
                          "producing severe anterograde amnesia",
                     correct=True),
                _opt("C", "Synthesize hippocampus-dependent loss",
                     correct=False),
                _opt("D", "Synthesize hippocampal mediation",
                     correct=False),
            ],
        )
        ok, reason = self.gate.check(q)
        self.assertFalse(ok, f"should flag unilateral distractor; reason: {reason}")
        self.assertIn("unilateral", reason.lower())
        self.assertIn("a", reason.lower())  # cites option A

    def test_unilateral_stem_bilateral_distractor_fails(self):
        # Inverse direction — stem says unilateral, distractor says bilateral
        q = _q(
            stem=(
                "Edgar Howard sustained a unilateral right-hemisphere "
                "lesion confirmed on imaging."
            ),
            options=[
                _opt("A", "Predict contralateral hemiplegia from the "
                          "right-hemisphere lesion", correct=True),
                _opt("B", "Predict bilateral hemiplegia from a "
                          "single-hemisphere stroke", correct=False),
                _opt("C", "Predict ipsilateral motor sparing",
                     correct=False),
                _opt("D", "Predict normal motor function", correct=False),
            ],
        )
        ok, reason = self.gate.check(q)
        self.assertFalse(ok)
        self.assertIn("bilateral", reason.lower())

    # ─ Bypass cases ────────────────────────────────────────────────

    def test_no_laterality_term_passes(self):
        q = _q(
            stem=(
                "Patient presents with motor weakness following a "
                "vascular event in the right hemisphere."
            ),
            options=[
                _opt("A", "Predict contralateral hemiplegia", correct=True),
                _opt("B", "Predict ipsilateral weakness", correct=False),
                _opt("C", "Predict normal function", correct=False),
                _opt("D", "Predict motor sparing", correct=False),
            ],
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_both_lateralities_in_stem_passes(self):
        # Stem references both — comparing them, not asserting one
        q = _q(
            stem=(
                "Compare bilateral lesions to unilateral lesions in "
                "predicting motor recovery."
            ),
            options=[
                _opt("A", "Bilateral lesions cause more severe loss",
                     correct=True),
                _opt("B", "Unilateral lesions are always asymptomatic",
                     correct=False),
                _opt("C", "Bilateral and unilateral are equivalent",
                     correct=False),
                _opt("D", "Recovery is identical", correct=False),
            ],
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok, "stem with both lateralities is comparing them, not asserting one")

    def test_clean_distractors_pass(self):
        # Stem says bilateral, all distractors stay bilateral or laterality-neutral
        q = _q(
            stem="Tatum sustained bilateral hippocampal lesions.",
            options=[
                _opt("A", "Predict severe anterograde amnesia from "
                          "bilateral hippocampal disconnection",
                     correct=True),
                _opt("B", "Predict gradual recovery from re-myelination",
                     correct=False),
                _opt("C", "Predict cortical compensation", correct=False),
                _opt("D", "Predict no measurable deficit", correct=False),
            ],
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_correct_option_with_inverted_laterality_passes(self):
        # The gate only inspects DISTRACTORS — correct option is exempt
        # (the gate is for stem-eliminable distractors, not for
        # checking correct's consistency, which other gates handle)
        q = _q(
            stem="Bilateral lesions are documented.",
            options=[
                _opt("A", "Unilateral injury is the textbook case",
                     correct=True),  # correct, even if odd
                _opt("B", "Bilateral injury — option B", correct=False),
                _opt("C", "Bilateral — option C", correct=False),
                _opt("D", "Bilateral — option D", correct=False),
            ],
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok, "gate only inspects distractors, not correct")

    def test_no_options_passes(self):
        q = _q(stem="Bilateral damage observed.", options=[])
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)


class TestUniversalDenial(unittest.TestCase):
    def setUp(self):
        self.gate = UniversalDenialGate()

    # ─ Canonical fail cases (from D7-PHY-076 audit) ────────────────

    def test_lester_pattern_fails(self):
        # The original user-flagged case
        q = _q(
            stem=(
                "After bilateral hippocampal damage Lester Nichols cannot "
                "form new declarative memories but still recalls his "
                "wedding from a decade earlier."
            ),
            options=[
                _opt("A", "Anterograde amnesia with preserved remote "
                          "declarative memory", correct=True),
                _opt("B", "Procedural memory deficits", correct=False),
                _opt("C", "Retrograde amnesia erases all pre-injury "
                          "memories regardless of consolidation",
                     correct=False),
                _opt("D", "Working memory disruption", correct=False),
            ],
        )
        ok, reason = self.gate.check(q)
        self.assertFalse(ok, f"should flag universal denial; reason: {reason}")
        self.assertIn("universal", reason.lower())

    def test_hm_pre_surgical_memories_fails(self):
        # Direct from audit: X-03 distractor A. Stem and distractor both
        # use 'declarative memories' lexically — within the gate's reach.
        q = _q(
            stem=(
                "H.M. underwent bilateral medial temporal lobectomy. "
                "His pre-surgical declarative memories were entirely "
                "spared, but he could not form new declarative memories."
            ),
            options=[
                _opt("A", "Anterograde declarative amnesia with intact "
                          "remote declarative memory", correct=True),
                _opt("B", "Working memory deficits", correct=False),
                _opt("C", "H.M.'s hippocampal resection erased all "
                          "pre-surgical declarative memories from "
                          "remote childhood onward",
                     correct=False),
                _opt("D", "Procedural learning impairment", correct=False),
            ],
        )
        ok, reason = self.gate.check(q)
        self.assertFalse(ok, f"should flag X-03 A pattern; reason: {reason}")
        self.assertIn("declarative", str(reason).lower())

    def test_mara_pattern_bypasses_no_lexical_overlap(self):
        # Documented limitation: when stem cites the preserved instance
        # ('wedding', 'childhood address') WITHOUT using the category
        # word ('memory/memories'), the regex has no shared-word signal
        # to fire on. The Sonnet audit script catches these semantic
        # cases — the heuristic is intentionally narrower.
        q = _q(
            stem=(
                "Mara Cromwell sustained bilateral medial temporal "
                "lobe damage. She vividly recounts her wedding from "
                "fifteen years ago and recites her childhood address."
            ),
            options=[
                _opt("A", "Anterograde amnesia with spared remote "
                          "declarative memory", correct=True),
                _opt("B", "Working memory deficits", correct=False),
                _opt("C", "Hippocampal injury typically erasing all "
                          "remote declarative memories from storage",
                     correct=False),
                _opt("D", "Procedural learning impairment", correct=False),
            ],
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(
            ok,
            "no lexical overlap between stem's named instances and "
            "distractor's category word — regex cannot catch this; "
            "Sonnet audit (Layer D) handles semantic cases",
        )

    # ─ Bypass cases ────────────────────────────────────────────────

    def test_no_preservation_marker_in_stem_passes(self):
        # Stem doesn't say anything is preserved/intact/spared — gate
        # has no signal to fire on
        q = _q(
            stem=(
                "Patient presents to the clinic with anterograde amnesia "
                "following a hypoxic event."
            ),
            options=[
                _opt("A", "Hippocampal damage producing the deficit",
                     correct=True),
                _opt("B", "All cortical regions globally erased",
                     correct=False),  # universal denier but no stem signal
                _opt("C", "Working memory deficits", correct=False),
                _opt("D", "Frontal lobe involvement", correct=False),
            ],
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok, "no preservation marker in stem → bypass")

    def test_no_universal_quantifier_in_distractor_passes(self):
        q = _q(
            stem=(
                "She still recalls her wedding from twenty years ago "
                "despite the recent injury."
            ),
            options=[
                _opt("A", "Spared remote declarative memory", correct=True),
                _opt("B", "Some pre-injury memories may be lost",
                     correct=False),  # no universal quantifier
                _opt("C", "Working memory deficits", correct=False),
                _opt("D", "Procedural memory issues", correct=False),
            ],
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_no_shared_category_words_passes(self):
        # Distractor has universal denial about a totally different category
        # than what the stem preserves — no overlap → bypass
        q = _q(
            stem="The patient still rides a bicycle proficiently.",
            options=[
                _opt("A", "Procedural memory preserved", correct=True),
                _opt("B", "All cardiovascular markers erased",
                     correct=False),  # universal but unrelated
                _opt("C", "Motor function maintained", correct=False),
                _opt("D", "Routine activities continue", correct=False),
            ],
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok, "universal denial in unrelated category → bypass")

    def test_correct_option_universal_quantifier_passes(self):
        # The gate only inspects distractors. If correct uses universal
        # language (e.g., as part of a content-correct claim), gate ignores.
        q = _q(
            stem="She still recalls her wedding twenty years post-injury.",
            options=[
                _opt("A", "All recent declarative memories erased — "
                          "anterograde phenotype",
                     correct=True),  # correct option uses 'all'
                _opt("B", "Working memory affected", correct=False),
                _opt("C", "Procedural deficits", correct=False),
                _opt("D", "Frontal involvement", correct=False),
            ],
        )
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)


class TestRegistration(unittest.TestCase):
    def test_both_gates_registered_in_pipeline(self):
        names = [g.name for g in create_gate_pipeline()]
        self.assertIn("laterality_integrity", names)
        self.assertIn("universal_denial", names)


if __name__ == "__main__":
    unittest.main()
