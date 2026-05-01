"""Tests for Phase 3 prompt-tightening (Plans N + O).

Plan N: T3+ correct-option text MUST begin with an application verb.
        Eliminates the noun-phrase opener anti-pattern.

Plan O: Distractor scope tightening — "MUST mention ALL of the following
        concept names" with explicit count and example, rather than
        "MUST reference these concepts" (which the LLM treats as
        set-of-options).

Both fixes are pure prompt-builder edits — no agent logic changes.
"""
import re
import unittest

import conftest  # noqa: F401  — sets sys.path

from pipeline.prompts import build_user_prompt


def _build(*, tier, correct_answer_form=None, distractor_plan=None):
    """Minimal harness exercising build_user_prompt with both planners
    populated so we can inspect the rendered sections."""
    return build_user_prompt(
        anchor_info={"chapter_title": "Test", "anchor_id_v2": "X", "uid": "X"},
        passage="",
        anchor_data=[{"uid": "X", "verbatim_anchor": "test", "testable_fact": "t"}],
        source_type="anchor_grounded",
        variant_num=1,
        domain_name="Test",
        difficulty_tier=tier,
        correct_answer_form=correct_answer_form,
        distractor_plan=distractor_plan,
    )


class TestPlanN_VerbAtStart(unittest.TestCase):
    """Plan N: verb-at-start mandate fires for T3+ but not T1/T2."""

    def _form(self, verb_pool):
        return {
            "required_verb": verb_pool[0] if verb_pool else "",
            "verb_pool": verb_pool,
            "option_form_constraint": "(form)",
            "option_length_constraint": "(length)",
            "permitted_concept_ids": [],
            "permitted_concept_labels": [],
            "permitted_vocabulary": [],
            "max_concept_count": 1,
        }

    # Regex pattern matching "must begin/start" case-insensitively. Lets
    # the prompt's exact wording polish without breaking the test contract.
    _START_MANDATE_RE = re.compile(r"(?i)must\s+(begin|start)\s+with")

    def test_t3_renders_verb_at_start_mandate(self):
        verb_pool = ["predict", "determine", "apply", "evaluate"]
        prompt = _build(tier=3, correct_answer_form=self._form(verb_pool))
        # Mandate is present (regardless of exact wording)
        self.assertRegex(prompt, self._START_MANDATE_RE,
                         "T3 prompt must contain a verb-at-start mandate")
        # Capitalized verbs from the pool listed (any of them — semantic check)
        verbs_present = sum(1 for v in verb_pool if v.capitalize() in prompt)
        self.assertGreaterEqual(verbs_present, 2,
                                "at least 2 verbs from the pool should appear")
        # Some forbidden-opener language exists (regardless of exact phrasing)
        self.assertRegex(prompt, r"(?i)(bare.?label|noun.?phrase|forbidden)",
                         "prompt should call out the bare-label anti-pattern")
        # Corpus-level fix: mandate must apply to ALL FOUR options
        # uniformly, not just the correct answer. Defeats the verb-
        # rotation frequency skew at scale (47% Predict-led correct).
        self.assertRegex(prompt, r"(?i)(all four|same verb|uniform)",
                         "verb mandate should apply to ALL options uniformly, "
                         "not just the correct option (corpus-level tell fix)")

    def test_t4_renders_verb_at_start_mandate(self):
        verb_pool = ["integrate", "synthesize", "evaluate", "justify"]
        prompt = _build(tier=4, correct_answer_form=self._form(verb_pool))
        self.assertRegex(prompt, self._START_MANDATE_RE)
        # At least one tier-keyed verb appears in capitalized form
        self.assertTrue(
            any(v.capitalize() in prompt for v in verb_pool),
            "T4 prompt should list at least one synthesis verb",
        )

    def test_t1_does_not_render_verb_at_start_mandate(self):
        # T1 stems are recognition-style (definitional), so verb-at-start
        # is unnatural. Mandate must not fire.
        verb_pool = ["identify", "recognize", "name"]
        prompt = _build(tier=1, correct_answer_form=self._form(verb_pool))
        self.assertNotRegex(prompt, self._START_MANDATE_RE)

    def test_t2_does_not_render_verb_at_start_mandate(self):
        verb_pool = ["identify", "describe", "classify"]
        prompt = _build(tier=2, correct_answer_form=self._form(verb_pool))
        self.assertNotRegex(prompt, self._START_MANDATE_RE)

    def test_empty_verb_pool_skips_clause(self):
        # If somehow verb_pool is empty, the clause must not render
        # (degenerate but defensive).
        form = self._form([])
        prompt = _build(tier=3, correct_answer_form=form)
        self.assertNotRegex(prompt, self._START_MANDATE_RE)


class TestT3MechanismMarkerPrompt(unittest.TestCase):
    """T3 (Apply) prompt MUST instruct the LLM that the correct option
    contains a mechanism/causal marker. Prevention layer for the labeling-
    drift pattern that ApplyIdentityGate's mechanism check catches at
    validation. This test ensures the requirement reaches the initial
    prompt (not just retry guidance), reducing retry frequency."""

    def _form(self, verb_pool):
        return {
            "required_verb": verb_pool[0] if verb_pool else "",
            "verb_pool": verb_pool,
            "option_form_constraint": "(form)",
            "option_length_constraint": "(length)",
            "permitted_concept_ids": [],
            "permitted_concept_labels": [],
            "permitted_vocabulary": [],
            "max_concept_count": 1,
        }

    # Regex matching the causal-anchor requirement. Tolerant of
    # wording polish (case-insensitive, broad alternation). Phase 10
    # broadened the contract from "mechanism/causal marker" to
    # "CAUSAL ANCHOR — mechanism marker OR criterion-application
    # marker" so both phrasings are accepted.
    _MECH_MARKER_RE = re.compile(
        r"(?i)(mechanism\s+marker|causal\s+anchor|"
        r"criterion[- ]application\s+marker)"
    )

    def test_t3_renders_mechanism_marker_clause(self):
        verb_pool = ["predict", "determine", "apply", "choose", "select"]
        prompt = _build(tier=3, correct_answer_form=self._form(verb_pool))
        self.assertRegex(
            prompt, self._MECH_MARKER_RE,
            "T3 prompt must instruct the LLM to include a causal "
            "anchor (mechanism marker OR criterion-application marker)",
        )
        # Concrete mechanism examples must appear
        self.assertTrue(
            any(m in prompt for m in ("reflecting", "from", "via", "through")),
            "T3 prompt should give concrete mechanism-marker examples",
        )
        # Phase 10 — concrete criterion-application examples must appear
        self.assertTrue(
            any(m in prompt.lower() for m in (
                "based on", "given that", "criterion", "threshold", "cutoff"
            )),
            "T3 prompt should give criterion-application examples for "
            "criteria-driven content (DSM, ethical thresholds)",
        )
        # Forbidden labeling examples must appear
        self.assertTrue(
            any(p in prompt.lower() for p in ("as part of", "as the", "alongside")),
            "T3 prompt should warn against labeling-as-prediction patterns",
        )

    def test_t1_does_not_render_mechanism_marker_clause(self):
        prompt = _build(
            tier=1,
            correct_answer_form=self._form(["identify", "recognize"]),
        )
        self.assertNotRegex(prompt, self._MECH_MARKER_RE)

    def test_t2_does_not_render_mechanism_marker_clause(self):
        prompt = _build(
            tier=2,
            correct_answer_form=self._form(["identify", "describe"]),
        )
        self.assertNotRegex(prompt, self._MECH_MARKER_RE)

    def test_t4_does_not_render_mechanism_marker_clause(self):
        # T4 has its own identity (Analyze/Evaluate). T3-specific
        # mechanism mandate doesn't apply at synthesis tier.
        prompt = _build(
            tier=4,
            correct_answer_form=self._form(["integrate", "evaluate"]),
        )
        self.assertNotRegex(prompt, self._MECH_MARKER_RE)


class TestPlanO_DistractorScopeTightening(unittest.TestCase):
    """Plan O: distractor section must say 'MUST mention ALL of the
    following concept names' with explicit count and example."""

    def _full_form(self, labels):
        return {
            "required_verb": "predict",
            "verb_pool": ["predict", "determine"],
            "option_form_constraint": "(form)",
            "option_length_constraint": "(length)",
            "permitted_concept_ids": [f"id-{i}" for i in range(len(labels))],
            "permitted_concept_labels": labels,
            "permitted_vocabulary": [],
            "max_concept_count": len(labels),
        }

    def _focused_plan(self):
        return {
            "mode": "focused",
            "slots": [
                {"slot": 1, "distractor_level": 2,
                 "misconception_id": "m1", "misconception_label": "ml1",
                 "misconception_type": "similar_property"},
                {"slot": 2, "distractor_level": 3,
                 "misconception_id": "m2", "misconception_label": "ml2",
                 "misconception_type": "similar_property"},
                {"slot": 3, "distractor_level": 4,
                 "misconception_id": "m3", "misconception_label": "ml3",
                 "misconception_type": "similar_property"},
            ],
        }

    def test_two_concepts_renders_all_with_example(self):
        prompt = _build(
            tier=3,
            correct_answer_form=self._full_form(["Hemiplegia",
                                                  "Cerebrovascular Stroke"]),
            distractor_plan=self._focused_plan(),
        )
        # "ALL N" semantic — captures phrasings like "ALL 2", "all 2",
        # "all of the 2", etc.
        self.assertRegex(prompt, r"(?i)all\s+(of\s+the\s+)?2\b",
                         "T3 with 2 concepts should explicitly count to ALL 2")
        # Both labels appear (semantic — both must be referenced)
        self.assertIn("Hemiplegia", prompt)
        self.assertIn("Cerebrovascular Stroke", prompt)
        # Some "both" or paired emphasis pattern appears
        self.assertRegex(prompt, r"(?i)\bboth\b",
                         "prompt should reference both concepts paired")
        # Asymmetric-scope warning references count or "fewer"
        self.assertRegex(prompt, r"(?i)(<\s*2|less than 2|fewer)")

    def test_single_concept_renders_without_example(self):
        # When there's only 1 permitted concept, the "BOTH" example doesn't
        # apply but the section still renders the distractor scope rule.
        prompt = _build(
            tier=3,
            correct_answer_form=self._full_form(["Hemiplegia"]),
            distractor_plan=self._focused_plan(),
        )
        self.assertRegex(prompt, r"(?i)all\s+(of\s+the\s+)?1\b")
        self.assertIn("Hemiplegia", prompt)

    def test_three_concepts_renders_all_three(self):
        prompt = _build(
            tier=4,
            correct_answer_form=self._full_form(["A", "B", "C"]),
            distractor_plan=self._focused_plan(),
        )
        self.assertRegex(prompt, r"(?i)all\s+(of\s+the\s+)?3\b")
        self.assertRegex(prompt, r"(?i)(<\s*3|less than 3|fewer)")


class TestPhase9_StemFactIntegrity(unittest.TestCase):
    """Phase 9 / Layer A: distractor section must instruct the LLM not to
    invert facts the stem has already stated (laterality, named subject's
    preserved abilities, observed findings). Audit on D7-PHY-076 found
    10/18 questions had this pattern; rule prevents it at the prompt
    layer rather than catching it post-hoc.

    Tier-agnostic: the rule applies to any focused-mode distractor plan
    because rich stems can appear at any tier (though prevalence scales
    T1 0% → T4 100%). Test renders at multiple tiers to confirm.
    """

    def _form(self, labels):
        return {
            "required_verb": "predict",
            "verb_pool": ["predict", "determine"],
            "option_form_constraint": "(form)",
            "option_length_constraint": "(length)",
            "permitted_concept_ids": [f"id-{i}" for i in range(len(labels))],
            "permitted_concept_labels": labels,
            "permitted_vocabulary": [],
            "max_concept_count": len(labels),
        }

    def _focused_plan(self):
        return {
            "mode": "focused",
            "slots": [
                {"slot": s, "distractor_level": s + 1,
                 "misconception_id": f"m{s}", "misconception_label": f"ml{s}",
                 "misconception_type": "similar_property"}
                for s in (1, 2, 3)
            ],
        }

    _STEM_FACT_RULE_RE = re.compile(
        r"(?i)distractors\s+must\s+not\s+contradict\s+facts"
    )
    _MECHANISM_ALT_RE = re.compile(r"(?i)mechanism\s+the\s+question\s+tests")

    def test_t3_renders_stem_fact_integrity_rule(self):
        prompt = _build(
            tier=3,
            correct_answer_form=self._form(["X"]),
            distractor_plan=self._focused_plan(),
        )
        self.assertRegex(prompt, self._STEM_FACT_RULE_RE,
                         "T3 prompt must contain the stem-fact integrity rule")
        # Concrete examples must appear (laterality + preserved-ability)
        self.assertRegex(prompt, r"(?i)bilateral",
                         "rule should cite laterality as an example fact")
        self.assertRegex(prompt, r"(?i)(rides a bicycle|recalls)",
                         "rule should cite a preserved-ability example")
        # The mechanism-alternative framing must appear (why the rule has teeth)
        self.assertRegex(prompt, self._MECHANISM_ALT_RE,
                         "rule should redirect distractors to mechanism-based "
                         "wrongness, not surface-fact inversion")

    def test_t4_renders_stem_fact_integrity_rule(self):
        # T4 had 100% flag rate in the audit — rule must apply here too
        prompt = _build(
            tier=4,
            correct_answer_form=self._form(["A", "B"]),
            distractor_plan=self._focused_plan(),
        )
        self.assertRegex(prompt, self._STEM_FACT_RULE_RE)

    def test_t1_renders_stem_fact_integrity_rule(self):
        # T1 had 0% flag rate but rule is universally true; defensive coverage
        prompt = _build(
            tier=1,
            correct_answer_form=self._form(["X"]),
            distractor_plan=self._focused_plan(),
        )
        self.assertRegex(prompt, self._STEM_FACT_RULE_RE)

    def test_open_mode_does_not_render_rule(self):
        # The clause lives inside the focused-mode block. Open-mode
        # generation has no pre-assigned distractor plan and skips this
        # whole section — verify the rule is absent.
        prompt = _build(
            tier=3,
            correct_answer_form=self._form(["X"]),
            distractor_plan=None,
        )
        self.assertNotRegex(prompt, self._STEM_FACT_RULE_RE)


if __name__ == "__main__":
    unittest.main()
