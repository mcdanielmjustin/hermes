"""Tests for the option-claim auto-strip helpers in pipeline.agents.

Calibration showed the LLM produces "X because Y" in option text on ~25%
of T3/T4 generations even with explicit prompt rules forbidding it
(5/9 calibration failures). The deterministic auto-strip in the
assembler moves the offending clause from option.text into
option.explanation BEFORE validation, so the violation never reaches
the OptionClaimGate.

This test suite locks the strip and merge behavior so it can't drift
away from OptionClaimGate's marker set.
"""
import unittest

import conftest  # noqa: F401  — sets sys.path

from pipeline.agents import (
    split_off_reasoning_clause,
    merge_reasoning_into_explanation,
)


class TestSplitOffReasoningClause(unittest.TestCase):
    def test_strips_because(self):
        text = "Compound X is an antagonist because it lacks intrinsic activity"
        head, clause = split_off_reasoning_clause(text)
        self.assertEqual(head, "Compound X is an antagonist")
        self.assertEqual(clause, "because it lacks intrinsic activity")

    def test_strips_since(self):
        text = "Receptor activity decreases sharply since the antagonist is blocking"
        head, clause = split_off_reasoning_clause(text)
        self.assertEqual(head, "Receptor activity decreases sharply")
        self.assertEqual(clause, "since the antagonist is blocking")

    def test_strips_due_to(self):
        text = "Postsynaptic effect intensifies due to direct receptor stimulation"
        head, clause = split_off_reasoning_clause(text)
        self.assertEqual(head, "Postsynaptic effect intensifies")
        self.assertEqual(clause, "due to direct receptor stimulation")

    def test_strips_owing_to(self):
        text = "Neurotransmitter synthesis halts owing to enzymatic depletion"
        head, clause = split_off_reasoning_clause(text)
        self.assertEqual(head, "Neurotransmitter synthesis halts")
        self.assertEqual(clause, "owing to enzymatic depletion")

    def test_strips_in_order_to(self):
        text = "The neuron generates an action potential in order to propagate the signal"
        head, clause = split_off_reasoning_clause(text)
        self.assertEqual(head, "The neuron generates an action potential")
        self.assertEqual(clause, "in order to propagate the signal")

    def test_strips_punctuation_separator(self):
        text = "Compound X is an antagonist; because it lacks intrinsic activity"
        head, clause = split_off_reasoning_clause(text)
        self.assertEqual(head, "Compound X is an antagonist")
        self.assertEqual(clause, "because it lacks intrinsic activity")

    def test_strips_at_first_marker_only(self):
        # Only the first marker is the split point. Subsequent markers go
        # into the clause as-is.
        text = "Compound X is acting as an antagonist because Y, since Z"
        head, clause = split_off_reasoning_clause(text)
        self.assertEqual(head, "Compound X is acting as an antagonist")
        self.assertIn("because Y", clause)
        self.assertIn("since Z", clause)

    def test_no_marker_returns_unchanged(self):
        text = "Compound X is an antagonist"
        head, clause = split_off_reasoning_clause(text)
        self.assertEqual(head, text)
        self.assertEqual(clause, "")

    def test_empty_text_returns_unchanged(self):
        head, clause = split_off_reasoning_clause("")
        self.assertEqual(head, "")
        self.assertEqual(clause, "")

    def test_short_head_skips_strip(self):
        # If stripping leaves < 20 chars of claim, the whole option is
        # mostly reasoning — we can't auto-clean it cleanly. Return
        # unchanged so the OptionClaimGate fires and a retry kicks in.
        text = "Brief claim because of complete enzymatic blockade at the receptor"
        head, clause = split_off_reasoning_clause(text)
        # head would be "Brief claim" (11 chars) — below absolute min 20
        self.assertEqual(head, text)
        self.assertEqual(clause, "")

    def test_yes_because_skipped(self):
        # Calibration regression: "Yes, because Y" produced 5-char stubs.
        # Bumped min head from 4 to 20 chars. This case must skip strip.
        text = "Yes, because the receptor is now fully saturated with the agonist"
        head, clause = split_off_reasoning_clause(text)
        self.assertEqual(head, text)
        self.assertEqual(clause, "")

    def test_parity_guard_skips_when_imbalanced(self):
        # Calibration: M-05 ended up A:77 B:67 C:10 D:72 because C got
        # stripped down to a stub. The parity guard skips strip when the
        # head would be < 55% of the longest sibling's original length.
        text = "Brief stub-ish claim because of detailed reasoning that follows"
        head, clause = split_off_reasoning_clause(text, max_sibling_len=80)
        # head would be "Brief stub-ish claim" (20 chars) —
        # 20 < 0.55 * 80 = 44, so strip is skipped
        self.assertEqual(head, text)
        self.assertEqual(clause, "")

    def test_parity_guard_allows_balanced_strip(self):
        # When the head is at least 55% of max sibling, strip proceeds.
        text = ("This is a fully developed claim about the compound, "
                "because intrinsic activity is the deciding factor")
        head, clause = split_off_reasoning_clause(text, max_sibling_len=80)
        # head ≈ 51 chars; 51 >= 0.55 * 80 = 44, strip proceeds
        self.assertNotEqual(head, text)
        self.assertNotIn("because", head.lower())
        self.assertIn("because", clause.lower())

    def test_max_sibling_zero_disables_relative_guard(self):
        # max_sibling_len=0 keeps backward compatibility — only absolute
        # min applies.
        text = "Sufficiently long claim because of this reason"
        head, clause = split_off_reasoning_clause(text, max_sibling_len=0)
        self.assertNotEqual(head, text)
        self.assertIn("because", clause.lower())

    def test_word_boundary_does_not_match_inside_word(self):
        # "ever since" contains "since" but as a separate word — DO match
        # (word-boundary is satisfied). However, "absince" or similar
        # substrings should NOT match.
        text = "An absincere claim"  # contrived; "since" not a separate word
        head, clause = split_off_reasoning_clause(text)
        self.assertEqual(head, text)
        self.assertEqual(clause, "")

    def test_case_insensitive(self):
        text = "Postsynaptic effect decreases BECAUSE the receptor is blocked"
        head, clause = split_off_reasoning_clause(text)
        self.assertEqual(head, "Postsynaptic effect decreases")
        self.assertIn("BECAUSE", clause)


class TestMergeReasoningIntoExplanation(unittest.TestCase):
    def test_appends_to_existing_explanation(self):
        out = merge_reasoning_into_explanation(
            "An antagonist binds without activating.",
            "because it lacks intrinsic activity",
        )
        self.assertIn("An antagonist binds without activating.", out)
        self.assertIn("Because it lacks intrinsic activity", out)

    def test_capitalizes_first_letter_when_standalone(self):
        out = merge_reasoning_into_explanation(
            "", "because it lacks intrinsic activity"
        )
        self.assertEqual(out, "Because it lacks intrinsic activity")

    def test_skips_duplicate_clause(self):
        # If the explanation already mentions the clause, don't double-add.
        existing = "Antagonists block because they lack intrinsic activity."
        out = merge_reasoning_into_explanation(
            existing, "because they lack intrinsic activity"
        )
        self.assertEqual(out, existing)

    def test_empty_clause_returns_explanation_unchanged(self):
        out = merge_reasoning_into_explanation("Existing.", "")
        self.assertEqual(out, "Existing.")

    def test_separator_handles_existing_period(self):
        out = merge_reasoning_into_explanation(
            "Existing fact.", "because of Y"
        )
        # Single space after period (existing already ends with period)
        self.assertIn("Existing fact. Because", out)

    def test_separator_adds_period_when_missing(self):
        out = merge_reasoning_into_explanation(
            "Existing fact", "because of Y"
        )
        self.assertIn("Existing fact. Because", out)


class TestEndToEndOptionTextStrip(unittest.TestCase):
    """End-to-end: simulate the assembler's auto-strip pass on options."""

    def _strip_option(self, option):
        head, clause = split_off_reasoning_clause(option["text"])
        if clause:
            option["text"] = head
            option["explanation"] = merge_reasoning_into_explanation(
                option.get("explanation", ""), clause
            )
        return option

    def test_realistic_calibration_failure_pattern(self):
        # Direct example from D7-PHY-195 calibration regen:
        # "option C (correct) has reasoning marker 'because': Reject the
        #  trainee's proposal and classify both compounds as antagonists,
        #  because both lack intrinsic activity."
        opt = {
            "letter": "C",
            "is_correct": True,
            "text": ("Reject the trainee's proposal and classify both compounds "
                     "as antagonists, because both lack intrinsic activity"),
            "explanation": "Antagonist activity is defined by binding without producing a response.",
        }
        cleaned = self._strip_option(opt)
        self.assertNotIn("because", cleaned["text"].lower())
        # Application verb ("classify") survives in option text
        self.assertIn("classify", cleaned["text"])
        self.assertIn("because both lack intrinsic activity",
                      cleaned["explanation"].lower())

    def test_distractor_pattern_also_stripped(self):
        # Calibration: "option D (distractor) has reasoning marker 'because':
        # Only Compound X is an antagonist; Compound Y fails the definition
        # because antagonists must..."
        opt = {
            "letter": "D",
            "is_correct": False,
            "text": ("Only Compound X is an antagonist; Compound Y fails the "
                     "definition because antagonists must produce a measurable response"),
            "explanation": "",
        }
        cleaned = self._strip_option(opt)
        self.assertNotIn("because", cleaned["text"].lower())
        self.assertIn("because antagonists must produce", cleaned["explanation"].lower())


if __name__ == "__main__":
    unittest.main()
