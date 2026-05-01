"""Tests for build_correction_prompt — multi-issue correction.

Earlier the function took (gate_name, reason) and only addressed one
failure per retry. With max_attempts=2, a question that failed BOTH
attribution AND length-balance gates would be dropped (single retry
can only fix one issue, then the other fails on the 2nd attempt).

The new signature accepts a list of (gate_name, reason) tuples and
bundles all guidance into one correction prompt — the LLM addresses
every failure in one retry.
"""
import unittest

import conftest  # noqa: F401  — sets sys.path

from pipeline.prompts import build_correction_prompt


class TestSingleFailure(unittest.TestCase):
    def test_returns_original_when_no_failures(self):
        result = build_correction_prompt("Original prompt.", [])
        self.assertEqual(result, "Original prompt.")

    def test_appends_correction_section(self):
        result = build_correction_prompt(
            "Original prompt.",
            [("attribution", "Squire (2004) detected in stem")],
        )
        self.assertIn("Original prompt.", result)
        self.assertIn("CORRECTION REQUIRED", result)
        self.assertIn("attribution", result)
        self.assertIn("Squire (2004)", result)

    def test_includes_static_guidance_for_known_gate(self):
        result = build_correction_prompt(
            "Original prompt.",
            [("attribution", "x")],
        )
        # Static guidance from _GATE_GUIDANCE for "attribution" — verify
        # it mentions the researcher topic somewhere (regardless of exact
        # phrasing, which polishes over time).
        self.assertRegex(result, r"(?i)researcher",
                         "correction for attribution should mention researchers")


class TestMultipleFailures(unittest.TestCase):
    def test_two_failures_both_addressed(self):
        result = build_correction_prompt(
            "Original prompt.",
            [
                ("attribution", "Squire (2004) detected"),
                ("option_length_balance", "ratio 1.95 exceeds 1.7"),
            ],
        )
        # Both gate names appear in the correction
        self.assertIn("attribution", result)
        self.assertIn("option_length_balance", result)
        # Both specific reasons appear
        self.assertIn("Squire (2004)", result)
        self.assertIn("1.95", result)
        # Both gates' guidance is rendered semantically (regardless of
        # exact phrasing — these used to assert exact substrings that
        # broke on minor wording polish).
        self.assertRegex(result, r"(?i)researcher",
                         "attribution guidance should mention researchers")
        self.assertRegex(result, r"(?i)(length|character)",
                         "length-balance guidance should reference length/chars")

    def test_three_failures_all_addressed(self):
        result = build_correction_prompt(
            "Original prompt.",
            [
                ("attribution", "Squire (2004) detected"),
                ("option_length_balance", "ratio 2.0"),
                ("anchor_grounding", "tested_concept not in brief"),
            ],
        )
        for gate in ("attribution", "option_length_balance", "anchor_grounding"):
            self.assertIn(gate, result)

    def test_plural_count_in_header(self):
        result = build_correction_prompt(
            "Original prompt.",
            [("attribution", "x"), ("option_length_balance", "y")],
        )
        self.assertIn("failed 2 validation gates", result)
        self.assertIn("CORRECTIONS REQUIRED", result)  # plural

    def test_singular_count_in_header(self):
        result = build_correction_prompt(
            "Original prompt.",
            [("attribution", "x")],
        )
        self.assertIn("failed 1 validation gate", result)
        self.assertIn("CORRECTION REQUIRED", result)  # singular


class TestUnknownGate(unittest.TestCase):
    def test_falls_back_to_generic_guidance(self):
        # An unknown gate name should still produce a correction with the
        # raw failure reason (no _GATE_GUIDANCE entry).
        result = build_correction_prompt(
            "Original.",
            [("unknown_gate", "something specific went wrong")],
        )
        self.assertIn("unknown_gate", result)
        self.assertIn("something specific went wrong", result)


if __name__ == "__main__":
    unittest.main()
