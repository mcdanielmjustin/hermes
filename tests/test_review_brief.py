"""Tests for the brief-generation second-pass review.

The review function is a critique-and-revise step that catches LLM
hallucinations the structural validator misses. These tests use a mock
Anthropic client so we don't burn API credit verifying control flow.
"""
import json
import sys
import pathlib
import unittest
from unittest.mock import MagicMock

import conftest  # noqa: F401  — sets sys.path

# generate_anchor_briefs lives in scripts/ — make it importable
SCRIPTS_DIR = pathlib.Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import generate_anchor_briefs as gab


# ── Fixtures ──────────────────────────────────────────────────

GOOD_BRIEF = {
    "core_claims": [
        "Implicit memory is preserved in amnesic patients",
        "Procedural learning depends on the basal ganglia",
        "Cerebellum mediates conditioned motor responses",
    ],
    "concepts": [
        {"concept_id": "implicit-memory", "label": "Implicit Memory",
         "description": "Nondeclarative memory."},
        {"concept_id": "basal-ganglia", "label": "Basal Ganglia",
         "description": "Subcortical structures."},
    ],
    "misconceptions": [
        {"misconception_id": "implicit-vs-explicit",
         "label": "Confusing implicit and explicit memory",
         "type": "opposite_direction",
         "concepts_involved": ["implicit-memory"]},
        {"misconception_id": "ganglia-vs-cortex",
         "label": "Believing cortex handles habits",
         "type": "similar_property",
         "concepts_involved": ["basal-ganglia", "implicit-memory"]},
        {"misconception_id": "memory-as-unitary",
         "label": "Treating memory as one system",
         "type": "overgeneralization",
         "concepts_involved": ["implicit-memory", "basal-ganglia"]},
    ],
    "question_angles": [
        {"type": "definitional", "description": "..."},
        {"type": "clinical_application", "description": "..."},
        {"type": "neuroanatomical", "description": "..."},
    ],
}


def _mock_response(payload):
    """Build a fake Anthropic response object holding the given JSON payload."""
    response = MagicMock()
    response.content = [MagicMock(text=json.dumps(payload))]
    response.usage = MagicMock(input_tokens=500, output_tokens=400)
    return response


# ── briefs_meaningfully_differ ─────────────────────────────────

class TestBriefsMeaningfullyDiffer(unittest.TestCase):
    def test_identical_briefs_no_diff(self):
        self.assertFalse(gab.briefs_meaningfully_differ(GOOD_BRIEF, GOOD_BRIEF))

    def test_concept_change_detected(self):
        revised = json.loads(json.dumps(GOOD_BRIEF))  # deep copy
        revised["concepts"][0]["description"] = "Different description"
        self.assertTrue(gab.briefs_meaningfully_differ(GOOD_BRIEF, revised))

    def test_misconception_added_detected(self):
        revised = json.loads(json.dumps(GOOD_BRIEF))
        revised["misconceptions"].append({
            "misconception_id": "new-confusion",
            "label": "New",
            "type": "similar_name",
            "concepts_involved": ["implicit-memory"],
        })
        self.assertTrue(gab.briefs_meaningfully_differ(GOOD_BRIEF, revised))

    def test_irrelevant_top_level_keys_ignored(self):
        # Adding metadata keys shouldn't trigger a meaningful-difference flag.
        revised = json.loads(json.dumps(GOOD_BRIEF))
        revised["uid"] = "test"
        revised["irrelevant_field"] = "anything"
        self.assertFalse(gab.briefs_meaningfully_differ(GOOD_BRIEF, revised))


# ── review_brief() ─────────────────────────────────────────────

class TestReviewBrief(unittest.TestCase):
    def test_unchanged_review_returns_same_brief(self):
        client = MagicMock()
        # Reviewer returns the brief as-is (no changes warranted)
        client.messages.create.return_value = _mock_response(GOOD_BRIEF)

        reviewed, tokens = gab.review_brief(
            client, GOOD_BRIEF,
            verbatim="anchor verbatim",
            testable="testable fact",
            passage_text="passage",
        )
        self.assertGreater(tokens, 0)
        self.assertFalse(gab.briefs_meaningfully_differ(GOOD_BRIEF, reviewed))

    def test_review_returns_revised_brief(self):
        client = MagicMock()
        revised_brief = json.loads(json.dumps(GOOD_BRIEF))
        revised_brief["core_claims"][0] = "Revised claim with tighter scope"
        client.messages.create.return_value = _mock_response(revised_brief)

        reviewed, _ = gab.review_brief(
            client, GOOD_BRIEF,
            verbatim="...", testable="...", passage_text="...",
        )
        self.assertTrue(gab.briefs_meaningfully_differ(GOOD_BRIEF, reviewed))
        self.assertEqual(
            reviewed["core_claims"][0],
            "Revised claim with tighter scope",
        )

    def test_review_strips_markdown_fences(self):
        # Reviewer occasionally wraps JSON in ```json ... ``` even though
        # told not to. The function must tolerate this.
        client = MagicMock()
        wrapped_text = "```json\n" + json.dumps(GOOD_BRIEF) + "\n```"
        response = MagicMock()
        response.content = [MagicMock(text=wrapped_text)]
        response.usage = MagicMock(input_tokens=100, output_tokens=100)
        client.messages.create.return_value = response

        reviewed, _ = gab.review_brief(
            client, GOOD_BRIEF,
            verbatim="...", testable="...", passage_text="...",
        )
        self.assertEqual(reviewed["core_claims"], GOOD_BRIEF["core_claims"])

    def test_review_json_failure_falls_back_to_draft(self):
        client = MagicMock()
        # All retry attempts produce unparseable text
        bad_response = MagicMock()
        bad_response.content = [MagicMock(text="not valid json")]
        bad_response.usage = MagicMock(input_tokens=100, output_tokens=100)
        client.messages.create.return_value = bad_response

        reviewed, tokens = gab.review_brief(
            client, GOOD_BRIEF,
            verbatim="...", testable="...", passage_text="...",
            max_retries=1,
        )
        # Falls back to the original draft so the anchor isn't dropped
        self.assertIs(reviewed, GOOD_BRIEF)
        self.assertEqual(tokens, 0)


# ── build_review_prompt ────────────────────────────────────────

class TestBuildReviewPrompt(unittest.TestCase):
    def test_includes_source_material(self):
        prompt = gab.build_review_prompt(
            GOOD_BRIEF,
            verbatim="An agonist binds to receptors.",
            testable="Agonists activate receptors.",
            passage_text="Textbook discussion of receptor pharmacology.",
        )
        self.assertIn("agonist binds to receptors", prompt.lower())
        self.assertIn("activate receptors", prompt.lower())
        self.assertIn("textbook discussion", prompt.lower())

    def test_includes_brief_for_review(self):
        prompt = gab.build_review_prompt(
            GOOD_BRIEF,
            verbatim="...", testable="...", passage_text="...",
        )
        self.assertIn("implicit-memory", prompt)
        self.assertIn("basal-ganglia", prompt)

    def test_handles_missing_passage(self):
        prompt = gab.build_review_prompt(
            GOOD_BRIEF,
            verbatim="...", testable="...", passage_text="",
        )
        self.assertIn("No textbook passage", prompt)


if __name__ == "__main__":
    unittest.main()
