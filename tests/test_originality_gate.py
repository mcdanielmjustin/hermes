"""Tests for OriginalityGate.

Catches verbatim copying from anchor source material into question text.
Uses 5-gram token overlap with stop-words removed.
"""
import unittest

import conftest  # noqa: F401  — sets sys.path

from pipeline.gates import OriginalityGate


SOURCE_TEXT = (
    "Nondeclarative memory includes procedural skill learning, habit "
    "formation, classical conditioning, and priming. It is preserved "
    "in amnesic patients with hippocampal damage. Procedural learning "
    "depends on the basal ganglia, while classically conditioned motor "
    "responses depend on the cerebellum."
)


def _q(*, stem, correct_text="A clean answer.", correct_explanation="",
       distractor_texts=None, source=None):
    distractor_texts = distractor_texts or ["d1", "d2", "d3"]
    options = [
        {"letter": "A", "text": correct_text,
         "explanation": correct_explanation, "is_correct": True}
    ]
    for L, t in zip(("B", "C", "D"), distractor_texts):
        options.append({"letter": L, "text": t, "explanation": "",
                        "is_correct": False})
    return {
        "question_stem": stem,
        "options": options,
        "anchor_content_summaries": [source] if source else [SOURCE_TEXT],
    }


class TestOriginalityViolations(unittest.TestCase):
    def setUp(self):
        self.gate = OriginalityGate(hard_threshold=0.5,
                                    soft_threshold=0.3,
                                    ngram_size=5)

    def test_verbatim_copy_fails(self):
        # Stem copies large chunk of the source text
        q = _q(stem="Nondeclarative memory includes procedural skill learning "
                    "habit formation classical conditioning and priming "
                    "preserved in amnesic patients with hippocampal damage "
                    "procedural learning depends on the basal ganglia while "
                    "classically conditioned motor responses depend on the "
                    "cerebellum")
        ok, reason = self.gate.check(q)
        self.assertFalse(ok)
        self.assertIn("verbatim", reason.lower())

    def test_paraphrase_passes(self):
        # Paraphrased — same concepts, different wording
        q = _q(stem="A patient with hippocampal injury can still learn new "
                    "motor skills. Which structure mediates this preserved "
                    "ability?",
               correct_text="The basal ganglia mediates skill acquisition.")
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)


class TestOriginalityEdgeCases(unittest.TestCase):
    def setUp(self):
        self.gate = OriginalityGate(hard_threshold=0.5,
                                    soft_threshold=0.3)

    def test_empty_source_passes(self):
        q = _q(stem="A novel question.", source="")
        q["anchor_content_summaries"] = []
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_empty_question_passes(self):
        q = {
            "question_stem": "",
            "options": [],
            "anchor_content_summaries": [SOURCE_TEXT],
        }
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)

    def test_short_text_below_ngram_size_passes(self):
        # Source shorter than 5 tokens — no ngrams to compare
        q = _q(stem="Original wording.",
               source="Two short.")
        ok, _ = self.gate.check(q)
        self.assertTrue(ok)


class TestThresholdConfigurability(unittest.TestCase):
    def test_strict_threshold_catches_more(self):
        # Strict: any non-trivial overlap fails. Loose: only near-verbatim.
        strict = OriginalityGate(hard_threshold=0.05, soft_threshold=0.01)
        loose = OriginalityGate(hard_threshold=0.95, soft_threshold=0.9)

        # Stem with some shared 5-grams but not large overlap
        q = _q(stem="Procedural skill learning habit formation classical "
                    "conditioning is preserved in patients.")
        strict_ok, _ = strict.check(q)
        loose_ok, _ = loose.check(q)
        self.assertFalse(strict_ok,
                         "strict threshold should hard-fail on partial overlap")
        self.assertTrue(loose_ok,
                        "loose threshold should pass on partial overlap")


if __name__ == "__main__":
    unittest.main()
