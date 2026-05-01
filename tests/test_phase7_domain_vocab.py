"""Tests for Phase 7 — Curated domain vocabulary pool.

Phase 7 adds a per-domain curated/bootstrapped vocabulary pool that
broadens the L3 scaffold's term list at T1/T2. Three coupled changes:

1. CorrectAnswerFormPlannerAgent appends domain_vocab at T1/T2 only
   (priority 3, after concept descriptions and brief-internal pool).
2. Cap raises from 8 → 12 at T1/T2; T3/T4 stays at 8.
3. Prompt enforcement: at T1/T2, distractor vocab rule changes from
   "at least ONE" to "at least TWO distinct" terms per distractor.

Plus orchestrator file-loader for the curated JSON.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

import conftest  # noqa: F401  — sets sys.path

from pipeline.orchestrator import QuestionOrchestrator
from pipeline.prompts import build_user_prompt


def _form(*, tier, permitted_vocab):
    """Minimal correct_answer_form payload exercising the vocab section."""
    return {
        "required_verb": "predict" if tier >= 3 else "identify",
        "verb_pool": ["predict", "determine"] if tier >= 3 else
                     ["identify", "recognize", "name"],
        "option_form_constraint": "(form)",
        "option_length_constraint": "(length)",
        "permitted_concept_ids": ["c1"],
        "permitted_concept_labels": ["Concept One"],
        "permitted_vocabulary": permitted_vocab,
        "max_concept_count": 1,
    }


def _build(*, tier, permitted_vocab, distractor_plan=None):
    return build_user_prompt(
        anchor_info={"chapter_title": "Test", "anchor_id_v2": "X", "uid": "X"},
        passage="",
        anchor_data=[{"uid": "X", "verbatim_anchor": "test", "testable_fact": "t"}],
        source_type="anchor_grounded",
        variant_num=1,
        domain_name="Test",
        difficulty_tier=tier,
        correct_answer_form=_form(tier=tier, permitted_vocab=permitted_vocab),
        distractor_plan=distractor_plan,
    )


class TestDistractorVocabRuleByTier(unittest.TestCase):
    """Phase 7 prompt enforcement: at T1/T2 distractor must contain
    "at least TWO distinct" pool terms; at T3/T4 keep "at least ONE"."""

    _VOCAB = ["alphavocabxxxxx", "betavocabxxxxx",
              "gammavocabxxxxx", "deltavocabxxxxx"]

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

    def test_t1_renders_two_distinct_rule(self):
        prompt = _build(tier=1, permitted_vocab=self._VOCAB,
                        distractor_plan=self._focused_plan())
        self.assertIn("TWO distinct", prompt,
                      "T1 prompt should require at least TWO distinct terms")

    def test_t2_renders_two_distinct_rule(self):
        prompt = _build(tier=2, permitted_vocab=self._VOCAB,
                        distractor_plan=self._focused_plan())
        self.assertIn("TWO distinct", prompt,
                      "T2 prompt should require at least TWO distinct terms")

    def test_t3_keeps_one_rule(self):
        prompt = _build(tier=3, permitted_vocab=self._VOCAB,
                        distractor_plan=self._focused_plan())
        # T3 should NOT bump to TWO; the strict cluster path handles
        # diversification at apply tier.
        self.assertNotIn("TWO distinct", prompt,
                         "T3 must not bump to TWO — keeps ONE rule")
        self.assertIn("at least ONE", prompt,
                      "T3 must keep 'at least ONE' rule")

    def test_t4_keeps_one_rule(self):
        prompt = _build(tier=4, permitted_vocab=self._VOCAB,
                        distractor_plan=self._focused_plan())
        self.assertNotIn("TWO distinct", prompt,
                         "T4 must not bump to TWO — keeps ONE rule")
        self.assertIn("at least ONE", prompt,
                      "T4 must keep 'at least ONE' rule")


class TestOrchestratorDomainVocabLoader(unittest.TestCase):
    """Phase 7: orchestrator caches per-domain vocabulary from the JSON
    file. Missing file → empty list (degrades cleanly)."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.vocab_dir = Path(self.tmpdir)

    def tearDown(self):
        # Cleanup
        for f in self.vocab_dir.glob("*.json"):
            f.unlink()
        os.rmdir(self.tmpdir)

    def _write_vocab(self, code, vocab):
        path = self.vocab_dir / f"{code}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump({
                "domain_code": code,
                "domain_name": code,
                "curated": False,
                "version": 1,
                "vocabulary": vocab,
            }, f)

    def test_loads_vocabulary_list(self):
        self._write_vocab("BPSY", ["aaa", "bbb", "ccc"])
        orch = QuestionOrchestrator()
        out = orch._load_domain_vocab(str(self.vocab_dir), "BPSY")
        self.assertEqual(out, ["aaa", "bbb", "ccc"])

    def test_caches_per_domain(self):
        self._write_vocab("BPSY", ["aaa"])
        orch = QuestionOrchestrator()
        out1 = orch._load_domain_vocab(str(self.vocab_dir), "BPSY")
        # Mutate the file — cache should serve original
        self._write_vocab("BPSY", ["zzz"])
        out2 = orch._load_domain_vocab(str(self.vocab_dir), "BPSY")
        self.assertEqual(out1, out2,
                         "second call should use cached value, not re-read")

    def test_missing_file_returns_empty(self):
        orch = QuestionOrchestrator()
        out = orch._load_domain_vocab(str(self.vocab_dir), "MISSING")
        self.assertEqual(out, [],
                         "missing file must return empty list, not crash")

    def test_none_dir_returns_empty(self):
        orch = QuestionOrchestrator()
        out = orch._load_domain_vocab(None, "BPSY")
        self.assertEqual(out, [],
                         "None dir must return empty list (not configured)")

    def test_malformed_json_returns_empty(self):
        path = self.vocab_dir / "BAD.json"
        with open(path, "w", encoding="utf-8") as f:
            f.write("{not valid json")
        orch = QuestionOrchestrator()
        out = orch._load_domain_vocab(str(self.vocab_dir), "BAD")
        self.assertEqual(out, [],
                         "malformed JSON must degrade to empty list")


if __name__ == "__main__":
    unittest.main()
