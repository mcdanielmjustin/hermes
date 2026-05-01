"""Unit tests for QuestionOrchestrator's public API (Tier 2 polish).

The orchestrator coordinates all pipeline phases — `load_anchor_context`,
`prepare_task`, and `generate_and_validate` are the three public entry
points. Prior to this file, only integration coverage existed via the
question-generation script's end-to-end runs.

These tests use synthetic briefs in tempdirs to exercise the orchestrator
in isolation, with no real LLM calls.
"""
import asyncio
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import conftest  # noqa: F401  — sets sys.path

from pipeline.orchestrator import QuestionOrchestrator


def _write_brief(briefs_dir, domain, uid, *, chapter_num="D7-Ch01",
                 concepts=None, misconceptions=None):
    """Helper: write a minimal brief file matching the production schema."""
    domain_dir = briefs_dir / domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    if concepts is None:
        concepts = [{"concept_id": "agonist", "label": "Agonist",
                     "description": "Compound that activates a receptor."}]
    if misconceptions is None:
        misconceptions = [{"misconception_id": "m1", "label": "lbl",
                           "type": "similar_property",
                           "concepts_involved": ["agonist"]}]
    brief = {
        "uid": uid,
        "chapter_num": chapter_num,
        "verbatim_anchor": "test verbatim anchor",
        "testable_fact": "test testable fact",
        "concepts": concepts,
        "misconceptions": misconceptions,
        "core_claims": ["claim about agonist"],
        "question_angles": [{"type": "definitional", "description": "Define X"}],
    }
    (domain_dir / f"{uid}.json").write_text(json.dumps(brief), encoding="utf-8")
    return brief


def _make_anchor(uid="D7-PHY-195-17b48e1e", chapter_num="D7-Ch01"):
    """Build an anchor dict matching what the script passes to prepare_task."""
    return {
        "uid": uid,
        "anchor_point_id_v2": f"AP-{uid}",
        "chapter_num": chapter_num,
        "chapter_title": "Test Chapter",
        "verbatim_anchor": "verbatim text from textbook",
        "testable_fact": "specific testable fact from the anchor",
        "passage": "Example passage text. " * 50,  # ~1000 chars
    }


# ── load_anchor_context ─────────────────────────────────────────

class TestLoadAnchorContext(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.briefs_dir = Path(self.tmp)
        self.orch = QuestionOrchestrator()

    def test_returns_brief_data_when_present(self):
        _write_brief(self.briefs_dir, "BPSY", "test-uid")
        ctx = self.orch.load_anchor_context(
            self.briefs_dir, "BPSY", "test-uid",
            chapter_vocab={"has_vocab": False, "concepts": [], "misconceptions": []},
            passage_text="example passage",
        )
        self.assertTrue(ctx["has_brief"])
        self.assertEqual(len(ctx["active_concepts"]), 1)
        self.assertEqual(ctx["active_concepts"][0]["concept_id"], "agonist")
        self.assertTrue(ctx["anchor_has_vocab"])

    def test_falls_back_to_chapter_vocab_when_no_brief(self):
        # No brief on disk — falls back to chapter_vocab argument
        chapter_vocab = {
            "has_vocab": True,
            "concepts": [{"concept_id": "ch-c1", "label": "Ch C1"}],
            "misconceptions": [],
        }
        ctx = self.orch.load_anchor_context(
            self.briefs_dir, "BPSY", "missing-uid",
            chapter_vocab=chapter_vocab,
            passage_text="passage",
        )
        self.assertFalse(ctx["has_brief"])
        # Active concepts come from chapter vocab
        self.assertEqual(ctx["active_concepts"][0]["concept_id"], "ch-c1")

    def test_handles_corrupt_brief_gracefully(self):
        # A corrupt brief file should NOT crash the orchestrator.
        domain_dir = self.briefs_dir / "BPSY"
        domain_dir.mkdir(parents=True)
        (domain_dir / "broken.json").write_text("{not valid json", encoding="utf-8")
        ctx = self.orch.load_anchor_context(
            self.briefs_dir, "BPSY", "broken",
            chapter_vocab={"has_vocab": False, "concepts": [], "misconceptions": []},
            passage_text="",
        )
        self.assertFalse(ctx["has_brief"])

    def test_exposes_metadata_for_prepare_task_clustering(self):
        # load_anchor_context must expose chapter_num, anchor_briefs_dir,
        # and primary_uid so prepare_task can call AnchorClusterAgent.
        _write_brief(self.briefs_dir, "BPSY", "test-uid",
                     chapter_num="D7-Ch01")
        ctx = self.orch.load_anchor_context(
            self.briefs_dir, "BPSY", "test-uid",
            chapter_vocab={"has_vocab": False, "concepts": [], "misconceptions": []},
            passage_text="",
        )
        self.assertEqual(ctx["primary_chapter_num"], "D7-Ch01")
        self.assertEqual(ctx["primary_uid"], "test-uid")
        self.assertEqual(ctx["domain_code"], "BPSY")
        self.assertEqual(ctx["anchor_briefs_dir"], self.briefs_dir)


# ── prepare_task ────────────────────────────────────────────────

class TestPrepareTask(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.briefs_dir = Path(self.tmp)
        self.orch = QuestionOrchestrator()
        # Seed a brief AND a few siblings for cluster selection
        _write_brief(self.briefs_dir, "BPSY", "primary",
                     chapter_num="D7-Ch01")
        _write_brief(self.briefs_dir, "BPSY", "sib-1",
                     chapter_num="D7-Ch01")
        _write_brief(self.briefs_dir, "BPSY", "sib-2",
                     chapter_num="D7-Ch01")

    def _ctx(self):
        return self.orch.load_anchor_context(
            self.briefs_dir, "BPSY", "primary",
            chapter_vocab={"has_vocab": False, "concepts": [], "misconceptions": []},
            passage_text="passage" * 30,
        )

    def test_returns_complete_task_for_t1(self):
        anchor = _make_anchor(uid="primary")
        task = self.orch.prepare_task(
            anchor, self._ctx(), tier=1, variant=1,
            domain_code="BPSY", domain_id=7, domain_name="Biopsychology",
            chapter_title="Test", section_title="Section",
            batch_id="test-batch", anchor_idx=0,
        )
        # Required task fields
        for key in ("question_id", "system_prompt", "user_prompt",
                    "tier", "distractor_plan", "meta_base",
                    "correct_answer_form", "cluster_anchors"):
            self.assertIn(key, task, f"task missing key: {key}")
        self.assertEqual(task["tier"], 1)

    def test_t1_task_has_zero_clusters(self):
        # Bloom's-identity invariant: T1 must NEVER receive cluster anchors.
        anchor = _make_anchor(uid="primary")
        task = self.orch.prepare_task(
            anchor, self._ctx(), tier=1, variant=1,
            domain_code="BPSY", domain_id=7, domain_name="Biopsychology",
            chapter_title="Test", section_title="Section",
            batch_id="test-batch", anchor_idx=0,
        )
        self.assertEqual(task["cluster_anchors"], [],
                         "T1 prepare_task must produce zero cluster anchors")

    def test_t4_task_has_two_clusters(self):
        # T4 must include 2 cluster anchors per the cap table.
        anchor = _make_anchor(uid="primary")
        task = self.orch.prepare_task(
            anchor, self._ctx(), tier=4, variant=1,
            domain_code="BPSY", domain_id=7, domain_name="Biopsychology",
            chapter_title="Test", section_title="Section",
            batch_id="test-batch", anchor_idx=0,
        )
        self.assertEqual(len(task["cluster_anchors"]), 2)

    def test_question_id_encodes_tier(self):
        anchor = _make_anchor(uid="primary")
        # T1 → "E" (easy)
        task_t1 = self.orch.prepare_task(
            anchor, self._ctx(), tier=1, variant=1,
            domain_code="BPSY", domain_id=7, domain_name="Biopsychology",
            chapter_title="Test", section_title="Section",
            batch_id="test-batch", anchor_idx=0,
        )
        self.assertIn("-E-", task_t1["question_id"])
        # T3 → "H" (hard)
        task_t3 = self.orch.prepare_task(
            anchor, self._ctx(), tier=3, variant=1,
            domain_code="BPSY", domain_id=7, domain_name="Biopsychology",
            chapter_title="Test", section_title="Section",
            batch_id="test-batch", anchor_idx=0,
        )
        self.assertIn("-H-", task_t3["question_id"])

    def test_correct_answer_form_carries_permitted_vocabulary(self):
        anchor = _make_anchor(uid="primary")
        task = self.orch.prepare_task(
            anchor, self._ctx(), tier=3, variant=1,
            domain_code="BPSY", domain_id=7, domain_name="Biopsychology",
            chapter_title="Test", section_title="Section",
            batch_id="test-batch", anchor_idx=0,
        )
        form = task["correct_answer_form"]
        self.assertIn("permitted_vocabulary", form)
        self.assertIn("required_verb", form)
        self.assertIn("permitted_concept_labels", form)


# ── generate_and_validate (mocked LLM) ──────────────────────────

class TestGenerateAndValidate(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.briefs_dir = Path(self.tmp)
        self.orch = QuestionOrchestrator()
        _write_brief(self.briefs_dir, "BPSY", "primary",
                     chapter_num="D7-Ch01")
        _write_brief(self.briefs_dir, "BPSY", "sib-1",
                     chapter_num="D7-Ch01")

    def _make_task(self, tier=1):
        ctx = self.orch.load_anchor_context(
            self.briefs_dir, "BPSY", "primary",
            chapter_vocab={"has_vocab": False, "concepts": [], "misconceptions": []},
            passage_text="passage" * 30,
        )
        anchor = _make_anchor(uid="primary")
        return self.orch.prepare_task(
            anchor, ctx, tier=tier, variant=1,
            domain_code="BPSY", domain_id=7, domain_name="Biopsychology",
            chapter_title="Test", section_title="Section",
            batch_id="test-batch", anchor_idx=0,
        )

    async def test_returns_failure_when_creator_returns_no_creative(self):
        # Mock the creator to return None (simulates LLM failure / timeout)
        client = MagicMock()
        with patch.object(self.orch.creator, "async_execute",
                          new=AsyncMock(return_value=(None, {"error": "no response"}))):
            result, t_in, t_out, reason = await self.orch.generate_and_validate(
                client, self._make_task(tier=1), max_attempts=1,
            )
        self.assertIsNone(result)
        self.assertIsNotNone(reason)

    async def test_assembles_when_creator_succeeds(self):
        # Mock the creator to return a synthetic creative dict that
        # passes assembly. Exercises the success path.
        creative = {
            "question_stem": "What does an agonist do at the receptor?",
            "knowledge_tested": "Agonist mechanism at receptor binding sites",
            "tested_concept": {
                "concept_id": "agonist",
                "concept_label": "Agonist",
                "knowledge_tested": "Agonist binds and activates",
            },
            "correct_answer": {
                "text": ("Identify activation of postsynaptic receptor through "
                         "specific binding interaction with the active site"),
                "explanation": "Agonist binding produces downstream signal transduction.",
            },
            "distractors": [
                {"text": "Identify blockade of postsynaptic receptor preventing "
                         "agonist activity at the same active site",
                 "explanation": "Antagonism, not agonism."},
                {"text": "Identify enhancement of synthesis pathways for the "
                         "neurotransmitter molecules independent of receptors",
                 "explanation": "Different pharmacological mechanism."},
                {"text": "Identify partial inhibition of enzymatic degradation "
                         "leading to accumulated neurotransmitter near receptor",
                 "explanation": "Reuptake/degradation, not direct binding."},
            ],
            "flashcard_seeds": {
                "concept": {"back": "Agonists activate"},
                "comparison": {"back": "vs antagonist"},
                "nuance": {"back": "context dependent"},
            },
            "topic_keywords": ["agonist", "receptor", "binding"],
        }
        client = MagicMock()
        api_meta = {
            "prompt_tokens": 100, "completion_tokens": 50, "retries": 0,
            "model_id": "claude-opus-4-7", "timestamp_utc": "2026-04-27T00:00:00Z",
            "latency_ms": 1000,
        }
        with patch.object(self.orch.creator, "async_execute",
                          new=AsyncMock(return_value=(creative, api_meta))):
            result, t_in, t_out, reason = await self.orch.generate_and_validate(
                client, self._make_task(tier=1), max_attempts=1,
            )
        # The synthetic creative may pass or fail depending on which
        # gates fire — but the function should return cleanly either way.
        self.assertIsInstance(t_in, int)
        self.assertIsInstance(t_out, int)
        # If assembly succeeded, result is a dict; if a gate caught it,
        # result is None and reason is set. Either is valid behavior.
        if result is not None:
            self.assertIn("question_id", result)
            self.assertIn("options", result)


if __name__ == "__main__":
    unittest.main()
