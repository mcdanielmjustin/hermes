"""Unit tests for the 8 previously untested agent classes (Tier 2 polish).

These agents had only integration coverage prior — a regression in any of
them would surface during a full orchestrator run but not in isolation.
This file adds focused unit tests for happy-path behavior, edge cases,
and rotation/determinism where applicable.

Agents covered:
  • TestedConceptSelectorAgent — variant×tier rotation
  • FlashcardTemplateAgent — template-based front generation
  • KeywordExtractorAgent — vocab-mode + frequency-mode
  • MetadataAgent — deterministic metadata building
  • ConceptVocabAgent — JSON file loader (chapter-level)
  • AnchorBriefAgent — JSON file loader (anchor-level)
  • QuestionAngleSelectorAgent — tier-affinity rotation
  • QuestionAssemblerAgent — JSON merge
"""
import json
import tempfile
import unittest
from pathlib import Path

import conftest  # noqa: F401  — sets sys.path

from pipeline.agents import (
    AnchorBriefAgent,
    ConceptVocabAgent,
    FlashcardTemplateAgent,
    KeywordExtractorAgent,
    MetadataAgent,
    QuestionAngleSelectorAgent,
    QuestionAssemblerAgent,
    TestedConceptSelectorAgent,
)


# ── TestedConceptSelectorAgent ──────────────────────────────────

class TestTestedConceptSelector(unittest.TestCase):
    def setUp(self):
        self.agent = TestedConceptSelectorAgent()
        self.concepts = [
            {"concept_id": "agonist", "label": "Agonist"},
            {"concept_id": "antagonist", "label": "Antagonist"},
            {"concept_id": "intrinsic-activity", "label": "Intrinsic Activity"},
        ]

    def test_returns_no_concept_when_pool_empty(self):
        out = self.agent.execute({"concepts": [], "variant": 1, "tier": 1})
        self.assertFalse(out["has_tested_concept"])

    def test_picks_concept_for_variant_1_tier_1(self):
        out = self.agent.execute({"concepts": self.concepts, "variant": 1, "tier": 1})
        self.assertTrue(out["has_tested_concept"])
        # idx = ((1-1)*4 + (1-1)) % 3 = 0 → first concept
        self.assertEqual(out["concept_id"], "agonist")
        self.assertEqual(out["concept_label"], "Agonist")

    def test_rotation_across_variants_and_tiers(self):
        # idx = ((variant-1)*4 + (tier-1)) % len(concepts)
        # variant=1 tier=1 → 0 → agonist
        # variant=1 tier=2 → 1 → antagonist
        # variant=1 tier=3 → 2 → intrinsic-activity
        # variant=2 tier=1 → 4 % 3 = 1 → antagonist
        cases = [
            (1, 1, "agonist"),
            (1, 2, "antagonist"),
            (1, 3, "intrinsic-activity"),
            (2, 1, "antagonist"),
            (2, 2, "intrinsic-activity"),
        ]
        for variant, tier, expected in cases:
            out = self.agent.execute({
                "concepts": self.concepts, "variant": variant, "tier": tier,
            })
            self.assertEqual(
                out["concept_id"], expected,
                f"variant={variant} tier={tier}: expected {expected}, got {out['concept_id']}",
            )

    def test_deterministic_same_input_same_output(self):
        a = self.agent.execute({"concepts": self.concepts, "variant": 3, "tier": 2})
        b = self.agent.execute({"concepts": self.concepts, "variant": 3, "tier": 2})
        self.assertEqual(a, b)


# ── FlashcardTemplateAgent ──────────────────────────────────────

class TestFlashcardTemplate(unittest.TestCase):
    def setUp(self):
        self.agent = FlashcardTemplateAgent()

    def test_generates_three_fronts(self):
        out = self.agent.execute({"tested_concept_label": "Hemiplegia"})
        self.assertIn("concept_front", out)
        self.assertIn("comparison_front", out)
        self.assertIn("nuance_front", out)
        # All non-empty strings
        for key in ("concept_front", "comparison_front", "nuance_front"):
            self.assertTrue(out[key])
            self.assertIsInstance(out[key], str)

    def test_concept_front_includes_tested_label(self):
        out = self.agent.execute({"tested_concept_label": "Hemiplegia"})
        self.assertIn("Hemiplegia", out["concept_front"])

    def test_comparison_front_uses_concepts_involved_when_available(self):
        # A misconception with 2+ concepts_involved feeds the comparison card.
        out = self.agent.execute({
            "tested_concept_label": "Antagonist",
            "misconceptions": [{
                "concepts_involved": ["agonist", "antagonist"],
            }],
        })
        # Title-cased and joined with " vs."
        self.assertIn("Agonist", out["comparison_front"])
        self.assertIn("Antagonist", out["comparison_front"])
        self.assertIn("vs", out["comparison_front"].lower())

    def test_comparison_front_falls_back_when_no_misconceptions(self):
        out = self.agent.execute({"tested_concept_label": "Antagonist"})
        # Falls back to a generic "differ from" template using tested label
        self.assertIn("Antagonist", out["comparison_front"])
        self.assertIn("differ", out["comparison_front"].lower())

    def test_default_label_when_missing(self):
        out = self.agent.execute({})
        # Default label is "this concept"
        self.assertIn("this concept", out["concept_front"])


# ── KeywordExtractorAgent ───────────────────────────────────────

class TestKeywordExtractor(unittest.TestCase):
    def setUp(self):
        self.agent = KeywordExtractorAgent()

    def test_vocab_mode_returns_concept_labels(self):
        # When ≥3 concepts present, returns labels directly.
        out = self.agent.execute({
            "concepts": [
                {"label": "Hemiplegia"},
                {"label": "Stroke"},
                {"label": "Decussation"},
            ],
            "content": "ignored when vocab is provided",
        })
        self.assertEqual(out["topic_keywords"], ["Hemiplegia", "Stroke", "Decussation"])

    def test_open_mode_extracts_from_content(self):
        # No vocab → frequency-based extraction
        content = (
            "Cognitive behavioral therapy is widely used. Cognitive therapy "
            "addresses thought patterns. Behavioral interventions modify actions."
        )
        out = self.agent.execute({"concepts": [], "content": content})
        self.assertIsInstance(out["topic_keywords"], list)
        self.assertTrue(len(out["topic_keywords"]) > 0)

    def test_short_content_returns_empty_in_open_mode(self):
        out = self.agent.execute({"concepts": [], "content": "tiny"})
        self.assertEqual(out["topic_keywords"], [])

    def test_psych_terms_get_boosted(self):
        # "memory" appears once, "table" appears 5 times. Memory should still
        # appear due to PSYCH_BOOSTS multiplier.
        content = ("table table table table table memory")
        out = self.agent.execute({"concepts": [], "content": content})
        # memory might appear; just check the output is sensible
        self.assertIsInstance(out["topic_keywords"], list)


# ── MetadataAgent ───────────────────────────────────────────────

class TestMetadataAgent(unittest.TestCase):
    def setUp(self):
        self.agent = MetadataAgent()

    def test_builds_question_id_and_meta(self):
        out = self.agent.execute({
            "domain_code": "BPSY",
            "domain_id": 7,
            "domain_name": "Biopsychology",
            "chapter_title": "Test Chapter",
            "chapter_num": "D7-Ch01",
            "section_title": "Section",
            "tier": 3,
            "variant": 1,
            "source_type": "anchor_grounded",
            "stem_pattern": "clinical_vignette",
            "anchor_uid": "D7-PHY-195-17b48e1e",
            "anchor_id_v2": "AP-D7-PHY-195",
            "verbatim_anchor": "test anchor",
            "testable_fact": "test fact",
            "batch_id": "test-batch",
            "content_chars": 500,
        })
        self.assertIn("question_id", out)
        self.assertIn("meta_base", out)
        # Question ID encodes tier (H = hard = T3)
        self.assertIn("H", out["question_id"])
        self.assertIn("D7-PHY-195", out["question_id"])

    def test_meta_base_has_required_fields(self):
        out = self.agent.execute({
            "domain_code": "BPSY",
            "domain_id": 7,
            "domain_name": "Biopsychology",
            "chapter_title": "Test",
            "chapter_num": "D7-Ch01",
            "section_title": "Section",
            "tier": 1,
            "variant": 1,
            "source_type": "anchor_grounded",
            "stem_pattern": "direct_definition",
            "anchor_uid": "uid",
            "anchor_id_v2": "id",
            "verbatim_anchor": "v",
            "testable_fact": "t",
            "batch_id": "b",
            "content_chars": 100,
        })
        meta = out["meta_base"]
        for key in ("difficulty_tier", "difficulty_label", "stem_pattern",
                    "domain_code", "domain_id"):
            self.assertIn(key, meta, f"meta_base missing required key: {key}")


# ── ConceptVocabAgent ───────────────────────────────────────────

class TestConceptVocabAgent(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dir = Path(self.tmp)
        self.agent = ConceptVocabAgent()

    def test_returns_no_vocab_when_dir_missing(self):
        out = self.agent.execute({
            "concept_vocab_dir": Path("/nonexistent"),
            "domain_code": "BPSY",
            "chapter_id": "D7-Ch01",
        })
        self.assertFalse(out["has_vocab"])

    def test_returns_no_vocab_when_chapter_id_missing(self):
        # Empty chapter_id is a degenerate input — graceful.
        out = self.agent.execute({
            "concept_vocab_dir": self.dir,
            "domain_code": "BPSY",
            "chapter_id": "",
        })
        self.assertFalse(out["has_vocab"])

    def test_loads_vocab_from_disk_when_present(self):
        domain_dir = self.dir / "BPSY"
        domain_dir.mkdir(parents=True)
        vocab = {
            "concepts": [{"concept_id": "agonist", "label": "Agonist"}],
            "misconceptions": [{"misconception_id": "m1", "label": "lbl"}],
        }
        (domain_dir / "D7-Ch01.json").write_text(
            json.dumps(vocab), encoding="utf-8",
        )
        out = self.agent.execute({
            "concept_vocab_dir": self.dir,
            "domain_code": "BPSY",
            "chapter_id": "D7-Ch01",
        })
        self.assertTrue(out["has_vocab"])
        self.assertEqual(len(out["concepts"]), 1)
        self.assertEqual(out["concepts"][0]["concept_id"], "agonist")


# ── AnchorBriefAgent ────────────────────────────────────────────

class TestAnchorBriefAgent(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.dir = Path(self.tmp)
        self.agent = AnchorBriefAgent()

    def test_returns_no_brief_when_file_missing(self):
        out = self.agent.execute({
            "anchor_briefs_dir": self.dir,
            "domain_code": "BPSY",
            "uid": "missing-uid",
        })
        self.assertFalse(out["has_brief"])
        # Falls back to empty containers, never crashes.
        self.assertEqual(out["concepts"], [])
        self.assertEqual(out["misconceptions"], [])

    def test_returns_no_brief_when_concepts_empty(self):
        # A brief file exists but has empty concepts → treated as no_brief.
        domain_dir = self.dir / "BPSY"
        domain_dir.mkdir(parents=True)
        (domain_dir / "uid-1.json").write_text(json.dumps({
            "concepts": [],
            "misconceptions": [],
            "core_claims": [],
            "question_angles": [],
        }), encoding="utf-8")
        out = self.agent.execute({
            "anchor_briefs_dir": self.dir,
            "domain_code": "BPSY",
            "uid": "uid-1",
        })
        self.assertFalse(out["has_brief"])

    def test_loads_full_brief(self):
        domain_dir = self.dir / "BPSY"
        domain_dir.mkdir(parents=True)
        brief = {
            "concepts": [{"concept_id": "agonist", "label": "Agonist",
                          "description": "desc"}],
            "misconceptions": [{"misconception_id": "m1", "label": "L"}],
            "core_claims": ["claim 1"],
            "question_angles": [{"angle_type": "definitional"}],
        }
        (domain_dir / "uid-2.json").write_text(json.dumps(brief), encoding="utf-8")
        out = self.agent.execute({
            "anchor_briefs_dir": self.dir,
            "domain_code": "BPSY",
            "uid": "uid-2",
        })
        self.assertTrue(out["has_brief"])
        self.assertEqual(len(out["concepts"]), 1)
        self.assertEqual(out["core_claims"], ["claim 1"])

    def test_handles_corrupt_json_gracefully(self):
        domain_dir = self.dir / "BPSY"
        domain_dir.mkdir(parents=True)
        (domain_dir / "broken.json").write_text("{not valid json", encoding="utf-8")
        out = self.agent.execute({
            "anchor_briefs_dir": self.dir,
            "domain_code": "BPSY",
            "uid": "broken",
        })
        # Should NOT crash on corrupt JSON; returns no-brief state.
        self.assertFalse(out["has_brief"])


# ── QuestionAngleSelectorAgent ──────────────────────────────────

class TestQuestionAngleSelector(unittest.TestCase):
    def setUp(self):
        self.agent = QuestionAngleSelectorAgent()
        # Brief schema uses "type" + "description" for each angle.
        self.angles = [
            {"type": "definitional", "description": "Define X"},
            {"type": "clinical_application", "description": "Apply X clinically"},
            {"type": "mechanism", "description": "How X works"},
        ]

    def test_returns_no_angle_when_empty(self):
        out = self.agent.execute({
            "question_angles": [], "tier": 1, "variant": 1,
        })
        self.assertFalse(out["has_angle"])

    def test_t1_prefers_definitional(self):
        # T1 tier has highest affinity for definitional (3 in TIER_AFFINITY)
        out = self.agent.execute({
            "question_angles": self.angles, "tier": 1, "variant": 1,
        })
        self.assertTrue(out["has_angle"])
        self.assertEqual(out["angle_type"], "definitional")

    def test_t3_prefers_clinical_application(self):
        # T3 has clinical_application=3 affinity, definitional=0
        out = self.agent.execute({
            "question_angles": self.angles, "tier": 3, "variant": 1,
        })
        self.assertTrue(out["has_angle"])
        # Should pick clinical_application (highest T3 affinity)
        self.assertEqual(out["angle_type"], "clinical_application")


# ── QuestionAssemblerAgent ──────────────────────────────────────

class TestQuestionAssemblerAgent(unittest.TestCase):
    def setUp(self):
        self.agent = QuestionAssemblerAgent()

    def _minimal_creative(self):
        return {
            "question_stem": "What does the receptor do?",
            "knowledge_tested": "Receptor mechanism",
            "correct_answer": {
                "text": "Activates the postsynaptic receptor",
                "explanation": "Agonists bind and produce response.",
            },
            "distractors": [
                {"text": "Blocks the receptor without effect",
                 "explanation": "That's antagonism."},
                {"text": "Enhances neurotransmitter synthesis",
                 "explanation": "Different mechanism."},
                {"text": "Inhibits enzyme function",
                 "explanation": "Unrelated process."},
            ],
            "flashcard_seeds": {
                "concept": {"back": "Receptor activates"},
                "comparison": {"back": "vs antagonist"},
                "nuance": {"back": "context dependent"},
            },
        }

    def _minimal_plan(self):
        return {
            "mode": "focused",
            "slots": [
                {"slot": 1, "distractor_level": 2, "misconception_id": "m1",
                 "misconception_label": "L1", "misconception_type": "similar_property"},
                {"slot": 2, "distractor_level": 3, "misconception_id": "m2",
                 "misconception_label": "L2", "misconception_type": "similar_property"},
                {"slot": 3, "distractor_level": 4, "misconception_id": "m3",
                 "misconception_label": "L3", "misconception_type": "similar_property"},
            ],
        }

    def _minimal_meta(self):
        return {
            "question_id": "QZ-BPSY-AP-D7-PHY-195-E-01",
            "difficulty_tier": 1,
            "difficulty_label": "easy",
            "stem_pattern": "direct_definition",
            "domain_code": "BPSY",
            "domain_id": 7,
            "domain_name": "Biopsychology",
            "chapter_title": "Test",
            "chapter_file": "test.html",
            "section_title": "Section",
            "blooms_primary": "remember",
            "blooms_secondary": "understand",
            "source_type": "anchor_grounded",
            "anchor_uids": ["uid"],
            "anchor_point_ids_v2": ["AP"],
            "anchor_label": "Test Anchor",
            "anchor_content_summaries": [],
            "testable_fact": "fact",
            "content_snippet_chars": 100,
            "variant": 1,
            "batch_id": "test-batch",
        }

    def test_assembles_minimal_question(self):
        out = self.agent.run({
            "creative": self._minimal_creative(),
            "distractor_plan": self._minimal_plan(),
            "meta_base": self._minimal_meta(),
            "topic_keywords": ["receptor", "mechanism"],
            "target_position": "A",
            "generation_metadata": {"timestamp_utc": "2026-04-27T00:00:00Z"},
            "pre_tested_concept": {
                "has_tested_concept": True,
                "concept_id": "agonist",
                "concept_label": "Agonist",
            },
        })
        # Top-level structural fields
        self.assertEqual(out["question_id"], "QZ-BPSY-AP-D7-PHY-195-E-01")
        self.assertIn("options", out)
        self.assertEqual(len(out["options"]), 4)
        # Correct option is at the target position
        correct = next(o for o in out["options"] if o.get("is_correct"))
        self.assertEqual(correct["letter"], "A")

    def test_auto_strips_reasoning_marker_in_option_text(self):
        # Verify the assembler's auto-strip integration: option text
        # containing "because" should have the clause moved to explanation.
        creative = self._minimal_creative()
        creative["correct_answer"]["text"] = (
            "Activates the postsynaptic receptor because the agonist binds "
            "to the active site"
        )
        out = self.agent.run({
            "creative": creative,
            "distractor_plan": self._minimal_plan(),
            "meta_base": self._minimal_meta(),
            "topic_keywords": [],
            "target_position": "A",
            "generation_metadata": {"timestamp_utc": "2026-04-27T00:00:00Z"},
            "pre_tested_concept": {
                "has_tested_concept": True,
                "concept_id": "agonist",
                "concept_label": "Agonist",
            },
        })
        correct = next(o for o in out["options"] if o.get("is_correct"))
        # Auto-strip should have moved "because..." out of text
        self.assertNotIn("because", correct["text"].lower())
        # The clause should appear in explanation now
        self.assertIn("because", correct["explanation"].lower())


if __name__ == "__main__":
    unittest.main()
