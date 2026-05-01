"""Tests for ConceptRegistry — cross-brief concept ID stability.

The registry deduplicates concept_ids when scaling to thousands of briefs.
Three resolution paths must all work, and the persistence layer must not
lose data across save/load cycles.
"""
import json
import tempfile
import pathlib
import unittest

import conftest  # noqa: F401  — sets sys.path

from pipeline.concept_registry import (
    ConceptRegistry, canonicalize_brief, _normalize_label,
)


class TestNormalizeLabel(unittest.TestCase):
    def test_lowercases(self):
        self.assertEqual(_normalize_label("Antagonist"), "antagonist")

    def test_strips_punctuation(self):
        self.assertEqual(_normalize_label("Cannon-Bard Theory"),
                         "cannon bard theory")

    def test_collapses_whitespace(self):
        self.assertEqual(_normalize_label("Hippocampal   Memory"),
                         "hippocampal memory")

    def test_empty(self):
        self.assertEqual(_normalize_label(""), "")
        self.assertEqual(_normalize_label(None), "")


class TestLookupOrRegister(unittest.TestCase):
    def setUp(self):
        # In-memory registry for each test
        self.tmpdir = tempfile.mkdtemp()
        self.path = pathlib.Path(self.tmpdir) / "registry.json"
        self.r = ConceptRegistry(self.path)

    def test_new_concept_registers(self):
        cid, is_new = self.r.lookup_or_register(
            "agonist", "Agonist", "A ligand that activates receptors.",
            brief_uid="brief-1",
        )
        self.assertEqual(cid, "agonist")
        self.assertTrue(is_new)
        self.assertEqual(self.r.stats()["total_concepts"], 1)

    def test_exact_id_match_reuses(self):
        self.r.lookup_or_register(
            "agonist", "Agonist", "...", brief_uid="brief-1",
        )
        cid, is_new = self.r.lookup_or_register(
            "agonist", "Agonist", "...", brief_uid="brief-2",
        )
        self.assertEqual(cid, "agonist")
        self.assertFalse(is_new)
        # Appearance recorded
        self.assertIn("brief-2",
                      self.r.data["concepts"]["agonist"]["appears_in"])

    def test_label_match_aliases(self):
        # First brief uses "agonist"
        self.r.lookup_or_register(
            "agonist", "Agonist", "...", brief_uid="brief-1",
        )
        # Second brief proposes "receptor-agonist" but same human label.
        cid, is_new = self.r.lookup_or_register(
            "receptor-agonist", "Agonist", "...", brief_uid="brief-2",
        )
        self.assertEqual(cid, "agonist")
        self.assertFalse(is_new)
        # Alias remembered
        self.assertEqual(self.r.data["aliases"]["receptor-agonist"], "agonist")

    def test_alias_match_resolves(self):
        # Set up an alias via label-match path
        self.r.lookup_or_register("agonist", "Agonist", "...",
                                  brief_uid="brief-1")
        self.r.lookup_or_register("receptor-agonist", "Agonist", "...",
                                  brief_uid="brief-2")
        # Now look up the alias directly
        cid, is_new = self.r.lookup_or_register(
            "receptor-agonist", "Agonist (different label)", "...",
            brief_uid="brief-3",
        )
        self.assertEqual(cid, "agonist")
        self.assertFalse(is_new)

    def test_different_concepts_dont_collide(self):
        # "Antagonist" and "Agonist" have very similar labels but the
        # registry doesn't fuzzy-match — they should stay separate.
        a, _ = self.r.lookup_or_register("agonist", "Agonist", "...",
                                         brief_uid="brief-1")
        b, _ = self.r.lookup_or_register("antagonist", "Antagonist", "...",
                                         brief_uid="brief-1")
        self.assertNotEqual(a, b)
        self.assertEqual(self.r.stats()["total_concepts"], 2)

    def test_appearance_tracked_uniquely(self):
        self.r.lookup_or_register("agonist", "Agonist", "...",
                                  brief_uid="brief-1")
        self.r.lookup_or_register("agonist", "Agonist", "...",
                                  brief_uid="brief-1")  # same brief
        self.r.lookup_or_register("agonist", "Agonist", "...",
                                  brief_uid="brief-2")
        appearances = self.r.data["concepts"]["agonist"]["appears_in"]
        self.assertEqual(appearances, ["brief-1", "brief-2"])


class TestPersistence(unittest.TestCase):
    def test_save_and_reload(self):
        tmpdir = tempfile.mkdtemp()
        path = pathlib.Path(tmpdir) / "registry.json"

        r1 = ConceptRegistry(path)
        r1.lookup_or_register("agonist", "Agonist", "desc",
                              brief_uid="brief-1")
        r1.save()

        # Reload from disk
        r2 = ConceptRegistry(path)
        self.assertEqual(r2.stats()["total_concepts"], 1)
        self.assertIn("agonist", r2.data["concepts"])
        self.assertEqual(r2.data["concepts"]["agonist"]["label"], "Agonist")


class TestCanonicalizeBrief(unittest.TestCase):
    def setUp(self):
        tmpdir = tempfile.mkdtemp()
        self.path = pathlib.Path(tmpdir) / "registry.json"
        self.r = ConceptRegistry(self.path)
        # Seed with one canonical concept
        self.r.lookup_or_register("agonist", "Agonist", "first version",
                                  brief_uid="seed")

    def test_remaps_aliased_concept_id(self):
        brief = {
            "uid": "test-brief",
            "concepts": [
                {"concept_id": "receptor-agonist",
                 "label": "Agonist", "description": "..."}
            ],
            "misconceptions": [],
        }
        n_new, n_aliased = canonicalize_brief(brief, self.r)
        self.assertEqual(n_new, 0)
        self.assertEqual(n_aliased, 1)
        self.assertEqual(brief["concepts"][0]["concept_id"], "agonist")

    def test_remaps_misconception_concepts_involved(self):
        brief = {
            "uid": "test-brief",
            "concepts": [
                {"concept_id": "receptor-agonist", "label": "Agonist",
                 "description": "..."},
                {"concept_id": "antagonist", "label": "Antagonist",
                 "description": "..."},
            ],
            "misconceptions": [
                {"misconception_id": "agonist-vs-antagonist",
                 "label": "...",
                 "type": "opposite_direction",
                 "concepts_involved": ["receptor-agonist", "antagonist"]},
            ],
        }
        canonicalize_brief(brief, self.r)
        # The aliased ID should be remapped here too.
        self.assertEqual(
            brief["misconceptions"][0]["concepts_involved"],
            ["agonist", "antagonist"],
        )

    def test_new_concepts_register_fresh(self):
        brief = {
            "uid": "test-brief",
            "concepts": [
                {"concept_id": "intrinsic-activity",
                 "label": "Intrinsic Activity", "description": "..."},
            ],
            "misconceptions": [],
        }
        n_new, n_aliased = canonicalize_brief(brief, self.r)
        self.assertEqual(n_new, 1)
        self.assertEqual(n_aliased, 0)
        self.assertIn("intrinsic-activity", self.r.data["concepts"])


if __name__ == "__main__":
    unittest.main()
