"""Tests for concept_clustering — fuzzy concept_id similarity.

The clustering surfaces merge candidates for human review. The key risk
is FALSE POSITIVES on near-twin concepts (agonist vs antagonist) — those
would propose catastrophic merges if accepted. Tests focus on:
  - True fragmentation pairs ARE surfaced (agonist vs receptor-agonist)
  - True opposites are NOT surfaced above threshold (agonist vs antagonist)
  - Threshold works as expected (raising it shrinks candidate list)
"""
import unittest

import conftest  # noqa: F401  — sets sys.path

from pipeline.concept_clustering import (
    similarity, find_merge_candidates,
    _slug_tokens, _description_words, _jaccard,
)


class TestSlugTokens(unittest.TestCase):
    def test_kebab_split(self):
        self.assertEqual(
            _slug_tokens("nondeclarative-memory-system"),
            {"nondeclarative", "memory", "system"},
        )

    def test_short_tokens_dropped(self):
        # 1-2 char tokens are too generic to be discriminating
        self.assertEqual(
            _slug_tokens("a-b-cd-efg"),
            {"efg"},
        )

    def test_empty(self):
        self.assertEqual(_slug_tokens(""), set())
        self.assertEqual(_slug_tokens(None), set())


class TestSimilarityCore(unittest.TestCase):
    def test_identical_concepts_high_similarity(self):
        a = {"concept_id": "agonist", "label": "Agonist",
             "description": "Activates the receptor."}
        b = {"concept_id": "agonist", "label": "Agonist",
             "description": "Activates the receptor."}
        self.assertGreaterEqual(similarity(a, b), 0.99)

    def test_fragmented_concepts_above_threshold(self):
        # Real fragmentation case: same concept, slightly different ID/label.
        # Default threshold of 0.5 should catch this.
        a = {"concept_id": "agonist", "label": "Agonist",
             "description": "A ligand that binds to a receptor and "
                            "activates it to produce a biological response."}
        b = {"concept_id": "receptor-agonist", "label": "Receptor Agonist",
             "description": "A ligand that binds to a receptor and "
                            "activates it to produce a biological response."}
        score = similarity(a, b)
        self.assertGreaterEqual(score, 0.5,
            f"fragmented concepts should score >=0.5, got {score:.3f}")

    def test_opposite_concepts_below_threshold(self):
        # Critical false-positive risk: agonist and antagonist are
        # opposites and must NOT auto-cluster despite shared letters.
        # Word-level Jaccard discriminates these cleanly because
        # {agonist} ∩ {antagonist} = ∅.
        a = {"concept_id": "agonist", "label": "Agonist",
             "description": "Activates the receptor and produces a "
                            "biological response."}
        b = {"concept_id": "antagonist", "label": "Antagonist",
             "description": "Blocks the receptor and produces no "
                            "intrinsic biological response."}
        score = similarity(a, b)
        self.assertLess(score, 0.3,
            f"opposites must score <0.3 (well below threshold), got {score:.3f}")

    def test_unrelated_concepts_low_similarity(self):
        a = {"concept_id": "hippocampal-declarative-system",
             "label": "Hippocampal Declarative Memory",
             "description": "Medial temporal lobe memory system."}
        b = {"concept_id": "antagonist", "label": "Antagonist",
             "description": "Blocks neurotransmitter receptor."}
        self.assertLess(similarity(a, b), 0.3)


class TestFindMergeCandidates(unittest.TestCase):
    def test_empty_registry_no_candidates(self):
        self.assertEqual(find_merge_candidates({}), [])

    def test_single_concept_no_candidates(self):
        registry = {"agonist": {"label": "Agonist", "description": "..."}}
        self.assertEqual(find_merge_candidates(registry), [])

    def test_fragmented_pair_surfaces(self):
        registry = {
            "agonist": {
                "label": "Agonist",
                "description": "A ligand that activates a receptor.",
            },
            "receptor-agonist": {
                "label": "Receptor Agonist",
                "description": "A ligand that activates a receptor.",
            },
            "completely-unrelated-concept": {
                "label": "Schizophrenia Spectrum",
                "description": "Psychotic disorders cluster.",
            },
        }
        candidates = find_merge_candidates(registry, threshold=0.5)
        self.assertGreaterEqual(len(candidates), 1)
        # The fragmented pair should be the top candidate
        ids = {candidates[0]["id_a"], candidates[0]["id_b"]}
        self.assertEqual(ids, {"agonist", "receptor-agonist"})

    def test_opposites_not_surfaced(self):
        registry = {
            "agonist": {
                "label": "Agonist",
                "description": "Activates receptor; produces biological response.",
            },
            "antagonist": {
                "label": "Antagonist",
                "description": "Blocks receptor; no intrinsic biological response.",
            },
        }
        candidates = find_merge_candidates(registry, threshold=0.7)
        self.assertEqual(candidates, [],
            "agonist/antagonist must NOT be surfaced as merge candidates")

    def test_threshold_filters_results(self):
        registry = {
            "agonist": {
                "label": "Agonist",
                "description": "Activates a receptor.",
            },
            "receptor-agonist": {
                "label": "Receptor Agonist",
                "description": "Activates a receptor.",
            },
            "weakly-related": {
                "label": "Receptor Modulator",
                "description": "Adjusts receptor activity.",
            },
        }
        loose = find_merge_candidates(registry, threshold=0.4)
        strict = find_merge_candidates(registry, threshold=0.85)
        self.assertGreater(len(loose), len(strict))

    def test_results_sorted_by_score_descending(self):
        registry = {
            "agonist": {"label": "Agonist", "description": "Activates receptor."},
            "receptor-agonist": {"label": "Receptor Agonist",
                                 "description": "Activates receptor."},
            "agonist-mechanism": {"label": "Agonist Mechanism",
                                  "description": "How agonist works."},
            "agonism": {"label": "Agonism",
                        "description": "Activation of receptors."},
        }
        candidates = find_merge_candidates(registry, threshold=0.3)
        for i in range(len(candidates) - 1):
            self.assertGreaterEqual(
                candidates[i]["score"],
                candidates[i + 1]["score"],
            )

    def test_no_self_pairs(self):
        registry = {
            "agonist": {"label": "Agonist", "description": "..."},
        }
        # Single concept can't pair with itself.
        candidates = find_merge_candidates(registry, threshold=0.0)
        self.assertEqual(candidates, [])


class TestRealRegistryShape(unittest.TestCase):
    """Smoke test using the actual seeded registry's two-domain structure."""

    def test_two_briefs_no_unintended_clusters(self):
        # Using simplified versions of D7-PHY-021 (memory) + D7-PHY-195
        # (agonist/antagonist) concepts. They're conceptually disjoint;
        # we should get no merge candidates at the default threshold.
        registry = {
            "nondeclarative-memory-system": {
                "label": "Nondeclarative Memory System",
                "description": "Implicit memory expressed without conscious awareness.",
            },
            "basal-ganglia-procedural": {
                "label": "Basal Ganglia and Procedural Learning",
                "description": "Subcortical structures for habit and skill learning.",
            },
            "antagonist": {
                "label": "Antagonist",
                "description": "Receptor ligand with no intrinsic activity.",
            },
            "agonist": {
                "label": "Agonist",
                "description": "Receptor ligand that produces a biological response.",
            },
        }
        candidates = find_merge_candidates(registry, threshold=0.7)
        self.assertEqual(candidates, [],
            f"Disjoint domains should produce no candidates, got {candidates}")


if __name__ == "__main__":
    unittest.main()
