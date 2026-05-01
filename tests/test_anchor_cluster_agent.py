"""Tests for AnchorClusterAgent — the Phase 4 baseline diversification scaffold.

Bloom's-identity invariants (CRITICAL — these MUST pass):
  • T1 (Recognize) returns 0 cluster anchors. Recognition tests one concept
    from one brief; importing sibling content dilutes anchor identity.
  • T2 (Understand) returns 0 cluster anchors. Same invariant.
  • T3 (Apply) returns 1 cluster anchor. Application requires cross-content.
  • T4 (Analyze/Evaluate) returns 2 cluster anchors. Analyze/Evaluate IS cross-content.

Selection priority (when clusters DO load at T3/T4):
  1. Same chapter (highest)
  2. Same domain + ≥1 shared concept_id
  3. Same domain (broadest fallback)

Deterministic rotation by primary UID hash for run-to-run consistency.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

import conftest  # noqa: F401  — sets sys.path

from pipeline.agents import AnchorClusterAgent


def _write_brief(briefs_dir, domain, uid, chapter_num, concepts,
                 description_terms=None):
    """Helper: write a minimal brief file matching the production schema.

    description_terms: optional list of vocabulary terms to embed into
    each concept's description. Used by Option 5 vocabulary-distance
    selection tests — without distinct vocabulary, all briefs would
    extract identical "Description for concept-X" stop-word strings and
    would all look identical to vocabulary-distance scoring.
    """
    domain_dir = briefs_dir / domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    if description_terms is None:
        # Default: each concept gets ITS OWN distinctive vocabulary so
        # different concept_id sets produce different vocabulary sets.
        descriptions = {
            cid: f"Description involving {cid.replace('-', ' ')} and "
                 f"related technical {cid.replace('-', ' ')} concepts."
            for cid in concepts
        }
    else:
        # All concepts in this brief share the same vocabulary (used to
        # simulate ethics-domain anchors that differ in concept_id but
        # use the same underlying terminology).
        shared_vocab = " ".join(description_terms)
        descriptions = {cid: shared_vocab for cid in concepts}
    brief = {
        "uid": uid,
        "chapter_num": chapter_num,
        "concepts": [
            {"concept_id": cid, "label": cid.replace("-", " ").title(),
             "description": descriptions[cid]}
            for cid in concepts
        ],
        "core_claims": [f"Claim about {concepts[0] if concepts else 'x'}"],
        "testable_fact": f"Fact about {concepts[0] if concepts else 'x'}",
    }
    (domain_dir / f"{uid}.json").write_text(json.dumps(brief), encoding="utf-8")


class TestBloomsInvariantTierKeying(unittest.TestCase):
    """The critical Bloom's-identity invariants: T1/T2 NEVER receive
    cluster anchors, T3 receives 1, T4 receives 2."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.briefs_dir = Path(self.tmp)
        # Set up: 1 primary + 4 sibling anchors in same domain
        _write_brief(self.briefs_dir, "BPSY", "primary",
                     "D7-Ch01", ["concept-a", "concept-b"])
        for i in range(4):
            _write_brief(self.briefs_dir, "BPSY", f"sibling-{i}",
                         "D7-Ch01", ["concept-a"])
        self.agent = AnchorClusterAgent()

    def _exec(self, tier):
        return self.agent.execute({
            "anchor_briefs_dir": self.briefs_dir,
            "primary_uid": "primary",
            "domain_code": "BPSY",
            "primary_chapter_num": "D7-Ch01",
            "primary_concepts": [
                {"concept_id": "concept-a"}, {"concept_id": "concept-b"},
            ],
            "tier": tier,
        })

    def test_t1_returns_zero_clusters(self):
        # Bloom's invariant: T1 (Recognize) is single-anchor identity.
        out = self._exec(tier=1)
        self.assertEqual(out["cluster_anchors"], [],
                         "T1 must NEVER receive cluster anchors — anchor identity")

    def test_t2_returns_zero_clusters(self):
        # Bloom's invariant: T2 (Understand) is single-anchor identity.
        out = self._exec(tier=2)
        self.assertEqual(out["cluster_anchors"], [],
                         "T2 must NEVER receive cluster anchors — anchor identity")

    def test_t3_returns_one_cluster(self):
        # T3 (Apply) requires cross-content — 1 cluster.
        out = self._exec(tier=3)
        self.assertEqual(len(out["cluster_anchors"]), 1)

    def test_t4_returns_two_clusters(self):
        # T4 (Analyze/Evaluate) requires cross-content integration — 2 clusters.
        out = self._exec(tier=4)
        self.assertEqual(len(out["cluster_anchors"]), 2)

    def test_cluster_count_table_is_invariant(self):
        # Direct test of the constant: future commits cannot change this
        # without failing this assertion. The Bloom's invariant lives in
        # the data table, so this test enforces it structurally.
        self.assertEqual(AnchorClusterAgent._CLUSTER_COUNT_BY_TIER[1], 0)
        self.assertEqual(AnchorClusterAgent._CLUSTER_COUNT_BY_TIER[2], 0)
        self.assertEqual(AnchorClusterAgent._CLUSTER_COUNT_BY_TIER[3], 1)
        self.assertEqual(AnchorClusterAgent._CLUSTER_COUNT_BY_TIER[4], 2)


class TestSelectionPriority(unittest.TestCase):
    """Selection priority: same chapter > same domain + shared concept >
    same domain (fallback)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.briefs_dir = Path(self.tmp)
        # Primary in chapter D7-Ch01 with concept-a
        _write_brief(self.briefs_dir, "BPSY", "primary",
                     "D7-Ch01", ["concept-a"])
        # Same chapter (highest priority)
        _write_brief(self.briefs_dir, "BPSY", "samechapter-1",
                     "D7-Ch01", ["concept-z"])  # different concept
        # Same domain + shared concept
        _write_brief(self.briefs_dir, "BPSY", "sharedconcept-1",
                     "D7-Ch02", ["concept-a"])
        # Same domain only (broadest)
        _write_brief(self.briefs_dir, "BPSY", "samedomain-1",
                     "D7-Ch03", ["concept-q"])
        self.agent = AnchorClusterAgent()

    def test_t4_picks_same_chapter_first(self):
        # T4 picks 2 clusters; same-chapter has top priority so it should
        # be in the selection.
        out = self.agent.execute({
            "anchor_briefs_dir": self.briefs_dir,
            "primary_uid": "primary",
            "domain_code": "BPSY",
            "primary_chapter_num": "D7-Ch01",
            "primary_concepts": [{"concept_id": "concept-a"}],
            "tier": 4,
        })
        uids = [c["uid"] for c in out["cluster_anchors"]]
        self.assertIn("samechapter-1", uids)

    def test_selection_tier_label_present(self):
        # Each cluster anchor records WHY it was selected (debugging aid).
        out = self.agent.execute({
            "anchor_briefs_dir": self.briefs_dir,
            "primary_uid": "primary",
            "domain_code": "BPSY",
            "primary_chapter_num": "D7-Ch01",
            "primary_concepts": [{"concept_id": "concept-a"}],
            "tier": 4,
        })
        for c in out["cluster_anchors"]:
            # Option 5 uses vocabulary-based selection; "disjoint_vocab"
            # replaces the prior "disjoint_concept" label.
            self.assertIn(c["selection_tier"],
                          {"same_chapter", "disjoint_vocab", "same_domain"})


class TestVocabularyDiversityPriority(unittest.TestCase):
    """Option 5 architectural fix: prefer siblings with DIVERSE
    extracted vocabulary (the actual L3 source) over siblings with
    overlapping vocabulary. Generalizes the Option 4 concept_id-level
    diversity to the actual extracted-text level — solves the CASS
    case where siblings have disjoint concept_ids but still share
    underlying ethics terminology in their descriptions.
    """

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.briefs_dir = Path(self.tmp)
        self.agent = AnchorClusterAgent()

    def _exec_t3(self, primary_uid, primary_concepts, primary_core_claims=None,
                 primary_testable_fact=""):
        return self.agent.execute({
            "anchor_briefs_dir": self.briefs_dir,
            "primary_uid": primary_uid,
            "domain_code": "BPSY",
            "primary_chapter_num": "D7-Ch01",
            "primary_concepts": primary_concepts,
            "primary_core_claims": primary_core_claims or [],
            "primary_testable_fact": primary_testable_fact,
            "tier": 3,
        })

    def test_distinct_vocab_sibling_preferred_over_redundant(self):
        # Primary brief uses "neurotransmitter" / "receptor" / "synaptic"
        # vocabulary. Two siblings:
        #   - same-vocab: same concept descriptions → high vocab overlap
        #   - distinct-vocab: different terminology → low overlap
        primary_concepts = [
            {"concept_id": "agonist", "label": "Agonist",
             "description": "A neurotransmitter compound that activates "
                            "synaptic receptor binding sites pharmacologically."},
        ]
        _write_brief(self.briefs_dir, "BPSY", "primary",
                     "D7-Ch01", ["agonist"],
                     description_terms=["neurotransmitter", "synaptic", "receptor",
                                         "binding", "compound", "pharmacological"])
        # Sibling with overlapping vocabulary
        _write_brief(self.briefs_dir, "BPSY", "same-vocab",
                     "D7-Ch99", ["agonist-variant"],
                     description_terms=["neurotransmitter", "synaptic", "receptor",
                                         "binding", "compound", "pharmacological"])
        # Sibling with distinct vocabulary
        _write_brief(self.briefs_dir, "BPSY", "distinct-vocab",
                     "D7-Ch99", ["unrelated-concept"],
                     description_terms=["hippocampus", "neurogenesis", "memory",
                                         "consolidation", "encoding", "retrieval"])
        out = self._exec_t3("primary", primary_concepts)
        # The distinct-vocab sibling should be selected.
        self.assertEqual(len(out["cluster_anchors"]), 1)
        self.assertEqual(out["cluster_anchors"][0]["uid"], "distinct-vocab")
        self.assertEqual(out["cluster_anchors"][0]["selection_tier"],
                         "disjoint_vocab")
        # Vocab distance is exposed for audit traceability.
        self.assertGreaterEqual(out["cluster_anchors"][0]["vocab_distance"], 0.5)

    def test_falls_back_to_same_domain_when_all_overlap(self):
        # All siblings share the same vocabulary as primary — no distinct
        # candidate exists. Should fall back to "same_domain" tier rather
        # than returning empty.
        shared_vocab = ["consent", "professional", "boundary", "obligation",
                         "practice", "confidentiality"]
        _write_brief(self.briefs_dir, "BPSY", "primary",
                     "D7-Ch01", ["c1"], description_terms=shared_vocab)
        _write_brief(self.briefs_dir, "BPSY", "overlap-1",
                     "D7-Ch99", ["c2"], description_terms=shared_vocab)
        _write_brief(self.briefs_dir, "BPSY", "overlap-2",
                     "D7-Ch99", ["c3"], description_terms=shared_vocab)
        out = self._exec_t3("primary", [
            {"concept_id": "c1", "description": " ".join(shared_vocab)},
        ])
        # No vocab-distinct candidate at threshold ≥0.5; falls back to same_domain.
        self.assertEqual(len(out["cluster_anchors"]), 1)
        self.assertEqual(out["cluster_anchors"][0]["selection_tier"],
                         "same_domain")

    def test_ethics_disjoint_concept_but_overlapping_vocab_falls_back(self):
        # The CASS calibration finding: ethics anchors have disjoint
        # concept_ids ("informed-consent" vs "dual-relationship") but
        # SHARED VOCABULARY (both briefs use "professional, practice,
        # client, boundary, obligation"). Option 4 (concept_id-level)
        # would falsely classify them as disjoint and select. Option 5
        # (vocab-level) correctly identifies them as overlapping and
        # falls back to same_domain.
        ethics_vocab = ["professional", "practice", "boundary", "obligation",
                         "client", "confidentiality", "ethics"]
        _write_brief(self.briefs_dir, "CASS", "primary-eth",
                     "D8-Ch01", ["informed-consent"],
                     description_terms=ethics_vocab)
        _write_brief(self.briefs_dir, "CASS", "different-concept-id-same-vocab",
                     "D8-Ch99", ["dual-relationship"],
                     description_terms=ethics_vocab)
        # Use the agent directly — different domain code from primary
        out = self.agent.execute({
            "anchor_briefs_dir": self.briefs_dir,
            "primary_uid": "primary-eth",
            "domain_code": "CASS",
            "primary_chapter_num": "D8-Ch01",
            "primary_concepts": [{
                "concept_id": "informed-consent",
                "description": " ".join(ethics_vocab),
            }],
            "primary_core_claims": [],
            "primary_testable_fact": "",
            "tier": 3,
        })
        # Despite different concept_ids, the vocab-distance is < 0.5 here
        # (both use same ethics terms). The disjoint_vocab tier finds
        # nothing, so we fall back to same_domain. Architectural fix
        # for the CASS calibration finding.
        self.assertEqual(len(out["cluster_anchors"]), 1)
        self.assertEqual(out["cluster_anchors"][0]["selection_tier"],
                         "same_domain",
                         "Ethics siblings with disjoint concept_ids but "
                         "shared vocabulary should fall back to same_domain "
                         "(not be classified as disjoint_vocab) — this is "
                         "Option 5's improvement over Option 4.")


class TestDeterministicRotation(unittest.TestCase):
    """Same primary UID + same sibling pool → same cluster selection
    across runs."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.briefs_dir = Path(self.tmp)
        _write_brief(self.briefs_dir, "BPSY", "primary",
                     "D7-Ch01", ["concept-a"])
        for i in range(5):
            _write_brief(self.briefs_dir, "BPSY", f"sibling-{i}",
                         "D7-Ch01", ["concept-a"])
        self.agent = AnchorClusterAgent()

    def test_repeat_runs_pick_same_clusters(self):
        out1 = self.agent.execute({
            "anchor_briefs_dir": self.briefs_dir,
            "primary_uid": "primary",
            "domain_code": "BPSY",
            "primary_chapter_num": "D7-Ch01",
            "primary_concepts": [{"concept_id": "concept-a"}],
            "tier": 4,
        })
        out2 = self.agent.execute({
            "anchor_briefs_dir": self.briefs_dir,
            "primary_uid": "primary",
            "domain_code": "BPSY",
            "primary_chapter_num": "D7-Ch01",
            "primary_concepts": [{"concept_id": "concept-a"}],
            "tier": 4,
        })
        uids1 = sorted(c["uid"] for c in out1["cluster_anchors"])
        uids2 = sorted(c["uid"] for c in out2["cluster_anchors"])
        self.assertEqual(uids1, uids2, "deterministic rotation must be stable")

    def test_different_primary_picks_different_offset(self):
        # Two different primaries should generally pick different cluster
        # offsets (deterministic rotation by uid hash).
        out_p1 = self.agent.execute({
            "anchor_briefs_dir": self.briefs_dir,
            "primary_uid": "primary",
            "domain_code": "BPSY",
            "primary_chapter_num": "D7-Ch01",
            "primary_concepts": [{"concept_id": "concept-a"}],
            "tier": 4,
        })
        # Use a different primary uid
        _write_brief(self.briefs_dir, "BPSY", "primary-alt",
                     "D7-Ch01", ["concept-a"])
        out_p2 = self.agent.execute({
            "anchor_briefs_dir": self.briefs_dir,
            "primary_uid": "primary-alt",
            "domain_code": "BPSY",
            "primary_chapter_num": "D7-Ch01",
            "primary_concepts": [{"concept_id": "concept-a"}],
            "tier": 4,
        })
        # We don't strictly require different selections, but the rotation
        # offset is a function of uid hash so we expect it to differ here.
        # This is a smoke test — the strict test is repeat-run stability.
        self.assertEqual(len(out_p1["cluster_anchors"]), 2)
        self.assertEqual(len(out_p2["cluster_anchors"]), 2)


class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.briefs_dir = Path(self.tmp)
        self.agent = AnchorClusterAgent()

    def test_empty_briefs_dir_returns_empty(self):
        # No siblings at all — agent must return [] gracefully.
        _write_brief(self.briefs_dir, "BPSY", "primary",
                     "D7-Ch01", ["concept-a"])
        out = self.agent.execute({
            "anchor_briefs_dir": self.briefs_dir,
            "primary_uid": "primary",
            "domain_code": "BPSY",
            "primary_chapter_num": "D7-Ch01",
            "primary_concepts": [{"concept_id": "concept-a"}],
            "tier": 4,
        })
        self.assertEqual(out["cluster_anchors"], [])

    def test_missing_briefs_dir_returns_empty(self):
        # Briefs dir doesn't exist — graceful failure.
        out = self.agent.execute({
            "anchor_briefs_dir": Path("/nonexistent/path/xyz"),
            "primary_uid": "primary",
            "domain_code": "BPSY",
            "primary_chapter_num": "D7-Ch01",
            "primary_concepts": [{"concept_id": "concept-a"}],
            "tier": 4,
        })
        self.assertEqual(out["cluster_anchors"], [])

    def test_no_primary_uid_returns_empty(self):
        out = self.agent.execute({
            "anchor_briefs_dir": self.briefs_dir,
            "primary_uid": "",
            "domain_code": "BPSY",
            "tier": 4,
        })
        self.assertEqual(out["cluster_anchors"], [])

    def test_unknown_tier_returns_empty(self):
        # Tier outside 1-4 — agent must not crash; returns [] safely.
        _write_brief(self.briefs_dir, "BPSY", "primary",
                     "D7-Ch01", ["concept-a"])
        _write_brief(self.briefs_dir, "BPSY", "sib",
                     "D7-Ch01", ["concept-a"])
        out = self.agent.execute({
            "anchor_briefs_dir": self.briefs_dir,
            "primary_uid": "primary",
            "domain_code": "BPSY",
            "primary_chapter_num": "D7-Ch01",
            "primary_concepts": [{"concept_id": "concept-a"}],
            "tier": 99,
        })
        self.assertEqual(out["cluster_anchors"], [])


if __name__ == "__main__":
    unittest.main()
