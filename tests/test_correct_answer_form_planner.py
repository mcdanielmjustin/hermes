"""Tests for CorrectAnswerFormPlannerAgent.

The agent is the Layer-3 (pre-LLM scaffolding) for the correct answer's
form. It mirrors DistractorPlannerAgent's strategy: collapse the LLM's
"decide what verb / shape / scope the correct answer takes" decision so
the prompt carries explicit constraints instead of hopeful guidance.

Without this scaffold, calibration showed:
  • T3 Bloom's bare-label answers — 4/5 violation rate
  • option_claim "X because Y" — 6/9 of all failures
  • scope_match concept-bloat — 2/9 failures on concept-rich briefs
"""
import unittest

import conftest  # noqa: F401  — sets sys.path

from pipeline.agents import CorrectAnswerFormPlannerAgent


CONCEPTS = [
    {"concept_id": "agonist", "label": "Agonist",
     "description": "A compound that binds and activates the receptor producing intrinsic activity."},
    {"concept_id": "antagonist", "label": "Antagonist",
     "description": "A compound that binds the receptor without producing intrinsic activity, blocking endogenous agonist binding."},
    {"concept_id": "receptor-blockade", "label": "Receptor Blockade",
     "description": "Mechanism by which an antagonist prevents agonist activation of the postsynaptic receptor."},
    {"concept_id": "intrinsic-activity", "label": "Intrinsic Activity",
     "description": "The capacity of a bound ligand to produce a downstream cellular response."},
    {"concept_id": "synthesis-inhibition", "label": "Synthesis Inhibition",
     "description": "Pharmacological reduction of neurotransmitter manufacture via enzyme blockade."},
]


class TestVerbAssignment(unittest.TestCase):
    def setUp(self):
        self.agent = CorrectAnswerFormPlannerAgent()

    def test_t1_picks_recognition_verb(self):
        out = self.agent.execute({"tier": 1, "variant": 1,
                                  "concepts": CONCEPTS})
        self.assertIn(out["required_verb"],
                      {"identify", "recognize", "name", "define", "label"})

    def test_t3_picks_application_verb(self):
        out = self.agent.execute({"tier": 3, "variant": 1,
                                  "concepts": CONCEPTS})
        application_verbs = {"predict", "determine", "apply", "evaluate",
                             "distinguish", "choose", "select", "infer"}
        self.assertIn(out["required_verb"], application_verbs)

    def test_t4_picks_analyze_evaluate_verb(self):
        out = self.agent.execute({"tier": 4, "variant": 1,
                                  "concepts": CONCEPTS})
        # T4 (Analyze/Evaluate) — the verb pool spans evaluation,
        # justification, comparison/integration. Pure "Create" verbs
        # (build, design) aren't in scope for MCQ format.
        analyze_evaluate_verbs = {"integrate", "synthesize", "evaluate",
                                  "justify", "reconcile", "weigh"}
        self.assertIn(out["required_verb"], analyze_evaluate_verbs)

    def test_verb_pool_returned_for_prompt(self):
        # The prompt lists the full pool so the LLM has flexibility within
        # the cognitive level — required_verb is the recommendation,
        # verb_pool is the permitted set.
        out = self.agent.execute({"tier": 3, "variant": 1,
                                  "concepts": CONCEPTS})
        self.assertGreaterEqual(len(out["verb_pool"]), 4)
        self.assertIn(out["required_verb"], out["verb_pool"])

    def test_variants_rotate_verbs(self):
        # Different variants should pick different verbs from the pool to
        # maximize cognitive-act diversity within a tier.
        verbs = set()
        for variant in range(1, 6):
            out = self.agent.execute({"tier": 3, "variant": variant,
                                      "concepts": CONCEPTS})
            verbs.add(out["required_verb"])
        self.assertGreater(len(verbs), 1,
                           "variants should rotate through the verb pool")


class TestConceptCap(unittest.TestCase):
    def setUp(self):
        self.agent = CorrectAnswerFormPlannerAgent()

    def test_t1_caps_at_one_concept(self):
        out = self.agent.execute({"tier": 1, "variant": 1,
                                  "concepts": CONCEPTS})
        self.assertEqual(out["max_concept_count"], 1)
        self.assertLessEqual(len(out["permitted_concept_ids"]), 1)

    def test_t3_caps_at_two_concepts(self):
        out = self.agent.execute({"tier": 3, "variant": 1,
                                  "concepts": CONCEPTS})
        self.assertEqual(out["max_concept_count"], 2)
        self.assertLessEqual(len(out["permitted_concept_ids"]), 2)

    def test_t4_caps_at_three_concepts(self):
        out = self.agent.execute({"tier": 4, "variant": 1,
                                  "concepts": CONCEPTS})
        self.assertEqual(out["max_concept_count"], 3)
        self.assertLessEqual(len(out["permitted_concept_ids"]), 3)

    def test_concept_rich_brief_capped_correctly(self):
        # The original scope_match failure: brief with 5 concepts produced
        # correct answers referencing all 5. The cap (3 for T4) prevents
        # this — the LLM is told it MAY only reference these N concepts.
        out = self.agent.execute({"tier": 4, "variant": 1,
                                  "concepts": CONCEPTS})  # 5 concepts in
        self.assertEqual(len(out["permitted_concept_ids"]), 3)


class TestPrimaryConceptAlignment(unittest.TestCase):
    def setUp(self):
        self.agent = CorrectAnswerFormPlannerAgent()

    def test_primary_concept_seeded_first(self):
        # The answer-plan must align with the distractor-plan: distractors
        # target the tested concept's misconceptions, so the correct
        # answer must reference that same concept (otherwise distractors
        # are off-topic for the answer).
        out = self.agent.execute({
            "tier": 3, "variant": 1, "concepts": CONCEPTS,
            "primary_concept_id": "antagonist",
        })
        self.assertEqual(out["permitted_concept_ids"][0], "antagonist")

    def test_primary_concept_unknown_falls_back_to_rotation(self):
        # If TestedConceptSelector has no concept (e.g., empty brief),
        # the planner just takes the first concepts in order.
        out = self.agent.execute({
            "tier": 3, "variant": 1, "concepts": CONCEPTS,
            "primary_concept_id": None,
        })
        self.assertEqual(len(out["permitted_concept_ids"]), 2)


class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.agent = CorrectAnswerFormPlannerAgent()

    def test_empty_concepts_returns_empty_permitted(self):
        out = self.agent.execute({"tier": 3, "variant": 1, "concepts": []})
        self.assertEqual(out["permitted_concept_ids"], [])
        self.assertEqual(out["permitted_concept_labels"], [])
        # Still returns a verb so the prompt has the cognitive constraint.
        self.assertTrue(out["required_verb"])

    def test_unknown_tier_falls_back_to_t1(self):
        out = self.agent.execute({"tier": 99, "variant": 1,
                                  "concepts": CONCEPTS})
        self.assertIn(out["required_verb"],
                      {"identify", "recognize", "name", "define", "label"})

    def test_option_form_constraint_present(self):
        # Static string forbidding causal markers in option text — the
        # primary fix for option_claim violations (6/9 of calibration
        # failures contained "because" / "since" in option.text).
        out = self.agent.execute({"tier": 3, "variant": 1,
                                  "concepts": CONCEPTS})
        constraint = out["option_form_constraint"]
        self.assertIn("because", constraint.lower())
        self.assertIn("explanation", constraint.lower())

    def test_option_form_includes_vocabulary_distribution(self):
        # Plan K: vocabulary distribution rule added to _OPTION_FORM after
        # D7-PHY-209 audit found 29% synonym-uniqueness on non-EXCEPT
        # questions. Forces distractors to share the correct's vocabulary.
        out = self.agent.execute({"tier": 3, "variant": 1,
                                  "concepts": CONCEPTS})
        constraint = out["option_form_constraint"]
        self.assertTrue(
            "distractor" in constraint.lower(),
            "option form must reference distractor vocabulary distribution",
        )
        self.assertTrue(
            any(t in constraint.lower() for t in ("technical term", "vocabulary")),
            "option form must call out technical-vocabulary distribution",
        )

    def test_option_length_constraint_present(self):
        # Length scaffold added after calibration regen showed length-
        # balance regression (5/20) when prompt-only length rules
        # weren't followed. Closes the multi-layer pattern for length.
        out = self.agent.execute({"tier": 3, "variant": 1,
                                  "concepts": CONCEPTS})
        constraint = out["option_length_constraint"]
        self.assertTrue(constraint, "option_length_constraint must be set")
        # Check that the constraint mentions explicit char range and parity
        self.assertTrue(
            any(token in constraint.lower() for token in ("character", "char")),
            "length constraint must specify a character measure",
        )
        self.assertIn("median", constraint.lower(),
                      "length constraint must explicitly target median parity "
                      "to defeat correct=longest tell")

    def test_option_length_caps_correct_at_median(self):
        # Plan J: tightened from "within ±20% of median" to "≤ median" after
        # D7-PHY-209 audit found 47-53% strict-longest correct. The "≤" or
        # "lean shorter" wording is the structural fix.
        out = self.agent.execute({"tier": 3, "variant": 1,
                                  "concepts": CONCEPTS})
        constraint = out["option_length_constraint"].lower()
        # Must contain "≤" symbol or "shorter" or "never the longest" — any
        # phrasing that bounds correct at-or-below median (not centered on it).
        cap_signals = ("≤", "shorter", "never the longest",
                       "must not be the longest")
        self.assertTrue(
            any(s in constraint for s in cap_signals),
            "length constraint must cap correct at-or-below median, not "
            "center it (the centering wording let LLM produce 47-53% "
            "strict-longest correct on D7-PHY-209)",
        )


class TestVocabularyScaffold(unittest.TestCase):
    """L3 vocabulary scaffold extracts technical terms from concept
    descriptions and exposes them so the distractor section can require
    distractors to share vocabulary with the correct option. Closes the
    L1+L2-only gap that left KeywordDistributionGate firing at ~30% on
    D7-PHY-209 phase 2 v3 batches.
    """

    def setUp(self):
        self.agent = CorrectAnswerFormPlannerAgent()

    def test_vocabulary_extracted_from_descriptions(self):
        out = self.agent.execute({
            "tier": 3, "variant": 1, "concepts": CONCEPTS,
            "primary_concept_id": "antagonist",
        })
        vocab = out["permitted_vocabulary"]
        self.assertTrue(vocab, "vocabulary must be non-empty when concepts have descriptions")
        # Antagonist's description contains 'producing', 'intrinsic',
        # 'activity', 'blocking', 'endogenous', 'binding' — at least
        # several should be in the extracted vocabulary.
        antagonist_terms = {"producing", "intrinsic", "activity",
                            "blocking", "endogenous", "binding"}
        overlap = antagonist_terms & set(vocab)
        self.assertGreaterEqual(
            len(overlap), 2,
            f"vocabulary should pull from antagonist description, got {vocab}",
        )

    def test_vocabulary_excludes_short_words(self):
        out = self.agent.execute({
            "tier": 3, "variant": 1, "concepts": CONCEPTS,
            "primary_concept_id": "antagonist",
        })
        vocab = out["permitted_vocabulary"]
        for w in vocab:
            self.assertGreaterEqual(
                len(w), 6,
                f"vocabulary words must be ≥6 chars; got '{w}'",
            )

    def test_vocabulary_capped(self):
        # Keep distractor over-constraint at bay — cap total vocabulary.
        out = self.agent.execute({
            "tier": 4, "variant": 1, "concepts": CONCEPTS,
            "primary_concept_id": "antagonist",
        })
        self.assertLessEqual(
            len(out["permitted_vocabulary"]), 8,
            "vocabulary must be capped at 8 terms to avoid over-constraining",
        )

    def test_vocabulary_empty_when_no_descriptions(self):
        bare_concepts = [
            {"concept_id": "x", "label": "X"},
            {"concept_id": "y", "label": "Y"},
        ]
        out = self.agent.execute({
            "tier": 3, "variant": 1, "concepts": bare_concepts,
        })
        self.assertEqual(out["permitted_vocabulary"], [])

    def test_vocabulary_dedupes_across_concepts(self):
        # If multiple permitted concepts share a term in their descriptions,
        # the term should appear once in the vocabulary.
        concepts_with_overlap = [
            {"concept_id": "a", "label": "A", "primary_concept_id": "a",
             "description": "Mechanism produces neurotransmitter binding cascades."},
            {"concept_id": "b", "label": "B",
             "description": "Process disrupts neurotransmitter binding selectively."},
        ]
        out = self.agent.execute({
            "tier": 4, "variant": 1, "concepts": concepts_with_overlap,
            "primary_concept_id": "a",
        })
        vocab = out["permitted_vocabulary"]
        # No duplicates
        self.assertEqual(len(vocab), len(set(vocab)),
                         "vocabulary must be deduplicated")


class TestDeterminism(unittest.TestCase):
    def setUp(self):
        self.agent = CorrectAnswerFormPlannerAgent()

    def test_same_input_same_output(self):
        # The whole point of pre-LLM scaffolding is determinism — running
        # twice must produce identical assignments so the contract is
        # stable across retries.
        a = self.agent.execute({"tier": 3, "variant": 2, "concepts": CONCEPTS,
                                "primary_concept_id": "agonist"})
        b = self.agent.execute({"tier": 3, "variant": 2, "concepts": CONCEPTS,
                                "primary_concept_id": "agonist"})
        self.assertEqual(a, b)


class TestBriefInternalPoolT1T2(unittest.TestCase):
    """Phase 6: T1/T2 broader brief-internal vocabulary pool.

    At recognition / understand tiers, vocabulary IS the test, but
    Bloom's invariant denies cluster anchors at those tiers. The only
    place to grow the term pool is the primary brief itself — pull from
    core_claims and testable_fact in addition to concept descriptions.

    T3/T4 keep narrow extraction (concept descriptions only) — cluster
    anchors are their diversification path.
    """

    def setUp(self):
        self.agent = CorrectAnswerFormPlannerAgent()

    # Use a primary brief with terms that DON'T appear in any concept
    # description above. That way we can detect whether the brief-
    # internal pool extension fired (terms in vocab) vs not (absent).
    _CORE_CLAIMS = [
        "Frontal lobe damage produces dysexecutive syndrome characterized by impaired planning.",
        "Prefrontal lesions disrupt working memory updating processes.",
    ]
    _TESTABLE_FACT = (
        "Phineas Gage's injury demonstrated personality alterations following "
        "ventromedial cortical destruction."
    )

    def test_t1_includes_core_claims_terms(self):
        out = self.agent.execute({
            "tier": 1, "variant": 1, "concepts": CONCEPTS,
            "primary_concept_id": "antagonist",
            "primary_core_claims": self._CORE_CLAIMS,
            "primary_testable_fact": self._TESTABLE_FACT,
        })
        vocab = set(out["permitted_vocabulary"])
        # Terms from core_claims (≥6 chars, not in concept descriptions).
        # "frontal", "dysexecutive", "prefrontal", "lesions", "planning",
        # "working", "memory", "updating", "processes", "syndrome", etc.
        brief_internal_candidates = {
            "frontal", "dysexecutive", "prefrontal", "lesions",
            "planning", "working", "memory", "updating", "processes",
            "syndrome", "impaired",
        }
        overlap = brief_internal_candidates & vocab
        self.assertGreaterEqual(
            len(overlap), 1,
            f"T1 vocab should include ≥1 term from core_claims; got {vocab}",
        )

    def test_t2_includes_testable_fact_terms(self):
        # Use bare concepts so testable_fact is the only source for some terms.
        out = self.agent.execute({
            "tier": 2, "variant": 1, "concepts": CONCEPTS,
            "primary_concept_id": "antagonist",
            "primary_core_claims": [],
            "primary_testable_fact": self._TESTABLE_FACT,
        })
        vocab = set(out["permitted_vocabulary"])
        # testable_fact-only terms: "phineas", "personality", "alterations",
        # "ventromedial", "cortical", "destruction".
        testable_candidates = {
            "personality", "alterations", "ventromedial", "cortical",
            "destruction", "demonstrated", "injury",
        }
        overlap = testable_candidates & vocab
        self.assertGreaterEqual(
            len(overlap), 1,
            f"T2 vocab should include ≥1 term from testable_fact; got {vocab}",
        )

    def test_t3_excludes_brief_internal_pool(self):
        # T3 must keep narrow extraction — cluster anchors handle T3+
        # diversification. Brief-internal terms (core_claims, testable_fact)
        # MUST NOT appear in T3's permitted_vocabulary.
        out = self.agent.execute({
            "tier": 3, "variant": 1, "concepts": CONCEPTS,
            "primary_concept_id": "antagonist",
            "primary_core_claims": self._CORE_CLAIMS,
            "primary_testable_fact": self._TESTABLE_FACT,
        })
        vocab = set(out["permitted_vocabulary"])
        # These terms appear ONLY in core_claims/testable_fact, NOT in the
        # CONCEPTS descriptions. T3 must not pick them up.
        brief_only_terms = {
            "dysexecutive", "prefrontal", "ventromedial", "phineas",
            "personality", "alterations",
        }
        leaked = brief_only_terms & vocab
        self.assertEqual(
            leaked, set(),
            f"T3 must not include brief-internal pool terms; leaked: {leaked}",
        )

    def test_t4_excludes_brief_internal_pool(self):
        # Same invariant as T3 — narrow extraction at analyze/evaluate tier.
        out = self.agent.execute({
            "tier": 4, "variant": 1, "concepts": CONCEPTS,
            "primary_concept_id": "antagonist",
            "primary_core_claims": self._CORE_CLAIMS,
            "primary_testable_fact": self._TESTABLE_FACT,
        })
        vocab = set(out["permitted_vocabulary"])
        brief_only_terms = {
            "dysexecutive", "prefrontal", "ventromedial", "phineas",
            "personality", "alterations",
        }
        leaked = brief_only_terms & vocab
        self.assertEqual(
            leaked, set(),
            f"T4 must not include brief-internal pool terms; leaked: {leaked}",
        )

    def test_t1_no_brief_data_falls_back_cleanly(self):
        # Defensive: T1 with no core_claims / testable_fact still produces
        # a well-formed vocab (just the concept-description terms).
        out = self.agent.execute({
            "tier": 1, "variant": 1, "concepts": CONCEPTS,
            "primary_concept_id": "antagonist",
        })
        # Just assert no crash and vocab is well-formed (list, capped).
        self.assertIsInstance(out["permitted_vocabulary"], list)
        self.assertLessEqual(len(out["permitted_vocabulary"]), 12)


class TestDomainVocabPoolT1T2(unittest.TestCase):
    """Phase 7: curated domain vocabulary pool.

    Domain pool is priority-3 (after concept descriptions and brief-
    internal pool). Lands at T1/T2 only — T3/T4 ignore it (cluster
    anchors are their diversification path). Cap raises to 12 at T1/T2
    so the wider pool has landing room.

    Bloom's-compliant: vocabulary only, no concepts imported.
    """

    # Domain pool with terms that aren't in CONCEPTS descriptions or
    # any brief-internal pool — so we can detect whether they pass
    # through to permitted_vocabulary.
    _DOMAIN_VOCAB = [
        "neuroplasticity", "encephalopathy", "myelination",
        "oligodendrocyte", "phagocytosis", "interneuron",
        "synaptogenesis", "neurodegeneration", "demyelination",
        "neurotransmission", "axoplasmic", "potentiation",
    ]

    def setUp(self):
        self.agent = CorrectAnswerFormPlannerAgent()

    def test_t1_includes_domain_vocab(self):
        out = self.agent.execute({
            "tier": 1, "variant": 1, "concepts": CONCEPTS,
            "primary_concept_id": "antagonist",
            "domain_vocab": self._DOMAIN_VOCAB,
        })
        vocab = set(out["permitted_vocabulary"])
        overlap = set(self._DOMAIN_VOCAB) & vocab
        self.assertGreaterEqual(
            len(overlap), 1,
            f"T1 vocab should include ≥1 domain term; got {vocab}",
        )

    def test_t2_includes_domain_vocab(self):
        out = self.agent.execute({
            "tier": 2, "variant": 1, "concepts": CONCEPTS,
            "primary_concept_id": "antagonist",
            "domain_vocab": self._DOMAIN_VOCAB,
        })
        vocab = set(out["permitted_vocabulary"])
        overlap = set(self._DOMAIN_VOCAB) & vocab
        self.assertGreaterEqual(
            len(overlap), 1,
            f"T2 vocab should include ≥1 domain term; got {vocab}",
        )

    def test_t3_excludes_domain_vocab(self):
        out = self.agent.execute({
            "tier": 3, "variant": 1, "concepts": CONCEPTS,
            "primary_concept_id": "antagonist",
            "domain_vocab": self._DOMAIN_VOCAB,
        })
        vocab = set(out["permitted_vocabulary"])
        leaked = set(self._DOMAIN_VOCAB) & vocab
        self.assertEqual(
            leaked, set(),
            f"T3 must not include domain pool terms; leaked: {leaked}",
        )

    def test_t4_excludes_domain_vocab(self):
        out = self.agent.execute({
            "tier": 4, "variant": 1, "concepts": CONCEPTS,
            "primary_concept_id": "antagonist",
            "domain_vocab": self._DOMAIN_VOCAB,
        })
        vocab = set(out["permitted_vocabulary"])
        leaked = set(self._DOMAIN_VOCAB) & vocab
        self.assertEqual(
            leaked, set(),
            f"T4 must not include domain pool terms; leaked: {leaked}",
        )

    def test_t1_t2_cap_raised_to_12(self):
        # Need enough source terms to actually fill past 8. Use a large
        # domain pool so the cap behavior is exercised.
        big_pool = [
            f"domainterm{i:02d}xxxxx" for i in range(20)
        ]
        out = self.agent.execute({
            "tier": 1, "variant": 1, "concepts": CONCEPTS,
            "primary_concept_id": "antagonist",
            "domain_vocab": big_pool,
        })
        self.assertLessEqual(len(out["permitted_vocabulary"]), 12,
                             "T1 cap should be 12")
        self.assertGreater(len(out["permitted_vocabulary"]), 8,
                           "T1 with 20+ source terms should land more than 8 "
                           "(pre-Phase-7 cap was 8)")

    def test_t3_t4_cap_stays_at_8(self):
        # T3/T4 must keep the strict 8-term cap.
        big_pool = [f"domainterm{i:02d}xxxxx" for i in range(20)]
        out = self.agent.execute({
            "tier": 3, "variant": 1, "concepts": CONCEPTS,
            "primary_concept_id": "antagonist",
            "domain_vocab": big_pool,  # ignored at T3/T4
        })
        self.assertLessEqual(len(out["permitted_vocabulary"]), 8,
                             "T3 cap should remain at 8")

    def test_short_domain_terms_filtered(self):
        # The form planner's threshold is 6+ chars (matches existing
        # extraction behavior). Short noise in the pool gets dropped.
        pool = ["abc", "in", "a", "very", "neuroplasticity"]
        out = self.agent.execute({
            "tier": 1, "variant": 1, "concepts": CONCEPTS,
            "primary_concept_id": "antagonist",
            "domain_vocab": pool,
        })
        vocab = set(out["permitted_vocabulary"])
        self.assertNotIn("abc", vocab)
        self.assertNotIn("in", vocab)
        self.assertNotIn("very", vocab)
        # 6+ char term should land
        self.assertIn("neuroplasticity", vocab)

    def test_empty_domain_vocab_no_crash(self):
        out = self.agent.execute({
            "tier": 1, "variant": 1, "concepts": CONCEPTS,
            "primary_concept_id": "antagonist",
            "domain_vocab": [],
        })
        self.assertIsInstance(out["permitted_vocabulary"], list)

    def test_missing_domain_vocab_key_no_crash(self):
        # No domain_vocab key in data dict — should fall back to current
        # Phase 6 behavior with cap 12 at T1/T2.
        out = self.agent.execute({
            "tier": 1, "variant": 1, "concepts": CONCEPTS,
            "primary_concept_id": "antagonist",
        })
        self.assertIsInstance(out["permitted_vocabulary"], list)


if __name__ == "__main__":
    unittest.main()
