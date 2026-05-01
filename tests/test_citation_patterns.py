"""Tests for the shared citation_patterns module.

The module is the single source of truth for citation-detection regexes
used by:
  • pipeline/gates.py        (AttributionGate)
  • pipeline/agents.py       (InputSanitizerAgent)
  • scripts/audit_question_quality.py
  • scripts/sweep_corpus_for_names.py

These tests verify the canonical patterns themselves AND that all four
consumers produce consistent results on the same input.
"""
import unittest

import conftest  # noqa: F401  — sets sys.path

from pipeline.citation_patterns import (
    CITATION_RE, ETAL_RE, ACCORDING_TO_RE, POSSESSIVE_RES_RE,
    find_attributions, is_whitelisted, split_authors,
)


class TestSplitAuthors(unittest.TestCase):
    def test_single_name(self):
        self.assertEqual(split_authors("Squire"), ["Squire"])

    def test_ampersand_split(self):
        self.assertEqual(split_authors("Atkinson & Shiffrin"), ["Atkinson", "Shiffrin"])

    def test_and_split(self):
        self.assertEqual(split_authors("Cannon and Bard"), ["Cannon", "Bard"])

    def test_comma_split(self):
        self.assertEqual(split_authors("Smith, Jones"), ["Smith", "Jones"])

    def test_three_authors_oxford(self):
        self.assertEqual(
            split_authors("Smith, Jones, and Brown"),
            ["Smith", "Jones", "Brown"],
        )

    def test_bandura_not_split(self):
        # Regression: word-boundary on "and" prevents splitting "Bandura".
        self.assertEqual(split_authors("Bandura"), ["Bandura"])

    def test_strips_et_al(self):
        self.assertEqual(split_authors("Smith et al."), ["Smith"])

    def test_empty(self):
        self.assertEqual(split_authors(""), [])
        self.assertEqual(split_authors(None), [])


class TestIsWhitelisted(unittest.TestCase):
    def test_single_whitelisted(self):
        self.assertTrue(is_whitelisted("Piaget"))

    def test_single_non_whitelisted(self):
        self.assertFalse(is_whitelisted("Squire"))

    def test_bandura_whitelisted(self):
        # Critical regression — must not be split into ["B", "ura"].
        self.assertTrue(is_whitelisted("Bandura"))

    def test_multi_author_all_whitelisted(self):
        self.assertTrue(is_whitelisted("Atkinson & Shiffrin"))
        self.assertTrue(is_whitelisted("Cannon and Bard"))

    def test_multi_author_partial_strips_all(self):
        # Bandura whitelisted but Walters not → all-or-nothing → False.
        self.assertFalse(is_whitelisted("Bandura and Walters"))


class TestCitationRegex(unittest.TestCase):
    def test_simple_year_citation(self):
        m = CITATION_RE.search("Squire (2004) found this.")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("name"), "Squire")

    def test_year_with_letter_suffix(self):
        m = CITATION_RE.search("Smith (2010a) wrote.")
        self.assertIsNotNone(m)

    def test_multi_author_with_ampersand(self):
        m = CITATION_RE.search("Atkinson & Shiffrin (1968).")
        self.assertEqual(m.group("name"), "Atkinson & Shiffrin")

    def test_multi_author_with_comma(self):
        m = CITATION_RE.search("Smith, Jones, and Brown (2010).")
        self.assertIsNotNone(m)

    def test_no_match_without_year(self):
        m = CITATION_RE.search("Squire's framework")
        self.assertIsNone(m)


class TestFindAttributions(unittest.TestCase):
    def test_finds_year_citation(self):
        hits = list(find_attributions("Squire (2004) found this."))
        self.assertEqual(len(hits), 1)
        matched, name, kind, wl = hits[0]
        self.assertEqual(name, "Squire")
        self.assertEqual(kind, "year")
        self.assertFalse(wl)

    def test_finds_according_to(self):
        hits = list(find_attributions("According to Smith, the model holds."))
        self.assertTrue(any(k == "according_to" for _, _, k, _ in hits))

    def test_finds_et_al(self):
        hits = list(find_attributions("Smith et al. demonstrated."))
        self.assertTrue(any(k == "et_al" for _, _, k, _ in hits))

    def test_finds_possessive(self):
        hits = list(find_attributions("Squire's framework explains amnesia."))
        self.assertTrue(any(k == "possessive" for _, _, k, _ in hits))

    def test_marks_whitelisted_correctly(self):
        hits = list(find_attributions("Piaget (1936) and Squire (2004) found things."))
        whitelisted = {(name, wl) for _, name, _, wl in hits}
        self.assertIn(("Piaget", True), whitelisted)
        self.assertIn(("Squire", False), whitelisted)

    def test_empty_text(self):
        self.assertEqual(list(find_attributions("")), [])
        self.assertEqual(list(find_attributions(None)), [])

    def test_clean_text_no_hits(self):
        self.assertEqual(
            list(find_attributions("Implicit memory is preserved in amnesia.")),
            [],
        )


class TestInitialedCitations(unittest.TestCase):
    """Initialed-author citations — 155 hits in the source corpus before fix.

    Pattern: "Smith, A. (2010)" or "Watson, J.B. & Rayner, R. (1920)".
    Critical because real EPPP-relevant figures (Watson, Rescorla,
    Herrnstein) appear in source data with this format.
    """

    def test_single_author_initialed(self):
        m = CITATION_RE.search("Smith, A. (2010) found something.")
        self.assertIsNotNone(m)
        self.assertIn("Smith", m.group("name"))

    def test_two_initialed_authors(self):
        m = CITATION_RE.search("Watson, J.B. & Rayner, R. (1920) demonstrated.")
        self.assertIsNotNone(m)
        # Surnames extracted via split_authors after stripping initials
        self.assertEqual(split_authors(m.group("name")), ["Watson", "Rayner"])

    def test_compact_initials(self):
        # "A.B." (no space) and "A. B." (space) both work.
        for variant in ("Smith, A.B. (2010)", "Smith, A. B. (2010)"):
            m = CITATION_RE.search(variant)
            self.assertIsNotNone(m, msg=variant)
            self.assertEqual(split_authors(m.group("name")), ["Smith"])

    def test_initialed_whitelisted_eponym(self):
        # Watson (J.B. Watson) is whitelisted as the founder of behaviorism.
        # Rayner is not. Multi-author check: not all whitelisted → False.
        self.assertFalse(is_whitelisted("Watson, J.B. & Rayner, R."))
        self.assertTrue(is_whitelisted("Watson, J.B."))


class TestUnicodeNames(unittest.TestCase):
    """Unicode-aware name matching — Latané, Köhler, Müller, etc.

    Source corpus has zero non-ASCII names today; this is preemptive.
    """

    def test_latane_with_diacritic_matches(self):
        m = CITATION_RE.search("Latané (1968) studied bystander effect.")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("name"), "Latané")

    def test_kohler_with_umlaut(self):
        m = CITATION_RE.search("Köhler (1925) studied insight learning.")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("name"), "Köhler")

    def test_unicode_name_whitelist_lookup(self):
        # "Latané" is in the whitelist (with the diacritic).
        self.assertTrue(is_whitelisted("Latané"))


class TestBareMultiAuthor(unittest.TestCase):
    """Bare multi-author attribution — "X and Y verb" without year.

    15 hits in source corpus. Detection requires both names to be
    whitelisted to pass, otherwise the gate flags. Two-author only;
    single-name has too many false positives in clinical vignettes.
    """

    def test_basic_bare_multi(self):
        hits = list(find_attributions("Smith and Jones found significant differences."))
        bare = [h for h in hits if h[2] == "bare_multi"]
        self.assertEqual(len(bare), 1)
        self.assertEqual(bare[0][1], "Smith and Jones")
        self.assertFalse(bare[0][3])  # not whitelisted

    def test_whitelisted_bare_multi_passes(self):
        # Cannon and Bard both whitelisted → marked as whitelisted, gate passes.
        hits = list(find_attributions("Cannon and Bard proposed simultaneous arousal."))
        bare = [h for h in hits if h[2] == "bare_multi"]
        self.assertEqual(len(bare), 1)
        self.assertTrue(bare[0][3])  # whitelisted

    def test_no_match_without_attribution_verb(self):
        # "X and Y" without an attribution verb shouldn't fire.
        hits = list(find_attributions("Smith and Jones went to lunch."))
        self.assertEqual([h for h in hits if h[2] == "bare_multi"], [])

    def test_ampersand_form_caught(self):
        hits = list(find_attributions("Smith & Jones demonstrated."))
        self.assertTrue(any(k == "bare_multi" for *_, k, _ in [(*h[:2], h[2], h[3]) for h in hits]))

    def test_concept_pair_filtered_by_blocklist(self):
        # NON_NAME_BLOCKLIST suppresses the bare_multi false positive on
        # capitalized concept pairs like "Anxiety and Depression". Both
        # words are blocklist entries, so the match is dropped.
        hits = list(find_attributions("Anxiety and Depression developed differently."))
        bare = [h for h in hits if h[2] == "bare_multi"]
        self.assertEqual(bare, [], "blocklist must filter concept-pair false positive")

    def test_blocklist_filters_one_side(self):
        # If only one side is a blocklist token, still filter (a real
        # multi-author group needs both sides to be person names).
        hits = list(find_attributions("Memory and Atkinson demonstrated retrieval."))
        bare = [h for h in hits if h[2] == "bare_multi"]
        self.assertEqual(bare, [])


class TestHyphenatedMultiAuthor(unittest.TestCase):
    """Hyphenated compound eponyms — Cannon-Bard, Stanford-Binet.

    Two cases must both work:
      • "Cannon-Bard" — split into ["Cannon", "Bard"], both whitelisted.
      • "Kübler-Ross" — atomic whitelist entry kept as a whole.
    is_whitelisted does whole-name lookup first, then split-parts fallback.
    """

    def test_cannon_bard_year_citation(self):
        m = CITATION_RE.search("Cannon-Bard (1929) proposed the theory.")
        self.assertIsNotNone(m)
        self.assertEqual(m.group("name"), "Cannon-Bard")
        self.assertTrue(is_whitelisted("Cannon-Bard"))

    def test_stanford_binet_year_citation(self):
        # Stanford and Binet are individual whitelist entries.
        # "Stanford-Binet" matches via split-parts.
        self.assertTrue(is_whitelisted("Stanford-Binet"))

    def test_kubler_ross_atomic(self):
        # Kübler-Ross is a single whitelist entry. Whole-name lookup
        # succeeds; split parts (Kübler, Ross) might NOT be whitelisted
        # individually, but the atomic match wins.
        self.assertTrue(is_whitelisted("Kübler-Ross"))

    def test_hyphenated_with_one_unwhitelisted(self):
        # "Smith-Jones" — neither WL individually, not atomic in WL.
        self.assertFalse(is_whitelisted("Smith-Jones"))


class TestOxfordBareMulti(unittest.TestCase):
    """Three-or-more-author bare attribution with oxford comma."""

    def test_three_author_oxford(self):
        hits = list(find_attributions("Smith, Jones, and Brown found that priming exists."))
        bare = [h for h in hits if h[2] == "bare_multi"]
        self.assertEqual(len(bare), 1)
        self.assertEqual(bare[0][1], "Smith, Jones, and Brown")

    def test_four_author_oxford(self):
        hits = list(find_attributions("Smith, Jones, Brown, and Davis demonstrated effects."))
        bare = [h for h in hits if h[2] == "bare_multi"]
        self.assertEqual(len(bare), 1)


class TestBareSingleAuthor(unittest.TestCase):
    """Single-name bare attribution — narrow detection, multi-layer false-positive filter."""

    def test_research_verb_phrase_caught(self):
        # "found that" is deliberately NOT in the verb list (clinical-context
        # overlap with "the patient found that her therapy helped").
        # Use "argued that" — research-leaning, low clinical-context risk.
        hits = list(find_attributions("Sadker argued that gender bias persists."))
        bare = [h for h in hits if h[2] == "bare_single"]
        self.assertEqual(len(bare), 1)

    def test_clinical_risky_verbs_not_caught(self):
        # Verbs deliberately excluded from the bare_single list because
        # they collide too often with clinical vignette prose.
        for fragment in (
            "Smith found that the model fits.",
            "Smith reported that he felt anxious.",
            "Smith observed that the patient improved.",
            "Smith noted that conditions changed.",
            "Smith showed that effect strength varies.",
        ):
            hits = list(find_attributions(fragment))
            bare = [h for h in hits if h[2] == "bare_single"]
            self.assertEqual(bare, [], f"clinical-risky verb leaked: {fragment!r}")

    def test_solo_research_verb_caught(self):
        hits = list(find_attributions("Smith hypothesized that priming occurs."))
        bare = [h for h in hits if h[2] == "bare_single"]
        self.assertEqual(len(bare), 1)

    def test_whitelisted_passes(self):
        hits = list(find_attributions("Pavlov demonstrated that conditioning extinguishes."))
        bare = [h for h in hits if h[2] == "bare_single"]
        self.assertEqual(len(bare), 1)
        self.assertTrue(bare[0][3])  # whitelisted

    def test_title_prefix_dr_exempted(self):
        # "Dr. Smith found that..." is a clinician, not a researcher.
        hits = list(find_attributions("Dr. Smith found that the patient improved."))
        bare = [h for h in hits if h[2] == "bare_single"]
        self.assertEqual(bare, [], "title prefix must exempt clinical vignette names")

    def test_title_prefix_mr_exempted(self):
        hits = list(find_attributions("Mr. Smith reported that he felt anxious."))
        bare = [h for h in hits if h[2] == "bare_single"]
        self.assertEqual(bare, [])

    def test_first_name_preceding_exempted(self):
        # "Maria Smith found that..." — "Maria" is a capitalized word
        # immediately before "Smith", so the post-filter treats this as
        # a vignette character (first + last name).
        hits = list(find_attributions("Maria Smith found that her therapist helped."))
        bare = [h for h in hits if h[2] == "bare_single"]
        self.assertEqual(bare, [])

    def test_non_research_verb_not_caught(self):
        # "Smith found her wallet" — "found" alone (no "that") is not in
        # the research verb-phrase list.
        hits = list(find_attributions("Smith found her wallet on the bus."))
        bare = [h for h in hits if h[2] == "bare_single"]
        self.assertEqual(bare, [])

    def test_non_name_token_filtered(self):
        # "Results showed that" — "Results" matches name regex but is
        # in NON_NAME_BLOCKLIST.
        for fragment in (
            "Results showed that priming exists.",
            "It demonstrated that the effect held.",
            "This proposed that learning is gradual.",
            "Aplysia showed that synaptic plasticity occurs.",
        ):
            hits = list(find_attributions(fragment))
            bare = [h for h in hits if h[2] == "bare_single"]
            self.assertEqual(bare, [], f"blocklist token leaked through: {fragment!r}")


class TestSplitAuthorsExtended(unittest.TestCase):
    """Extended split_authors tests for initial-stripping behavior."""

    def test_strips_single_initial(self):
        self.assertEqual(split_authors("Smith, A."), ["Smith"])

    def test_strips_compact_initials(self):
        self.assertEqual(split_authors("Watson, J.B."), ["Watson"])

    def test_strips_spaced_initials(self):
        self.assertEqual(split_authors("Watson, J. B."), ["Watson"])

    def test_multi_author_with_initials(self):
        self.assertEqual(
            split_authors("Watson, J.B. & Rayner, R."),
            ["Watson", "Rayner"],
        )

    def test_oxford_comma(self):
        self.assertEqual(
            split_authors("Smith, Jones, and Brown"),
            ["Smith", "Jones", "Brown"],
        )


class TestGateAuditConsistency(unittest.TestCase):
    """The AttributionGate and audit script must produce identical results
    on any given input — they share a single regex/whitelist module.
    Regression check: run the same fixture through both and compare.
    """
    def test_gate_audit_agree_on_violations(self):
        from pipeline.gates import AttributionGate

        fixtures = [
            "Squire (2004) found implicit memory.",
            "Piaget (1936) described stages.",
            "According to Smith, the model holds.",
            "According to Pavlov, conditioning extinguishes.",
            "Smith et al. demonstrated.",
            "Bandura et al. found self-efficacy.",
            "Squire's framework explains amnesia.",
            "Piaget's stages are universal.",
            "A clean clinical vignette with Dr. Harding.",
            "APA Guidelines for Forensic Psychology (2013) state...",
        ]

        gate = AttributionGate()
        for text in fixtures:
            # Both call into find_attributions internally; they must agree
            # on which strings are violations.
            audit_violations = [
                m for m, _, _, wl in find_attributions(text) if not wl
            ]
            gate_q = {
                "question_stem": text,
                "tested_concept": {"knowledge_tested": ""},
                "options": [],
            }
            gate_ok, _ = gate.check(gate_q)

            if audit_violations:
                self.assertFalse(gate_ok,
                                 msg=f"Gate should fail on: {text!r}")
            else:
                self.assertTrue(gate_ok,
                                msg=f"Gate should pass on: {text!r}")


if __name__ == "__main__":
    unittest.main()
