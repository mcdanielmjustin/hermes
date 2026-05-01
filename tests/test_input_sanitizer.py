"""Tests for InputSanitizerAgent.

The sanitizer strips researcher citations from anchor inputs before they
reach the LLM. It must:
  - Remove non-whitelisted author/year citations entirely (incl. dangling colons)
  - Preserve whitelisted eponyms with the year stripped
  - Handle multi-author citations (X & Y, X and Y, X, Y, and Z)
  - Handle "et al." patterns
  - Be idempotent
  - NOT split whitelisted names that contain "and" as a substring (Bandura regression)
"""
import unittest

import conftest  # noqa: F401  — sets sys.path

from pipeline.agents import InputSanitizerAgent


class TestSanitizerCitationStripping(unittest.TestCase):
    def test_strips_leading_non_whitelisted_citation(self):
        result = InputSanitizerAgent.sanitize(
            "Squire (2004): Nondeclarative memory is preserved in amnesia."
        )
        self.assertEqual(result, "Nondeclarative memory is preserved in amnesia.")

    def test_keeps_whitelisted_eponym_drops_year(self):
        result = InputSanitizerAgent.sanitize(
            "Piaget (1936): Children develop sensorimotor stages."
        )
        self.assertEqual(result, "Piaget: Children develop sensorimotor stages.")

    def test_multi_author_both_whitelisted(self):
        result = InputSanitizerAgent.sanitize(
            "Atkinson & Shiffrin (1968): Multi-store model."
        )
        self.assertEqual(result, "Atkinson & Shiffrin: Multi-store model.")

    def test_multi_author_neither_whitelisted_strips_all(self):
        result = InputSanitizerAgent.sanitize(
            "Smith and Jones (1985) showed working memory limits."
        )
        self.assertEqual(result, "showed working memory limits.")

    def test_et_al_strip(self):
        result = InputSanitizerAgent.sanitize(
            "Smith et al. (2019) found something."
        )
        self.assertEqual(result, "found something.")

    def test_no_year_no_change(self):
        text = "Cannon-Bard theory states emotion and arousal occur simultaneously."
        self.assertEqual(InputSanitizerAgent.sanitize(text), text)

    def test_clean_text_unchanged(self):
        text = "Implicit memory is preserved in amnesic patients with hippocampal damage."
        self.assertEqual(InputSanitizerAgent.sanitize(text), text)

    def test_eponym_condition_unchanged(self):
        # "Huntington's disease" — eponymous condition, no citation.
        text = "Huntington's disease impairs procedural learning."
        self.assertEqual(InputSanitizerAgent.sanitize(text), text)

    def test_idempotent(self):
        once = InputSanitizerAgent.sanitize("Squire (2004): Memory.")
        twice = InputSanitizerAgent.sanitize(once)
        self.assertEqual(once, twice)


class TestSanitizerEdgeCases(unittest.TestCase):
    def test_empty_string(self):
        self.assertEqual(InputSanitizerAgent.sanitize(""), "")

    def test_none_returns_none(self):
        self.assertIsNone(InputSanitizerAgent.sanitize(None))

    def test_non_string_passthrough(self):
        self.assertEqual(InputSanitizerAgent.sanitize(42), 42)

    def test_collapses_double_spaces(self):
        result = InputSanitizerAgent.sanitize("Squire  (2004):  Memory.")
        self.assertEqual(result, "Memory.")

    def test_year_with_letter_suffix(self):
        # Citations like "Smith (2010a)" are common in psych literature.
        result = InputSanitizerAgent.sanitize("Smith (2010a) found...")
        self.assertEqual(result, "found...")

    def test_preserves_newlines(self):
        # _DOUBLE_SPACE_RE collapses spaces/tabs only, not newlines.
        # Multi-line content (currently not used in briefs) must survive
        # sanitization without paragraph collapse.
        text = "First claim.\nSecond claim with Squire (2004) cited.\nThird claim."
        result = InputSanitizerAgent.sanitize(text)
        # Newlines survive; only the citation got stripped from line 2.
        self.assertEqual(
            result,
            "First claim.\nSecond claim with cited.\nThird claim.",
        )

    def test_collapses_only_horizontal_whitespace(self):
        result = InputSanitizerAgent.sanitize("Word1   Word2\tWord3")
        # Multiple spaces → single space; tab preserved as single whitespace via collapse.
        # The actual behavior: \t collapses to a single space when adjacent
        # to other whitespace via [ \t]{2,}; alone it stays as tab.
        self.assertNotIn("   ", result)  # No triple spaces remain


class TestBanduraRegression(unittest.TestCase):
    """Names containing 'and' as a substring must NOT be split.

    Earlier bug: re.split(r'\\s*(?:&|and|,)\\s*', 'Bandura') returned ['B', 'ura']
    because the alternation matched 'and' inside 'Bandura'. Fix uses \\band\\b
    (word boundaries) so 'and' only matches as a whole word.

    Bandura is the only currently-whitelisted name with this property, but
    the bug would affect any future addition (e.g., a hypothetical "Vandura").
    """

    def test_bandura_with_year_kept(self):
        result = InputSanitizerAgent.sanitize(
            "Bandura (1977): Self-efficacy beliefs predict behavior change."
        )
        self.assertEqual(result, "Bandura: Self-efficacy beliefs predict behavior change.")

    def test_bandura_alone_unchanged(self):
        text = "Bandura proposed reciprocal determinism."
        self.assertEqual(InputSanitizerAgent.sanitize(text), text)

    def test_multi_author_with_bandura_partial_whitelist(self):
        # Best-of-breed policy: Bandura is whitelisted, Walters is not, so
        # only Bandura survives. Year is dropped along with the non-WL author.
        # Cleaner than the old all-or-nothing strip ('studied imitation.'
        # with broken grammar).
        result = InputSanitizerAgent.sanitize(
            "Bandura and Walters (1963) studied imitation."
        )
        self.assertEqual(result, "Bandura studied imitation.")


class TestBestOfBreedSanitizer(unittest.TestCase):
    """Mixed-whitelist multi-author citations keep whitelisted authors only.

    Old behavior was all-or-nothing: any non-whitelisted co-author → strip
    everything. New behavior salvages whitelisted authors, producing
    cleaner downstream prompt text.
    """

    def test_mixed_keeps_only_whitelisted(self):
        # Bandura whitelisted, Walters not → keep "Bandura", drop "Walters" + year.
        result = InputSanitizerAgent.sanitize(
            "Bandura and Walters (1963) studied imitation."
        )
        self.assertEqual(result, "Bandura studied imitation.")

    def test_initialed_authors_with_one_whitelisted(self):
        # Watson whitelisted, Rayner not. Initials get stripped during
        # whitelist check; reconstruction uses bare surname.
        result = InputSanitizerAgent.sanitize(
            "Watson, J.B. & Rayner, R. (1920) demonstrated conditioning."
        )
        self.assertEqual(result, "Watson demonstrated conditioning.")

    def test_all_whitelisted_preserves_original_form(self):
        # Both whitelisted → preserve original "Cannon and Bard" formatting.
        result = InputSanitizerAgent.sanitize(
            "Cannon and Bard (1929) proposed the theory."
        )
        self.assertEqual(result, "Cannon and Bard proposed the theory.")

    def test_none_whitelisted_strips_all(self):
        result = InputSanitizerAgent.sanitize(
            "Smith and Jones (1985) showed limits."
        )
        self.assertEqual(result, "showed limits.")

    def test_three_authors_mixed(self):
        # Smith not WL, Jones not WL, Bandura WL → keep only Bandura.
        result = InputSanitizerAgent.sanitize(
            "Smith, Jones, and Bandura (2010) found something."
        )
        self.assertEqual(result, "Bandura found something.")

    def test_real_corpus_example_watson_rayner(self):
        # Verbatim citation from anchor_points.csv, demonstrating that
        # the initialed + best-of-breed combination resolves a real
        # corpus pattern (155 such citations existed before this fix).
        result = InputSanitizerAgent.sanitize(
            "Watson, J.B. & Rayner, R. (1920): Conditioned fear in Little Albert."
        )
        self.assertEqual(result, "Watson: Conditioned fear in Little Albert.")


class TestSanitizerExecuteMethod(unittest.TestCase):
    def test_execute_returns_dict_with_three_fields(self):
        agent = InputSanitizerAgent()
        out = agent.execute({
            "verbatim_anchor": "Squire (2004): Memory dissociations.",
            "testable_fact": "Implicit memory is preserved.",
            "core_claims": [
                "Squire (2004) showed implicit memory is preserved.",
                "Procedural learning depends on basal ganglia.",
            ],
        })
        self.assertEqual(out["verbatim_anchor"], "Memory dissociations.")
        self.assertEqual(out["testable_fact"], "Implicit memory is preserved.")
        self.assertEqual(out["core_claims"][0], "showed implicit memory is preserved.")
        self.assertEqual(out["core_claims"][1], "Procedural learning depends on basal ganglia.")

    def test_execute_handles_missing_fields(self):
        agent = InputSanitizerAgent()
        out = agent.execute({})
        self.assertEqual(out["verbatim_anchor"], "")
        self.assertEqual(out["testable_fact"], "")
        self.assertEqual(out["core_claims"], [])

    def test_execute_handles_none_core_claims(self):
        agent = InputSanitizerAgent()
        out = agent.execute({"core_claims": None})
        self.assertEqual(out["core_claims"], [])


if __name__ == "__main__":
    unittest.main()
