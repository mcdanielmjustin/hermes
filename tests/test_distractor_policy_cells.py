"""Phase A2.5 — distractor_policy cell-matrix tests.

Verifies:
  - Cell shape accepts override_thresholds, co_firing_required,
    classification_prior fields with defaults.
  - DEFAULT_T1_T2 / DEFAULT_T3 / DEFAULT_T4 have the expected thresholds.
  - resolve() falls back to tier-keyed defaults when no domain-specific
    cell matches.
  - Domain-specific cells (BPSY, PETH) still resolve correctly under the
    extended Cell shape.
  - threshold_for() returns the correct float or None.
"""
from __future__ import annotations

import unittest

import conftest  # noqa: F401  — sets sys.path

from pipeline.distractor_policy import (
    Cell,
    DEFAULT,
    DEFAULT_T1_T2,
    DEFAULT_T3,
    DEFAULT_T4,
    POLICY_BPSY,
    POLICY_PETH,
    resolve,
)


class TestCellShape(unittest.TestCase):
    """Cell accepts new fields with defaults; validates inputs."""

    def test_cell_accepts_default_no_overrides(self):
        c = Cell(
            gate_action="strict",
            correction_strategy="swap_for_content_distractor",
        )
        self.assertEqual(c.override_thresholds, ())
        self.assertFalse(c.co_firing_required)
        self.assertIsNone(c.classification_prior)
        self.assertIsNone(c.threshold_for("english_gap_scanner"))

    def test_cell_accepts_override_thresholds_tuple(self):
        c = Cell(
            gate_action="strict",
            correction_strategy="swap_for_content_distractor",
            override_thresholds=(("english_gap_scanner", 0.85),),
        )
        self.assertAlmostEqual(c.threshold_for("english_gap_scanner"), 0.85)
        self.assertIsNone(c.threshold_for("schema_labeling"))

    def test_cell_rejects_malformed_threshold_entry(self):
        with self.assertRaises(ValueError):
            Cell(
                gate_action="strict",
                correction_strategy="swap_for_content_distractor",
                override_thresholds=(("english_gap_scanner",),),  # missing threshold
            )

    def test_cell_rejects_threshold_out_of_range(self):
        with self.assertRaises(ValueError):
            Cell(
                gate_action="strict",
                correction_strategy="swap_for_content_distractor",
                override_thresholds=(("english_gap_scanner", 1.5),),
            )

    def test_cell_rejects_invalid_classification_prior(self):
        with self.assertRaises(ValueError):
            Cell(
                gate_action="strict",
                correction_strategy="swap_for_content_distractor",
                classification_prior="bogus",
            )


class TestTierKeyedDefaults(unittest.TestCase):
    """DEFAULT_TX cells have the expected thresholds matching the plan."""

    def test_t1_t2_threshold_is_0_75(self):
        # T1/T2: any fired override-eligible signature wins
        # (UQ 0.85, laterality 0.75, num_ratio 0.80 all >= 0.75).
        self.assertAlmostEqual(
            DEFAULT_T1_T2.threshold_for("english_gap_scanner"), 0.75, places=3
        )

    def test_t3_threshold_is_0_85(self):
        # T3: only universal_quantifier (0.85) reaches the threshold.
        self.assertAlmostEqual(
            DEFAULT_T3.threshold_for("english_gap_scanner"), 0.85, places=3
        )

    def test_t4_threshold_is_0_95(self):
        # T4: no single signature reaches; effectively advisory.
        self.assertAlmostEqual(
            DEFAULT_T4.threshold_for("english_gap_scanner"), 0.95, places=3
        )

    def test_t4_has_classification_prior_content_gap(self):
        self.assertEqual(DEFAULT_T4.classification_prior, "content_gap")

    def test_t4_co_firing_required(self):
        self.assertTrue(DEFAULT_T4.co_firing_required)


class TestResolveTierFallback(unittest.TestCase):
    """resolve() falls back to tier-keyed default when no domain-specific
    cell matches. This is the new A2.5 behavior."""

    def test_unknown_domain_t1_falls_to_t1_t2_default(self):
        cell = resolve(tier=1)
        self.assertEqual(cell.note, DEFAULT_T1_T2.note)

    def test_unknown_domain_t2_falls_to_t1_t2_default(self):
        cell = resolve(tier=2)
        self.assertEqual(cell.note, DEFAULT_T1_T2.note)

    def test_unknown_domain_t3_falls_to_t3_default(self):
        cell = resolve(tier=3)
        self.assertEqual(cell.note, DEFAULT_T3.note)

    def test_unknown_domain_t4_falls_to_t4_default(self):
        cell = resolve(tier=4)
        self.assertEqual(cell.note, DEFAULT_T4.note)

    def test_unknown_tier_falls_to_bare_default(self):
        cell = resolve(tier=None)
        self.assertEqual(cell.note, DEFAULT.note)

    def test_known_domain_no_matching_cell_falls_to_tier_default(self):
        # BPSY at (1, "anchor_grounded", "novel_pattern_not_in_matrix")
        # — should fall through to DEFAULT_T1_T2.
        cell = resolve(
            tier=1, domain_code="BPSY",
            source_type="anchor_grounded",
            stem_pattern="novel_pattern_not_in_matrix",
        )
        self.assertEqual(cell.note, DEFAULT_T1_T2.note)


class TestDomainSpecificCellsStillResolve(unittest.TestCase):
    """The existing BPSY and PETH cells must still resolve correctly
    under the extended Cell shape (regression check). After S2,
    domain cells inherit the tier-default's detector fields when they
    use defaults — so domain cells now expose tier-default thresholds."""

    def test_bpsy_t4_integrated_contrast_prompt_resolves(self):
        cell = resolve(
            tier=4, domain_code="BPSY",
            source_type="integrated", stem_pattern="contrast_prompt",
        )
        # Existing cell — gate_action='audit', correction='judgment_error'.
        # Those win on the domain side.
        self.assertEqual(cell.gate_action, "audit")
        self.assertEqual(cell.correction_strategy, "judgment_error")
        # S2: domain cell uses default for override_thresholds → inherits
        # DEFAULT_T4's threshold of 0.95 (and co_firing + content_gap prior).
        self.assertAlmostEqual(
            cell.threshold_for("english_gap_scanner"), 0.95, places=3,
            msg="S2: BPSY T4 domain cell must inherit DEFAULT_T4's threshold"
        )
        self.assertTrue(cell.co_firing_required,
                        "S2: BPSY T4 domain cell inherits T4 co_firing_required")
        self.assertEqual(cell.classification_prior, "content_gap",
                         "S2: BPSY T4 domain cell inherits T4 content_gap prior")

    def test_peth_t3_integrated_clinical_vignette_resolves(self):
        cell = resolve(
            tier=3, domain_code="PETH",
            source_type="integrated", stem_pattern="clinical_vignette",
        )
        self.assertEqual(cell.gate_action, "framework_aware")
        self.assertEqual(cell.correction_strategy, "framework_misapplication")
        # S2: PETH T3 domain cell inherits DEFAULT_T3's threshold (0.85).
        self.assertAlmostEqual(
            cell.threshold_for("english_gap_scanner"), 0.85, places=3,
            msg="S2: PETH T3 domain cell must inherit DEFAULT_T3's threshold"
        )

    def test_bpsy_t2_anchor_grounded_paraphrase_inherits_t1_t2_threshold(self):
        cell = resolve(
            tier=2, domain_code="BPSY",
            source_type="anchor_grounded", stem_pattern="paraphrase",
        )
        self.assertEqual(cell.gate_action, "audit")  # domain cell wins
        # S2: domain cell uses default for override_thresholds →
        # inherits DEFAULT_T1_T2's 0.75.
        self.assertAlmostEqual(
            cell.threshold_for("english_gap_scanner"), 0.75, places=3,
        )


class TestS2Composition(unittest.TestCase):
    """S2 explicit composition tests — verify domain cells that DO set
    A2.5 fields keep their own values, while ones that don't inherit."""

    def test_synthetic_domain_cell_with_explicit_threshold_keeps_own(self):
        """A domain cell that explicitly sets override_thresholds should
        keep its own values (tier-default does NOT override)."""
        from pipeline.distractor_policy import _compose_with_tier_default
        domain_cell = Cell(
            gate_action="audit",
            correction_strategy="judgment_error",
            override_thresholds=(("english_gap_scanner", 0.99),),
        )
        composed = _compose_with_tier_default(domain_cell, tier=1)
        self.assertAlmostEqual(
            composed.threshold_for("english_gap_scanner"), 0.99, places=3,
            msg="explicit domain threshold must NOT be overridden by tier-default"
        )

    def test_synthetic_domain_cell_without_threshold_inherits(self):
        from pipeline.distractor_policy import _compose_with_tier_default
        domain_cell = Cell(
            gate_action="audit",
            correction_strategy="judgment_error",
            # override_thresholds defaulted to ()
        )
        composed = _compose_with_tier_default(domain_cell, tier=4)
        self.assertAlmostEqual(
            composed.threshold_for("english_gap_scanner"), 0.95, places=3,
            msg="empty domain threshold must inherit DEFAULT_T4 (0.95)"
        )

    def test_co_firing_propagates_either_direction(self):
        """If the tier-default OR the domain cell requires co-firing,
        the composed cell requires co-firing (more conservative wins)."""
        from pipeline.distractor_policy import _compose_with_tier_default
        # Domain cell doesn't require co-firing; tier-default does (T4):
        domain_cell = Cell(
            gate_action="audit", correction_strategy="judgment_error",
        )
        composed_t4 = _compose_with_tier_default(domain_cell, tier=4)
        self.assertTrue(composed_t4.co_firing_required,
                        "T4 default requires co-firing; composed must too")

        # Domain cell requires; tier-default (T1) doesn't:
        domain_cell_with_cofire = Cell(
            gate_action="audit", correction_strategy="judgment_error",
            co_firing_required=True,
        )
        composed_t1 = _compose_with_tier_default(domain_cell_with_cofire, tier=1)
        self.assertTrue(composed_t1.co_firing_required,
                        "Domain cell requires co-firing; composed must too")

    def test_classification_prior_domain_wins(self):
        """If the domain cell sets classification_prior, that wins.
        If domain cell uses None, tier-default fills in."""
        from pipeline.distractor_policy import _compose_with_tier_default
        # Domain cell sets english_gap; tier-default (T4) sets content_gap:
        domain_cell = Cell(
            gate_action="audit", correction_strategy="judgment_error",
            classification_prior="english_gap",
        )
        composed = _compose_with_tier_default(domain_cell, tier=4)
        self.assertEqual(composed.classification_prior, "english_gap",
                         "domain-specific prior must win")

        # Domain cell None; tier-default (T4) sets content_gap:
        domain_cell_unset = Cell(
            gate_action="audit", correction_strategy="judgment_error",
        )
        composed = _compose_with_tier_default(domain_cell_unset, tier=4)
        self.assertEqual(composed.classification_prior, "content_gap",
                         "domain cell with None prior must inherit tier-default")


class TestThresholdForSemantics(unittest.TestCase):
    """threshold_for() returns float or None correctly."""

    def test_threshold_for_unknown_detector_is_none(self):
        cell = DEFAULT_T1_T2
        self.assertIsNone(cell.threshold_for("nonexistent_detector"))

    def test_threshold_for_returns_float(self):
        cell = DEFAULT_T3
        result = cell.threshold_for("english_gap_scanner")
        self.assertIsInstance(result, float)
        self.assertEqual(result, 0.85)


if __name__ == "__main__":
    unittest.main()
