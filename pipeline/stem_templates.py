"""Stem template library for stem-rewrite rescues.

When a question's stem over-specifies in a way that traps any plausible
distractor as english_gap (e.g., BPSY MX-7 prints "no measurable change
in postsynaptic firing" — any distractor about D2 antagonism + post-
synaptic vocabulary lexically conflicts), the right intervention is to
rewrite the STEM, not the distractors. The new stem must:

1. Preserve the testable concept (what the student must understand).
2. Stay at the same Bloom's tier (T1 Remember / T2 Understand / T3 Apply
   / T4 Evaluate).
3. Remove the over-specification — i.e., DON'T print the discriminator
   variable's value (the locus, the direction, the specific finding,
   the explicit count, the explicit framing).

A StemTemplate encodes the constraints for a specific (domain, flavor,
tier, discriminator) cell. Templates are SKELETONS — Phase 20d's
stem_rewrite strategy fills the slots from the anchor's brief
(concept_explanation + discriminators) and the original question's
correct option.

Templates grow organically: start with cells we have observed failures
on; add more as patterns recur.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class StemTemplate:
    """A stem-rewrite skeleton for a specific (domain, flavor, tier,
    discriminator) cell.

    Used by scripts/rescue_failed_questions.py:rescue_stem_rewrite to
    rewrite a stem that triggers english_gap by over-specifying. The
    rewrite preserves the testable concept and Bloom's tier; it removes
    the variable the student is supposed to resolve.
    """
    domain_code: str           # e.g., "BPSY", "CASS", "PMET", "SOCU"
    flavor: str                # e.g., "mechanism", "test_psychometric"
    bloom_tier: int            # 1-4
    discriminator: str         # e.g., "locus_pre_vs_post"
    # The skeleton stem — describes the intervention/observation/setup
    # without printing the discriminator's value. Includes {slot}
    # placeholders the rewrite prompt fills from anchor brief +
    # original question.
    stem_skeleton: str
    # Constraints the rewrite prompt enforces. Each rule says what to
    # OMIT; "do not print X" rules guard against re-introducing the
    # over-specification.
    omit_rules: tuple[str, ...]
    # Shape for the correct answer. Used to validate the rewrite
    # preserves the answer's testable form (e.g., a T3 Apply correct
    # option must contain a mechanism marker via apply_identity gate).
    correct_answer_shape: str
    # Notes about typical misconception patterns this template traps.
    # The rescue prompt uses these to ensure distractors stay aligned
    # with the rewritten stem.
    expected_misconception_axes: tuple[str, ...] = field(default_factory=tuple)


# ── Initial template library ────────────────────────────────
# Phase 20d starter set. Covers the 4 chapters that landed in review
# under Phase 19's deterministic auditor, plus the universal pattern
# the schema-labeling exception (Phase 20a) leaves uncaught: stems
# that print specific findings and trap mechanism-flavored distractors.

# 1. BPSY mechanism T3 — the MX-7 / D2-antagonist pattern.
# Stem must describe a binding event without printing the receptor's
# functional outcome ("no measurable change in postsynaptic firing")
# that traps distractors about postsynaptic activity.
TEMPLATE_BPSY_MECHANISM_T3_LOCUS = StemTemplate(
    domain_code="BPSY",
    flavor="mechanism",
    bloom_tier=3,
    discriminator="locus_pre_vs_post",
    stem_skeleton=(
        "{scenario_lead}. {agent_description} A researcher administers "
        "{compound_name}, a {compound_class}, and records "
        "{observation_target} shortly after administration."
    ),
    omit_rules=(
        "do not print whether the compound has its own postsynaptic effect",
        "do not print whether activity increases or decreases",
        "do not print whether the locus is presynaptic or postsynaptic",
        "do not print the magnitude or direction of the receptor's response",
    ),
    correct_answer_shape="mechanism_specific_with_marker",
    expected_misconception_axes=(
        "compound has its own intrinsic activity (vs. zero intrinsic activity)",
        "presynaptic vs postsynaptic locus confusion",
        "agonist vs antagonist confusion",
    ),
)

# 2. CASS test_psychometric T4 — the MMPI-2 split-pattern case.
# Stem must describe a profile without printing values that universal
# claims can lexically contradict ("F normal, FB elevated" + distractor
# saying "throughout the entire protocol").
TEMPLATE_CASS_TEST_PSYCHOMETRIC_T4_PATTERN = StemTemplate(
    domain_code="CASS",
    flavor="test_psychometric",
    bloom_tier=4,
    discriminator="scale_pattern_interpretation",
    stem_skeleton=(
        "{client_lead}. The clinician evaluates {test_instrument} validity "
        "in the context of {clinical_context}. Asked to interpret what the "
        "configuration of validity scales suggests about response style, "
        "{interpretation_question}."
    ),
    omit_rules=(
        "do not print numeric scale values (e.g., 'F = 65', 'FB elevated')",
        "do not print whether scales are normal/elevated/depressed",
        "do not state whether the pattern is consistent or inconsistent",
        "do not name the specific clinical conclusion",
    ),
    correct_answer_shape="pattern_specific_with_inferential_marker",
    expected_misconception_axes=(
        "universal pattern claim (vs. localized split)",
        "wrong scale-pattern interpretation",
        "wrong validity-scale referent",
    ),
)

# 3. PMET statistical T2 — the IV/DV definitional pattern.
# Stem must describe the design without printing exact variable counts
# that distractor inversions can lexically contradict. The schema-
# labeling clause from Phase 20a makes count-inversion distractors
# classify as content_gap, but the stem-rewrite path is still useful
# as an alternate route when the schema clause doesn't catch a specific
# inversion shape (e.g., when the distractor introduces an off-topic
# concept like "statistical test family" that flips to topic_realm).
TEMPLATE_PMET_STATISTICAL_T2_DESIGN = StemTemplate(
    domain_code="PMET",
    flavor="statistical",
    bloom_tier=2,
    discriminator="design_role_attribution",
    stem_skeleton=(
        "A researcher conducts a study {scenario_setup}. {variable_summary} "
        "How should the design's variable structure be characterized in "
        "terms of independent and dependent variables?"
    ),
    omit_rules=(
        "do not print the specific count of factors (e.g., '3 factors')",
        "do not print the specific count of outcomes (e.g., '2 outcomes')",
        "do use generic terms like 'several' or 'multiple' if needed",
        "do not pre-label any variable as 'IV' or 'DV' in the stem",
    ),
    correct_answer_shape="iv_dv_role_definition_specific",
    expected_misconception_axes=(
        "swap which group is IV vs DV",
        "introduce off-topic statistical concept",
        "collapse the IV/DV distinction",
    ),
)

# 4. SOCU applied_cultural T3 — the internalized-racism framework
# pattern. Stem must describe behaviors without explicitly naming
# every observation as racially-tied (which makes "unrelated to race"
# distractors lexically rejectable).
TEMPLATE_SOCU_APPLIED_CULTURAL_T3_FRAMEWORK = StemTemplate(
    domain_code="SOCU",
    flavor="applied_cultural",
    bloom_tier=3,
    discriminator="framework_direction",
    stem_skeleton=(
        "{client_lead}. {client_name} describes {behavioral_pattern} and "
        "reports {affective_state} dating to adolescence. The clinician "
        "considers how to characterize the underlying psychological process."
    ),
    omit_rules=(
        "do not explicitly state every observation is tied to racial group membership",
        "do describe behaviors and affects in clinically neutral language",
        "do not pre-label the pattern (e.g., 'internalized racism', "
        "'racial identity', or 'self-schema')",
    ),
    correct_answer_shape="framework_application_with_directional_specificity",
    expected_misconception_axes=(
        "frame denial (claiming pattern is 'unrelated to' the implicit framework)",
        "wrong direction within framework (e.g., externalized vs internalized)",
        "wrong framework category (e.g., self-schema vs identity)",
    ),
)


# ── Alternate templates ────────────────────────────────────
# Each (domain, flavor, tier) cell has at least one ALTERNATE template
# probing the same concept from a different cognitive angle. If the
# canonical fails re-validation, rescue_chapter_stems.py falls through
# to the alternate. Per Phase 20d's three-pass strategy: try canonical,
# try up to 2 alternates, then fall back to fix_question.

# 5. BPSY mechanism T3 ALTERNATE — direction-of-change probe.
# Same concept (D2 antagonism's pharmacological consequence) but the
# discriminator is "↑ vs ↓" rather than "pre vs post". Used when the
# canonical's locus framing won't fit the source anchor's natural
# stem voice.
TEMPLATE_BPSY_MECHANISM_T3_DIRECTION = StemTemplate(
    domain_code="BPSY",
    flavor="mechanism",
    bloom_tier=3,
    discriminator="direction_of_change",
    stem_skeleton=(
        "{scenario_lead}. A clinician administers {compound_name} and "
        "monitors {downstream_marker} over the early hours after "
        "administration. The expected initial change in {downstream_marker} "
        "depends on {discriminator_question}."
    ),
    omit_rules=(
        "do not print whether the marker increases or decreases",
        "do not print the magnitude or sign of the predicted change",
        "do not name the underlying mechanism in the stem",
    ),
    correct_answer_shape="mechanism_specific_with_marker",
    expected_misconception_axes=(
        "wrong direction (increase vs decrease)",
        "no-change (treats antagonism as system shutdown)",
        "right direction wrong magnitude",
    ),
)

# 6. CASS test_psychometric T4 ALTERNATE — profile internal consistency.
# Probes the SAME pharmacology (validity-scale interpretation) but asks
# about consistency-scale signals (VRIN/TRIN) rather than F-FB
# split-pattern. Different aspect of test_psychometric knowledge.
TEMPLATE_CASS_TEST_PSYCHOMETRIC_T4_CONSISTENCY = StemTemplate(
    domain_code="CASS",
    flavor="test_psychometric",
    bloom_tier=4,
    discriminator="profile_internal_consistency",
    stem_skeleton=(
        "{client_lead}. The clinician reviews {test_instrument}'s "
        "consistency-scale signals (e.g., VRIN, TRIN) alongside the "
        "broader profile and considers what the response pattern implies "
        "about test-taking style. {interpretation_question}."
    ),
    omit_rules=(
        "do not print VRIN or TRIN values",
        "do not state whether the pattern is random or fixed responding",
        "do not name the specific response style in the stem",
    ),
    correct_answer_shape="response_style_with_test_specific_marker",
    expected_misconception_axes=(
        "conflate VRIN with TRIN (different inconsistency types)",
        "wrong response-style attribution",
        "wrong validity-scale referent",
    ),
)

# 7. PMET statistical T2 ALTERNATE — variable-type classification logic.
# Probes the SAME concept (IV vs DV) but the discriminator is HOW the
# variable was measured (continuous vs categorical) rather than its
# role in the design. Different angle on the same definitional schema.
TEMPLATE_PMET_STATISTICAL_T2_VARIABLE_TYPE = StemTemplate(
    domain_code="PMET",
    flavor="statistical",
    bloom_tier=2,
    discriminator="variable_classification_logic",
    stem_skeleton=(
        "{study_setup}. {investigator_question}. To select an appropriate "
        "analytic approach, what reasoning correctly classifies the "
        "variables in this design as independent vs dependent?"
    ),
    omit_rules=(
        "do not print which specific variable is the IV or DV",
        "do not pre-classify variables as continuous or categorical",
        "do not state the analytic test in the stem",
    ),
    correct_answer_shape="iv_dv_classification_principle",
    expected_misconception_axes=(
        "swap IV/DV definitions",
        "conflate variable role with measurement scale",
        "introduce unrelated classification dimension",
    ),
)

# 8. SOCU applied_cultural T3 ALTERNATE — intersectional emergence probe.
# Probes the SAME concept (which framework explains the pattern) but
# from the angle of WHICH framework component is the most predictive.
# Useful when canonical's framework_direction angle doesn't fit (e.g.,
# the concept isn't directional but multi-dimensional).
TEMPLATE_SOCU_APPLIED_CULTURAL_T3_INTERSECTIONAL = StemTemplate(
    domain_code="SOCU",
    flavor="applied_cultural",
    bloom_tier=3,
    discriminator="intersectional_emergence",
    stem_skeleton=(
        "{client_lead}. {client_name} reports {behavioral_pattern} that "
        "appears across {context_breadth}. The clinician considers which "
        "psychological framework component most directly accounts for "
        "the pattern's emergence and persistence."
    ),
    omit_rules=(
        "do not pre-name the framework in the stem",
        "do not explicitly attribute the behaviors to any single dimension",
        "do not describe the pattern as racial, gender, class, or "
        "another single-axis explanation",
    ),
    correct_answer_shape="framework_component_with_emergence_marker",
    expected_misconception_axes=(
        "single-axis attribution where multi-axis is correct",
        "naming the framework category but wrong component",
        "treating the pattern as decontextualized individual psychology",
    ),
)


# Registry indexed by (domain_code, flavor, bloom_tier, discriminator).
# Phase 20d expanded: each (domain, flavor, tier) cell has a CANONICAL
# template plus at least one ALTERNATE. rescue_chapter_stems.py walks
# the alternates if the canonical fails re-validation.
TEMPLATE_REGISTRY: dict[tuple[str, str, int, str], StemTemplate] = {
    (t.domain_code, t.flavor, t.bloom_tier, t.discriminator): t
    for t in (
        # Canonicals (one per cell — original Phase 20d set)
        TEMPLATE_BPSY_MECHANISM_T3_LOCUS,
        TEMPLATE_CASS_TEST_PSYCHOMETRIC_T4_PATTERN,
        TEMPLATE_PMET_STATISTICAL_T2_DESIGN,
        TEMPLATE_SOCU_APPLIED_CULTURAL_T3_FRAMEWORK,
        # Alternates (one per cell — Phase 20d expansion)
        TEMPLATE_BPSY_MECHANISM_T3_DIRECTION,
        TEMPLATE_CASS_TEST_PSYCHOMETRIC_T4_CONSISTENCY,
        TEMPLATE_PMET_STATISTICAL_T2_VARIABLE_TYPE,
        TEMPLATE_SOCU_APPLIED_CULTURAL_T3_INTERSECTIONAL,
    )
}


def get_template(domain_code: str, flavor: str, bloom_tier: int,
                 discriminator: str | None = None) -> StemTemplate | None:
    """Return the matching StemTemplate, or None if no template covers
    this cell. If `discriminator` is provided, requires exact match;
    if None, returns the first template at (domain, flavor, tier).
    """
    if discriminator:
        return TEMPLATE_REGISTRY.get((domain_code, flavor, bloom_tier, discriminator))
    # Fallback: any template at (domain, flavor, tier)
    for (d, f, t, _disc), tpl in TEMPLATE_REGISTRY.items():
        if d == domain_code and f == flavor and t == bloom_tier:
            return tpl
    return None


def get_alternate_templates(domain_code: str, flavor: str,
                            bloom_tier: int) -> list[StemTemplate]:
    """Return all templates at (domain_code, flavor, bloom_tier),
    sorted so the canonical (most-tested) is first. Used for the
    fallback chain when canonical fails validation.
    """
    return [
        tpl for (d, f, t, _disc), tpl in TEMPLATE_REGISTRY.items()
        if d == domain_code and f == flavor and t == bloom_tier
    ]


def list_covered_cells() -> list[tuple[str, str, int, str]]:
    """List all (domain, flavor, tier, discriminator) cells with a
    template registered. Used for diagnostics and for stem_rewrite to
    decide whether it has anything to try."""
    return list(TEMPLATE_REGISTRY.keys())
