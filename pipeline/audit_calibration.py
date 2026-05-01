"""Flavor-aware calibration text for the 3-class audit prompt.

The standalone audit (scripts/audit_stem_contradictions.py) and the
ship_readiness pipeline call audit_question with the question alone.
Without flavor context, the auditor's uniform "english_gap if rejectable
by reading alone" rule misclassifies cases where the rejection technically
depends on stage-name → age-range knowledge or framework-name → scope
knowledge — both of which are content knowledge in their domain, not
lexical pattern-matching.

This module exposes per-flavor exception clauses that get appended to
the audit prompt. Each clause names what kind of knowledge the flavor
relies on and instructs the auditor to grant content_gap for that
knowledge type, reserving english_gap for genuine lexical-only
contradictions independent of the named-entity knowledge.

The in-line StemEliminableDistractorGate has its own cell-aware
addendum logic (pipeline/gates.py:_build_mode_addendum). The two paths
overlap conceptually but operate on different inputs (cell vs flavor).
A future refactor could unify them through a single calibration
resolver that takes both. For now, this module covers the audit-only
path; the gate path stays as is.
"""
from __future__ import annotations

from pipeline.anchor_flavor import flavor_for_anchor


# Phase 20a: schema-labeling sub-rule. The auditor's uniform "english_gap
# = lexical rejection without concept knowledge" rule misclassifies
# distractors when the stem's plain-English vocabulary cues a definitional
# schema (e.g., "categorized" → IVs, "measured" → DVs) and the distractor
# swaps which named category attaches to which facet. Rejecting such a
# distractor genuinely requires schema knowledge, not lexical comparison.
#
# This sub-rule appends to the relevant flavor blocks below. CRITICAL
# precondition: the exception applies ONLY when no universal-quantifier
# contradiction exists. If the distractor uses "all", "every", "throughout",
# "entire", "any", "always", "never" against a stem-stated specific case,
# english_gap stands regardless of any schema-labeling pattern. This keeps
# the Lester/wedding canonical case classified english_gap.
_SCHEMA_LABELING_SUB_RULE = (
    "\n\nSCHEMA-LABELING SUB-RULE (this flavor): When the stem describes a "
    "structured situation using plain-English category cues (e.g., "
    "'categorized', 'measured', 'manipulated', 'assigned') and a distractor "
    "swaps which definitional label (IV/DV, encoding/storage/retrieval, "
    "agonist/antagonist, sensitivity/specificity, Type I/II error, "
    "fixed/random effects, between/within subjects, internal/external "
    "validity, classical/operant, assimilation/accommodation, etic/emic, "
    "distal/proximal) attaches to which facet of the situation, classify "
    "as CONTENT_GAP, not english_gap. Rejecting requires the student to "
    "know which definitional label applies to which kind of operation.\n"
    "\n"
    "PRECONDITION: this sub-rule applies ONLY when no universal-quantifier "
    "contradiction exists in the distractor. If the distractor uses 'all', "
    "'every', 'throughout', 'entire', 'any', 'always', or 'never' against "
    "a stem-stated specific case (e.g., stem says 'F normal, FB elevated'; "
    "distractor says 'overreporting throughout the entire protocol'), "
    "classify as ENGLISH_GAP regardless of any schema-labeling pattern. "
    "The Lester/wedding case ('ALL pre-injury memories' vs stated wedding) "
    "is ENGLISH_GAP."
)


# Per-flavor exception clauses. Empty string = no exception (default
# uniform rule applies). Each clause:
# 1. Names what kind of named-entity knowledge the flavor relies on.
# 2. Instructs the auditor to classify named-entity knowledge as
#    content_gap, NOT english_gap.
# 3. Specifies the residual english_gap criteria — when contradiction
#    is in a specific stem-stated fact independent of named-entity
#    knowledge.
# 4. (Phase 20a) Where schema-labeling patterns are common in the flavor,
#    appends _SCHEMA_LABELING_SUB_RULE to add the IV/DV-style exception.

_FLAVOR_AUDIT_ADDENDA: dict[str, str] = {
    "developmental_stage": (
        "\n\nFLAVOR CALIBRATION (developmental_stage): Named developmental "
        "stages (e.g., 'concrete operational', 'preoperational', "
        "'sensorimotor substage 6', 'formal operations') imply specific "
        "age ranges as part of their definition. Knowing WHICH age range "
        "corresponds to a named stage IS domain knowledge in this content "
        "area — not a lexical fact. A distractor that places a stage-named "
        "claim at the wrong age tests stage-knowledge → CLASSIFY AS "
        "CONTENT_GAP, NOT english_gap. Reserve english_gap for "
        "contradictions with stem-stated facts that are independent of "
        "stage-name knowledge (e.g., a stem describing a child's behavior "
        "as 'puzzled' contradicted by 'actively examining' — that is a "
        "lexical mental-state contradiction)."
    ),
    "framework": (
        "\n\nFLAVOR CALIBRATION (framework): Named frameworks (APA "
        "Standards, DSM criteria, legal authorities, ethical principles) "
        "imply specific scope and content as part of their definition. "
        "Knowing what an APA Standard or DSM criterion actually says is "
        "domain knowledge. A distractor that invokes the wrong standard, "
        "misstates a framework's content, or misapplies a principle "
        "tests framework-knowledge → CLASSIFY AS CONTENT_GAP. Reserve "
        "english_gap for contradictions with specific stem-stated facts "
        "(named subject's documented action, stated date, stated setting) "
        "that are independent of framework content."
    ),
    "mechanism": (
        "\n\nFLAVOR CALIBRATION (mechanism): Named drugs, receptors, "
        "neurotransmitters, pathways, and anatomical structures imply "
        "specific pharmacological/neurological properties. Knowing what "
        "'MAOI', 'D2 antagonist', 'pyramidal decussation', or 'intrinsic "
        "activity' implies is domain knowledge. A distractor that "
        "misstates a mechanism property tests mechanism-knowledge → "
        "CLASSIFY AS CONTENT_GAP. Reserve english_gap for cases where "
        "the stem prints a specific finding (e.g., 'no measurable "
        "postsynaptic activity', a numeric assay value, a stated "
        "laterality) and the distractor lexically contradicts it "
        "independent of mechanism knowledge."
    ),
    "clinical_disease": (
        "\n\nFLAVOR CALIBRATION (clinical_disease): Named diseases "
        "(Parkinson's, Alzheimer's, schizophrenia, sundowning, etc.) "
        "imply specific etiology, symptoms, imaging findings, and "
        "treatment as part of their clinical definition. Word etymology "
        "(e.g., 'sundowning' suggests evening) is part of disease-"
        "knowledge, not a lexical fact independent of clinical content. "
        "A distractor that confuses two diseases or misstates etiology/"
        "symptoms tests differential-diagnosis knowledge → CLASSIFY AS "
        "CONTENT_GAP. Reserve english_gap for contradictions with "
        "explicitly-stated stem details (the named subject's described "
        "symptoms, stated imaging findings, stated lab values) "
        "independent of disease-name knowledge."
    ),
    "cognitive_process": (
        "\n\nFLAVOR CALIBRATION (cognitive_process): Memory systems "
        "(sensory/short-term/long-term, working memory, declarative/"
        "procedural), attention mechanisms, and learning paradigms "
        "(classical/operant conditioning, encoding/storage/retrieval) "
        "imply specific properties. Knowing which system handles what "
        "is domain knowledge. A distractor that confuses process stages, "
        "memory systems, or learning mechanisms tests process-knowledge "
        "→ CLASSIFY AS CONTENT_GAP. Reserve english_gap for lexical-"
        "only contradiction with a printed stem fact (e.g., the stem "
        "states 'still recalls his wedding'; distractor says 'erases ALL "
        "pre-injury memories')."
    ),
    "applied_cultural": (
        "\n\nFLAVOR CALIBRATION (applied_cultural): Named identity-"
        "development stages (Cross's Nigrescence, Helms's White Identity, "
        "Atkinson's Minority Identity Development) and cultural frameworks "
        "imply specific characteristics and developmental sequences. "
        "Knowing what a stage entails (e.g., 'autonomy' implies developed/"
        "internalized non-racist identity) is domain knowledge. A "
        "distractor naming the wrong stage or misattributing stage "
        "characteristics is CONTENT_GAP. Reserve english_gap for lexical "
        "contradictions with stated awareness/behavior states (e.g., stem "
        "says 'puzzled'; distractor implies 'actively examining') that "
        "don't depend on stage-knowledge."
    ),
    "social_process": (
        "\n\nFLAVOR CALIBRATION (social_process): Named biases (actor-"
        "observer, fundamental attribution error, correspondence bias), "
        "group processes (groupthink, conformity, deindividuation), and "
        "attribution mechanisms imply specific cognitive operations. A "
        "distractor naming the wrong bias or misattributing process "
        "tests process-knowledge → CONTENT_GAP. Reserve english_gap "
        "for lexical contradictions with specific stem-stated facts "
        "independent of process-naming."
    ),
    "diagnostic_criterion": (
        "\n\nFLAVOR CALIBRATION (diagnostic_criterion): DSM criterion "
        "sets imply specific symptom counts, durations, exclusions, and "
        "specifiers. Knowing the actual DSM threshold is domain "
        "knowledge. A distractor citing the wrong criterion count, "
        "wrong duration, or missing exclusion is CONTENT_GAP. Reserve "
        "english_gap for cases where the stem prints a specific symptom "
        "or duration value and the distractor's analysis depends on a "
        "value not in the stem (e.g., distractor cites 'failure to meet "
        "criteria due to childhood-onset absence' when stem explicitly "
        "states childhood symptoms were confirmed — that is english_gap)."
    ),
    "statistical": (
        "\n\nFLAVOR CALIBRATION (statistical): Test names (ANOVA, "
        "MANOVA, t-test, chi-square), effect sizes, and assumption "
        "labels imply specific properties (number of IVs/DVs handled, "
        "level-of-measurement requirements, etc.). A distractor naming "
        "the wrong test for the design, misstating assumptions, or "
        "misinterpreting effect-size magnitude is CONTENT_GAP — "
        "rejecting requires knowing what each test does. Reserve "
        "english_gap for distractors that contradict specific numeric "
        "values printed in the stem (e.g., stem says n=200; distractor "
        "claims 'underpowered with n<50') independent of test-knowledge."
    ),
    "therapeutic_modality": (
        "\n\nFLAVOR CALIBRATION (therapeutic_modality): Named "
        "therapeutic modalities (CBT, DBT, MI, ACT, psychodynamic, "
        "person-centered) imply specific techniques and theoretical "
        "principles. A distractor naming the wrong modality or "
        "technique-substitution is CONTENT_GAP. Reserve english_gap "
        "for lexical contradictions with the named subject's stated "
        "presenting concerns or session events independent of modality-"
        "knowledge."
    ),
    "test_psychometric": (
        "\n\nFLAVOR CALIBRATION (test_psychometric): Test names (MMPI-2, "
        "WISC-V, WAIS-IV, Rorschach) imply specific subtests, validity "
        "scales, and score interpretations. Knowing what F, FB, K, or "
        "the Working Memory Index measures is domain knowledge. A "
        "distractor that names the wrong subtest, misinterprets a "
        "validity-scale pattern, or misstates score relationships is "
        "CONTENT_GAP. Reserve english_gap for distractors that "
        "numerically contradict specific scale values or scores stated "
        "in the stem (e.g., stem says 'F within normal limits, FB "
        "elevated'; distractor claims 'parallel F and FB elevations' — "
        "that is english_gap because the F value is stated)."
    ),
    "selection_psychometric": (
        "\n\nFLAVOR CALIBRATION (selection_psychometric): Selection-"
        "validity concepts (Taylor-Russell tables, base rates, selection "
        "ratios, criterion contamination, incremental validity) imply "
        "specific quantitative relationships. A distractor misstating "
        "these relationships (e.g., 'highest gain at high base rate' "
        "when Taylor-Russell shows the opposite) is CONTENT_GAP. "
        "Reserve english_gap for lexical contradictions with specific "
        "stem-stated values independent of the conceptual relationship."
    ),
    # generic: default uniform rule applies (no addendum)
    "generic": "",
}


# Phase 20a: append the schema-labeling sub-rule to flavors where the
# pattern (stem cues a definitional schema; distractor swaps labels) is
# most common. The sub-rule is THE SAME ACROSS FLAVORS but referenced
# from a single source (_SCHEMA_LABELING_SUB_RULE) so refinements stay
# coherent. Per Plan-agent review (regression risk #c): keeping the rule
# inside flavor blocks (vs as a second appended chunk) preserves each
# flavor's own "Reserve english_gap for X" carve-out, since the
# precondition "applies ONLY when no universal-quantifier contradiction
# exists" is built into the sub-rule itself.
_FLAVORS_WITH_SCHEMA_LABELING: tuple[str, ...] = (
    "statistical",         # IV/DV, between/within, fixed/random, internal/external validity
    "cognitive_process",   # encoding/storage/retrieval, classical/operant, assimilation/accommodation
    "mechanism",           # agonist/antagonist, presynaptic/postsynaptic
    "test_psychometric",   # sensitivity/specificity, validity vs reliability
    "applied_cultural",    # etic/emic, distal/proximal
    "diagnostic_criterion", # Type I/II error, etc.
    # Phase 22a bridge: missing flavors that empirically host schema-
    # labeling patterns. The deterministic structural classifier in
    # `pipeline.schema_labeling_classifier` is the authoritative
    # override — this list still feeds the prompt-side sub-rule as
    # belt-and-suspenders. Augmenting reduces upstream prompt failure
    # rate so the override doesn't carry the entire load.
    "social_process",      # refutational/supportive, conformity/compliance
    "framework",           # APA Standard X vs Y, DSM criterion A vs B
)
for _flavor in _FLAVORS_WITH_SCHEMA_LABELING:
    if _flavor in _FLAVOR_AUDIT_ADDENDA and _FLAVOR_AUDIT_ADDENDA[_flavor]:
        _FLAVOR_AUDIT_ADDENDA[_flavor] = (
            _FLAVOR_AUDIT_ADDENDA[_flavor] + _SCHEMA_LABELING_SUB_RULE
        )


def audit_addendum_for_flavor(flavor: str | None) -> str:
    """Return the audit calibration addendum for a flavor, or "" if
    the flavor has no exception clause defined."""
    if not flavor:
        return ""
    return _FLAVOR_AUDIT_ADDENDA.get(flavor, "")


def audit_addendum_for_question(question: dict) -> str:
    """Derive flavor from the question's metadata or anchor_uid and
    return the matching addendum.

    Resolution order:
    1. question["generation_metadata"]["flavor"] — set by post-Phase-15
       generations (preferred; reflects the flavor used at gen time).
    2. flavor_for_anchor(anchor_uid, domain_code) — derived for legacy
       questions without metadata.
    3. Empty string if no flavor can be determined.
    """
    md = question.get("generation_metadata") or {}
    flavor = md.get("flavor")
    if not flavor:
        anchor_uids = question.get("anchor_uids") or []
        anchor_uid = anchor_uids[0] if anchor_uids else None
        domain_code = question.get("domain_code")
        if anchor_uid or domain_code:
            flavor = flavor_for_anchor(anchor_uid, domain_code)
    return audit_addendum_for_flavor(flavor)
