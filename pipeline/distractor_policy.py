"""Adaptive distractor-quality decision matrix.

Resolves a `Cell` (gate action + correction strategy) from the four
context dimensions:

  - tier: 1-4 Bloom's
  - domain_code: 9 EPPP domains (BPSY, PETH, ...)
  - pedagogical_content_type: domain-keyed taxonomy (e.g., BPSY
    'concept_definition', PETH 'framework_application'); 'unknown' when
    the anchor brief hasn't been classified yet
  - misconception_type: per-distractor, one of VALID_MISCONCEPTION_TYPES

P0 (current) behavior: every resolution returns DEFAULT (current strict
behavior). The function exists so callers and gates can adopt the API
without a behavior change. P1 populates the matrix with empirically-
informed cells; P5 adds per-domain overrides.

Resolution order (most-specific wins):
  (tier, domain, content_type, misconception_type)
  → (tier, domain, content_type)
  → (tier, content_type)
  → (tier,)
  → DEFAULT

A module-level `Counter` records every resolution so cell coverage can
be inspected at end-of-run via `get_resolution_stats()`. Used in
P-1 to inform which cells deserve P1 authoring effort.
"""
from __future__ import annotations

import threading
from collections import Counter
from dataclasses import dataclass
from typing import Optional


# ── Enumerations ──────────────────────────────────────────────

GATE_ACTIONS = (
    # Currently populated in production matrices:
    "audit",            # default — Sonnet classifies, gate flags only english_gap
    "strict",           # call audit with no calibration hint (DEFAULT cell)
    "framework_aware",  # call audit with ethics-framework calibration hint

    # Historically populated, currently dead but reserved for future use:
    "skip",             # bypass audit entirely — used pre-3-class redesign
                        # for cells where audit was over-flagging. Kept in
                        # case a cell needs it again (e.g., a known-good
                        # content-type that doesn't benefit from audit).
    "permissive",       # call audit with relaxed calibration. Replaced by
                        # "audit" + cell-specific calibration hints.
)
CORRECTION_STRATEGIES = (
    "swap_for_content_distractor",
    "mechanism_inversion",
    "judgment_error",
    "rewrite_stem_to_observation",
    "framework_misapplication",
)


@dataclass(frozen=True)
class Cell:
    """A policy cell — what the gate should do, how the retry should
    correct, and (Phase A2.5) which deterministic detectors are
    override-eligible at this cell with what minimum confidence.

    A2.5 fields (additive, all default to "no override"):
      - override_thresholds: tuple of (detector_id, min_confidence) pairs.
        A detector signal at confidence ≥ min_confidence triggers
        OVERRIDE_TO at audit time. Detectors not in this tuple emit
        ADVISORY only at this cell. Empty tuple = no overrides.
      - co_firing_required: when True, two or more enabled detectors
        must agree on the same letter before any override fires.
        (Currently a flag; the wiring to enforce it is staged for a
        later iteration of A2.5 — single-detector behavior holds for
        the MVP since the english_gap_scanner only emits one signature
        per distractor.)
      - classification_prior: when set ("english_gap" | "content_gap"),
        the audit's verdict on borderline distractors (soft_flag class)
        is biased toward this prior. Default None = LLM verdict stands.
    """
    gate_action: str
    correction_strategy: str
    note: str = ""  # human-readable rationale, optional
    # ── A2.5 detector-aware fields ──────────────────────────────
    override_thresholds: tuple[tuple[str, float], ...] = ()
    co_firing_required: bool = False
    classification_prior: Optional[str] = None

    def __post_init__(self):
        if self.gate_action not in GATE_ACTIONS:
            raise ValueError(
                f"gate_action {self.gate_action!r} not in {GATE_ACTIONS}")
        if self.correction_strategy not in CORRECTION_STRATEGIES:
            raise ValueError(
                f"correction_strategy {self.correction_strategy!r} not in "
                f"{CORRECTION_STRATEGIES}")
        for entry in self.override_thresholds:
            if not (isinstance(entry, tuple) and len(entry) == 2):
                raise ValueError(
                    f"override_thresholds entries must be (detector_id, "
                    f"threshold) tuples; got {entry!r}"
                )
            did, thr = entry
            if not isinstance(did, str) or not did:
                raise ValueError(f"detector_id must be non-empty string; got {did!r}")
            if not (0.0 <= float(thr) <= 1.0):
                raise ValueError(
                    f"threshold must be in [0.0, 1.0]; got {thr!r}"
                )
        if self.classification_prior is not None:
            if self.classification_prior not in ("english_gap", "content_gap"):
                raise ValueError(
                    f"classification_prior must be None | 'english_gap' | "
                    f"'content_gap'; got {self.classification_prior!r}"
                )

    def threshold_for(self, detector_id: str) -> Optional[float]:
        """Return the override-eligibility threshold for ``detector_id``,
        or None if this detector is not override-eligible at this cell.
        """
        for did, thr in self.override_thresholds:
            if did == detector_id:
                return float(thr)
        return None


# ── DEFAULT ────────────────────────────────────────────────────
# Current gate behavior: strict detection, basic correction guidance.
# Every P0 resolution returns this. P1 introduces cell-specific entries.
DEFAULT = Cell(
    gate_action="strict",
    correction_strategy="swap_for_content_distractor",
    note="default — pre-matrix behavior",
)


# ── A2.5: Tier-keyed default cells ──────────────────────────────
# When no domain-specific cell matches, resolve() falls back to a
# tier-keyed default that controls detector override behavior.
# Confidence thresholds are graduated by tier:
#   T1/T2: 0.75 (any fired override-eligible signature wins)
#   T3:    0.85 (only universal_quantifier conf=0.85 fires)
#   T4:    0.95 (effectively advisory until co-firing wires up; single
#                signatures alone at 0.85 don't reach this floor)
# This recognizes that lexical contradictions at higher Bloom's tiers
# are more often content_gap-disguised-as-english_gap (the student is
# expected to read carefully and apply concept knowledge to recognize
# the contradiction).

DEFAULT_T1_T2 = Cell(
    gate_action="strict",
    correction_strategy="swap_for_content_distractor",
    note="T1/T2 default — aggressive override on any fired override-"
         "eligible signature (universal_quantifier 0.85, laterality 0.75, "
         "numeric_ratio 0.80 all >= threshold 0.75)",
    override_thresholds=(("english_gap_scanner", 0.75),),
)

DEFAULT_T3 = Cell(
    gate_action="strict",
    correction_strategy="swap_for_content_distractor",
    note="T3 default — conservative override; only universal_quantifier "
         "(conf 0.85) reaches the threshold. Laterality and numeric_ratio "
         "remain advisory.",
    override_thresholds=(("english_gap_scanner", 0.85),),
)

DEFAULT_T4 = Cell(
    gate_action="strict",
    correction_strategy="swap_for_content_distractor",
    note="T4 default — most conservative. No single english_gap_scanner "
         "signature reaches 0.95. Effectively advisory until co_firing "
         "is wired (a future iteration of A2.5). classification_prior "
         "biases borderline cases toward content_gap.",
    override_thresholds=(("english_gap_scanner", 0.95),),
    co_firing_required=True,
    classification_prior="content_gap",
)


def _tier_default(tier: Optional[int]) -> Cell:
    """Return the tier-keyed default cell. Falls back to DEFAULT if
    tier is unknown."""
    if tier in (1, 2):
        return DEFAULT_T1_T2
    if tier == 3:
        return DEFAULT_T3
    if tier == 4:
        return DEFAULT_T4
    return DEFAULT


def _compose_with_tier_default(domain_cell: Cell, tier: Optional[int]) -> Cell:
    """S2 — Compose a domain-specific cell with the tier-default for the
    A2.5 detector fields.

    Why: domain-specific cells in `POLICY_BPSY` and `POLICY_PETH` were
    authored before A2.5 introduced `override_thresholds`,
    `co_firing_required`, and `classification_prior`. They have empty
    defaults for those fields. Without composition, `resolve()` would
    return the domain cell directly and silently disable A2.5's
    detector promotion at every populated cell — even though the
    curators didn't intend to opt out of overrides; the field just
    didn't exist when they authored.

    Composition rule: domain cell ALWAYS wins on `gate_action`,
    `correction_strategy`, `note` (the original P1+ contract). For the
    A2.5 fields, the tier-default fills in IF the domain cell uses
    defaults (empty `override_thresholds`, False `co_firing_required`,
    None `classification_prior`). A domain cell that explicitly sets
    any A2.5 field keeps its own value — the field is opt-in for
    domain cells that have specific override semantics.
    """
    tier_default = _tier_default(tier)

    # If domain cell explicitly sets override_thresholds (non-empty),
    # respect it. Otherwise inherit tier-default's.
    composed_thresholds = (
        domain_cell.override_thresholds
        if domain_cell.override_thresholds
        else tier_default.override_thresholds
    )
    # co_firing: True wins (more conservative). Domain cell can opt IN to
    # co-firing even if tier-default doesn't require it; cannot opt out
    # of tier-default's requirement.
    composed_cofire = (
        domain_cell.co_firing_required
        or tier_default.co_firing_required
    )
    # classification_prior: domain cell wins if it sets one (overrides
    # tier-default's prior); otherwise tier-default's prior fills in.
    composed_prior = (
        domain_cell.classification_prior
        if domain_cell.classification_prior is not None
        else tier_default.classification_prior
    )

    return Cell(
        gate_action=domain_cell.gate_action,
        correction_strategy=domain_cell.correction_strategy,
        note=domain_cell.note,
        override_thresholds=composed_thresholds,
        co_firing_required=composed_cofire,
        classification_prior=composed_prior,
    )


# ── Per-domain matrix slots (P1+ will populate) ───────────────
# Sparse dict-of-dicts. Each domain's matrix is keyed by tuples of
# whatever dimensions the cell cares about. Resolution falls back from
# more-specific to less-specific keys, then to DEFAULT.

POLICY_BPSY: dict[tuple, Cell] = {
    # ── T4 evaluation patterns: structurally content-gap-heavy ─────
    # Under the 3-class audit (introduced 2026-04-28 evening), Sonnet
    # classifies each distractor as english_gap / content_gap / clean.
    # These cells were previously SKIP because Sonnet's binary view
    # mis-flagged content_gap distractors as stem-eliminable. With the
    # 3-class output, we let Sonnet classify — the calibration hint in
    # the cell's note biases Sonnet toward content_gap for distractors
    # that vary on technical vocabulary whose meaning IS the test.
    (4, "integrated", "contrast_prompt"): Cell(
        gate_action="audit",
        correction_strategy="judgment_error",
        note="contrast_prompt prints diagnostic criteria to enable "
             "comparison; distractors that misclassify entities "
             "typically require concept knowledge to recognize as wrong",
    ),
    (4, "integrated", "subtle_error"): Cell(
        gate_action="audit",
        correction_strategy="judgment_error",
        note="subtle_error quotes student reasoning containing technical "
             "claims; distractors evaluating the reasoning engage those "
             "claims; rejecting requires understanding the framework",
    ),
    (4, "integrated", "competing_evidence"): Cell(
        gate_action="audit",
        correction_strategy="judgment_error",
        note="balancing multiple evidence lines — distractors are "
             "alternative weightings of the same evidence",
    ),
    (4, "integrated", "integration"): Cell(
        gate_action="audit",
        correction_strategy="judgment_error",
        note="synthesizing 2+ concepts — distractors are "
             "mis-integrations requiring concept knowledge to reject",
    ),
    # T4 best_answer stays DEFAULT (strict) — relativistic comparison
    # of plausible options, less structural-ceiling risk.

    # ── T2 concept-definition cells ──────────────────────────
    # Stems must restate the diagnostic criterion to be answerable;
    # distractors that vary on the criterion are typically content_gap
    # (require knowing what the criterion means). 3-class audit lets
    # Sonnet classify; calibration hint pushes toward content_gap.
    (2, "anchor_grounded", "paraphrase"): Cell(
        gate_action="audit",
        correction_strategy="rewrite_stem_to_observation",
        note="'Which property best classifies' style — definitional "
             "restatement is structural; distractors that contradict "
             "the printed criterion are typically content_gap",
    ),
    (2, "integrated", "simple_application"): Cell(
        gate_action="audit",
        correction_strategy="rewrite_stem_to_observation",
        note="apply concept to described compound; description "
             "necessarily restates the diagnostic criterion",
    ),

    # ── T3 apply patterns: keep audit + mechanism_inversion ──
    # T3 strict catches real cases (PHY-195 H-03 was a real catch).
    # Use mechanism-inversion correction so retries fix via mechanism.
    # No calibration hint needed — T3 stem_eliminable failures tend
    # to be real english_gap.
    (3, "integrated", "clinical_vignette"): Cell(
        gate_action="strict",
        correction_strategy="mechanism_inversion",
        note="clinical case — mechanism-inversion correction",
    ),
    (3, "integrated", "scenario_completion"): Cell(
        gate_action="strict",
        correction_strategy="mechanism_inversion",
        note="scenario prediction — mechanism-inversion correction",
    ),

    # All other cells fall through to DEFAULT (strict + swap_for_content).
}

POLICY_PETH: dict[tuple, Cell] = {
    # ── Ethics framework-application cells ─────────────────────
    # Ethics distractors that apply the WRONG framework correctly
    # (e.g., applying research ethics rules to a clinical case) test
    # framework selection — a content-knowledge gap, not a reading
    # gap. The framework_aware calibration hint biases Sonnet toward
    # content_gap classification for these cases.
    #
    # Pattern coverage (per P-1 data, 5 PETH anchor briefs on disk):
    # PETH T3 underuses mechanism-flavored patterns (case_analysis,
    # mechanism_application, error_identification) — those are
    # biopsych-shaped. PETH T3 instead concentrates on
    # clinical_vignette/scenario_completion where ethical scenarios
    # are presented and the student selects the right ethical action.

    # T2 understand-tier ethics patterns
    (2, "anchor_grounded", "paraphrase"): Cell(
        gate_action="framework_aware",
        correction_strategy="framework_misapplication",
        note="ethics paraphrase — restates an APA standard or ethical "
             "principle; distractors that vary on the principle's "
             "scope/applicability are typically content_gap (require "
             "knowing what the principle says)",
    ),
    (2, "integrated", "comparison"): Cell(
        gate_action="framework_aware",
        correction_strategy="framework_misapplication",
        note="comparing two ethical principles or APA standards; "
             "distractors confuse adjacent principles — typically "
             "content_gap (requires distinguishing the principles)",
    ),

    # T3 apply-tier ethics patterns
    (3, "integrated", "clinical_vignette"): Cell(
        gate_action="framework_aware",
        correction_strategy="framework_misapplication",
        note="ethics scenario — psychologist faces ethical decision; "
             "distractors apply wrong framework (e.g., research ethics "
             "vs clinical ethics) — content_gap by design",
    ),
    (3, "integrated", "scenario_completion"): Cell(
        gate_action="framework_aware",
        correction_strategy="framework_misapplication",
        note="ethics scenario completion — student picks ethically "
             "correct next step; distractors are framework misapplications",
    ),
    (3, "integrated", "case_analysis"): Cell(
        gate_action="framework_aware",
        correction_strategy="framework_misapplication",
        note="ethics case analysis — analyze a multi-stakeholder "
             "ethical case; distractors miss key stakeholder or "
             "misapply standard",
    ),

    # T4 evaluate-tier ethics patterns — like BPSY T4 cells, rich
    # vignettes naturally produce content_gap distractors.
    (4, "integrated", "contrast_prompt"): Cell(
        gate_action="framework_aware",
        correction_strategy="framework_misapplication",
        note="ethics contrast — compare two ethical interpretations; "
             "distractors weight ethical considerations differently",
    ),
    (4, "integrated", "subtle_error"): Cell(
        gate_action="framework_aware",
        correction_strategy="framework_misapplication",
        note="evaluate quoted ethical reasoning; distractors engage "
             "the framework chain at different points",
    ),
    (4, "integrated", "competing_evidence"): Cell(
        gate_action="framework_aware",
        correction_strategy="framework_misapplication",
        note="balancing competing ethical principles (e.g., autonomy "
             "vs beneficence); distractors apply wrong weighting",
    ),
    (4, "integrated", "best_answer"): Cell(
        gate_action="framework_aware",
        correction_strategy="framework_misapplication",
        note="best ethical action — relativistic pick among defensible "
             "options; distractors are framework-suboptimal but not "
             "absolutely wrong",
    ),
    (4, "integrated", "integration"): Cell(
        gate_action="framework_aware",
        correction_strategy="framework_misapplication",
        note="synthesize 2+ ethical frameworks (e.g., APA Code + state "
             "law); distractors are mis-integrations",
    ),
}

DOMAIN_OVERRIDES: dict[str, dict[tuple, Cell]] = {
    "BPSY": POLICY_BPSY,
    "PETH": POLICY_PETH,
}


# ── Resolution stats (cell-coverage instrumentation) ─────────
# Counts every resolution for cell-coverage analysis. Caller queries
# `get_resolution_stats()` at end-of-run to see the distribution.

_resolution_counter: Counter = Counter()

# Lock guarding _resolution_counter mutations. Counter.update / __iadd__
# aren't strictly atomic in pure Python — under parallel orchestrator
# instances (e.g., the 27-anchor parallel sampling), races could lose
# counts. The lock is cheap; resolve() is called on the order of
# microseconds per call, so contention is negligible.
_resolution_lock = threading.Lock()


def get_resolution_stats() -> dict[tuple, int]:
    """Return a snapshot of resolution counts keyed by
    (tier, domain, content_type, misconception_type)."""
    with _resolution_lock:
        return dict(_resolution_counter)


def reset_resolution_stats() -> None:
    """Clear the resolution counter. Call at the start of a run if you
    want per-run stats rather than per-process."""
    with _resolution_lock:
        _resolution_counter.clear()


# ── resolve() ─────────────────────────────────────────────────

def resolve(
    tier: Optional[int] = None,
    domain_code: Optional[str] = None,
    pedagogical_content_type: Optional[str] = None,
    misconception_type: Optional[str] = None,
    source_type: Optional[str] = None,
    stem_pattern: Optional[str] = None,
) -> Cell:
    """Resolve a Cell from the matrix dimensions.

    Per P-1 finding (2026-04-28): `stem_pattern` is the empirical
    content-type signal, not the proposed `pedagogical_content_type`
    (which would have been a meta-classification). Cells are keyed on
    `(tier, source_type, stem_pattern)` for general behavior. The
    `pedagogical_content_type` arg is retained for API stability but
    is currently unused; future work may layer it in if/when briefs
    carry richer signals than patterns can encode.

    Resolution order:
      1. `(tier, source_type, stem_pattern)` in domain-specific matrix
      2. `(tier, stem_pattern)` in domain-specific matrix
      3. `(tier,)` in domain-specific matrix
      4. DEFAULT

    misconception_type is recorded for instrumentation but NOT consumed
    by any gate today. It's diagnostic intent metadata: DistractorPlannerAgent
    pre-assigns one of `pipeline.VALID_MISCONCEPTION_TYPES` per distractor
    so the question carries diagnostic information for downstream remediation,
    but no validation gate currently reads the field. P3 (deferred) would
    promote it to a primary cell-resolution key. Recording it here lets
    cell-coverage analysis see the per-misconception-type distribution.

    Resolution order: only the (t, src, pat) layer is populated in current
    matrices. The historically-supported (t, pat) and (t,) fallback layers
    were removed in 2026-04-29 cleanup — no cells existed at those keys, so
    they were dead code. If a future iteration wants per-tier or
    per-(tier, pattern) cells (e.g., a tier-wide policy override), reinstate
    the fallback layers.
    """
    # Normalize unknowns so the counter buckets cleanly
    ct = pedagogical_content_type or "unknown"
    mt = misconception_type or "unknown"
    dc = domain_code or "unknown"
    src = source_type or "unknown"
    pat = stem_pattern or "unknown"
    t = tier if tier is not None else 0

    with _resolution_lock:
        _resolution_counter[(t, dc, src, pat, mt)] += 1

    matrix = DOMAIN_OVERRIDES.get(dc, {})
    if (t, src, pat) in matrix:
        # S2: compose the domain cell with the tier-default so the
        # A2.5 detector fields aren't silently nulled out by domain
        # cells authored pre-A2.5. Domain cell wins on the original
        # gate_action/correction_strategy fields; tier-default fills
        # in detector fields when domain cell uses defaults.
        return _compose_with_tier_default(matrix[(t, src, pat)], tier)
    # A2.5: fall back to tier-keyed default for detector override behavior
    # before the bare DEFAULT. This is what gives T1/T2 vs T3 vs T4 their
    # different override thresholds in the absence of a domain-specific
    # cell (which is the common case).
    return _tier_default(tier)
