"""Single source of truth for distractor-quality classification.

The 3-class scheme (ENGLISH_GAP / CONTENT_GAP / CLEAN) is referenced in
multiple places — the audit prompt, the in-line gate's parsing logic,
and the v2 generation prompt. Without a SSOT these can drift: a tweak
to the audit's class definitions wouldn't automatically reach the
generation prompt's design directive, leading to subtle gen/audit
disagreement that surfaces as inflated english_gap rates only on
specific content.

This module defines:

  - The class-name constants (ENGLISH_GAP, CONTENT_GAP, CLEAN) — string
    literals matching the audit's JSON output, used for comparison in
    `pipeline.gates.StemEliminableDistractorGate.check`.
  - Canonical examples for each class — drawn from real failures
    observed in stress-tests and the corpus audit.
  - The discriminating-rule heuristic — the question Opus and Sonnet
    both apply: "could a student reject this distractor by re-reading
    the stem alone?"
  - `prompt_block_for_audit()` and `prompt_block_for_generation()` —
    helpers that emit the canonical 3-class definitions formatted
    appropriately for each consumer's prompt template.

Pattern-classification (directional / stage_mental_state / etc.) is
out of scope for this module — see `pipeline.failure_patterns` (TIER 2
item C) when it lands. This module is just the english_gap/content_gap/
clean trichotomy.
"""
from __future__ import annotations

from dataclasses import dataclass


# ── Class name constants ──────────────────────────────────────
# Match the JSON `class` field emitted by the audit prompt and read
# by the in-line gate's classifications-list parser. DO NOT change
# these strings without also updating the audit prompt and the gate.
ENGLISH_GAP = "english_gap"
CONTENT_GAP = "content_gap"
CLEAN = "clean"
# Phase 20b: 4th class for the auditor to express confidence-borderline
# cases. Soft_flagged distractors do NOT block ship — they surface in
# the manifest as a separate dimension for optional human review.
# Empirically motivated: temperature=0 reduces but does not eliminate
# Sonnet's classification jitter on borderline cases (observed run-to-
# run flips between english_gap and clean on PMET IV/DV B and SOCU
# race-frame). Forcing a binary on borderlines is the source of jitter;
# soft_flag lets the auditor say "uncertain" instead of guessing.
SOFT_FLAG = "soft_flag"

CLASS_NAMES: frozenset[str] = frozenset({ENGLISH_GAP, CONTENT_GAP, CLEAN, SOFT_FLAG})


# ── Canonical examples ───────────────────────────────────────
# One worked example per class. Drawn from real corpus content that
# the audit has classified consistently across multiple runs.

@dataclass(frozen=True)
class CanonicalExample:
    """A worked example of a class, with stem/distractor/why fields."""
    class_name: str
    stem: str
    distractor: str
    why: str  # one-sentence explanation of why this distractor is in the class


CANONICAL_EXAMPLES: tuple[CanonicalExample, ...] = (
    CanonicalExample(
        class_name=ENGLISH_GAP,
        stem=("After bilateral hippocampal damage, Lester Nichols cannot "
              "form new declarative memories but still recalls his wedding "
              "from a decade earlier."),
        distractor=("Retrograde amnesia erases all pre-injury memories "
                    "regardless of when they were consolidated."),
        why=('"all" is contradicted by the wedding case in the stem. '
             "Student rejects without knowing what retrograde amnesia is."),
    ),
    CanonicalExample(
        class_name=CONTENT_GAP,
        stem=("A compound binds a receptor but produces no measurable "
              "postsynaptic activity on its own."),
        distractor=("It has intrinsic activity that mimics the endogenous "
                    "neurotransmitter."),
        why=("rejecting requires knowing intrinsic activity ⇒ measurable "
             "postsynaptic effect. Tests concept knowledge directly."),
    ),
    CanonicalExample(
        class_name=CLEAN,
        stem=("CT imaging confirms an acute thromboembolic infarct in the "
              "right cerebral hemisphere involving the motor cortex."),
        distractor=("Predict hemiplegia consistent with closed head trauma "
                    "rather than pyramidal infarction."),
        why=('stem doesn\'t say "not closed head trauma"; rejecting '
             "requires knowing thromboembolic ≠ trauma etiology."),
    ),
    CanonicalExample(
        class_name=SOFT_FLAG,
        stem=("Leon prefers lighter-skinned colleagues, avoids events "
              "where most attendees share his darker complexion, and "
              "has felt ashamed of his own appearance since adolescence."),
        distractor=("Predict generalized self-esteem patterns reflecting "
                    "negative self-schema unrelated to racial group "
                    "membership."),
        why=("ambiguous — could be english_gap (frame denial: 'unrelated "
             "to X' contradicts stem's stated X-framing) OR clean (the "
             "distractor is just naming a different analytic frame, not "
             "contradicting a specific fact). When you cannot decide "
             "between two classes, soft_flag the case rather than "
             "forcing a binary."),
    ),
)


# ── Discriminating heuristic ─────────────────────────────────
# The question both Opus (generation) and Sonnet (audit) apply.
# Identical wording on both sides keeps the architecture coherent.
DISCRIMINATING_RULE = (
    "Could a student who hasn't studied the concept reject this "
    "distractor by re-reading the stem alone?\n"
    "  - YES → english_gap. (Forbidden — quality failure.)\n"
    "  - NO, but a real contradiction exists once you know the concept "
    "→ content_gap. (Preferred — the workhorse.)\n"
    "  - NO direct contradiction at all, just a wrong-but-plausible "
    "alternative → clean. (Preferred — secondary.)\n"
    "  - GENUINELY UNCERTAIN between two classes after applying the "
    "rule → soft_flag. (Use sparingly — only when classification is "
    "honestly borderline. Do NOT use as an out from making a clear "
    "call.) Soft_flagged distractors do not block ship; they surface "
    "for optional human review."
)


# ── Class definitions (one-paragraph each) ───────────────────

CLASS_DEFINITIONS: dict[str, str] = {
    ENGLISH_GAP: (
        "A student can reject the distractor by lexical comparison with "
        "the stem alone. The contradiction is in printed words, not "
        "concepts."
    ),
    CONTENT_GAP: (
        "A real contradiction exists, but recognizing it requires "
        "invoking concept knowledge. The distractor LOOKS plausible "
        "until you know what a technical term actually means."
    ),
    CLEAN: (
        "A wrong-but-plausible alternative not directly contradicting "
        "any stem fact. Rejection requires applying concept knowledge."
    ),
    SOFT_FLAG: (
        "Genuinely uncertain between two classes after honest "
        "application of the discriminating rule. Soft_flagged "
        "distractors are NOT counted as quality failures (they don't "
        "block ship), but they are surfaced in the audit manifest for "
        "optional human review. Use only when the classification is "
        "actually borderline — not as an out from a call you could "
        "make."
    ),
}


# ── Prompt-block helpers ─────────────────────────────────────
# Both helpers emit the canonical examples in a form appropriate to
# their consumer. Audit gets a "classify into one of these" instruction;
# generation gets a "produce content_gap or clean, never english_gap"
# directive. Same examples in both — drift-prevention.


def _format_example(ex: CanonicalExample, label: str) -> str:
    """Render a CanonicalExample with a label like 'Canonical example' or
    'PREFERRED'."""
    return (
        f"  {label}:\n"
        f"    Stem: \"{ex.stem}\"\n"
        f"    Distractor: \"{ex.distractor}\"\n"
        f"    Why: {ex.why}"
    )


def prompt_block_for_audit() -> str:
    """Return the 4-class prompt block formatted for the audit-side
    classification instruction (Sonnet-facing).

    The audit asks Sonnet to CLASSIFY each distractor. The block leads
    with the class definitions, includes one canonical example per
    class, and ends with the discriminating rule.

    Phase 20b: SOFT_FLAG was added as a fourth class for genuine
    auditor uncertainty. It does NOT block ship.
    """
    blocks = []
    blocks.append("THE FOUR CLASSES:\n")
    for class_name in (ENGLISH_GAP, CONTENT_GAP, CLEAN, SOFT_FLAG):
        ex = next(e for e in CANONICAL_EXAMPLES if e.class_name == class_name)
        label = {
            ENGLISH_GAP: "ENGLISH_GAP (FORBIDDEN — quality failure)",
            CONTENT_GAP: "CONTENT_GAP (PREFERRED — the workhorse)",
            CLEAN: "CLEAN (PREFERRED — secondary)",
            SOFT_FLAG: "SOFT_FLAG (acceptable — use only when genuinely uncertain)",
        }[class_name]
        blocks.append(f"\n{label}:")
        blocks.append(CLASS_DEFINITIONS[class_name])
        blocks.append(_format_example(ex, "Canonical example"))
    blocks.append("\nCLASSIFICATION RULE:")
    blocks.append(DISCRIMINATING_RULE)
    return "\n".join(blocks)


def prompt_block_for_generation() -> str:
    """Return the 3-class prompt block formatted for the generation-side
    design directive (Opus-facing).

    The generation prompt asks Opus to DESIGN distractors that fall in
    content_gap or clean. The block emphasizes "produce content_gap or
    clean, never english_gap" as the design rule.
    """
    blocks = []
    blocks.append("THE THREE CLASSES (downstream auditor will classify each "
                  "distractor; design with this in mind):\n")
    for class_name in (ENGLISH_GAP, CONTENT_GAP, CLEAN):
        ex = next(e for e in CANONICAL_EXAMPLES if e.class_name == class_name)
        label = {
            ENGLISH_GAP: "ENGLISH_GAP (FORBIDDEN — quality failure)",
            CONTENT_GAP: "CONTENT_GAP (PREFERRED — the workhorse)",
            CLEAN: "CLEAN (PREFERRED — secondary)",
        }[class_name]
        blocks.append(f"\n{label}:")
        blocks.append(CLASS_DEFINITIONS[class_name])
        blocks.append(_format_example(ex, "Canonical example"))
    blocks.append("\nTHE DESIGN DIRECTIVE:")
    blocks.append("Produce content_gap (preferred) or clean (acceptable). "
                  "Never produce english_gap. Discriminating test:")
    blocks.append(DISCRIMINATING_RULE)
    return "\n".join(blocks)


def is_known_class(class_name: str) -> bool:
    """Return True if `class_name` matches one of the canonical class
    constants. Useful for parser validation."""
    return class_name in CLASS_NAMES
