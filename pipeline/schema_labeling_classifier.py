"""Phase 22: Deterministic structural classifier for schema-labeling.

The complement (and authoritative override) to the prompt-side schema-
labeling sub-rule in ``audit_calibration._SCHEMA_LABELING_SUB_RULE``.
Phase 21 validation showed that prompting alone is the weakest
reinforcer in the audit stack: multi-pass quorum (Phase 21a) and
cross-model verifier (Phase 21c) reproduced the same wrong verdict on
SOCU/persuasion M-01 because both followed the same prompt. This
module provides a deterministic code path that fires structurally
and overrides the audit's english_gap verdict to content_gap when
the schema-labeling pattern is detected.

The pattern: a stem describes a structured situation using paired-
named-concepts (IV/DV, agonist/antagonist, refutational/supportive)
and a distractor swaps which label attaches to which facet. Rejecting
requires schema knowledge — content_gap, not english_gap.

Design decisions:

- Pure function, no API calls, no I/O.
- Universal-quantifier guard as a HARD precondition: distractors with
  'all', 'every', 'throughout', 'entire', 'any', 'always', or 'never'
  remain english_gap regardless of any paired-concept signal. This
  protects the Lester/wedding canonical case and the BPSY postsynaptic-
  firing over-specification case.
- Three-tier signal hierarchy. Confidence is graded for telemetry;
  the override decision is a single boolean threshold (>= 0.5).
    Tier A (1.0) brief-boosted: discriminators carry a labeled-pair
                  shape (e.g. 'refutational_vs_supportive_defense');
                  both pair members in stem; one in distractor.
    Tier B (0.5) lexical fallback: pair from LABEL_PAIRS appears in
                  stem within proximity window; one member in
                  distractor.
    Tier C (0.0) no signal.
- ``LABEL_PAIRS`` is an explicit, audited list. Don't expand reactively
  — push gardening to brief discriminators where it belongs (one
  source of truth with chapter-level review).

The override is traced on overridden entries via:
  ``structural_override = "schema_labeling"``
  ``original_class = "english_gap"``
  ``structural_override_confidence = 1.0 | 0.5``
  ``structural_override_reason = "<short trace>"``
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# ── Constants ────────────────────────────────────────────────

# Labeled pairs the classifier recognises by shape. Each pair is
# (label_a, label_b), case-insensitive, matched on word boundaries.
# Adding a pair is a structural change — it requires a unit test.
# Do NOT expand reactively to chase coverage; push pattern coverage
# into brief.discriminators (Tier A) where it belongs.
LABEL_PAIRS: tuple[tuple[str, str], ...] = (
    ("agonist", "antagonist"),
    ("refutational", "supportive"),
    ("encoding", "retrieval"),
    ("encoding", "storage"),
    ("storage", "retrieval"),
    ("sensitivity", "specificity"),
    ("classical", "operant"),
    ("assimilation", "accommodation"),
    ("etic", "emic"),
    ("distal", "proximal"),
    ("between", "within"),         # subjects
    ("fixed", "random"),           # effects
    ("internal", "external"),      # validity
    ("presynaptic", "postsynaptic"),
    ("intrinsic", "extrinsic"),
    ("pre-exposure", "post-exposure"),
    ("independent variable", "dependent variable"),
)

# Uppercase-required pairs — short tokens like "IV"/"DV" or "Type I" /
# "Type II" generate too many false positives if matched case-insensitively
# (IV inside intravenous, give, etc.). These match only with their stated
# casing or the canonical alternative form.
UPPERCASE_PAIRS: tuple[tuple[str, str], ...] = (
    ("IV", "DV"),
    ("Type I", "Type II"),
)

# Universal quantifiers that signal a lexical contradiction independent
# of schema-labeling. If any appears in the distractor, the override
# does NOT fire — english_gap stands.
UNIVERSAL_QUANTIFIERS: tuple[str, ...] = (
    "all", "every", "throughout", "entire", "any", "always", "never",
)

# Maximum character distance between paired tokens in the stem for
# Tier B lexical detection. Empirically a paired-concept stem keeps
# both labels within ~150 chars; 200 gives slack for explanatory clauses.
TIER_B_PROXIMITY_CHARS: int = 200

_UNIVERSAL_RE = re.compile(
    r"\b(" + "|".join(UNIVERSAL_QUANTIFIERS) + r")\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SchemaLabelingSignal:
    """Result of running ``classify_distractor`` on one (stem, distractor)
    pair. ``fired`` is the override decision; the other fields are for
    telemetry, debugging, and manifest tracing.
    """
    fired: bool
    confidence: float
    reason: str
    pair_matched: tuple[str, str] | None
    universal_quantifier_blocked: bool
    brief_boosted: bool


_NEGATIVE = SchemaLabelingSignal(
    fired=False,
    confidence=0.0,
    reason="no_signal",
    pair_matched=None,
    universal_quantifier_blocked=False,
    brief_boosted=False,
)


# ── Brief-discriminator parsing ──────────────────────────────

_VS_RE = re.compile(
    r"^([a-z][a-z0-9]*)_vs_([a-z][a-z0-9]*)(?:_[a-z][a-z0-9_]*)?$",
    re.IGNORECASE,
)


def labeled_pair_discriminators(
    discriminators: list[str] | None,
) -> list[tuple[str, str]]:
    """Parse brief discriminators of the form ``X_vs_Y`` (with optional
    suffix like ``_definition`` or ``_timing``) into ``(X, Y)`` tuples.

    Returns an empty list if ``discriminators`` is None/empty or contains
    no labeled-pair shapes. Tolerant of single-axis discriminators
    (``intentionality_level``) — they are silently dropped.
    """
    if not discriminators:
        return []
    pairs: list[tuple[str, str]] = []
    for item in discriminators:
        if not isinstance(item, str):
            continue
        m = _VS_RE.match(item.strip())
        if m is None:
            continue
        a = m.group(1).lower().replace("_", " ").strip()
        b = m.group(2).lower().replace("_", " ").strip()
        if a and b and a != b:
            pairs.append((a, b))
    return pairs


# ── Lexical detection helpers ────────────────────────────────

def _has_universal_quantifier(distractor_text: str) -> bool:
    if not distractor_text:
        return False
    return _UNIVERSAL_RE.search(distractor_text) is not None


def _find_all_positions(text: str, token: str, *,
                        case_sensitive: bool = False) -> list[int]:
    """Find all occurrences of ``token`` in ``text`` on word boundaries.

    Returns a list of start indices (empty if no match). Multi-word
    tokens (e.g. ``independent variable``) match as a literal sequence
    on word boundaries at each end.
    """
    if not text or not token:
        return []
    flags = 0 if case_sensitive else re.IGNORECASE
    pat = r"\b" + re.escape(token) + r"\b"
    return [m.start() for m in re.finditer(pat, text, flags)]


def _min_cross_distance(positions_a: list[int],
                        positions_b: list[int]) -> int | None:
    """Smallest absolute distance between any element of ``positions_a``
    and any element of ``positions_b``. Returns None if either is empty."""
    if not positions_a or not positions_b:
        return None
    return min(abs(a - b) for a in positions_a for b in positions_b)


def _both_in_stem_within_window(
    stem: str, label_a: str, label_b: str, *,
    case_sensitive: bool = False,
    window: int = TIER_B_PROXIMITY_CHARS,
) -> bool:
    pa = _find_all_positions(stem, label_a, case_sensitive=case_sensitive)
    pb = _find_all_positions(stem, label_b, case_sensitive=case_sensitive)
    d = _min_cross_distance(pa, pb)
    return d is not None and d <= window


def _distractor_mentions_either(
    distractor_text: str, label_a: str, label_b: str, *,
    case_sensitive: bool = False,
) -> bool:
    if (_find_all_positions(distractor_text, label_a,
                            case_sensitive=case_sensitive)
            or _find_all_positions(distractor_text, label_b,
                                   case_sensitive=case_sensitive)):
        return True
    return False


# ── Public classifier ────────────────────────────────────────

def classify_distractor(
    *,
    stem: str,
    distractor_text: str,
    contradicted_stem_fact: str | None = None,
    discriminators: list[str] | None = None,
) -> SchemaLabelingSignal:
    """Detect whether ``(stem, distractor)`` is a schema-labeling swap.

    Three-tier hierarchy (Tier A > Tier B > no signal). Universal-
    quantifier guard is a hard precondition: any match in the
    distractor blocks the override regardless of tier signal.

    Returns a ``SchemaLabelingSignal``. The override decision is
    ``signal.fired``; ``signal.confidence`` is for telemetry.
    """
    # Universal-quantifier guard — checked once, applies to every tier.
    blocked = _has_universal_quantifier(distractor_text or "")

    # Tier A: brief-boosted. Try first — if a brief explicitly names a
    # labeled pair, that's the highest-precision input.
    brief_pairs = labeled_pair_discriminators(discriminators)
    for (a, b) in brief_pairs:
        if (_both_in_stem_within_window(stem or "", a, b)
                and _distractor_mentions_either(distractor_text or "", a, b)):
            if blocked:
                return SchemaLabelingSignal(
                    fired=False, confidence=1.0,
                    reason=f"tier_a_blocked_by_universal_quantifier:{a}_vs_{b}",
                    pair_matched=(a, b),
                    universal_quantifier_blocked=True,
                    brief_boosted=True,
                )
            return SchemaLabelingSignal(
                fired=True, confidence=1.0,
                reason=f"tier_a_brief:{a}_vs_{b}",
                pair_matched=(a, b),
                universal_quantifier_blocked=False,
                brief_boosted=True,
            )

    # Tier B: lexical fallback against canonical LABEL_PAIRS.
    for (a, b) in LABEL_PAIRS:
        if (_both_in_stem_within_window(stem or "", a, b)
                and _distractor_mentions_either(distractor_text or "", a, b)):
            if blocked:
                return SchemaLabelingSignal(
                    fired=False, confidence=0.5,
                    reason=f"tier_b_blocked_by_universal_quantifier:{a}_vs_{b}",
                    pair_matched=(a, b),
                    universal_quantifier_blocked=True,
                    brief_boosted=False,
                )
            return SchemaLabelingSignal(
                fired=True, confidence=0.5,
                reason=f"tier_b_lexical:{a}_vs_{b}",
                pair_matched=(a, b),
                universal_quantifier_blocked=False,
                brief_boosted=False,
            )

    # Tier B (uppercase-required): IV/DV, Type I/II.
    for (a, b) in UPPERCASE_PAIRS:
        if (_both_in_stem_within_window(stem or "", a, b, case_sensitive=True)
                and _distractor_mentions_either(distractor_text or "", a, b,
                                                case_sensitive=True)):
            if blocked:
                return SchemaLabelingSignal(
                    fired=False, confidence=0.5,
                    reason=f"tier_b_blocked_by_universal_quantifier:{a}_vs_{b}",
                    pair_matched=(a, b),
                    universal_quantifier_blocked=True,
                    brief_boosted=False,
                )
            return SchemaLabelingSignal(
                fired=True, confidence=0.5,
                reason=f"tier_b_lexical:{a}_vs_{b}",
                pair_matched=(a, b),
                universal_quantifier_blocked=False,
                brief_boosted=False,
            )

    # Tier C: no signal.
    return _NEGATIVE


# ── Override application ────────────────────────────────────

def apply_schema_labeling_override(
    question: dict,
    classifications: list[dict],
    discriminators: list[str] | None = None,
) -> tuple[list[dict], int]:
    """For each ``english_gap`` entry in ``classifications``, run the
    structural classifier; on fire, demote to ``content_gap`` and stamp
    structural-override metadata. Returns ``(new_classifications,
    override_count)``. Does not mutate the input list.

    Override metadata added to overridden entries:
      - ``class`` -> ``"content_gap"``
      - ``structural_override`` -> ``"schema_labeling"``
      - ``original_class`` -> ``"english_gap"`` (preserved for trace)
      - ``structural_override_confidence`` -> 1.0 or 0.5
      - ``structural_override_reason`` -> the signal's reason field
      - ``contradicted_stem_fact`` removed (no longer applicable)
    """
    from pipeline.quality_taxonomy import ENGLISH_GAP, CONTENT_GAP

    stem = (question or {}).get("question_stem", "") or ""
    new_list: list[dict] = []
    override_count = 0
    for c in classifications or []:
        if c.get("class") != ENGLISH_GAP:
            new_list.append(c)
            continue
        signal = classify_distractor(
            stem=stem,
            distractor_text=c.get("distractor_text", "") or "",
            contradicted_stem_fact=c.get("contradicted_stem_fact"),
            discriminators=discriminators,
        )
        if not signal.fired:
            new_list.append(c)
            continue
        entry = dict(c)
        entry["class"] = CONTENT_GAP
        entry["structural_override"] = "schema_labeling"
        entry["original_class"] = ENGLISH_GAP
        entry["structural_override_confidence"] = signal.confidence
        entry["structural_override_reason"] = signal.reason
        entry.pop("contradicted_stem_fact", None)
        new_list.append(entry)
        override_count += 1
    return new_list, override_count
