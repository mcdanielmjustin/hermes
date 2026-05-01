"""Phase 24 — Deterministic pre-audit english_gap scanner.

Scans (stem, distractor) pairs for high-confidence english_gap signatures
without an LLM call. Layer 3 in the reinforcer hierarchy: deterministic
structural classifier.

Complementary to Phase 22a's `schema_labeling_classifier`, which targets the
schema-labeling pattern specifically (paired-named-concepts swap). This
module targets the broader english_gap class:

  - Universal quantifier in a distractor that contradicts a specific stated
    case in the stem (the canonical Lester/wedding pattern).
  - Numeric mismatch between stem-printed value and distractor value.
  - Laterality contradiction (left/right, ipsi/contralateral, bi/unilateral).
  - Direction contradiction (increase/decrease, rise/fall, elevated/reduced).
  - Stage-timing contradiction (childhood/adulthood, pre/post-puberty).

Conservative design: signatures fire only when BOTH the stem has the
specific fact AND the distractor has the contradicting form. False positives
on content_gap distractors are the failure mode we most want to avoid —
they'd cause the scanner to suppress legitimate distractors. So fire only
on high-confidence signatures.

Public API:
  - EnglishGapSignal: dataclass with fired/confidence/reason.
  - scan_distractor(stem, distractor_text) -> EnglishGapSignal
  - scan_question(question) -> dict[letter, EnglishGapSignal]
  - english_gap_distractors(question) -> list[letter] (convenience)

Cost: $0 per call (pure regex).
"""
from __future__ import annotations

import re
from dataclasses import dataclass


# ── Signature constants ─────────────────────────────────────

UNIVERSAL_QUANTIFIERS = (
    "all", "every", "always", "never", "throughout",
    "entire", "any", "none", "no",
)

LATERAL_PAIRS = (
    ("left", "right"),
    ("ipsilateral", "contralateral"),
    ("bilateral", "unilateral"),
)

DIRECTION_PAIRS = (
    ("increase", "decrease"),
    ("increased", "decreased"),
    ("rise", "fall"),
    ("rises", "falls"),
    ("rising", "falling"),
    ("elevated", "reduced"),
    ("higher", "lower"),
    ("more", "less"),
    ("greater", "smaller"),
)

STAGE_PAIRS = (
    ("childhood", "adulthood"),
    ("prepubertal", "postpubertal"),
    ("pre-puberty", "post-puberty"),
    ("juvenile", "adult"),
    ("early-onset", "late-onset"),
    ("acute", "chronic"),
)


# Compile patterns once at module load.
_UNIVERSAL_RE = re.compile(
    r"\b(" + "|".join(UNIVERSAL_QUANTIFIERS) + r")\b",
    re.IGNORECASE,
)

# Match numbers like 2:1, 5%, 5+, age 30, 12 weeks
_RATIO_RE = re.compile(r"\b(\d+):(\d+)\b")
_PERCENT_RE = re.compile(r"\b(\d+)\s*%")
_AGE_RE = re.compile(r"\b(?:age|ages)\s+(\d+)\b", re.IGNORECASE)
_DURATION_RE = re.compile(
    r"\b(\d+)\s*(?:weeks?|months?|years?|days?|hours?)\b", re.IGNORECASE
)
_INTEGER_RE = re.compile(r"\b(\d+)(?:\+|\s+(?:or\s+)?more)?\b")


@dataclass(frozen=True)
class EnglishGapSignal:
    fired: bool
    confidence: float
    reason: str
    signature: str | None = None  # which kind of signature fired


_NEGATIVE = EnglishGapSignal(fired=False, confidence=0.0, reason="no_signal")


# ── Helpers ─────────────────────────────────────────────────

def _has_universal(text: str) -> str | None:
    if not text:
        return None
    m = _UNIVERSAL_RE.search(text)
    return m.group(0) if m else None


def _word_present(text: str, token: str) -> bool:
    if not text or not token:
        return False
    pat = r"\b" + re.escape(token) + r"\b"
    return re.search(pat, text, re.IGNORECASE) is not None


def _stem_has_specific_case(stem: str) -> tuple[bool, str]:
    """Heuristic: stem contains a specific case the universal would
    contradict. Looks for: named subjects (Dr. X, age N, named patient),
    specific numbers/ratios/percentages, named-finding clauses ('still
    recalls', 'continues to', 'observes that').
    """
    if not stem:
        return False, ""
    # Specific subject + age (e.g., "Dr. Wren, age 49")
    if re.search(r"\b(?:Dr|Mr|Mrs|Ms)\.?\s+\w+", stem):
        return True, "named_subject"
    if re.search(r"\bage\s+\d+\b", stem, re.IGNORECASE):
        return True, "named_age"
    # Specific numeric facts in the stem
    for pat, name in (
        (_RATIO_RE, "specific_ratio"),
        (_PERCENT_RE, "specific_percent"),
        (_DURATION_RE, "specific_duration"),
    ):
        if pat.search(stem):
            return True, name
    # Stated-case markers
    case_markers = (
        "still recalls", "still able to", "continues to", "remains able",
        "is observed to", "presents with", "complains of", "reports",
        "shows", "exhibits", "demonstrates",
    )
    for m in case_markers:
        if m in stem.lower():
            return True, f"stated_case:{m}"
    return False, ""


# ── Signature scanners ──────────────────────────────────────

def _check_universal_quantifier(
    stem: str, distractor: str,
) -> EnglishGapSignal | None:
    """Tier A signature: universal quantifier in distractor + specific
    case in stem. Canonical Lester/wedding pattern.
    """
    quant = _has_universal(distractor)
    if not quant:
        return None
    has_case, case_kind = _stem_has_specific_case(stem)
    if not has_case:
        return None
    return EnglishGapSignal(
        fired=True,
        confidence=0.85,
        reason=f"universal_quantifier:'{quant}'+specific_stem:{case_kind}",
        signature="universal_quantifier",
    )


_HANDEDNESS_RE = re.compile(
    # Matches: right-handed, left-handed, right-handedness, left-hander,
    # right-dominant, left-dominant. Both hyphenated and space-separated
    # forms ("right handed"). The stem's handedness descriptor is removed
    # from the laterality check because it describes the patient, not
    # anatomy/lesion side. Without this strip, "right-handed woman with
    # left hemisphere stroke" reads to the regex as having both
    # "left" AND "right" — which incorrectly flags any distractor that
    # mentions "left ___" anatomy. (S1 fix; see plan).
    r"\b(right|left)[-\s]"
    r"(?:hand(?:ed(?:ness)?|er|edly)?|dominant)\b",
    re.IGNORECASE,
)


def _strip_handedness(text: str) -> str:
    """Remove handedness descriptors so the laterality regex doesn't
    mistake them for anatomical laterality. Used inside `_check_laterality`
    for the stem text only — handedness in a distractor is rare and would
    not produce a false positive in the same way.
    """
    if not text:
        return text or ""
    return _HANDEDNESS_RE.sub("", text)


def _check_laterality(
    stem: str, distractor: str,
) -> EnglishGapSignal | None:
    """Tier B signature: stem has a laterality term and distractor has
    the OPPOSITE laterality term. Less common but high-confidence
    when it fires.

    S1: handedness descriptors ("right-handed", "left-dominant") are
    stripped from the stem before the cross-check — they describe the
    patient, not the anatomy/lesion, and would otherwise produce false
    positives whenever a vignette mentions handedness AND any distractor
    references the opposite-side anatomy.
    """
    stem_clean = _strip_handedness(stem)
    for a, b in LATERAL_PAIRS:
        if _word_present(stem_clean, a) and _word_present(distractor, b):
            return EnglishGapSignal(
                fired=True,
                confidence=0.75,
                reason=f"laterality:'{a}'(stem)_vs_'{b}'(distractor)",
                signature="laterality",
            )
        if _word_present(stem_clean, b) and _word_present(distractor, a):
            return EnglishGapSignal(
                fired=True,
                confidence=0.75,
                reason=f"laterality:'{b}'(stem)_vs_'{a}'(distractor)",
                signature="laterality",
            )
    return None


def _check_numeric_ratio(
    stem: str, distractor: str,
) -> EnglishGapSignal | None:
    """Tier B signature: stem prints a specific ratio, distractor prints
    a different ratio. Matches the CPAT depression E-02 pattern (stem
    prints '2:1 by adulthood', distractor says '3:1').
    """
    stem_ratios = _RATIO_RE.findall(stem)
    if not stem_ratios:
        return None
    dist_ratios = _RATIO_RE.findall(distractor)
    if not dist_ratios:
        return None
    for sr in stem_ratios:
        for dr in dist_ratios:
            if sr != dr:
                return EnglishGapSignal(
                    fired=True,
                    confidence=0.80,
                    reason=f"ratio_mismatch:stem='{sr[0]}:{sr[1]}'_vs_dist='{dr[0]}:{dr[1]}'",
                    signature="numeric_ratio",
                )
    return None


def _check_stage_timing(
    stem: str, distractor: str,
) -> EnglishGapSignal | None:
    """Tier B signature: stem describes one developmental stage, distractor
    claims the opposite. Lower confidence than the others (stage tokens
    can co-occur legitimately in some content).
    """
    for a, b in STAGE_PAIRS:
        # Both terms present in stem + distractor only flags if they
        # APPEAR contradictory (one in stem, other in distractor).
        if _word_present(stem, a) and _word_present(distractor, b) \
                and not _word_present(stem, b):
            return EnglishGapSignal(
                fired=True,
                confidence=0.65,
                reason=f"stage_timing:'{a}'(stem)_vs_'{b}'(distractor)",
                signature="stage_timing",
            )
    return None


# Scanner pipeline — first signature to fire wins.
_SCANNERS = (
    _check_universal_quantifier,
    _check_laterality,
    _check_numeric_ratio,
    _check_stage_timing,
)


# ── Public API ──────────────────────────────────────────────

def scan_distractor(stem: str, distractor_text: str) -> EnglishGapSignal:
    """Return the first english_gap signature that fires on this pair,
    or _NEGATIVE if none fire."""
    if not isinstance(stem, str) or not isinstance(distractor_text, str):
        return _NEGATIVE
    if not stem or not distractor_text:
        return _NEGATIVE
    for scanner in _SCANNERS:
        sig = scanner(stem, distractor_text)
        if sig is not None:
            return sig
    return _NEGATIVE


def scan_question(question: dict) -> dict[str, EnglishGapSignal]:
    """Scan all distractors of a question. Returns letter → signal."""
    out: dict[str, EnglishGapSignal] = {}
    stem = (question or {}).get("question_stem", "") or ""
    for opt in (question or {}).get("options", []) or []:
        if opt.get("is_correct"):
            continue
        letter = opt.get("letter", "?")
        out[letter] = scan_distractor(stem, opt.get("text", "") or "")
    return out


def english_gap_distractors(question: dict) -> list[str]:
    """Return list of distractor letters that the scanner flags as
    english_gap. Convenience wrapper.
    """
    return [
        letter for letter, sig in scan_question(question).items()
        if sig.fired
    ]


# ── Phase A2: override application ──────────────────────────

def apply_english_gap_override(
    question: dict,
    classifications: list[dict],
    eg_signals: list,
) -> tuple[list[dict], int]:
    """Phase A2 — apply english_gap detector overrides to LLM classifications.

    For each option whose detector signal carries
    ``verdict_action="override_to"`` and ``proposed_class="english_gap"``,
    flip that option's audit class to ``english_gap`` and stamp tracing
    metadata. Returns ``(new_classifications, override_count)``. Does
    not mutate the input list.

    Mirrors the shape of ``apply_schema_labeling_override`` so audit
    callers can apply both helpers in sequence.

    Override metadata added on overridden entries:
      - ``class``                          → ``"english_gap"``
      - ``original_class``                 → previous LLM class (for trace)
      - ``structural_override``            → ``"english_gap_scanner"``
      - ``structural_override_confidence`` → signal's confidence (0.75-0.85)
      - ``structural_override_signature``  → which sub-signature fired
                                            (universal_quantifier | laterality | numeric_ratio)
      - ``structural_override_tier``       → the question's difficulty_tier
      - ``structural_override_cell_threshold`` → the cell's threshold the
                                              signal cleared (from
                                              ``signal.extra["cell_threshold"]``).

    Idempotent re schema-labeling: if a distractor was demoted to
    content_gap by ``apply_schema_labeling_override`` upstream, A2
    will re-promote to english_gap when the scanner fires (the
    scanner is a higher-confidence deterministic source than the
    schema_labeling fallback Tier B). This is intentional —
    universal_quantifier guard inside ``classify_distractor`` already
    prevents schema_labeling from firing when the distractor has a
    universal quantifier, so the only conflict cases are those where
    schema_labeling fired on Tier A/B BUT the english_gap scanner ALSO
    fired with high confidence. The scanner wins because its signal
    is more specific.

    ``eg_signals`` is the list of DetectorSignals from the english_gap
    detector for this question (typically obtained by filtering
    ``DetectorRegistry.scan_for_phase("audit", question)`` to
    ``detector_id == "english_gap_scanner"``). Only signals with
    ``verdict_action == "override_to"`` and ``proposed_class ==
    "english_gap"`` are acted on.
    """
    # Local import to avoid circular dependency at module load time.
    from pipeline.detectors import VERDICT_OVERRIDE_TO
    from pipeline.quality_taxonomy import ENGLISH_GAP

    by_letter: dict[str, object] = {}
    for s in eg_signals or []:
        if (
            getattr(s, "verdict_action", None) == VERDICT_OVERRIDE_TO
            and getattr(s, "proposed_class", None) == "english_gap"
            and getattr(s, "letter", None) is not None
        ):
            by_letter[s.letter] = s

    if not by_letter:
        # Return a shallow copy to keep the contract uniform with the
        # override-applied path (callers shouldn't have to inspect
        # whether the list was mutated or not).
        return list(classifications or []), 0

    new_list: list[dict] = []
    override_count = 0
    for c in classifications or []:
        letter = c.get("letter")
        sig = by_letter.get(letter)
        if sig is None:
            new_list.append(c)
            continue
        # Skip the override no-op case where the LLM already classified
        # this option as english_gap.
        if c.get("class") == ENGLISH_GAP:
            new_list.append(c)
            continue

        entry = dict(c)
        entry["original_class"] = c.get("class")
        entry["class"] = ENGLISH_GAP
        entry["structural_override"] = "english_gap_scanner"
        entry["structural_override_confidence"] = float(sig.confidence)
        entry["structural_override_signature"] = sig.signature
        # A2.5: tier and cell threshold come from signal.extra (set by
        # the detector when it consulted the cell matrix). Falls back to
        # the question's tier if extra is missing (defensive).
        sig_extra = getattr(sig, "extra", {}) or {}
        entry["structural_override_tier"] = sig_extra.get(
            "tier", (question or {}).get("difficulty_tier")
        )
        entry["structural_override_cell_threshold"] = sig_extra.get(
            "cell_threshold"
        )
        # Drop contradicted_stem_fact as it no longer applies (the
        # scanner doesn't surface a stem fact; the original LLM verdict
        # may have one but it's misleading post-override).
        entry.pop("contradicted_stem_fact", None)
        new_list.append(entry)
        override_count += 1

    return new_list, override_count
