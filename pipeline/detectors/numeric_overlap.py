"""numeric_overlap detector — Phase A5a.

Detects WISC age-band-style overlap zones where:
  - Stem mentions a numeric value (age, score, threshold)
  - Multiple options reference named instruments / categories whose
    valid ranges OVERLAP with that value

The canonical case from today's E2E-100 D8 anchor: stem says "below
WISC-V floor by one month"; options A (WAIS-IV) and D (WISC-V) both
have valid age bands that overlap at 16:0–16:11. The keyed answer is
defensible AND so is the alternative — ambiguity, not a clean test.

Tier behavior:
  - T2/T3: VERDICT_BLOCK on fire (over-flag a content_gap is acceptable
    at lower tiers because the resolution is "clarify the stem")
  - T4: VERDICT_ADVISORY on fire (overlap zones can be the LEGITIMATE
    test at evaluate-tier — clinical judgment under boundary cases is
    a real competency)

A5 ships an OPINIONATED instrument-range table (WPPSI/WISC/WAIS at
canonical age bands). Future work could mine more ranges from the
corpus (DSM age criteria, CBCL/Achenbach subscales, etc.).

Confidence 0.7 — the regex match is reliable but the SEMANTIC
ambiguity claim depends on the specific instrument ranges; future
expansions should include their own range tables.
"""
from __future__ import annotations

import re

from . import (
    Detector,
    DetectorSignal,
    PHASE_AUDIT,
    PHASE_GENERATION,
    VERDICT_ADVISORY,
    VERDICT_BLOCK,
)


# Canonical instrument age ranges (years.months → decimal years).
# Sourced from publisher manuals; see plan A5 step-by-step. Decimal is
# (years + months/12) so 16:11 → 16.92.
_INSTRUMENT_AGE_RANGES: dict[str, tuple[float, float]] = {
    "wppsi-iv": (2.5, 7.58),
    "wppsi": (2.5, 7.58),    # generic alias
    "wisc-v": (6.0, 16.92),
    "wisc": (6.0, 16.92),    # generic alias
    "wais-iv": (16.0, 90.92),
    "wais": (16.0, 90.92),
    "wiat-iii": (4.0, 50.92),
    "wiat": (4.0, 50.92),
}


def _extract_age_in_years(text: str) -> list[float]:
    """Extract every age value in `text` as decimal years.

    Patterns:
      - "age 16" / "ages 16" / "aged 16"
      - "16-year-old" / "16 years old"
      - "16:11" (years:months notation; converts to decimal)
      - "16 years, 11 months" / "16 years 11 months"
    """
    ages: list[float] = []

    # 16:11 notation
    for m in re.finditer(r"\b(\d{1,2}):(\d{1,2})\b", text):
        years = int(m.group(1))
        months = int(m.group(2))
        if 0 <= months < 12 and 0 < years < 110:
            ages.append(years + months / 12.0)

    # "age N" / "aged N" / "ages N"
    for m in re.finditer(r"\b(?:age|aged|ages)\s+(\d{1,3})\b", text, re.IGNORECASE):
        ages.append(float(m.group(1)))

    # "N-year-old" / "N year old"
    for m in re.finditer(
        r"\b(\d{1,3})[-\s]year[-\s]old\b", text, re.IGNORECASE,
    ):
        ages.append(float(m.group(1)))

    # "N years, M months" / "N years M months"
    for m in re.finditer(
        r"\b(\d{1,3})\s+years?\s*,?\s*(\d{1,2})\s+months?\b",
        text,
        re.IGNORECASE,
    ):
        years = int(m.group(1))
        months = int(m.group(2))
        if 0 <= months < 12:
            ages.append(years + months / 12.0)

    return ages


def _instruments_in(text: str) -> list[str]:
    """Find named instruments in `text`. Returns lowercase canonical
    keys. Handles uppercase variants like "WISC-V" → "wisc-v"."""
    out: list[str] = []
    text_lower = (text or "").lower()
    for key in _INSTRUMENT_AGE_RANGES:
        if re.search(r"\b" + re.escape(key) + r"\b", text_lower):
            out.append(key)
    return out


def _option_supports_age(age: float, instrument: str) -> bool:
    """True if the instrument's age range covers `age`."""
    rng = _INSTRUMENT_AGE_RANGES.get(instrument)
    if not rng:
        return False
    lo, hi = rng
    return lo <= age <= hi


class NumericOverlapDetector(Detector):
    """Fires when ≥2 distractors reference instruments whose age ranges
    overlap with a stem-stated age — the keyed answer is defensible
    AND so is the alternative.

    Detection logic (per question):
      1. Extract stem ages
      2. For each option, find named instruments
      3. For each pair of options, check if both reference instruments
         that include any of the stem's ages
      4. If ≥2 options each support a stem age via their referenced
         instrument: fire on the question (letter=None — set-level
         ambiguity, not per-letter)
    """

    detector_id = "numeric_overlap"
    phases = (PHASE_GENERATION, PHASE_AUDIT)

    def scan(
        self,
        question: dict,
        context: dict | None = None,
    ) -> list[DetectorSignal]:
        question = question or {}
        stem = question.get("question_stem", "") or ""
        stem_ages = _extract_age_in_years(stem)
        if not stem_ages:
            return [DetectorSignal(
                detector_id=self.detector_id,
                letter=None,
                fired=False,
                confidence=0.0,
                signature=None,
                verdict_action=VERDICT_ADVISORY,
                reason="no age value in stem",
            )]

        # Find options whose referenced instruments could support a stem age.
        # Track UNIQUE LETTERS (not [letter, instrument, age] tuples), so
        # an option naming both "wisc-v" and the alias "wisc" counts as
        # one supporting option.
        supporting: list[tuple[str, str, float]] = []  # (letter, instrument, age)
        supporting_letters: set[str] = set()
        for opt in question.get("options") or []:
            letter = opt.get("letter", "?")
            text = opt.get("text", "") or ""
            instruments = _instruments_in(text)
            matched_for_letter = False
            for inst in instruments:
                if matched_for_letter:
                    break  # one supporting instrument per letter is enough
                for age in stem_ages:
                    if _option_supports_age(age, inst):
                        supporting.append((letter, inst, age))
                        supporting_letters.add(letter)
                        matched_for_letter = True
                        break

        if len(supporting_letters) < 2:
            return [DetectorSignal(
                detector_id=self.detector_id,
                letter=None,
                fired=False,
                confidence=0.0,
                signature=None,
                verdict_action=VERDICT_ADVISORY,
                reason=(
                    f"only {len(supporting_letters)} option(s) support stem age "
                    f"via named instrument"
                ),
            )]

        # Tier-conditional verdict: T2/T3 BLOCK; T4 ADVISORY.
        tier = question.get("difficulty_tier")
        if tier == 4:
            verdict = VERDICT_ADVISORY
        else:
            verdict = VERDICT_BLOCK

        return [DetectorSignal(
            detector_id=self.detector_id,
            letter=None,
            fired=True,
            confidence=0.7,
            signature="numeric_overlap",
            verdict_action=verdict,
            reason=(
                f"{len(supporting_letters)} options reference instruments whose "
                f"age ranges include stem age — overlap zone ambiguity"
            ),
            extra={"supporting_options": supporting, "stem_ages": stem_ages},
        )]


__all__ = ["NumericOverlapDetector"]
