"""numeric_overlap fixer — Phase A6.

When the numeric_overlap detector fires at T2/T3 (T4 is advisory only),
the stem's numeric value falls in an overlap zone where multiple
options' instruments could legitimately apply. The fix: tighten the
stem's numeric value to a band that's UNAMBIGUOUSLY in one
instrument's range.

Strategy:
  1. Read the keyed (correct) option's referenced instrument.
  2. Look up that instrument's age range from the canonical table.
  3. Compute a value mid-range (avoiding boundaries).
  4. Replace the stem's age value with the new unambiguous value.

This is the only fixer that modifies the stem (the others fix only
the flagged distractor). The plan explicitly authorizes stem-rewrite
for numeric_overlap signature.

T4 questions never reach this fixer because A2.5's cell config makes
numeric_overlap ADVISORY at T4 (overlap zones can be the legitimate
analysis test at T4). This fixer's `handles_signatures` is intersected
with the actual detector verdict at dispatch time — only T2/T3 BLOCK
signals hit here.
"""
from __future__ import annotations

import re

from pipeline.fixers import Fixer
from pipeline.detectors import DetectorSignal
from pipeline.detectors.numeric_overlap import _INSTRUMENT_AGE_RANGES, _instruments_in


def _correct_option_instruments(question: dict) -> list[str]:
    """Find named instruments in the keyed (correct) option's text.
    Returns lowercase instrument keys."""
    for opt in question.get("options") or []:
        if opt.get("is_correct"):
            return _instruments_in(opt.get("text", "") or "")
    return []


def _replace_age_in_stem(stem: str, new_age_years: float) -> str | None:
    """Replace the first age value in the stem with `new_age_years`
    (rendered as integer years). Returns None if no replaceable age
    pattern found."""
    new_int = int(round(new_age_years))

    # Try in order: "16-year-old" → "<N>-year-old"
    pat = re.compile(r"\b(\d{1,3})[-\s]year[-\s]old\b", re.IGNORECASE)
    m = pat.search(stem)
    if m:
        return pat.sub(f"{new_int}-year-old", stem, count=1)

    # "age 16" → "age <N>"
    pat = re.compile(r"\b(age|aged|ages)\s+\d{1,3}\b", re.IGNORECASE)
    m = pat.search(stem)
    if m:
        verb = m.group(1)
        return pat.sub(f"{verb} {new_int}", stem, count=1)

    # "16:11" notation → keep year, drop month suffix to land mid-range
    pat = re.compile(r"\b\d{1,2}:\d{1,2}\b")
    m = pat.search(stem)
    if m:
        new_year = new_int
        new_month = 6  # mid-year as a stable choice
        return pat.sub(f"{new_year}:{new_month:02d}", stem, count=1)

    return None


class NumericOverlapFixer(Fixer):
    fixer_id = "numeric_overlap_fixer"
    handles_signatures = ("numeric_overlap",)

    async def fix(
        self,
        client,
        question: dict,
        signal: DetectorSignal,
        semaphore,
    ) -> dict:
        question = question or {}

        # T4 is advisory only at the detector level — but defense in depth.
        if question.get("difficulty_tier") == 4:
            return question

        instruments = _correct_option_instruments(question)
        if not instruments:
            return question

        # Pick the first canonical (non-alias) instrument range. Aliases
        # like "wppsi"/"wisc" point at the same range as their canonical
        # forms; either works.
        rng = None
        chosen_inst = None
        for inst in instruments:
            r = _INSTRUMENT_AGE_RANGES.get(inst)
            if r is None:
                continue
            rng = r
            chosen_inst = inst
            break
        if rng is None:
            return question

        lo, hi = rng
        # Pick mid-range, then move to the next integer year if mid-range
        # is too close to a boundary (within 0.5 years).
        midpoint = (lo + hi) / 2.0
        # Snap to integer year for stem readability.
        new_age = round(midpoint)
        # Ensure new_age is well within the range with margin.
        if new_age - lo < 1.0:
            new_age = int(lo + 2)
        if hi - new_age < 1.0:
            new_age = int(hi - 2)

        new_stem = _replace_age_in_stem(
            question.get("question_stem", "") or "",
            new_age,
        )
        if new_stem is None:
            return question

        new_q = dict(question)
        new_q["question_stem"] = new_stem
        new_q["_routed_fixer"] = (
            f"numeric_overlap:stem_age_to_{new_age}_for_{chosen_inst}"
        )
        return new_q


__all__ = ["NumericOverlapFixer"]
