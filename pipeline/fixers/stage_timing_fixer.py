"""stage_timing fixer — Phase B2.

Handles `stage_timing` signature from `english_gap_scanner`. The
detector fires when stem mentions one developmental stage (childhood)
and a distractor mentions the OPPOSITE stage (adulthood). Pure lexical
contradiction.

Strategy: deterministic regex flip — replace the distractor's stage
term with the stem's stage term. Mirrors `laterality_fixer.py`'s
pattern. No LLM call needed.

Stage pairs come from `pipeline.english_gap_scanner.STAGE_PAIRS`
(single source of truth — the detector and fixer share the same list).
"""
from __future__ import annotations

import re

from pipeline.fixers import Fixer
from pipeline.detectors import DetectorSignal
from pipeline.english_gap_scanner import STAGE_PAIRS


def _word_present(text: str, token: str) -> bool:
    if not text or not token:
        return False
    pat = r"\b" + re.escape(token) + r"\b"
    return re.search(pat, text, re.IGNORECASE) is not None


def _find_stem_stage(stem: str) -> tuple[str, str] | None:
    """Return (stem_stage, opposite_stage) for the stage the stem
    asserts EXCLUSIVELY (one of the pair present, the other not).
    Returns None if no exclusive stage found (stem has both or neither)."""
    for a, b in STAGE_PAIRS:
        a_in = _word_present(stem, a)
        b_in = _word_present(stem, b)
        if a_in and not b_in:
            return a, b
        if b_in and not a_in:
            return b, a
    return None


def _replace_word(text: str, old: str, new: str) -> str:
    """Replace `old` with `new` on word boundaries, case-insensitively,
    preserving the case of the FIRST letter where reasonable."""
    pat = re.compile(r"\b" + re.escape(old) + r"\b", re.IGNORECASE)

    def _sub(m):
        original = m.group(0)
        if original[:1].isupper():
            return new[:1].upper() + new[1:]
        return new

    return pat.sub(_sub, text)


class StageTimingFixer(Fixer):
    """Deterministic fixer: flip the distractor's stage term to match
    the stem's exclusive stage."""

    fixer_id = "stage_timing_fixer"
    handles_signatures = ("stage_timing",)

    async def fix(
        self,
        client,
        question: dict,
        signal: DetectorSignal,
        semaphore,
    ) -> dict:
        question = question or {}
        letter = signal.letter
        if not letter:
            return question

        stem = question.get("question_stem", "") or ""
        pair = _find_stem_stage(stem)
        if pair is None:
            # Stem doesn't have an exclusive stage — can't flip safely.
            return question

        stem_stage, opposite_stage = pair

        options = list(question.get("options") or [])
        target_idx = next(
            (i for i, o in enumerate(options) if o.get("letter") == letter),
            None,
        )
        if target_idx is None:
            return question

        target_opt = options[target_idx]
        if target_opt.get("is_correct"):
            return question

        original_text = target_opt.get("text", "") or ""
        if not _word_present(original_text, opposite_stage):
            # Distractor doesn't actually contain the expected opposite —
            # detector's claim doesn't match. Defensive return.
            return question

        new_text = _replace_word(original_text, opposite_stage, stem_stage)
        # Same edit on the explanation — keeps the trace consistent
        original_explanation = target_opt.get("explanation", "") or ""
        new_explanation = (
            _replace_word(original_explanation, opposite_stage, stem_stage)
            if original_explanation else ""
        )

        new_options = list(options)
        new_options[target_idx] = dict(target_opt)
        new_options[target_idx]["text"] = new_text
        if new_explanation:
            new_options[target_idx]["explanation"] = new_explanation
        new_options[target_idx]["_routed_fixer"] = "stage_timing:deterministic_flip"

        new_q = dict(question)
        new_q["options"] = new_options
        return new_q


__all__ = ["StageTimingFixer"]
