"""laterality fixer — Phase A6.

Deterministic-only fix: flips the inverted laterality term in the
flagged distractor back to match the stem. Zero LLM calls.

Strategy:
  1. Identify the laterality term in the stem (left / right, bilateral /
     unilateral, ipsilateral / contralateral).
  2. The flagged distractor contains the OPPOSITE term (per detector).
  3. Replace the opposite term with the stem's term.
  4. Validate: re-running the english_gap scanner's _check_laterality
     on the patched distractor should NOT fire.

If the deterministic edit can't proceed (stem has both terms, or
distractor doesn't contain the expected opposite), return the original
question unchanged.
"""
from __future__ import annotations

import re

from pipeline.fixers import Fixer
from pipeline.detectors import DetectorSignal
from pipeline.english_gap_scanner import LATERAL_PAIRS


def _word_present(text: str, token: str) -> bool:
    if not text or not token:
        return False
    pat = r"\b" + re.escape(token) + r"\b"
    return re.search(pat, text, re.IGNORECASE) is not None


def _find_stem_laterality(stem: str) -> tuple[str, str] | None:
    """Return (stem_term, opposite_term) for the laterality the stem
    asserts EXCLUSIVELY (one of the pair present, the other not).
    Returns None if no exclusive laterality found."""
    for a, b in LATERAL_PAIRS:
        a_in = _word_present(stem, a)
        b_in = _word_present(stem, b)
        if a_in and not b_in:
            return a, b
        if b_in and not a_in:
            return b, a
    return None


def _replace_word(text: str, old: str, new: str) -> str:
    """Replace `old` with `new` in `text` on word boundaries,
    case-insensitively, preserving the original case of the FIRST letter
    where reasonable."""
    pat = re.compile(r"\b" + re.escape(old) + r"\b", re.IGNORECASE)

    def _sub(m):
        original = m.group(0)
        # If original was capitalized, capitalize the replacement.
        if original[:1].isupper():
            return new[:1].upper() + new[1:]
        return new

    return pat.sub(_sub, text)


class LateralityFixer(Fixer):
    fixer_id = "laterality_fixer"
    handles_signatures = ("laterality",)

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
        pair = _find_stem_laterality(stem)
        if pair is None:
            return question

        stem_term, opposite_term = pair

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
        if not _word_present(original_text, opposite_term):
            return question

        new_text = _replace_word(original_text, opposite_term, stem_term)
        # Same edit on the explanation (if any), so the trace still
        # reflects the corrected laterality.
        original_explanation = target_opt.get("explanation", "") or ""
        new_explanation = _replace_word(original_explanation, opposite_term, stem_term)

        new_options = list(options)
        new_options[target_idx] = dict(target_opt)
        new_options[target_idx]["text"] = new_text
        if new_explanation:
            new_options[target_idx]["explanation"] = new_explanation
        new_options[target_idx]["_routed_fixer"] = "laterality:deterministic_flip"

        new_q = dict(question)
        new_q["options"] = new_options
        return new_q


__all__ = ["LateralityFixer"]
