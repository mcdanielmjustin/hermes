"""numeric_ratio fixer — Phase B2.

Handles `numeric_ratio` signature from `english_gap_scanner`. The
detector fires when the stem prints a specific ratio (e.g. "2:1") and
a distractor prints a CONFLICTING ratio (e.g. "3:1"). The student
rejects this without concept knowledge — pure lexical contradiction.

Strategy: rewrite the distractor to express a wrong-but-plausible
NON-NUMERIC claim. Drop the conflicting ratio entirely. Sonnet
single-call with narrow prompt + UQ guard (don't introduce universal
quantifiers in the rewrite).

Per the plan, this is the simpler of two strategies. The alternative
(adjust the stem to remove the printed ratio) is more invasive — keeps
all distractors but rewrites stem. We use distractor-rewrite because
A6's design contract is "touch only the flagged option."
"""
from __future__ import annotations

import re

from pipeline.fixers import Fixer
from pipeline.fixers._helpers import parse_fixer_json
from pipeline.detectors import DetectorSignal


_SONNET_MODEL_ID = "claude-sonnet-4-6"

# UQ guard: reject rewrites that introduce universal quantifiers
# (would create a different english_gap pattern).
_UQ_RE = re.compile(
    r"\b(all|every|always|never|throughout|entire|any|none|no)\b",
    re.IGNORECASE,
)

# Numeric ratio pattern: matches "2:1", "10:3", etc. Used to detect
# whether the rewritten distractor contains a ratio (we want to reject
# rewrites that just substitute one wrong ratio for another — the
# whole point is to avoid the lexical contradiction at all).
_RATIO_RE = re.compile(r"\b\d+:\d+\b")


_FIX_PROMPT = """You are minimally repairing one distractor in a multiple-choice question. The original distractor printed a specific NUMERIC RATIO ("{distractor_ratio}") that DIRECTLY CONTRADICTS a ratio printed in the stem ("{stem_ratio}"). A student can reject this distractor by reading the stem alone — no concept knowledge needed. That's an "english_gap" failure.

Your task: rewrite the distractor to express a WRONG-BUT-PLAUSIBLE claim WITHOUT printing any numeric ratio. The distractor should be wrong because of a concept misunderstanding, not because the printed numbers contradict the stem.

ABSOLUTE PRESERVATION RULES (violating these is worse than the original problem):
- Do NOT change which option is correct. Letter {letter} stays as a distractor.
- Do NOT print any numeric ratio (X:Y format) in the rewrite. Drop the conflicting ratio entirely.
- Do NOT add universal quantifiers (all, every, always, never, throughout, entire, any, none, no).
- Do NOT introduce new lexical contradictions with the stem.
- Preserve the misconception being tested — refine the EXPRESSION, not the target concept.
- All four options remain. Only the text of letter {letter} may change.

QUESTION STEM:
{stem}

ALL OPTIONS (you may only modify {letter}):
{options_block}

THE FLAGGED OPTION ({letter}):
"{distractor_text}"

OUTPUT — single JSON object (no preamble, no markdown):

{{
  "letter": "{letter}",
  "new_text": "<rewritten distractor without any numeric ratio>",
  "new_explanation": "<rewritten explanation, 1-2 sentences>"
}}"""


def _extract_ratios(text: str) -> list[str]:
    """Return all 'X:Y' ratios in the text."""
    if not text:
        return []
    return _RATIO_RE.findall(text)


class NumericRatioFixer(Fixer):
    """Fixes numeric_ratio english_gap by rewriting the distractor to
    drop the conflicting numeric ratio."""

    fixer_id = "numeric_ratio_fixer"
    handles_signatures = ("numeric_ratio",)

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
        stem_ratios = _extract_ratios(stem)
        if not stem_ratios:
            # Detector said numeric_ratio fired but we can't find a ratio
            # in the stem — defensive return. Shouldn't happen.
            return question

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
        distractor_ratios = _extract_ratios(original_text)
        if not distractor_ratios:
            # Detector's claim doesn't match — distractor doesn't contain
            # a ratio. Defensive return.
            return question

        # Build options block for context.
        options_block_lines = []
        for o in options:
            mark = "[CORRECT]" if o.get("is_correct") else "[distractor]"
            options_block_lines.append(
                f"  {o.get('letter','?')} {mark}: {o.get('text','')}"
            )
        options_block = "\n".join(options_block_lines)

        prompt = _FIX_PROMPT.format(
            letter=letter,
            stem=stem,
            stem_ratio=stem_ratios[0],
            distractor_ratio=distractor_ratios[0],
            options_block=options_block,
            distractor_text=original_text,
        )

        async with semaphore:
            try:
                response = await client.messages.create(
                    model=_SONNET_MODEL_ID,
                    max_tokens=512,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text if response.content else ""
            except Exception:
                return question

        parsed = parse_fixer_json(text)
        if parsed is None:
            return question

        new_text = (parsed.get("new_text") or "").strip()
        new_explanation = (parsed.get("new_explanation") or "").strip()
        if not new_text:
            return question

        # Invariant guards:
        # - Reject rewrites that introduce a universal quantifier
        # - Reject rewrites that still contain a numeric ratio (defeats
        #   the whole purpose of this fixer)
        if _UQ_RE.search(new_text):
            return question
        if _RATIO_RE.search(new_text):
            return question

        new_options = list(options)
        new_options[target_idx] = dict(target_opt)
        new_options[target_idx]["text"] = new_text
        if new_explanation:
            new_options[target_idx]["explanation"] = new_explanation
        new_options[target_idx]["_routed_fixer"] = "numeric_ratio:llm_rewrite"

        new_q = dict(question)
        new_q["options"] = new_options
        return new_q


__all__ = ["NumericRatioFixer"]
