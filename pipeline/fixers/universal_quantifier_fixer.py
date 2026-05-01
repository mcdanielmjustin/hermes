"""universal_quantifier fixer — Phase A6.

Removes a universal-quantifier distractor's offending word and asks
Sonnet for a minimal targeted rewrite that turns it from a universal
denial into a wrong-but-plausible specific claim.

Strategy:
  1. Drop the universal quantifier word deterministically (no LLM).
  2. If the resulting distractor still makes sense (>=10 words remain),
     keep it as-is (no LLM call needed).
  3. Otherwise, ask Sonnet to rewrite, with a NARROW prompt that:
     - Names the offending word
     - Names the stem fact the original universal contradicted
     - Forbids introducing new lexical contradictions or universal
       quantifiers
     - Preserves all other distractors as context

Only ONE option is rewritten — the one named by signal.letter.
"""
from __future__ import annotations

import re

from pipeline.fixers import Fixer
from pipeline.fixers._helpers import parse_fixer_json
from pipeline.detectors import DetectorSignal


# The same UQ tokens the english_gap_scanner detects. Word boundaries
# on each side; case-insensitive removal.
_UQ_TOKENS: tuple[str, ...] = (
    "all", "every", "always", "never", "throughout",
    "entire", "any", "none", "no",
)
_UQ_RE = re.compile(
    r"\b(" + "|".join(_UQ_TOKENS) + r")\b",
    re.IGNORECASE,
)

# Sonnet pricing (matching scripts/audit_stem_contradictions imports).
_SONNET_MODEL_ID = "claude-sonnet-4-6"

_FIX_PROMPT = """You are minimally repairing one distractor in a multiple-choice question. The original distractor used a universal quantifier ("{word}") that the stem directly contradicts via a specific stated case. Your task: rewrite this distractor so it expresses a wrong-but-plausible SPECIFIC claim instead of a universal denial.

ABSOLUTE PRESERVATION RULES (violating these is worse than the original problem):
- Do NOT change which option is correct. Letter {letter} stays as a distractor.
- Do NOT add NEW universal quantifiers (all, every, always, never, throughout, entire, any, none, no).
- Do NOT introduce new lexical contradictions with the stem.
- Do NOT change the misconception being tested — only refine its expression.
- All four options remain. Only the text of letter {letter} may change.

STEM:
{stem}

ALL OPTIONS (you may only modify {letter}):
{options_block}

The flagged option is letter {letter}, currently:
"{distractor_text}"

OUTPUT — single JSON object (no preamble, no markdown):

{{
  "letter": "{letter}",
  "new_text": "<rewritten distractor>",
  "new_explanation": "<rewritten explanation, 1-2 sentences>"
}}"""


def _drop_uq_word(text: str) -> tuple[str, str | None]:
    """Strip the first universal-quantifier word from `text`.
    Returns (new_text, dropped_word) or (text, None) if no UQ found."""
    m = _UQ_RE.search(text or "")
    if not m:
        return text, None
    word = m.group(1)
    new_text = (text[:m.start()] + text[m.end():]).strip()
    # Collapse any double-spacing introduced by the deletion
    new_text = re.sub(r"\s+", " ", new_text)
    return new_text, word


class UniversalQuantifierFixer(Fixer):
    fixer_id = "universal_quantifier_fixer"
    handles_signatures = ("universal_quantifier",)

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

        options = list(question.get("options") or [])
        target_idx = next(
            (i for i, o in enumerate(options) if o.get("letter") == letter),
            None,
        )
        if target_idx is None:
            return question

        target_opt = options[target_idx]
        if target_opt.get("is_correct"):
            # Never modify the correct option.
            return question

        original_text = target_opt.get("text", "") or ""
        stripped, dropped = _drop_uq_word(original_text)
        if dropped is None:
            # Detector said UQ but we can't find it — return unchanged.
            return question

        # If stripped text is still substantive, keep the deterministic edit.
        # ≥10 words is a heuristic threshold — short enough to allow brief
        # distractors, long enough to avoid producing nonsense.
        if len(stripped.split()) >= 10:
            new_options = list(options)
            new_options[target_idx] = dict(target_opt)
            new_options[target_idx]["text"] = stripped
            new_options[target_idx]["_routed_fixer"] = "universal_quantifier:deterministic_drop"
            new_q = dict(question)
            new_q["options"] = new_options
            return new_q

        # Otherwise, Sonnet rewrites. Build a "all options" preview block.
        options_block_lines = []
        for o in options:
            mark = "[CORRECT]" if o.get("is_correct") else "[distractor]"
            options_block_lines.append(
                f"  {o.get('letter','?')} {mark}: {o.get('text','')}"
            )
        options_block = "\n".join(options_block_lines)

        prompt = _FIX_PROMPT.format(
            word=dropped,
            letter=letter,
            stem=question.get("question_stem", ""),
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

        new_text = parsed.get("new_text", "").strip()
        new_explanation = parsed.get("new_explanation", "").strip()
        if not new_text:
            return question

        # Invariant guard: if the LLM re-introduced a universal quantifier,
        # reject the rewrite.
        if _UQ_RE.search(new_text):
            return question

        new_options = list(options)
        new_options[target_idx] = dict(target_opt)
        new_options[target_idx]["text"] = new_text
        if new_explanation:
            new_options[target_idx]["explanation"] = new_explanation
        new_options[target_idx]["_routed_fixer"] = "universal_quantifier:llm_rewrite"

        new_q = dict(question)
        new_q["options"] = new_options
        return new_q


__all__ = ["UniversalQuantifierFixer"]
