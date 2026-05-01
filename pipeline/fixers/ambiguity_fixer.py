"""ambiguity fixer — Phase A7+.

Handles `llm_ambiguity` signals from `pipeline.detectors.llm_ambiguity`.
Reads the defensible-alternative argument from signal.extra['argument']
and asks Sonnet to rewrite the flagged distractor so it's NO LONGER
defensibly correct under that alternative interpretation.

Strategy:
  1. Read the signal's argument — this is the LLM's explanation for
     WHY the distractor could be defended.
  2. Build a narrow Sonnet prompt that:
     - Names the distractor and its current text
     - Quotes the defensible-alternative argument
     - Asks for a rewrite that makes the distractor unambiguously wrong
       while preserving the misconception being tested
  3. Apply invariant guards: correct option unchanged, all 4 options
     present, no new universal quantifiers introduced.

This is the FIRST fixer that targets dq ambiguity directly. It
complements A6's surface fixers (universal_quantifier, laterality,
schema_labeling, numeric_overlap) by handling the deeper ambiguity
patterns that regex can't see.
"""
from __future__ import annotations

import re

from pipeline.fixers import Fixer
from pipeline.fixers._helpers import parse_fixer_json
from pipeline.detectors import DetectorSignal


_SONNET_MODEL_ID = "claude-sonnet-4-6"

_FIX_PROMPT = """You are minimally repairing one distractor in a multiple-choice question. An LLM-backed audit found this distractor to be DEFENSIBLY CORRECT under the alternative interpretation below — meaning a knowledgeable test-taker could argue it's also correct, undermining the question's discrimination.

Your task: rewrite the distractor so it's UNAMBIGUOUSLY WRONG while still testing the same misconception. Do this by removing the specific reading that lets the alternative interpretation succeed — narrow the distractor's claim to one that's clearly false under the stem.

ABSOLUTE PRESERVATION RULES (violating these is worse than the original ambiguity):
- Do NOT change which option is correct. Letter {letter} stays as a distractor.
- Do NOT add universal quantifiers (all, every, always, never, throughout, entire, any, none, no).
- Do NOT introduce new lexical contradictions with the stem (don't make it english_gap).
- Preserve the misconception being tested — refine the EXPRESSION, not the target concept.
- All four options remain. Only the text of letter {letter} may change.

QUESTION STEM:
{stem}

ALL OPTIONS (you may only modify {letter}):
{options_block}

THE FLAGGED OPTION ({letter}):
"{distractor_text}"

DEFENSIBLE-ALTERNATIVE ARGUMENT (why the audit flagged this):
{argument}

OUTPUT — single JSON object (no preamble, no markdown):

{{
  "letter": "{letter}",
  "new_text": "<rewritten distractor that's unambiguously wrong>",
  "new_explanation": "<rewritten explanation, 1-2 sentences>"
}}"""


_UQ_RE = re.compile(
    r"\b(all|every|always|never|throughout|entire|any|none|no)\b",
    re.IGNORECASE,
)


class AmbiguityFixer(Fixer):
    """Routes `llm_ambiguity` signals to a Sonnet-backed minimal rewrite
    that removes the defensible alternative."""

    fixer_id = "ambiguity_fixer"
    handles_signatures = ("llm_ambiguity",)

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

        # Argument is the LLM's reason for flagging — needed in the
        # rewrite prompt to ground the fix.
        argument = (signal.extra or {}).get("argument", "") or signal.reason or ""
        if not argument:
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
        if not original_text:
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
            stem=question.get("question_stem", ""),
            options_block=options_block,
            distractor_text=original_text,
            argument=argument,
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

        # Invariant guard: reject rewrites that introduce universal
        # quantifiers (would create english_gap).
        if _UQ_RE.search(new_text):
            return question

        new_options = list(options)
        new_options[target_idx] = dict(target_opt)
        new_options[target_idx]["text"] = new_text
        if new_explanation:
            new_options[target_idx]["explanation"] = new_explanation
        new_options[target_idx]["_routed_fixer"] = "ambiguity:llm_rewrite"

        new_q = dict(question)
        new_q["options"] = new_options
        return new_q


__all__ = ["AmbiguityFixer"]
