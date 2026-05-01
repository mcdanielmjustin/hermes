"""schema_labeling fixer — Phase A6.

When the schema_labeling classifier fired (LABEL_PAIRS Tier B match) on
a distractor, it means the distractor swaps which label attaches to
which member of a paired-named-concept (IV/DV, agonist/antagonist,
etc.). The audit override demoted english_gap → content_gap (correct).
But if downstream still wants this distractor *rewritten* (e.g. because
the pattern is judged unintentional in the context), this fixer
performs the swap correction.

Strategy:
  - The detector's signal.extra carries `pair_matched=(a,b)`.
  - Determine which member is in the stem and which in the distractor.
  - Swap: replace the distractor's mention with the OTHER member.
  - Deterministic — no LLM call needed.

NOTE: most of the time, schema_labeling demotion is sufficient and
this fixer doesn't need to run. It's a fallback for cases where the
audit-time demotion left the distractor in place but the question
still flagged for fix.
"""
from __future__ import annotations

import re

from pipeline.fixers import Fixer
from pipeline.detectors import DetectorSignal


def _word_present(text: str, token: str) -> bool:
    if not text or not token:
        return False
    pat = r"\b" + re.escape(token) + r"\b"
    return re.search(pat, text, re.IGNORECASE) is not None


def _replace_word(text: str, old: str, new: str) -> str:
    pat = re.compile(r"\b" + re.escape(old) + r"\b", re.IGNORECASE)

    def _sub(m):
        original = m.group(0)
        if original[:1].isupper():
            return new[:1].upper() + new[1:]
        return new

    return pat.sub(_sub, text)


class SchemaLabelingFixer(Fixer):
    fixer_id = "schema_labeling_fixer"
    # Both Tier A (brief discriminators) and Tier B (LABEL_PAIRS lexical)
    # surface as one of these signature labels via the detector wrapper.
    handles_signatures = ("tier_a_brief", "tier_b_lexical")

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

        pair = (signal.extra or {}).get("pair_matched")
        if not pair or len(pair) != 2:
            return question
        a, b = pair[0], pair[1]
        if not a or not b:
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

        # Decide which side is in the distractor; swap it for the other.
        a_in_dist = _word_present(original_text, a)
        b_in_dist = _word_present(original_text, b)
        if a_in_dist and not b_in_dist:
            new_text = _replace_word(original_text, a, b)
        elif b_in_dist and not a_in_dist:
            new_text = _replace_word(original_text, b, a)
        else:
            # Both or neither present in distractor — can't safely swap.
            return question

        new_options = list(options)
        new_options[target_idx] = dict(target_opt)
        new_options[target_idx]["text"] = new_text
        new_options[target_idx]["_routed_fixer"] = "schema_labeling:deterministic_swap"

        new_q = dict(question)
        new_q["options"] = new_options
        return new_q


__all__ = ["SchemaLabelingFixer"]
