"""defensible_alternative detector — Phase A5e (T4 advisory stub).

Heuristic stub for T4 ambiguity. Counts how many options share enough
content vocabulary with the stem's testable_fact (or anchor content
summary) to be plausibly correct under a reasonable interpretation.

Goliath's current corpus has ~22% T4 dq_major rate, with ambiguity as
the dominant failure mode (per E2E-100 simulation). This detector is
a coarse first cut — it surfaces questions where ≥2 options have
substantial content-overlap with the stem, signaling that the keyed
answer may not be uniquely defensible.

This is NOT a robust ambiguity detector. The full solution lives in
A7 (LLM-backed ambiguity audit). A5e ships an advisory-only stub so
the corpus has *some* T4 ambiguity tracking pre-A7.

Tier behavior: T4 ONLY. Other tiers return fired=False.

Verdict: VERDICT_ADVISORY always (low-confidence heuristic). Confidence
0.4 — well below any cell's override threshold; informational only.
"""
from __future__ import annotations

import re

from . import (
    Detector,
    DetectorSignal,
    PHASE_AUDIT,
    VERDICT_ADVISORY,
)


# Stopwords excluded from content-overlap counting. Kept minimal; the
# pipeline's `pipeline.stopwords.BASE_FULL` is the authoritative source
# but importing it here would couple modules. Use a focused subset.
_STOP: frozenset[str] = frozenset({
    "the", "a", "an", "and", "or", "but", "is", "are", "was", "were",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must", "shall",
    "to", "of", "in", "on", "at", "by", "for", "with", "from", "as",
    "this", "that", "these", "those", "it", "its", "they", "them",
    "their", "there", "which", "what", "who", "whom", "whose", "where",
    "when", "why", "how", "all", "any", "some", "no", "not", "only",
    "if", "then", "than", "so", "such", "more", "most", "less", "least",
    "very", "much", "many", "few", "same", "different", "other", "another",
    "between", "among", "into", "through", "during", "before", "after",
    "under", "over", "above", "below", "out", "off",
})

_MIN_WORD_LEN = 5  # Filter short words; focus on content tokens.


def _content_words(text: str) -> set[str]:
    """Extract content-word set from text. Lowercase, length ≥ 5,
    not a stopword, alphanumeric."""
    if not text:
        return set()
    tokens = re.findall(r"\b[a-zA-Zà-öø-ÿ][a-zA-Zà-öø-ÿ0-9-]+\b", text.lower())
    return {
        t for t in tokens
        if len(t) >= _MIN_WORD_LEN and t not in _STOP
    }


class DefensibleAlternativeDetector(Detector):
    """T4-only stub: fires when ≥2 distractors share substantial content
    vocabulary with the stem's testable_fact (or anchor summary), hinting
    that the keyed answer may not be uniquely defensible.

    Threshold: each candidate option shares ≥3 content words with the
    stem's testable fact. If ≥2 options meet this bar, fire (advisory).
    """

    detector_id = "defensible_alternative"
    phases = (PHASE_AUDIT,)
    MIN_SHARED_PER_OPTION = 3
    MIN_OPTIONS_TO_FIRE = 2

    def scan(
        self,
        question: dict,
        context: dict | None = None,
    ) -> list[DetectorSignal]:
        question = question or {}
        tier = question.get("difficulty_tier")
        if tier != 4:
            return [DetectorSignal(
                detector_id=self.detector_id,
                letter=None,
                fired=False,
                confidence=0.0,
                signature=None,
                verdict_action=VERDICT_ADVISORY,
                reason=f"detector applies to T4 only (got T{tier})",
            )]

        # Build the "stem context" — testable_fact preferred, fall back
        # to anchor_content_summaries, then to the stem itself.
        stem_context = (
            question.get("testable_fact")
            or " ".join(question.get("anchor_content_summaries") or [])
            or question.get("question_stem", "")
            or ""
        )
        stem_words = _content_words(stem_context)
        if len(stem_words) < self.MIN_SHARED_PER_OPTION:
            return [DetectorSignal(
                detector_id=self.detector_id,
                letter=None,
                fired=False,
                confidence=0.0,
                signature=None,
                verdict_action=VERDICT_ADVISORY,
                reason=(
                    f"stem context has only {len(stem_words)} content words "
                    f"(need ≥{self.MIN_SHARED_PER_OPTION})"
                ),
            )]

        # Count sharing per option (distractors only — the correct option
        # SHOULD share with the stem; we're looking at distractors that
        # also share heavily, suggesting they could be defensibly correct).
        sharing: list[tuple[str, int]] = []
        for opt in question.get("options") or []:
            if opt.get("is_correct"):
                continue
            letter = opt.get("letter", "?")
            text = opt.get("text", "") or ""
            opt_words = _content_words(text)
            shared = stem_words & opt_words
            if len(shared) >= self.MIN_SHARED_PER_OPTION:
                sharing.append((letter, len(shared)))

        if len(sharing) < self.MIN_OPTIONS_TO_FIRE:
            return [DetectorSignal(
                detector_id=self.detector_id,
                letter=None,
                fired=False,
                confidence=0.0,
                signature=None,
                verdict_action=VERDICT_ADVISORY,
                reason=(
                    f"only {len(sharing)} distractor(s) meet sharing threshold "
                    f"(need ≥{self.MIN_OPTIONS_TO_FIRE})"
                ),
            )]

        return [DetectorSignal(
            detector_id=self.detector_id,
            letter=None,
            fired=True,
            confidence=0.4,
            signature="defensible_alternative",
            verdict_action=VERDICT_ADVISORY,
            reason=(
                f"T4 with {len(sharing)} distractors sharing ≥"
                f"{self.MIN_SHARED_PER_OPTION} content words with stem fact: "
                f"{sharing}"
            ),
            extra={"sharing": sharing, "tier": 4},
        )]


__all__ = ["DefensibleAlternativeDetector"]
