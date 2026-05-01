"""meta_evaluative detector — Phase A5c.

Detects stems containing meta-evaluative modifiers ("correctly", "best",
"most", "option") that bias the question toward a particular answer
without testing concept knowledge. Goliath's prompt explicitly forbids
these — a stem like "which option BEST describes..." cues the test-
taker to look for the strongest-sounding answer rather than the
correct one.

Tier-blind: stem-hygiene violation at every tier.

Verdict: VERDICT_BLOCK on fire — the stem itself is the issue, not a
specific distractor. signal.letter=None (whole-question scope).
Confidence 1.0 (literal token presence).
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


# Meta-evaluative modifiers. These are the canonical hedge words goliath's
# stem-hygiene rule explicitly bans. "option" is included because stems
# like "which option..." re-introduce question-form cuing.
_META_TOKENS: tuple[str, ...] = (
    "correctly",
    "best",
    "most",
    "option",
    "which option",
    "the option",
    "the best",
    "the most",
    "best describes",
    "most accurately",
    "most likely",
    "most appropriate",
)

# Word-boundary regex per token. Multi-word phrases match as literal
# sequences with internal whitespace tolerance.
_META_RE = re.compile(
    r"\b(" + "|".join(re.escape(t) for t in _META_TOKENS) + r")\b",
    re.IGNORECASE,
)


class MetaEvaluativeDetector(Detector):
    """Fires when the stem contains a meta-evaluative modifier.

    Caveat: the regex matches "best" / "most" anywhere in the stem.
    Some legitimate clinical prose uses these words ("most patients
    present with..."). The plan accepts this as a tradeoff — the
    detector is conservative-stub at A5; later work could narrow to
    stem-final position only ("which X best describes Y?" pattern).
    """

    detector_id = "meta_evaluative"
    phases = (PHASE_GENERATION, PHASE_AUDIT)

    def scan(
        self,
        question: dict,
        context: dict | None = None,
    ) -> list[DetectorSignal]:
        question = question or {}
        stem = question.get("question_stem", "") or ""
        m = _META_RE.search(stem)
        if not m:
            return [DetectorSignal(
                detector_id=self.detector_id,
                letter=None,
                fired=False,
                confidence=0.0,
                signature=None,
                verdict_action=VERDICT_ADVISORY,
                reason="no meta-evaluative modifier in stem",
            )]
        token = m.group(1).lower()
        return [DetectorSignal(
            detector_id=self.detector_id,
            letter=None,
            fired=True,
            confidence=1.0,
            signature="meta_evaluative",
            verdict_action=VERDICT_BLOCK,
            reason=f"stem contains meta-evaluative '{token}'",
            extra={"token": token},
        )]


__all__ = ["MetaEvaluativeDetector"]
