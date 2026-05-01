"""imperative_lead detector — Phase A5b.

Detects distractors that begin with an imperative verb ("Identify X",
"Predict Y", "Classify Z"). Imperative leads make options sound like
questions instead of answers — a style violation goliath's prompt
explicitly forbids.

Tier-blind: violates style at every Bloom's tier.

Verdict: VERDICT_BLOCK on fire (the orchestrator gate-loop treats this
as gate failure at PHASE_GENERATION; the audit consumer ignores BLOCK
signals from non-english_gap detectors, so the audit path is
unaffected). Confidence 0.95 — regex match is unambiguous.
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


# Imperative verbs that signal an instruction lead. Word boundaries on
# each side; case-insensitive matching at the start of the distractor.
_IMPERATIVE_VERBS: frozenset[str] = frozenset({
    "identify", "predict", "classify", "determine", "choose",
    "recognize", "distinguish", "differentiate", "compute",
    "calculate", "compare", "contrast", "list", "describe",
    "explain", "evaluate", "select",
})

_IMPERATIVE_RE = re.compile(
    r"^\s*(" + "|".join(_IMPERATIVE_VERBS) + r")\b",
    re.IGNORECASE,
)


class ImperativeLeadDetector(Detector):
    """Fires on distractors whose first non-whitespace token is an
    imperative verb."""

    detector_id = "imperative_lead"
    phases = (PHASE_GENERATION, PHASE_AUDIT)

    def scan(
        self,
        question: dict,
        context: dict | None = None,
    ) -> list[DetectorSignal]:
        question = question or {}
        out: list[DetectorSignal] = []
        for opt in question.get("options") or []:
            if opt.get("is_correct"):
                continue
            letter = opt.get("letter", "?")
            text = opt.get("text", "") or ""
            m = _IMPERATIVE_RE.match(text)
            if not m:
                out.append(DetectorSignal(
                    detector_id=self.detector_id,
                    letter=letter,
                    fired=False,
                    confidence=0.0,
                    signature=None,
                    verdict_action=VERDICT_ADVISORY,
                    reason="no imperative lead",
                ))
                continue
            verb = m.group(1).lower()
            out.append(DetectorSignal(
                detector_id=self.detector_id,
                letter=letter,
                fired=True,
                confidence=0.95,
                signature="imperative_lead",
                verdict_action=VERDICT_BLOCK,
                reason=f"distractor {letter} begins with imperative '{verb}'",
                extra={"imperative": verb},
            ))
        return out


__all__ = ["ImperativeLeadDetector"]
