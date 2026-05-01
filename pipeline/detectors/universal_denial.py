"""universal_denial detector — Phase A1 wrapper around UniversalDenialGate.

Behavior contract (A1):
  - Wraps `pipeline.gates.UniversalDenialGate.check`.
  - Emits one BLOCK signal when the gate fails; one fired=False
    advisory signal when it passes.
  - phase=PHASE_GENERATION — the gate runs at gen-time today.

The underlying gate looks for a preservation marker in the stem
(`still`, `intact`, `vividly`, etc.) paired with a universal quantifier
+ denial verb in a distractor (`erases ALL`, `forgets every`, etc.)
within ~80 chars, AND requires a shared content word ≥6 chars between
stem and distractor.
"""
from __future__ import annotations

import re

from . import (
    Detector,
    DetectorSignal,
    PHASE_GENERATION,
    VERDICT_ADVISORY,
    VERDICT_BLOCK,
)
from pipeline.gates import UniversalDenialGate

_LETTER_RE = re.compile(r"\bdistractor\s+([A-D])\b")


class UniversalDenialDetector(Detector):
    """Wraps `UniversalDenialGate` so it speaks DetectorSignal.

    The underlying gate is narrow by design — only fires on the
    canonical preservation-marker + universal-denial pattern with
    ≥1 shared content word between stem and distractor. Sonnet audit
    catches semantic versions this regex misses; A1 preserves both.
    """

    detector_id = "universal_denial"
    phases = (PHASE_GENERATION,)

    def __init__(self) -> None:
        self._gate = UniversalDenialGate()

    def scan(
        self,
        question: dict,
        context: dict | None = None,
    ) -> list[DetectorSignal]:
        ok, reason = self._gate.check(question or {}, context)
        if ok:
            return [DetectorSignal(
                detector_id=self.detector_id,
                letter=None,
                fired=False,
                confidence=1.0,
                signature=None,
                verdict_action=VERDICT_ADVISORY,
                reason="ok",
            )]

        m = _LETTER_RE.search(reason or "")
        letter = m.group(1) if m else None

        return [DetectorSignal(
            detector_id=self.detector_id,
            letter=letter,
            fired=True,
            confidence=1.0,
            signature="universal_denial",
            verdict_action=VERDICT_BLOCK,
            reason=reason or "universal denial detected",
        )]


__all__ = ["UniversalDenialDetector"]
