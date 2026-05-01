"""laterality detector — Phase A1 wrapper around LateralityIntegrityGate.

Behavior contract (A1):
  - Wraps `pipeline.gates.LateralityIntegrityGate.check`.
  - Single signal per question (the underlying gate fails on the first
    inverted-laterality distractor it finds; we surface that as one
    BLOCK signal carrying the gate's `reason`).
  - phase=PHASE_GENERATION — the gate runs at gen-time today; A1
    preserves that.

The underlying gate produces (ok, reason) — we translate ok=False into
a BLOCK signal and ok=True into a fired=False advisory.
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
from pipeline.gates import LateralityIntegrityGate

# Regex used to extract the offending option letter from the gate's
# reason string. The gate currently formats failures as
#   "distractor X asserts 'unilateral' laterality while the stem ..."
# so we pull the first single-letter capital after "distractor".
_LETTER_RE = re.compile(r"\bdistractor\s+([A-D])\b")


class LateralityDetector(Detector):
    """Wraps `LateralityIntegrityGate` so it speaks DetectorSignal.

    The underlying gate is conservative — it bypasses unless the stem
    asserts ONE laterality exclusively (`bilateral` XOR `unilateral`),
    then fails on any distractor that asserts the opposite. A1
    preserves this behavior unchanged.
    """

    detector_id = "laterality_integrity"
    phases = (PHASE_GENERATION,)

    def __init__(self) -> None:
        self._gate = LateralityIntegrityGate()

    def scan(
        self,
        question: dict,
        context: dict | None = None,
    ) -> list[DetectorSignal]:
        ok, reason = self._gate.check(question or {}, context)
        if ok:
            # Emit a fired=False advisory so observability reflects the
            # detector ran. Letter=None because there's no per-letter
            # finding — the whole-question check passed.
            return [DetectorSignal(
                detector_id=self.detector_id,
                letter=None,
                fired=False,
                confidence=1.0,
                signature=None,
                verdict_action=VERDICT_ADVISORY,
                reason="ok",
            )]

        # Failed: extract the offending letter from the gate's reason
        # if possible (the gate formats "distractor X asserts ...").
        m = _LETTER_RE.search(reason or "")
        letter = m.group(1) if m else None

        return [DetectorSignal(
            detector_id=self.detector_id,
            letter=letter,
            fired=True,
            confidence=1.0,
            signature="laterality_inversion",
            verdict_action=VERDICT_BLOCK,
            reason=reason or "laterality inversion detected",
        )]


__all__ = ["LateralityDetector"]
