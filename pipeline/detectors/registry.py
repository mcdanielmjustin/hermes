"""DetectorRegistry — the central place that knows which detectors exist
and runs them in phase-bounded passes.

Used at three insertion points (per the architecture plan):

  - Point A — `pipeline/orchestrator.py` post-assembly: run detectors
    with phase=PHASE_GENERATION; treat BLOCK verdicts as gate failures.
  - Point B — `scripts/audit_stem_contradictions.py` post-classification:
    run detectors with phase=PHASE_AUDIT; apply OVERRIDE_TO verdicts to
    LLM classifications; record ADVISORY signals on the manifest.
  - Point C — `scripts/ship_readiness.py` pre-routing: query
    chapter-aggregated detector signals to inform routing decisions.

The factory `create_detector_registry()` returns a registry pre-populated
with goliath's existing detectors, wrapped in the new interface but
preserving today's behavior (A1 invariant).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Iterable

from . import (
    Detector,
    DetectorSignal,
    PHASE_AUDIT,
    PHASE_AUDIT_LLM,
    PHASE_GENERATION,
    VALID_PHASES,
)


class DetectorRegistry:
    """Holds a collection of Detector instances and exposes phase-bounded
    iteration.

    Detectors register their phase tags via the `phases` class attribute.
    The registry indexes them by phase so `scan_for_phase("audit")` is a
    cheap dict lookup, not a scan.
    """

    def __init__(self) -> None:
        self._by_id: dict[str, Detector] = {}
        self._by_phase: dict[str, list[Detector]] = defaultdict(list)

    def register(self, detector: Detector) -> None:
        """Add a detector to the registry. Idempotent on detector_id —
        re-registering a detector with the same id replaces the prior
        entry (intentional: lets tests inject mocks).
        """
        det_id = detector.detector_id
        if not det_id:
            raise ValueError("detector must have a non-empty detector_id")
        if not detector.phases:
            raise ValueError(
                f"detector {det_id!r} declares no phases; nothing to register"
            )
        for ph in detector.phases:
            if ph not in VALID_PHASES:
                raise ValueError(
                    f"detector {det_id!r} declares unknown phase {ph!r}; "
                    f"must be one of {sorted(VALID_PHASES)}"
                )

        # Replace any prior registration of the same id.
        if det_id in self._by_id:
            old = self._by_id[det_id]
            for ph in old.phases:
                lst = self._by_phase.get(ph, [])
                self._by_phase[ph] = [d for d in lst if d.detector_id != det_id]

        self._by_id[det_id] = detector
        for ph in detector.phases:
            self._by_phase[ph].append(detector)

    def get(self, detector_id: str) -> Detector | None:
        """Look up a detector by id. Returns None if not registered."""
        return self._by_id.get(detector_id)

    def detectors_for_phase(self, phase: str) -> list[Detector]:
        """Return the (ordered) list of detectors registered for a phase.
        Order is registration order — relevant for audit-time, where
        detectors that produce overrides should run before detectors
        that only emit advisory signals.
        """
        if phase not in VALID_PHASES:
            raise ValueError(
                f"unknown phase {phase!r}; must be one of {sorted(VALID_PHASES)}"
            )
        return list(self._by_phase.get(phase, []))

    def scan_for_phase(
        self,
        phase: str,
        question: dict,
        context: dict | None = None,
    ) -> list[DetectorSignal]:
        """Run every detector registered for `phase` against the question
        and concatenate their signals. Each detector is called with the
        same context dict — detectors that need phase-specific extras
        should pull them from context.

        Detectors must not raise; if one does, the registry catches the
        exception, emits no signals from it, and continues. This prevents
        a single buggy detector from blowing up an entire generation /
        audit run. (Buggy detectors are caught by tests; the registry's
        job is graceful degradation.)
        """
        out: list[DetectorSignal] = []
        for det in self.detectors_for_phase(phase):
            try:
                signals = det.scan(question, context) or []
            except Exception as e:
                # Emit a synthetic advisory signal so the failure is
                # visible on the manifest, but don't propagate.
                from . import VERDICT_ADVISORY
                out.append(DetectorSignal(
                    detector_id=det.detector_id,
                    letter=None,
                    fired=False,
                    confidence=0.0,
                    signature="exception",
                    verdict_action=VERDICT_ADVISORY,
                    reason=f"detector raised: {type(e).__name__}: {e}",
                ))
                continue
            for s in signals:
                if not isinstance(s, DetectorSignal):
                    raise TypeError(
                        f"detector {det.detector_id!r} returned non-Signal: "
                        f"{type(s).__name__}"
                    )
            out.extend(signals)
        return out

    def all_detectors(self) -> Iterable[Detector]:
        """Iterate over every registered detector. Order is registration
        order across all phases."""
        return list(self._by_id.values())

    async def scan_for_phase_async(
        self,
        phase: str,
        question: dict,
        context: dict | None = None,
    ) -> list[DetectorSignal]:
        """Async-aware variant of `scan_for_phase`. Used for
        PHASE_AUDIT_LLM detectors that make LLM calls. Each detector
        with an `async_scan` method is awaited; sync `scan` detectors
        also work (their results are folded in transparently).

        Detectors that raise are caught (synthetic advisory signal
        emitted) per the same contract as the sync path.
        """
        out: list[DetectorSignal] = []
        for det in self.detectors_for_phase(phase):
            try:
                if hasattr(det, "async_scan"):
                    signals = await det.async_scan(question, context) or []
                else:
                    signals = det.scan(question, context) or []
            except Exception as e:
                from . import VERDICT_ADVISORY
                out.append(DetectorSignal(
                    detector_id=det.detector_id,
                    letter=None,
                    fired=False,
                    confidence=0.0,
                    signature="exception",
                    verdict_action=VERDICT_ADVISORY,
                    reason=f"async detector raised: {type(e).__name__}: {e}",
                ))
                continue
            for s in signals:
                if not isinstance(s, DetectorSignal):
                    raise TypeError(
                        f"async detector {det.detector_id!r} returned non-Signal: "
                        f"{type(s).__name__}"
                    )
            out.extend(signals)
        return out


# ── Factory ─────────────────────────────────────────────────

def create_detector_registry() -> DetectorRegistry:
    """Build the canonical registry with all of goliath's existing
    detectors wrapped behind the new interface.

    Phase A1 ships these (behavior-preserving wrappers):
      - english_gap (advisory at audit; A2 promotes to override at T1/T2)
      - schema_labeling (override_to:content_gap at audit — preserves
        the existing override behavior at audit_stem_contradictions:784)
      - laterality (block at generation — wraps LateralityIntegrityGate)
      - universal_denial (block at generation — wraps UniversalDenialGate)

    Detectors are imported lazily so importing the registry doesn't
    transitively load Sonnet clients or other heavy modules.
    """
    # Local imports keep the registry module light at import time.
    from .english_gap import EnglishGapDetector
    from .schema_labeling import SchemaLabelingDetector
    from .laterality import LateralityDetector
    from .universal_denial import UniversalDenialDetector
    # Phase A5 — new tier-aware structural classifiers.
    from .numeric_overlap import NumericOverlapDetector
    from .imperative_lead import ImperativeLeadDetector
    from .meta_evaluative import MetaEvaluativeDetector
    from .lead_form_parallelism import LeadFormParallelismDetector
    from .defensible_alternative import DefensibleAlternativeDetector
    # Phase A7 — LLM-backed signal-shaped detectors (advisory only).
    from .llm_ambiguity import LlmAmbiguityDetector
    from .llm_fact_check import LlmFactCheckDetector

    registry = DetectorRegistry()
    registry.register(EnglishGapDetector())
    registry.register(SchemaLabelingDetector())
    registry.register(LateralityDetector())
    registry.register(UniversalDenialDetector())
    registry.register(NumericOverlapDetector())
    registry.register(ImperativeLeadDetector())
    registry.register(MetaEvaluativeDetector())
    registry.register(LeadFormParallelismDetector())
    registry.register(DefensibleAlternativeDetector())
    registry.register(LlmAmbiguityDetector())
    registry.register(LlmFactCheckDetector())
    return registry


__all__ = [
    "DetectorRegistry",
    "create_detector_registry",
]
