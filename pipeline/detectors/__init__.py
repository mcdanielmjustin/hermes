"""Phase A1 — Detector Registry foundation.

This module defines the unified `Detector` ABC and `DetectorSignal`
dataclass that every deterministic (or LLM-backed signal-shaped) check
in goliath ultimately produces. The registry (in `registry.py`)
collects detectors and runs them in phase-bounded passes.

Design contract:

- A detector's job is to look at a question and produce zero or more
  DetectorSignals describing what it found. The signal's
  `verdict_action` tells downstream code what to do with it:

    BLOCK         — generation-time gate failure (used by gen-time pass)
    OVERRIDE_TO   — audit verdict override (use `proposed_class`)
    ADVISORY      — informational only; trace/log; no action

- A detector declares which phases it runs in via the `phases` tuple.
  The registry filters by phase. A detector can run in multiple phases
  (e.g., english_gap runs at both "audit" and "generation" once Phase A3
  ships).

- Detectors must be pure: same inputs → same outputs. No I/O. No global
  state mutation. This is what makes them deterministic and overridable.
  LLM-backed detectors (Phase A7) live behind the same interface but
  declare phase="audit_llm" so they run last and only when the
  deterministic gates pass.

A1 contract: this module + the registry + thin wrappers around the
existing detectors must produce signals that are bit-identical to
today's behavior. No semantic changes.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


# ── Verdict actions ──────────────────────────────────────────

#: Generation-time gate failure. The orchestrator should reject the
#: assembled question and route to retry / correction. Carries the
#: human-readable `reason` for use in the correction prompt.
VERDICT_BLOCK = "block"

#: Audit verdict override. The audit's classification for this option
#: should be replaced by `proposed_class`. Tracing fields (signature,
#: confidence, reason) preserved for manifest entries.
VERDICT_OVERRIDE_TO = "override_to"

#: Informational only. The signal is logged on the manifest but does
#: not change the audit verdict or block generation. Use when a detector
#: has fired but its confidence is too low to override (or A1's "no
#: behavior change" requirement keeps it advisory until A2 promotes it).
VERDICT_ADVISORY = "advisory"

VALID_VERDICT_ACTIONS = frozenset({
    VERDICT_BLOCK, VERDICT_OVERRIDE_TO, VERDICT_ADVISORY,
})


# ── Phase tags ───────────────────────────────────────────────

#: Generation-time pass: detectors in this phase run after the orchestrator
#: assembles a question, before the validation gate loop. A BLOCK verdict
#: here causes the orchestrator to route to correction.
PHASE_GENERATION = "generation"

#: Audit-time pass (deterministic): detectors run after the LLM audit's
#: classifications are produced, before flagged_distractors derivation.
#: An OVERRIDE_TO verdict here replaces the LLM's class for that option.
PHASE_AUDIT = "audit"

#: Audit-time pass (LLM-backed): detectors that themselves use an LLM,
#: shaped to the same DetectorSignal interface. Run after deterministic
#: detectors so they don't fight regex verdicts.
PHASE_AUDIT_LLM = "audit_llm"

VALID_PHASES = frozenset({
    PHASE_GENERATION, PHASE_AUDIT, PHASE_AUDIT_LLM,
})


# ── Signal ───────────────────────────────────────────────────

@dataclass(frozen=True)
class DetectorSignal:
    """One observation by one detector on one question element.

    ``letter`` is the option letter (A/B/C/D) for distractor-level signals;
    None for stem-level or whole-question signals.

    ``fired`` distinguishes "I ran and saw nothing" from "I ran and
    detected the pattern." Negative-fire signals are still emitted
    (with fired=False) so a manifest can show the detector ran without
    producing a finding — useful for telemetry and false-negative analysis.

    ``proposed_class`` is only meaningful when verdict_action ==
    VERDICT_OVERRIDE_TO. It carries the destination class
    (e.g., "english_gap", "content_gap").
    """
    detector_id: str
    letter: str | None
    fired: bool
    confidence: float
    signature: str | None
    verdict_action: str
    proposed_class: str | None = None
    reason: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.verdict_action not in VALID_VERDICT_ACTIONS:
            raise ValueError(
                f"invalid verdict_action: {self.verdict_action!r}; "
                f"must be one of {sorted(VALID_VERDICT_ACTIONS)}"
            )
        if self.verdict_action == VERDICT_OVERRIDE_TO and not self.proposed_class:
            raise ValueError(
                "verdict_action=override_to requires proposed_class"
            )
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError(
                f"confidence must be in [0.0, 1.0]; got {self.confidence}"
            )


# ── Detector ABC ─────────────────────────────────────────────

class Detector(ABC):
    """Abstract base for every detector wrapper.

    Subclasses declare:
      - detector_id (class attribute): a short stable identifier.
      - phases (class attribute): tuple of phase tags this detector
        runs in. Most detectors will be single-phase; multi-phase is
        useful when the same logic runs at gen-time AND audit-time
        (Phase A3 mirrors).

    Subclasses implement:
      - scan(question, context) -> list[DetectorSignal]

    The `context` dict carries phase-specific extras (e.g.
    `discriminators` from the brief at audit, or anchor metadata at
    generation). Detectors that don't need context should ignore it.

    A1 invariant: wrappers around existing scanners must produce signals
    such that the existing code paths (audit override, gate failure)
    behave identically when the registry is consulted.
    """

    #: Short stable identifier — used for logging, manifest entries,
    #: and `register()` lookups.
    detector_id: str = "base_detector"

    #: Phase tags this detector runs in.
    phases: tuple[str, ...] = ()

    @abstractmethod
    def scan(
        self,
        question: dict,
        context: dict | None = None,
    ) -> list[DetectorSignal]:
        """Run the detector and return all signals it produces.

        Should not raise on malformed input — return an empty list or
        a fired=False signal instead.
        """
        raise NotImplementedError


# Public re-exports for convenience.
__all__ = [
    "Detector",
    "DetectorSignal",
    "VERDICT_BLOCK",
    "VERDICT_OVERRIDE_TO",
    "VERDICT_ADVISORY",
    "VALID_VERDICT_ACTIONS",
    "PHASE_GENERATION",
    "PHASE_AUDIT",
    "PHASE_AUDIT_LLM",
    "VALID_PHASES",
]
