"""english_gap detector — wraps the Phase 24 deterministic scanner.

A1 contract (DONE 2026-04-29): emits ADVISORY signals at audit time;
behavior bit-identical to direct `scan_question` calls.

A2 contract (DONE 2026-04-29): high-confidence signatures on T1/T2
emit OVERRIDE_TO with proposed_class="english_gap". T3/T4 advisory.

A2.5 contract (this file's current behavior): override-eligibility is
read from `pipeline.distractor_policy.resolve(tier, source_type,
stem_pattern).threshold_for("english_gap_scanner")`. The threshold is
tier-conditional via `DEFAULT_T1_T2` / `DEFAULT_T3` / `DEFAULT_T4` (and
domain-specific cells when populated). When threshold is None or
signal.confidence < threshold, the signal is ADVISORY.

Override-eligible signatures (the scanner's high-confidence set):
  - universal_quantifier (conf 0.85)
  - laterality           (conf 0.75)
  - numeric_ratio        (conf 0.80)

Always advisory:
  - stage_timing (conf 0.65) — too low for any cell threshold

Why tier-conditional thresholds: at higher Bloom's tiers, lexical
contradictions are more likely content_gap-disguised-as-english_gap
(the student is supposed to read carefully and apply concept knowledge
to recognize the contradiction). T1/T2 = aggressive override; T3 =
only highest-confidence signature; T4 = effectively advisory.
"""
from __future__ import annotations

from . import (
    Detector,
    DetectorSignal,
    PHASE_AUDIT,
    PHASE_GENERATION,
    VERDICT_ADVISORY,
    VERDICT_OVERRIDE_TO,
)
from pipeline.english_gap_scanner import scan_question
from pipeline import distractor_policy


# Signatures that are override-eligible AT ALL (the scanner's
# high-confidence set). The actual override decision per signature is
# the conjunction (signature ∈ this set) AND (confidence ≥ cell
# threshold). stage_timing (conf 0.65) is excluded because no cell
# threshold can promote it without false positives, even at T1.
OVERRIDE_ELIGIBLE_SIGNATURES = frozenset({
    "universal_quantifier",
    "laterality",
    "numeric_ratio",
})

# A2 historical constant — preserved for tests/imports that reference
# it. Now informational only; the actual tier gating reads from the
# cell matrix's `override_thresholds`.
OVERRIDE_ELIGIBLE_TIERS = frozenset({1, 2})


class EnglishGapDetector(Detector):
    """Wraps the Phase 24 deterministic english_gap scanner.

    A2.5: override-eligibility per (tier, source_type, stem_pattern) cell.
    The cell's `threshold_for("english_gap_scanner")` returns the minimum
    fired-confidence to trigger OVERRIDE_TO. Signals below the threshold
    (or signatures not in the eligible set) emit ADVISORY.

    The override application primitive is
    `pipeline.english_gap_scanner.apply_english_gap_override` (called
    from the audit script).
    """

    detector_id = "english_gap_scanner"
    # A3: detector also runs at generation time. The same scan() logic
    # produces the same signals; the orchestrator's gate loop interprets
    # OVERRIDE_TO at gen-time as "block this generation; re-prompt with
    # targeted correction." Behind GOLIATH_DETECTORS_AT_GEN env flag
    # for measurement-before-default-on.
    phases = (PHASE_AUDIT, PHASE_GENERATION)

    def scan(
        self,
        question: dict,
        context: dict | None = None,
    ) -> list[DetectorSignal]:
        question = question or {}
        tier = question.get("difficulty_tier")
        source_type = question.get("source_type")
        stem_pattern = question.get("stem_pattern")
        domain_code = question.get("domain_code")

        # A2.5: cell determines override threshold.
        cell = distractor_policy.resolve(
            tier=tier,
            domain_code=domain_code,
            source_type=source_type,
            stem_pattern=stem_pattern,
        )
        threshold = cell.threshold_for("english_gap_scanner")

        raw = scan_question(question)
        out: list[DetectorSignal] = []
        for letter, sig in raw.items():
            should_override = (
                bool(sig.fired)
                and sig.signature in OVERRIDE_ELIGIBLE_SIGNATURES
                and threshold is not None
                and float(sig.confidence) >= float(threshold)
            )
            if should_override:
                out.append(DetectorSignal(
                    detector_id=self.detector_id,
                    letter=letter,
                    fired=True,
                    confidence=float(sig.confidence),
                    signature=sig.signature,
                    verdict_action=VERDICT_OVERRIDE_TO,
                    proposed_class="english_gap",
                    reason=sig.reason or "",
                    extra={
                        "tier": tier,
                        "cell_threshold": threshold,
                        "cell_note": cell.note,
                    },
                ))
            else:
                out.append(DetectorSignal(
                    detector_id=self.detector_id,
                    letter=letter,
                    fired=bool(sig.fired),
                    confidence=float(sig.confidence),
                    signature=sig.signature,
                    verdict_action=VERDICT_ADVISORY,
                    reason=sig.reason or "",
                    extra={
                        "tier": tier,
                        "cell_threshold": threshold,
                    },
                ))
        return out


__all__ = ["EnglishGapDetector", "OVERRIDE_ELIGIBLE_SIGNATURES",
           "OVERRIDE_ELIGIBLE_TIERS"]
