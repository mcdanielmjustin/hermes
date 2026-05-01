"""schema_labeling detector — Phase A1 wrapper around
schema_labeling_classifier.

Behavior contract (A1):
  - Wraps `pipeline.schema_labeling_classifier.classify_distractor`.
  - For every distractor in the question, runs the classifier against
    the (stem, distractor_text, discriminators) triple.
  - On fire, emits a signal with verdict_action=OVERRIDE_TO and
    proposed_class="content_gap" — preserves the existing override
    behavior at `scripts/audit_stem_contradictions.py:784`.
  - On no-fire, emits a fired=False advisory signal so the manifest
    can show the detector ran.

The existing `apply_schema_labeling_override` function still exists at
the call site for now (A1 only adds the registry wrapper alongside it).
The actual rewiring of the audit script to consume from the registry
happens in Task 8 — and `apply_schema_labeling_override` stays in place
as the application primitive (the registry produces signals; the audit
code applies them via the same helper).
"""
from __future__ import annotations

from . import (
    Detector,
    DetectorSignal,
    PHASE_AUDIT,
    VERDICT_ADVISORY,
    VERDICT_OVERRIDE_TO,
)
from pipeline.schema_labeling_classifier import classify_distractor


class SchemaLabelingDetector(Detector):
    """Wraps the Phase 22 schema-labeling structural classifier.

    Tier A (brief discriminators, conf=1.0): paired-named-concept
        discriminators on the brief drive override.
    Tier B (LABEL_PAIRS lexical fallback, conf=0.5): canonical pairs
        in stem with one member in distractor.
    Universal-quantifier guard: blocks override regardless of tier.

    Reads the brief discriminators (when present) from the question's
    `_discriminators` field — the same convention as
    `apply_schema_labeling_override`.
    """

    detector_id = "schema_labeling"
    phases = (PHASE_AUDIT,)

    def scan(
        self,
        question: dict,
        context: dict | None = None,
    ) -> list[DetectorSignal]:
        question = question or {}
        stem = question.get("question_stem", "") or ""
        # Discriminators come either via question dict (existing convention)
        # or via context dict (new convention for the registry path).
        discriminators = (
            question.get("_discriminators")
            or (context or {}).get("discriminators")
        )

        out: list[DetectorSignal] = []
        for opt in question.get("options") or []:
            if opt.get("is_correct"):
                continue
            letter = opt.get("letter", "?")
            distractor_text = opt.get("text", "") or ""
            sig = classify_distractor(
                stem=stem,
                distractor_text=distractor_text,
                discriminators=discriminators,
            )
            if sig.fired:
                out.append(DetectorSignal(
                    detector_id=self.detector_id,
                    letter=letter,
                    fired=True,
                    confidence=float(sig.confidence),
                    signature=(
                        "tier_a_brief" if sig.brief_boosted
                        else "tier_b_lexical"
                    ),
                    verdict_action=VERDICT_OVERRIDE_TO,
                    proposed_class="content_gap",
                    reason=sig.reason or "",
                    extra={
                        "pair_matched": sig.pair_matched,
                        "brief_boosted": sig.brief_boosted,
                        "universal_quantifier_blocked": (
                            sig.universal_quantifier_blocked
                        ),
                    },
                ))
            else:
                # Emit a fired=False advisory so the manifest can show
                # the detector ran on this option (and any blocked-by-
                # universal-quantifier trace lives somewhere observable).
                out.append(DetectorSignal(
                    detector_id=self.detector_id,
                    letter=letter,
                    fired=False,
                    confidence=float(sig.confidence),
                    signature=None,
                    verdict_action=VERDICT_ADVISORY,
                    reason=sig.reason or "no_signal",
                    extra={
                        "pair_matched": sig.pair_matched,
                        "brief_boosted": sig.brief_boosted,
                        "universal_quantifier_blocked": (
                            sig.universal_quantifier_blocked
                        ),
                    },
                ))
        return out


__all__ = ["SchemaLabelingDetector"]
