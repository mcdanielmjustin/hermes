"""Phase A6 — Fixer Registry.

Each `Fixer` handles ONE detector signature. When the audit's flagged
distractors include a known signature, the fix dispatch routes to the
matching fixer instead of the generic `self_critique` /
`fix_question` path.

Why this is needed (the regression A6 eliminates):

Today's fix-and-export run on the E2E-100 bad set showed `dq_major`
INCREASED 8 → 15 because `pipeline/self_critique.py`'s prompt is
english_gap-focused. When run on a flagged distractor, it rewrites the
text to remove the lexical contradiction. But the rewrite can introduce
NEW ambiguity — overlap-zone phrasing, defensible alternatives,
mismatched cognitive demand. Surface-fix breaks dq invariants.

A6 routes each detector signature to a specialized fixer that:
  - Touches ONLY the option flagged by the signature
  - Knows what the signature targets (so it doesn't introduce other
    failure modes while fixing the named one)
  - Uses deterministic edits where possible; minimal LLM rewrite where not
  - PRESERVES is_correct invariance (the option marked correct stays
    correct; never swap which option is correct)

Public API:

  class Fixer(ABC):
      fixer_id: str
      handles_signatures: tuple[str, ...]
      async def fix(self, client, question, signal, semaphore) -> dict

  class FixerRegistry:
      register(fixer)
      fixer_for_signature(signature) -> Fixer | None

  create_fixer_registry() -> FixerRegistry
      Returns the canonical registry pre-populated with goliath's fixers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable

from pipeline.detectors import DetectorSignal


class Fixer(ABC):
    """Abstract base for every signature-routed fixer.

    Subclasses declare:
      - fixer_id (class attribute): short stable identifier.
      - handles_signatures (class attribute): tuple of detector signature
        strings this fixer handles. The registry indexes by signature.

    Subclasses implement:
      - fix(client, question, signal, semaphore) -> patched_question

    The patched question MUST preserve:
      - question_id
      - is_correct on the correct option (never swap)
      - all 4 options (no fewer, no more)
      - difficulty_tier
    The fixer MAY modify:
      - the flagged option's text and explanation
      - the stem ONLY if the signature explicitly requires (e.g.,
        numeric_overlap_fixer adjusts stem age band)

    On failure (LLM error, parse failure, invariant violation), the
    fixer returns the ORIGINAL question unchanged. This is by design:
    a fix that breaks invariants is worse than no fix.
    """

    fixer_id: str = "base_fixer"
    handles_signatures: tuple[str, ...] = ()

    @abstractmethod
    async def fix(
        self,
        client,
        question: dict,
        signal: DetectorSignal,
        semaphore,
    ) -> dict:
        """Return a patched question, or the original on failure."""
        raise NotImplementedError


class FixerRegistry:
    """Maps detector signature → Fixer instance."""

    def __init__(self) -> None:
        self._by_signature: dict[str, Fixer] = {}
        self._by_fixer_id: dict[str, Fixer] = {}

    def register(self, fixer: Fixer) -> None:
        if not fixer.fixer_id:
            raise ValueError("fixer must have a non-empty fixer_id")
        if not fixer.handles_signatures:
            raise ValueError(
                f"fixer {fixer.fixer_id!r} declares no signatures; "
                f"nothing to register"
            )
        self._by_fixer_id[fixer.fixer_id] = fixer
        for sig in fixer.handles_signatures:
            self._by_signature[sig] = fixer

    def fixer_for_signature(self, signature: str | None) -> Fixer | None:
        """Return the fixer registered for `signature`, or None if no
        fixer handles it."""
        if not signature:
            return None
        return self._by_signature.get(signature)

    def all_fixers(self) -> Iterable[Fixer]:
        return list(self._by_fixer_id.values())


def create_fixer_registry() -> FixerRegistry:
    """Build the canonical fixer registry pre-populated with goliath's
    signature-routed fixers.

    Phase A6 ships:
      - universal_quantifier_fixer
      - laterality_fixer
      - schema_labeling_fixer
      - numeric_overlap_fixer

    Detectors not represented here (imperative_lead, meta_evaluative,
    lead_form_parallelism, defensible_alternative) fall through to the
    legacy fix_question / self_critique path until their fixers are
    written.
    """
    from .universal_quantifier_fixer import UniversalQuantifierFixer
    from .laterality_fixer import LateralityFixer
    from .schema_labeling_fixer import SchemaLabelingFixer
    from .numeric_overlap_fixer import NumericOverlapFixer
    from .ambiguity_fixer import AmbiguityFixer
    # Phase B2 — close english_gap_scanner signature coverage.
    from .numeric_ratio_fixer import NumericRatioFixer
    from .stage_timing_fixer import StageTimingFixer

    registry = FixerRegistry()
    registry.register(UniversalQuantifierFixer())
    registry.register(LateralityFixer())
    registry.register(SchemaLabelingFixer())
    registry.register(NumericOverlapFixer())
    registry.register(AmbiguityFixer())
    registry.register(NumericRatioFixer())
    registry.register(StageTimingFixer())
    return registry


__all__ = ["Fixer", "FixerRegistry", "create_fixer_registry"]
