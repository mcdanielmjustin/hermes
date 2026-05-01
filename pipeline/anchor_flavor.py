"""Anchor-flavor derivation for the v2 generation prompt (P6).

The prompt's "DOMAIN-PREFERRED WRONGNESS MODES" block needs to dispatch
on a per-anchor pedagogical flavor, not just `domain_code`. Some
domains contain anchor-prefix sub-classes with distinct flavors:

- D9 PETH covers BOTH pharmacology (`D9-PHY-*`, mechanism) AND ethics
  (`D9-ETH-*`, framework) within one domain code.
- D7 BPSY contains physiology (`D7-PHY-*`), learning/memory
  (`D7-LEA-*`), and pathology (`D7-PPA-*`) sub-classes.
- D5 SOCU contains clinical-cultural (`D5-CLI-*`) and social-process
  (`D5-SOC-*`) splits.

This module derives a flavor at task-build time from the anchor UID
prefix, with a per-domain default fallback. The mapping is small and
stable; it lives in code rather than as a stored brief field so that
prompts pick up changes immediately without backfill.

P6 v2 generation prompts read this flavor to select the right
"preferred wrongness mode" example block (mechanism / framework /
stage-theory / definitional / etc.). The matrix's audit-side
behavior (POLICY_BPSY, POLICY_PETH cells) is independent of this
mapping; flavor is generation-side only.
"""
from __future__ import annotations


# Prefix → flavor. Match by uid.startswith(prefix).
ANCHOR_FLAVOR: dict[str, str] = {
    # PETH (D9) hybrid split — the originating motivation for this module
    "D9-PHY-": "mechanism",
    "D9-ETH-": "framework",

    # BPSY (D7) sub-classes
    "D7-PHY-": "mechanism",
    "D7-LEA-": "cognitive_process",
    "D7-PPA-": "clinical_disease",

    # SOCU (D5) split
    "D5-CLI-": "applied_cultural",
    "D5-SOC-": "social_process",

    # CASS (D8) — assessment vs assessment-ethics
    "D8-PAS-": "test_psychometric",
    "D8-ETH-": "framework",
}


# Per-domain default when no prefix matches. Used as a safety net if a
# new prefix appears without a flavor entry.
DOMAIN_DEFAULT_FLAVOR: dict[str, str] = {
    "PMET": "statistical",
    "LDEV": "developmental_stage",
    "CPAT": "diagnostic_criterion",
    "PTHE": "therapeutic_modality",
    "WDEV": "selection_psychometric",
    "BPSY": "mechanism",
    "PETH": "framework",
    "SOCU": "applied_cultural",
    "CASS": "test_psychometric",
}


# All flavors that the v2 generation prompt has a preferred-wrongness
# block for. Used by tests and by the prompt builder to validate that
# a derived flavor maps to a known block.
KNOWN_FLAVORS: frozenset[str] = frozenset({
    "mechanism",
    "framework",
    "cognitive_process",
    "clinical_disease",
    "applied_cultural",
    "social_process",
    "test_psychometric",
    "statistical",
    "developmental_stage",
    "diagnostic_criterion",
    "therapeutic_modality",
    "selection_psychometric",
    "generic",
})


def flavor_for_anchor(anchor_uid: str | None,
                       domain_code: str | None) -> str:
    """Resolve the pedagogical flavor for an anchor.

    Resolution order:
      1. Match `anchor_uid` against `ANCHOR_FLAVOR` prefixes (longest
         first to avoid ambiguity).
      2. Fall back to `DOMAIN_DEFAULT_FLAVOR[domain_code]`.
      3. Fall back to "generic" if neither matches.
    """
    if anchor_uid:
        # Match longest prefix first so "D9-PHY-" doesn't collide with
        # a hypothetical "D9-" entry.
        for prefix in sorted(ANCHOR_FLAVOR, key=len, reverse=True):
            if anchor_uid.startswith(prefix):
                return ANCHOR_FLAVOR[prefix]
    if domain_code and domain_code in DOMAIN_DEFAULT_FLAVOR:
        return DOMAIN_DEFAULT_FLAVOR[domain_code]
    return "generic"


# ── Import-time validation ───────────────────────────────────
# Catches drift bugs early: if a developer adds a flavor to
# ANCHOR_FLAVOR or DOMAIN_DEFAULT_FLAVOR but forgets to add the
# corresponding wrongness block in pipeline.prompts:_FLAVOR_WRONGNESS_BLOCKS,
# import-time assertion fires immediately rather than silently
# falling through to "generic" at generation time.
def _validate_flavor_completeness() -> None:
    """Assert every flavor referenced in ANCHOR_FLAVOR and
    DOMAIN_DEFAULT_FLAVOR has a corresponding wrongness block in
    pipeline.prompts. Lazy-imported to avoid circular dependency.
    """
    # Lazy import — pipeline.prompts may not be fully initialized at
    # this point if anchor_flavor is imported during prompts module load.
    from pipeline.prompts import _FLAVOR_WRONGNESS_BLOCKS
    referenced = set(ANCHOR_FLAVOR.values()) | set(DOMAIN_DEFAULT_FLAVOR.values())
    missing = referenced - set(_FLAVOR_WRONGNESS_BLOCKS.keys())
    if missing:
        raise RuntimeError(
            f"anchor_flavor: flavors {sorted(missing)} are referenced in "
            f"ANCHOR_FLAVOR / DOMAIN_DEFAULT_FLAVOR but have no wrongness "
            f"block in pipeline.prompts:_FLAVOR_WRONGNESS_BLOCKS. Add the "
            f"missing block(s) so v2 generation prompts include the right "
            f"per-flavor guidance."
        )


# Run validation eagerly when this module is imported. Test runs and
# normal generation runs both pay this cost; it's a few set ops.
# A failure here is an immediate hard error — better than silent
# fall-through to "generic" guidance at generation time.
try:
    _validate_flavor_completeness()
except ImportError:
    # pipeline.prompts not available yet (circular import or test
    # context). Skip validation; tests/unit checks must verify
    # completeness explicitly.
    pass
