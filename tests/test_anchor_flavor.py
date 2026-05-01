"""Unit tests for pipeline.anchor_flavor flavor resolution."""
from __future__ import annotations

import sys
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pipeline.anchor_flavor import (
    flavor_for_anchor, ANCHOR_FLAVOR, DOMAIN_DEFAULT_FLAVOR, KNOWN_FLAVORS,
    _validate_flavor_completeness,
)


# ── Prefix-class dispatch ────────────────────────────────────

def test_d9_phy_resolves_to_mechanism():
    """The originating motivation: D9 is hybrid pharm+ethics; pharm
    anchors get mechanism flavor."""
    assert flavor_for_anchor("D9-PHY-006-80f8b0cb", "PETH") == "mechanism"


def test_d9_eth_resolves_to_framework():
    assert flavor_for_anchor("D9-ETH-157-a9b092b2", "PETH") == "framework"


def test_d7_phy_resolves_to_mechanism():
    assert flavor_for_anchor("D7-PHY-184-d58e5b78", "BPSY") == "mechanism"


def test_d7_lea_resolves_to_cognitive_process():
    """BPSY learning/memory anchors get a distinct flavor from physiology."""
    assert flavor_for_anchor("D7-LEA-005-d96dbb89", "BPSY") == "cognitive_process"


def test_d7_ppa_resolves_to_clinical_disease():
    assert flavor_for_anchor("D7-PPA-001-7944e52c", "BPSY") == "clinical_disease"


def test_d5_cli_resolves_to_applied_cultural():
    assert flavor_for_anchor("D5-CLI-154-4334e6da", "SOCU") == "applied_cultural"


def test_d5_soc_resolves_to_social_process():
    """SOCU CLI/SOC split: empirically distinct failure modes."""
    assert flavor_for_anchor("D5-SOC-013-2c70b19a", "SOCU") == "social_process"


def test_d8_pas_resolves_to_test_psychometric():
    assert flavor_for_anchor("D8-PAS-099-79df765b", "CASS") == "test_psychometric"


def test_d8_eth_resolves_to_framework():
    """D8 has both PAS (assessment) and ETH (assessment-ethics) prefixes."""
    assert flavor_for_anchor("D8-ETH-007-47352c8e", "CASS") == "framework"


# ── Domain-default fallback ──────────────────────────────────

def test_unknown_prefix_falls_back_to_domain_default():
    """An anchor with no prefix match falls back to the domain default."""
    # Hypothetical "D7-NEW-..." prefix (not in ANCHOR_FLAVOR yet)
    assert flavor_for_anchor("D7-NEW-001-xyz", "BPSY") == "mechanism"  # BPSY default


def test_pmet_default_is_statistical():
    assert flavor_for_anchor(None, "PMET") == "statistical"


def test_ldev_default_is_developmental_stage():
    assert flavor_for_anchor(None, "LDEV") == "developmental_stage"


def test_cpat_default_is_diagnostic_criterion():
    assert flavor_for_anchor(None, "CPAT") == "diagnostic_criterion"


def test_pthe_default_is_therapeutic_modality():
    assert flavor_for_anchor(None, "PTHE") == "therapeutic_modality"


def test_wdev_default_is_selection_psychometric():
    assert flavor_for_anchor(None, "WDEV") == "selection_psychometric"


# ── Generic fallback ─────────────────────────────────────────

def test_no_uid_no_domain_falls_back_to_generic():
    assert flavor_for_anchor(None, None) == "generic"


def test_unknown_domain_falls_back_to_generic():
    assert flavor_for_anchor(None, "XXXX") == "generic"


# ── Longest-prefix-match safety ─────────────────────────────

def test_longer_prefix_wins():
    """If a hypothetical short prefix collides with a longer specific
    one, the longer one must win. Currently no collisions exist, but
    the code uses sorted-by-length-descending to be defensive."""
    # No real collision in current data — this verifies the algorithm
    # by inspection: all current prefixes are 7 chars (e.g., 'D9-PHY-')
    # so the longest-first sort is a no-op for now. Test confirms the
    # sort doesn't break when prefixes are uniform length.
    assert flavor_for_anchor("D9-PHY-test", "PETH") == "mechanism"
    assert flavor_for_anchor("D9-ETH-test", "PETH") == "framework"


# ── Schema completeness ─────────────────────────────────────

def test_known_flavors_set_covers_anchor_flavor_values():
    """Every flavor mentioned in ANCHOR_FLAVOR or DOMAIN_DEFAULT_FLAVOR
    must be in KNOWN_FLAVORS (used as documentation/test-helper)."""
    used = set(ANCHOR_FLAVOR.values()) | set(DOMAIN_DEFAULT_FLAVOR.values())
    missing = used - KNOWN_FLAVORS
    assert not missing, f"flavors used but not in KNOWN_FLAVORS: {missing}"


def test_validate_flavor_completeness_passes():
    """Import-time validation: every flavor referenced has a wrongness
    block in pipeline.prompts._FLAVOR_WRONGNESS_BLOCKS."""
    # Should not raise
    _validate_flavor_completeness()


def test_validate_flavor_completeness_catches_missing():
    """Inject a bogus flavor and verify validation catches it."""
    from pipeline.prompts import _FLAVOR_WRONGNESS_BLOCKS
    # Save originals
    original_anchor = dict(ANCHOR_FLAVOR)
    try:
        ANCHOR_FLAVOR["TEST-BOGUS-"] = "nonexistent_flavor_for_test"
        try:
            _validate_flavor_completeness()
            assert False, "validation should have raised"
        except RuntimeError as e:
            assert "nonexistent_flavor_for_test" in str(e)
    finally:
        # Restore
        ANCHOR_FLAVOR.clear()
        ANCHOR_FLAVOR.update(original_anchor)


if __name__ == "__main__":
    import inspect
    funcs = [f for n, f in globals().items() if n.startswith("test_") and inspect.isfunction(f)]
    failures = []
    for f in funcs:
        try:
            f()
            print(f"PASS {f.__name__}")
        except AssertionError as e:
            failures.append((f.__name__, str(e)))
            print(f"FAIL {f.__name__}: {e}")
        except Exception as e:
            failures.append((f.__name__, f"{type(e).__name__}: {e}"))
            print(f"ERROR {f.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(funcs) - len(failures)}/{len(funcs)} passed")
    sys.exit(1 if failures else 0)
