"""Unit tests for pipeline.distractor_policy matrix + resolve()."""
from __future__ import annotations

import sys
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pipeline.distractor_policy import (
    Cell, DEFAULT, GATE_ACTIONS, CORRECTION_STRATEGIES,
    POLICY_BPSY, POLICY_PETH, DOMAIN_OVERRIDES,
    resolve, get_resolution_stats, reset_resolution_stats,
)


# ── Cell dataclass ───────────────────────────────────────────

def test_cell_accepts_valid_enums():
    c = Cell(gate_action="audit", correction_strategy="judgment_error")
    assert c.gate_action == "audit"
    assert c.correction_strategy == "judgment_error"


def test_cell_rejects_invalid_gate_action():
    try:
        Cell(gate_action="bogus", correction_strategy="judgment_error")
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "gate_action" in str(e)


def test_cell_rejects_invalid_correction_strategy():
    try:
        Cell(gate_action="audit", correction_strategy="bogus")
        assert False, "should have raised ValueError"
    except ValueError as e:
        assert "correction_strategy" in str(e)


def test_cell_is_frozen():
    c = Cell(gate_action="audit", correction_strategy="judgment_error")
    try:
        c.gate_action = "strict"  # type: ignore
        assert False, "frozen dataclass should reject mutation"
    except Exception:
        pass


def test_default_cell_has_strict_action():
    """DEFAULT must be the safe fallback — strict + swap-for-content."""
    assert DEFAULT.gate_action == "strict"
    assert DEFAULT.correction_strategy == "swap_for_content_distractor"


# ── Enum completeness ───────────────────────────────────────

def test_gate_actions_contains_currently_used_values():
    assert "audit" in GATE_ACTIONS
    assert "strict" in GATE_ACTIONS
    assert "framework_aware" in GATE_ACTIONS
    # Reserved (currently dead but documented):
    assert "skip" in GATE_ACTIONS
    assert "permissive" in GATE_ACTIONS


def test_correction_strategies_complete():
    assert "swap_for_content_distractor" in CORRECTION_STRATEGIES
    assert "judgment_error" in CORRECTION_STRATEGIES
    assert "rewrite_stem_to_observation" in CORRECTION_STRATEGIES
    assert "mechanism_inversion" in CORRECTION_STRATEGIES
    assert "framework_misapplication" in CORRECTION_STRATEGIES


# ── POLICY contents ─────────────────────────────────────────

def test_policy_bpsy_has_expected_cells():
    """Sanity check: BPSY matrix has the cells we authored in P1+."""
    expected_keys = [
        (4, "integrated", "contrast_prompt"),
        (4, "integrated", "subtle_error"),
        (4, "integrated", "competing_evidence"),
        (4, "integrated", "integration"),
        (2, "anchor_grounded", "paraphrase"),
        (2, "integrated", "simple_application"),
        (3, "integrated", "clinical_vignette"),
        (3, "integrated", "scenario_completion"),
    ]
    for key in expected_keys:
        assert key in POLICY_BPSY, f"missing BPSY cell: {key}"


def test_policy_peth_uses_framework_aware_throughout():
    """All PETH cells should use framework_aware action."""
    for key, cell in POLICY_PETH.items():
        assert cell.gate_action == "framework_aware", \
            f"PETH cell {key} should be framework_aware, got {cell.gate_action}"


def test_domain_overrides_includes_bpsy_and_peth():
    assert "BPSY" in DOMAIN_OVERRIDES
    assert "PETH" in DOMAIN_OVERRIDES
    assert DOMAIN_OVERRIDES["BPSY"] is POLICY_BPSY
    assert DOMAIN_OVERRIDES["PETH"] is POLICY_PETH


# ── resolve() behavior ──────────────────────────────────────

def test_resolve_hits_bpsy_cell():
    """T4 contrast_prompt in BPSY should resolve to the audit cell."""
    reset_resolution_stats()
    cell = resolve(tier=4, domain_code="BPSY", source_type="integrated",
                   stem_pattern="contrast_prompt")
    assert cell.gate_action == "audit"
    assert cell.correction_strategy == "judgment_error"


def test_resolve_hits_peth_cell():
    """T3 clinical_vignette in PETH should resolve to framework_aware."""
    cell = resolve(tier=3, domain_code="PETH", source_type="integrated",
                   stem_pattern="clinical_vignette")
    assert cell.gate_action == "framework_aware"
    assert cell.correction_strategy == "framework_misapplication"


def test_resolve_falls_through_to_default_for_unknown_domain():
    """A2.5: when a (tier, source, pattern) has no matching cell AND
    the tier is recognized, resolve() now returns the tier-keyed default
    (DEFAULT_T4 here) rather than the bare DEFAULT. Behavior preserved
    in `gate_action` ('strict' in both) but the tier-default carries the
    detector override threshold."""
    from pipeline.distractor_policy import DEFAULT_T4
    cell = resolve(tier=4, domain_code="UNKNOWN", source_type="integrated",
                   stem_pattern="contrast_prompt")
    assert cell.gate_action == "strict"
    assert cell == DEFAULT_T4


def test_resolve_falls_through_to_default_for_unmapped_cell():
    """A (tier, source, pattern) with no matching cell falls to the
    tier-keyed default (A2.5 change)."""
    # T1 BPSY direct_definition — not currently in matrix
    from pipeline.distractor_policy import DEFAULT_T1_T2
    cell = resolve(tier=1, domain_code="BPSY", source_type="anchor_grounded",
                   stem_pattern="direct_definition")
    assert cell == DEFAULT_T1_T2


def test_resolve_with_all_none_returns_default():
    cell = resolve()
    assert cell == DEFAULT


def test_resolve_normalizes_none_inputs():
    """None values for any input should normalize and not crash."""
    cell = resolve(tier=None, domain_code=None, source_type=None,
                   stem_pattern=None)
    assert cell == DEFAULT


# ── Resolution counter ──────────────────────────────────────

def test_resolution_counter_records_calls():
    reset_resolution_stats()
    resolve(tier=4, domain_code="BPSY", source_type="integrated",
            stem_pattern="contrast_prompt", misconception_type="similar_property")
    stats = get_resolution_stats()
    expected_key = (4, "BPSY", "integrated", "contrast_prompt", "similar_property")
    assert expected_key in stats
    assert stats[expected_key] == 1


def test_resolution_counter_increments_on_repeat():
    reset_resolution_stats()
    for _ in range(3):
        resolve(tier=2, domain_code="BPSY", source_type="anchor_grounded",
                stem_pattern="paraphrase")
    stats = get_resolution_stats()
    expected_key = (2, "BPSY", "anchor_grounded", "paraphrase", "unknown")
    assert stats[expected_key] == 3


def test_reset_clears_counter():
    reset_resolution_stats()
    resolve(tier=4, domain_code="BPSY", source_type="integrated",
            stem_pattern="contrast_prompt")
    assert len(get_resolution_stats()) > 0
    reset_resolution_stats()
    assert len(get_resolution_stats()) == 0


def test_resolution_counter_thread_safe_under_concurrency():
    """Multiple threads calling resolve simultaneously shouldn't lose
    counts. Under the GIL, simple ops are usually safe, but the lock
    formalizes the contract."""
    import threading
    reset_resolution_stats()
    iterations_per_thread = 100
    n_threads = 8

    def worker():
        for _ in range(iterations_per_thread):
            resolve(tier=4, domain_code="BPSY", source_type="integrated",
                    stem_pattern="contrast_prompt")

    threads = [threading.Thread(target=worker) for _ in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    stats = get_resolution_stats()
    expected_key = (4, "BPSY", "integrated", "contrast_prompt", "unknown")
    expected_count = n_threads * iterations_per_thread
    assert stats.get(expected_key, 0) == expected_count, \
        f"expected {expected_count}, got {stats.get(expected_key, 0)}"


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
