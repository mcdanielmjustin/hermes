"""Unit tests for pipeline.quality_taxonomy SSOT module."""
from __future__ import annotations

import sys
import pathlib

# Path setup so tests can run standalone (mirrors what production scripts do)
REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pipeline.quality_taxonomy import (
    ENGLISH_GAP, CONTENT_GAP, CLEAN, SOFT_FLAG, CLASS_NAMES,
    CANONICAL_EXAMPLES, CLASS_DEFINITIONS, DISCRIMINATING_RULE,
    is_known_class, prompt_block_for_audit, prompt_block_for_generation,
)


# ── Constants ────────────────────────────────────────────────

def test_class_constants_match_audit_prompt_strings():
    """The constants must match the JSON `class` field strings the
    audit prompt instructs Sonnet to emit. Drift would silently break
    classification parsing."""
    assert ENGLISH_GAP == "english_gap"
    assert CONTENT_GAP == "content_gap"
    assert CLEAN == "clean"
    assert SOFT_FLAG == "soft_flag"


def test_class_names_set_is_complete():
    # Phase 20b: 4-class taxonomy includes soft_flag
    assert CLASS_NAMES == frozenset({ENGLISH_GAP, CONTENT_GAP, CLEAN, SOFT_FLAG})


def test_is_known_class():
    assert is_known_class(ENGLISH_GAP)
    assert is_known_class(CONTENT_GAP)
    assert is_known_class(CLEAN)
    assert is_known_class(SOFT_FLAG)
    assert not is_known_class("unknown")
    assert not is_known_class("")


# ── Canonical examples ───────────────────────────────────────

def test_each_class_has_at_least_one_canonical_example():
    seen_classes = {ex.class_name for ex in CANONICAL_EXAMPLES}
    assert ENGLISH_GAP in seen_classes
    assert CONTENT_GAP in seen_classes
    assert CLEAN in seen_classes
    assert SOFT_FLAG in seen_classes


def test_canonical_examples_have_required_fields():
    for ex in CANONICAL_EXAMPLES:
        assert ex.class_name in CLASS_NAMES
        assert ex.stem and len(ex.stem) > 30
        assert ex.distractor and len(ex.distractor) > 20
        assert ex.why and len(ex.why) > 20


def test_class_definitions_cover_all_classes():
    for cls in (ENGLISH_GAP, CONTENT_GAP, CLEAN, SOFT_FLAG):
        assert cls in CLASS_DEFINITIONS
        assert len(CLASS_DEFINITIONS[cls]) > 50  # non-trivial paragraph


# ── Prompt-block helpers ─────────────────────────────────────

def test_audit_block_contains_all_three_classes():
    block = prompt_block_for_audit()
    # The block uses uppercase class labels in the headings
    assert "ENGLISH_GAP" in block
    assert "CONTENT_GAP" in block
    assert "CLEAN" in block


def test_audit_block_contains_canonical_example_text():
    block = prompt_block_for_audit()
    # Lester appears in the english_gap canonical example
    assert "Lester" in block
    # Intrinsic activity appears in the content_gap canonical example
    assert "intrinsic activity" in block
    # Thromboembolic appears in the clean canonical example
    assert "thromboembolic" in block


def test_audit_block_contains_classification_rule():
    block = prompt_block_for_audit()
    assert "CLASSIFICATION RULE" in block


def test_generation_block_contains_all_three_classes():
    block = prompt_block_for_generation()
    assert "ENGLISH_GAP" in block
    assert "CONTENT_GAP" in block
    assert "CLEAN" in block


def test_generation_block_contains_design_directive():
    block = prompt_block_for_generation()
    assert "DESIGN DIRECTIVE" in block
    assert "Produce content_gap" in block
    assert "Never produce english_gap" in block


def test_audit_and_generation_blocks_share_canonical_examples():
    """Drift-prevention: examples for the three FORCING classes must
    appear in both prompts. The soft_flag canonical is audit-only —
    Opus should never DESIGN a soft_flag distractor (the auditor uses
    soft_flag to express uncertainty about an existing classification,
    not as a design target)."""
    audit = prompt_block_for_audit()
    gen = prompt_block_for_generation()
    for ex in CANONICAL_EXAMPLES:
        marker = ex.stem[:40]
        # Audit must include all canonical examples (it classifies into
        # all four classes including soft_flag).
        assert marker in audit, f"audit missing canonical: {marker}"
        # Generation must include only the three forcing classes
        # (english_gap, content_gap, clean). soft_flag is audit-only.
        if ex.class_name == SOFT_FLAG:
            continue
        assert marker in gen, f"generation missing canonical: {marker}"


# ── Discriminating rule ─────────────────────────────────────

def test_discriminating_rule_mentions_all_three_classes():
    assert ENGLISH_GAP in DISCRIMINATING_RULE
    assert CONTENT_GAP in DISCRIMINATING_RULE
    assert CLEAN in DISCRIMINATING_RULE


def test_discriminating_rule_phrases_yes_no():
    """The rule must phrase the heuristic as a yes/no question with
    branching outcomes."""
    rule = DISCRIMINATING_RULE.lower()
    assert "yes" in rule
    assert "no" in rule


if __name__ == "__main__":
    # Allow `python tests/test_quality_taxonomy.py` standalone invocation
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
    print(f"\n{len(funcs) - len(failures)}/{len(funcs)} passed")
    sys.exit(1 if failures else 0)
