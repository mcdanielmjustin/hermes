"""Unit tests for scripts.audit_stem_contradictions.aggregate_classifications.

Phase 21a quorum logic: pure function, no API calls. Tests cover:
- Single-pass passthrough (n_passes=1 case)
- Unanimous classifications (3/3 same class)
- Majority votes (2/3 vs 1/3)
- Tie-breaks (1/1/1 split → most conservative class wins)
- Soft_flag preservation (any pass = soft_flag → keep)
- Multi-distractor coverage (different letters, different votes)
- Edge cases (empty passes, missing letters)
"""
from __future__ import annotations

import sys
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_stem_contradictions import aggregate_classifications


def _entry(letter, cls, **extras):
    """Helper: build a classification entry dict."""
    e = {"letter": letter, "class": cls, "distractor_text": f"d-{letter}",
         "explanation": f"why-{cls}"}
    if cls in ("english_gap", "content_gap"):
        e["contradicted_stem_fact"] = f"fact-{cls}"
    if cls == "soft_flag":
        e["ambiguous_between"] = extras.get("ambiguous_between",
                                             ["english_gap", "content_gap"])
    return e


# ── Single-pass passthrough ──────────────────────────────────

def test_single_pass_returns_pass_unchanged():
    pass_one = [_entry("A", "english_gap"), _entry("B", "content_gap")]
    result = aggregate_classifications([pass_one])
    assert len(result) == 2
    a = next(e for e in result if e["letter"] == "A")
    b = next(e for e in result if e["letter"] == "B")
    assert a["class"] == "english_gap"
    assert b["class"] == "content_gap"


# ── Unanimous votes ──────────────────────────────────────────

def test_unanimous_english_gap():
    passes = [
        [_entry("A", "english_gap")],
        [_entry("A", "english_gap")],
        [_entry("A", "english_gap")],
    ]
    result = aggregate_classifications(passes)
    assert len(result) == 1
    assert result[0]["class"] == "english_gap"
    assert result[0]["_pass_count"] == {"english_gap": 3}


def test_unanimous_clean():
    passes = [
        [_entry("A", "clean")],
        [_entry("A", "clean")],
        [_entry("A", "clean")],
    ]
    result = aggregate_classifications(passes)
    assert result[0]["class"] == "clean"


# ── Majority votes ───────────────────────────────────────────

def test_majority_two_vs_one():
    passes = [
        [_entry("A", "english_gap")],
        [_entry("A", "english_gap")],
        [_entry("A", "content_gap")],
    ]
    result = aggregate_classifications(passes)
    assert result[0]["class"] == "english_gap"
    assert result[0]["_pass_count"] == {"english_gap": 2, "content_gap": 1}


def test_majority_content_gap_wins_over_clean():
    passes = [
        [_entry("A", "content_gap")],
        [_entry("A", "content_gap")],
        [_entry("A", "clean")],
    ]
    result = aggregate_classifications(passes)
    assert result[0]["class"] == "content_gap"


# ── Tie-breaks (conservative order) ──────────────────────────

def test_tie_three_classes_picks_most_conservative():
    """1-1-1 split → english_gap (most conservative)."""
    passes = [
        [_entry("A", "english_gap")],
        [_entry("A", "content_gap")],
        [_entry("A", "clean")],
    ]
    result = aggregate_classifications(passes)
    assert result[0]["class"] == "english_gap"


def test_tie_two_way_content_vs_clean():
    """Even split between content_gap and clean → content_gap wins."""
    passes = [
        [_entry("A", "content_gap")],
        [_entry("A", "clean")],
    ]
    result = aggregate_classifications(passes)
    assert result[0]["class"] == "content_gap"


def test_tie_two_way_eg_vs_clean():
    """english_gap and clean tied → english_gap wins (conservative)."""
    passes = [
        [_entry("A", "english_gap")],
        [_entry("A", "clean")],
    ]
    result = aggregate_classifications(passes)
    assert result[0]["class"] == "english_gap"


# ── Soft_flag preservation ───────────────────────────────────

def test_any_soft_flag_preserved():
    """If any pass produced soft_flag, the aggregate is soft_flag."""
    passes = [
        [_entry("A", "english_gap")],
        [_entry("A", "english_gap")],
        [_entry("A", "soft_flag", ambiguous_between=["english_gap", "clean"])],
    ]
    result = aggregate_classifications(passes)
    assert result[0]["class"] == "soft_flag"
    assert result[0]["ambiguous_between"] == ["english_gap", "clean"]


def test_soft_flag_majority_still_soft_flag():
    """2 of 3 soft_flag → still soft_flag."""
    passes = [
        [_entry("A", "soft_flag")],
        [_entry("A", "soft_flag")],
        [_entry("A", "english_gap")],
    ]
    result = aggregate_classifications(passes)
    assert result[0]["class"] == "soft_flag"


# ── Multi-distractor coverage ───────────────────────────────

def test_multiple_distractors_independent_aggregation():
    passes = [
        [_entry("A", "english_gap"), _entry("B", "clean"), _entry("C", "content_gap")],
        [_entry("A", "english_gap"), _entry("B", "clean"), _entry("C", "english_gap")],
        [_entry("A", "content_gap"), _entry("B", "clean"), _entry("C", "english_gap")],
    ]
    result = aggregate_classifications(passes)
    by_letter = {e["letter"]: e for e in result}
    assert by_letter["A"]["class"] == "english_gap"  # 2-1 majority
    assert by_letter["B"]["class"] == "clean"        # unanimous
    assert by_letter["C"]["class"] == "english_gap"  # 2-1 majority


def test_letters_not_in_all_passes_still_aggregate():
    """If pass 2 missed letter B but passes 1 and 3 had it,
    aggregation still works on the available votes."""
    passes = [
        [_entry("A", "english_gap"), _entry("B", "clean")],
        [_entry("A", "english_gap")],  # B missing
        [_entry("A", "english_gap"), _entry("B", "clean")],
    ]
    result = aggregate_classifications(passes)
    by_letter = {e["letter"]: e for e in result}
    assert by_letter["A"]["class"] == "english_gap"
    assert by_letter["B"]["class"] == "clean"
    # B has only 2 votes (passes 1 + 3); pass count reflects this
    assert by_letter["B"]["_pass_count"] == {"clean": 2}


# ── Edge cases ──────────────────────────────────────────────

def test_empty_passes_returns_empty():
    assert aggregate_classifications([]) == []


def test_passes_with_empty_classifications():
    """If all passes have empty classifications, result is empty."""
    assert aggregate_classifications([[], [], []]) == []


def test_contradicted_stem_fact_preserved_for_english_gap():
    passes = [
        [_entry("A", "english_gap")],
        [_entry("A", "english_gap")],
    ]
    result = aggregate_classifications(passes)
    assert "contradicted_stem_fact" in result[0]
    assert result[0]["contradicted_stem_fact"] == "fact-english_gap"


def test_contradicted_stem_fact_omitted_for_clean():
    passes = [
        [_entry("A", "clean")],
        [_entry("A", "clean")],
    ]
    result = aggregate_classifications(passes)
    # clean entries don't have contradicted_stem_fact in our representation
    assert result[0].get("contradicted_stem_fact") is None or \
           "contradicted_stem_fact" not in result[0]


# ── Pass count introspection ────────────────────────────────

def test_pass_count_reflects_actual_votes():
    passes = [
        [_entry("A", "english_gap")],
        [_entry("A", "content_gap")],
        [_entry("A", "english_gap")],
        [_entry("A", "clean")],
    ]
    result = aggregate_classifications(passes)
    assert result[0]["_pass_count"] == {
        "english_gap": 2,
        "content_gap": 1,
        "clean": 1,
    }


if __name__ == "__main__":
    import inspect
    funcs = [f for n, f in globals().items()
             if n.startswith("test_") and inspect.isfunction(f)]
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
