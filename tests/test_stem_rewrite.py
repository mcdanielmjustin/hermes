"""Unit tests for Phase 20d stem-rewrite dispatch + heuristic.

The over-specification signal (``should_rewrite_stem``) is a pure
function and exhaustively unit-tested here. The ``rewrite_stem``
async path that calls Sonnet is integration-tested separately
(it makes a real API call); the unit layer only covers the
threshold logic and the dispatch contract in ``fix_question``.
"""
from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_stem_contradictions import should_rewrite_stem  # noqa: E402


def _q(n_distractors: int) -> dict:
    """Build a synthetic question with ``n_distractors`` distractors and
    one correct option."""
    options = [
        {"letter": chr(ord("A") + i), "is_correct": False}
        for i in range(n_distractors)
    ]
    options.append({"letter": chr(ord("A") + n_distractors), "is_correct": True})
    return {"options": options}


def _audit(n_flagged: int, start: str = "A") -> dict:
    """Build an audit_result with ``n_flagged`` distractors flagged."""
    return {
        "flagged_distractors": [
            {"letter": chr(ord(start) + i)} for i in range(n_flagged)
        ]
    }


# ── Threshold semantics ────────────────────────────────────

def test_all_distractors_flagged_fires():
    """3/3 flagged is the canonical CPAT E-02 case."""
    assert should_rewrite_stem(_audit(3), _q(3)) is True


def test_majority_flagged_fires_at_default_threshold():
    """2/3 = 66.7% >= 50% threshold."""
    assert should_rewrite_stem(_audit(2), _q(3)) is True


def test_minority_flagged_does_not_fire():
    """1/3 = 33% < 50% threshold."""
    assert should_rewrite_stem(_audit(1), _q(3)) is False


def test_no_flags_does_not_fire():
    assert should_rewrite_stem({"flagged_distractors": []}, _q(3)) is False


def test_missing_flagged_key_does_not_fire():
    assert should_rewrite_stem({}, _q(3)) is False


def test_no_distractors_does_not_fire():
    """Defensive: malformed question with no distractors."""
    q = {"options": [{"letter": "A", "is_correct": True}]}
    assert should_rewrite_stem(_audit(0), q) is False


def test_no_options_does_not_fire():
    """Defensive: question missing options entirely."""
    assert should_rewrite_stem(_audit(0), {}) is False


# ── Custom thresholds ─────────────────────────────────────

def test_custom_threshold_higher():
    """threshold=1.0 requires ALL distractors flagged."""
    assert should_rewrite_stem(_audit(3), _q(3), threshold=1.0) is True
    assert should_rewrite_stem(_audit(2), _q(3), threshold=1.0) is False


def test_custom_threshold_lower():
    """threshold=0.34 fires on 1/3 (33% < 34% so still false; 2/3 yes)."""
    assert should_rewrite_stem(_audit(1), _q(3), threshold=0.34) is False
    assert should_rewrite_stem(_audit(2), _q(3), threshold=0.34) is True


# ── Edge cases ────────────────────────────────────────────

def test_4_distractor_question():
    """If a question has 4 distractors (rare), 2/4 = 50% should fire."""
    assert should_rewrite_stem(_audit(2), _q(4)) is True
    assert should_rewrite_stem(_audit(1), _q(4)) is False


def test_fires_regardless_of_which_letters_are_flagged():
    """The threshold is count-based; specific letters don't matter."""
    # B and D flagged out of 4 distractors A-D
    audit = {"flagged_distractors": [{"letter": "B"}, {"letter": "D"}]}
    assert should_rewrite_stem(audit, _q(4)) is True


# ── Standalone runner ──────────────────────────────────────

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
