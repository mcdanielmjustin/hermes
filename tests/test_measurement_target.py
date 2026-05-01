"""Unit tests for `pipeline.measurement_target`.

Phase 26a foundation. Schema validation + serialization round-trip.
"""
from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pipeline.measurement_target import (  # noqa: E402
    MeasurementTarget, DistractorDiagnosis,
    to_dict, from_dict, is_valid,
    CANONICAL_EXAMPLE, VALID_DISCRIMINATION_LEVELS,
)


def _good_dict():
    return {
        "competency_claim": "Test claim.",
        "knower_state": "Knows X.",
        "non_knower_state": "Doesn't know X.",
        "expected_correct_letter": "B",
        "expected_correct_reasoning": "Picks B because they know X.",
        "distractor_diagnoses": [
            {"letter": "A", "candidate_state": "thinks Y", "diagnostic_meaning": "confuses X with Y"},
            {"letter": "C", "candidate_state": "thinks Z", "diagnostic_meaning": "missing concept"},
            {"letter": "D", "candidate_state": "thinks W", "diagnostic_meaning": "incomplete"},
        ],
        "discrimination_prediction_level": "strong",
        "discrimination_prediction_rationale": "Specific cognitive errors per distractor.",
    }


# ── Schema validation ─────────────────────────────────────

def test_valid_target_passes():
    ok, reason = is_valid(_good_dict())
    assert ok, f"valid target failed: {reason}"


def test_canonical_example_validates():
    """The canonical example used in prompts must itself validate."""
    d = to_dict(CANONICAL_EXAMPLE)
    ok, reason = is_valid(d)
    assert ok, f"canonical example failed: {reason}"


def test_missing_competency_claim_fails():
    d = _good_dict()
    del d["competency_claim"]
    ok, reason = is_valid(d)
    assert not ok
    assert "competency_claim" in reason


def test_empty_competency_claim_fails():
    d = _good_dict()
    d["competency_claim"] = ""
    ok, reason = is_valid(d)
    assert not ok


def test_whitespace_only_field_fails():
    d = _good_dict()
    d["knower_state"] = "   "
    ok, reason = is_valid(d)
    assert not ok


def test_invalid_discrimination_level_fails():
    d = _good_dict()
    d["discrimination_prediction_level"] = "incredible"
    ok, reason = is_valid(d)
    assert not ok
    assert "discrimination_prediction_level" in reason


def test_each_valid_discrimination_level_works():
    for level in VALID_DISCRIMINATION_LEVELS:
        d = _good_dict()
        d["discrimination_prediction_level"] = level
        ok, reason = is_valid(d)
        assert ok, f"level {level} failed: {reason}"


def test_empty_distractor_diagnoses_fails():
    d = _good_dict()
    d["distractor_diagnoses"] = []
    ok, reason = is_valid(d)
    assert not ok
    assert "distractor_diagnoses" in reason


def test_distractor_diagnosis_missing_field_fails():
    d = _good_dict()
    d["distractor_diagnoses"][0] = {"letter": "A"}  # missing other fields
    ok, reason = is_valid(d)
    assert not ok


def test_non_dict_input_fails():
    ok, reason = is_valid("not a dict")
    assert not ok


# ── Serialization round-trip ──────────────────────────────

def test_round_trip_preserves_canonical_example():
    d = to_dict(CANONICAL_EXAMPLE)
    restored = from_dict(d)
    assert restored.competency_claim == CANONICAL_EXAMPLE.competency_claim
    assert restored.expected_correct_letter == CANONICAL_EXAMPLE.expected_correct_letter
    assert len(restored.distractor_diagnoses) == len(CANONICAL_EXAMPLE.distractor_diagnoses)
    for orig, new in zip(CANONICAL_EXAMPLE.distractor_diagnoses, restored.distractor_diagnoses):
        assert orig.letter == new.letter
        assert orig.candidate_state == new.candidate_state
        assert orig.diagnostic_meaning == new.diagnostic_meaning


def test_from_dict_tolerates_extra_keys():
    d = _good_dict()
    d["extra_field"] = "ignored"
    # Should still produce a valid MeasurementTarget (extra fields silently dropped)
    restored = from_dict(d)
    assert restored.competency_claim == "Test claim."


def test_from_dict_tolerates_missing_optional_fields():
    """Build a partial dict; from_dict provides defaults."""
    d = {
        "competency_claim": "x",
        "knower_state": "y",
        "non_knower_state": "z",
        "expected_correct_letter": "A",
        "expected_correct_reasoning": "w",
        # distractor_diagnoses, discrimination fields default
    }
    target = from_dict(d)
    assert target.competency_claim == "x"
    assert target.distractor_diagnoses == []
    assert target.discrimination_prediction_level == "moderate"


# ── Standalone runner ─────────────────────────────────────

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
