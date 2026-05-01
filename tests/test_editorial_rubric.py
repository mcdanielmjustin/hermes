"""Unit tests for pipeline.editorial_rubric.

Covers the rubric module's constants, validators, and prompt content.
The actual editorial classification (Sonnet call) is integration-tested
elsewhere; this module tests the pure parts.
"""
from __future__ import annotations

import sys
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pipeline.editorial_rubric import (
    EDITORIAL_RUBRIC_PROMPT, EDITORIAL_CLASSES,
    is_known_editorial_class, CLEAN, MINOR, MAJOR,
)


# ── Class constants ─────────────────────────────────────────

def test_class_constants_match_expected_strings():
    assert CLEAN == "clean"
    assert MINOR == "minor"
    assert MAJOR == "major"


def test_editorial_classes_complete():
    assert EDITORIAL_CLASSES == frozenset({CLEAN, MINOR, MAJOR})


def test_is_known_editorial_class():
    assert is_known_editorial_class(CLEAN)
    assert is_known_editorial_class(MINOR)
    assert is_known_editorial_class(MAJOR)
    assert not is_known_editorial_class("unknown")
    assert not is_known_editorial_class("")
    assert not is_known_editorial_class(None)


# ── Rubric prompt content ───────────────────────────────────

def test_rubric_mentions_all_dimensions():
    """The rubric should explicitly cover all 5 dimensions."""
    rubric = EDITORIAL_RUBRIC_PROMPT
    assert "Style" in rubric
    assert "Clinical formatting" in rubric
    assert "Sensitivity" in rubric
    assert "Distractor structure" in rubric
    assert "Stem clarity" in rubric


def test_rubric_specifies_three_classes():
    """The rubric should instruct Sonnet to choose from clean/minor/major."""
    rubric = EDITORIAL_RUBRIC_PROMPT
    assert "clean" in rubric
    assert "minor" in rubric
    assert "major" in rubric


def test_rubric_specifies_what_NOT_to_classify_on():
    """The rubric must explicitly exclude english_gap (separate audit)
    and Bloom's tier (separate gate) and factual correctness."""
    rubric = EDITORIAL_RUBRIC_PROMPT
    assert "english_gap" in rubric
    assert "Bloom" in rubric
    # Should mention that the editorial pass doesn't judge factual content
    assert ("factual" in rubric.lower() or "SME" in rubric or
            "correctness" in rubric.lower())


def test_rubric_specifies_json_output_format():
    """The rubric must request structured JSON output."""
    rubric = EDITORIAL_RUBRIC_PROMPT
    assert "JSON" in rubric or "json" in rubric
    assert "editorial_class" in rubric
    assert "issues" in rubric
    assert "summary" in rubric


def test_rubric_has_stem_and_options_placeholders():
    """The rubric needs {stem} and {options_block} for interpolation."""
    rubric = EDITORIAL_RUBRIC_PROMPT
    assert "{stem}" in rubric
    assert "{options_block}" in rubric


def test_rubric_includes_named_subject_pattern():
    """Clinical formatting should reference the EPPP-canonical named-
    subject pattern."""
    rubric = EDITORIAL_RUBRIC_PROMPT
    # Either "Dr. [Name]" pattern or general clinical formatting language
    assert "Dr." in rubric or "named-subject" in rubric.lower()


def test_rubric_warns_about_double_negatives():
    """Stem clarity must explicitly call out double negatives."""
    rubric = EDITORIAL_RUBRIC_PROMPT
    assert "double negative" in rubric.lower()


def test_rubric_calls_out_parallelism():
    """Distractor structure must mention parallel form."""
    rubric = EDITORIAL_RUBRIC_PROMPT
    assert "parallel" in rubric.lower()


# ── Sensitivity dimension ───────────────────────────────────

def test_rubric_addresses_gendered_pronouns():
    rubric = EDITORIAL_RUBRIC_PROMPT
    assert "gender" in rubric.lower() or "pronoun" in rubric.lower()


def test_rubric_addresses_race_ethnicity():
    rubric = EDITORIAL_RUBRIC_PROMPT
    assert ("race" in rubric.lower() or "ethnicity" in rubric.lower() or
            "racial" in rubric.lower())


def test_rubric_avoids_ableist_examples():
    rubric = EDITORIAL_RUBRIC_PROMPT
    assert "ableist" in rubric.lower() or "stigmatiz" in rubric.lower()


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
