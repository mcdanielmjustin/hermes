"""Phase 26a — Measurement target as a first-class artifact.

The frame shift: each question carries an explicit articulation of what
knowledge state it discriminates. This makes the IMPLICIT goal of the
question EXPLICIT, so:

  - audit can verify that the question actually probes the declared target
  - generation can take the target as input and produce items aimed at it
  - curriculum design can reason over which competencies are covered
  - empirical validation has a falsifiable prediction (knowers pick X; non-knowers pick Y for reason Z)

This module defines the schema, validators, and helpers. The actual
inference (extracting targets from existing questions) lives in
``scripts/diagnosis/infer_measurement_target.py``. Target-driven
generation lives in ``scripts/diagnosis/generate_from_target.py``.

Schema design: the target is FALSIFIABLE. Vague targets like "tests
depression epidemiology" are not allowed. Targets must state
propositions a candidate either does or doesn't have, and predict
which option each knowledge-state would pick.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, field
from typing import Any


VALID_DISCRIMINATION_LEVELS = frozenset({"strong", "moderate", "weak"})


@dataclass(frozen=True)
class DistractorDiagnosis:
    """What a candidate's pick of THIS distractor reveals about their
    knowledge state. Specific, falsifiable, not vague."""
    letter: str
    candidate_state: str  # what they wrongly believe (1 sentence)
    diagnostic_meaning: str  # what their pick reveals (1 sentence)


@dataclass(frozen=True)
class MeasurementTarget:
    """The first-class measurement-instrument specification per question.

    Required fields:
      - competency_claim: 1 sentence on what the candidate must be able to do.
      - knower_state: specific propositions a knower has (1-2 sentences).
      - non_knower_state: what's missing (1-2 sentences).
      - expected_correct_letter: predicted correct option letter.
      - expected_correct_reasoning: cognitive steps a knower takes to pick.
      - distractor_diagnoses: per-distractor candidate_state + diagnostic_meaning.
      - discrimination_prediction_level: "strong" | "moderate" | "weak".
      - discrimination_prediction_rationale: why this level.
    """
    competency_claim: str
    knower_state: str
    non_knower_state: str
    expected_correct_letter: str
    expected_correct_reasoning: str
    distractor_diagnoses: list[DistractorDiagnosis] = field(default_factory=list)
    discrimination_prediction_level: str = "moderate"
    discrimination_prediction_rationale: str = ""


def to_dict(target: MeasurementTarget) -> dict[str, Any]:
    """Serialize a MeasurementTarget to a dict for JSON storage."""
    return asdict(target)


def from_dict(d: dict[str, Any]) -> MeasurementTarget:
    """Deserialize from dict. Tolerates missing optional fields."""
    diagnoses_data = d.get("distractor_diagnoses") or []
    diagnoses = [
        DistractorDiagnosis(
            letter=item.get("letter", "?"),
            candidate_state=item.get("candidate_state", ""),
            diagnostic_meaning=item.get("diagnostic_meaning", ""),
        )
        for item in diagnoses_data if isinstance(item, dict)
    ]
    return MeasurementTarget(
        competency_claim=d.get("competency_claim", ""),
        knower_state=d.get("knower_state", ""),
        non_knower_state=d.get("non_knower_state", ""),
        expected_correct_letter=d.get("expected_correct_letter", ""),
        expected_correct_reasoning=d.get("expected_correct_reasoning", ""),
        distractor_diagnoses=diagnoses,
        discrimination_prediction_level=d.get("discrimination_prediction_level", "moderate"),
        discrimination_prediction_rationale=d.get("discrimination_prediction_rationale", ""),
    )


def is_valid(d: dict[str, Any]) -> tuple[bool, str]:
    """Return (True, "ok") if the target dict has all required fields
    populated; else (False, reason). Helps catch under-specified targets
    that would be useless for audit/generation downstream.
    """
    if not isinstance(d, dict):
        return False, f"target is not a dict (got {type(d).__name__})"
    required = (
        "competency_claim", "knower_state", "non_knower_state",
        "expected_correct_letter", "expected_correct_reasoning",
        "distractor_diagnoses", "discrimination_prediction_level",
        "discrimination_prediction_rationale",
    )
    for field_name in required:
        if field_name not in d:
            return False, f"missing field: {field_name}"
    # String fields must be non-empty
    for field_name in (
        "competency_claim", "knower_state", "non_knower_state",
        "expected_correct_letter", "expected_correct_reasoning",
        "discrimination_prediction_rationale",
    ):
        v = d.get(field_name)
        if not isinstance(v, str) or not v.strip():
            return False, f"empty or non-string: {field_name}"
    # discrimination level must be one of the canonical values
    level = d.get("discrimination_prediction_level")
    if level not in VALID_DISCRIMINATION_LEVELS:
        return False, (
            f"invalid discrimination_prediction_level: {level!r} "
            f"(must be one of {sorted(VALID_DISCRIMINATION_LEVELS)})"
        )
    # distractor_diagnoses must be a non-empty list
    diagnoses = d.get("distractor_diagnoses")
    if not isinstance(diagnoses, list) or not diagnoses:
        return False, "distractor_diagnoses must be a non-empty list"
    for i, diag in enumerate(diagnoses):
        if not isinstance(diag, dict):
            return False, f"distractor_diagnoses[{i}] is not a dict"
        for sub_field in ("letter", "candidate_state", "diagnostic_meaning"):
            v = diag.get(sub_field)
            if not isinstance(v, str) or not v.strip():
                return False, f"distractor_diagnoses[{i}].{sub_field} empty or non-string"
    return True, "ok"


# ── Canonical example for prompt anchoring ────────────────

CANONICAL_EXAMPLE = MeasurementTarget(
    competency_claim=(
        "The candidate can identify the term that names the developmental "
        "epidemiological pattern of sex differences in major depressive "
        "disorder (similar prepubertal rates, divergence at puberty, "
        "approaching 2:1 female-to-male ratio in adulthood)."
    ),
    knower_state=(
        "Knows the canonical developmental MDD epidemiology: roughly equal "
        "rates in childhood, female rates rise during puberty, ~2:1 "
        "female-to-male ratio by adulthood. Knows this pattern is named "
        "'pubertal divergence'."
    ),
    non_knower_state=(
        "May know that adult depression is more common in females but does "
        "not know the developmental timing or the canonical name for the "
        "pattern."
    ),
    expected_correct_letter="C",
    expected_correct_reasoning=(
        "A knower recognizes the term 'pubertal divergence' as naming the "
        "specific timing pattern (childhood-equivalent → puberty-emerging) "
        "and selects C without needing the specific 2:1 ratio printed in "
        "the stem."
    ),
    distractor_diagnoses=[
        DistractorDiagnosis(
            letter="A",
            candidate_state=(
                "Believes the female predominance is established from "
                "early childhood onward, without the pubertal-onset detail."
            ),
            diagnostic_meaning=(
                "Reveals partial knowledge of the adult ratio without "
                "developmental timing."
            ),
        ),
        DistractorDiagnosis(
            letter="B",
            candidate_state=(
                "Believes the sex difference reverses across the lifespan "
                "(male-predominant in childhood → female-predominant in "
                "adulthood)."
            ),
            diagnostic_meaning=(
                "Confuses MDD with conduct disorder or ADHD, which DO "
                "show male predominance in childhood."
            ),
        ),
        DistractorDiagnosis(
            letter="D",
            candidate_state=(
                "Believes lifetime prevalence is equivalent across sexes."
            ),
            diagnostic_meaning=(
                "Reveals lack of basic sex-difference knowledge in MDD "
                "epidemiology."
            ),
        ),
    ],
    discrimination_prediction_level="strong",
    discrimination_prediction_rationale=(
        "Each distractor maps to a specific cognitive error documented "
        "in epidemiology textbooks. A knower selects C deterministically; "
        "non-knowers distribute across A/B/D depending on their specific "
        "gap. Predicted point-biserial correlation in the 0.30-0.45 range."
    ),
)
