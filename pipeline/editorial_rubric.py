"""Editorial rubric for the Phase 21b editorial-pass audit.

Single source of truth for what an "editorially clean" question looks
like. The rubric describes:

  Style — APA Manual basics: clear voice, parallel structure across
  distractors, no double negatives in stems, present-tense scientific
  voice, concise framing.

  Clinical formatting — EPPP-specific conventions: named-subject
  formatting ("Dr. X consulting at Y"), patient-first language, age-
  marker consistency, contextual setting rendered consistently.

  Sensitivity — gendered pronouns where neutral framing is more
  appropriate, race/ethnicity handling matches APA standards, ableist
  or pejorative language flagged.

  Distractor structure — parallel grammatical form, consistent leading
  verb, consistent tense, length within ~30 chars of one another.

  Stem clarity — single ask (no compound questions), no "test the
  test" phrasing, no double negatives.

The rubric is delivered to Sonnet at audit time. Output classes:

  clean — no editorial concerns
  minor — fixable polish issues (e.g., parallelism break, slightly
          long distractor); advisory, doesn't block ship
  major — fundamental issue (cultural insensitivity, ambiguous stem,
          double negative impossible to parse); routes chapter to
          review for human attention

This module exposes:
  - EDITORIAL_RUBRIC_PROMPT: the rubric body interpolated into the
    audit prompt for each question
  - EDITORIAL_CLASSES: the set of valid editorial_class values
  - is_known_editorial_class(): validation helper
"""
from __future__ import annotations

CLEAN = "clean"
MINOR = "minor"
MAJOR = "major"

EDITORIAL_CLASSES: frozenset[str] = frozenset({CLEAN, MINOR, MAJOR})


def is_known_editorial_class(cls: str) -> bool:
    return cls in EDITORIAL_CLASSES


# The full prompt body. Interpolated into audit_editorial_quality.py's
# main prompt; do not include the {stem} / {options} placeholders here
# (those live in the audit script).
EDITORIAL_RUBRIC_PROMPT = """You are conducting an EDITORIAL review of a multiple-choice question for an EPPP exam-prep platform. Your role is NOT to evaluate concept correctness or distractor english_gap (a separate audit covers that). You are checking STYLE, CLINICAL FORMATTING, SENSITIVITY, and STRUCTURE.

CHECK THESE DIMENSIONS:

## Style (APA Manual basics)
- Clear, present-tense scientific voice
- No double negatives in stems
- Parallel grammatical form across distractors
- Concise framing — no padding phrases ("In this scenario, it is the case that...")
- Active voice preferred over passive when subject is clear

## Clinical formatting (EPPP conventions)
- Named-subject formatting consistent: "Dr. [Name] [verb-ing] at [Setting]" pattern
- Patient/client first ("Maria Chen, age 34, presents with...")
- Age markers consistent (numeric + comma, e.g., "age 34")
- Setting rendered consistently within a stem (e.g., don't switch "outpatient clinic" → "the office" mid-stem)

## Sensitivity (cultural + linguistic)
- Gendered pronouns: prefer neutral framing when sex/gender isn't load-bearing for the test
- Race/ethnicity: match APA standards (capitalized for racial groups; person-first when possible)
- Avoid ableist language ("crazy," "lame," "tone-deaf" used metaphorically)
- Avoid pejoratives or stigmatizing framings of clinical populations

## Distractor structure
- Parallel grammatical form (all start with the same part of speech / verb / noun-phrase shape)
- Consistent verb tense
- Length within ~30 chars of one another (no one distractor 3× longer than another)

## Stem clarity
- Single ask (one question per stem)
- No "test the test" phrasing ("Which of the following correctly describes...")
- No compound questions
- No double negatives

CLASSIFICATION RULE — assign ONE editorial_class per question:

  clean — no editorial concerns; would ship as-is in a polished test instrument
  minor — fixable polish issues (parallelism break, slightly long distractor, missed neutral framing where minor); advisory only — chapter ships, but a human polish pass would improve it
  major — fundamental issue requiring human attention before ship: cultural insensitivity, ambiguous compound stem, double negative obscuring the question, severely unparallel distractors, stigmatizing framing

DO NOT classify on:
- english_gap distractor patterns (separate audit)
- factual content correctness (a clinical SME's job, not editorial)
- which option is correct
- Bloom's-tier appropriateness (separate gate)

INSPECT THIS QUESTION:

STEM:
{stem}

OPTIONS:
{options_block}

OUTPUT FORMAT — single JSON object:

{{
  "editorial_class": "clean" | "minor" | "major",
  "issues": [
    {{"dimension": "style|clinical|sensitivity|distractor|stem",
      "severity": "minor|major",
      "description": "specific issue with file/option/text reference",
      "suggested_fix": "concrete actionable fix"}}
  ],
  "summary": "1-sentence overall verdict"
}}

If editorial_class is "clean", issues should be []."""
