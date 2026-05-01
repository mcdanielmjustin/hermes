"""Phase 27-29 — Diagnostic quality rubric (factual + ambiguity + tier-fit).

Production multi-criterion audit targeting the three quality dimensions
that the path-B pilot found systematically missing in goliath's existing
audit/scanner/editorial layers:

  - factual_correctness: stem, correct answer, distractor explanations
    are factually accurate. Path-B found 10% factual error rate.
  - ambiguity: there is exactly one defensible correct answer. Path-B
    found ~16% with multiple defensible interpretations.
  - tier_fit: cognitive demand of selecting the correct answer matches
    the labeled Bloom's tier. Path-B found 60% scope_creep flags.

Single Sonnet 4.6 call per question evaluates all three dimensions
(~$0.005/Q vs ~$0.015 for separate audits). Routes to review if ANY
criterion scores below 3 of 5.

Sibling to Phase 21b editorial_rubric and Phase Δ english_gap audit;
distinct dimensions, each handling a quality concern the others miss.
"""
from __future__ import annotations

CLEAN = "clean"
MINOR = "minor"
MAJOR = "major"
DIAGNOSTIC_QUALITY_CLASSES: frozenset[str] = frozenset({CLEAN, MINOR, MAJOR})


def is_known_diagnostic_quality_class(cls: str) -> bool:
    return cls in DIAGNOSTIC_QUALITY_CLASSES


# Single rubric prompt; interpolated by audit_diagnostic_quality.py
DIAGNOSTIC_QUALITY_RUBRIC_PROMPT = """You are conducting a DIAGNOSTIC QUALITY review of a multiple-choice question for an EPPP exam-prep platform. Your role is NOT to evaluate english_gap (separate audit) or surface editorial style (separate audit). You ARE checking three specific dimensions that determine whether the item FUNCTIONS as a diagnostic instrument:

CHECK THESE THREE DIMENSIONS:

## 1. Factual correctness (1-5 scale)

Are the factual claims in stem, correct answer, and distractor explanations accurate per current professional standards (DSM-5-TR, APA Standards, ASPPB conventions, peer-reviewed literature)?

  5 — All factual claims accurate; no errors detectable.
  4 — Accurate; minor terminology preferences only.
  3 — Mostly accurate; one minor imprecision (specific to citation style or non-load-bearing claim).
  2 — Material factual issue: stem/correct answer/distractor explanation contains a claim a domain expert would challenge as wrong.
  1 — Multiple factual errors; question would be retracted in production.

Common failures: incorrect drug schedules, misnamed disorders, wrong DSM criteria, outdated terminology, scope claims that don't survive expert review.

## 2. Ambiguity (1-5 scale)

Is there exactly ONE defensible correct answer? Could a knowledgeable candidate reasonably argue for a different option? Items with ambiguous correct answers fail psychometrically because expertise correlates with seeing the alternative reading.

For each non-correct option, explicitly ask: "could a domain expert defend this as correct given the stem?" If yes for any distractor, the item is ambiguous.

  5 — One defensible correct answer; distractors clearly distinguishable from correct.
  4 — Effectively unambiguous; minor edge case in one distractor.
  3 — One distractor is partially defensible under uncommon interpretation.
  2 — Two distractors could be defensibly correct; the keyed answer is preferred but not uniquely correct.
  1 — The keyed correct answer is itself wrong, OR a keyed distractor is mathematically/factually equivalent to correct.

Common failures: keyed distractor is mathematically equivalent (e.g., DFA ≡ MANOVA omnibus), keyed correct contains a false universal that a knowledgeable candidate spots.

## 3. Tier fit (1-5 scale)

Does the cognitive demand of SELECTING the correct answer match the labeled Bloom's tier?

  T1 (Remember): selecting requires retrieving a memorized fact/term/definition.
  T2 (Understand): selecting requires recognizing an example or distinction.
  T3 (Apply): selecting requires using a concept on a novel scenario.
  T4 (Evaluate): selecting requires judging among defensible alternatives, weighing evidence.

  5 — Cognitive demand precisely matches the labeled tier.
  4 — Demand matches with minor drift toward an adjacent tier.
  3 — Demand drifts one tier from label.
  2 — Cognitive demand notably misaligned (e.g., T3 vignette dressing for T1 recall).
  1 — Severely misaligned (e.g., T4 stem with T1 answer).

Common failures: scope_creep — vignette setting at higher tier but answer is just term recall.

CLASSIFICATION RULE — assign ONE diagnostic_quality_class per question based on the LOWEST score across the three dimensions:

  clean — all three dimensions ≥4. Production-ready.
  minor — at least one dimension at 3. Polish would improve.
  major — at least one dimension ≤2. Routes chapter to review for human attention.

DO NOT classify on:
- english_gap (separate audit)
- editorial style (separate audit)
- which option you think is best
- how interesting the question is

INSPECT THIS QUESTION:

Domain: {domain}
Bloom's tier: {tier} ({tier_name})
Anchor's testable fact: {testable_fact}

STEM:
{stem}

OPTIONS:
{options_block}

OUTPUT FORMAT — single JSON object:

{{
  "scores": {{
    "factual_correctness": 1-5,
    "ambiguity": 1-5,
    "tier_fit": 1-5
  }},
  "rationales": {{
    "factual_correctness": "1-sentence justification",
    "ambiguity": "1-sentence justification",
    "tier_fit": "1-sentence justification"
  }},
  "diagnostic_quality_class": "clean|minor|major",
  "issues": [
    {{"dimension": "factual_correctness|ambiguity|tier_fit",
      "severity": "minor|major",
      "description": "specific issue with file/option/text reference",
      "suggested_fix": "concrete actionable fix"}}
  ],
  "summary": "1-sentence overall verdict"
}}

If diagnostic_quality_class is "clean", issues should be []."""
