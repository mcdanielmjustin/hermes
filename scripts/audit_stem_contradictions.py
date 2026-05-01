"""Audit-mode LLM check for stem-contradicting distractors.

For each question in a saved batch, asks Sonnet 4.6 whether any
distractor's claim is directly contradicted by a stated fact in the
stem (i.e., eliminable by reading alone, not by content knowledge).

This is an OFFLINE audit, NOT a production gate — it runs on saved
batches to flag candidates for review/regeneration. It does not block
generation or add per-question retry cost.

The pattern this audit catches:

  BAD (stem-eliminable):
    Stem: "Lester still recalls his wedding from a decade earlier."
    Distractor: "Retrograde amnesia erases all pre-injury memories
                 regardless of when they were consolidated."
    -> Wedding is a pre-injury memory; stem says it's preserved;
       distractor's "all" is contradicted by the wedding case.
       Eliminable by reading alone.

  OK (content-eliminable):
    Stem: "CT confirms thromboembolic infarct in motor cortex."
    Distractor: "Hemiplegia consistent with closed head trauma rather
                 than pyramidal infarction."
    -> Requires the student to know thromboembolic ≠ closed head
       trauma (etiology categories). Not stem-contradicted.

Why Sonnet (not Haiku or Opus): the task is binary classification with
citation, requiring solid NLI but not Opus-level open-ended reasoning.
Sonnet is the marginal-value sweet spot — 5× cheaper than Opus, 1-2%
better than Haiku at subtle cases.

Usage:
  python scripts/audit_stem_contradictions.py path/to/batch.json
  python scripts/audit_stem_contradictions.py --dir data/quiz/BPSY/
  python scripts/audit_stem_contradictions.py exported_batch.json
    (handles JSON exports with `{"questions": [...]}` wrapper too)
  python scripts/audit_stem_contradictions.py path/to/batch.json --fix
    (Phase 11: Sonnet rewrites each flagged distractor; saves to
     `{input}.fixed.json` alongside the original. Empirically reduces
     flag count by ~1 per pass within Sonnet's audit-variance noise —
     useful as a first-pass cleanup before human review, NOT a
     one-shot eliminator. Costs ~$0.005 per flagged distractor.)

Outputs:
  - Per-question flags printed to stdout with citations
  - Summary table at end (questions audited, flags found, cost)
  - Full results saved as `{input_path}.audit.json` for downstream use
  - With --fix: patched batch saved as `{input_path}.fixed.json`
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys
from datetime import datetime, timezone

import anthropic

SCRIPT_DIR = pathlib.Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.audit_calibration import audit_addendum_for_question
from pipeline.schema_labeling_classifier import apply_schema_labeling_override
from pipeline.english_gap_scanner import (
    scan_question as _scan_question_for_eg,
    apply_english_gap_override,
)
from pipeline.detectors import PHASE_AUDIT
from pipeline.detectors.registry import create_detector_registry

# Phase A1: registry instance for audit-time detector calls. Lazily
# constructed on first use so importing this script doesn't pay the
# registry-creation cost when only utilities are needed.
_audit_detector_registry = None


def _get_audit_registry():
    global _audit_detector_registry
    if _audit_detector_registry is None:
        _audit_detector_registry = create_detector_registry()
    return _audit_detector_registry

# Sonnet 4.6 — the marginal-value sweet spot for this task. Closed
# binary classification with citation; Opus's open-ended reasoning
# depth is wasted here.
MODEL_ID = "claude-sonnet-4-6"
INPUT_PRICE_PER_M = 3.0
OUTPUT_PRICE_PER_M = 15.0

# Phase 21c: Haiku 4.5 for cross-model verification. Single Haiku pass
# alongside the Sonnet quorum is the second opinion — different model
# size means different blind spots, which is the goal. ship_readiness
# opts in via --cross-model-verify; default off. Disagreement on
# english_gap stays english_gap (conservative — trust the catch);
# other disagreements escalate to soft_flag.
HAIKU_MODEL_ID = "claude-haiku-4-5-20251001"
HAIKU_INPUT_PRICE_PER_M = 1.0
HAIKU_OUTPUT_PRICE_PER_M = 5.0

PROMPT = """You are auditing a multiple-choice question for distractor-quality issues. For EACH distractor (not the correct option), classify into ONE of four categories. The classification IS the audit decision — do not also output a separate flag.

THE FOUR CLASSES

ENGLISH_GAP — A student can reject the distractor by lexical comparison with the stem alone. No domain knowledge needed. The contradiction is in the printed words, not in the concepts.
  Canonical example:
    Stem: "After bilateral hippocampal damage, Lester Nichols cannot form new declarative memories but still recalls his wedding from a decade earlier."
    Distractor: "Retrograde amnesia erases ALL pre-injury memories regardless of when they were consolidated."
    Classification: english_gap.
    Why: the universal "all pre-injury memories" is directly contradicted by the specific stated case "wedding from a decade earlier" (a pre-injury memory). The student needs only to notice the contradiction between "all" and the wedding example — no knowledge of what retrograde amnesia means as a concept is required. This is a quality failure: the question becomes a reading-comprehension exercise.
  Pattern: universal quantifiers paired with specific stem counter-examples; laterality inversions where the stem names a specific side; distractors that contradict the named subject's preserved abilities or observed findings.

CONTENT_GAP — Rejecting the distractor requires invoking concept/domain knowledge. A real contradiction exists, but recognizing it requires knowing what some technical term or relationship means.
  Canonical example:
    Stem: "A compound binds a receptor but produces no measurable postsynaptic activity on its own."
    Distractor: "The compound exhibits intrinsic activity that mimics the endogenous neurotransmitter."
    Classification: content_gap.
    Why: the contradiction exists ("intrinsic activity" vs "no measurable postsynaptic activity"), BUT recognizing it requires knowing that "intrinsic activity" in receptor pharmacology means producing a measurable postsynaptic effect. A student who doesn't know what intrinsic activity means cannot reject the distractor. Picking it reveals a content-knowledge gap (didn't know what intrinsic activity entails) — exactly what the question is supposed to test.
  Pattern: stem describes a phenomenon using technical vocabulary whose meaning IS the test; the distractor varies on that defined meaning. The contradiction is real but invisible without concept knowledge.

CLEAN — No direct contradiction with stem facts. The distractor is a plausible-but-wrong alternative that requires applying the concept to determine wrongness.
  Canonical example:
    Stem: "CT imaging confirms an acute thromboembolic infarct in the right cerebral hemisphere involving the motor cortex."
    Distractor: "Predict hemiplegia consistent with closed head trauma rather than pyramidal infarction."
    Classification: clean.
    Why: the stem doesn't state "this is not closed head trauma" anywhere. There is no specific stem fact for the distractor to contradict. The student rejects this by knowing thromboembolic ≠ closed head trauma (different etiology categories). This is the most pedagogically pure distractor design.

SOFT_FLAG — Genuinely uncertain between two classes after applying the discriminating rule honestly. Use ONLY when the classification is actually borderline — not as an out from a call you could make.
  Canonical example:
    Stem: "Leon prefers lighter-skinned colleagues, avoids events where most attendees share his darker complexion, and has felt ashamed of his own appearance since adolescence."
    Distractor: "Predict generalized self-esteem patterns reflecting negative self-schema unrelated to racial group membership."
    Classification: soft_flag.
    Why: ambiguous between english_gap (frame denial: "unrelated to X" contradicts stem's stated X-framing) and clean (the distractor names a different analytic frame, not directly contradicting a fact). The auditor cannot honestly commit to one without overstating confidence. soft_flag distractors do NOT block ship — they surface for human review.
  When to use: when the same distractor would plausibly classify differently across runs. Sparingly.

CLASSIFICATION RULE — ask yourself for each distractor:
  "Could a student reject this distractor by re-reading the stem and the distractor alone, with NO domain knowledge invoked?"
  • YES, the contradiction is in the printed words → english_gap
  • NO, but a real contradiction exists once you know the concept → content_gap
  • NO direct contradiction at all, just a wrong-but-plausible alternative → clean
  • GENUINELY UNCERTAIN between two of the above (after honest application) → soft_flag

DO NOT include hedging language ("actually", "on closer inspection", "this requires knowing", "should not be flagged"). Commit to a classification. If you find yourself wanting to qualify between english_gap and content_gap or between english_gap and clean, that's exactly when soft_flag is appropriate — but use it sparingly.

THE PEDAGOGICAL TARGET: distractors should be content_gap or clean. english_gap distractors degrade the question to reading-comprehension and are the quality failure this audit catches. Do not over-flag — content_gap and clean distractors are FUNCTIONING distractors and must not be classified as english_gap. soft_flag is for honest borderlines, not for avoiding decisions.

INSPECT THIS QUESTION:

STEM:
{stem}

OPTIONS:
{options_block}
{mode_addendum}

OUTPUT FORMAT: Output your final JSON answer ONLY. Do not include any preamble, restart, correction text, or multiple JSON objects in sequence. If you reconsider during reasoning, emit ONLY your final decision — never include phrases like "Wait, let me redo this" or "Actually, on second thought" followed by a second JSON object. Your response must be a single, complete, valid JSON object.

Respond with valid JSON in this exact shape:
{{
  "classifications": [
    {{"letter": "X", "class": "english_gap", "distractor_text": "...", "contradicted_stem_fact": "...", "explanation": "..."}},
    {{"letter": "Y", "class": "content_gap", "distractor_text": "...", "contradicted_stem_fact": "...", "explanation": "..."}},
    {{"letter": "Z", "class": "clean", "distractor_text": "...", "explanation": "..."}},
    {{"letter": "W", "class": "soft_flag", "distractor_text": "...", "ambiguous_between": ["english_gap", "clean"], "explanation": "..."}}
  ]
}}

Include one classification entry per DISTRACTOR (skip the correct option). The `contradicted_stem_fact` field is required for english_gap and content_gap; omit or set to null for clean and soft_flag. The `ambiguous_between` field is required for soft_flag (list of the two classes you couldn't decide between) and omitted otherwise. The `explanation` field justifies the classification — be specific about WHY the classification fits."""


# ── Fix prompt (Phase 11) ─────────────────────────────────────
# When --fix is set, the audit findings feed back to Sonnet as
# rewrite directives. The model rewrites just the flagged distractor's
# text + explanation, preserving the misconception_id and the
# question's other fields. Cheaper than re-running the full Opus
# generation pipeline ($0.005/Q vs $0.30/Q) and surgical (only the
# flagged distractor changes — the correct answer, stem, and clean
# distractors stay intact).
FIX_PROMPT = """You are rewriting one distractor in a multiple-choice question. The current distractor is "english_gap" — its claim contradicts a stem fact in a way a student can detect by reading alone, with no concept knowledge invoked. That defeats the pedagogy because the question becomes reading comprehension.

Your job: rewrite the flagged distractor's `text` and `explanation` so it becomes CONTENT_GAP — wrong in a way that requires concept knowledge to recognize.

PRINCIPLES:
1. Targets the SAME misconception_id (preserved verbatim below) — distractors carry diagnostic intent.
2. Wrong via CONTENT KNOWLEDGE — recognizing the wrongness requires invoking what a technical term means, what a mechanism does, or what a framework actually says.
3. Does NOT lexically contradict the stem. Specifically: AVOID universal quantifiers ("all", "throughout", "entire", "any") that conflict with stated specifics. AVOID denying a stem-stated frame ("unrelated to X" when stem ties everything to X). AVOID inverting numeric counts the stem prints. AVOID repeating stem-vocabulary in a way that contradicts the stem's claim about that vocabulary.
4. Stays approximately the same character length as the other distractors.
5. Engages the same topic-realm vocabulary as the correct option.
6. The `text` field MUST be a noun phrase or short declarative claim. Reasoning markers (`because`, `since`, `due to`, `owing to`) are FORBIDDEN in text — those belong in the explanation.

PAIRED EXAMPLES (english_gap → content_gap rewrites):

Pattern A — Universal contradicts split:
  Stem: "F scale within normal limits, but FB scale markedly elevated"
  ❌ english_gap: "Evaluate as overreporting throughout the entire protocol"
     (Lexical: "entire" contradicts split pattern stated in stem)
  ✅ content_gap: "Evaluate as F-K dissimulation pattern indicating defensive responding"
     (Wrong scale-pattern interpretation; rejecting requires knowing F-K vs FB)

Pattern B — Stated-vocab contradiction:
  Stem: "compound binds D2 receptor but produces no measurable change in postsynaptic firing"
  ❌ english_gap: "Predict direct D2 inhibition exerting its own postsynaptic biological effect"
     (Lexical: "own postsynaptic effect" contradicts "no measurable change in postsynaptic firing")
  ✅ content_gap: "Predict direct D2 inhibition through allosteric modulation of channel kinetics"
     (Wrong mechanism — D2 antagonism doesn't act allosterically on channels; rejecting requires receptor pharmacology knowledge)

Pattern C — Numeric inversion:
  Stem: "researcher categorizes by 3 factors and measures 2 score outcomes"
  ❌ english_gap: "IVs are the 2 measured outcomes; DVs are the 3 categorized factors"
     (Lexical: 3↔2 inversion of stem-printed counts)
  ✅ content_gap: "IVs require continuous measurement scales; DVs require categorical levels"
     (Wrong DEFINITION of IV/DV — measurement-scale vs role; rejecting requires research-design knowledge)

Pattern D — Frame denial:
  Stem: "Leon prefers lighter-skinned colleagues, avoids events with darker-complexioned attendees"
  ❌ english_gap: "Predict generalized self-esteem patterns unrelated to racial group membership"
     (Lexical: "unrelated to race" denies the racial framing the stem establishes)
  ✅ content_gap: "Predict externalized racism patterns reflecting projection of racial attitudes onto out-group members"
     (Wrong direction within the framework — externalized vs internalized; rejecting requires knowing internalized racism's defining inward-direction)

KEY SHIFT: Instead of negating, repeating, or inverting stem content, vary on a CONCEPT-LEVEL property (mechanism, definition, direction within framework, scope of application). The contradiction should require domain knowledge to detect, not careful reading.

QUESTION STEM:
{stem}

CORRECT ANSWER (preserved as-is, do not modify):
  {correct_letter} [CORRECT]: {correct_text}

OTHER DISTRACTORS (preserved as-is, do not modify):
{other_distractors_block}

DISTRACTOR TO REWRITE — option {letter}:
  Current text: {flagged_text}
  Misconception ID to preserve: {misconception_id}
  Misconception type: {misconception_type}

WHY FLAGGED:
  Stem fact contradicted: "{contradicted_stem_fact}"
  Explanation: {flag_explanation}

Target character length: ~{target_length} characters (match the other distractors).

Respond ONLY with valid JSON in this exact shape:
{{"text": "<new distractor text>", "explanation": "<new explanation, 1-2 sentences explaining why this distractor reflects the misconception and what the student should learn from it>"}}"""


# ── Stem-rewrite prompt (Phase 20d) ────────────────────────────
# When a question's flagged-distractor rate is ≥50% (majority of
# distractors are english_gap), the signal is stem over-specification,
# not distractor design. Rewriting individual distractors can't escape
# a stem that prints the answer. Stem rewrite removes the giveaway
# facts while preserving the testable concept.
STEM_REWRITE_PROMPT = """You are rewriting the STEM of a multiple-choice question to fix STEM OVER-SPECIFICATION. The audit found that ≥50% of this question's distractors are english_gap — each lexically contradicts a fact stated in the stem. That signals the stem reveals too much about the answer; the question becomes a reading exercise rather than a probe of domain knowledge. The fix is at the stem level, not the distractor level.

PRINCIPLES:
1. REMOVE the specific facts that distractors lexically contradict — the printed numbers, ratios, durations, laterality, named outcomes, or stated mechanisms that allow elimination by reading alone.
2. PRESERVE the testable concept — the correct answer must STILL be the right answer to the new stem. The student's knowledge demand stays the same; only the reading shortcut is removed.
3. MAINTAIN Bloom's Tier {tier} cognitive demand:
   - Tier 1 (Remember): direct definitional interrogative; no scenario.
   - Tier 2 (Understand): brief context (≤2 sentences); ask for comprehension.
   - Tier 3 (Apply): novel scenario; ask for prediction/selection in that scenario.
   - Tier 4 (Evaluate): complex case with competing considerations.
4. DIRECT INTERROGATIVE phrasing — no "Which option correctly...", "Which best...", "Which most accurately...", "Which definition correctly...". Use "What is X?", "What distinguishes A from B?", "Which term describes Y?".
5. KEEP within ≤2 sentences for T1/T2; ≤4 sentences for T3/T4.

PAIRED EXAMPLES (over-specified stem → fixed stem):

Example A — T1 epidemiology (the canonical CPAT case):
  ❌ over-specified: "depression rates are similar for boys and girls until puberty, after which female rates rise substantially and approach a 2:1 ratio by adulthood. Which term describes this developmental pattern?"
     (Distractor A "stable 3:1 from early childhood" is contradicted by "until puberty"; distractor B "reverses by adulthood" is contradicted by "2:1 by adulthood"; distractor D "equivalent across all stages" is contradicted by both. ALL 3 fail by reading.)
  ✅ fixed: "Which term describes the well-documented epidemiological pattern of sex differences in depression incidence emerging across adolescence?"
     (Removes the specific ratios and timing. Correct answer "pubertal divergence approaching 2:1 in adulthood" still answers the question; distractors A/B/D no longer have lexical contradictions to exploit. The student must know the term to choose correctly.)

Example B — T3 mechanism:
  ❌ over-specified: "compound binds D2 receptor but produces no measurable change in postsynaptic firing on its own. Predict the mechanism."
     (A distractor claiming "own postsynaptic effect" is contradicted by "no measurable change". Reading defeats it.)
  ✅ fixed: "A compound binds the D2 receptor at high affinity. Researchers screen its functional profile for therapeutic potential. Which mechanism best fits a candidate antagonist at this target?"
     (Specific finding "no measurable change" removed; clinical/research framing intact; correct answer about antagonism still applies.)

CURRENT STEM:
{stem}

CORRECT ANSWER (preserved as-is — do NOT modify):
  {correct_letter} [CORRECT]: {correct_text}

TESTABLE CONCEPT: {tested_concept}

FLAGGED DISTRACTORS AND THE STEM FACTS THEY CONTRADICT:
{flagged_block}

OTHER (non-flagged) DISTRACTORS — preserve their answerability:
{other_block}

QUESTION TIER: {tier} ({tier_name})

OUTPUT — single JSON object only:
{{
  "stem": "the rewritten stem",
  "rationale": "1 sentence: what was removed, and why the rewrite preserves the correct answer's correctness"
}}"""


_TIER_NAMES = {1: "Remember", 2: "Understand", 3: "Apply", 4: "Evaluate"}


def should_rewrite_stem(audit_result: dict, question: dict,
                        threshold: float = 0.5) -> bool:
    """Phase 20d heuristic: when ≥``threshold`` fraction of a question's
    distractors are flagged english_gap, the stem is over-specifying
    and distractor-rewrite can't escape. Returns True if stem rewrite
    is the correct intervention.

    A typical question has 3 distractors; threshold=0.5 fires when
    ≥2 of 3 are flagged. The CPAT depression E-02 case (3/3 flagged,
    100%) is the canonical positive.
    """
    flagged = audit_result.get("flagged_distractors") or []
    if not flagged:
        return False
    options = question.get("options") or []
    distractors = [o for o in options if not o.get("is_correct")]
    if not distractors:
        return False
    rate = len(flagged) / len(distractors)
    return rate >= threshold


async def rewrite_stem(client, question: dict, audit_result: dict,
                       semaphore) -> dict:
    """Rewrite a question's stem to remove over-specification.

    Returns the same dict shape as ``fix_question``: ``{question_id,
    patched, question, usage, errors}``. The patched question carries
    a new ``question_stem`` with the giveaway facts stripped; the
    correct answer, distractors, and metadata are preserved.

    Phase 20d intervention. Triggered upstream by ``should_rewrite_stem``
    when ≥50% of distractors are english_gap (the over-specification
    signal). Uses Sonnet at temp=0 for determinism.
    """
    flagged = audit_result.get("flagged_distractors") or []
    if not flagged:
        return {
            "question_id": question.get("question_id", "?"),
            "patched": False,
            "question": question,
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "errors": [],
        }

    options = question.get("options") or []
    correct = next((o for o in options if o.get("is_correct")), None)
    distractors = [o for o in options if not o.get("is_correct")]
    if not correct or not distractors:
        return {
            "question_id": question.get("question_id", "?"),
            "patched": False,
            "question": question,
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "errors": ["no correct option or no distractors"],
        }

    flagged_letters = {f.get("letter") for f in flagged}
    flagged_block_lines: list[str] = []
    for flag in flagged:
        flagged_block_lines.append(
            f"  {flag.get('letter','?')}: {flag.get('distractor_text','')}\n"
            f"     contradicted by stem fact: \"{flag.get('contradicted_stem_fact','')}\""
        )
    flagged_block = "\n".join(flagged_block_lines) or "  (none)"

    other_block_lines: list[str] = []
    for d in distractors:
        if d.get("letter") in flagged_letters:
            continue
        other_block_lines.append(
            f"  {d.get('letter','?')}: {d.get('text','')}"
        )
    other_block = "\n".join(other_block_lines) or "  (none — all distractors flagged)"

    tier = question.get("difficulty_tier") or 1
    tier_name = _TIER_NAMES.get(tier, "Unknown")
    tested_concept = (
        (question.get("tested_concept") or {}).get("concept_label")
        or (question.get("tested_concept") or {}).get("knowledge_tested")
        or "(not specified)"
    )

    prompt = STEM_REWRITE_PROMPT.format(
        stem=question.get("question_stem", ""),
        correct_letter=correct.get("letter", "?"),
        correct_text=correct.get("text", ""),
        tested_concept=tested_concept,
        flagged_block=flagged_block,
        other_block=other_block,
        tier=tier,
        tier_name=tier_name,
    )

    async with semaphore:
        try:
            response = await client.messages.create(
                model=MODEL_ID,
                max_tokens=512,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text if response.content else ""
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        except Exception as e:
            return {
                "question_id": question.get("question_id", "?"),
                "patched": False,
                "question": question,
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "errors": [f"API error: {e}"],
            }

    parsed = parse_response(text)
    if not parsed or "stem" not in parsed:
        return {
            "question_id": question.get("question_id", "?"),
            "patched": False,
            "question": question,
            "usage": usage,
            "errors": ["rewrite parse failed"],
        }

    new_stem = parsed["stem"].strip()
    if not new_stem:
        return {
            "question_id": question.get("question_id", "?"),
            "patched": False,
            "question": question,
            "usage": usage,
            "errors": ["rewrite produced empty stem"],
        }

    patched_question = dict(question)
    patched_question["question_stem"] = new_stem
    # Trace the intervention on the question record for downstream
    # reporting — kept under the underscore convention so it's transient
    # and not persisted unless the caller chooses to.
    patched_question["_stem_rewrite_rationale"] = parsed.get("rationale", "")
    patched_question["_stem_rewrite_original_stem"] = question.get("question_stem", "")

    return {
        "question_id": question.get("question_id", "?"),
        "patched": True,
        "question": patched_question,
        "usage": usage,
        "errors": [],
        "intervention": "stem_rewrite",
    }


def load_questions(path: pathlib.Path) -> list[dict]:
    """Load questions from a chapter JSON or an exported batch JSON."""
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "questions" in data:
        return data["questions"]
    return []


def build_prompt(question: dict, mode_addendum: str | None = None) -> str:
    """Build the audit prompt for a question.

    If mode_addendum is None (default), derive a flavor-aware addendum
    from the question's metadata. Pass an explicit string (including
    "") to override — the in-line gate uses this to inject cell-aware
    calibration instead of flavor-aware.
    """
    stem = question.get("question_stem", "")
    options = question.get("options", []) or []
    options_block = "\n".join(
        f"  {o.get('letter', '?')} "
        f"{'[CORRECT]' if o.get('is_correct') else '[distractor]'}: "
        f"{o.get('text', '')}"
        for o in options
    )
    if mode_addendum is None:
        mode_addendum = audit_addendum_for_question(question)
    return PROMPT.format(
        stem=stem,
        options_block=options_block,
        mode_addendum=mode_addendum,
    )


def _extract_json_objects(text: str) -> list[str]:
    """Find every top-level {...} substring in `text`, in order.

    Tracks brace depth while respecting string literals so braces inside
    quoted JSON strings don't confuse the boundaries. Used to pick the
    LAST valid object when a model emits multiple in sequence (e.g., a
    self-correction stream like '{...wrong...}\\nWait, let me redo.\\n{...right...}').
    """
    candidates = []
    depth = 0
    start = -1
    in_str = False
    esc = False
    for i, c in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif c == '\\':
                esc = True
            elif c == '"':
                in_str = False
            continue
        if c == '"':
            in_str = True
        elif c == '{':
            if depth == 0:
                start = i
            depth += 1
        elif c == '}':
            depth -= 1
            if depth == 0 and start >= 0:
                candidates.append(text[start:i + 1])
                start = -1
    return candidates


def parse_response(text: str) -> dict | None:
    """Extract the JSON object from Sonnet's response. Tolerant of code
    fences, surrounding prose, and self-correction streams (multiple
    JSON objects in sequence — the last one is taken as the model's
    final answer)."""
    if not text:
        return None
    # Strip markdown code fences if present
    cleaned = text.strip()
    if "```json" in cleaned:
        cleaned = cleaned.split("```json", 1)[1]
        if "```" in cleaned:
            cleaned = cleaned.split("```", 1)[0]
    elif "```" in cleaned:
        cleaned = cleaned.split("```", 1)[1]
        if "```" in cleaned:
            cleaned = cleaned.split("```", 1)[0]
    cleaned = cleaned.strip()

    # Happy path: response is a single JSON object.
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass

    # Fallback: extract every top-level JSON object and try them from
    # last to first. When a model self-corrects mid-response, the final
    # object is the intended answer; older parsers that span "first { to
    # last }" silently merged corrections into invalid JSON.
    for cand in reversed(_extract_json_objects(cleaned)):
        try:
            return json.loads(cand)
        except json.JSONDecodeError:
            continue
    return None


def derive_flagged_from_classifications(parsed: dict | None) -> list[dict]:
    """Backward-compat shim: derive the legacy `flagged_distractors`
    list from the new `classifications` array. The audit prompt now
    outputs three classes (english_gap, content_gap, clean); only
    english_gap entries become flags. Returns [] if parsed is None or
    has no classifications.
    """
    if not parsed:
        return []
    # Import here to avoid a top-level dependency from this script on
    # the pipeline package (audit script can run standalone).
    from pipeline.quality_taxonomy import ENGLISH_GAP
    # Direct classifications path (new prompt)
    classes = parsed.get("classifications") or []
    flags = []
    for c in classes:
        if c.get("class") == ENGLISH_GAP:
            flags.append({
                "letter": c.get("letter", "?"),
                "distractor_text": c.get("distractor_text", ""),
                "contradicted_stem_fact": c.get("contradicted_stem_fact", ""),
                "explanation": c.get("explanation", ""),
            })
    if flags or classes:
        return flags
    # Legacy path (old prompt format) — preserve original key
    return parsed.get("flagged_distractors", []) or []


# ── Phase 21a: aggregation logic for multi-pass audit ──────────
# Pure function — no API calls. Unit-testable in isolation.
# Per-distractor majority vote across N classification passes.
# Tie-breaks (e.g., 1-1-1 split among english_gap/content_gap/clean):
# pick the most-conservative class so the audit doesn't fail OPEN.
# Conservatism order: english_gap > content_gap > clean.
# Soft_flag is preserved (any pass producing soft_flag → keep it),
# since it's already an "uncertain" signal — quorum can't override.
_CONSERVATISM_ORDER = ("english_gap", "content_gap", "clean")


def aggregate_classifications(passes: list[list[dict]]) -> list[dict]:
    """Combine N classification arrays into one via per-distractor
    majority vote.

    Args:
      passes: list of classifications arrays from N audit calls. Each
              inner list contains per-distractor dicts with at least
              {"letter": "A", "class": "english_gap", ...}.

    Returns:
      Single classifications array, one entry per distractor letter
      seen in the passes.

    Aggregation rules:
      - For each letter: collect all classifications across passes.
      - If any pass classified as soft_flag → emit soft_flag (preserve
        uncertainty).
      - Otherwise pick the majority class.
      - Ties → pick the most conservative class (english_gap >
        content_gap > clean).
      - Aggregated explanation/contradicted_stem_fact: pick from a
        pass whose class matches the chosen class (first such pass).
      - Add per-distractor "_pass_count" with the per-class vote count
        for downstream introspection.
    """
    if not passes:
        return []

    # Group classifications by letter
    by_letter: dict[str, list[dict]] = {}
    for pass_classifications in passes:
        for entry in pass_classifications:
            letter = entry.get("letter", "?")
            by_letter.setdefault(letter, []).append(entry)

    aggregated = []
    for letter in sorted(by_letter.keys()):
        entries = by_letter[letter]
        # Tally classes
        votes: dict[str, int] = {}
        for e in entries:
            cls = e.get("class", "")
            votes[cls] = votes.get(cls, 0) + 1

        # Soft_flag preservation — any pass that flagged uncertainty wins
        if "soft_flag" in votes:
            chosen_class = "soft_flag"
        else:
            # Majority vote with conservative tie-break
            max_count = max(votes.values())
            top_classes = [c for c, v in votes.items() if v == max_count]
            if len(top_classes) == 1:
                chosen_class = top_classes[0]
            else:
                # Tie — pick most conservative
                for cls in _CONSERVATISM_ORDER:
                    if cls in top_classes:
                        chosen_class = cls
                        break
                else:
                    # Should not reach (top_classes is non-empty), but
                    # fall back to first class for safety.
                    chosen_class = top_classes[0]

        # Pick the first pass entry matching the chosen class as the
        # representative entry (carries explanation, contradicted_stem_fact)
        representative = next(
            (e for e in entries if e.get("class") == chosen_class),
            entries[0],
        )

        agg_entry = {
            "letter": letter,
            "class": chosen_class,
            "distractor_text": representative.get("distractor_text", ""),
            "explanation": representative.get("explanation", ""),
            "_pass_count": dict(votes),
        }
        # Preserve contradicted_stem_fact for english_gap/content_gap
        if chosen_class in ("english_gap", "content_gap"):
            agg_entry["contradicted_stem_fact"] = representative.get(
                "contradicted_stem_fact", ""
            )
        # Preserve ambiguous_between for soft_flag
        if chosen_class == "soft_flag":
            agg_entry["ambiguous_between"] = representative.get(
                "ambiguous_between", []
            )
        aggregated.append(agg_entry)

    return aggregated


async def _audit_question_single(client, question: dict, semaphore,
                                  model_id: str = MODEL_ID) -> dict:
    """One audit pass. Internal helper — callers use audit_question."""
    async with semaphore:
        prompt = build_prompt(question)
        try:
            response = await client.messages.create(
                model=model_id,
                max_tokens=2048,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text if response.content else ""
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            parsed = parse_response(text)
            classifications = (parsed or {}).get("classifications", []) or []
            return {
                "classifications": classifications,
                "raw_response": text if parsed is None else None,
                "usage": usage,
                "error": None if parsed else "json_parse_failed",
            }
        except Exception as e:
            return {
                "classifications": [],
                "raw_response": None,
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "error": str(e),
            }


async def audit_question(client, question: dict, semaphore,
                         n_passes: int = 1, model_id: str = MODEL_ID) -> dict:
    """Audit a single question. Returns a dict with classifications,
    backward-compat flagged_distractors (derived from english_gap), and
    usage.

    Args:
      n_passes: number of independent audit calls to make. With > 1,
                classifications are aggregated via per-distractor
                majority vote (Phase 21a). Default 1 preserves prior
                behavior. ship_readiness uses 3 to dampen single-pass
                jitter on borderline cases.
      model_id: model identifier for the audit. Default Sonnet 4.6.
                Phase 21c can pass HAIKU_MODEL_ID for cross-model verify.
    """
    if n_passes < 1:
        n_passes = 1

    # Run N passes (sequentially within the per-question semaphore)
    pass_results = []
    for _ in range(n_passes):
        result = await _audit_question_single(client, question, semaphore, model_id)
        pass_results.append(result)

    # Combine usage across passes
    total_in = sum(r.get("usage", {}).get("input_tokens", 0) for r in pass_results)
    total_out = sum(r.get("usage", {}).get("output_tokens", 0) for r in pass_results)
    combined_usage = {"input_tokens": total_in, "output_tokens": total_out}

    # Detect any-pass error (we tolerate partial errors and aggregate
    # only over the successful passes, but flag if zero passes succeeded)
    successful_passes = [r for r in pass_results if not r.get("error")]
    error = None
    if not successful_passes:
        error = pass_results[0].get("error", "all_passes_failed")
    elif len(successful_passes) < n_passes:
        # Partial: some passes errored. Aggregate only the good ones.
        error = f"partial: {len(successful_passes)}/{n_passes} succeeded"

    if successful_passes:
        if n_passes == 1:
            # Single-pass: skip aggregation overhead, use the raw result
            classifications = successful_passes[0]["classifications"]
        else:
            # Multi-pass: aggregate
            classifications = aggregate_classifications(
                [r["classifications"] for r in successful_passes]
            )
    else:
        classifications = []

    # Phase A1/A2: gather all audit-time deterministic detector signals
    # via the registry. Used by the override-application steps below
    # (schema_labeling, english_gap) and projected into the manifest
    # shape (`scanner_signals`).
    registry = _get_audit_registry()
    audit_signals = registry.scan_for_phase(PHASE_AUDIT, question)
    eg_signals = [
        s for s in audit_signals
        if s.detector_id == "english_gap_scanner" and s.letter is not None
    ]

    # Phase 22a: deterministic schema-labeling override. Demotes
    # english_gap → content_gap when paired-named-concepts in the
    # stem are swapped in the distractor and no universal-quantifier
    # contradiction blocks. Brief discriminators (when threaded onto
    # the question dict via `_discriminators` — Phase 22c) provide
    # tier-A precision; absent the brief, Tier-B lexical fallback
    # against canonical LABEL_PAIRS still fires. The override fires
    # uniformly here so both Sonnet and Haiku audits land in the
    # same overridden state before cross-model reconciliation.
    discriminators = (question or {}).get("_discriminators")
    classifications, n_schema_overrides = apply_schema_labeling_override(
        question, classifications, discriminators=discriminators,
    )

    # Phase A2: deterministic english_gap override (T1/T2 only).
    # When the regex scanner fires a high-confidence signature
    # (universal_quantifier, laterality, numeric_ratio) on a T1/T2
    # question, the scanner wins over the LLM's classification —
    # including over a prior schema_labeling demotion to content_gap,
    # because the universal-quantifier guard in classify_distractor
    # already prevents schema_labeling from firing on universal-quantifier
    # cases. Only the rare overlap (Tier B lexical pair + scanner
    # signature on the same letter) re-promotes; manual review
    # recommended on those during validation. T3/T4 stays advisory
    # until A2.5 populates the cell matrix.
    classifications, n_eg_overrides = apply_english_gap_override(
        question, classifications, eg_signals,
    )

    flagged = derive_flagged_from_classifications(
        {"classifications": classifications}
    )

    # Manifest projection (Phase 24b / A1): the english_gap detector's
    # raw signals (regardless of override action) are surfaced for
    # ship_readiness's per-chapter aggregation and downstream consumers
    # comparing scanner flags vs audit english_gap.
    scanner_signals = {
        s.letter: {
            "fired": s.fired,
            "confidence": s.confidence,
            "reason": s.reason,
            "signature": s.signature,
        }
        for s in eg_signals
    }
    n_scanner_flags = sum(1 for s in eg_signals if s.fired)

    return {
        "question_id": question.get("question_id", "?"),
        "tier": question.get("difficulty_tier"),
        "stem": question.get("question_stem", "")[:200],
        "classifications": classifications,
        "flagged_distractors": flagged,
        "raw_response": (
            successful_passes[0].get("raw_response")
            if successful_passes else pass_results[0].get("raw_response")
        ),
        "usage": combined_usage,
        "error": error,
        "n_passes": n_passes,
        "model_id": model_id,
        "schema_labeling_overrides_count": n_schema_overrides,
        "english_gap_override_count": n_eg_overrides,
        "scanner_signals": scanner_signals,
        "scanner_flags_count": n_scanner_flags,
    }


async def fix_question(client, question: dict, audit_result: dict,
                       semaphore) -> dict:
    """Rewrite each flagged distractor in a question via Sonnet, OR
    (Phase 20d) rewrite the stem when the over-specification signal
    fires — chaining a residual distractor-rewrite pass if the stem
    rewrite leaves <50% flagged.

    Returns a dict with the patched question (or the original if no
    flags / errors), the fix usage, and any errors.

    Strategy:
    - Phase 20d dispatch: if ≥50% of distractors are flagged
      english_gap, the signal is stem over-specification, not
      distractor design. Stem-rewrite is the correct first intervention;
      individual distractor rewrites can't escape a stem that prints
      the answer. After stem rewrite, re-audit; if residual english_gap
      distractors remain (typically 0-1), run a SECOND pass of
      distractor-rewrite on those residuals. The result dict includes
      ``intervention="stem_rewrite"`` or
      ``intervention="stem_rewrite+distractor_rewrite"``.
    - Default: one Sonnet call per flagged distractor — keeps each
      rewrite focused and lets us preserve the OTHER flagged distractors
      (if any) as part of the "do not modify" context for each call.
    """
    flagged = audit_result.get("flagged_distractors") or []
    if not flagged:
        return {
            "question_id": question.get("question_id", "?"),
            "patched": False,
            "question": question,
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "errors": [],
        }

    # Phase 20d: dispatch to stem-rewrite when over-specification signal
    # fires. Distractor-rewrite cannot escape a stem that lexically
    # implies the answer; stem-rewrite removes the implication. Chain a
    # residual distractor-rewrite pass if any flags remain after re-audit.
    if should_rewrite_stem(audit_result, question):
        stem_res = await rewrite_stem(client, question, audit_result, semaphore)
        if not stem_res.get("patched"):
            return stem_res
        rewritten_q = stem_res["question"]

        # Re-audit the rewritten question (single pass for cost; the
        # downstream ship_readiness re-audit will run multi-pass quorum).
        post_audit = await audit_question(
            client, rewritten_q, semaphore, n_passes=1,
        )
        residual_flags = post_audit.get("flagged_distractors") or []

        # No residuals → stem rewrite alone converged.
        if not residual_flags:
            stem_res["intervention"] = "stem_rewrite"
            return stem_res

        # Residuals → run distractor-rewrite on the post-stem-rewrite
        # question against the new audit's flagged list. Capped at 1
        # iteration: more residuals than the threshold's worth implies
        # something more fundamentally broken than over-specification.
        dist_res = await _fix_distractors(
            client, rewritten_q, post_audit, semaphore,
        )

        # Merge usage from the three calls (stem-rewrite + re-audit +
        # distractor-rewrite). The re-audit's usage isn't surfaced via
        # fix accounting today — caller's downstream re-audit will
        # invoice that line item separately. Keep stem + distractor
        # here for fix-cost reporting fidelity.
        merged_usage = {
            "input_tokens": (
                stem_res.get("usage", {}).get("input_tokens", 0)
                + post_audit.get("usage", {}).get("input_tokens", 0)
                + dist_res.get("usage", {}).get("input_tokens", 0)
            ),
            "output_tokens": (
                stem_res.get("usage", {}).get("output_tokens", 0)
                + post_audit.get("usage", {}).get("output_tokens", 0)
                + dist_res.get("usage", {}).get("output_tokens", 0)
            ),
        }
        # Distractor-rewrite returns the further-patched question;
        # preserve the stem-rewrite trace fields the dist pass ignored.
        final_q = dist_res.get("question") or rewritten_q
        if "_stem_rewrite_rationale" in rewritten_q:
            final_q["_stem_rewrite_rationale"] = rewritten_q["_stem_rewrite_rationale"]
        if "_stem_rewrite_original_stem" in rewritten_q:
            final_q["_stem_rewrite_original_stem"] = rewritten_q["_stem_rewrite_original_stem"]

        return {
            "question_id": question.get("question_id", "?"),
            "patched": True,
            "fixes_applied": dist_res.get("fixes_applied", 0),
            "fixes_attempted": dist_res.get("fixes_attempted", 0),
            "question": final_q,
            "usage": merged_usage,
            "errors": stem_res.get("errors", []) + dist_res.get("errors", []),
            "intervention": "stem_rewrite+distractor_rewrite",
            "stem_residual_flags": len(residual_flags),
        }

    return await _fix_distractors(client, question, audit_result, semaphore)


async def _fix_distractors(client, question: dict, audit_result: dict,
                            semaphore) -> dict:
    """Distractor-rewrite primitive (was the body of fix_question
    pre-Phase-20d). One Sonnet call per flagged distractor; preserves
    misconception_id, slot, letter, is_correct — only text and
    explanation are rewritten.
    """
    flagged = audit_result.get("flagged_distractors") or []
    if not flagged:
        return {
            "question_id": question.get("question_id", "?"),
            "patched": False,
            "question": question,
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "errors": [],
        }
    options = question.get("options", []) or []
    correct = next((o for o in options if o.get("is_correct")), None)
    distractors = [o for o in options if not o.get("is_correct")]
    if not correct or not distractors:
        return {
            "question_id": question.get("question_id", "?"),
            "patched": False,
            "question": question,
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "errors": ["no correct option or no distractors — cannot fix"],
        }

    # Median distractor length as the target (same heuristic the
    # OptionLengthBalanceGate uses).
    distractor_lengths = sorted(len(d.get("text", "") or "") for d in distractors)
    target_length = distractor_lengths[len(distractor_lengths) // 2] if distractor_lengths else 80

    # Mutable copy of options we'll patch in place
    patched_options = [dict(o) for o in options]
    total_in = 0
    total_out = 0
    errors: list[str] = []

    for flag in flagged:
        letter = flag.get("letter", "?")
        target_idx = next(
            (i for i, o in enumerate(patched_options)
             if o.get("letter") == letter and not o.get("is_correct")),
            None,
        )
        if target_idx is None:
            errors.append(f"flagged letter {letter}: not found among distractors")
            continue

        target_opt = patched_options[target_idx]

        # Build the "other distractors" block (the patched, ordered
        # ones excluding the one we're rewriting). The correct answer
        # is shown separately so the LLM doesn't accidentally rewrite
        # toward it.
        other_block_lines = []
        for o in patched_options:
            if o is target_opt or o.get("is_correct"):
                continue
            other_block_lines.append(
                f"  {o.get('letter', '?')} [distractor]: {o.get('text', '')}"
            )
        other_distractors_block = "\n".join(other_block_lines) or "  (none)"

        prompt = FIX_PROMPT.format(
            stem=question.get("question_stem", ""),
            correct_letter=correct.get("letter", "?"),
            correct_text=correct.get("text", ""),
            other_distractors_block=other_distractors_block,
            letter=letter,
            flagged_text=flag.get("distractor_text", target_opt.get("text", "")),
            misconception_id=target_opt.get("misconception_id", "?"),
            misconception_type=target_opt.get("misconception_type", "?"),
            contradicted_stem_fact=flag.get("contradicted_stem_fact", ""),
            flag_explanation=flag.get("explanation", ""),
            target_length=target_length,
        )

        async with semaphore:
            try:
                response = await client.messages.create(
                    model=MODEL_ID,
                    max_tokens=1024,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text if response.content else ""
                total_in += response.usage.input_tokens
                total_out += response.usage.output_tokens
            except Exception as e:
                errors.append(f"letter {letter}: API error {e}")
                continue

        parsed = parse_response(text)
        if not parsed or "text" not in parsed:
            errors.append(f"letter {letter}: rewrite parse failed")
            continue

        # Patch the distractor in place — preserve ID/concept/slot/
        # misconception/letter/is_correct, only update text + explanation.
        target_opt["text"] = parsed["text"]
        if "explanation" in parsed:
            target_opt["explanation"] = parsed["explanation"]

    patched_question = dict(question)
    patched_question["options"] = patched_options

    return {
        "question_id": question.get("question_id", "?"),
        "patched": len(flagged) - len(errors) > 0,
        "fixes_applied": len(flagged) - len(errors),
        "fixes_attempted": len(flagged),
        "question": patched_question,
        "usage": {"input_tokens": total_in, "output_tokens": total_out},
        "errors": errors,
    }


def load_api_key() -> str:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    for p in [pathlib.Path(".env"), pathlib.Path.home() / ".env"]:
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("\"'")
    raise RuntimeError("No API key. Set ANTHROPIC_API_KEY env var.")


async def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "path",
        help="Path to a single batch JSON file OR a directory of them",
    )
    parser.add_argument(
        "--workers", type=int, default=5,
        help="Concurrent API workers (default: 5)",
    )
    parser.add_argument(
        "--api-key", default=None, help="Override API key",
    )
    parser.add_argument(
        "--fix", action="store_true",
        help="After auditing, rewrite each flagged distractor via Sonnet "
             "and save the patched batch to <batch>.fixed.json. "
             "Costs ~$0.005 per flagged distractor.",
    )
    parser.add_argument(
        "--passes", type=int, default=1,
        help="Number of audit passes per question (Phase 21a quorum). "
             "Default 1 (single-pass, current behavior). Use 3 to reduce "
             "single-pass jitter on borderline cases via majority vote. "
             "Cost scales linearly with passes.",
    )
    args = parser.parse_args()

    target_path = pathlib.Path(args.path)
    if target_path.is_dir():
        batch_paths = sorted(p for p in target_path.glob("*.json")
                              if "_backup" not in p.name and "audit" not in p.name)
    else:
        batch_paths = [target_path]

    if not batch_paths:
        print("No batch files found.", file=sys.stderr)
        sys.exit(1)

    api_key = args.api_key or load_api_key()
    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(args.workers)

    grand_total_questions = 0
    grand_total_flagged = 0
    grand_total_cost = 0.0
    audit_started = datetime.now(timezone.utc).isoformat()

    for batch_path in batch_paths:
        questions = load_questions(batch_path)
        if not questions:
            print(f"\n--- {batch_path.name}: empty ---", flush=True)
            continue

        print(f"\n=== {batch_path.name} ({len(questions)} questions) ===",
              flush=True)

        # Concurrent audit
        tasks = [audit_question(client, q, semaphore, n_passes=args.passes)
                 for q in questions]
        results = await asyncio.gather(*tasks)

        # Print per-question flags
        batch_flagged = 0
        batch_cost = 0.0
        for r in results:
            cost = (
                r["usage"]["input_tokens"] / 1e6 * INPUT_PRICE_PER_M
                + r["usage"]["output_tokens"] / 1e6 * OUTPUT_PRICE_PER_M
            )
            batch_cost += cost
            if r.get("error"):
                print(f"  ! {r['question_id']}: {r['error']}", flush=True)
                continue
            for f in r.get("flagged_distractors", []):
                batch_flagged += 1
                print(f"\n  [T{r['tier']}] {r['question_id']}", flush=True)
                print(f"    STEM: {r['stem']}{'...' if len(r['stem']) >= 200 else ''}",
                      flush=True)
                print(f"    Distractor {f.get('letter', '?')}: "
                      f"{f.get('distractor_text', '')}", flush=True)
                print(f"    Contradicted by: '{f.get('contradicted_stem_fact', '')}'",
                      flush=True)
                print(f"    Why: {f.get('explanation', '')}", flush=True)

        # Summary line for this batch
        print(f"\n  Batch summary: {batch_flagged} flagged / "
              f"{len(questions)} audited, ${batch_cost:.4f}", flush=True)

        # Save full audit to a sibling file
        audit_out = batch_path.with_suffix(".audit.json")
        with open(audit_out, "w", encoding="utf-8") as fh:
            json.dump({
                "audit_metadata": {
                    "audited_at": audit_started,
                    "model": MODEL_ID,
                    "source_batch": batch_path.name,
                    "questions_audited": len(questions),
                    "flagged_count": batch_flagged,
                    "cost_usd": round(batch_cost, 4),
                },
                "results": results,
            }, fh, indent=2, ensure_ascii=False)
        print(f"  Saved audit -> {audit_out.name}", flush=True)

        # ── Phase 11: --fix step ────────────────────────────────────
        # Rewrite flagged distractors via Sonnet and save the patched
        # batch to a sibling .fixed.json file. The original batch is
        # preserved so the human can compare or roll back.
        if args.fix and batch_flagged > 0:
            print(f"\n  Rewriting flagged distractors (--fix)...", flush=True)
            fix_tasks = [
                fix_question(client, q, r, semaphore)
                for q, r in zip(questions, results)
            ]
            fix_results = await asyncio.gather(*fix_tasks)
            fix_in = sum(fr["usage"]["input_tokens"] for fr in fix_results)
            fix_out = sum(fr["usage"]["output_tokens"] for fr in fix_results)
            fix_cost = (
                fix_in / 1e6 * INPUT_PRICE_PER_M
                + fix_out / 1e6 * OUTPUT_PRICE_PER_M
            )
            patched_count = sum(1 for fr in fix_results if fr.get("patched"))
            applied = sum(fr.get("fixes_applied", 0) for fr in fix_results)
            attempted = sum(fr.get("fixes_attempted", 0) for fr in fix_results)
            for fr in fix_results:
                for err in fr.get("errors") or []:
                    print(f"    ! {fr['question_id']}: {err}", flush=True)

            patched_questions = [fr["question"] for fr in fix_results]
            fixed_out = batch_path.with_suffix(".fixed.json")
            with open(fixed_out, "w", encoding="utf-8") as fh:
                # Preserve the original batch's top-level shape.
                # Original is a list-at-top-level; we keep that.
                json.dump(patched_questions, fh, indent=2,
                          ensure_ascii=False)
            print(
                f"  Patched {patched_count} question(s), "
                f"{applied}/{attempted} distractor rewrites succeeded, "
                f"${fix_cost:.4f}",
                flush=True,
            )
            print(f"  Saved fixed batch -> {fixed_out.name}", flush=True)
            batch_cost += fix_cost

        grand_total_questions += len(questions)
        grand_total_flagged += batch_flagged
        grand_total_cost += batch_cost

    print(f"\n{'='*60}", flush=True)
    print(f"GRAND TOTAL: {grand_total_questions} questions audited, "
          f"{grand_total_flagged} stem-eliminable distractors flagged",
          flush=True)
    print(f"Total cost: ${grand_total_cost:.4f}", flush=True)
    print(f"{'='*60}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
