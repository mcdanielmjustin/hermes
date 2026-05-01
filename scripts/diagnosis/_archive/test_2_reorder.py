"""Diagnosis Test 2 — JSON output reorder (Layer 4 + Layer 6).

Hypothesis: putting `distractors` before `question_stem` in the JSON output
template causes Opus to commit to misconceptions before composing the stem,
breaking the negation-from-stem path that produces english_gap distractors.

Approach: build a minimal-but-representative system prompt that carries the
key generation rules from goliath's `pipeline/prompts.py`. Generate one
question per tier on CPAT D3-PPA-003 with TWO variants:

  CONTROL: JSON template has stem-first (matches goliath's current output).
  INTERVENTION: JSON template has distractors-first + WRITING ORDER hint.

Both variants get the SAME body; only the output-format section differs.
Audit both via Sonnet n_passes=3. Compare english_gap rates.

Cost: ~$3 (8 generations Opus + 8 audits Sonnet × 3 passes).

Run:
  python scripts/diagnosis/test_2_reorder.py
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys
from datetime import datetime, timezone

import anthropic

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_stem_contradictions import (  # noqa: E402
    audit_question, parse_response, load_api_key,
    INPUT_PRICE_PER_M, OUTPUT_PRICE_PER_M,
)

OPUS_MODEL_ID = "claude-opus-4-7"
OPUS_INPUT_PRICE_PER_M = 15.0
OPUS_OUTPUT_PRICE_PER_M = 75.0

ANCHOR_UID = "D3-PPA-003-bda8c4e9"
ANCHOR_TESTABLE_FACT = (
    "The lifetime prevalence of major depressive disorder shows a robust "
    "sex difference: rates are roughly equivalent in childhood, diverge "
    "during puberty, and reach approximately 2:1 (female-to-male) by "
    "adulthood. This pattern is termed pubertal divergence and is one of "
    "the most consistent findings in psychiatric epidemiology."
)
ANCHOR_DOMAIN = "CPAT (Clinical Psychopathology)"
ANCHOR_CHAPTER = "When the Light Goes Out: Depression — Its Models and Who It Strikes"

OUT_DIR = REPO_ROOT / "data" / ".diagnosis" / "test_2"

TIER_INSTRUCTIONS = {
    1: ("Tier 1 — Remember (recall): direct definitional question. NO scenario, "
        "no vignette, no named subject. Stem asks for a term, label, or "
        "definitional characterization."),
    2: ("Tier 2 — Understand (comprehension): brief context (1-2 sentences max). "
        "Ask for recognition of an example, distinction, or classification. NO "
        "evaluative framings ('most appropriate'); those are T4."),
    3: ("Tier 3 — Apply (application): novel scenario (≤4 sentences). Ask for "
        "predicting an outcome, selecting an action, or evaluating a trade-off "
        "based on scenario-specific details."),
    4: ("Tier 4 — Evaluate (analysis): complex case (≥2 sentences) with competing "
        "claims or conjunctive complexity. Ask for judging, critiquing, or "
        "synthesizing across concepts."),
}

# Body shared between both variants — captures the distillation of goliath's
# system prompt that's relevant to english_gap. NOT goliath's full prompt
# (that would be ~3000 tokens); this is the minimum needed for a fair test
# of the JSON-order variable.
SHARED_BODY = """You are an expert EPPP exam question author. Generate a multiple-choice question grounded in the anchor provided.

## Distractor Quality Framework (CRITICAL)

Every distractor will be classified by a downstream audit as one of:

ENGLISH_GAP (FORBIDDEN) — A student can reject the distractor by lexical comparison with the stem alone. No domain knowledge needed. The contradiction is in printed words.
  Example: Stem says "Lester recalls his wedding from a decade earlier"; distractor says "Retrograde amnesia erases ALL pre-injury memories." The universal "all" is contradicted by the wedding example. Student needs no concept knowledge.

CONTENT_GAP (PREFERRED) — Rejecting requires invoking concept knowledge. A real contradiction exists, but recognizing it requires knowing what a technical term means.

CLEAN (PREFERRED) — No direct contradiction with stem facts. Plausible-but-wrong; rejection requires applying the concept.

DESIGN RULE: For each distractor, ask "could a student reject this by re-reading the stem alone, with NO domain knowledge?" If YES, you have written english_gap. Redesign.

## Stem & Distractor Format Hygiene

Stem: pose ONE direct question. FORBIDDEN: "Which option correctly identifies", "Which best describes", "Which most accurately", or any meta-evaluative modifiers ("correctly", "best", "most", "option") qualifying the answer choice.

Distractors: options are ANSWERS, not INSTRUCTIONS. FORBIDDEN: imperative-verb leads ("Identify the X", "Classify Y", "Recognize Z", "Predict X" as a lead form, "Evaluate option (a)"). Options must be noun phrases or short declarative claims.

All four options must share the same grammatical form and start with the same part of speech.

## Anchor (factual basis — do not deviate)

{anchor_testable_fact}

Domain: {anchor_domain}
Chapter: {anchor_chapter}

## Difficulty Tier

{tier_instruction}

## Distractor design

Generate 3 wrong answers. Each probes a DIFFERENT misconception about the anchor's testable fact. Wrong via concept knowledge, NOT via lexical contradiction with the stem."""

OUTPUT_FORMAT_CONTROL = """
## Output Format

Return ONLY valid JSON (no markdown, no preamble, no explanation):

{{
  "question_stem": "...",
  "tested_concept": "...",
  "correct_answer": {{
    "letter": "A|B|C|D",
    "text": "...",
    "explanation": "1-2 sentences"
  }},
  "distractors": [
    {{"letter": "A|B|C|D", "text": "...", "explanation": "1-2 sentences"}},
    {{"letter": "A|B|C|D", "text": "...", "explanation": "1-2 sentences"}},
    {{"letter": "A|B|C|D", "text": "...", "explanation": "1-2 sentences"}}
  ]
}}

Each option (correct + 3 distractors) gets a distinct letter A through D."""

OUTPUT_FORMAT_INTERVENTION = """
## Output Format

WRITING ORDER (this matters): Design your output in this order. Do NOT write the stem first.

1. tested_concept — what concept is being tested.
2. distractors — design THREE specific cognitive errors a student might make about the tested concept. Each is wrong via concept knowledge. You do NOT yet have a stem to reference; the distractors must stand on their own as wrong-answers about the tested concept.
3. correct_answer — write the correct answer that makes those specific distractors clearly wrong.
4. question_stem — compose the stem to fit. The stem should be answerable by selecting the correct option among these specific distractors. The stem MUST NOT print specific facts (numbers, ratios, directions, named outcomes) that any distractor lexically contradicts — if you find yourself writing such a fact, redesign the distractor instead.

Return ONLY valid JSON (no markdown, no preamble, no explanation):

{{
  "tested_concept": "...",
  "distractors": [
    {{"letter": "A|B|C|D", "text": "...", "explanation": "1-2 sentences"}},
    {{"letter": "A|B|C|D", "text": "...", "explanation": "1-2 sentences"}},
    {{"letter": "A|B|C|D", "text": "...", "explanation": "1-2 sentences"}}
  ],
  "correct_answer": {{
    "letter": "A|B|C|D",
    "text": "...",
    "explanation": "1-2 sentences"
  }},
  "question_stem": "..."
}}

Each option (correct + 3 distractors) gets a distinct letter A through D."""


def _build_prompt(tier: int, intervention: bool) -> str:
    body = SHARED_BODY.format(
        anchor_testable_fact=ANCHOR_TESTABLE_FACT,
        anchor_domain=ANCHOR_DOMAIN,
        anchor_chapter=ANCHOR_CHAPTER,
        tier_instruction=TIER_INSTRUCTIONS[tier],
    )
    out_fmt = OUTPUT_FORMAT_INTERVENTION if intervention else OUTPUT_FORMAT_CONTROL
    return body + out_fmt


def _opus_cost(usage: dict) -> float:
    return (
        usage.get("input_tokens", 0) / 1e6 * OPUS_INPUT_PRICE_PER_M
        + usage.get("output_tokens", 0) / 1e6 * OPUS_OUTPUT_PRICE_PER_M
    )


def _sonnet_cost(usage: dict) -> float:
    return (
        usage.get("input_tokens", 0) / 1e6 * INPUT_PRICE_PER_M
        + usage.get("output_tokens", 0) / 1e6 * OUTPUT_PRICE_PER_M
    )


async def generate_one(client, tier: int, intervention: bool, semaphore) -> dict:
    """Generate one question. Returns a goliath-shaped question dict."""
    prompt = _build_prompt(tier, intervention)
    async with semaphore:
        try:
            response = await client.messages.create(
                model=OPUS_MODEL_ID,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text if response.content else ""
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        except Exception as e:
            return {"error": f"api_error: {e}", "tier": tier,
                    "intervention": intervention, "usage": {"input_tokens": 0, "output_tokens": 0}}

    parsed = parse_response(text)
    if not parsed:
        return {"error": "parse_failed", "raw": text[:300], "tier": tier,
                "intervention": intervention, "usage": usage}

    # Convert response into goliath-shaped question dict for the audit
    correct = parsed.get("correct_answer") or {}
    distractors = parsed.get("distractors") or []
    options = []
    correct_letter = correct.get("letter", "?")
    for d in distractors:
        options.append({
            "letter": d.get("letter", "?"),
            "text": d.get("text", ""),
            "is_correct": False,
            "explanation": d.get("explanation", ""),
        })
    options.append({
        "letter": correct_letter,
        "text": correct.get("text", ""),
        "is_correct": True,
        "explanation": correct.get("explanation", ""),
    })
    options.sort(key=lambda o: o.get("letter", ""))

    question = {
        "question_id": f"TEST2-T{tier}-{'INT' if intervention else 'CTRL'}",
        "difficulty_tier": tier,
        "question_stem": parsed.get("question_stem", ""),
        "options": options,
        "tested_concept": parsed.get("tested_concept", ""),
        "_writing_order_hint": "distractors_first" if intervention else "stem_first",
        "_generation_usage": usage,
    }
    return question


def _english_gap_count(audit_results: list[dict]) -> int:
    return sum(
        1
        for r in audit_results if not r.get("error")
        for c in (r.get("classifications") or [])
        if c.get("class") == "english_gap"
    )


def _summarize(audit_results: list[dict]) -> dict:
    counts = {"english_gap": 0, "content_gap": 0, "clean": 0, "soft_flag": 0, "error": 0}
    for r in audit_results:
        if r.get("error"):
            counts["error"] += 1
            continue
        for c in r.get("classifications") or []:
            cls = c.get("class")
            if cls in counts:
                counts[cls] += 1
    return counts


async def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    api_key = load_api_key()
    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(4)

    # Generate 4 control + 4 intervention (one per tier)
    print("=== Generation: 4 control (stem-first) + 4 intervention (distractors-first) ===")
    gen_tasks = [generate_one(client, tier, False, semaphore) for tier in (1, 2, 3, 4)]
    gen_tasks += [generate_one(client, tier, True, semaphore) for tier in (1, 2, 3, 4)]
    questions = await asyncio.gather(*gen_tasks)
    control = questions[:4]
    intervention = questions[4:]

    gen_cost = 0.0
    parse_failures = 0
    for q in questions:
        if q.get("error"):
            parse_failures += 1
        gen_cost += _opus_cost(q.get("_generation_usage") or q.get("usage") or {})
    print(f"  parse failures: {parse_failures}")
    print(f"  generation cost: ${gen_cost:.4f}")
    print()

    # Filter out errored generations
    valid_control = [q for q in control if not q.get("error")]
    valid_intervention = [q for q in intervention if not q.get("error")]

    if not valid_control or not valid_intervention:
        print("ERROR: insufficient valid generations to compare")
        return

    # Audit both
    print(f"=== Audit (control: {len(valid_control)} q, intervention: {len(valid_intervention)} q) ===")
    audit_tasks_ctrl = [audit_question(client, q, semaphore, n_passes=3) for q in valid_control]
    audit_tasks_int = [audit_question(client, q, semaphore, n_passes=3) for q in valid_intervention]
    audits_ctrl = await asyncio.gather(*audit_tasks_ctrl)
    audits_int = await asyncio.gather(*audit_tasks_int)

    audit_cost = sum(_sonnet_cost(r.get("usage", {})) for r in audits_ctrl + audits_int)

    ctrl_eg = _english_gap_count(audits_ctrl)
    int_eg = _english_gap_count(audits_int)
    ctrl_summary = _summarize(audits_ctrl)
    int_summary = _summarize(audits_int)

    print(f"  CONTROL    summary: {ctrl_summary}")
    print(f"  INTERVENTION summary: {int_summary}")
    print(f"  audit cost: ${audit_cost:.4f}")
    print()

    print("=" * 60)
    print(f"CONTROL english_gap     : {ctrl_eg} / {sum(ctrl_summary[c] for c in ('english_gap','content_gap','clean','soft_flag'))} distractors")
    print(f"INTERVENTION english_gap: {int_eg} / {sum(int_summary[c] for c in ('english_gap','content_gap','clean','soft_flag'))} distractors")
    delta = int_eg - ctrl_eg
    print(f"DELTA: {delta:+d}")
    total_cost = gen_cost + audit_cost
    print(f"Total test cost: ${total_cost:.4f}")
    print("=" * 60)

    # Save artifacts
    artifact = {
        "test": "test_2_reorder",
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "anchor_uid": ANCHOR_UID,
        "control": {
            "questions": [{"qid": q.get("question_id"), "stem": q.get("question_stem", "")[:200], "options": q.get("options")} for q in valid_control],
            "audit_summary": ctrl_summary,
            "english_gap_count": ctrl_eg,
        },
        "intervention": {
            "questions": [{"qid": q.get("question_id"), "stem": q.get("question_stem", "")[:200], "options": q.get("options")} for q in valid_intervention],
            "audit_summary": int_summary,
            "english_gap_count": int_eg,
        },
        "delta_english_gap": delta,
        "generation_cost_usd": round(gen_cost, 4),
        "audit_cost_usd": round(audit_cost, 4),
        "total_cost_usd": round(total_cost, 4),
    }
    with open(OUT_DIR / "results.json", "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2, ensure_ascii=False)
    print()
    print(f"Artifacts saved to {OUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
