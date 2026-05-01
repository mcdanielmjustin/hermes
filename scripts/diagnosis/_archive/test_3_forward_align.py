"""Diagnosis Test 3 — Forward-aligned objective (Layer 2 + Layer 9).

Hypothesis: replacing the system prompt's leading paragraph with "your output
will be classified by [audit's exact rubric verbatim], optimize for
english_gap=0" shifts Opus's optimization target from "be a good question
writer" to "produce questions that pass the audit." The audit's exact wording
becomes the objective rather than a paraphrase.

Approach: same CONTROL/INTERVENTION pattern as Test 2. Both share the SAME
body and SAME (stem-first) JSON output template. Only the LEADING PARAGRAPH
differs. CONTROL leads with "you are an EPPP exam question author"; INTERVENTION
leads with "your output will be classified by [paste audit prompt verbatim]".

To isolate from Test 2's confounder: BOTH variants in this test use the
SAME (stem-first) JSON output. So this test isolates the leading-paragraph
intervention.

Cost: ~$1.50.
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
from test_2_reorder import (  # noqa: E402
    OPUS_MODEL_ID, OPUS_INPUT_PRICE_PER_M, OPUS_OUTPUT_PRICE_PER_M,
    ANCHOR_UID, ANCHOR_TESTABLE_FACT, ANCHOR_DOMAIN, ANCHOR_CHAPTER,
    TIER_INSTRUCTIONS, OUTPUT_FORMAT_CONTROL,
    _opus_cost, _sonnet_cost, _english_gap_count, _summarize,
)

OUT_DIR = REPO_ROOT / "data" / ".diagnosis" / "test_3"

# CONTROL leading paragraph (matches goliath's current style):
LEADING_CONTROL = """You are an expert EPPP exam question author. Generate a multiple-choice question grounded in the anchor provided."""

# INTERVENTION leading paragraph: forward-aligned with the audit's exact rubric
# pasted verbatim. This is "your output will be classified by [this rubric]".
LEADING_INTERVENTION = """## YOUR PRIMARY OBJECTIVE

Your output will be classified by a downstream Sonnet 4.6 audit using the rubric below verbatim. Your SUCCESS CRITERION is: every distractor classifies as CONTENT_GAP or CLEAN. NEVER english_gap.

THE AUDIT'S EXACT RUBRIC — your output is evaluated against this text, not paraphrases:

ENGLISH_GAP — A student can reject the distractor by lexical comparison with the stem alone. No domain knowledge needed. The contradiction is in the printed words, not in the concepts.
  Canonical example:
    Stem: "After bilateral hippocampal damage, Lester Nichols cannot form new declarative memories but still recalls his wedding from a decade earlier."
    Distractor: "Retrograde amnesia erases ALL pre-injury memories regardless of when they were consolidated."
    Why english_gap: the universal "all pre-injury memories" is contradicted by "wedding from a decade earlier" — student rejects without concept knowledge.
  Pattern: universal quantifiers vs specific stem counter-examples; laterality inversions; distractors that contradict stem-stated findings.

CONTENT_GAP — Rejecting the distractor requires invoking concept/domain knowledge. A real contradiction exists, but recognizing it requires knowing what some technical term means.
  Canonical example:
    Stem: "compound binds receptor but produces no measurable postsynaptic activity on its own."
    Distractor: "compound has intrinsic activity that mimics the endogenous neurotransmitter."
    Why content_gap: rejecting requires knowing that "intrinsic activity" implies a measurable postsynaptic effect.

CLEAN — No direct contradiction with stem facts. Plausible-but-wrong; rejection requires applying the concept.

CLASSIFICATION RULE for each distractor (the audit applies this to your output):
  "Could a student reject this distractor by re-reading the stem and the distractor alone, with NO domain knowledge invoked?"
  • YES → english_gap (FAIL)
  • NO, but a real contradiction exists once you know the concept → content_gap (PASS)
  • NO direct contradiction at all, just a wrong-but-plausible alternative → clean (PASS)

OPTIMIZE for the audit's verdict. Every other rule below supports this primary objective.

---

You are an EPPP exam question author. Generate a multiple-choice question grounded in the anchor provided."""

SHARED_BODY_TAIL = """

## Stem & Distractor Format Hygiene

Stem: pose ONE direct question. FORBIDDEN: "Which option correctly identifies", "Which best describes", "Which most accurately", or any meta-evaluative modifiers.

Distractors: options are ANSWERS, not INSTRUCTIONS. FORBIDDEN: imperative-verb leads ("Identify the X", "Predict X" as a lead form, "Evaluate option (a)").

All four options must share the same grammatical form.

## Anchor (factual basis — do not deviate)

{anchor_testable_fact}

Domain: {anchor_domain}
Chapter: {anchor_chapter}

## Difficulty Tier

{tier_instruction}

## Distractor design

Generate 3 wrong answers. Each probes a DIFFERENT misconception. Wrong via concept knowledge, NOT via lexical contradiction with the stem."""


def _build_prompt(tier: int, intervention: bool) -> str:
    leading = LEADING_INTERVENTION if intervention else LEADING_CONTROL
    body = SHARED_BODY_TAIL.format(
        anchor_testable_fact=ANCHOR_TESTABLE_FACT,
        anchor_domain=ANCHOR_DOMAIN,
        anchor_chapter=ANCHOR_CHAPTER,
        tier_instruction=TIER_INSTRUCTIONS[tier],
    )
    return leading + body + OUTPUT_FORMAT_CONTROL  # both use stem-first JSON


async def generate_one(client, tier: int, intervention: bool, semaphore) -> dict:
    prompt = _build_prompt(tier, intervention)
    async with semaphore:
        try:
            response = await client.messages.create(
                model=OPUS_MODEL_ID,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text if response.content else ""
            usage = {"input_tokens": response.usage.input_tokens,
                     "output_tokens": response.usage.output_tokens}
        except Exception as e:
            return {"error": f"api_error: {e}", "tier": tier,
                    "intervention": intervention, "usage": {"input_tokens": 0, "output_tokens": 0}}

    parsed = parse_response(text)
    if not parsed:
        return {"error": "parse_failed", "raw": text[:300], "tier": tier,
                "intervention": intervention, "usage": usage}

    correct = parsed.get("correct_answer") or {}
    distractors = parsed.get("distractors") or []
    options = []
    for d in distractors:
        options.append({"letter": d.get("letter", "?"), "text": d.get("text", ""),
                        "is_correct": False, "explanation": d.get("explanation", "")})
    options.append({"letter": correct.get("letter", "?"), "text": correct.get("text", ""),
                    "is_correct": True, "explanation": correct.get("explanation", "")})
    options.sort(key=lambda o: o.get("letter", ""))

    return {
        "question_id": f"TEST3-T{tier}-{'INT' if intervention else 'CTRL'}",
        "difficulty_tier": tier,
        "question_stem": parsed.get("question_stem", ""),
        "options": options,
        "tested_concept": parsed.get("tested_concept", ""),
        "_intervention": intervention,
        "_generation_usage": usage,
    }


async def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = anthropic.AsyncAnthropic(api_key=load_api_key())
    semaphore = asyncio.Semaphore(4)

    print("=== Test 3: forward-aligned objective ===")
    print("Generating 4 control + 4 intervention (both stem-first JSON)...")
    gen_tasks = [generate_one(client, t, False, semaphore) for t in (1, 2, 3, 4)]
    gen_tasks += [generate_one(client, t, True, semaphore) for t in (1, 2, 3, 4)]
    questions = await asyncio.gather(*gen_tasks)
    control = [q for q in questions[:4] if not q.get("error")]
    intervention = [q for q in questions[4:] if not q.get("error")]
    gen_cost = sum(_opus_cost(q.get("_generation_usage") or q.get("usage") or {}) for q in questions)
    print(f"  parse failures: {sum(1 for q in questions if q.get('error'))}")
    print(f"  generation cost: ${gen_cost:.4f}")

    print(f"\n=== Audit ===")
    audits_ctrl = await asyncio.gather(*[audit_question(client, q, semaphore, n_passes=3) for q in control])
    audits_int = await asyncio.gather(*[audit_question(client, q, semaphore, n_passes=3) for q in intervention])
    audit_cost = sum(_sonnet_cost(r.get("usage", {})) for r in audits_ctrl + audits_int)

    ctrl_eg = _english_gap_count(audits_ctrl)
    int_eg = _english_gap_count(audits_int)
    ctrl_summary = _summarize(audits_ctrl)
    int_summary = _summarize(audits_int)

    print(f"  CONTROL summary: {ctrl_summary}")
    print(f"  INTERVENTION summary: {int_summary}")
    print(f"  audit cost: ${audit_cost:.4f}")

    print("=" * 60)
    print(f"CONTROL english_gap     : {ctrl_eg}")
    print(f"INTERVENTION english_gap: {int_eg}")
    print(f"DELTA: {int_eg - ctrl_eg:+d}")
    print(f"Total cost: ${gen_cost + audit_cost:.4f}")
    print("=" * 60)

    with open(OUT_DIR / "results.json", "w", encoding="utf-8") as fh:
        json.dump({
            "test": "test_3_forward_align",
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "control_summary": ctrl_summary, "control_eg": ctrl_eg,
            "intervention_summary": int_summary, "intervention_eg": int_eg,
            "delta_eg": int_eg - ctrl_eg,
            "total_cost_usd": round(gen_cost + audit_cost, 4),
            "control_questions": [
                {"qid": q.get("question_id"), "stem": q.get("question_stem", "")[:200],
                 "options": q.get("options")} for q in control],
            "intervention_questions": [
                {"qid": q.get("question_id"), "stem": q.get("question_stem", "")[:200],
                 "options": q.get("options")} for q in intervention],
        }, fh, indent=2, ensure_ascii=False)
    print(f"Artifacts: {OUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
