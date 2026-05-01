"""Diagnosis Test 4 — cognitive_error schema field (Layer 5 + Layer 3).

Hypothesis: adding a required `cognitive_error` field per distractor that Opus
must populate BEFORE the option text forces misconception-first thinking inside
the single-call architecture. Negation as shortcut becomes harder because
cognitive_error must express a specific student misunderstanding (not a literal
negation).

Approach: same CONTROL/INTERVENTION pattern. CONTROL uses standard schema.
INTERVENTION schema requires cognitive_error before text per distractor.

Both variants use the SAME body and SAME stem-first JSON output template
(modulo the schema-field addition in INTERVENTION). Isolates the schema-field
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
    TIER_INSTRUCTIONS, SHARED_BODY,
    _opus_cost, _sonnet_cost, _english_gap_count, _summarize,
)

OUT_DIR = REPO_ROOT / "data" / ".diagnosis" / "test_4"

OUTPUT_FORMAT_CONTROL = """
## Output Format

Return ONLY valid JSON:

{{
  "question_stem": "...",
  "tested_concept": "...",
  "correct_answer": {{"letter": "A|B|C|D", "text": "...", "explanation": "..."}},
  "distractors": [
    {{"letter": "...", "text": "...", "explanation": "..."}},
    {{"letter": "...", "text": "...", "explanation": "..."}},
    {{"letter": "...", "text": "...", "explanation": "..."}}
  ]
}}"""

OUTPUT_FORMAT_INTERVENTION = """
## Output Format

For each distractor, the `cognitive_error` field is the DESIGN SPEC. The `text` field is the EXPRESSION. You MUST populate `cognitive_error` BEFORE writing `text`. The text must be a specific cognitive error expressed in answer form — NOT a literal negation of stem facts (which would be english_gap).

Return ONLY valid JSON:

{{
  "question_stem": "...",
  "tested_concept": "...",
  "correct_answer": {{"letter": "A|B|C|D", "text": "...", "explanation": "..."}},
  "distractors": [
    {{
      "letter": "...",
      "cognitive_error": "1 sentence: WHAT the student wrongly believes (a specific misunderstanding). 1 sentence: WHY a competent student would not believe it (the concept knowledge that distinguishes).",
      "text": "...the option text expressing this cognitive_error in answer form...",
      "explanation": "..."
    }},
    {{...}},
    {{...}}
  ]
}}

The `cognitive_error` field is mandatory and must be substantive (not platitudes). The `text` must manifest the specific cognitive_error described."""


def _build_prompt(tier: int, intervention: bool) -> str:
    body = SHARED_BODY.format(
        anchor_testable_fact=ANCHOR_TESTABLE_FACT,
        anchor_domain=ANCHOR_DOMAIN,
        anchor_chapter=ANCHOR_CHAPTER,
        tier_instruction=TIER_INSTRUCTIONS[tier],
    )
    out_fmt = OUTPUT_FORMAT_INTERVENTION if intervention else OUTPUT_FORMAT_CONTROL
    return body + out_fmt


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
    cognitive_errors = []
    for d in distractors:
        options.append({"letter": d.get("letter", "?"), "text": d.get("text", ""),
                        "is_correct": False, "explanation": d.get("explanation", "")})
        if intervention:
            cognitive_errors.append({"letter": d.get("letter", "?"),
                                      "cognitive_error": d.get("cognitive_error", "")})
    options.append({"letter": correct.get("letter", "?"), "text": correct.get("text", ""),
                    "is_correct": True, "explanation": correct.get("explanation", "")})
    options.sort(key=lambda o: o.get("letter", ""))

    return {
        "question_id": f"TEST4-T{tier}-{'INT' if intervention else 'CTRL'}",
        "difficulty_tier": tier,
        "question_stem": parsed.get("question_stem", ""),
        "options": options,
        "_cognitive_errors": cognitive_errors,
        "_intervention": intervention,
        "_generation_usage": usage,
    }


async def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    client = anthropic.AsyncAnthropic(api_key=load_api_key())
    semaphore = asyncio.Semaphore(4)

    print("=== Test 4: cognitive_error schema field ===")
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

    print("=" * 60)
    print(f"CONTROL english_gap     : {ctrl_eg}")
    print(f"INTERVENTION english_gap: {int_eg}")
    print(f"DELTA: {int_eg - ctrl_eg:+d}")
    print(f"Total cost: ${gen_cost + audit_cost:.4f}")
    print("=" * 60)

    with open(OUT_DIR / "results.json", "w", encoding="utf-8") as fh:
        json.dump({
            "test": "test_4_cognitive_error",
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
                 "options": q.get("options"), "cognitive_errors": q.get("_cognitive_errors")}
                for q in intervention],
        }, fh, indent=2, ensure_ascii=False)
    print(f"Artifacts: {OUT_DIR}")


if __name__ == "__main__":
    asyncio.run(main())
