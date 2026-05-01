"""Phase 26b — Target-driven generation prototype.

Demonstrates the frame shift in the GENERATIVE direction: take a
measurement_target as INPUT and produce an item that probes it. The
target's distractor_diagnoses become the generation spec — Opus
constructs distractor text that manifests each declared candidate_state.

This is the architectural pair to Phase 26a's inference: inference
extracts targets from existing items; generation produces items from
targets. Together they prove that measurement_target is the right
intermediate representation.

Usage:
  python scripts/diagnosis/generate_from_target.py
  (uses the canonical example target by default; demonstrates the flow)

Output:
  data/.diagnosis/measurement_targets/generated_<timestamp>.json
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
    parse_response, load_api_key, audit_question,
    INPUT_PRICE_PER_M, OUTPUT_PRICE_PER_M,
)
from pipeline.measurement_target import (  # noqa: E402
    MeasurementTarget, DistractorDiagnosis, CANONICAL_EXAMPLE, to_dict,
)

OPUS_MODEL_ID = "claude-opus-4-7"
OPUS_INPUT_PRICE_PER_M = 15.0
OPUS_OUTPUT_PRICE_PER_M = 75.0

OUT_DIR = REPO_ROOT / "data" / ".diagnosis" / "measurement_targets"


GENERATE_FROM_TARGET_PROMPT = """You are an expert EPPP item writer constructing a multiple-choice question from a measurement-instrument specification. The MEASUREMENT TARGET tells you exactly what knowledge state to discriminate, what each distractor should reveal, and what level of discrimination to aim for.

Your job: produce the question STEM and OPTIONS such that:
1. A candidate in the `knower_state` picks the correct option for the reason given.
2. A candidate matching each `candidate_state` in the distractor_diagnoses picks the corresponding distractor.
3. The stem does NOT print specific facts that distractors lexically contradict (forbidden — that's english_gap).
4. The cognitive demand matches the labeled Bloom's tier.

STEM HYGIENE (these prevent english_gap and editorial issues):
- NEVER print numbers, ratios, named outcomes, lateralities, or stage timings that any distractor lexically contradicts.
- NEVER use meta-evaluative modifiers in the stem ("correctly", "best", "most", "option").
- NEVER lead distractors with imperative verbs ("Identify", "Predict", "Classify"). Distractors must be answer-form (noun phrases or declarative claims).
- All four option leads must share grammatical form.

ANCHOR (factual basis):
{anchor_testable_fact}

Domain: {domain}
Bloom's Tier: {tier} ({tier_name})

MEASUREMENT TARGET:

{target_json}

OUTPUT — single JSON object (no preamble, no markdown):

{{
  "question_stem": "...",
  "options": [
    {{"letter": "A", "text": "...", "is_correct": false, "explanation": "..."}},
    {{"letter": "B", "text": "...", "is_correct": false, "explanation": "..."}},
    {{"letter": "C", "text": "...", "is_correct": false, "explanation": "..."}},
    {{"letter": "D", "text": "...", "is_correct": false, "explanation": "..."}}
  ]
}}

The option marked is_correct=true MUST be the one matching the target's expected_correct_letter. Each non-correct option's text MUST manifest its declared candidate_state from the target.distractor_diagnoses."""


TIER_NAMES = {1: "Remember", 2: "Understand", 3: "Apply", 4: "Evaluate"}


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


async def generate_from_target(
    client, anchor_testable_fact: str, domain: str, tier: int,
    target: MeasurementTarget, semaphore,
) -> dict:
    """Generate one question from a measurement_target. Returns dict with
    question + usage + errors."""
    target_json = json.dumps(to_dict(target), indent=2)
    prompt = GENERATE_FROM_TARGET_PROMPT.format(
        anchor_testable_fact=anchor_testable_fact,
        domain=domain,
        tier=tier,
        tier_name=TIER_NAMES.get(tier, "Unknown"),
        target_json=target_json,
    )

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
            return {"error": f"api_error: {e}", "question": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0}}

    parsed = parse_response(text)
    if not parsed:
        return {"error": "parse_failed", "raw": text[:300],
                "question": None, "usage": usage}

    options = parsed.get("options") or []
    # Enforce expected_correct_letter — set is_correct=true on that one.
    for o in options:
        o["is_correct"] = (o.get("letter") == target.expected_correct_letter)

    question = {
        "question_id": f"GENERATED-FROM-TARGET-T{tier}",
        "difficulty_tier": tier,
        "question_stem": parsed.get("question_stem", ""),
        "options": options,
        "_target_used": to_dict(target),
    }
    return {"question": question, "usage": usage}


async def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    api_key = load_api_key()
    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(2)

    # Use the canonical depression-epidemiology target
    target = CANONICAL_EXAMPLE
    anchor_testable_fact = (
        "The lifetime prevalence of major depressive disorder shows a "
        "robust sex difference: rates are roughly equivalent in childhood, "
        "diverge during puberty, and reach approximately 2:1 (female-to-"
        "male) by adulthood. This pattern is termed pubertal divergence."
    )
    domain = "CPAT (Clinical Psychopathology)"
    tier = 1  # Remember-tier definitional question

    print("=== Target-driven generation ===")
    print(f"Domain: {domain}")
    print(f"Tier: {tier}")
    print(f"Target competency: {target.competency_claim[:120]}...")
    print()

    result = await generate_from_target(
        client, anchor_testable_fact, domain, tier, target, semaphore,
    )

    gen_cost = _opus_cost(result.get("usage", {}))
    if result.get("error"):
        print(f"ERROR: {result['error']}")
        sys.exit(1)

    q = result["question"]
    print("Generated question:")
    print(f"  Stem: {q['question_stem']}")
    print(f"  Options:")
    for o in q.get("options", []):
        marker = "[+]" if o.get("is_correct") else "   "
        print(f"    {marker} {o.get('letter')}: {o.get('text', '')[:130]}")
    print()
    print(f"Generation cost: ${gen_cost:.4f}")
    print()

    # Validate against the audit
    print("=== Audit on generated question ===")
    audit_result = await audit_question(client, q, semaphore, n_passes=3)
    audit_cost = _sonnet_cost(audit_result.get("usage", {}))
    classifications = audit_result.get("classifications") or []
    eg_count = sum(1 for c in classifications if c.get("class") == "english_gap")
    print(f"  english_gap: {eg_count} / {len(classifications)}")
    print(f"  per-distractor: {[(c.get('letter'), c.get('class')) for c in classifications]}")
    print(f"  audit cost: ${audit_cost:.4f}")
    print()

    # Sanity check: did the correct answer match the target's expected letter?
    correct_option = next((o for o in q.get("options", []) if o.get("is_correct")), None)
    target_letter = target.expected_correct_letter
    actual_letter = (correct_option or {}).get("letter")
    print(f"Expected correct letter (from target): {target_letter}")
    print(f"Actual correct letter (in question):  {actual_letter}")
    print(f"Match: {target_letter == actual_letter}")
    print()

    # Save
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    artifact_path = OUT_DIR / f"generated_{timestamp.replace(':', '-')}.json"
    with open(artifact_path, "w", encoding="utf-8") as fh:
        json.dump({
            "timestamp": timestamp,
            "target": to_dict(target),
            "anchor_testable_fact": anchor_testable_fact,
            "domain": domain,
            "tier": tier,
            "generated_question": q,
            "audit_result": {
                "classifications": classifications,
                "english_gap_count": eg_count,
            },
            "letter_match": target_letter == actual_letter,
            "total_cost_usd": round(gen_cost + audit_cost, 4),
        }, fh, indent=2, ensure_ascii=False)
    print(f"Artifact: {artifact_path}")


if __name__ == "__main__":
    asyncio.run(main())
