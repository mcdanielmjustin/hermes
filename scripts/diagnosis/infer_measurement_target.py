"""Phase 26a — Post-hoc measurement_target inference.

Takes existing audit-clean goliath questions and extracts an explicit
measurement_target spec for each, using Opus 4.7 in a "reverse-engineer
the diagnostic intent" role.

This is BACKFILL: existing questions have implicit targets; the script
makes them explicit. Subsequent target-validation audits and target-
driven generation rely on this artifact.

The prompt insists on FALSIFIABLE targets — specific propositions a
candidate either has or lacks, with predictions about which option each
knowledge state would pick. Vague targets like "tests depression
epidemiology" are explicitly rejected.

Usage:
  python scripts/diagnosis/infer_measurement_target.py [chapter.json]
  python scripts/diagnosis/infer_measurement_target.py --sample 20

Output:
  data/.diagnosis/measurement_targets/<chapter_basename>.targets.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import random
import sys
from datetime import datetime, timezone

import anthropic

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_stem_contradictions import parse_response, load_api_key  # noqa: E402
from pipeline.measurement_target import (  # noqa: E402
    is_valid, CANONICAL_EXAMPLE, to_dict,
)

OPUS_MODEL_ID = "claude-opus-4-7"
OPUS_INPUT_PRICE_PER_M = 15.0
OPUS_OUTPUT_PRICE_PER_M = 75.0

QUIZ_DIR = REPO_ROOT / "data" / "quiz"
OUT_DIR = REPO_ROOT / "data" / ".diagnosis" / "measurement_targets"


INFER_PROMPT = """You are a psychometrician reverse-engineering the MEASUREMENT TARGET of an existing multiple-choice question. Your job is to make the IMPLICIT diagnostic intent EXPLICIT — what knowledge state distinguishes a candidate who picks the correct answer from a candidate who picks each distractor.

CRITICAL RULES:

1. **Be falsifiable.** Vague targets like "tests understanding of depression epidemiology" are useless. Useful targets state propositions: "distinguishes candidates who know that the F:M ratio for adult MDD is approximately 2:1 from those who don't."

2. **Be specific to the actual question.** Read the stem and options carefully. The target should explain why a knowing candidate picks the correct option specifically (not just any reasonable option).

3. **Distractor diagnoses must be different.** Each distractor's pick reveals a DIFFERENT cognitive error. If two distractors have the same diagnosis, the question has redundant distractors — note this in the rationale.

4. **Predict discrimination honestly.** "Strong" means the question would have point-biserial >0.30 in real test-taker data — the correct answer is clearly distinguished by knowing the concept. "Weak" means the correct answer could plausibly be picked by guessing or partial knowledge.

CANONICAL EXAMPLE (target for an MDD-epidemiology T1 question):

```json
{canonical_json}
```

Notice how each distractor diagnosis names a SPECIFIC cognitive error (not "they don't know epidemiology" but "they confuse MDD with conduct disorder which DOES show male-predominance in childhood").

QUESTION TO ANALYZE:

Domain: {domain}
Difficulty Tier: {tier} ({tier_name})
Anchor: {anchor_uid}
Question ID: {question_id}

STEM:
{stem}

OPTIONS:
{options_block}

OUTPUT — single JSON object exactly matching this schema (no preamble, no markdown):

{{
  "competency_claim": "1 sentence on what the candidate must be able to do",
  "knower_state": "1-2 sentences on the specific propositions a knower has",
  "non_knower_state": "1-2 sentences on what's missing",
  "expected_correct_letter": "X",
  "expected_correct_reasoning": "the cognitive steps a knower takes to pick the correct option",
  "distractor_diagnoses": [
    {{
      "letter": "...",
      "candidate_state": "what they wrongly believe (1 sentence)",
      "diagnostic_meaning": "what their pick reveals about their knowledge (1 sentence)"
    }},
    ...one entry per distractor (skip the correct option)...
  ],
  "discrimination_prediction_level": "strong|moderate|weak",
  "discrimination_prediction_rationale": "why this level — be honest"
}}"""


TIER_NAMES = {1: "Remember", 2: "Understand", 3: "Apply", 4: "Evaluate"}


def _opus_cost(usage: dict) -> float:
    return (
        usage.get("input_tokens", 0) / 1e6 * OPUS_INPUT_PRICE_PER_M
        + usage.get("output_tokens", 0) / 1e6 * OPUS_OUTPUT_PRICE_PER_M
    )


def _build_options_block(options: list[dict]) -> str:
    lines = []
    for o in options:
        marker = "[CORRECT]" if o.get("is_correct") else "[distractor]"
        lines.append(f"  {o.get('letter','?')} {marker}: {o.get('text','')}")
    return "\n".join(lines)


async def infer_one(client, question: dict, semaphore) -> dict:
    """Infer a measurement_target for one question. Returns
      {question_id, target (or None), valid (bool), reason, usage, errors}.
    """
    qid = question.get("question_id", "?")
    domain = question.get("domain_code") or question.get("_domain") or "?"
    anchor_uids = question.get("anchor_uids") or []
    anchor_uid = anchor_uids[0] if anchor_uids else "?"
    tier = question.get("difficulty_tier") or 1
    tier_name = TIER_NAMES.get(tier, "Unknown")
    stem = question.get("question_stem", "") or ""
    options = question.get("options") or []
    options_block = _build_options_block(options)

    canonical_json = json.dumps(to_dict(CANONICAL_EXAMPLE), indent=2)
    prompt = INFER_PROMPT.format(
        canonical_json=canonical_json,
        domain=domain,
        tier=tier,
        tier_name=tier_name,
        anchor_uid=anchor_uid,
        question_id=qid,
        stem=stem,
        options_block=options_block,
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
            return {"question_id": qid, "target": None, "valid": False,
                    "reason": f"api_error: {e}", "usage": {"input_tokens": 0, "output_tokens": 0}}

    parsed = parse_response(text)
    if not parsed:
        return {"question_id": qid, "target": None, "valid": False,
                "reason": "parse_failed", "usage": usage}

    valid, reason = is_valid(parsed)
    return {"question_id": qid, "target": parsed if valid else None,
            "raw_target": parsed, "valid": valid, "reason": reason,
            "usage": usage}


async def process_chapter(client, chapter_path: pathlib.Path,
                          semaphore: asyncio.Semaphore) -> dict:
    """Process all questions in one chapter file."""
    with open(chapter_path, encoding="utf-8") as fh:
        questions = json.load(fh)
    if not isinstance(questions, list):
        return {"path": str(chapter_path), "error": "not a question list"}

    print(f"  Processing {chapter_path.name}: {len(questions)} questions")
    results = await asyncio.gather(*[
        infer_one(client, q, semaphore) for q in questions
    ])
    return {
        "path": str(chapter_path),
        "n_questions": len(questions),
        "n_valid": sum(1 for r in results if r["valid"]),
        "n_invalid": sum(1 for r in results if not r["valid"]),
        "results": results,
    }


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=None,
                        help="single chapter JSON path; or omit to use --sample")
    parser.add_argument("--sample", type=int, default=0,
                        help="random-sample N questions across all domains")
    parser.add_argument("--seed", type=int, default=8421)
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    api_key = load_api_key()
    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(4)

    if args.path:
        # Single chapter mode
        chapter_path = pathlib.Path(args.path).resolve()
        if not chapter_path.exists():
            print(f"ERROR: {chapter_path} not found")
            sys.exit(1)
        chapter_result = await process_chapter(client, chapter_path, semaphore)
        out = chapter_result
    else:
        # Sample mode
        if args.sample <= 0:
            print("ERROR: must supply --sample N or a chapter path")
            sys.exit(1)
        rng = random.Random(args.seed)
        all_questions = []
        for domain_dir in sorted(QUIZ_DIR.iterdir()):
            if not domain_dir.is_dir():
                continue
            for ch in domain_dir.glob("*.json"):
                try:
                    with open(ch, encoding="utf-8") as fh:
                        qs = json.load(fh)
                    if not isinstance(qs, list):
                        continue
                    for q in qs:
                        q_copy = dict(q)
                        q_copy["_chapter_path"] = str(ch)
                        all_questions.append(q_copy)
                except (json.JSONDecodeError, OSError):
                    continue
        rng.shuffle(all_questions)
        sampled = all_questions[:args.sample]
        print(f"Sampled {len(sampled)} of {len(all_questions)} questions")

        print("Running inference (Opus 4.7)...")
        results = await asyncio.gather(*[
            infer_one(client, q, semaphore) for q in sampled
        ])
        out = {
            "n_sampled": len(sampled),
            "n_valid": sum(1 for r in results if r["valid"]),
            "n_invalid": sum(1 for r in results if not r["valid"]),
            "results": results,
        }

    total_cost = sum(_opus_cost(r.get("usage", {})) for r in out.get("results", []))
    out["total_cost_usd"] = round(total_cost, 4)
    out["timestamp"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    timestamp_safe = out["timestamp"].replace(":", "-")
    artifact_path = OUT_DIR / f"inference_{timestamp_safe}.json"
    with open(artifact_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    stable_path = OUT_DIR / "inference_latest.json"
    with open(stable_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)

    print()
    print(f"Total cost: ${total_cost:.4f}")
    print(f"Valid targets:   {out['n_valid']}/{out.get('n_sampled') or out.get('n_questions')}")
    print(f"Invalid targets: {out['n_invalid']}")
    print(f"Artifact: {artifact_path}")
    if out['n_invalid']:
        print()
        print("Invalid-target reasons:")
        for r in out["results"]:
            if not r["valid"]:
                print(f"  {r['question_id']}: {r['reason']}")


if __name__ == "__main__":
    asyncio.run(main())
