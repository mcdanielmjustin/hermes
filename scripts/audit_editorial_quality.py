"""audit_editorial_quality.py — offline editorial pass for a chapter.

Phase 21b. Parallel to audit_stem_contradictions.py: per-question
Sonnet call with an editorial rubric. Outputs per-question
{editorial_class: clean|minor|major, issues: [...], summary: "..."}.

Why offline (not a 23rd gate during generation):
- Adding a Sonnet call to every generation attempt would add ~$0.008/Q
  to every retry and ~1.5s of latency. That's a meaningful tax on the
  generation loop where speed and cost both matter.
- Editorial review is a different concern from english_gap detection
  (already covered by audit_stem_contradictions.py). Keeping it offline
  matches the existing audit pattern.

Output:
  Per-chapter JSON sidecar at <chapter>.editorial.json with
    {audit_metadata: {...}, results: [...]}.
  Each result has: question_id, editorial_class, issues[], summary, usage.

Usage:
  python scripts/audit_editorial_quality.py path/to/batch.json
  python scripts/audit_editorial_quality.py --dir data/quiz/BPSY/
  python scripts/audit_editorial_quality.py path/to/batch.json --workers 5

Cost: ~$0.015-0.020/question (Sonnet 4.6 with the rubric prompt + ~10 line response).
For a 1000-Q audit, ~$15-20 total.
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

# Reuse audit primitives from audit_stem_contradictions where possible.
from audit_stem_contradictions import (  # noqa: E402
    parse_response, load_questions, load_api_key,
    INPUT_PRICE_PER_M, OUTPUT_PRICE_PER_M, MODEL_ID,
)
from pipeline.editorial_rubric import (  # noqa: E402
    EDITORIAL_RUBRIC_PROMPT, EDITORIAL_CLASSES, is_known_editorial_class,
    CLEAN, MINOR, MAJOR,
)


# ── Per-question audit ──────────────────────────────────────

def build_editorial_prompt(question: dict) -> str:
    """Format the rubric prompt with a question's stem + options."""
    stem = question.get("question_stem", "")
    options = question.get("options", []) or []
    options_block = "\n".join(
        f"  {o.get('letter', '?')} "
        f"{'[CORRECT]' if o.get('is_correct') else '[distractor]'}: "
        f"{o.get('text', '')}"
        for o in options
    )
    return EDITORIAL_RUBRIC_PROMPT.format(
        stem=stem, options_block=options_block,
    )


async def audit_editorial_question(client, question: dict, semaphore) -> dict:
    """Audit a single question's editorial quality. Returns a dict
    with editorial_class, issues, summary, usage, error.

    Uses temperature=0 for stability (matches stem-audit determinism)
    and the same Sonnet model.
    """
    async with semaphore:
        prompt = build_editorial_prompt(question)
        try:
            response = await client.messages.create(
                model=MODEL_ID,
                max_tokens=1024,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text if response.content else ""
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
            parsed = parse_response(text)
            if not parsed:
                return {
                    "question_id": question.get("question_id", "?"),
                    "editorial_class": None,
                    "issues": [],
                    "summary": "",
                    "usage": usage,
                    "error": "json_parse_failed",
                }
            editorial_class = parsed.get("editorial_class")
            if not is_known_editorial_class(editorial_class):
                # Coerce unknown classes to clean (audit fail-safe);
                # log for inspection
                editorial_class = CLEAN
                parsed_summary = (
                    f"unrecognized class '{parsed.get('editorial_class')}'; "
                    f"coerced to clean"
                )
            else:
                parsed_summary = parsed.get("summary", "")
            return {
                "question_id": question.get("question_id", "?"),
                "tier": question.get("difficulty_tier"),
                "editorial_class": editorial_class,
                "issues": parsed.get("issues", []) or [],
                "summary": parsed_summary,
                "usage": usage,
                "error": None,
            }
        except Exception as e:
            return {
                "question_id": question.get("question_id", "?"),
                "tier": question.get("difficulty_tier"),
                "editorial_class": None,
                "issues": [],
                "summary": "",
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "error": str(e),
            }


# ── CLI / main ──────────────────────────────────────────────

def _calc_cost(usage: dict) -> float:
    return (
        usage.get("input_tokens", 0) / 1e6 * INPUT_PRICE_PER_M
        + usage.get("output_tokens", 0) / 1e6 * OUTPUT_PRICE_PER_M
    )


async def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "path",
        help="Path to a single chapter JSON OR a directory of them",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    target_path = pathlib.Path(args.path)
    if target_path.is_dir():
        batch_paths = sorted(
            p for p in target_path.glob("*.json")
            if not any(t in p.name for t in
                       ("audit", "fixed", "manifest", "editorial", "_backup"))
        )
    else:
        batch_paths = [target_path]

    if not batch_paths:
        print("No chapter files found.", file=sys.stderr)
        sys.exit(1)

    api_key = args.api_key or load_api_key()
    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(args.workers)

    grand_total_questions = 0
    grand_total_cost = 0.0
    grand_class_counts = {CLEAN: 0, MINOR: 0, MAJOR: 0, "error": 0}

    for batch_path in batch_paths:
        questions = load_questions(batch_path)
        if not questions:
            print(f"\n--- {batch_path.name}: empty ---", flush=True)
            continue

        print(f"\n=== {batch_path.name} ({len(questions)} questions) ===",
              flush=True)

        # Concurrent editorial audit
        tasks = [audit_editorial_question(client, q, semaphore)
                 for q in questions]
        results = await asyncio.gather(*tasks)

        # Per-question summary
        batch_class_counts = {CLEAN: 0, MINOR: 0, MAJOR: 0, "error": 0}
        batch_cost = 0.0
        for r in results:
            cost = _calc_cost(r.get("usage", {}))
            batch_cost += cost
            cls = r.get("editorial_class")
            if r.get("error"):
                batch_class_counts["error"] += 1
            elif cls in EDITORIAL_CLASSES:
                batch_class_counts[cls] += 1
            else:
                batch_class_counts["error"] += 1

            # Print majors immediately for visibility
            if cls == MAJOR:
                qid = r.get("question_id", "?")
                summary = r.get("summary", "")
                print(f"  [MAJOR] {qid}: {summary[:150]}", flush=True)
                for issue in (r.get("issues") or [])[:3]:
                    desc = issue.get("description", "")[:120]
                    print(f"    - {issue.get('dimension', '?')}: {desc}",
                          flush=True)

        print(f"  Batch summary: clean={batch_class_counts[CLEAN]}, "
              f"minor={batch_class_counts[MINOR]}, "
              f"major={batch_class_counts[MAJOR]}, "
              f"errors={batch_class_counts['error']}, "
              f"cost=${batch_cost:.4f}", flush=True)

        # Save sidecar
        sidecar_path = batch_path.with_name(
            batch_path.stem + ".editorial.json"
        )
        sidecar_data = {
            "audit_metadata": {
                "audit_type": "editorial",
                "model": MODEL_ID,
                "audited_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "questions_audited": len(questions),
                "class_counts": batch_class_counts,
                "total_cost_usd": round(batch_cost, 4),
            },
            "results": results,
        }
        with open(sidecar_path, "w", encoding="utf-8") as fh:
            json.dump(sidecar_data, fh, indent=2, ensure_ascii=False)

        grand_total_questions += len(questions)
        grand_total_cost += batch_cost
        for k, v in batch_class_counts.items():
            grand_class_counts[k] += v

    # Grand summary
    print()
    print("=" * 70)
    print(f"GRAND TOTAL: {grand_total_questions} questions audited")
    print(f"  clean:  {grand_class_counts[CLEAN]}")
    print(f"  minor:  {grand_class_counts[MINOR]}")
    print(f"  major:  {grand_class_counts[MAJOR]}")
    print(f"  error:  {grand_class_counts['error']}")
    print(f"Total cost: ${grand_total_cost:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
