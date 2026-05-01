"""Phase 27-29 — Offline diagnostic-quality audit.

Multi-criterion audit covering factual_correctness, ambiguity, and
tier_fit in a single Sonnet 4.6 call per question. Productionized
version of the path-B pilot's three highest-leverage criteria.

Sibling to:
  - scripts/audit_stem_contradictions.py (english_gap audit)
  - scripts/audit_editorial_quality.py (editorial style audit)
This audit covers three orthogonal dimensions the others miss.

Usage:
  python scripts/audit_diagnostic_quality.py path/to/chapter.json
  python scripts/audit_diagnostic_quality.py --dir data/quiz/

Output:
  Per-question class (clean/minor/major) printed
  Sidecar at <chapter>.diagnostic_quality.json
  Summary table at end (questions audited, class counts, cost)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
from datetime import datetime, timezone

import anthropic

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline.diagnostic_quality_rubric import (  # noqa: E402
    DIAGNOSTIC_QUALITY_RUBRIC_PROMPT,
    is_known_diagnostic_quality_class,
    CLEAN, MINOR, MAJOR,
)
from audit_stem_contradictions import (  # noqa: E402
    parse_response, load_api_key, INPUT_PRICE_PER_M, OUTPUT_PRICE_PER_M,
)

MODEL_ID = "claude-sonnet-4-6"

TIER_NAMES = {1: "Remember", 2: "Understand", 3: "Apply", 4: "Evaluate"}


def _calc_cost(usage: dict) -> float:
    return (
        usage.get("input_tokens", 0) / 1e6 * INPUT_PRICE_PER_M
        + usage.get("output_tokens", 0) / 1e6 * OUTPUT_PRICE_PER_M
    )


def _build_options_block(options: list[dict]) -> str:
    lines = []
    for o in options:
        marker = "[CORRECT]" if o.get("is_correct") else "[distractor]"
        lines.append(f"  {o.get('letter','?')} {marker}: {o.get('text','')}")
    return "\n".join(lines)


async def audit_diagnostic_quality_question(
    client, question: dict, semaphore,
) -> dict:
    """Run the diagnostic-quality audit on one question. Returns the
    parsed verdict + usage."""
    qid = question.get("question_id", "?")
    domain = question.get("domain_code") or "?"
    tier = question.get("difficulty_tier") or 1
    tier_name = TIER_NAMES.get(tier, "Unknown")
    testable_fact = question.get("testable_fact") or "(not specified)"
    stem = question.get("question_stem", "") or ""
    options = question.get("options") or []
    options_block = _build_options_block(options)

    prompt = DIAGNOSTIC_QUALITY_RUBRIC_PROMPT.format(
        domain=domain,
        tier=tier,
        tier_name=tier_name,
        testable_fact=testable_fact,
        stem=stem,
        options_block=options_block,
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
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        except Exception as e:
            return {"question_id": qid, "error": f"api_error: {e}",
                    "diagnostic_quality_class": None,
                    "usage": {"input_tokens": 0, "output_tokens": 0}}

    parsed = parse_response(text)
    if not parsed:
        return {"question_id": qid, "error": "parse_failed",
                "diagnostic_quality_class": None, "usage": usage}

    cls = parsed.get("diagnostic_quality_class")
    if not is_known_diagnostic_quality_class(cls):
        return {"question_id": qid,
                "error": f"unknown_class: {cls}",
                "diagnostic_quality_class": None, "usage": usage}

    return {
        "question_id": qid,
        "diagnostic_quality_class": cls,
        "scores": parsed.get("scores") or {},
        "rationales": parsed.get("rationales") or {},
        "issues": parsed.get("issues") or [],
        "summary": parsed.get("summary"),
        "usage": usage,
    }


def _load_chapter(path: pathlib.Path) -> list[dict]:
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(d, list):
        return d
    if isinstance(d, dict) and "questions" in d:
        return d["questions"]
    return []


def _save_sidecar(chapter_path: pathlib.Path, results: list[dict]) -> pathlib.Path:
    sidecar_path = chapter_path.with_name(
        chapter_path.stem + ".diagnostic_quality.json"
    )
    payload = {
        "audit_metadata": {
            "model": MODEL_ID,
            "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "n_questions": len(results),
        },
        "results": results,
    }
    with open(sidecar_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)
    return sidecar_path


async def audit_chapter(client, chapter_path: pathlib.Path,
                        semaphore: asyncio.Semaphore) -> dict:
    questions = _load_chapter(chapter_path)
    if not questions:
        print(f"  [skip] {chapter_path.name}: empty or unreadable")
        return {"path": str(chapter_path), "n_questions": 0}

    print(f"=== {chapter_path.name} ({len(questions)} questions) ===")
    results = await asyncio.gather(*[
        audit_diagnostic_quality_question(client, q, semaphore)
        for q in questions
    ])

    counts = {CLEAN: 0, MINOR: 0, MAJOR: 0, "error": 0}
    cost = 0.0
    major_qids = []
    for r in results:
        cls = r.get("diagnostic_quality_class")
        if r.get("error") or cls is None:
            counts["error"] += 1
        elif cls in counts:
            counts[cls] += 1
        if cls == MAJOR:
            major_qids.append(r.get("question_id"))
        cost += _calc_cost(r.get("usage", {}))

    print(f"  Batch summary: clean={counts[CLEAN]} minor={counts[MINOR]} "
          f"major={counts[MAJOR]} errors={counts['error']}, ${cost:.4f}")
    if major_qids:
        for qid in major_qids:
            r = next(x for x in results if x.get("question_id") == qid)
            issue_str = "; ".join(
                f"{i.get('dimension')}:{i.get('severity')}"
                for i in r.get("issues") or []
            )
            print(f"    [MAJOR] {qid}: {issue_str}")

    sidecar = _save_sidecar(chapter_path, results)
    print(f"  Saved -> {sidecar.name}")
    return {
        "path": str(chapter_path),
        "n_questions": len(questions),
        "counts": counts,
        "cost_usd": round(cost, 4),
        "results": results,
    }


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", default=None,
                        help="single chapter JSON OR directory to scan")
    parser.add_argument("--dir", default=None,
                        help="directory of chapters")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()

    if not args.path and not args.dir:
        print("ERROR: supply a path or --dir")
        sys.exit(1)

    api_key = args.api_key or load_api_key()
    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(args.workers)

    chapter_paths: list[pathlib.Path] = []
    target = pathlib.Path(args.path or args.dir).resolve()
    if target.is_file():
        chapter_paths = [target]
    elif target.is_dir():
        chapter_paths = sorted(
            p for p in target.rglob("*.json")
            if not any(t in p.name for t in
                       ("audit", "fixed", "manifest", "diagnostic_quality", "editorial"))
        )

    if not chapter_paths:
        print(f"No chapter files found at {target}")
        sys.exit(1)

    print(f"Auditing {len(chapter_paths)} chapter(s)...")
    print()

    chapter_results = []
    for ch_path in chapter_paths:
        result = await audit_chapter(client, ch_path, semaphore)
        chapter_results.append(result)

    # Grand total
    total_clean = sum(r.get("counts", {}).get(CLEAN, 0) for r in chapter_results)
    total_minor = sum(r.get("counts", {}).get(MINOR, 0) for r in chapter_results)
    total_major = sum(r.get("counts", {}).get(MAJOR, 0) for r in chapter_results)
    total_error = sum(r.get("counts", {}).get("error", 0) for r in chapter_results)
    total_cost = sum(r.get("cost_usd", 0.0) for r in chapter_results)
    total_questions = sum(r.get("n_questions", 0) for r in chapter_results)

    print()
    print("=" * 70)
    print(f"GRAND TOTAL: {total_questions} questions audited")
    print(f"  clean: {total_clean}")
    print(f"  minor: {total_minor}")
    print(f"  major: {total_major}")
    print(f"  errors: {total_error}")
    print(f"Total cost: ${total_cost:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
