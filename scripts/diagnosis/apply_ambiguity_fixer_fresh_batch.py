"""Apply the AmbiguityFixer to the dq_major questions from the fresh
batch audit. Direct test: does the ambiguity_fixer move dq_major
numbers on fresh-generation content (where it matters most)?

Pipeline:
  1. Load fresh batch audit artifact
  2. Filter to questions with dq_class == "major"
  3. Run llm_ambiguity detector to get defensible-alternative argument
  4. Dispatch to AmbiguityFixer
  5. Re-audit dq
  6. Aggregate before/after dq_class distribution

Cost: ~$2-3 (Sonnet ambiguity x 11 + Sonnet rewrite x ~11 + Sonnet
dq audit x 11).
"""
from __future__ import annotations

import asyncio
import glob
import json
import pathlib
import sys
from datetime import datetime, timezone

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, OSError):
    pass

import anthropic

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_stem_contradictions import (  # noqa: E402
    load_api_key, INPUT_PRICE_PER_M, OUTPUT_PRICE_PER_M,
)
from audit_diagnostic_quality import audit_diagnostic_quality_question  # noqa: E402
from pipeline.detectors import PHASE_AUDIT_LLM, VERDICT_OVERRIDE_TO  # noqa: E402
from pipeline.detectors.registry import create_detector_registry  # noqa: E402
from pipeline.fixers import create_fixer_registry  # noqa: E402

E2E_DIR = REPO_ROOT / "data" / ".diagnosis" / "measurement_targets"


def _sonnet_cost(usage: dict) -> float:
    if not usage:
        return 0.0
    return (
        usage.get("input_tokens", 0) / 1e6 * INPUT_PRICE_PER_M
        + usage.get("output_tokens", 0) / 1e6 * OUTPUT_PRICE_PER_M
    )


def _load_target_question(question_id: str) -> dict | None:
    """Find the question in the fresh-batch chapters by question_id."""
    chapters = [
        REPO_ROOT / "data/quiz/BPSY/the-frontal-lobe-executive-function-motor-control-and-prefro.json",
        REPO_ROOT / "data/quiz/CASS/the-therapeutic-perimeter-standard-10-and-the-ethics-of-clin.json",
        REPO_ROOT / "data/quiz/CPAT/wired-differently-from-the-start-adhd-autism-and-neurodevelo.json",
    ]
    for ch in chapters:
        try:
            data = json.loads(ch.read_text(encoding="utf-8"))
        except Exception:
            continue
        for q in data:
            if q.get("question_id") == question_id:
                return q
    return None


async def _process_one(client, q: dict, semaphore,
                       detector_registry, fixer_registry) -> dict:
    rec: dict = {
        "question_id": q.get("question_id"),
        "anchor_uid": (q.get("anchor_uids") or [None])[0],
        "tier": q.get("difficulty_tier"),
        "domain": q.get("domain_code"),
    }

    # Step 1: pre-fix dq audit
    pre_dq = await audit_diagnostic_quality_question(client, q, semaphore)
    rec["pre_dq_class"] = pre_dq.get("diagnostic_quality_class")
    rec["pre_dq_scores"] = pre_dq.get("scores", {})
    pre_dq_cost = _sonnet_cost(pre_dq.get("usage", {}))

    # Step 2: run llm_ambiguity detector to surface defensible-alternative
    # arguments
    ambiguity_signals = await detector_registry.scan_for_phase_async(
        PHASE_AUDIT_LLM, q,
        context={"client": client, "semaphore": semaphore},
    )
    fired_amb = [
        s for s in ambiguity_signals
        if s.detector_id == "llm_ambiguity" and s.fired
    ]
    rec["llm_ambiguity_fires"] = len(fired_amb)
    rec["llm_ambiguity_letters"] = [s.letter for s in fired_amb]
    amb_cost = sum(
        _sonnet_cost((s.extra or {}).get("usage") or {})
        for s in ambiguity_signals
        if s.detector_id == "llm_ambiguity"
    )

    # Step 3: dispatch AmbiguityFixer
    patched = q
    fixers_applied: list[str] = []
    for sig in fired_amb:
        fixer = fixer_registry.fixer_for_signature(sig.signature)
        if fixer is None:
            continue
        patched_before = patched
        patched = await fixer.fix(client, patched, sig, semaphore)
        if patched != patched_before:
            fixers_applied.append(f"{fixer.fixer_id}:{sig.letter}")
    rec["fixers_applied"] = fixers_applied

    # Step 4: post-fix dq audit
    post_dq = await audit_diagnostic_quality_question(client, patched, semaphore)
    rec["post_dq_class"] = post_dq.get("diagnostic_quality_class")
    rec["post_dq_scores"] = post_dq.get("scores", {})
    post_dq_cost = _sonnet_cost(post_dq.get("usage", {}))

    rec["total_cost"] = pre_dq_cost + amb_cost + post_dq_cost
    return rec


async def main():
    # Load latest fresh batch audit artifact
    audits = sorted(E2E_DIR.glob("fresh_batch_audit_*.json"))
    if not audits:
        print("ERROR: no fresh_batch_audit artifact found")
        sys.exit(1)
    audit_path = audits[-1]
    print(f"Loading: {audit_path.name}")
    fresh_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    per_question = fresh_audit.get("per_question") or []

    # Filter to dq_major questions
    major_qids = [
        r["question_id"] for r in per_question
        if r.get("post_dq_class") == "major"
    ]
    print(f"dq_major questions: {len(major_qids)}")

    questions = []
    for qid in major_qids:
        q = _load_target_question(qid)
        if q:
            questions.append(q)
    print(f"Loaded {len(questions)} questions for ambiguity fixing")
    print()

    if not questions:
        print("No questions to process.")
        return

    api_key = load_api_key()
    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(3)

    detector_registry = create_detector_registry()
    fixer_registry = create_fixer_registry()

    print("Running ambiguity fixer chain on dq_major questions...")
    results = await asyncio.gather(*[
        _process_one(client, q, semaphore, detector_registry, fixer_registry)
        for q in questions
    ])

    # Aggregate
    pre_class = {"clean": 0, "minor": 0, "major": 0, None: 0}
    post_class = {"clean": 0, "minor": 0, "major": 0, None: 0}
    for r in results:
        pre_class[r["pre_dq_class"]] = pre_class.get(r["pre_dq_class"], 0) + 1
        post_class[r["post_dq_class"]] = post_class.get(r["post_dq_class"], 0) + 1
    fixers_total = sum(len(r["fixers_applied"]) for r in results)
    total_cost = sum(r["total_cost"] for r in results)

    print()
    print("=" * 70)
    print(f"Ambiguity fixer on fresh-batch dq_major ({len(results)} questions)")
    print("=" * 70)
    print()
    print(f"dq_class:")
    print(f"  pre:  clean={pre_class['clean']:2d}  minor={pre_class['minor']:2d}  "
          f"major={pre_class['major']:2d}  parse_err={pre_class[None]:2d}")
    print(f"  post: clean={post_class['clean']:2d}  minor={post_class['minor']:2d}  "
          f"major={post_class['major']:2d}  parse_err={post_class[None]:2d}")
    print()
    delta_major = post_class['major'] - pre_class['major']
    delta_clean = post_class['clean'] - pre_class['clean']
    print(f"  delta major: {delta_major:+d}")
    print(f"  delta clean: {delta_clean:+d}")
    print()
    print(f"Ambiguity fixers applied: {fixers_total}")
    print(f"Total API cost: ${total_cost:.4f}")
    print()
    print("Per question:")
    for r in results:
        print(f"  {r['question_id']} (T{r['tier']}) "
              f"pre={r['pre_dq_class']} post={r['post_dq_class']} "
              f"fixers={len(r['fixers_applied'])}")

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out_path = E2E_DIR / f"ambiguity_fixer_freshbatch_{timestamp.replace(':', '-')}.json"
    out_path.write_text(json.dumps({
        "timestamp": timestamp,
        "n_questions": len(results),
        "pre_class": pre_class,
        "post_class": post_class,
        "fixers_total": fixers_total,
        "total_cost": round(total_cost, 4),
        "per_question": results,
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print()
    print(f"Artifact: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
