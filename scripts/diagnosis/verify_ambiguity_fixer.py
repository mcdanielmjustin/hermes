"""Phase A7+ verification — ambiguity fixer on the questions A7 flagged.

Runs the full chain on T4 questions:
  1. Re-run llm_ambiguity (Sonnet) to surface defensible alternatives
  2. For each fired signal: dispatch to AmbiguityFixer
  3. Re-audit dq (existing diagnostic_quality audit)
  4. Compare dq_ambiguity score pre vs post

Picks ~6 T4 questions from E2E-100 where the audit's dq_ambiguity
score < 5 (i.e. flagged ambiguous). These are the cases where
llm_ambiguity is most likely to fire and the fixer should produce a
measurable lift.

Cost estimate: ~$2 (Sonnet ambiguity x 6 + Sonnet rewrite x ~6 +
Sonnet dq audit x 6).

Usage:
    cd C:/Users/mcdan/goliath
    python scripts/diagnosis/verify_ambiguity_fixer.py
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
from audit_diagnostic_quality import (  # noqa: E402
    audit_diagnostic_quality_question,
)
from pipeline.detectors import PHASE_AUDIT_LLM, VERDICT_ADVISORY
from pipeline.detectors.registry import create_detector_registry
from pipeline.fixers import create_fixer_registry

E2E_DIR = REPO_ROOT / "data" / ".diagnosis" / "measurement_targets"


def _sonnet_cost(usage: dict) -> float:
    if not usage:
        return 0.0
    return (
        usage.get("input_tokens", 0) / 1e6 * INPUT_PRICE_PER_M
        + usage.get("output_tokens", 0) / 1e6 * OUTPUT_PRICE_PER_M
    )


def _reconstruct_question(r: dict) -> dict:
    options = []
    for o in r.get("generated_options", []):
        options.append({
            "letter": o.get("letter"),
            "text": o.get("text", "") or "",
            "is_correct": bool(o.get("is_correct")),
            "explanation": "",
        })
    return {
        "question_id": f"AMB-VERIFY-{r.get('source_qid', '?')}",
        "domain_code": r.get("domain"),
        "difficulty_tier": r.get("tier"),
        "question_stem": r.get("generated_stem_preview", "") or "",
        "options": options,
        "testable_fact": r.get("target_competency", "") or "",
    }


def _pick_subset(results: list[dict], n: int = 6) -> list[dict]:
    """Pick T4 questions where audit's dq_ambiguity is < 5 (flagged)."""
    out = []
    for r in results:
        if r.get("phase_failed"):
            continue
        if r.get("tier") != 4:
            continue
        amb = (r.get("diagnostic_quality_scores") or {}).get("ambiguity") or 5
        if amb >= 5:
            continue
        out.append(r)
        if len(out) >= n:
            break
    return out


async def _run_one(client, question: dict, semaphore,
                   detector_registry, fixer_registry) -> dict:
    record: dict = {
        "question_id": question["question_id"],
        "tier": question.get("difficulty_tier"),
        "domain": question.get("domain_code"),
    }

    # Step 1: pre-fix dq audit (baseline ambiguity score on the
    # reconstructed question — note this re-runs the audit so it may
    # differ slightly from the e2e_100 stored score due to LLM jitter).
    pre_dq = await audit_diagnostic_quality_question(client, question, semaphore)
    record["pre_dq_class"] = pre_dq.get("diagnostic_quality_class")
    record["pre_dq_scores"] = pre_dq.get("scores", {})
    pre_dq_cost = _sonnet_cost(pre_dq.get("usage", {}))

    # Step 2: run llm_ambiguity detector.
    ambiguity_signals = await detector_registry.scan_for_phase_async(
        PHASE_AUDIT_LLM, question,
        context={"client": client, "semaphore": semaphore},
    )
    fired_amb = [
        s for s in ambiguity_signals
        if s.detector_id == "llm_ambiguity" and s.fired
    ]
    record["llm_ambiguity_fires"] = len(fired_amb)
    record["llm_ambiguity_letters"] = [s.letter for s in fired_amb]
    ambiguity_cost = sum(
        _sonnet_cost((s.extra or {}).get("usage") or {})
        for s in ambiguity_signals
        if s.detector_id == "llm_ambiguity"
    )

    # Step 3: dispatch to AmbiguityFixer for each fired signal.
    patched = question
    fixers_applied: list[str] = []
    fix_cost = 0.0
    for sig in fired_amb:
        fixer = fixer_registry.fixer_for_signature(sig.signature)
        if fixer is None:
            continue
        patched_before = patched
        patched = await fixer.fix(client, patched, sig, semaphore)
        if patched is not patched_before and patched != patched_before:
            fixers_applied.append(f"{fixer.fixer_id}:{sig.letter}")
        # Note: fixer doesn't expose its own cost; estimate via single
        # Sonnet call if a rewrite happened.
        if any(o.get("_routed_fixer", "").startswith("ambiguity:")
               for o in patched.get("options") or []):
            fix_cost += 0.005  # rough estimate; production should track usage

    record["fixers_applied"] = fixers_applied

    # Step 4: post-fix dq audit.
    post_dq = await audit_diagnostic_quality_question(client, patched, semaphore)
    record["post_dq_class"] = post_dq.get("diagnostic_quality_class")
    record["post_dq_scores"] = post_dq.get("scores", {})
    post_dq_cost = _sonnet_cost(post_dq.get("usage", {}))

    record["total_cost"] = pre_dq_cost + ambiguity_cost + post_dq_cost + fix_cost
    return record


async def main():
    files = sorted(glob.glob(str(E2E_DIR / "e2e_100_2026-*.json")))
    if not files:
        print("ERROR: no e2e_100 artifact found")
        sys.exit(1)
    artifact = pathlib.Path(files[-1])
    print(f"Loading: {artifact.name}")
    e2e = json.loads(artifact.read_text(encoding="utf-8"))
    sample = _pick_subset(e2e.get("results", []), n=6)
    print(f"Selected {len(sample)} T4 questions with dq_ambiguity < 5")
    print()

    api_key = load_api_key()
    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(3)

    detector_registry = create_detector_registry()
    fixer_registry = create_fixer_registry()

    questions = [_reconstruct_question(r) for r in sample]

    print("Running ambiguity fixer chain...")
    results = await asyncio.gather(*[
        _run_one(client, q, semaphore, detector_registry, fixer_registry)
        for q in questions
    ])

    # Aggregate
    pre_amb_sum = sum((r["pre_dq_scores"].get("ambiguity") or 0) for r in results)
    post_amb_sum = sum((r["post_dq_scores"].get("ambiguity") or 0) for r in results)
    pre_class_dist = {}
    post_class_dist = {}
    for r in results:
        pre_class_dist[r["pre_dq_class"]] = pre_class_dist.get(r["pre_dq_class"], 0) + 1
        post_class_dist[r["post_dq_class"]] = post_class_dist.get(r["post_dq_class"], 0) + 1
    fixers_total = sum(len(r["fixers_applied"]) for r in results)
    total_cost = sum(r["total_cost"] for r in results)

    print()
    print("=" * 70)
    print("Ambiguity fixer verification")
    print("=" * 70)
    print(f"Subset: {len(results)} T4 questions where audit flagged ambiguity")
    print()
    print(f"Mean dq_ambiguity score: {pre_amb_sum / len(results):.2f} -> {post_amb_sum / len(results):.2f}")
    print(f"  delta: {(post_amb_sum - pre_amb_sum) / len(results):+.2f}")
    print()
    print(f"dq_class distribution:")
    print(f"  pre:  {pre_class_dist}")
    print(f"  post: {post_class_dist}")
    print()
    print(f"Total fixers applied: {fixers_total}")
    print(f"Total API cost: ${total_cost:.4f}")
    print()

    # Per-question detail
    print("Per question:")
    for r in results:
        print(f"  {r['question_id']} ({r['tier']}) "
              f"pre_amb={r['pre_dq_scores'].get('ambiguity')} "
              f"-> post_amb={r['post_dq_scores'].get('ambiguity')}  "
              f"class {r['pre_dq_class']}->{r['post_dq_class']}  "
              f"fixers={len(r['fixers_applied'])}")

    # Save artifact
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    timestamp_safe = timestamp.replace(":", "-")
    out_path = E2E_DIR / f"ambiguity_fixer_verification_{timestamp_safe}.json"
    out_path.write_text(json.dumps({
        "timestamp": timestamp,
        "source": artifact.name,
        "subset_size": len(results),
        "pre_amb_mean": pre_amb_sum / len(results),
        "post_amb_mean": post_amb_sum / len(results),
        "pre_class_dist": pre_class_dist,
        "post_class_dist": post_class_dist,
        "total_fixers_applied": fixers_total,
        "total_cost": round(total_cost, 4),
        "per_question": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print(f"Artifact: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
