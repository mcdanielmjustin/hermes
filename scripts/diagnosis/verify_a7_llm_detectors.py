"""Phase A7 verification — LLM-backed detectors on 10 E2E-100 questions.

Runs llm_ambiguity + llm_fact_check on a 10-question subset
(prioritizing T4 to exercise both detectors). Reports:
  - Detector fire counts
  - Correlation with the existing dq audit's ambiguity score
  - Per-question and total cost

Cost estimate: ~$3-5 (Sonnet ambiguity ~$0.01/q × 10 = $0.10;
Opus fact-check ~$0.05/q × T4-count = up to $0.30; plus prompt overhead).

Usage:
    cd C:/Users/mcdan/goliath
    python scripts/diagnosis/verify_a7_llm_detectors.py
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
from pipeline.detectors import PHASE_AUDIT_LLM
from pipeline.detectors.registry import create_detector_registry

OPUS_INPUT_PRICE_PER_M = 15.0
OPUS_OUTPUT_PRICE_PER_M = 75.0

E2E_DIR = REPO_ROOT / "data" / ".diagnosis" / "measurement_targets"


def _sonnet_cost(usage: dict) -> float:
    if not usage:
        return 0.0
    return (
        usage.get("input_tokens", 0) / 1e6 * INPUT_PRICE_PER_M
        + usage.get("output_tokens", 0) / 1e6 * OUTPUT_PRICE_PER_M
    )


def _opus_cost(usage: dict) -> float:
    if not usage:
        return 0.0
    return (
        usage.get("input_tokens", 0) / 1e6 * OPUS_INPUT_PRICE_PER_M
        + usage.get("output_tokens", 0) / 1e6 * OPUS_OUTPUT_PRICE_PER_M
    )


def _reconstruct_question(r: dict) -> dict:
    options = []
    for o in r.get("generated_options", []):
        options.append({
            "letter": o.get("letter"),
            "text": o.get("text", "") or "",
            "is_correct": bool(o.get("is_correct")),
        })
    return {
        "question_id": f"A7-VERIFY-{r.get('source_qid', '?')}",
        "domain_code": r.get("domain"),
        "difficulty_tier": r.get("tier"),
        "question_stem": r.get("generated_stem_preview", "") or "",
        "options": options,
    }


def _pick_subset(results: list[dict], n: int = 10) -> list[dict]:
    """Pick n questions weighted toward T4 (so fact-check has cases to
    exercise) and toward dq_major/minor (where ambiguity is most likely)."""
    valid = [r for r in results if not r.get("phase_failed")]
    # Prioritize T4 dq_major, then T4 dq_minor, then any T4, then T3.
    t4_major = [r for r in valid if r.get("tier") == 4 and r.get("diagnostic_quality_class") == "major"]
    t4_minor = [r for r in valid if r.get("tier") == 4 and r.get("diagnostic_quality_class") == "minor"]
    t4_clean = [r for r in valid if r.get("tier") == 4 and r.get("diagnostic_quality_class") == "clean"]
    t3 = [r for r in valid if r.get("tier") == 3]

    out: list[dict] = []
    for source in (t4_major, t4_minor, t4_clean, t3):
        for r in source:
            if len(out) >= n:
                break
            if r in out:
                continue
            out.append(r)
        if len(out) >= n:
            break
    return out[:n]


async def main():
    files = sorted(glob.glob(str(E2E_DIR / "e2e_100_2026-*.json")))
    if not files:
        print(f"ERROR: no e2e_100 artifact found")
        sys.exit(1)
    artifact = pathlib.Path(files[-1])
    print(f"Loading: {artifact.name}")
    e2e = json.loads(artifact.read_text(encoding="utf-8"))
    sample = _pick_subset(e2e.get("results", []), n=10)
    print(f"Selected {len(sample)} questions for A7 verification")
    tier_dist = {}
    for r in sample:
        tier_dist[r.get("tier")] = tier_dist.get(r.get("tier"), 0) + 1
    print(f"Tier distribution: {dict(sorted(tier_dist.items()))}")
    print()

    api_key = load_api_key()
    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(3)

    registry = create_detector_registry()

    print("Running A7 LLM-backed detectors on 10 questions...")
    per_question: list[dict] = []
    for r in sample:
        q = _reconstruct_question(r)
        signals = await registry.scan_for_phase_async(
            PHASE_AUDIT_LLM, q, context={"client": client, "semaphore": semaphore},
        )
        # Aggregate per detector
        ambiguity_fired = []
        fact_check_fired = []
        ambiguity_cost = 0.0
        fact_check_cost = 0.0
        for s in signals:
            if s.detector_id == "llm_ambiguity":
                usage = (s.extra or {}).get("usage")
                if usage:
                    ambiguity_cost = _sonnet_cost(usage)
                if s.fired:
                    ambiguity_fired.append(s)
            elif s.detector_id == "llm_fact_check":
                usage = (s.extra or {}).get("usage")
                if usage:
                    fact_check_cost = _opus_cost(usage)
                if s.fired:
                    fact_check_fired.append(s)
        per_question.append({
            "source_qid": r.get("source_qid"),
            "tier": r.get("tier"),
            "audit_dq_class": r.get("diagnostic_quality_class"),
            "audit_dq_ambiguity_score": r.get("diagnostic_quality_scores", {}).get("ambiguity"),
            "audit_dq_factual_score": r.get("diagnostic_quality_scores", {}).get("factual_correctness"),
            "llm_ambiguity_fires": len(ambiguity_fired),
            "llm_ambiguity_letters": [s.letter for s in ambiguity_fired],
            "llm_fact_check_fires": len(fact_check_fired),
            "llm_fact_check_letters": [s.letter for s in fact_check_fired],
            "ambiguity_cost": round(ambiguity_cost, 4),
            "fact_check_cost": round(fact_check_cost, 4),
        })
        print(f"  {r.get('source_qid')} (T{r.get('tier')}) "
              f"audit_amb={r.get('diagnostic_quality_scores', {}).get('ambiguity')} "
              f"llm_amb_fires={len(ambiguity_fired)} "
              f"llm_fact_fires={len(fact_check_fired)}")

    total_amb_fires = sum(p["llm_ambiguity_fires"] for p in per_question)
    total_fact_fires = sum(p["llm_fact_check_fires"] for p in per_question)
    total_cost = sum(p["ambiguity_cost"] + p["fact_check_cost"] for p in per_question)

    # Correlation analysis: where audit dq_ambiguity score < 5, do
    # llm_ambiguity fires correlate?
    low_amb_questions = [p for p in per_question if (p["audit_dq_ambiguity_score"] or 5) < 5]
    high_amb_questions = [p for p in per_question if (p["audit_dq_ambiguity_score"] or 5) == 5]
    low_amb_with_fires = sum(1 for p in low_amb_questions if p["llm_ambiguity_fires"] > 0)
    high_amb_with_fires = sum(1 for p in high_amb_questions if p["llm_ambiguity_fires"] > 0)

    print()
    print("=" * 70)
    print("Phase A7 — LLM-backed detector verification")
    print("=" * 70)
    print(f"Subset: {len(per_question)} questions from E2E-100")
    print()
    print(f"Total fires:")
    print(f"  llm_ambiguity:  {total_amb_fires}")
    print(f"  llm_fact_check: {total_fact_fires} (T4 only)")
    print()
    print(f"Correlation with audit dq ambiguity score:")
    print(f"  Audit ambiguity < 5 (flagged): {len(low_amb_questions)} questions")
    print(f"    of which llm_ambiguity fired: {low_amb_with_fires}")
    print(f"  Audit ambiguity = 5 (clean):  {len(high_amb_questions)} questions")
    print(f"    of which llm_ambiguity fired: {high_amb_with_fires}")
    print()
    print(f"Total API cost: ${total_cost:.4f}")

    # Save artifact
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    timestamp_safe = timestamp.replace(":", "-")
    out_path = E2E_DIR / f"a7_verification_{timestamp_safe}.json"
    out_path.write_text(json.dumps({
        "timestamp": timestamp,
        "source_artifact": artifact.name,
        "n_questions": len(per_question),
        "total_ambiguity_fires": total_amb_fires,
        "total_fact_check_fires": total_fact_fires,
        "low_amb_audit_count": len(low_amb_questions),
        "low_amb_with_llm_fires": low_amb_with_fires,
        "high_amb_audit_count": len(high_amb_questions),
        "high_amb_with_llm_fires": high_amb_with_fires,
        "total_cost": round(total_cost, 4),
        "per_question": per_question,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print(f"Artifact: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
