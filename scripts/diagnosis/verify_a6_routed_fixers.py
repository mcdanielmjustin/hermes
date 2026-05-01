"""Phase A6 verification — routed fixers vs self_critique on 10 bad questions.

Demonstrates whether routed fixers eliminate the dq_major regression that
today's fix-and-export run produced (dq_major 8 → 15 because
self_critique introduces new ambiguity while fixing english_gap).

Method:
  1. Load the most recent fix-and-export result from Downloads (the
     run that produced the 8 → 15 regression).
  2. Pick 10 representative bad questions, prioritizing signatured cases
     (universal_quantifier or laterality fires from
     english_gap_scanner).
  3. For each: re-audit english_gap → if a routed fixer matches the
     signature, dispatch to it → re-audit eg + dq → record.
  4. Compare dq_major count pre vs post against the prior fix run.

Cost: ~$1-3 (most fixers are deterministic; only Sonnet calls are the
re-audit passes).

Usage:
    cd C:/Users/mcdan/goliath
    python scripts/diagnosis/verify_a6_routed_fixers.py

Output:
    data/.diagnosis/measurement_targets/a6_routed_fixers_verification_<ts>.json
    Console summary
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
    audit_question, load_api_key,
    INPUT_PRICE_PER_M, OUTPUT_PRICE_PER_M,
)
from audit_diagnostic_quality import (  # noqa: E402
    audit_diagnostic_quality_question,
)
from pipeline.detectors import VERDICT_OVERRIDE_TO, PHASE_AUDIT
from pipeline.detectors.registry import create_detector_registry
from pipeline.fixers import create_fixer_registry

DOWNLOADS_DIR = pathlib.Path.home() / "Downloads"
OUT_DIR = REPO_ROOT / "data" / ".diagnosis" / "measurement_targets"


def _sonnet_cost(usage: dict) -> float:
    return (
        usage.get("input_tokens", 0) / 1e6 * INPUT_PRICE_PER_M
        + usage.get("output_tokens", 0) / 1e6 * OUTPUT_PRICE_PER_M
    )


def _reconstruct_question(rec: dict) -> dict:
    """Build a question dict from a fix-and-export bad record's BEFORE
    state (pre-self_critique). This is the state where routed fixers
    have something to act on; using AFTER would be testing the audit
    on already-fixed questions and routed fixers would never fire."""
    before = rec.get("before") or {}
    options = []
    for o in before.get("options") or []:
        options.append({
            "letter": o.get("letter"),
            "text": o.get("text", "") or "",
            "is_correct": bool(o.get("is_correct")),
            "explanation": o.get("explanation", ""),
        })
    return {
        "question_id": f"A6-VERIFY-{rec.get('source_qid', '?')}",
        "domain_code": rec.get("domain"),
        "difficulty_tier": rec.get("tier"),
        "question_stem": before.get("stem", "") or "",
        "options": options,
        "testable_fact": "(carried from fix-and-export run pre-fix state)",
    }


def _pick_10_subset(records: list[dict]) -> list[dict]:
    """Pick 10 representative bad records, prioritizing signatured cases."""
    # Categorize: had-eg-flag, had-dq-flag, both, parse-failures.
    eg_flagged = [
        r for r in records if r.get("was_bad")
        and r.get("before", {}).get("english_gap_count", 0) > 0
    ]
    dq_majors = [
        r for r in records if r.get("was_bad")
        and r.get("before", {}).get("diagnostic_quality_class") == "major"
    ]
    dq_minors = [
        r for r in records if r.get("was_bad")
        and r.get("before", {}).get("diagnostic_quality_class") == "minor"
    ]
    # Aim for 4 eg-flagged + 4 dq_major + 2 dq_minor (10 total).
    out = list(eg_flagged[:4])
    seen = {r.get("source_qid") for r in out}
    for r in dq_majors:
        if len(out) >= 8:
            break
        if r.get("source_qid") in seen:
            continue
        out.append(r)
        seen.add(r.get("source_qid"))
    for r in dq_minors:
        if len(out) >= 10:
            break
        if r.get("source_qid") in seen:
            continue
        out.append(r)
        seen.add(r.get("source_qid"))
    return out[:10]


async def _run_one(client, question: dict, semaphore,
                   detector_registry, fixer_registry) -> dict:
    """Audit eg → if signature matches, route to fixer → re-audit eg + dq."""
    record: dict = {
        "question_id": question["question_id"],
        "tier": question.get("difficulty_tier"),
        "domain": question.get("domain_code"),
    }

    # Step 1: english_gap audit (n=1, just to get classifications + flagged).
    eg_initial = await audit_question(client, question, semaphore, n_passes=1)
    initial_eg_count = sum(
        1 for c in eg_initial.get("classifications") or []
        if c.get("class") == "english_gap"
    )
    record["before_eg_count"] = initial_eg_count
    record["initial_audit_cost"] = _sonnet_cost(eg_initial.get("usage", {}))

    # Step 2: scan via detector registry to surface signatures.
    detector_signals = detector_registry.scan_for_phase(
        PHASE_AUDIT, question, context={
            "tier": question.get("difficulty_tier"),
            "source_type": question.get("source_type"),
            "stem_pattern": question.get("stem_pattern"),
            "domain_code": question.get("domain_code"),
        },
    )

    # Step 3: route any OVERRIDE_TO signal to a fixer if registered.
    patched = question
    fixers_applied: list[str] = []
    for sig in detector_signals:
        if not sig.fired:
            continue
        if sig.verdict_action != VERDICT_OVERRIDE_TO:
            continue
        fixer = fixer_registry.fixer_for_signature(sig.signature)
        if fixer is None:
            continue
        try:
            patched = await fixer.fix(client, patched, sig, semaphore)
            fixers_applied.append(f"{fixer.fixer_id}:{sig.signature}:{sig.letter}")
        except Exception as e:
            record.setdefault("fixer_errors", []).append(
                f"{fixer.fixer_id}: {type(e).__name__}: {e}"
            )

    record["fixers_applied"] = fixers_applied

    # Step 4: re-audit english_gap (n=3 quorum) and diagnostic_quality.
    eg_final = await audit_question(client, patched, semaphore, n_passes=3)
    final_eg_count = sum(
        1 for c in eg_final.get("classifications") or []
        if c.get("class") == "english_gap"
    )
    record["after_eg_count"] = final_eg_count
    record["final_eg_audit_cost"] = _sonnet_cost(eg_final.get("usage", {}))

    # Inject metadata for dq audit.
    patched_for_dq = dict(patched)
    patched_for_dq["domain_code"] = patched_for_dq.get("domain_code") or question.get("domain_code")
    patched_for_dq["difficulty_tier"] = patched_for_dq.get("difficulty_tier") or question.get("difficulty_tier")

    dq_final = await audit_diagnostic_quality_question(client, patched_for_dq, semaphore)
    record["after_dq_class"] = dq_final.get("diagnostic_quality_class")
    record["after_dq_scores"] = dq_final.get("scores", {})
    record["final_dq_audit_cost"] = _sonnet_cost(dq_final.get("usage", {}))

    record["total_cost"] = (
        record["initial_audit_cost"]
        + record["final_eg_audit_cost"]
        + record["final_dq_audit_cost"]
    )
    return record


async def main():
    # Load latest fix-and-export from Downloads.
    paths = sorted(DOWNLOADS_DIR.glob("goliath-e2e100-fixed-*.json"))
    if not paths:
        print("ERROR: no goliath-e2e100-fixed-*.json found in ~/Downloads")
        sys.exit(1)
    fix_export_path = paths[-1]
    print(f"Loading fix-and-export: {fix_export_path.name}")
    fix_export = json.loads(fix_export_path.read_text(encoding="utf-8"))
    bad_records = [r for r in fix_export.get("questions", []) if r.get("was_bad")]
    print(f"Bad records in source: {len(bad_records)}")

    sample = _pick_10_subset(bad_records)
    print(f"Selected subset: {len(sample)} questions")
    print()

    # Pre-fix dq major count from the source (the 8 → 15 regression context).
    pre_fix_dq_majors = sum(
        1 for r in sample
        if r.get("before", {}).get("diagnostic_quality_class") == "major"
    )
    pre_fix_dq_minors = sum(
        1 for r in sample
        if r.get("before", {}).get("diagnostic_quality_class") == "minor"
    )
    print(f"Subset baseline: {pre_fix_dq_majors} dq_major + {pre_fix_dq_minors} dq_minor pre-fix")
    print()

    api_key = load_api_key()
    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(3)

    detector_registry = create_detector_registry()
    fixer_registry = create_fixer_registry()

    questions = [_reconstruct_question(r) for r in sample]

    print("Running routed fixers + re-audit on 10 questions...")
    results = await asyncio.gather(*[
        _run_one(client, q, semaphore, detector_registry, fixer_registry)
        for q in questions
    ])

    total_cost = sum(r["total_cost"] for r in results)
    post_dq_classes = [r["after_dq_class"] for r in results]
    post_dq_majors = sum(1 for c in post_dq_classes if c == "major")
    post_dq_minors = sum(1 for c in post_dq_classes if c == "minor")
    post_dq_clean = sum(1 for c in post_dq_classes if c == "clean")

    eg_clean_pre = sum(1 for r in results if r["before_eg_count"] == 0)
    eg_clean_post = sum(1 for r in results if r["after_eg_count"] == 0)

    fixers_used: dict[str, int] = {}
    for r in results:
        for f in r.get("fixers_applied") or []:
            fid = f.split(":")[0]
            fixers_used[fid] = fixers_used.get(fid, 0) + 1

    print()
    print("=" * 70)
    print("Phase A6 — routed fixers verification")
    print("=" * 70)
    print(f"Subset: 10 bad questions from {fix_export_path.name}")
    print()
    print(f"Pre-A6 baseline (this subset, from prior fix-and-export run):")
    print(f"  dq_major: {pre_fix_dq_majors}/10")
    print(f"  dq_minor: {pre_fix_dq_minors}/10")
    print()
    print(f"Post-A6 (routed fixers + re-audit):")
    print(f"  dq_major: {post_dq_majors}/10  ", end="")
    if post_dq_majors > pre_fix_dq_majors:
        print(f"REGRESSION (+{post_dq_majors - pre_fix_dq_majors})")
    elif post_dq_majors < pre_fix_dq_majors:
        print(f"IMPROVEMENT (-{pre_fix_dq_majors - post_dq_majors})")
    else:
        print("(equal — regression eliminated vs prior 8 -> 15)")
    print(f"  dq_minor: {post_dq_minors}/10")
    print(f"  dq_clean: {post_dq_clean}/10")
    print()
    print(f"english_gap clean: {eg_clean_pre} -> {eg_clean_post}")
    print()
    print(f"Routed fixers used:")
    for fid, count in fixers_used.items():
        print(f"  {fid}: {count}")
    print()
    print(f"Total API cost: ${total_cost:.4f}")

    # Save artifact
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    timestamp_safe = timestamp.replace(":", "-")
    artifact_path = OUT_DIR / f"a6_routed_fixers_verification_{timestamp_safe}.json"
    artifact_path.write_text(json.dumps({
        "timestamp": timestamp,
        "source": fix_export_path.name,
        "subset_size": len(sample),
        "pre_fix_dq_majors": pre_fix_dq_majors,
        "pre_fix_dq_minors": pre_fix_dq_minors,
        "post_a6_dq_majors": post_dq_majors,
        "post_a6_dq_minors": post_dq_minors,
        "post_a6_dq_clean": post_dq_clean,
        "eg_clean_pre": eg_clean_pre,
        "eg_clean_post": eg_clean_post,
        "fixers_used": fixers_used,
        "total_cost": round(total_cost, 4),
        "per_question_results": results,
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    print()
    print(f"Artifact: {artifact_path}")


if __name__ == "__main__":
    asyncio.run(main())
