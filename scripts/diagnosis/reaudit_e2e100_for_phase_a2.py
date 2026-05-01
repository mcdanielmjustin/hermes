"""Phase A2 verification — measure the english_gap override effect on E2E-100.

A2 promotes the english_gap_scanner from advisory to override on T1/T2
questions when high-confidence signatures (universal_quantifier,
laterality, numeric_ratio) fire. This script measures the override's
effect on the existing E2E-100 fixture WITHOUT re-running the LLM audit
(deterministic, $0 cost).

What "lift" means at this layer
-------------------------------

A2 flips classifications TOWARD english_gap (catches cases the LLM
missed). On the same corpus, this will *increase* the count of english_gap
distractors visible at audit. The plan's "91% → 94% english_gap clean"
trajectory is post-A2 + fix: A2 surfaces more flags; fix rewrites the
flagged distractors; re-audit shows fewer flags.

This script reports the *catch count* — how many T1/T2 cases A2 would
override. That's the actionable input for the fix pipeline.

Output
------
- Console summary (per-tier override counts, projected new flag rate)
- `data/.diagnosis/measurement_targets/a2_override_simulation_<timestamp>.json`

Usage
-----
    cd C:/Users/mcdan/goliath
    python scripts/diagnosis/reaudit_e2e100_for_phase_a2.py
"""
from __future__ import annotations

import glob
import json
import pathlib
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, OSError):
    pass

# Reuse repo's sys.path convention (scripts/ provides shared_constants).
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pipeline.detectors.english_gap import (  # noqa: E402
    EnglishGapDetector,
    OVERRIDE_ELIGIBLE_SIGNATURES,
    OVERRIDE_ELIGIBLE_TIERS,
)
from pipeline.detectors import VERDICT_OVERRIDE_TO  # noqa: E402

E2E_DIR = REPO_ROOT / "data" / ".diagnosis" / "measurement_targets"


def _reconstruct_question(r: dict) -> dict:
    """Same shape used in fix_and_export_e2e_100.py — the artifact stores
    truncated stems (240 chars) and options (200 chars). For the regex
    scanner this is sufficient for most signatures (universal_quantifier,
    laterality, numeric_ratio are all surface patterns)."""
    options = []
    for o in r.get("generated_options", []):
        options.append({
            "letter": o.get("letter"),
            "text": o.get("text", "") or "",
            "is_correct": bool(o.get("is_correct")),
        })
    return {
        "question_id": f"E2E100-{r.get('source_qid', '?')}",
        "domain_code": r.get("domain"),
        "difficulty_tier": r.get("tier"),
        "question_stem": r.get("generated_stem_preview", "") or "",
        "options": options,
    }


def _baseline_per_letter_class(r: dict) -> dict[str, str]:
    """Read the audit's per-distractor class from the artifact's
    english_gap_per_distractor field. Returns dict[letter, class].
    """
    out: dict[str, str] = {}
    for entry in r.get("english_gap_per_distractor") or []:
        letter = entry.get("letter")
        cls = entry.get("class")
        if letter and cls:
            out[letter] = cls
    return out


def main():
    files = sorted(glob.glob(str(E2E_DIR / "e2e_100_2026-*.json")))
    if not files:
        print(f"ERROR: no e2e_100 artifact found in {E2E_DIR}")
        sys.exit(1)
    artifact = pathlib.Path(files[-1])
    print(f"Loading: {artifact.name}")

    with open(artifact, encoding="utf-8") as fh:
        e2e = json.load(fh)
    results = e2e.get("results", [])
    n = len(results)
    print(f"Total questions: {n}")
    print()

    detector = EnglishGapDetector()

    overrides_per_question: list[dict] = []
    overrides_total = 0
    questions_with_override = 0
    questions_newly_unclean = 0   # was clean (eg=0); now has >=1 override
    questions_more_unclean = 0    # was not-clean (eg>=1); count grew
    by_tier: dict[int, dict] = defaultdict(
        lambda: {"questions": 0, "overrides": 0, "questions_with_override": 0}
    )
    by_signature: Counter = Counter()

    for r in results:
        if r.get("phase_failed"):
            continue
        tier = r.get("tier")
        q = _reconstruct_question(r)
        baseline_classes = _baseline_per_letter_class(r)
        baseline_eg_count = r.get("english_gap_count", 0) or 0

        signals = detector.scan(q)
        # Filter to override signals only.
        overrides_this_q: list[dict] = []
        for s in signals:
            if s.verdict_action != VERDICT_OVERRIDE_TO:
                continue
            # Override only if baseline class wasn't already english_gap.
            baseline = baseline_classes.get(s.letter, "?")
            if baseline == "english_gap":
                continue
            overrides_this_q.append({
                "letter": s.letter,
                "signature": s.signature,
                "confidence": s.confidence,
                "baseline_class": baseline,
                "reason": s.reason,
            })
            by_signature[s.signature] += 1

        by_tier[tier]["questions"] += 1
        if overrides_this_q:
            overrides_per_question.append({
                "source_qid": r.get("source_qid"),
                "anchor_uid": r.get("anchor_uid"),
                "domain": r.get("domain"),
                "tier": tier,
                "baseline_eg_count": baseline_eg_count,
                "new_eg_count": baseline_eg_count + len(overrides_this_q),
                "stem_preview": (q.get("question_stem") or "")[:200],
                "overrides": overrides_this_q,
            })
            overrides_total += len(overrides_this_q)
            questions_with_override += 1
            by_tier[tier]["questions_with_override"] += 1
            by_tier[tier]["overrides"] += len(overrides_this_q)
            if baseline_eg_count == 0:
                questions_newly_unclean += 1
            else:
                questions_more_unclean += 1

    # Baseline metrics from artifact aggregate (the original e2e_100 run).
    baseline_summary = e2e.get("aggregate") or {}
    baseline_clean = baseline_summary.get("english_gap_clean") or sum(
        1 for r in results if (r.get("english_gap_count") or 0) == 0
    )
    baseline_clean_pct = (baseline_clean / n) * 100 if n else 0
    projected_clean = baseline_clean - questions_newly_unclean
    projected_clean_pct = (projected_clean / n) * 100 if n else 0

    print("=" * 70)
    print("Phase A2 — english_gap override simulation on E2E-100")
    print("=" * 70)
    print()
    print(f"Total override-eligible signatures: {sorted(OVERRIDE_ELIGIBLE_SIGNATURES)}")
    print(f"Override-eligible tiers:            {sorted(OVERRIDE_ELIGIBLE_TIERS)}")
    print()
    print(f"Baseline clean (from artifact): {baseline_clean}/{n} ({baseline_clean_pct:.1f}%)")
    print()
    print("Override outcomes:")
    print(f"  Total override events:           {overrides_total}")
    print(f"  Questions with >=1 override:      {questions_with_override}/{n}")
    print(f"    of which newly become unclean: {questions_newly_unclean}")
    print(f"    of which already unclean:      {questions_more_unclean}")
    print()
    print(f"Projected post-A2-audit clean rate (BEFORE fix):")
    print(f"  {projected_clean}/{n} ({projected_clean_pct:.1f}%)")
    print(f"  delta: {projected_clean_pct - baseline_clean_pct:+.1f}pp")
    print(f"  (A2 surfaces MORE eg flags; the plan's 91→94% lift requires fix on top)")
    print()
    print("By tier:")
    for tier in sorted(by_tier.keys()):
        st = by_tier[tier]
        print(f"  T{tier}: questions={st['questions']:3d}  "
              f"overrides={st['overrides']:3d}  "
              f"questions_with_override={st['questions_with_override']:3d}")
    print()
    print("By signature (which patterns fired most):")
    for sig, count in by_signature.most_common():
        print(f"  {sig:30s} {count}")
    print()

    # Save artifact
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    timestamp_safe = timestamp.replace(":", "-")
    out_path = E2E_DIR / f"a2_override_simulation_{timestamp_safe}.json"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump({
            "timestamp": timestamp,
            "source_artifact": artifact.name,
            "n_questions": n,
            "baseline_clean_count": baseline_clean,
            "baseline_clean_pct": round(baseline_clean_pct, 2),
            "overrides_total": overrides_total,
            "questions_with_override": questions_with_override,
            "questions_newly_unclean": questions_newly_unclean,
            "questions_more_unclean": questions_more_unclean,
            "projected_clean_count_pre_fix": projected_clean,
            "projected_clean_pct_pre_fix": round(projected_clean_pct, 2),
            "by_tier": {str(k): v for k, v in by_tier.items()},
            "by_signature": dict(by_signature),
            "override_details": overrides_per_question,
        }, fh, indent=2, ensure_ascii=False)

    print(f"Artifact: {out_path}")
    print()
    print("Next: manual inspection — review override_details in the artifact,")
    print("verify each override is correct, then ship A2.")


if __name__ == "__main__":
    main()
