"""B5 — verify B3's tightened prompt on PMET D1-LEA-008.

Audits the 8 questions on this anchor (2 pre-B3 from anchor-4 partial
run + 6 new with B3-tightened prompt). Reports dq + eg distribution
and compares to the 34% dq clean baseline from the fresh-batch audit.

Cost: ~$1 (8 questions × ~$0.10 audit).
"""
from __future__ import annotations

import asyncio
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
from audit_diagnostic_quality import audit_diagnostic_quality_question  # noqa: E402

import os
_ANCHOR = os.environ.get("VERIFY_ANCHOR", "D1-LEA-008-f5ad52bd")
_CHAPTER = os.environ.get(
    "VERIFY_CHAPTER",
    "data/quiz/PMET/the-architecture-of-association-how-classical-conditioning-b.json",
)
TARGET_CHAPTER = REPO_ROOT / _CHAPTER
TARGET_ANCHOR = _ANCHOR
OUT_DIR = REPO_ROOT / "data" / ".diagnosis" / "measurement_targets"


def _sonnet_cost(usage):
    if not usage: return 0.0
    return (usage.get("input_tokens", 0) / 1e6 * INPUT_PRICE_PER_M
            + usage.get("output_tokens", 0) / 1e6 * OUTPUT_PRICE_PER_M)


async def _audit_one(client, q, semaphore):
    eg = await audit_question(client, q, semaphore, n_passes=3)
    dq = await audit_diagnostic_quality_question(client, q, semaphore)
    eg_count = sum(1 for c in eg.get("classifications") or [] if c.get("class") == "english_gap")
    return {
        "question_id": q.get("question_id"),
        "tier": q.get("difficulty_tier"),
        "eg_count": eg_count,
        "dq_class": dq.get("diagnostic_quality_class"),
        "dq_scores": dq.get("scores", {}),
        "cost": _sonnet_cost(eg.get("usage", {})) + _sonnet_cost(dq.get("usage", {})),
    }


async def main():
    data = json.loads(TARGET_CHAPTER.read_text(encoding="utf-8"))
    questions = [q for q in data if TARGET_ANCHOR in (q.get("anchor_uids") or [])]
    print(f"Auditing {len(questions)} questions on PMET D1-LEA-008")
    print()

    client = anthropic.AsyncAnthropic(api_key=load_api_key())
    semaphore = asyncio.Semaphore(4)

    results = await asyncio.gather(*[_audit_one(client, q, semaphore) for q in questions])

    n = len(results)
    eg_clean = sum(1 for r in results if r["eg_count"] == 0)
    dq = {"clean": 0, "minor": 0, "major": 0, None: 0}
    for r in results:
        dq[r["dq_class"]] = dq.get(r["dq_class"], 0) + 1

    by_tier = {}
    for r in results:
        by_tier.setdefault(r["tier"], []).append(r)

    total_cost = sum(r["cost"] for r in results)

    print("=" * 70)
    print(f"B5 — Tightened-prompt verification ({n} questions)")
    print("=" * 70)
    print()
    print(f"english_gap clean: {eg_clean}/{n} ({eg_clean*100//n}%)")
    print(f"dq distribution:")
    print(f"  clean={dq['clean']}  minor={dq['minor']}  major={dq['major']}  parse_err={dq[None]}")
    if n:
        print(f"  dq clean rate: {dq['clean']*100/n:.0f}%")
    print()
    print("By tier:")
    for tier in sorted(by_tier.keys()):
        rs = by_tier[tier]
        eg_c = sum(1 for r in rs if r["eg_count"] == 0)
        dq_c = sum(1 for r in rs if r["dq_class"] == "clean")
        dq_m = sum(1 for r in rs if r["dq_class"] == "major")
        print(f"  T{tier}: n={len(rs)}  eg_clean={eg_c}/{len(rs)}  "
              f"dq_clean={dq_c}/{len(rs)}  dq_major={dq_m}")
    print()
    print(f"Audit cost: ${total_cost:.4f}")

    print()
    print("=" * 70)
    print("vs Fresh-Batch Baseline (B3 prompt change effect)")
    print("=" * 70)
    print(f"  metric            this run    baseline (Apr 29)")
    print(f"  english_gap clean {eg_clean*100//n if n else 0:>3}%       90%")
    if n:
        dq_pct = dq['clean']*100/n
        delta = dq_pct - 34
        print(f"  dq clean          {dq_pct:>3.0f}%       34%   delta {delta:+.0f}pp")

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out = OUT_DIR / f"b5_verify_{timestamp.replace(':', '-')}.json"
    out.write_text(json.dumps({
        "timestamp": timestamp,
        "n_questions": n,
        "eg_clean": eg_clean,
        "dq_distribution": dq,
        "by_tier": {str(t): len(rs) for t, rs in by_tier.items()},
        "total_cost": round(total_cost, 4),
        "results": results,
    }, indent=2, default=str), encoding="utf-8")
    print()
    print(f"Artifact: {out}")


if __name__ == "__main__":
    asyncio.run(main())
