"""Audit the fresh batch (3 anchors, 44 questions) and report quality
metrics vs the E2E-100 baseline.

For each question in the 3 newly-generated chapters:
  1. Run english_gap audit (Sonnet quorum n=3) — A1+A2.5 in effect
  2. Run diagnostic_quality audit (Sonnet single-pass)
  3. Run routed_fix dispatch on flagged questions (A6 wiring)
  4. Re-audit post-fix
  5. Aggregate per-anchor and overall quality

Compare against E2E-100 baseline:
  - 91% english_gap clean
  - 55% dq clean (T4: 26%)

Cost: ~$3-5 (audit) + ~$1-3 (fix on flagged) = $4-8 total.
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
from pipeline.detectors import VERDICT_OVERRIDE_TO, DetectorSignal  # noqa: E402
from pipeline.fixers import create_fixer_registry  # noqa: E402

TARGET_CHAPTERS = [
    REPO_ROOT / "data/quiz/BPSY/the-frontal-lobe-executive-function-motor-control-and-prefro.json",
    REPO_ROOT / "data/quiz/CASS/the-therapeutic-perimeter-standard-10-and-the-ethics-of-clin.json",
    REPO_ROOT / "data/quiz/CPAT/wired-differently-from-the-start-adhd-autism-and-neurodevelo.json",
]
TARGET_ANCHORS = {
    "D7-PHY-058-fedbfde8",
    "D8-ETH-024-c7600a57",
    "D3-PPA-034-60886d34",
}

OUT_DIR = REPO_ROOT / "data" / ".diagnosis" / "measurement_targets"


def _sonnet_cost(usage: dict) -> float:
    if not usage:
        return 0.0
    return (
        usage.get("input_tokens", 0) / 1e6 * INPUT_PRICE_PER_M
        + usage.get("output_tokens", 0) / 1e6 * OUTPUT_PRICE_PER_M
    )


def _load_target_questions() -> list[dict]:
    out = []
    for path in TARGET_CHAPTERS:
        if not path.exists():
            print(f"WARNING: {path.name} not found")
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"WARNING: {path.name} parse error: {e}")
            continue
        for q in data:
            anchor_uids = q.get("anchor_uids") or []
            if any(uid in TARGET_ANCHORS for uid in anchor_uids):
                q["_chapter_path"] = str(path)
                out.append(q)
    return out


async def _audit_one(client, q: dict, semaphore) -> dict:
    """Run eg + dq audit on one question, then route fixers, then re-audit."""
    out: dict = {
        "question_id": q.get("question_id"),
        "anchor_uid": (q.get("anchor_uids") or [None])[0],
        "tier": q.get("difficulty_tier"),
        "domain": q.get("domain_code"),
    }

    # Initial eg audit (n=3 quorum — production-grade)
    eg_init = await audit_question(client, q, semaphore, n_passes=3)
    eg_init_count = sum(
        1 for c in eg_init.get("classifications") or []
        if c.get("class") == "english_gap"
    )
    out["pre_eg_count"] = eg_init_count
    out["pre_eg_audit_cost"] = _sonnet_cost(eg_init.get("usage", {}))
    out["pre_eg_overrides"] = eg_init.get("english_gap_override_count", 0) or 0

    # Initial dq audit
    dq_init = await audit_diagnostic_quality_question(client, q, semaphore)
    out["pre_dq_class"] = dq_init.get("diagnostic_quality_class")
    out["pre_dq_scores"] = dq_init.get("scores", {})
    out["pre_dq_audit_cost"] = _sonnet_cost(dq_init.get("usage", {}))

    # Routed fixer dispatch (mirrors A6 wiring in ship_readiness)
    fixer_registry = create_fixer_registry()
    scanner_signals = eg_init.get("scanner_signals") or {}
    patched = q
    fixers_applied: list[str] = []
    for letter, sig_data in scanner_signals.items():
        if not sig_data.get("fired"):
            continue
        signature = sig_data.get("signature")
        fixer = fixer_registry.fixer_for_signature(signature)
        if fixer is None:
            continue
        sig = DetectorSignal(
            detector_id="english_gap_scanner",
            letter=letter,
            fired=True,
            confidence=float(sig_data.get("confidence") or 0.0),
            signature=signature,
            verdict_action=VERDICT_OVERRIDE_TO,
            proposed_class="english_gap",
            reason=sig_data.get("reason") or "",
        )
        try:
            new_patched = await fixer.fix(client, patched, sig, semaphore)
            if new_patched != patched:
                patched = new_patched
                fixers_applied.append(f"{fixer.fixer_id}:{letter}")
        except Exception as e:
            out.setdefault("fixer_errors", []).append(str(e))
    out["fixers_applied"] = fixers_applied

    # Post-fix re-audit (only if fixers ran)
    if fixers_applied:
        eg_post = await audit_question(client, patched, semaphore, n_passes=3)
        eg_post_count = sum(
            1 for c in eg_post.get("classifications") or []
            if c.get("class") == "english_gap"
        )
        out["post_eg_count"] = eg_post_count
        out["post_eg_audit_cost"] = _sonnet_cost(eg_post.get("usage", {}))
        dq_post = await audit_diagnostic_quality_question(client, patched, semaphore)
        out["post_dq_class"] = dq_post.get("diagnostic_quality_class")
        out["post_dq_scores"] = dq_post.get("scores", {})
        out["post_dq_audit_cost"] = _sonnet_cost(dq_post.get("usage", {}))
    else:
        out["post_eg_count"] = eg_init_count
        out["post_dq_class"] = out["pre_dq_class"]
        out["post_dq_scores"] = out["pre_dq_scores"]

    out["total_cost"] = (
        out.get("pre_eg_audit_cost", 0)
        + out.get("pre_dq_audit_cost", 0)
        + out.get("post_eg_audit_cost", 0)
        + out.get("post_dq_audit_cost", 0)
    )
    return out


async def main():
    questions = _load_target_questions()
    print(f"Loaded {len(questions)} questions from 3 fresh-batch anchors")
    print()

    api_key = load_api_key()
    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(4)

    print("Auditing + routed-fixing...")
    results = await asyncio.gather(*[
        _audit_one(client, q, semaphore) for q in questions
    ])

    # Aggregate
    n = len(results)
    pre_eg_clean = sum(1 for r in results if r["pre_eg_count"] == 0)
    post_eg_clean = sum(1 for r in results if r["post_eg_count"] == 0)
    pre_dq = {"clean": 0, "minor": 0, "major": 0, None: 0}
    post_dq = {"clean": 0, "minor": 0, "major": 0, None: 0}
    for r in results:
        pre_dq[r["pre_dq_class"]] = pre_dq.get(r["pre_dq_class"], 0) + 1
        post_dq[r["post_dq_class"]] = post_dq.get(r["post_dq_class"], 0) + 1

    by_tier = {1: [], 2: [], 3: [], 4: []}
    for r in results:
        by_tier.setdefault(r["tier"], []).append(r)

    fixers_total = sum(len(r["fixers_applied"]) for r in results)
    total_cost = sum(r["total_cost"] for r in results)
    eg_overrides_total = sum(r.get("pre_eg_overrides", 0) for r in results)

    print()
    print("=" * 70)
    print(f"Fresh-batch audit results ({n} questions)")
    print("=" * 70)
    print()
    print(f"english_gap clean: {pre_eg_clean}/{n} ({pre_eg_clean*100//n}%) "
          f"-> {post_eg_clean}/{n} ({post_eg_clean*100//n}%)  "
          f"[A2.5 overrides applied: {eg_overrides_total}]")
    print()
    print(f"diagnostic_quality:")
    print(f"  pre:  clean={pre_dq['clean']:2d}  minor={pre_dq['minor']:2d}  "
          f"major={pre_dq['major']:2d}  parse_err={pre_dq[None]:2d}")
    print(f"  post: clean={post_dq['clean']:2d}  minor={post_dq['minor']:2d}  "
          f"major={post_dq['major']:2d}  parse_err={post_dq[None]:2d}")
    print()
    print(f"Routed fixers applied: {fixers_total}")
    print(f"Audit + fix cost: ${total_cost:.4f}")
    print()
    print("By tier:")
    for tier in sorted(by_tier.keys()):
        rs = by_tier[tier]
        if not rs:
            continue
        eg_c = sum(1 for r in rs if r["post_eg_count"] == 0)
        dq_c = sum(1 for r in rs if r["post_dq_class"] == "clean")
        print(f"  T{tier}: n={len(rs):2d}  eg_clean={eg_c}/{len(rs)}  "
              f"dq_clean={dq_c}/{len(rs)}")
    print()
    print("By anchor:")
    by_anchor: dict = {}
    for r in results:
        by_anchor.setdefault(r["anchor_uid"], []).append(r)
    for uid, rs in sorted(by_anchor.items()):
        eg_c = sum(1 for r in rs if r["post_eg_count"] == 0)
        dq_c = sum(1 for r in rs if r["post_dq_class"] == "clean")
        dq_m = sum(1 for r in rs if r["post_dq_class"] == "major")
        print(f"  {uid}: n={len(rs):2d}  eg_clean={eg_c}/{len(rs)}  "
              f"dq_clean={dq_c}/{len(rs)}  dq_major={dq_m}")

    # Comparison table vs E2E-100 baseline
    print()
    print("=" * 70)
    print("vs E2E-100 baseline:")
    print("=" * 70)
    print(f"  english_gap clean: fresh={post_eg_clean*100//n}%  e2e100=91%")
    print(f"  dq clean:          fresh={post_dq['clean']*100//n}%  e2e100=55%")
    t4 = by_tier.get(4) or []
    if t4:
        t4_dq_c = sum(1 for r in t4 if r["post_dq_class"] == "clean")
        print(f"  T4 dq clean:       fresh={t4_dq_c*100//len(t4)}%  e2e100=26%")

    # Save artifact
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out_path = OUT_DIR / f"fresh_batch_audit_{timestamp.replace(':', '-')}.json"
    out_path.write_text(json.dumps({
        "timestamp": timestamp,
        "n_questions": n,
        "pre_eg_clean": pre_eg_clean,
        "post_eg_clean": post_eg_clean,
        "pre_dq_dist": pre_dq,
        "post_dq_dist": post_dq,
        "routed_fixers_applied": fixers_total,
        "total_cost": round(total_cost, 4),
        "per_question": results,
    }, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print()
    print(f"Artifact: {out_path}")


if __name__ == "__main__":
    asyncio.run(main())
