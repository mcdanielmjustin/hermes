"""Phase B1: Fix-and-export the 44 fresh-batch questions to ~/Downloads.

Loads the 3 fresh-batch chapters (BPSY frontal-lobe, CASS clinical-
ethics-perimeter, CPAT neurodevelopmental). Filters to the 3 target
anchors. For each question, runs:

  1. english_gap audit (n=3 quorum)
  2. diagnostic_quality audit (single pass)
  3. Routed fixer dispatch on scanner signatures (now includes B2's
     numeric_ratio + stage_timing fixers + the existing 5)
  4. Post-fix re-audit (eg + dq)
  5. Saves before/after to a unified export

Outputs:
  ~/Downloads/goliath-freshbatch-fixed-<timestamp>.json — full data
  ~/Downloads/goliath-freshbatch-fixed-<timestamp>.md   — human summary

Cost target: ~$3-5 (44 questions × ~$0.10 audit + occasional fixer LLM).
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

DOWNLOADS_DIR = pathlib.Path.home() / "Downloads"

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
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        for q in data:
            anchor_uids = q.get("anchor_uids") or []
            if any(uid in TARGET_ANCHORS for uid in anchor_uids):
                q["_chapter_path"] = str(path)
                out.append(q)
    return out


async def _process_one(client, q: dict, semaphore, fixer_registry) -> dict:
    rec: dict = {
        "question_id": q.get("question_id"),
        "anchor_uid": (q.get("anchor_uids") or [None])[0],
        "tier": q.get("difficulty_tier"),
        "domain": q.get("domain_code"),
        "before": {
            "stem": q.get("question_stem", "")[:300],
            "options": [
                {"letter": o.get("letter"), "text": (o.get("text") or "")[:200],
                 "is_correct": bool(o.get("is_correct"))}
                for o in q.get("options") or []
            ],
        },
    }

    # Pre-fix eg audit (n=3 quorum)
    eg_init = await audit_question(client, q, semaphore, n_passes=3)
    eg_init_count = sum(
        1 for c in eg_init.get("classifications") or []
        if c.get("class") == "english_gap"
    )
    rec["pre_eg_count"] = eg_init_count
    pre_eg_cost = _sonnet_cost(eg_init.get("usage", {}))

    # Pre-fix dq audit
    dq_init = await audit_diagnostic_quality_question(client, q, semaphore)
    rec["pre_dq_class"] = dq_init.get("diagnostic_quality_class")
    rec["pre_dq_scores"] = dq_init.get("scores", {})
    pre_dq_cost = _sonnet_cost(dq_init.get("usage", {}))

    # Routed fixer dispatch
    scanner_signals = eg_init.get("scanner_signals") or {}
    patched = q
    fixers_applied: list[str] = []
    fix_cost = 0.0
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
                # Estimate $0.005 per LLM-backed fix; deterministic = $0
                if "llm" in fixer.fixer_id or fixer.fixer_id == "numeric_ratio_fixer":
                    fix_cost += 0.005
        except Exception as e:
            rec.setdefault("fixer_errors", []).append(str(e))
    rec["fixers_applied"] = fixers_applied

    # Post-fix re-audit (only if anything changed)
    if fixers_applied:
        eg_post = await audit_question(client, patched, semaphore, n_passes=3)
        eg_post_count = sum(
            1 for c in eg_post.get("classifications") or []
            if c.get("class") == "english_gap"
        )
        rec["post_eg_count"] = eg_post_count
        rec["post_eg_audit_cost"] = _sonnet_cost(eg_post.get("usage", {}))
        dq_post = await audit_diagnostic_quality_question(client, patched, semaphore)
        rec["post_dq_class"] = dq_post.get("diagnostic_quality_class")
        rec["post_dq_scores"] = dq_post.get("scores", {})
        rec["post_dq_audit_cost"] = _sonnet_cost(dq_post.get("usage", {}))
    else:
        rec["post_eg_count"] = eg_init_count
        rec["post_dq_class"] = rec["pre_dq_class"]
        rec["post_dq_scores"] = rec["pre_dq_scores"]
        rec["post_eg_audit_cost"] = 0.0
        rec["post_dq_audit_cost"] = 0.0

    rec["after"] = {
        "stem": patched.get("question_stem", "")[:300],
        "options": [
            {"letter": o.get("letter"), "text": (o.get("text") or "")[:200],
             "is_correct": bool(o.get("is_correct")),
             "_routed_fixer": o.get("_routed_fixer")}
            for o in patched.get("options") or []
        ],
    }

    rec["cost_usd"] = round(
        pre_eg_cost + pre_dq_cost + fix_cost
        + rec["post_eg_audit_cost"] + rec["post_dq_audit_cost"],
        4,
    )
    return rec


def _render_md(report: dict) -> str:
    a = report["aggregate"]
    lines = [
        "# Goliath Fresh Batch — Fix & Export",
        "",
        f"Generated: {report['timestamp']}",
        "",
        "## Summary",
        "",
        f"- Total questions: **{a['n_total']}**",
        f"- Questions where routed fixers applied: **{a['n_fixers_used']}**",
        f"- Total cost: **${a['total_cost']:.4f}**",
        "",
        "### english_gap clean rate",
        "",
        f"- Pre-fix: {a['pre_eg_clean']}/{a['n_total']} ({a['pre_eg_clean_pct']:.0f}%)",
        f"- Post-fix: {a['post_eg_clean']}/{a['n_total']} ({a['post_eg_clean_pct']:.0f}%)",
        "",
        "### diagnostic_quality distribution",
        "",
        "| Class | Pre | Post |",
        "|---|---|---|",
        f"| clean | {a['pre_dq']['clean']} | {a['post_dq']['clean']} |",
        f"| minor | {a['pre_dq']['minor']} | {a['post_dq']['minor']} |",
        f"| major | {a['pre_dq']['major']} | {a['post_dq']['major']} |",
        f"| parse_err | {a['pre_dq']['parse_err']} | {a['post_dq']['parse_err']} |",
        "",
        "### By anchor",
        "",
        "| Anchor | n | post eg clean | post dq clean | post dq major |",
        "|---|---|---|---|---|",
    ]
    for uid, st in sorted(a["by_anchor"].items()):
        lines.append(
            f"| {uid} | {st['n']} | {st['post_eg_clean']}/{st['n']} | "
            f"{st['post_dq_clean']}/{st['n']} | {st['post_dq_major']} |"
        )

    lines.extend([
        "",
        "## Per-question detail",
        "",
    ])
    for q in report["questions"]:
        lines.append(
            f"### {q['question_id']} (T{q['tier']}, {q['domain']})"
        )
        lines.append("")
        lines.append(
            f"- Pre: eg={q['pre_eg_count']} dq={q['pre_dq_class']} "
            f"(scores={q['pre_dq_scores']})"
        )
        lines.append(
            f"- Post: eg={q['post_eg_count']} dq={q['post_dq_class']} "
            f"(scores={q['post_dq_scores']})"
        )
        if q.get("fixers_applied"):
            lines.append(f"- Fixers applied: {q['fixers_applied']}")
        lines.append("")
        lines.append("**Final stem:**")
        lines.append("")
        lines.append(f"> {q['after']['stem']}")
        lines.append("")
        lines.append("**Final options:**")
        for o in q["after"]["options"]:
            mark = "✓" if o.get("is_correct") else " "
            tag = f" [{o['_routed_fixer']}]" if o.get("_routed_fixer") else ""
            lines.append(f"- {mark} {o.get('letter')}: {o.get('text')}{tag}")
        lines.append("")
    return "\n".join(lines)


async def main():
    DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    questions = _load_target_questions()
    print(f"Loaded {len(questions)} questions from 3 fresh-batch chapters")
    print()

    api_key = load_api_key()
    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(4)

    fixer_registry = create_fixer_registry()
    print(f"Fixer registry: {[f.fixer_id for f in fixer_registry.all_fixers()]}")
    print()

    print(f"Processing {len(questions)} questions...")
    results = await asyncio.gather(*[
        _process_one(client, q, semaphore, fixer_registry) for q in questions
    ])

    # Aggregate
    n = len(results)
    pre_eg_clean = sum(1 for r in results if r["pre_eg_count"] == 0)
    post_eg_clean = sum(1 for r in results if r["post_eg_count"] == 0)
    pre_dq = {"clean": 0, "minor": 0, "major": 0, "parse_err": 0}
    post_dq = {"clean": 0, "minor": 0, "major": 0, "parse_err": 0}
    for r in results:
        pre_key = r["pre_dq_class"] if r["pre_dq_class"] in pre_dq else "parse_err"
        post_key = r["post_dq_class"] if r["post_dq_class"] in post_dq else "parse_err"
        pre_dq[pre_key] += 1
        post_dq[post_key] += 1

    by_anchor: dict = {}
    for r in results:
        uid = r["anchor_uid"]
        st = by_anchor.setdefault(uid, {"n": 0, "post_eg_clean": 0, "post_dq_clean": 0, "post_dq_major": 0})
        st["n"] += 1
        if r["post_eg_count"] == 0:
            st["post_eg_clean"] += 1
        if r["post_dq_class"] == "clean":
            st["post_dq_clean"] += 1
        if r["post_dq_class"] == "major":
            st["post_dq_major"] += 1

    n_fixers = sum(1 for r in results if r["fixers_applied"])
    total_cost = sum(r["cost_usd"] for r in results)

    aggregate = {
        "n_total": n,
        "n_fixers_used": n_fixers,
        "pre_eg_clean": pre_eg_clean,
        "pre_eg_clean_pct": pre_eg_clean * 100 / n if n else 0,
        "post_eg_clean": post_eg_clean,
        "post_eg_clean_pct": post_eg_clean * 100 / n if n else 0,
        "pre_dq": pre_dq,
        "post_dq": post_dq,
        "by_anchor": by_anchor,
        "total_cost": round(total_cost, 4),
    }

    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    timestamp_safe = timestamp.replace(":", "-")
    report = {
        "timestamp": timestamp,
        "aggregate": aggregate,
        "questions": results,
    }
    json_path = DOWNLOADS_DIR / f"goliath-freshbatch-fixed-{timestamp_safe}.json"
    md_path = DOWNLOADS_DIR / f"goliath-freshbatch-fixed-{timestamp_safe}.md"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    md_path.write_text(_render_md(report), encoding="utf-8")

    print()
    print("=" * 70)
    print("FRESH BATCH FIX & EXPORT SUMMARY")
    print("=" * 70)
    print(f"Total: {n}")
    print(f"english_gap clean: {pre_eg_clean} -> {post_eg_clean} of {n}")
    print(f"dq distribution:")
    print(f"  pre:  clean={pre_dq['clean']:2d}  minor={pre_dq['minor']:2d}  major={pre_dq['major']:2d}  parse_err={pre_dq['parse_err']:2d}")
    print(f"  post: clean={post_dq['clean']:2d}  minor={post_dq['minor']:2d}  major={post_dq['major']:2d}  parse_err={post_dq['parse_err']:2d}")
    print(f"Routed fixers used on: {n_fixers} questions")
    print(f"Total cost: ${total_cost:.4f}")
    print()
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")


if __name__ == "__main__":
    asyncio.run(main())
