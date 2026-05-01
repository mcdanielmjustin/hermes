"""rescue_chapter_stems.py — stem-rewrite rescue for review chapters.

Phase 20d. When ship_readiness leaves a chapter in `data/quiz_review/`
because auto-fix couldn't escape an english_gap distractor, the
underlying problem is often that the STEM over-specifies — it prints
the discriminator variable's value (a numeric finding, a directional
term, an explicit framing) so any plausible-but-wrong distractor that
engages the stem's vocabulary becomes lexically rejectable.

This script rewrites the STEM (not the distractors) using a template
library at `pipeline/stem_templates.py`. The template encodes:
  - The discriminator dimension being probed (e.g., locus_pre_vs_post)
  - Omit rules ("don't print the receptor outcome")
  - Correct answer shape ("mechanism marker required")

The rewrite preserves: the testable concept, the Bloom's tier, the
correct answer (semantically), and the distractors' identity. After
rewrite, the question is re-audited; if english_gap remains on the
same distractor, the existing fix_question flow is used as a single
fallback to rewrite that distractor (cap: 1 fallback). Beyond that,
the question surfaces to a separate review queue.

Usage:
  python scripts/rescue_chapter_stems.py data/quiz_review/BPSY/foo.json
  python scripts/rescue_chapter_stems.py data/quiz_review/ --workers 4
  python scripts/rescue_chapter_stems.py --dry-run data/quiz_review/

Output:
  data/quiz_rescue/{DOMAIN}/{filename}  — rewritten chapters
  logs/rescue_chapter_stems_<date>/     — per-rewrite reports

Cost: ~$0.05-0.10 per chapter (audit + 1 rewrite + re-audit, with
optional fallback fix).

Important: rescued questions are NOT automatically gate-validated for
Bloom's tier preservation. After rescue, run ship_readiness against
data/quiz_rescue/ to verify english_gap is resolved. For full Bloom-
level safety, the rewrites should pass through the generation
pipeline's identity gates before merge into data/quiz/. This script
produces CANDIDATES, not finalized questions.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import re
import sys
from datetime import datetime, timezone

import anthropic

SCRIPT_DIR = pathlib.Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from audit_stem_contradictions import (  # noqa: E402
    audit_question, fix_question, load_api_key,
    INPUT_PRICE_PER_M, OUTPUT_PRICE_PER_M, MODEL_ID,
    parse_response,
)
from pipeline.stem_templates import (  # noqa: E402
    StemTemplate, get_template, get_alternate_templates,
)
from pipeline.anchor_flavor import flavor_for_anchor  # noqa: E402
from pipeline.agents import AnchorBriefAgent  # noqa: E402


# ── Stem-rewrite prompt ─────────────────────────────────────

STEM_REWRITE_PROMPT = """You are rewriting the STEM of a multiple-choice question. The current stem over-specifies — it prints information that makes one distractor lexically rejectable without concept knowledge (an "english_gap" failure). Your job: rewrite the stem so the testable concept is preserved but the over-specification is removed, eliminating the lexical trap.

CRITICAL CONSTRAINTS:
1. PRESERVE the Bloom's tier (currently T{tier}). Do not drift the cognitive demand.
   - T1 = Remember: direct factual question, no scenario.
   - T2 = Understand: brief context (1-2 sentences max).
   - T3 = Apply: novel scenario (named subject, clinical context, "given X" / "after Y" setup).
   - T4 = Evaluate/Analyze: integrative scenario requiring synthesis.
2. PRESERVE the correct answer's meaning. The rewrite must still be answered correctly by the existing correct option's CLAIM (you may need to adjust slightly, but the testable concept stays the same).
3. PRESERVE the distractors as written. Don't rewrite them.
4. PRESERVE the testable concept (printed below as concept_explanation).
5. APPLY the omit rules below. Each rule says what NOT to print in the stem.
6. The new stem must NOT contain reasoning markers in noun-form claims, must NOT exceed normal stem length (≤300 chars for T1/T2, ≤500 for T3/T4).

CURRENT STEM (over-specifying):
{original_stem}

CURRENT CORRECT ANSWER (preserve):
{correct_letter}: {correct_text}

CURRENT DISTRACTORS (preserve):
{distractors_block}

WHY THIS QUESTION IS FLAGGED:
The auditor classified distractor {flagged_letter} as english_gap. Specific reason:
"{flag_explanation}"
The stem fact contradicted: "{contradicted_stem_fact}"

TESTABLE CONCEPT (the irreducible idea the student must invoke to answer correctly):
{concept_explanation}

DISCRIMINATOR DIMENSION (the variable the student must resolve):
{discriminator}

OMIT RULES (what the new stem MUST NOT contain):
{omit_rules_block}

YOUR TASK: write a new stem that probes the same concept, stays at T{tier}, and follows the omit rules. The new stem should describe the SETUP and ASK the question without giving away the discriminator's value.

Respond ONLY with valid JSON:
{{"new_stem": "<rewritten stem text>", "rationale": "<1-2 sentences explaining how the rewrite removes the lexical trap while preserving the testable concept and Bloom's tier>"}}"""


# ── Helpers ─────────────────────────────────────────────────

def _calc_cost(usage: dict) -> float:
    return (
        usage.get("input_tokens", 0) / 1e6 * INPUT_PRICE_PER_M
        + usage.get("output_tokens", 0) / 1e6 * OUTPUT_PRICE_PER_M
    )


def _format_options_block(options: list[dict]) -> str:
    return "\n".join(
        f"  {o.get('letter', '?')} "
        f"{'[CORRECT]' if o.get('is_correct') else '[distractor]'}: "
        f"{o.get('text', '')}"
        for o in options
    )


def _format_distractors_block(options: list[dict]) -> str:
    return "\n".join(
        f"  {o.get('letter', '?')}: {o.get('text', '')}"
        for o in options
        if not o.get("is_correct")
    )


def _correct_option(options: list[dict]) -> dict | None:
    for o in options:
        if o.get("is_correct"):
            return o
    return None


def _load_chapter(path: pathlib.Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        d = json.load(fh)
    return d if isinstance(d, list) else d.get("questions", [])


def _save_rescued_chapter(out_dir: pathlib.Path, source_path: pathlib.Path,
                           questions: list[dict]) -> pathlib.Path:
    """Save the rewritten chapter to data/quiz_rescue/{DOMAIN}/{filename}.

    Atomic write via temp + os.replace to match ship_readiness's
    _write_chapter pattern. A crash mid-write previously left a corrupt
    JSON file at the destination; the tempfile-then-rename pattern
    keeps the destination either fully-old-or-fully-new, never partial.
    """
    import os
    import tempfile
    domain = source_path.parent.name
    out_path = out_dir / domain / source_path.name
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(out_path.parent), prefix=".rescue_tmp_", suffix=".json",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(questions, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, str(out_path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return out_path


# ── Brief lookup ────────────────────────────────────────────

_brief_agent = AnchorBriefAgent()


def _load_brief(briefs_dir: pathlib.Path, domain_code: str,
                anchor_uid: str) -> dict:
    """Load the anchor brief (returns empty dict shape if missing)."""
    return _brief_agent.run({
        "anchor_briefs_dir": briefs_dir,
        "domain_code": domain_code,
        "uid": anchor_uid,
    })


# ── Per-question stem rewrite ───────────────────────────────

async def _rewrite_one_stem(
    client, question: dict, audit_finding: dict,
    template: StemTemplate, brief: dict, semaphore,
) -> tuple[dict | None, dict, str | None]:
    """Rewrite the stem of a question. Returns (new_question, usage, error)."""
    options = question.get("options", []) or []
    correct = _correct_option(options)
    if not correct:
        return None, {"input_tokens": 0, "output_tokens": 0}, "no_correct_option"

    omit_rules_block = "\n".join(f"  - {r}" for r in template.omit_rules)
    prompt = STEM_REWRITE_PROMPT.format(
        tier=question.get("difficulty_tier", "?"),
        original_stem=question.get("question_stem", ""),
        correct_letter=correct.get("letter", "?"),
        correct_text=correct.get("text", ""),
        distractors_block=_format_distractors_block(options),
        flagged_letter=audit_finding.get("letter", "?"),
        flag_explanation=audit_finding.get("explanation", "")[:500],
        contradicted_stem_fact=audit_finding.get("contradicted_stem_fact", ""),
        concept_explanation=brief.get("concept_explanation", "") or
                              "(no concept_explanation in brief — fallback to testable_fact)",
        discriminator=template.discriminator,
        omit_rules_block=omit_rules_block,
    )

    async with semaphore:
        try:
            response = await client.messages.create(
                model=MODEL_ID,
                max_tokens=1024,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception as e:
            return None, {"input_tokens": 0, "output_tokens": 0}, f"api_error: {e}"

    text = response.content[0].text if response.content else ""
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    parsed = parse_response(text)
    if not parsed or not parsed.get("new_stem"):
        return None, usage, "parse_or_missing_new_stem"

    # Deep-copy the question so subsequent mutations to rescue_metadata
    # (and any other nested dicts) don't leak back to the input. A
    # shallow dict() copy + setdefault would share the same nested dict
    # reference if rescue_metadata already exists on the input — we'd
    # overwrite the original's metadata, breaking re-rescue scenarios
    # where the input is itself a previously-rescued question.
    import copy
    new_question = copy.deepcopy(question)
    new_question["question_stem"] = parsed["new_stem"]
    # Mark the rescue lineage on the question so post-merge inspection
    # can identify rewrites and run them through full gate validation
    # before adding to data/quiz/.
    new_question.setdefault("rescue_metadata", {})
    new_question["rescue_metadata"]["stem_rewritten"] = True
    new_question["rescue_metadata"]["original_stem"] = question.get("question_stem", "")
    new_question["rescue_metadata"]["rewrite_template"] = (
        f"{template.domain_code}/{template.flavor}/T{template.bloom_tier}/"
        f"{template.discriminator}"
    )
    new_question["rescue_metadata"]["rewrite_rationale"] = parsed.get("rationale", "")
    new_question["rescue_metadata"]["rewritten_at"] = (
        datetime.now(timezone.utc).isoformat(timespec="seconds")
    )
    return new_question, usage, None


# ── Per-chapter orchestration ───────────────────────────────

async def process_chapter(
    client, source_path: pathlib.Path, briefs_dir: pathlib.Path,
    out_dir: pathlib.Path, semaphore,
) -> dict:
    """Process a single chapter: audit, rewrite stems for english_gap
    questions, re-audit, fall back to fix_question if needed, save
    rescued chapter."""
    questions = _load_chapter(source_path)
    if not questions:
        return {"chapter": str(source_path), "status": "empty", "cost": 0.0}

    # 1. Audit each question
    audit_tasks = [audit_question(client, q, semaphore) for q in questions]
    audit_results = await asyncio.gather(*audit_tasks)
    total_cost = sum(_calc_cost(r.get("usage", {})) for r in audit_results)

    rewritten_questions = []
    per_question_outcomes = []

    for q, audit in zip(questions, audit_results):
        # Find any english_gap distractor in this question's audit.
        english_gap_class = next(
            (c for c in (audit.get("classifications") or [])
             if c.get("class") == "english_gap"),
            None,
        )
        if english_gap_class is None:
            # Question has no english_gap — pass through unchanged.
            rewritten_questions.append(q)
            per_question_outcomes.append({
                "qid": q.get("question_id"),
                "outcome": "no_english_gap",
            })
            continue

        # Look up template for this question's cell.
        domain_code = q.get("domain_code", "?")
        tier = q.get("difficulty_tier", 0)
        anchor_uids = q.get("anchor_uids") or []
        anchor_uid = anchor_uids[0] if anchor_uids else None
        flavor_value = (
            (q.get("generation_metadata") or {}).get("flavor")
            or flavor_for_anchor(anchor_uid, domain_code)
        )

        candidates = get_alternate_templates(domain_code, flavor_value, tier)
        if not candidates:
            # No template registered for this cell — keep original; mark for review.
            rewritten_questions.append(q)
            per_question_outcomes.append({
                "qid": q.get("question_id"),
                "outcome": "no_template_for_cell",
                "cell": f"{domain_code}/{flavor_value}/T{tier}",
            })
            continue

        # Load the brief (concept_explanation + discriminators)
        brief = _load_brief(briefs_dir, domain_code, anchor_uid) if anchor_uid else {}

        # Try each candidate template; first one that re-audits clean wins.
        # Cap: 2 alternates (3 attempts total).
        rewrite_succeeded = False
        last_attempt_question = None
        for tpl in candidates[:3]:
            new_q, usage, err = await _rewrite_one_stem(
                client, q, english_gap_class, tpl, brief, semaphore,
            )
            total_cost += _calc_cost(usage)
            if err or new_q is None:
                continue
            # Re-audit the rewritten question
            re_audit = await audit_question(client, new_q, semaphore)
            total_cost += _calc_cost(re_audit.get("usage", {}))
            still_eg = any(
                c.get("class") == "english_gap"
                for c in (re_audit.get("classifications") or [])
            )
            last_attempt_question = new_q
            if not still_eg:
                rewritten_questions.append(new_q)
                per_question_outcomes.append({
                    "qid": q.get("question_id"),
                    "outcome": "rewrote_stem",
                    "template": (
                        f"{tpl.domain_code}/{tpl.flavor}/T{tpl.bloom_tier}/"
                        f"{tpl.discriminator}"
                    ),
                })
                rewrite_succeeded = True
                break

        if rewrite_succeeded:
            continue

        # Fall back to fix_question for the english_gap distractor
        # (single fallback, no recursion).
        if last_attempt_question is None:
            last_attempt_question = q
        # Build a synthetic audit_result for fix_question
        fallback_audit = {
            "flagged_distractors": [{
                "letter": english_gap_class.get("letter"),
                "distractor_text": english_gap_class.get("distractor_text", ""),
                "contradicted_stem_fact": english_gap_class.get(
                    "contradicted_stem_fact", ""),
                "explanation": english_gap_class.get("explanation", ""),
            }]
        }
        # Ensure misconception_type is set on the flagged distractor for fix_question
        for o in last_attempt_question.get("options", []):
            if not o.get("is_correct"):
                o.setdefault("misconception_type", "unknown")
                o.setdefault("misconception_id", o.get("letter", "?"))

        fix_result = await fix_question(
            client, last_attempt_question, fallback_audit, semaphore,
        )
        total_cost += _calc_cost(fix_result.get("usage", {}))
        if fix_result.get("patched"):
            patched_q = fix_result.get("question", last_attempt_question)
            patched_q.setdefault("rescue_metadata", {})
            patched_q["rescue_metadata"]["distractor_fix_applied"] = True
            rewritten_questions.append(patched_q)
            per_question_outcomes.append({
                "qid": q.get("question_id"),
                "outcome": "stem_then_distractor_fix",
            })
        else:
            rewritten_questions.append(last_attempt_question)
            per_question_outcomes.append({
                "qid": q.get("question_id"),
                "outcome": "did_not_converge",
            })

    out_path = _save_rescued_chapter(out_dir, source_path, rewritten_questions)
    return {
        "chapter": str(source_path),
        "out_path": str(out_path),
        "questions_processed": len(questions),
        "outcomes": per_question_outcomes,
        "cost_usd": round(total_cost, 4),
    }


# ── Main ────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "input_path",
        help="Path to a chapter JSON OR a directory of them",
    )
    parser.add_argument(
        "--out-dir", default="data/quiz_rescue",
        help="Output dir (default: data/quiz_rescue)",
    )
    parser.add_argument(
        "--briefs-dir", default="data/anchor_briefs",
        help="Anchor briefs directory",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    in_path = pathlib.Path(args.input_path).resolve()
    out_dir = pathlib.Path(args.out_dir).resolve()
    briefs_dir = pathlib.Path(args.briefs_dir).resolve()

    # Resolve chapter list
    if in_path.is_file():
        chapters = [in_path]
    else:
        chapters = sorted(in_path.rglob("*.json"))
        chapters = [c for c in chapters if not any(t in c.name
                    for t in ("audit", "fixed", "manifest"))]

    if not chapters:
        print(f"No chapters found at {in_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Input: {in_path}")
    print(f"Output: {out_dir}")
    print(f"Briefs: {briefs_dir}")
    print(f"Chapters: {len(chapters)}")
    print()

    if args.dry_run:
        for c in chapters:
            print(f"  [dry] would process: {c}")
        return

    api_key = args.api_key or load_api_key()
    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(args.workers)

    results = []
    for chapter_path in chapters:
        print(f"  Processing {chapter_path.name}...", flush=True)
        r = await process_chapter(
            client, chapter_path, briefs_dir, out_dir, semaphore,
        )
        results.append(r)
        for outcome in r.get("outcomes", []):
            print(f"    {outcome.get('qid', '?'):<40} {outcome.get('outcome')}")
        print(f"    cost: ${r.get('cost_usd', 0):.4f}")

    # Summary
    print()
    print("=" * 70)
    total_cost = sum(r.get("cost_usd", 0) for r in results)
    total_questions = sum(r.get("questions_processed", 0) for r in results)
    by_outcome: dict[str, int] = {}
    for r in results:
        for o in r.get("outcomes", []):
            by_outcome[o["outcome"]] = by_outcome.get(o["outcome"], 0) + 1
    print(f"Chapters processed:   {len(results)}")
    print(f"Questions processed:  {total_questions}")
    for outcome, count in sorted(by_outcome.items(), key=lambda x: -x[1]):
        print(f"  {outcome:<35} {count}")
    print(f"Total cost:           ${total_cost:.4f}")
    print()
    print("Next: python scripts/ship_readiness.py --quiz-dir "
          f"{out_dir} --ship-dir data/quiz_rescue_shippable "
          "--review-dir data/quiz_rescue_review")


if __name__ == "__main__":
    asyncio.run(main())
