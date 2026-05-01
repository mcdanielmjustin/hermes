"""Diagnosis Test 1 — Self-critique step (Layer 8).

Runs a single Opus call AFTER each baseline question with the audit's rubric in
context, asking Opus to self-classify and revise english_gap distractors. Compares
audit metrics before/after the self-critique step.

Hypothesis: in-loop semantic feedback (Opus reviewing its own output against the
audit's standard) substantially reduces english_gap rate. If so, this test
validates that the cheapest fix to goliath's plateau is a single extra Opus call
per question — not the multi-week distractors-first rebuild.

Cost on the CPAT D3-PPA-003 chapter (8 questions): ~$1.10 total
  - baseline audit (8 × 3 passes Sonnet) ~$0.50
  - self-critique calls (8 × 1 Opus call) ~$0.10
  - revised audit (8 × 3 passes Sonnet) ~$0.50

Run:
  python scripts/diagnosis/test_1_self_critique.py
"""
from __future__ import annotations

import asyncio
import json
import pathlib
import sys
from datetime import datetime, timezone

import anthropic

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from audit_stem_contradictions import (  # noqa: E402
    audit_question, parse_response, load_api_key,
    INPUT_PRICE_PER_M, OUTPUT_PRICE_PER_M,
)

OPUS_MODEL_ID = "claude-opus-4-7"
OPUS_INPUT_PRICE_PER_M = 15.0
OPUS_OUTPUT_PRICE_PER_M = 75.0

ANCHOR_UID = "D3-PPA-003-bda8c4e9"
CHAPTER_PATH = (
    REPO_ROOT / "data" / ".p21_focus" / "quiz" / "CPAT"
    / "when-the-light-goes-out-depression-its-models-and-who-it-str.json"
)
OUT_DIR = REPO_ROOT / "data" / ".diagnosis" / "test_1"

SELF_CRITIQUE_PROMPT = """You are reviewing your own draft of a multiple-choice question before it ships. The downstream audit will classify each of your distractors using the rubric below. Your job NOW is to (1) self-audit each distractor against this rubric, and (2) revise any distractor classified as english_gap.

THE FOUR CLASSES (the audit's exact rubric — your output WILL be evaluated against this):

ENGLISH_GAP — A student can reject the distractor by lexical comparison with the stem alone. No domain knowledge needed. The contradiction is in the printed words, not in the concepts.
  Canonical example:
    Stem: "After bilateral hippocampal damage, Lester Nichols cannot form new declarative memories but still recalls his wedding from a decade earlier."
    Distractor: "Retrograde amnesia erases ALL pre-injury memories regardless of when they were consolidated."
    Why english_gap: "all pre-injury memories" is contradicted by "wedding from a decade earlier" — student needs no concept knowledge.

CONTENT_GAP — Rejecting the distractor requires invoking concept/domain knowledge. A real contradiction exists, but recognizing it requires knowing what some technical term or relationship means.
  Canonical example:
    Stem: "A compound binds a receptor but produces no measurable postsynaptic activity on its own."
    Distractor: "The compound exhibits intrinsic activity that mimics the endogenous neurotransmitter."
    Why content_gap: rejecting requires knowing that "intrinsic activity" implies a measurable postsynaptic effect.

CLEAN — No direct contradiction with stem facts. Plausible-but-wrong; rejection requires applying the concept.

SOFT_FLAG — Genuinely uncertain between two classes after honest application. Use sparingly.

CLASSIFICATION RULE for each distractor:
  "Could a student reject this distractor by re-reading the stem and the distractor alone, with NO domain knowledge invoked?"
  • YES → english_gap (FORBIDDEN — must be revised)
  • NO, but a real contradiction exists once you know the concept → content_gap (PREFERRED)
  • NO direct contradiction at all, just a wrong-but-plausible alternative → clean (PREFERRED)
  • GENUINELY UNCERTAIN → soft_flag (acceptable, use sparingly)

YOUR DRAFT QUESTION:

STEM:
{stem}

OPTIONS:
{options_block}

INSTRUCTIONS:

Step 1 — Self-classify each distractor. For each non-correct option, decide its class against the rubric above.

Step 2 — If ANY distractor classifies as english_gap, revise the question to eliminate it. You MAY:
  - Rewrite the english_gap distractor's text and explanation to make it content_gap (target a specific concept misunderstanding rather than a lexical contradiction)
  - Rewrite the stem to remove the specific facts that the distractor lexically contradicts (use this when ≥50% of distractors are english_gap, indicating stem over-specification)

ABSOLUTE PRESERVATION RULES (violating these is a quality FAILURE worse than the english_gap you are fixing):

  - The option that is `is_correct: true` in YOUR DRAFT above MUST remain `is_correct: true` in your revision. NEVER swap which option is marked correct. The letter (A/B/C/D) of the correct option MUST stay the same.
  - The correct option's TEXT must answer the new question correctly. If the stem changes, you may update the correct option's text only as needed to maintain factual correctness against the new stem — but the option must still be the right answer.
  - You MUST preserve the Bloom's tier of the question.
  - You MUST keep exactly four options (one correct + three distractors).
  - Each distractor's `is_correct: false` flag must persist.

If you are tempted to move the correct flag to a different option because a distractor's text now seems "more correct," DO NOT. That means you have written a bad distractor. Rewrite the distractor instead.

Step 3 — Output a single JSON object with this exact shape:

{{
  "self_audit": [
    {{"letter": "X", "class": "english_gap|content_gap|clean|soft_flag", "reason": "1 sentence justification"}},
    ...one entry per non-correct option...
  ],
  "revised": true,
  "rationale": "1-2 sentences: what was changed and why (or 'no changes — all distractors classify as content_gap or clean')",
  "question": {{
    "question_stem": "...",
    "options": [
      {{"letter": "A", "text": "...", "is_correct": false, "explanation": "..."}},
      {{"letter": "B", "text": "...", "is_correct": true, "explanation": "..."}},
      {{"letter": "C", "text": "...", "is_correct": false, "explanation": "..."}},
      {{"letter": "D", "text": "...", "is_correct": false, "explanation": "..."}}
    ]
  }}
}}

If no revision is needed (all distractors are content_gap, clean, or soft_flag), set "revised": false but STILL include the question (unchanged) in the output for downstream consumption.

Output ONLY the JSON object. No preamble, no markdown fences, no commentary."""


def _opus_cost(usage: dict) -> float:
    return (
        usage.get("input_tokens", 0) / 1e6 * OPUS_INPUT_PRICE_PER_M
        + usage.get("output_tokens", 0) / 1e6 * OPUS_OUTPUT_PRICE_PER_M
    )


def _sonnet_cost(usage: dict) -> float:
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


async def self_critique_one(
    client, question: dict, semaphore,
) -> tuple[dict, dict, dict]:
    """Run one self-critique call. Returns (revised_question, parsed_response, usage)."""
    stem = question.get("question_stem", "") or ""
    options = question.get("options") or []
    options_block = _build_options_block(options)
    prompt = SELF_CRITIQUE_PROMPT.format(stem=stem, options_block=options_block)

    async with semaphore:
        try:
            # Opus 4.7 deprecated `temperature`; the API rejects the parameter.
            # Self-critique determinism is lost (default sampling); acceptable
            # for a single-anchor test, but worth noting in the journal.
            response = await client.messages.create(
                model=OPUS_MODEL_ID,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            text = response.content[0].text if response.content else ""
            usage = {
                "input_tokens": response.usage.input_tokens,
                "output_tokens": response.usage.output_tokens,
            }
        except Exception as e:
            return question, {"error": f"api_error: {e}"}, {"input_tokens": 0, "output_tokens": 0}

    parsed = parse_response(text)
    if not parsed or "question" not in parsed:
        return question, {"error": "parse_failed", "raw": text[:500]}, usage

    revised_q = dict(question)
    new_q = parsed["question"]
    if "question_stem" in new_q:
        revised_q["question_stem"] = new_q["question_stem"]
    if "options" in new_q and isinstance(new_q["options"], list):
        # Merge new option text/explanation into the existing option records,
        # preserving slot/concept_id/misconception fields the critique may not
        # have specified.
        new_by_letter = {o.get("letter"): o for o in new_q["options"]}
        merged_options = []
        for orig in options:
            letter = orig.get("letter")
            new = new_by_letter.get(letter)
            merged = dict(orig)
            if new:
                if "text" in new:
                    merged["text"] = new["text"]
                if "explanation" in new:
                    merged["explanation"] = new["explanation"]
                if "is_correct" in new:
                    merged["is_correct"] = new["is_correct"]
            merged_options.append(merged)
        revised_q["options"] = merged_options

    revised_q["_self_critique_rationale"] = parsed.get("rationale", "")
    revised_q["_self_audit"] = parsed.get("self_audit", [])
    revised_q["_revised"] = parsed.get("revised", False)
    return revised_q, parsed, usage


def _english_gap_count(audit_results: list[dict]) -> int:
    n = 0
    for r in audit_results:
        if r.get("error"):
            continue
        for c in r.get("classifications") or []:
            if c.get("class") == "english_gap":
                n += 1
    return n


def _summarize_audit(audit_results: list[dict]) -> dict:
    counts = {"english_gap": 0, "content_gap": 0, "clean": 0, "soft_flag": 0, "error": 0}
    for r in audit_results:
        if r.get("error"):
            counts["error"] += 1
            continue
        for c in r.get("classifications") or []:
            cls = c.get("class")
            if cls in counts:
                counts[cls] += 1
    return counts


async def main():
    if not CHAPTER_PATH.exists():
        print(f"ERROR: chapter not found at {CHAPTER_PATH}", file=sys.stderr)
        sys.exit(1)

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CHAPTER_PATH, encoding="utf-8") as fh:
        questions = json.load(fh)

    print(f"Loaded {len(questions)} questions from {CHAPTER_PATH.name}")
    print(f"Anchor: {ANCHOR_UID}")
    print()

    api_key = load_api_key()
    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(4)

    # ── Phase 1: BASELINE AUDIT ────────────────────────────────
    print("=== Phase 1: Baseline audit (Sonnet n_passes=3) ===")
    baseline_audits = await asyncio.gather(*[
        audit_question(client, q, semaphore, n_passes=3)
        for q in questions
    ])
    baseline_summary = _summarize_audit(baseline_audits)
    baseline_eg = _english_gap_count(baseline_audits)
    baseline_audit_cost = sum(_sonnet_cost(r.get("usage", {})) for r in baseline_audits)
    print(f"  classifications: {baseline_summary}")
    print(f"  english_gap distractors: {baseline_eg}")
    print(f"  cost: ${baseline_audit_cost:.4f}")
    print()

    # ── Phase 2: SELF-CRITIQUE per question ────────────────────
    print("=== Phase 2: Self-critique (Opus per question) ===")
    critique_results = await asyncio.gather(*[
        self_critique_one(client, q, semaphore)
        for q in questions
    ])
    revised_questions = [r[0] for r in critique_results]
    parsed_responses = [r[1] for r in critique_results]
    critique_costs = [_opus_cost(r[2]) for r in critique_results]
    critique_total_cost = sum(critique_costs)

    revised_count = sum(1 for q in revised_questions if q.get("_revised"))
    print(f"  questions self-critiqued: {len(revised_questions)}")
    print(f"  revisions applied: {revised_count}")
    print(f"  parse failures: {sum(1 for r in parsed_responses if r.get('error'))}")
    print(f"  cost: ${critique_total_cost:.4f}")
    print()

    # ── Phase 3: REVISED AUDIT ─────────────────────────────────
    print("=== Phase 3: Revised audit (Sonnet n_passes=3) ===")
    revised_audits = await asyncio.gather(*[
        audit_question(client, q, semaphore, n_passes=3)
        for q in revised_questions
    ])
    revised_summary = _summarize_audit(revised_audits)
    revised_eg = _english_gap_count(revised_audits)
    revised_audit_cost = sum(_sonnet_cost(r.get("usage", {})) for r in revised_audits)
    print(f"  classifications: {revised_summary}")
    print(f"  english_gap distractors: {revised_eg}")
    print(f"  cost: ${revised_audit_cost:.4f}")
    print()

    # ── Phase 4: REPORT ────────────────────────────────────────
    print("=" * 60)
    print(f"BASELINE  english_gap: {baseline_eg}")
    print(f"REVISED   english_gap: {revised_eg}")
    delta = revised_eg - baseline_eg
    pct_change = ((revised_eg - baseline_eg) / max(baseline_eg, 1)) * 100
    print(f"DELTA: {delta:+d} ({pct_change:+.0f}%)")
    print()
    total_cost = baseline_audit_cost + critique_total_cost + revised_audit_cost
    print(f"Total test cost: ${total_cost:.4f}")
    print("=" * 60)

    # ── Save artifacts ─────────────────────────────────────────
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    artifact = {
        "test": "test_1_self_critique",
        "timestamp": timestamp,
        "anchor_uid": ANCHOR_UID,
        "chapter_path": str(CHAPTER_PATH),
        "n_questions": len(questions),
        "baseline": {
            "summary": baseline_summary,
            "english_gap_count": baseline_eg,
            "audit_cost_usd": round(baseline_audit_cost, 4),
        },
        "self_critique": {
            "revisions_applied": revised_count,
            "parse_failures": sum(1 for r in parsed_responses if r.get("error")),
            "cost_usd": round(critique_total_cost, 4),
        },
        "revised": {
            "summary": revised_summary,
            "english_gap_count": revised_eg,
            "audit_cost_usd": round(revised_audit_cost, 4),
        },
        "delta_english_gap": delta,
        "pct_change_english_gap": round(pct_change, 1),
        "total_cost_usd": round(total_cost, 4),
    }
    artifact_path = OUT_DIR / "results.json"
    with open(artifact_path, "w", encoding="utf-8") as fh:
        json.dump(artifact, fh, indent=2, ensure_ascii=False)

    # Save per-question detail
    detail = []
    for q, revised_q, parsed, baseline_a, revised_a in zip(
        questions, revised_questions, parsed_responses, baseline_audits, revised_audits,
    ):
        detail.append({
            "question_id": q.get("question_id"),
            "baseline_classes": [
                {"letter": c.get("letter"), "class": c.get("class")}
                for c in (baseline_a.get("classifications") or [])
            ],
            "self_audit": parsed.get("self_audit") if isinstance(parsed, dict) else None,
            "revised": revised_q.get("_revised", False),
            "rationale": revised_q.get("_self_critique_rationale", ""),
            "revised_classes": [
                {"letter": c.get("letter"), "class": c.get("class")}
                for c in (revised_a.get("classifications") or [])
            ],
            "revised_question": {
                "question_stem": revised_q.get("question_stem"),
                "options": [
                    {"letter": o.get("letter"), "text": o.get("text"),
                     "is_correct": o.get("is_correct")}
                    for o in (revised_q.get("options") or [])
                ],
            },
        })
    detail_path = OUT_DIR / "per_question.json"
    with open(detail_path, "w", encoding="utf-8") as fh:
        json.dump(detail, fh, indent=2, ensure_ascii=False)

    print()
    print(f"Artifacts saved to {OUT_DIR}")
    print(f"  - {artifact_path.name}")
    print(f"  - {detail_path.name}")


if __name__ == "__main__":
    asyncio.run(main())
