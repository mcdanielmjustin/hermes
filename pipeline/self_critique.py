"""Phase 25 — Production self-critique step.

Extracted and hardened from `scripts/diagnosis/test_1_self_critique.py`. Layer 2
in the reinforcer hierarchy: in-loop semantic feedback that has Opus review its
own output against the audit's rubric before commit.

Design contract:
  - Single Opus call per question (~$0.005-0.01 cost; ~17% per-Q overhead at scale).
  - Strict `is_correct` preservation — the option marked correct in the input
    MUST remain correct in the output. The Test-1 false-positive incident
    (Opus silently swapped which option was correct) is structurally prevented
    by an "ABSOLUTE PRESERVATION RULES" block in the prompt.
  - Bloom's-tier preservation.
  - All four options preserved; only text/explanation may change.
  - The critique sees the audit's exact rubric (paste from
    `scripts/audit_stem_contradictions.py:PROMPT` THE FOUR CLASSES section).
  - If parse fails or API errors, the original question is returned unchanged.

Public API:
  - SELF_CRITIQUE_PROMPT: the prompt template.
  - self_critique_question(client, question, semaphore) -> dict
      Returns: {patched: bool, question: dict, usage: dict, errors: list,
                rationale: str, intervention: str}
"""
from __future__ import annotations

import asyncio
from typing import Any

OPUS_MODEL_ID = "claude-opus-4-7"
OPUS_INPUT_PRICE_PER_M = 15.0
OPUS_OUTPUT_PRICE_PER_M = 75.0


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

Step 1 — Self-classify each distractor against the rubric above.

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


def _build_options_block(options: list[dict]) -> str:
    lines = []
    for o in options:
        marker = "[CORRECT]" if o.get("is_correct") else "[distractor]"
        lines.append(f"  {o.get('letter','?')} {marker}: {o.get('text','')}")
    return "\n".join(lines)


def opus_cost(usage: dict) -> float:
    return (
        usage.get("input_tokens", 0) / 1e6 * OPUS_INPUT_PRICE_PER_M
        + usage.get("output_tokens", 0) / 1e6 * OPUS_OUTPUT_PRICE_PER_M
    )


async def self_critique_question(
    client, question: dict, semaphore: asyncio.Semaphore,
) -> dict[str, Any]:
    """Run one self-critique call on the question. Returns a dict matching
    the production fix-result contract:
      {
        "question_id": str,
        "patched": bool,        # whether the revised question differs
        "question": dict,        # the final question (revised or original)
        "usage": dict,           # Opus token usage
        "errors": list[str],
        "rationale": str,        # Opus's explanation of what changed
        "intervention": "self_critique",
      }

    Defensive: on parse failure or API error, returns the ORIGINAL question
    with errors recorded. The is_correct flags are preserved structurally
    (the prompt enforces it; if Opus violates, the merge logic respects the
    ORIGINAL is_correct assignment).
    """
    # Lazy import — caller must have anthropic in scope, but the module
    # itself doesn't directly depend on it. parse_response comes from the
    # audit script.
    import sys
    import pathlib
    repo_root = pathlib.Path(__file__).resolve().parent.parent
    if str(repo_root / "scripts") not in sys.path:
        sys.path.insert(0, str(repo_root / "scripts"))
    from audit_stem_contradictions import parse_response  # noqa: E402

    options = question.get("options") or []
    if not options:
        return {
            "question_id": question.get("question_id", "?"),
            "patched": False,
            "question": question,
            "usage": {"input_tokens": 0, "output_tokens": 0},
            "errors": ["no_options"],
            "rationale": "",
            "intervention": "self_critique",
        }

    stem = question.get("question_stem", "") or ""
    options_block = _build_options_block(options)
    prompt = SELF_CRITIQUE_PROMPT.format(stem=stem, options_block=options_block)

    async with semaphore:
        try:
            # Opus 4.7 deprecates `temperature`; do not pass it.
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
            return {
                "question_id": question.get("question_id", "?"),
                "patched": False,
                "question": question,
                "usage": {"input_tokens": 0, "output_tokens": 0},
                "errors": [f"api_error: {e}"],
                "rationale": "",
                "intervention": "self_critique",
            }

    parsed = parse_response(text)
    if not parsed:
        return {
            "question_id": question.get("question_id", "?"),
            "patched": False,
            "question": question,
            "usage": usage,
            "errors": ["parse_failed"],
            "rationale": "",
            "intervention": "self_critique",
        }

    # If Opus says no revision needed, return original.
    if not parsed.get("revised", False):
        return {
            "question_id": question.get("question_id", "?"),
            "patched": False,
            "question": question,
            "usage": usage,
            "errors": [],
            "rationale": parsed.get("rationale", ""),
            "intervention": "self_critique",
        }

    # Merge revised stem and option text/explanation while preserving
    # is_correct, slot, concept_id, misconception_id, etc. from the
    # original. This is the structural is_correct safeguard — even if
    # Opus violates the prompt, the original is_correct mapping wins.
    new_q = parsed.get("question") or {}
    revised = dict(question)
    if isinstance(new_q.get("question_stem"), str) and new_q["question_stem"].strip():
        revised["question_stem"] = new_q["question_stem"]

    new_options = new_q.get("options") or []
    new_by_letter = {o.get("letter"): o for o in new_options if isinstance(o, dict)}
    merged_options = []
    for orig in options:
        letter = orig.get("letter")
        new = new_by_letter.get(letter) or {}
        merged = dict(orig)
        if isinstance(new.get("text"), str):
            merged["text"] = new["text"]
        if isinstance(new.get("explanation"), str):
            merged["explanation"] = new["explanation"]
        # is_correct is NEVER taken from the new payload — original wins.
        # All other fields (slot, concept_id, misconception_id, etc.)
        # carry over from original.
        merged_options.append(merged)
    revised["options"] = merged_options

    # Trace fields — kept underscore-prefixed so they don't accidentally
    # land in production audit consumers expecting a clean schema.
    revised["_self_critique_rationale"] = parsed.get("rationale", "")
    revised["_self_audit"] = parsed.get("self_audit", [])

    return {
        "question_id": question.get("question_id", "?"),
        "patched": True,
        "question": revised,
        "usage": usage,
        "errors": [],
        "rationale": parsed.get("rationale", ""),
        "intervention": "self_critique",
    }
