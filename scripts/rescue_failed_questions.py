"""rescue_failed_questions.py — recover questions dropped at validation.

The generation pipeline's smart-retry loop runs each question through
the full validation-gate stack. After 2 failed attempts a question is
DROPPED — never written to data/quiz/. The rejected candidates are
preserved in the structured log (logs/*.log) as `validation_fail`
events, but no downstream tooling reads them.

This script reads those events, dispatches each to a per-gate rewrite
strategy, applies the rewrite via Sonnet (temp=0), and emits rescued
questions to data/quiz_rescue/{DOMAIN}/{chapter}.json. After review,
the user can either run ship_readiness against quiz_rescue/ or merge
selected rescues into data/quiz/.

Per-gate strategies:

  apply_identity   — T3 correct option lacks mechanism/criterion
                     marker. Rewrite the CORRECT option to add a
                     causal anchor.
  topic_realm      — distractors drift off the correct option's topic.
                     Rewrite distractors to share concept vocabulary.
  option_claim     — option text contains reasoning markers
                     (because/since/due to). Remove them; preserve
                     claim, move justification to explanation.
  stem_eliminable  — flagged distractor lexically contradicts a stem
                     fact. Use the existing fix_question logic.

Out of scope (low frequency or harder structurally):
  remember_identity (stem reshape) — 3 events
  understand_identity (length trim) — 3 events
  domain_expertise / blooms_cognitive_level / attribution / originality
  — 1 event each

Usage:
  python scripts/rescue_failed_questions.py logs/2026-04-28-quiz.log
  python scripts/rescue_failed_questions.py --workers 4 --dry-run
  python scripts/rescue_failed_questions.py --gates apply_identity,topic_realm

Cost: ~$0.02-0.05 per rescue (one Sonnet rewrite call per question;
stem_eliminable also re-audits to verify content_gap shape).

Output:
  data/quiz_rescue/{DOMAIN}/{chapter}.json   — rescued questions
  logs/rescue_<date>/summary.csv              — per-question outcome
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


# ── Failure parsing ──────────────────────────────────────────

GATE_RE = re.compile(r"^([\w_]+):")


def parse_log(log_path: pathlib.Path) -> dict[str, dict]:
    """Read a structured log file; return dict mapping question_id to
    the most-recent validation_fail event for that question."""
    fails: dict[str, dict] = {}
    if not log_path.exists():
        return fails
    with open(log_path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if event.get("event") != "validation_fail":
                continue
            qid = event.get("question_id")
            if qid:
                fails[qid] = event
    return fails


def gate_id_from_reason(reason: str) -> str:
    """Extract the gate_id from the start of a validation reason."""
    m = GATE_RE.match(reason or "")
    return m.group(1) if m else "unknown"


def tier_from_qid(qid: str) -> int:
    """Tier letter is the second-to-last token: E=1, M=2, H=3, X=4."""
    parts = qid.split("-")
    if len(parts) >= 2:
        letter = parts[-2]
        return {"E": 1, "M": 2, "H": 3, "X": 4}.get(letter, 0)
    return 0


def domain_from_qid(qid: str) -> str:
    """Domain code is the second token of the qid (QZ-{DOMAIN}-...)."""
    parts = qid.split("-")
    return parts[1] if len(parts) > 2 else "?"


# ── Rewrite prompts (one per gate) ──────────────────────────

APPLY_IDENTITY_PROMPT = """You are rewriting the CORRECT option of a multiple-choice question that failed validation. The validator says the correct option lacks a causal anchor — it names an outcome/category but doesn't link to a mechanism or criterion. T3 (Apply) questions require the correct answer to express HOW the determination is made.

VALIDATOR FEEDBACK:
{reason}

QUESTION STEM:
{stem}

CURRENT CORRECT OPTION ({correct_letter}):
{correct_text}

DISTRACTORS (kept as-is for context):
{distractors_block}

YOUR JOB: rewrite ONLY the correct option's `text` so it includes a MECHANISM marker (from/via/through/reflecting/producing/mediated by/with [impaired/reduced] X/by [verb-ing] Y) OR a CRITERION-APPLICATION marker (based on/given that/satisfying/failing/for exceeding the cutoff). Keep the same target concept and answer. Do NOT change the meaning, only add the causal anchor.

CONSTRAINTS:
- Preserve the same conceptual answer — just add the linking phrase.
- text length: similar to current correct option (within ~30 chars).
- text must be a noun phrase or short declarative — no "because"/"since" reasoning markers.

Respond with JSON only:
{{"text": "<rewritten correct option>", "explanation": "<1-2 sentences justifying why this is correct, citing the mechanism/criterion>"}}"""


TOPIC_REALM_PROMPT = """You are rewriting the DISTRACTORS of a multiple-choice question that failed validation. The validator says distractors are off-topic from the correct option's concept area. Distractors should be plausible-but-incorrect claims about the SAME topic, not unrelated alternatives.

VALIDATOR FEEDBACK:
{reason}

QUESTION STEM:
{stem}

CORRECT OPTION ({correct_letter}, kept as-is):
{correct_text}

DISTRACTORS TO REWRITE:
{distractors_block}

YOUR JOB: rewrite each distractor's `text` to engage the same topic vocabulary as the correct option, but apply the topic incorrectly. Each distractor should still target its original misconception type — if you can infer it from the current text, preserve that diagnostic intent. The new distractors must:
1. Use the same concept-domain vocabulary as the correct option.
2. Be wrong via concept-knowledge (content_gap or clean), NOT lexical contradiction with stem facts.
3. Stay similar character length to current distractors (±30 chars each).
4. Have NO reasoning markers (because/since/due to/owing to) in text.

Respond with JSON only:
{{"distractors": [
  {{"letter": "X", "text": "<rewritten>", "explanation": "<1-2 sentences>"}},
  ...
]}}
Include one entry for EACH distractor letter currently present (not the correct option)."""


OPTION_CLAIM_PROMPT = """You are rewriting options of a multiple-choice question that failed validation. The validator detected reasoning markers (because/since/due to/owing to) in option text fields. Option `text` is the CLAIM; the `explanation` field is for justification. Reasoning markers in text mean the claim and its justification have been collapsed.

VALIDATOR FEEDBACK:
{reason}

QUESTION STEM:
{stem}

CORRECT OPTION ({correct_letter}):
  Current text: {correct_text}

DISTRACTORS:
{distractors_block}

YOUR JOB: for EVERY option (correct + distractors), if the current text contains "because", "since", "due to", "owing to", "as a result of", or similar reasoning markers, rewrite the text to be a noun phrase or short declarative claim WITHOUT the marker. Move any justification content into the explanation field. If an option text is already clean, return it unchanged.

CONSTRAINTS:
- Preserve the option's claim — change only the framing, not the meaning.
- text length similar to current (±30 chars).

Respond with JSON only:
{{"options": [
  {{"letter": "A", "text": "<text>", "explanation": "<exp>"}},
  ...
]}}
Include all option letters present in the current question (correct AND distractors), in original order."""


SCOPE_MATCH_PROMPT = """You are rewriting DISTRACTORS of a multiple-choice question that failed validation. The validator says the correct option compares N concepts, but at least one distractor references FEWER concepts (under-scoped). Comparison/best-answer stems require symmetric scope across options — every option (correct AND distractors) should engage the same set of concepts, with distractors getting them wrong.

VALIDATOR FEEDBACK:
{reason}

QUESTION STEM:
{stem}

CORRECT OPTION ({correct_letter}, kept as-is):
{correct_text}

DISTRACTORS TO REWRITE (some may already be in-scope; rewrite the under-scoped ones):
{distractors_block}

YOUR JOB: for each under-scoped distractor (named in the validator feedback), rewrite the text so it ENGAGES the same concept set as the correct option but applies them WRONG. Each rewritten distractor should:
1. Reference the same concepts the correct option compares (so scope is symmetric).
2. Be wrong via concept-knowledge (content_gap or clean), NOT via lexical contradiction with stem facts.
3. Preserve the diagnostic intent — if you can infer the original misconception, keep targeting it.
4. Stay similar character length to the correct option (±40 chars).
5. NO reasoning markers (because/since/due to) in text — those go in explanation.

If a distractor is already in-scope and not flagged, keep it unchanged.

Respond with JSON only:
{{"distractors": [
  {{"letter": "X", "text": "<rewritten>", "explanation": "<1-2 sentences>"}},
  ...
]}}
Include one entry for EACH distractor letter currently present (not the correct option), in original order."""


STEM_KEYWORD_DISTRIBUTION_PROMPT = """You are rewriting DISTRACTORS of a multiple-choice question that failed validation. The validator says the correct option contains stem keywords that NO distractor uses, letting students keyword-match the correct answer without engaging the concept.

VALIDATOR FEEDBACK:
{reason}

QUESTION STEM:
{stem}

CORRECT OPTION ({correct_letter}, kept as-is):
{correct_text}

DISTRACTORS TO REWRITE:
{distractors_block}

YOUR JOB: rewrite distractors so the keywords flagged by the validator are DISTRIBUTED across distractors (used by 1-2 distractors, in contexts that make THOSE distractors wrong). The goal is keyword-matching alone shouldn't identify the correct option.

CONSTRAINTS:
1. Each rewritten distractor should still target a real misconception (don't water down).
2. Use the flagged keywords WHERE THEY MAKE THE DISTRACTOR WRONG — e.g., if the keyword is "general", a distractor might say "the General Memory Index" (wrong index name) or "general intelligence factor" (wrong measurement attribution).
3. Stay similar character length to other distractors (±30 chars).
4. NO reasoning markers (because/since/due to) in text.
5. Preserve the diagnostic intent (don't change which misconception each distractor targets).

If a distractor is already correctly using flagged keywords, keep it unchanged.

Respond with JSON only:
{{"distractors": [
  {{"letter": "X", "text": "<rewritten>", "explanation": "<1-2 sentences>"}},
  ...
]}}
Include one entry for EACH distractor letter currently present (not the correct option), in original order."""


# ── Rewrite calls ───────────────────────────────────────────

async def _sonnet_rewrite(client, prompt: str, semaphore, max_tokens: int = 1024) -> tuple[dict | None, dict, str | None]:
    """Send a rewrite prompt to Sonnet at temp=0; return (parsed, usage, error)."""
    async with semaphore:
        try:
            response = await client.messages.create(
                model=MODEL_ID,
                max_tokens=max_tokens,
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
    if not parsed:
        return None, usage, "parse_error"
    return parsed, usage, None


def _format_distractors(options: list[dict], correct_letter: str) -> str:
    return "\n".join(
        f"  {o.get('letter', '?')}: {o.get('text', '')}"
        for o in options
        if o.get("letter") != correct_letter
    )


def _correct_option(options: list[dict]) -> dict | None:
    for o in options:
        if o.get("is_correct"):
            return o
    return None


def _build_question_from_failure(event: dict) -> dict:
    """Build a question dict suitable for downstream use (audit, save)
    from a validation_fail log event."""
    qid = event.get("question_id", "?")
    return {
        "question_id": qid,
        "domain_code": domain_from_qid(qid),
        "difficulty_tier": tier_from_qid(qid),
        "question_stem": event.get("question_stem", ""),
        "options": event.get("options", []) or [],
        "anchor_uids": [_anchor_from_qid(qid)] if _anchor_from_qid(qid) else [],
        "rescue_metadata": {
            "rescued_from": "validation_fail_log",
            "original_gate": gate_id_from_reason(event.get("reason", "")),
            "original_reason": event.get("reason", ""),
            "rescued_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        },
    }


def _anchor_from_qid(qid: str) -> str | None:
    """Extract the anchor UID portion from a question_id like
    'QZ-PMET-AP-D1-RMS-01-1-H-01'. The anchor portion is everything
    between the third token (AP) and the variant marker (E/M/H/X)."""
    parts = qid.split("-")
    if len(parts) < 5:
        return None
    # Find the index of the variant letter (E/M/H/X) from the right
    for i in range(len(parts) - 1, 2, -1):
        if parts[i] in ("E", "M", "H", "X") and i < len(parts) - 1:
            # anchor portion is parts[3:i] joined back with '-'
            return "-".join(parts[3:i])
    return None


# ── Per-gate dispatch ───────────────────────────────────────

async def rescue_apply_identity(client, event: dict, semaphore) -> tuple[dict | None, dict, str | None]:
    """Rewrite the correct option to add mechanism/criterion language."""
    options = event.get("options", []) or []
    correct = _correct_option(options)
    if not correct:
        return None, {"input_tokens": 0, "output_tokens": 0}, "no_correct_option"
    prompt = APPLY_IDENTITY_PROMPT.format(
        reason=event.get("reason", ""),
        stem=event.get("question_stem", ""),
        correct_letter=correct.get("letter", "?"),
        correct_text=correct.get("text", ""),
        distractors_block=_format_distractors(options, correct.get("letter", "?")),
    )
    parsed, usage, err = await _sonnet_rewrite(client, prompt, semaphore, max_tokens=1024)
    if err or not parsed:
        return None, usage, err or "no_parsed"
    new_text = parsed.get("text")
    new_exp = parsed.get("explanation", correct.get("explanation", ""))
    if not new_text:
        return None, usage, "missing_text_in_response"
    # Build rescued question
    q = _build_question_from_failure(event)
    new_options = []
    for o in options:
        if o.get("letter") == correct.get("letter"):
            new_options.append({**o, "text": new_text, "explanation": new_exp})
        else:
            new_options.append(dict(o))
    q["options"] = new_options
    return q, usage, None


async def rescue_topic_realm(client, event: dict, semaphore) -> tuple[dict | None, dict, str | None]:
    """Rewrite distractors to share topic with the correct option."""
    options = event.get("options", []) or []
    correct = _correct_option(options)
    if not correct:
        return None, {"input_tokens": 0, "output_tokens": 0}, "no_correct_option"
    prompt = TOPIC_REALM_PROMPT.format(
        reason=event.get("reason", ""),
        stem=event.get("question_stem", ""),
        correct_letter=correct.get("letter", "?"),
        correct_text=correct.get("text", ""),
        distractors_block=_format_distractors(options, correct.get("letter", "?")),
    )
    parsed, usage, err = await _sonnet_rewrite(client, prompt, semaphore, max_tokens=2048)
    if err or not parsed:
        return None, usage, err or "no_parsed"
    rewrites = {d.get("letter"): d for d in parsed.get("distractors") or []}
    if not rewrites:
        return None, usage, "missing_distractors_in_response"
    q = _build_question_from_failure(event)
    new_options = []
    for o in options:
        if o.get("letter") == correct.get("letter"):
            new_options.append(dict(o))
            continue
        rw = rewrites.get(o.get("letter"))
        if rw and rw.get("text"):
            new_options.append({
                **o,
                "text": rw["text"],
                "explanation": rw.get("explanation", o.get("explanation", "")),
            })
        else:
            # Strategy missed this letter — preserve original
            new_options.append(dict(o))
    q["options"] = new_options
    return q, usage, None


async def _rescue_with_distractor_rewriting_prompt(
    client, event: dict, semaphore, prompt_template: str,
) -> tuple[dict | None, dict, str | None]:
    """Shared logic for rescue strategies that rewrite distractors via a
    prompt template containing {reason}, {stem}, {correct_letter},
    {correct_text}, {distractors_block} placeholders. Used by
    topic_realm, scope_match, and stem_keyword_distribution."""
    options = event.get("options", []) or []
    correct = _correct_option(options)
    if not correct:
        return None, {"input_tokens": 0, "output_tokens": 0}, "no_correct_option"
    prompt = prompt_template.format(
        reason=event.get("reason", ""),
        stem=event.get("question_stem", ""),
        correct_letter=correct.get("letter", "?"),
        correct_text=correct.get("text", ""),
        distractors_block=_format_distractors(options, correct.get("letter", "?")),
    )
    parsed, usage, err = await _sonnet_rewrite(client, prompt, semaphore, max_tokens=2048)
    if err or not parsed:
        return None, usage, err or "no_parsed"
    rewrites = {d.get("letter"): d for d in parsed.get("distractors") or []}
    if not rewrites:
        return None, usage, "missing_distractors_in_response"
    q = _build_question_from_failure(event)
    new_options = []
    for o in options:
        if o.get("letter") == correct.get("letter"):
            new_options.append(dict(o))
            continue
        rw = rewrites.get(o.get("letter"))
        if rw and rw.get("text"):
            new_options.append({
                **o,
                "text": rw["text"],
                "explanation": rw.get("explanation", o.get("explanation", "")),
            })
        else:
            new_options.append(dict(o))
    q["options"] = new_options
    return q, usage, None


async def rescue_scope_match(client, event: dict, semaphore) -> tuple[dict | None, dict, str | None]:
    """Rewrite under-scoped distractors to engage the same concept set
    as the correct option (with wrong claims)."""
    return await _rescue_with_distractor_rewriting_prompt(
        client, event, semaphore, SCOPE_MATCH_PROMPT,
    )


async def rescue_stem_keyword_distribution(client, event: dict, semaphore) -> tuple[dict | None, dict, str | None]:
    """Rewrite distractors so flagged stem keywords are distributed
    across distractors (where they make the distractor wrong)."""
    return await _rescue_with_distractor_rewriting_prompt(
        client, event, semaphore, STEM_KEYWORD_DISTRIBUTION_PROMPT,
    )


async def rescue_option_claim(client, event: dict, semaphore) -> tuple[dict | None, dict, str | None]:
    """Strip reasoning markers from option texts."""
    options = event.get("options", []) or []
    correct = _correct_option(options)
    if not correct:
        return None, {"input_tokens": 0, "output_tokens": 0}, "no_correct_option"
    prompt = OPTION_CLAIM_PROMPT.format(
        reason=event.get("reason", ""),
        stem=event.get("question_stem", ""),
        correct_letter=correct.get("letter", "?"),
        correct_text=correct.get("text", ""),
        distractors_block=_format_distractors(options, correct.get("letter", "?")),
    )
    parsed, usage, err = await _sonnet_rewrite(client, prompt, semaphore, max_tokens=2048)
    if err or not parsed:
        return None, usage, err or "no_parsed"
    rewrites = {d.get("letter"): d for d in parsed.get("options") or []}
    if not rewrites:
        return None, usage, "missing_options_in_response"
    q = _build_question_from_failure(event)
    new_options = []
    for o in options:
        rw = rewrites.get(o.get("letter"))
        if rw and rw.get("text"):
            new_options.append({
                **o,
                "text": rw["text"],
                "explanation": rw.get("explanation", o.get("explanation", "")),
            })
        else:
            new_options.append(dict(o))
    q["options"] = new_options
    return q, usage, None


async def rescue_stem_eliminable(client, event: dict, semaphore) -> tuple[dict | None, dict, str | None]:
    """Use the existing audit + fix_question pipeline."""
    q = _build_question_from_failure(event)
    # Need to add misconception_type/id default for fix_question
    for o in q["options"]:
        if not o.get("is_correct"):
            o.setdefault("misconception_type", "unknown")
            o.setdefault("misconception_id", o.get("letter", "?"))
    audit_result = await audit_question(client, q, semaphore)
    if audit_result.get("error"):
        return None, audit_result.get("usage", {}), f"audit_error: {audit_result['error']}"
    # If audit no longer flags anything, the original may have been already fine;
    # but we still rescue (apply trivial copy) so it lands in the rescue dir.
    fix_result = await fix_question(client, q, audit_result, semaphore)
    if fix_result.get("errors"):
        # Partial errors are OK as long as we have a patched question
        if not fix_result.get("patched"):
            return None, fix_result.get("usage", {}), "fix_did_not_patch"
    rescued = fix_result.get("question", q)
    total_in = (audit_result.get("usage", {}).get("input_tokens", 0)
                + fix_result.get("usage", {}).get("input_tokens", 0))
    total_out = (audit_result.get("usage", {}).get("output_tokens", 0)
                 + fix_result.get("usage", {}).get("output_tokens", 0))
    return rescued, {"input_tokens": total_in, "output_tokens": total_out}, None


GATE_DISPATCH = {
    "apply_identity": rescue_apply_identity,
    "topic_realm": rescue_topic_realm,
    "option_claim": rescue_option_claim,
    "stem_eliminable_distractor": rescue_stem_eliminable,
    # Phase 20-rescue extension: scope_match and stem_keyword_distribution
    # are both distractor-rewriting strategies (same shape as topic_realm)
    # that share the _rescue_with_distractor_rewriting_prompt helper.
    "scope_match": rescue_scope_match,
    "stem_keyword_distribution": rescue_stem_keyword_distribution,
}


# ── Persistence ─────────────────────────────────────────────

def _calc_cost(usage: dict) -> float:
    return (
        usage.get("input_tokens", 0) / 1e6 * INPUT_PRICE_PER_M
        + usage.get("output_tokens", 0) / 1e6 * OUTPUT_PRICE_PER_M
    )


def _save_rescue(q: dict, out_root: pathlib.Path) -> pathlib.Path:
    """Save a rescued question. Output path:
    {out_root}/{DOMAIN}/{qid}.json  — one file per rescue (no chapter
    grouping; safer for downstream merge review).
    """
    domain = q.get("domain_code", "UNKNOWN")
    qid = q.get("question_id", "unknown")
    safe_qid = re.sub(r"[^A-Za-z0-9_-]", "_", qid)
    domain_dir = out_root / domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    path = domain_dir / f"{safe_qid}.json"
    with open(path, "w", encoding="utf-8") as fh:
        # Wrap as a single-item list so ship_readiness can ingest with
        # the same shape as a chapter file.
        json.dump([q], fh, indent=2, ensure_ascii=False)
    return path


# ── Main ────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "log", nargs="?", default="logs/2026-04-28-quiz.log",
        help="Path to a structured log file (default: logs/2026-04-28-quiz.log)",
    )
    parser.add_argument(
        "--out-dir", default="data/quiz_rescue",
        help="Output directory for rescued questions (default: data/quiz_rescue)",
    )
    parser.add_argument(
        "--gates", default=",".join(GATE_DISPATCH.keys()),
        help="Comma-separated gate IDs to rescue (default: all supported)",
    )
    parser.add_argument("--workers", type=int, default=4, help="Concurrent API workers")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse + report without API calls or writes")
    parser.add_argument("--api-key", default=None, help="Override API key")
    parser.add_argument(
        "--limit", type=int, default=0,
        help="Max questions to rescue (0 = all). Useful for budget-limited test runs.",
    )
    args = parser.parse_args()

    log_path = pathlib.Path(args.log).resolve()
    out_root = pathlib.Path(args.out_dir).resolve()
    selected_gates = set(args.gates.split(","))
    invalid = selected_gates - set(GATE_DISPATCH.keys())
    if invalid:
        print(f"ERROR: unsupported gate(s): {invalid}. "
              f"Supported: {set(GATE_DISPATCH.keys())}", file=sys.stderr)
        sys.exit(1)

    fails = parse_log(log_path)
    if not fails:
        print(f"No validation_fail events found in {log_path}")
        return

    # Filter by selected gates
    targets: list[tuple[str, dict, str]] = []  # (qid, event, gate_id)
    skipped_by_gate: dict[str, int] = {}
    for qid, event in fails.items():
        gate = gate_id_from_reason(event.get("reason", ""))
        if gate not in selected_gates:
            skipped_by_gate[gate] = skipped_by_gate.get(gate, 0) + 1
            continue
        targets.append((qid, event, gate))

    if args.limit:
        targets = targets[:args.limit]

    print(f"Log:          {log_path}")
    print(f"Out dir:      {out_root}")
    print(f"Total fails:  {len(fails)}")
    print(f"Selected:     {len(targets)} (gates: {','.join(sorted(selected_gates))})")
    if skipped_by_gate:
        skip_str = ", ".join(f"{g}:{n}" for g, n in skipped_by_gate.items())
        print(f"Skipped:      {skip_str}")
    print()

    if args.dry_run:
        print("Dry run; would attempt rescue on:")
        for qid, _, gate in targets[:20]:
            print(f"  [{gate}] {qid}")
        if len(targets) > 20:
            print(f"  ... ({len(targets) - 20} more)")
        return

    if not targets:
        return

    api_key = args.api_key or load_api_key()
    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(args.workers)

    async def _worker(qid: str, event: dict, gate: str) -> dict:
        rescuer = GATE_DISPATCH[gate]
        rescued, usage, err = await rescuer(client, event, semaphore)
        outcome = {
            "question_id": qid,
            "gate": gate,
            "cost_usd": round(_calc_cost(usage), 4),
            "error": err,
            "saved_to": None,
        }
        if rescued and not err:
            try:
                path = _save_rescue(rescued, out_root)
                outcome["saved_to"] = str(path)
            except Exception as e:
                outcome["error"] = f"save_error: {e}"
        return outcome

    tasks = [_worker(qid, ev, gate) for qid, ev, gate in targets]
    results = await asyncio.gather(*tasks)

    # Aggregate
    total_cost = sum(r["cost_usd"] for r in results)
    rescued_n = sum(1 for r in results if r.get("saved_to"))
    by_gate: dict[str, list[dict]] = {}
    for r in results:
        by_gate.setdefault(r["gate"], []).append(r)

    # Per-question console summary
    for r in results:
        marker = "[ok]" if r.get("saved_to") else f"[err: {r['error']}]"
        print(f"  {marker:<35} {r['gate']:<26} {r['question_id']}  "
              f"${r['cost_usd']:.4f}")
    print()

    # Roll-up
    print("=" * 70)
    print(f"Total attempts: {len(results)}")
    print(f"Rescued OK:     {rescued_n}")
    print(f"Failed:         {len(results) - rescued_n}")
    print(f"Total cost:     ${total_cost:.4f}")
    print()
    print("By gate:")
    for gate in sorted(by_gate):
        rs = by_gate[gate]
        ok = sum(1 for r in rs if r.get("saved_to"))
        print(f"  {gate:<32} {ok}/{len(rs)} rescued  "
              f"${sum(r['cost_usd'] for r in rs):.4f}")
    print(f"\nRescued questions written under: {out_root}")
    print("Next step: inspect the rescued JSON files, then run "
          "ship_readiness against this directory:")
    print(f"  python scripts/ship_readiness.py --quiz-dir {out_root} "
          f"--ship-dir data/quiz_rescue_shippable --review-dir data/quiz_rescue_review")

    # Write CSV report
    log_dir = pathlib.Path("logs") / f"rescue_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}"
    log_dir.mkdir(parents=True, exist_ok=True)
    csv_path = log_dir / "summary.csv"
    with open(csv_path, "w", encoding="utf-8") as fh:
        fh.write("question_id,gate,cost_usd,error,saved_to\n")
        for r in results:
            fh.write(f"{r['question_id']},{r['gate']},{r['cost_usd']:.4f},"
                     f"{r['error'] or ''},{r['saved_to'] or ''}\n")
    print(f"CSV report: {csv_path}")


if __name__ == "__main__":
    asyncio.run(main())
