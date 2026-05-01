"""ship_readiness.py — audit + auto-fix gate before bundling for production.

Phase 18 closes the gap between generation and bundling. Generation emits
chapters into `data/quiz/{DOMAIN}/*.json` with ~4% english_gap distractors
under v1 (~1% under v2 opt-in). The bundle script (`build_quiz_bundle.py`)
ships whatever's there directly to PassEPPP-website. Without an
operational quality gate, those english_gap distractors reach users.

This script is the missing layer:

  Generation (data/quiz/) → ship_readiness → ready/review separation
                                          → bundle reads only ready content

Per chapter:
  1. Run the 3-class audit.
  2. If 0 english_gap → copy original to ship_dir (READY).
  3. If english_gap > 0 and --no-fix → copy original to review_dir (REVIEW).
  4. Else → run --fix to rewrite english_gap distractors → re-audit.
     - If 0 english_gap remaining → copy fixed chapter to ship_dir (READY).
     - Else → copy original to review_dir (REVIEW, fix did not converge).

Manifest emitted at `{ship_dir}/manifest.json` records per-chapter status,
audit timestamps, fix convergence rate, and total cost. Bundle script reads
from ship_dir, refusing to run if manifest is missing or stale.

Idempotent: re-running on unchanged source data is a near-no-op (file
hashes compared against manifest entries; unchanged chapters skip
re-audit).

Usage:
  python scripts/ship_readiness.py
  python scripts/ship_readiness.py --no-fix              # audit-only, no rewrites
  python scripts/ship_readiness.py --workers 8           # higher concurrency
  python scripts/ship_readiness.py --threshold 0.02      # tolerate up to 2% english_gap

Cost: ~$0.014/question audit + ~$0.005/distractor fix. For a 1000-question
batch with ~4% english_gap rate: ~$15 audit + ~$0.20 fix = ~$15.20 total
quality-control overhead.
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import pathlib
import shutil
import sys
import tempfile
from datetime import datetime, timezone

import anthropic

SCRIPT_DIR = pathlib.Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

# Reuse existing audit + fix primitives — no duplication.
from audit_stem_contradictions import (  # noqa: E402
    audit_question, fix_question, load_api_key,
    derive_flagged_from_classifications,
    INPUT_PRICE_PER_M, OUTPUT_PRICE_PER_M,
    MODEL_ID as SONNET_MODEL_ID,
    HAIKU_MODEL_ID, HAIKU_INPUT_PRICE_PER_M, HAIKU_OUTPUT_PRICE_PER_M,
)
# Phase 21b: editorial pass. Optional second offline audit per
# question. Imported lazily here; only invoked when --editorial flag
# is set, so the import happens regardless but the calls don't.
from audit_editorial_quality import audit_editorial_question  # noqa: E402
from pipeline.editorial_rubric import (  # noqa: E402
    CLEAN as EDITORIAL_CLEAN,
    MINOR as EDITORIAL_MINOR,
    MAJOR as EDITORIAL_MAJOR,
)


# ── Config ───────────────────────────────────────────────────

DEFAULT_QUIZ_DIR = REPO_ROOT / "data" / "quiz"
DEFAULT_SHIP_DIR = REPO_ROOT / "data" / "quiz_shippable"
DEFAULT_REVIEW_DIR = REPO_ROOT / "data" / "quiz_review"
DEFAULT_BRIEFS_DIR = REPO_ROOT / "data" / "anchor_briefs"
MANIFEST_FILENAME = "manifest.json"
MANIFEST_VERSION = 1


# ── Helpers ──────────────────────────────────────────────────

def _file_sha256(path: pathlib.Path) -> str:
    """Return hex SHA-256 of file contents. Used for idempotency:
    re-running on unchanged source files skips re-audit."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_chapter(path: pathlib.Path) -> list[dict]:
    """Load a chapter JSON. Tolerates list-shape and {questions: [...]}
    wrappers. Returns [] on error (caller logs)."""
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return []
    if isinstance(d, list):
        return d
    if isinstance(d, dict) and "questions" in d:
        return d["questions"]
    return []


def _write_chapter(path: pathlib.Path, questions: list[dict]) -> None:
    """Write a chapter JSON atomically (temp + rename)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        dir=str(path.parent), prefix=".ship_tmp_", suffix=".json"
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(questions, fh, indent=2, ensure_ascii=False)
        os.replace(tmp, str(path))
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _calc_cost(usage: dict, model_id: str = SONNET_MODEL_ID) -> float:
    """Compute USD cost from a usage dict with input_tokens/output_tokens.

    Phase 21c: dispatches on model_id so Haiku-priced calls (cross-model
    verify) cost-out correctly. Defaults to Sonnet pricing — every
    existing caller (fix, editorial) uses Sonnet.
    """
    if model_id == HAIKU_MODEL_ID:
        return (
            usage.get("input_tokens", 0) / 1e6 * HAIKU_INPUT_PRICE_PER_M
            + usage.get("output_tokens", 0) / 1e6 * HAIKU_OUTPUT_PRICE_PER_M
        )
    return (
        usage.get("input_tokens", 0) / 1e6 * INPUT_PRICE_PER_M
        + usage.get("output_tokens", 0) / 1e6 * OUTPUT_PRICE_PER_M
    )


# ── Per-chapter workflow ────────────────────────────────────

def _reconcile_cross_model(
    sonnet_results: list[dict],
    haiku_results: list[dict],
) -> list[dict]:
    """Phase 21c per-question per-distractor reconciliation between the
    Sonnet quorum and a single Haiku pass.

    Rules:
      - If either model's audit failed entirely (error and no
        classifications): defer to the other side.
      - Per-distractor letter (matched by letter):
          * Agreement → Sonnet's verdict stands.
          * Disagreement involving english_gap → keep english_gap
            (conservative; trust the catch). Mark cross_model_disagreement
            on that entry.
          * Disagreement not involving english_gap → escalate to
            soft_flag with ambiguous_between recording the two classes.
            Mark cross_model_disagreement.
      - Letters Haiku missed (e.g., omitted from its response) keep
        Sonnet's verdict unchanged.

    Returns audit-result dicts in Sonnet's shape, with classifications
    replaced by the reconciled set, flagged_distractors re-derived,
    and a per-result `cross_model_disagreement_count` added.
    """
    reconciled = []
    for s_r, h_r in zip(sonnet_results, haiku_results):
        s_classes = s_r.get("classifications") or []
        h_classes = h_r.get("classifications") or []
        s_failed = bool(s_r.get("error")) and not s_classes
        h_failed = bool(h_r.get("error")) and not h_classes

        # If exactly one side failed, defer to the other entirely.
        if s_failed and not h_failed:
            entry = dict(h_r)
            entry["cross_model_disagreement_count"] = 0
            entry["cross_model_deferred_to"] = "haiku"
            reconciled.append(entry)
            continue
        if h_failed and not s_failed:
            entry = dict(s_r)
            entry["cross_model_disagreement_count"] = 0
            entry["cross_model_deferred_to"] = "sonnet_haiku_failed"
            reconciled.append(entry)
            continue
        if s_failed and h_failed:
            entry = dict(s_r)
            entry["cross_model_disagreement_count"] = 0
            reconciled.append(entry)
            continue

        h_by_letter = {c.get("letter"): c for c in h_classes}
        new_classes = []
        disagreements = 0
        for sc in s_classes:
            letter = sc.get("letter")
            hc = h_by_letter.get(letter)
            if hc is None:
                new_classes.append(sc)
                continue
            s_class = sc.get("class", "")
            h_class = hc.get("class", "")
            if s_class == h_class:
                new_classes.append(sc)
                continue
            # Phase 22b: structural override is deterministic — when a
            # schema-labeling override fired on either side, that verdict
            # is authoritative regardless of model disagreement. Avoid
            # the false-disagreement that would otherwise emit soft_flag
            # on a structurally-resolved case. Sonnet's overridden entry
            # is preferred when both fired (they agree by construction).
            s_struct = sc.get("structural_override") == "schema_labeling"
            h_struct = hc.get("structural_override") == "schema_labeling"
            if s_struct or h_struct:
                winner = sc if s_struct else hc
                new_classes.append(winner)
                continue
            disagreements += 1
            if "english_gap" in (s_class, h_class):
                # Conservative: keep english_gap regardless of which
                # side flagged it. If Sonnet flagged it, keep Sonnet's
                # entry (carries contradicted_stem_fact); else promote
                # using Haiku's entry.
                base = sc if s_class == "english_gap" else hc
                entry = dict(base)
                entry["cross_model_disagreement"] = True
                entry["cross_model_other_class"] = (
                    h_class if s_class == "english_gap" else s_class
                )
                new_classes.append(entry)
            else:
                # Non-english_gap disagreement → soft_flag.
                entry = dict(sc)
                entry["class"] = "soft_flag"
                entry["ambiguous_between"] = sorted({s_class, h_class})
                entry["cross_model_disagreement"] = True
                entry["cross_model_other_class"] = h_class
                # contradicted_stem_fact doesn't apply to soft_flag
                entry.pop("contradicted_stem_fact", None)
                new_classes.append(entry)

        out_entry = dict(s_r)
        out_entry["classifications"] = new_classes
        out_entry["cross_model_disagreement_count"] = disagreements
        out_entry["flagged_distractors"] = derive_flagged_from_classifications(
            {"classifications": new_classes}
        )
        reconciled.append(out_entry)

    return reconciled


async def _audit_chapter(
    client, questions: list[dict], semaphore: asyncio.Semaphore,
    n_passes: int = 1, cross_model_verify: bool = False,
    brief_disc_index: dict[str, list[str]] | None = None,
) -> tuple[list[dict], float]:
    """Run the 3-class audit on every question in a chapter.

    Returns (audit_results, total_cost). Each audit_result dict matches
    the shape returned by audit_stem_contradictions.audit_question
    (classifications, flagged_distractors, usage, error).

    Phase 21a: n_passes > 1 enables multi-pass quorum to reduce
    single-pass jitter on borderline cases. ship_readiness defaults
    to 3 passes; per-pass cost scales linearly.

    Phase 21c: when cross_model_verify, additionally runs a single
    Haiku pass alongside the Sonnet quorum (concurrent with it) and
    reconciles per-distractor. Disagreement on english_gap stays
    english_gap; other disagreements become soft_flag. Cost includes
    Haiku usage at Haiku's pricing.

    Phase 22c: when ``brief_disc_index`` is provided, each question
    is enriched with ``_discriminators`` if its primary anchor has a
    brief on disk. The structural classifier inside ``audit_question``
    reads ``_discriminators`` for Tier-A schema-labeling detection
    (high precision). Questions without briefs fall through to
    Tier-B lexical detection. The enrichment uses fresh dicts so
    the persisted chapter record never carries the transient field.
    """
    if not questions:
        return [], 0.0

    audit_questions = _attach_discriminators(questions, brief_disc_index)

    sonnet_coro = asyncio.gather(*[
        audit_question(client, q, semaphore, n_passes=n_passes)
        for q in audit_questions
    ])

    if not cross_model_verify:
        sonnet_results = await sonnet_coro
        cost = sum(
            _calc_cost(r.get("usage", {}), r.get("model_id", SONNET_MODEL_ID))
            for r in sonnet_results
        )
        return sonnet_results, cost

    haiku_coro = asyncio.gather(*[
        audit_question(client, q, semaphore, n_passes=1,
                       model_id=HAIKU_MODEL_ID)
        for q in audit_questions
    ])
    sonnet_results, haiku_results = await asyncio.gather(
        sonnet_coro, haiku_coro,
    )
    sonnet_cost = sum(
        _calc_cost(r.get("usage", {}), r.get("model_id", SONNET_MODEL_ID))
        for r in sonnet_results
    )
    haiku_cost = sum(
        _calc_cost(r.get("usage", {}), HAIKU_MODEL_ID)
        for r in haiku_results
    )
    reconciled = _reconcile_cross_model(sonnet_results, haiku_results)
    return reconciled, sonnet_cost + haiku_cost


async def _fix_chapter(
    client, questions: list[dict], audit_results: list[dict],
    semaphore: asyncio.Semaphore,
) -> tuple[list[dict], float, dict]:
    """Run --fix on every flagged distractor in the chapter.

    Returns (fixed_questions, total_cost, summary) where summary has
    {fixes_attempted, fixes_succeeded, fixes_failed}.
    """
    fix_tasks = [
        fix_question(client, q, r, semaphore)
        for q, r in zip(questions, audit_results)
    ]
    fix_results = await asyncio.gather(*fix_tasks)

    fixed_questions = [fr["question"] for fr in fix_results]
    cost = sum(_calc_cost(fr.get("usage", {})) for fr in fix_results)
    summary = {
        "fixes_attempted": sum(fr.get("fixes_attempted", 0) for fr in fix_results),
        "fixes_succeeded": sum(fr.get("fixes_applied", 0) for fr in fix_results),
        "fixes_failed": sum(
            len(fr.get("errors") or [])
            for fr in fix_results
        ),
    }
    return fixed_questions, cost, summary


async def _routed_fix_chapter(
    client, questions: list[dict], audit_results: list[dict],
    semaphore: asyncio.Semaphore,
) -> tuple[list[dict], float, dict]:
    """Phase A6: dispatch detector-signature-routed fixers BEFORE the
    legacy fix_question / self_critique path.

    Each audit_result carries `scanner_signals` (the english_gap_scanner's
    per-letter signal dict from A1's manifest projection). For each
    fired signal whose signature has a registered fixer in
    `pipeline/fixers/create_fixer_registry`, dispatch to the routed
    fixer. Questions whose flags are ALL handled by routed fixers can
    skip the downstream legacy path; questions with residual flags fall
    through.

    Returns (patched_questions, total_cost, summary) where summary has:
      - routed_fixes_attempted
      - routed_fixes_applied
      - residual_flagged (count of questions still flagged after routed fix)
      - by_fixer (dict[fixer_id, count])
    """
    from pipeline.fixers import create_fixer_registry
    from pipeline.detectors import (
        DetectorSignal, VERDICT_OVERRIDE_TO,
    )

    fixer_registry = create_fixer_registry()
    patched_questions: list[dict] = []
    by_fixer: dict[str, int] = {}
    attempted = 0
    applied = 0
    residual_flagged = 0

    for q, audit_r in zip(questions, audit_results):
        scanner_signals = audit_r.get("scanner_signals") or {}
        if not scanner_signals:
            patched_questions.append(q)
            if audit_r.get("flagged_distractors"):
                residual_flagged += 1
            continue

        patched = q
        patched_any = False
        for letter, sig_data in scanner_signals.items():
            if not sig_data.get("fired"):
                continue
            signature = sig_data.get("signature")
            fixer = fixer_registry.fixer_for_signature(signature)
            if fixer is None:
                continue

            # Reconstruct a DetectorSignal so the fixer's interface
            # matches A6 production usage. proposed_class="english_gap"
            # because scanner_signals only carry english_gap_scanner data.
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
            attempted += 1
            try:
                new_patched = await fixer.fix(client, patched, sig, semaphore)
            except Exception:
                continue
            if new_patched is patched or new_patched == patched:
                continue
            patched = new_patched
            patched_any = True
            by_fixer[fixer.fixer_id] = by_fixer.get(fixer.fixer_id, 0) + 1

        patched_questions.append(patched)
        if patched_any:
            applied += 1
        elif audit_r.get("flagged_distractors"):
            residual_flagged += 1

    # Cost for routed fixers is hard to track precisely without each
    # fixer reporting its own usage. Estimate conservatively as $0.005
    # per LLM-backed fix (universal_quantifier may invoke Sonnet;
    # laterality and schema_labeling are deterministic; numeric_overlap
    # is deterministic). Future: each fixer could surface usage via the
    # patched question's metadata.
    total_cost = applied * 0.005

    summary = {
        "routed_fixes_attempted": attempted,
        "routed_fixes_applied": applied,
        "residual_flagged": residual_flagged,
        "by_fixer": by_fixer,
    }
    return patched_questions, total_cost, summary


async def _self_critique_chapter(
    client, questions: list[dict], audit_results: list[dict],
    semaphore: asyncio.Semaphore,
) -> tuple[list[dict], float, dict]:
    """Phase 30: alternative to _fix_chapter using whole-question
    Opus self-critique (Phase 25 module).

    Run self-critique only on questions whose audit flagged english_gap;
    pass through unchanged for clean questions.

    Returns (revised_questions, total_cost, summary) where summary has
    {self_critiques_attempted, self_critiques_revised,
    self_critiques_failed}.
    """
    # Lazy import — only loaded when --self-critique flag is set.
    from pipeline.self_critique import (  # noqa: E402
        self_critique_question, opus_cost,
    )

    revised_questions: list[dict] = []
    sc_results: list[dict] = []
    for q, audit_r in zip(questions, audit_results):
        flagged = audit_r.get("flagged_distractors") or []
        if not flagged:
            revised_questions.append(q)
            sc_results.append({
                "question_id": q.get("question_id"),
                "skipped": True,
                "patched": False,
                "usage": {"input_tokens": 0, "output_tokens": 0},
            })
            continue
        result = await self_critique_question(client, q, semaphore)
        revised_questions.append(result.get("question") or q)
        sc_results.append(result)

    total_cost = sum(
        opus_cost(r.get("usage", {})) for r in sc_results
    )
    summary = {
        "self_critiques_attempted": sum(
            1 for r in sc_results if not r.get("skipped")
        ),
        "self_critiques_revised": sum(
            1 for r in sc_results if r.get("patched")
        ),
        "self_critiques_failed": sum(
            1 for r in sc_results
            if r.get("errors") and not r.get("skipped")
        ),
    }
    return revised_questions, total_cost, summary


def _english_gap_count(audit_results: list[dict]) -> int:
    """Count english_gap classifications across audit results."""
    n = 0
    for r in audit_results:
        if r.get("error"):
            continue
        for c in r.get("classifications") or []:
            if c.get("class") == "english_gap":
                n += 1
    return n


def _soft_flag_count(audit_results: list[dict]) -> int:
    """Count soft_flag classifications across audit results.

    Phase 20b: soft_flag is a SEPARATE manifest dimension. It does NOT
    block ship — chapters with soft_flag distractors still route via
    english_gap logic for the ready/review decision. The count surfaces
    in the manifest for optional human review of borderline cases.
    """
    n = 0
    for r in audit_results:
        if r.get("error"):
            continue
        for c in r.get("classifications") or []:
            if c.get("class") == "soft_flag":
                n += 1
    return n


def _schema_labeling_overrides_count(audit_results: list[dict]) -> int:
    """Count schema-labeling structural overrides across audit results.

    Phase 22a: when ``apply_schema_labeling_override`` demotes an
    english_gap classification to content_gap, ``audit_question``
    surfaces a per-question ``schema_labeling_overrides_count``. This
    helper sums them at chapter scope. The counter is advisory — a
    deterministic structural classifier fired and the audit verdict
    was overridden by code, not by prompt. Surfaced in the manifest
    for weekly drift review (if the rate spikes, investigate whether
    the classifier is over-firing).
    """
    return sum(
        r.get("schema_labeling_overrides_count", 0)
        for r in audit_results
        if not r.get("error")
    )


def _english_gap_override_count(audit_results: list[dict]) -> int:
    """Count Phase A2 english_gap structural overrides across audit results.

    When ``apply_english_gap_override`` flips an LLM classification to
    english_gap on a T1/T2 question (because the regex scanner fired a
    high-confidence signature), ``audit_question`` surfaces a per-question
    ``english_gap_override_count``. This helper sums at chapter scope.

    Watch on the manifest: a sudden spike in this count means a chapter
    is producing many T1/T2 stems that lexically conflict with their
    distractors — the gen-time mirror in A3 should catch these earlier.
    A persistent low rate is healthy ("scanner doing its job"); a drop to
    zero may indicate the override wired off.
    """
    return sum(
        r.get("english_gap_override_count", 0)
        for r in audit_results
        if not r.get("error")
    )


def _scanner_flags_count(audit_results: list[dict]) -> int:
    """Count Phase 24 english_gap scanner flags across audit results.

    The scanner is deterministic and runs locally (no API). It detects
    high-confidence english_gap signatures (universal-quantifier +
    specific stem, numeric-ratio mismatch, laterality contradiction,
    stage-timing contradiction). Informational only — does NOT change
    audit verdicts. Triangulates with the LLM audit: when scanner
    fires AND audit also flagged english_gap, high confidence; when
    scanner fires AND audit did NOT flag, the audit may have missed
    a deterministic signature (worth manual review).
    """
    return sum(
        r.get("scanner_flags_count", 0)
        for r in audit_results
        if not r.get("error")
    )


def _scanner_audit_disagreement_count(audit_results: list[dict]) -> int:
    """Count distractors where the scanner flagged english_gap but the
    audit did NOT classify as english_gap. These are audit-blind-spot
    candidates worth investigating.
    """
    n = 0
    for r in audit_results:
        if r.get("error"):
            continue
        scanner = r.get("scanner_signals") or {}
        audit_classes = {
            c.get("letter"): c.get("class")
            for c in (r.get("classifications") or [])
        }
        for letter, sig in scanner.items():
            if sig.get("fired") and audit_classes.get(letter) != "english_gap":
                n += 1
    return n


# ── Phase 22c: brief loading at audit time ──────────────────

def _load_brief_discriminators(briefs_dir: pathlib.Path) -> dict[str, list[str]]:
    """Eagerly enumerate ``briefs_dir/**/*.json`` and return a mapping
    of ``anchor_uid → discriminators``. Skips briefs whose JSON fails
    to parse, lacks a ``uid`` field, or has an empty/missing
    ``discriminators`` list. Returns an empty dict if ``briefs_dir``
    does not exist.

    Phase 22c rationale: ship_readiness historically audited chapters
    in isolation. With the structural classifier in place
    (Phase 22a/b), brief discriminators provide Tier-A precision
    boost when present — at near-zero cost, since briefs are small
    JSON files (<2 KB each) and the inventory is currently 36 briefs
    against 1,567 anchors.
    """
    index: dict[str, list[str]] = {}
    if not briefs_dir.exists():
        return index
    for path in briefs_dir.rglob("*.json"):
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        uid = data.get("uid")
        discs = data.get("discriminators")
        if not uid or not isinstance(discs, list) or not discs:
            continue
        # Keep only string entries
        clean = [d for d in discs if isinstance(d, str) and d.strip()]
        if clean:
            index[uid] = clean
    return index


def _attach_discriminators(
    questions: list[dict],
    brief_disc_index: dict[str, list[str]] | None,
) -> list[dict]:
    """Return a NEW list of question dicts, each with ``_discriminators``
    set when the brief index has discriminators for the question's
    primary anchor_uid. Original ``questions`` list is NOT mutated.

    Underscore prefix on ``_discriminators`` marks the field transient —
    consumers (audit_question / apply_schema_labeling_override) read
    it; persistence layers should never write it. Because this
    function returns NEW dicts, the original ``questions`` retains
    its disk-shape and ``_write_chapter`` writes a clean record.
    """
    if not questions:
        return []
    if not brief_disc_index:
        return list(questions)
    enriched: list[dict] = []
    for q in questions:
        uids = q.get("anchor_uids") or []
        uid = uids[0] if uids else None
        discs = brief_disc_index.get(uid) if uid else None
        if discs:
            q2 = dict(q)
            q2["_discriminators"] = discs
            enriched.append(q2)
        else:
            enriched.append(q)
    return enriched


def _total_distractor_count(questions: list[dict]) -> int:
    """Count actual distractors (non-correct options) across all
    questions in the chapter. Used as the denominator for english_gap
    rate; do NOT hardcode '3' — questions may have variable option
    counts, and bad data may have malformed options."""
    total = 0
    for q in questions or []:
        for o in q.get("options", []) or []:
            if not o.get("is_correct"):
                total += 1
    return max(total, 1)  # avoid divide-by-zero on degenerate input


def _empty_editorial_counts() -> dict:
    return {
        EDITORIAL_CLEAN: 0, EDITORIAL_MINOR: 0,
        EDITORIAL_MAJOR: 0, "error": 0,
    }


async def _editorial_audit(
    client, questions: list[dict], semaphore: asyncio.Semaphore,
) -> tuple[dict, float]:
    """Phase 21b editorial-quality audit on a chapter's questions.

    Returns (class_counts, total_cost_usd) where class_counts is
    {clean, minor, major, error: N}. The audit is offline — it does
    not affect generation. ship_readiness uses this signal both for
    routing (any 'major' overrides ready→review) and manifest
    tracking; the rubric covers style, clinical formatting,
    sensitivity, and structure (orthogonal to english_gap).
    """
    if not questions:
        return _empty_editorial_counts(), 0.0
    tasks = [
        audit_editorial_question(client, q, semaphore) for q in questions
    ]
    results = await asyncio.gather(*tasks)
    counts = _empty_editorial_counts()
    total_cost = 0.0
    for r in results:
        cls = r.get("editorial_class")
        if r.get("error"):
            counts["error"] += 1
        elif cls in (EDITORIAL_CLEAN, EDITORIAL_MINOR, EDITORIAL_MAJOR):
            counts[cls] += 1
        else:
            counts["error"] += 1
        total_cost += _calc_cost(r.get("usage", {}))
    return counts, total_cost


async def _process_chapter(
    client, source_path: pathlib.Path,
    relative_path: str,
    ship_dir: pathlib.Path, review_dir: pathlib.Path,
    semaphore: asyncio.Semaphore,
    allow_fix: bool, threshold: float,
    audit_passes: int = 1,
    editorial_enabled: bool = False,
    cross_model_verify: bool = False,
    brief_disc_index: dict[str, list[str]] | None = None,
    self_critique_enabled: bool = False,
    routed_fixers_enabled: bool = False,
) -> dict:
    """Audit one chapter, optionally fix, then route to ship_dir or
    review_dir. Returns a manifest entry for this chapter.

    Phase 21b: when editorial_enabled, additionally run an offline
    editorial-quality audit on the final question set. Chapters with
    any 'major' editorial finding are routed to review even if their
    english_gap audit was clean — major editorial issues (cultural
    insensitivity, ambiguous compound stems) need human attention
    before ship. Minor issues are advisory; counts surface in the
    manifest but do not block ship.

    Phase 21c: when cross_model_verify, the Sonnet quorum is augmented
    with a Haiku 4.5 pass at every audit point. Per-distractor
    disagreements on english_gap stay english_gap (conservative);
    other disagreements escalate to soft_flag. The reconciled audit
    drives all routing — disagreement counts surface in the manifest
    for advisory review.
    """
    questions = _load_chapter(source_path)
    audited_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
    source_hash = _file_sha256(source_path)

    if not questions:
        entry = {
            "status": "empty",
            "english_gap_caught": 0,
            "english_gap_remaining": 0,
            # Phase 20b parity: every branch must emit soft_flag_count
            # so consumers that read entry["soft_flag_count"] directly
            # (vs entry.get) don't crash on empty chapters.
            "soft_flag_count": 0,
            "fixes_applied": 0,
            "fixes_failed": 0,
            "audit_cost_usd": 0.0,
            "fix_cost_usd": 0.0,
            "source_sha256": source_hash,
            "audited_at": audited_at,
        }
        if editorial_enabled:
            entry["editorial_class_counts"] = _empty_editorial_counts()
            entry["editorial_cost_usd"] = 0.0
        return entry

    # First audit pass
    audit_results, audit_cost = await _audit_chapter(
        client, questions, semaphore, n_passes=audit_passes,
        cross_model_verify=cross_model_verify,
        brief_disc_index=brief_disc_index,
    )
    eg_initial = _english_gap_count(audit_results)
    sf_initial = _soft_flag_count(audit_results)
    parse_errors = sum(1 for r in audit_results if r.get("error"))

    # Default state — overwritten when fix runs
    final_questions = questions
    eg_remaining = eg_initial
    sf_final = sf_initial
    fix_cost = 0.0
    fixes_applied = 0
    fixes_failed = 0
    re_audit_cost = 0.0
    # Phase 21c: track FINAL audit results for disagreement reporting.
    # Initial = audit_results; post-fix path overwrites with re_audit_results.
    final_audit_results = audit_results

    # Phase 1 routing: english_gap-driven decision before editorial
    if parse_errors > 0:
        # Audit fail-OPEN risk: a question whose audit response failed
        # to parse contributes 0 to eg_initial and would silently ship
        # under the threshold check. Treat any parse error as a
        # non-shippable signal regardless of english_gap count.
        status = "review"
        ready_via = None
        review_reason = f"audit_parse_errors={parse_errors}"
    else:
        # Threshold check: 0 english_gap or below tolerance → ship.
        # Denominator = actual distractor count, not hardcoded * 3.
        # Phase 20b: soft_flag is NOT counted toward eg_rate — it's a
        # separate manifest dimension that does not block ship.
        eg_rate = eg_initial / _total_distractor_count(questions)
        if eg_rate <= threshold:
            status = "ready"
            ready_via = "audit_only" if eg_initial == 0 else "tolerance"
            review_reason = None
        elif not allow_fix:
            status = "review"
            ready_via = None
            review_reason = "english_gap_above_threshold_and_no_fix"
        else:
            # Phase A6 routed-fixer pre-pass (when --routed-fixers flag
            # set). Dispatches detector-signature-routed fixers BEFORE
            # the legacy fix path. Routed fixers handle signatured cases
            # (universal_quantifier, laterality, schema_labeling,
            # numeric_overlap) deterministically or with narrow LLM
            # rewrites; questions with residual flags fall through to
            # self_critique / fix_question.
            routed_summary = None
            routed_cost = 0.0
            if routed_fixers_enabled:
                questions, routed_cost, routed_summary = await _routed_fix_chapter(
                    client, questions, audit_results, semaphore,
                )
                fix_cost += routed_cost
                # Re-audit if routed fixers patched anything; gives the
                # legacy path the updated audit state.
                if routed_summary["routed_fixes_applied"] > 0:
                    audit_results, _re_audit_cost = await _audit_chapter(
                        client, questions, semaphore, n_passes=audit_passes,
                        cross_model_verify=cross_model_verify,
                        brief_disc_index=brief_disc_index,
                    )
                    re_audit_cost += _re_audit_cost

            # Phase 30 dispatch: self-critique (Opus, whole-question) OR
            # fix_question (Sonnet, per-distractor). Self-critique handles
            # whole-question issues (stem rewrites, multi-distractor
            # rewrites) that per-distractor fix can't cleanly address.
            if self_critique_enabled:
                final_questions, fix_cost, sc_summary = await _self_critique_chapter(
                    client, questions, audit_results, semaphore,
                )
                fixes_applied = sc_summary["self_critiques_revised"]
                fixes_failed = sc_summary["self_critiques_failed"]
                fix_summary = {
                    "fixes_attempted": sc_summary["self_critiques_attempted"],
                    "fixes_succeeded": sc_summary["self_critiques_revised"],
                    "fixes_failed": sc_summary["self_critiques_failed"],
                    "intervention": "self_critique",
                }
            else:
                final_questions, fix_cost, fix_summary = await _fix_chapter(
                    client, questions, audit_results, semaphore,
                )
                fixes_applied = fix_summary["fixes_succeeded"]
                fixes_failed = fix_summary["fixes_failed"]

            # Re-audit the fixed chapter to verify convergence
            re_audit_results, re_audit_cost = await _audit_chapter(
                client, final_questions, semaphore, n_passes=audit_passes,
                cross_model_verify=cross_model_verify,
                brief_disc_index=brief_disc_index,
            )
            final_audit_results = re_audit_results
            eg_remaining = _english_gap_count(re_audit_results)
            # sf reported from FINAL audit (post-fix). Soft-flagged
            # distractors don't block ship; they surface for human review.
            sf_final = _soft_flag_count(re_audit_results)

            denom = _total_distractor_count(final_questions)
            if eg_remaining == 0 or (eg_remaining / denom) <= threshold:
                status = "ready"
                ready_via = "post_fix"
                review_reason = None
            else:
                status = "review"
                ready_via = None
                review_reason = "fix_did_not_converge"

    # Phase 21b editorial pass — runs when enabled regardless of status
    # so manifest reports complete editorial counts. Audit-target
    # mirrors what's being written to disk: shipping uses
    # final_questions, review uses originals (consistent with the
    # existing pattern that preserves un-modified source for review).
    editorial_counts = _empty_editorial_counts()
    editorial_cost = 0.0
    if editorial_enabled:
        eq_target = final_questions if status == "ready" else questions
        editorial_counts, editorial_cost = await _editorial_audit(
            client, eq_target, semaphore,
        )
        # Major editorial issue overrides ready→review. Minor issues
        # are advisory only — counts surface in manifest, don't block.
        if status == "ready" and editorial_counts[EDITORIAL_MAJOR] > 0:
            status = "review"
            ready_via = None
            review_reason = "editorial_major"

    # Decide output destination + content. Convention:
    # - ready paths write the final (post-fix when applicable) questions
    # - review paths write the ORIGINAL questions (preserve un-modified
    #   content for human inspection)
    # Exception: editorial_major override writes final_questions because
    # those are what would have shipped — the user needs to review the
    # questions whose editorial issue was detected.
    if status == "ready":
        out_path = ship_dir / relative_path
        out_questions = final_questions
    elif review_reason == "editorial_major":
        out_path = review_dir / relative_path
        out_questions = final_questions
    else:
        out_path = review_dir / relative_path
        out_questions = questions
    _write_chapter(out_path, out_questions)

    entry = {
        "status": status,
        "ready_via": ready_via,
        "review_reason": review_reason,
        "english_gap_caught": eg_initial,
        "english_gap_remaining": eg_remaining,
        "soft_flag_count": sf_final,
        # Phase 22a: structural overrides (english_gap → content_gap
        # via schema_labeling) summed across the FINAL audit pass.
        # Always reported when audit ran — the classifier is always-on.
        "schema_labeling_overrides_count": _schema_labeling_overrides_count(
            final_audit_results
        ),
        # Phase A2: english_gap structural overrides at T1/T2.
        # Distinct counter from schema_labeling_overrides — these are
        # cases where the deterministic scanner OVERRODE the LLM's
        # non-english_gap classification on a T1/T2 question.
        "english_gap_override_count": _english_gap_override_count(
            final_audit_results
        ),
        # Phase 24b: deterministic scanner signals. Informational only —
        # does NOT change routing. Always reported (the scanner is local
        # and always-on at $0 marginal cost). scanner_flags_count is the
        # count of distractors flagged by signature; scanner_audit_disagree
        # is the subset where scanner fired but audit did NOT classify
        # english_gap (potential audit blind spots).
        "scanner_flags_count": _scanner_flags_count(final_audit_results),
        "scanner_audit_disagreement_count": _scanner_audit_disagreement_count(
            final_audit_results
        ),
        "fixes_applied": fixes_applied,
        "fixes_failed": fixes_failed,
        "audit_cost_usd": round(audit_cost + re_audit_cost, 4),
        "fix_cost_usd": round(fix_cost, 4),
        "parse_errors": parse_errors,
        "source_sha256": source_hash,
        "audited_at": audited_at,
    }
    if editorial_enabled:
        entry["editorial_class_counts"] = editorial_counts
        entry["editorial_cost_usd"] = round(editorial_cost, 4)
    if cross_model_verify:
        entry["cross_model_disagreement_count"] = sum(
            r.get("cross_model_disagreement_count", 0)
            for r in final_audit_results
        )
    return entry


# ── Idempotency: skip if source unchanged ──────────────────

def _load_existing_manifest(manifest_path: pathlib.Path) -> dict:
    if not manifest_path.exists():
        return {}
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("manifest_version") != MANIFEST_VERSION:
            return {}
        return data
    except Exception:
        return {}


def _is_skip_reusable(source_path: pathlib.Path,
                       previous_entry: dict | None,
                       need_editorial: bool = False,
                       need_cross_model: bool = False) -> bool:
    """Return True if the chapter's hash matches the manifest entry —
    no need to re-audit.

    Phase 21b: editorial_enabled state may change between runs. If
    --editorial is on but the cached entry has no editorial fields, we
    can't reuse it (would leave editorial counts unfilled). If
    --editorial is off but the cached entry was routed to review for
    editorial_major, that decision is no longer valid; re-evaluate.

    Phase 21c: cross_model_verify state changes invalidate the cache
    in either direction. Turning it on requires populating the
    disagreement count; turning it off requires re-evaluating routing
    that may have been driven by Haiku-only english_gap promotions.
    """
    if not previous_entry:
        return False
    if need_editorial and "editorial_class_counts" not in previous_entry:
        return False
    if (not need_editorial
            and previous_entry.get("review_reason") == "editorial_major"):
        return False
    has_cross_model = "cross_model_disagreement_count" in previous_entry
    if need_cross_model and not has_cross_model:
        return False
    if not need_cross_model and has_cross_model:
        return False
    try:
        return previous_entry.get("source_sha256") == _file_sha256(source_path)
    except OSError:
        return False


# ── Main orchestration ──────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--quiz-dir", default=str(DEFAULT_QUIZ_DIR),
        help=f"Source directory (default: {DEFAULT_QUIZ_DIR})",
    )
    parser.add_argument(
        "--ship-dir", default=str(DEFAULT_SHIP_DIR),
        help=f"Ship-ready output directory (default: {DEFAULT_SHIP_DIR})",
    )
    parser.add_argument(
        "--review-dir", default=str(DEFAULT_REVIEW_DIR),
        help=f"Review-needed output directory (default: {DEFAULT_REVIEW_DIR})",
    )
    parser.add_argument(
        "--workers", type=int, default=4,
        help="Concurrent API workers (default: 4)",
    )
    parser.add_argument(
        "--no-fix", action="store_true",
        help="Audit only — don't run --fix on flagged distractors. Chapters "
             "with english_gap above threshold go to review.",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.0,
        help="Maximum english_gap rate to tolerate (default: 0.0 = zero "
             "tolerance). Rate = english_gap_distractors / total_distractors. "
             "Useful for staged rollouts.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-audit all chapters even if source unchanged since last run.",
    )
    parser.add_argument(
        "--audit-passes", type=int, default=3,
        help="Number of audit passes per question (Phase 21a quorum). "
             "Default 3 — multi-pass quorum reduces single-pass jitter on "
             "borderline cases via per-distractor majority vote. Set to 1 "
             "for legacy single-pass behavior. Cost scales linearly.",
    )
    parser.add_argument(
        "--editorial", action="store_true",
        help="Phase 21b: run an offline editorial-quality audit on each "
             "chapter (style, clinical formatting, sensitivity, distractor "
             "structure, stem clarity). Chapters with any 'major' editorial "
             "finding route to review even if english_gap is clean. Adds "
             "~$0.015-0.020/question. Default off.",
    )
    parser.add_argument(
        "--cross-model-verify", action="store_true",
        help="Phase 21c: alongside the Sonnet quorum, run a single Haiku 4.5 "
             "pass per question. Reconciles per-distractor — disagreement on "
             "english_gap stays english_gap (conservative); other "
             "disagreements escalate to soft_flag. Adds ~$0.005/question at "
             "Haiku pricing. Default off.",
    )
    parser.add_argument(
        "--routed-fixers", action="store_true",
        help="Phase A6: dispatch detector-signature-routed fixers BEFORE "
             "the legacy fix path. Universal_quantifier, laterality, "
             "schema_labeling, numeric_overlap, and llm_ambiguity (when "
             "audit-LLM detectors run) get specialized fixers that "
             "preserve invariants better than generic self_critique. "
             "Adds ~$0.005-0.01/Q on signatured cases. Default off.",
    )
    parser.add_argument(
        "--self-critique", action="store_true",
        help="Phase 30: when an audit flags english_gap, use Opus 4.7 "
             "self-critique (whole-question revision with strict "
             "is_correct preservation) instead of the default Sonnet "
             "fix_question (per-distractor surgical rewrites). Adds "
             "~$0.005-0.01/Q on flagged questions only. Best for "
             "stem-over-specification cases that fix_question can't "
             "escape via distractor rewrites alone. Default off.",
    )
    parser.add_argument(
        "--briefs-dir", default=str(DEFAULT_BRIEFS_DIR),
        help="Phase 22c: directory of anchor briefs to load discriminators "
             "from. Briefs are JSON files with `uid` and `discriminators` "
             "fields. When a question's anchor_uid matches a brief, its "
             "discriminators provide Tier-A precision boost to the schema-"
             "labeling structural classifier. Default: data/anchor_briefs/. "
             "Pass --briefs-dir '' to disable brief loading.",
    )
    parser.add_argument(
        "--api-key", default=None, help="Override API key",
    )
    args = parser.parse_args()

    quiz_dir = pathlib.Path(args.quiz_dir).resolve()
    ship_dir = pathlib.Path(args.ship_dir).resolve()
    review_dir = pathlib.Path(args.review_dir).resolve()

    if not quiz_dir.exists():
        print(f"ERROR: source directory does not exist: {quiz_dir}",
              file=sys.stderr)
        sys.exit(1)

    # Find all chapter files
    chapter_paths = []
    for p in sorted(quiz_dir.rglob("*.json")):
        if any(t in p.name for t in
               ("audit", "fixed", "bak", "backup", "_layer_", "_pre_", "manifest")):
            continue
        # relative path from quiz_dir, e.g., "BPSY/foo.json"
        rel = p.relative_to(quiz_dir).as_posix()
        chapter_paths.append((p, rel))

    if not chapter_paths:
        print(f"No chapter files found under {quiz_dir}")
        # Still write an empty manifest so downstream bundle script
        # can detect "ran but nothing to ship."
        ship_dir.mkdir(parents=True, exist_ok=True)
        with open(ship_dir / MANIFEST_FILENAME, "w", encoding="utf-8") as fh:
            json.dump({
                "manifest_version": MANIFEST_VERSION,
                "source_dir": str(quiz_dir),
                "audited_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "chapters": {},
                "summary": {
                    "chapters_total": 0, "chapters_ready": 0, "chapters_review": 0,
                    "english_gap_caught": 0, "english_gap_remaining": 0,
                    "total_cost_usd": 0.0,
                },
            }, fh, indent=2, ensure_ascii=False)
        return

    # Phase 22c: load brief discriminators once per run. Empty
    # briefs_dir, missing path, or "" arg disables brief loading;
    # the structural classifier still runs via Tier-B lexical fallback.
    brief_disc_index: dict[str, list[str]] = {}
    briefs_dir_path: pathlib.Path | None = None
    if args.briefs_dir:
        briefs_dir_path = pathlib.Path(args.briefs_dir).resolve()
        brief_disc_index = _load_brief_discriminators(briefs_dir_path)

    print(f"Source: {quiz_dir}")
    print(f"Ship dir: {ship_dir}")
    print(f"Review dir: {review_dir}")
    print(f"Found {len(chapter_paths)} chapter file(s)")
    if briefs_dir_path is not None:
        print(f"Briefs:   {briefs_dir_path} "
              f"({len(brief_disc_index)} brief(s) with discriminators)")
    fix_mode = "self_critique" if args.self_critique else "fix_question"
    print(f"Mode: {'audit-only' if args.no_fix else f'audit + {fix_mode}'}, "
          f"threshold={args.threshold}, workers={args.workers}, "
          f"audit_passes={args.audit_passes}, "
          f"editorial={'on' if args.editorial else 'off'}, "
          f"cross_model_verify="
          f"{'on (sonnet+haiku)' if args.cross_model_verify else 'off'}")
    print()

    # Load existing manifest for idempotency
    manifest_path = ship_dir / MANIFEST_FILENAME
    previous = _load_existing_manifest(manifest_path)
    previous_chapters = previous.get("chapters") or {}

    # Make output dirs
    ship_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    api_key = args.api_key or load_api_key()
    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(args.workers)

    chapters_manifest: dict[str, dict] = {}
    skipped = 0

    for source_path, rel in chapter_paths:
        prev_entry = previous_chapters.get(rel)
        if not args.force and _is_skip_reusable(
            source_path, prev_entry,
            need_editorial=args.editorial,
            need_cross_model=args.cross_model_verify,
        ):
            # Source unchanged — reuse previous decision
            chapters_manifest[rel] = prev_entry
            skipped += 1
            continue

        print(f"  Processing {rel}...", flush=True)
        entry = await _process_chapter(
            client, source_path, rel,
            ship_dir, review_dir, semaphore,
            allow_fix=not args.no_fix,
            threshold=args.threshold,
            audit_passes=args.audit_passes,
            editorial_enabled=args.editorial,
            cross_model_verify=args.cross_model_verify,
            brief_disc_index=brief_disc_index,
            self_critique_enabled=args.self_critique,
            routed_fixers_enabled=args.routed_fixers,
        )
        chapters_manifest[rel] = entry
        if entry["status"] == "ready":
            status_marker = "[ready]"
        elif entry["status"] == "review":
            status_marker = "[review]"
        else:
            status_marker = "[empty]"
        ed_part = ""
        if args.editorial:
            ec = entry.get("editorial_class_counts") or {}
            ed_part = (
                f" ed_clean={ec.get(EDITORIAL_CLEAN, 0)} "
                f"ed_minor={ec.get(EDITORIAL_MINOR, 0)} "
                f"ed_major={ec.get(EDITORIAL_MAJOR, 0)}"
            )
        cm_part = ""
        if args.cross_model_verify:
            cm_part = (
                f" disagreements="
                f"{entry.get('cross_model_disagreement_count', 0)}"
            )
        chapter_cost = (
            entry.get("audit_cost_usd", 0.0)
            + entry.get("fix_cost_usd", 0.0)
            + entry.get("editorial_cost_usd", 0.0)
        )
        print(
            f"    {status_marker} {rel}: "
            f"eg_caught={entry['english_gap_caught']} "
            f"eg_remaining={entry['english_gap_remaining']} "
            f"fixes={entry.get('fixes_applied', 0)}"
            f"{ed_part}{cm_part} "
            f"cost=${chapter_cost:.4f}",
            flush=True,
        )

    # Aggregate summary
    summary = {
        "chapters_total": len(chapters_manifest),
        "chapters_ready": sum(
            1 for e in chapters_manifest.values() if e["status"] == "ready"
        ),
        "chapters_review": sum(
            1 for e in chapters_manifest.values() if e["status"] == "review"
        ),
        "chapters_empty": sum(
            1 for e in chapters_manifest.values() if e["status"] == "empty"
        ),
        "chapters_skipped_unchanged": skipped,
        "english_gap_caught": sum(
            e.get("english_gap_caught", 0) for e in chapters_manifest.values()
        ),
        "english_gap_remaining": sum(
            e.get("english_gap_remaining", 0) for e in chapters_manifest.values()
        ),
        # Phase 20b: separate dimension. Total soft_flag distractors across
        # the corpus and chapters_with_soft_flag is the count of chapters
        # that surfaced any soft_flag for human review. NEITHER affects
        # ready/review routing — they are advisory metrics.
        "soft_flag_total": sum(
            e.get("soft_flag_count", 0) for e in chapters_manifest.values()
        ),
        "chapters_with_soft_flag": sum(
            1 for e in chapters_manifest.values() if e.get("soft_flag_count", 0) > 0
        ),
        # Phase 22a: deterministic schema-labeling overrides summed
        # across the corpus. Reported whether or not any chapter
        # overrode anything — a zero-line is informative ("classifier
        # ran but found no schema-labeling cases").
        "schema_labeling_overrides_total": sum(
            e.get("schema_labeling_overrides_count", 0)
            for e in chapters_manifest.values()
        ),
        "chapters_with_schema_labeling_override": sum(
            1 for e in chapters_manifest.values()
            if e.get("schema_labeling_overrides_count", 0) > 0
        ),
        # Phase A2: english_gap structural overrides (T1/T2 only) summed
        # across the corpus. Distinct from schema_labeling — these flip
        # toward english_gap, not away from it. Spike → many T1/T2 stems
        # creating lexical conflicts; A3 gen-time gates should reduce.
        "english_gap_overrides_total": sum(
            e.get("english_gap_override_count", 0)
            for e in chapters_manifest.values()
        ),
        "chapters_with_english_gap_override": sum(
            1 for e in chapters_manifest.values()
            if e.get("english_gap_override_count", 0) > 0
        ),
        # Phase 24b: deterministic english_gap scanner signals across
        # the corpus. Informational; does not affect routing. Surfaces
        # both raw flag count and audit-disagreement count for drift
        # detection.
        "scanner_flags_total": sum(
            e.get("scanner_flags_count", 0)
            for e in chapters_manifest.values()
        ),
        "chapters_with_scanner_flag": sum(
            1 for e in chapters_manifest.values()
            if e.get("scanner_flags_count", 0) > 0
        ),
        "scanner_audit_disagreement_total": sum(
            e.get("scanner_audit_disagreement_count", 0)
            for e in chapters_manifest.values()
        ),
        "fixes_applied": sum(
            e.get("fixes_applied", 0) for e in chapters_manifest.values()
        ),
        "fixes_failed": sum(
            e.get("fixes_failed", 0) for e in chapters_manifest.values()
        ),
        "total_cost_usd": round(sum(
            e.get("audit_cost_usd", 0.0)
            + e.get("fix_cost_usd", 0.0)
            + e.get("editorial_cost_usd", 0.0)
            for e in chapters_manifest.values()
        ), 4),
    }
    # Phase 21b editorial summary — only emitted when --editorial was
    # set OR when previous-run cached entries carry editorial data.
    # Major drives ready→review routing; minor is advisory.
    if args.editorial or any(
        "editorial_class_counts" in e for e in chapters_manifest.values()
    ):
        ed_clean = ed_minor = ed_major = ed_error = 0
        chapters_with_major = 0
        ed_cost = 0.0
        for e in chapters_manifest.values():
            ec = e.get("editorial_class_counts") or {}
            ed_clean += ec.get(EDITORIAL_CLEAN, 0)
            ed_minor += ec.get(EDITORIAL_MINOR, 0)
            ed_major += ec.get(EDITORIAL_MAJOR, 0)
            ed_error += ec.get("error", 0)
            if ec.get(EDITORIAL_MAJOR, 0) > 0:
                chapters_with_major += 1
            ed_cost += e.get("editorial_cost_usd", 0.0)
        summary["editorial_total_clean"] = ed_clean
        summary["editorial_total_minor"] = ed_minor
        summary["editorial_total_major"] = ed_major
        summary["editorial_total_errors"] = ed_error
        summary["chapters_with_editorial_major"] = chapters_with_major
        summary["editorial_cost_usd"] = round(ed_cost, 4)
    # Phase 21c cross-model summary — only when --cross-model-verify
    # was set OR when prior cached entries carry the disagreement field.
    # Disagreements are advisory metrics; routing has already been
    # adjusted by the reconciliation logic upstream.
    if args.cross_model_verify or any(
        "cross_model_disagreement_count" in e
        for e in chapters_manifest.values()
    ):
        total_disagreements = sum(
            e.get("cross_model_disagreement_count", 0)
            for e in chapters_manifest.values()
        )
        chapters_with_disagreement = sum(
            1 for e in chapters_manifest.values()
            if e.get("cross_model_disagreement_count", 0) > 0
        )
        summary["cross_model_disagreement_total"] = total_disagreements
        summary["chapters_with_cross_model_disagreement"] = (
            chapters_with_disagreement
        )

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "source_dir": str(quiz_dir),
        "ship_dir": str(ship_dir),
        "review_dir": str(review_dir),
        "audited_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "no_fix": args.no_fix,
            "threshold": args.threshold,
            "workers": args.workers,
            "force": args.force,
            "audit_passes": args.audit_passes,
            "editorial": args.editorial,
            "cross_model_verify": args.cross_model_verify,
        },
        "summary": summary,
        "chapters": chapters_manifest,
    }
    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, ensure_ascii=False)

    # Console summary
    print()
    print("=" * 70)
    print(f"Chapters total:     {summary['chapters_total']}")
    print(f"  ready:            {summary['chapters_ready']}")
    print(f"  review:           {summary['chapters_review']}")
    print(f"  empty:            {summary['chapters_empty']}")
    print(f"  skipped (cached): {summary['chapters_skipped_unchanged']}")
    print(f"english_gap caught:    {summary['english_gap_caught']}")
    print(f"english_gap remaining: {summary['english_gap_remaining']}")
    print(f"soft_flag total:       {summary.get('soft_flag_total', 0)} "
          f"({summary.get('chapters_with_soft_flag', 0)} chapter(s) — "
          "advisory, does not block ship)")
    print(f"schema-label overrides:{summary.get('schema_labeling_overrides_total', 0)} "
          f"({summary.get('chapters_with_schema_labeling_override', 0)} chapter(s) - "
          "structural english_gap -> content_gap demotions)")
    print(f"scanner flags:         {summary.get('scanner_flags_total', 0)} "
          f"({summary.get('chapters_with_scanner_flag', 0)} chapter(s)) - "
          f"audit-disagreement: {summary.get('scanner_audit_disagreement_total', 0)} "
          f"(scanner fired, audit did not catch)")
    if "editorial_total_major" in summary:
        print(f"editorial classes:     "
              f"clean={summary['editorial_total_clean']} "
              f"minor={summary['editorial_total_minor']} "
              f"major={summary['editorial_total_major']} "
              f"error={summary['editorial_total_errors']}")
        print(f"  chapters w/ major:   "
              f"{summary['chapters_with_editorial_major']} "
              f"(blocks ship — routed to review)")
        print(f"  editorial cost:      "
              f"${summary['editorial_cost_usd']:.4f}")
    if "cross_model_disagreement_total" in summary:
        print(f"cross-model verify:    "
              f"{summary['cross_model_disagreement_total']} "
              f"per-distractor disagreement(s) across "
              f"{summary['chapters_with_cross_model_disagreement']} "
              f"chapter(s) — advisory; reconciliation already applied")
    print(f"Fixes applied:         {summary['fixes_applied']}")
    print(f"Fixes failed:          {summary['fixes_failed']}")
    print(f"Total cost:            ${summary['total_cost_usd']:.4f}")
    print(f"Manifest:              {manifest_path}")
    print("=" * 70)

    # Exit code: nonzero if review chapters exist (so CI/automation
    # can detect "needs human attention").
    sys.exit(2 if summary["chapters_review"] > 0 else 0)


if __name__ == "__main__":
    asyncio.run(main())
