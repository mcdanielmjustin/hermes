"""audit_sentinels.py — pin audit verdicts on canary cases.

The Phase 19 deterministic auditor (temperature=0 + flavor-aware
calibration) catches real english_gap distractors that previous-noise
let through. As we iterate on the audit prompt and calibration in
Phase 20 (schema-labeling exception, soft_flag tier), there's a real
risk that a clause refinement in one place causes an unintended drift
on a different pattern. The Lester/wedding canonical case must stay
english_gap; the BPSY postsynaptic-firing case must stay english_gap;
PMET IV/DV must flip from english_gap (current over-flag) to
content_gap (post-20a).

This test pins those expected verdicts. It runs the live audit at
temperature=0 against five real questions from the corpus and asserts
the per-distractor classification for each sentinel matches the
checkpoint expectation.

Three checkpoints supported:

  baseline   pre-Phase-20a state. All 4 review-chapter english_gap
             distractors are still classified english_gap. Use this
             before applying any 20a changes to verify the auditor is
             at the expected state.
  post_20a   after the schema-labeling exception ships in the audit
             prompt. PMET IV/DV inversions flip to content_gap.
  post_22a   after the deterministic structural classifier
             (pipeline.schema_labeling_classifier) ships. PMET IV/DV
             remains content_gap (now via structural override, Tier B
             lexical match on the canonical IV/DV pair). SOCU/persuasion
             M-01:C (refutational↔supportive swap) flips to content_gap
             via the same mechanism. Negatives (Lester, BPSY postsynaptic,
             CASS MMPI) remain english_gap — universal-quantifier guard
             holds.

Usage:
  python tests/audit_sentinels.py --checkpoint baseline
  python tests/audit_sentinels.py --checkpoint post_20a
  python tests/audit_sentinels.py --checkpoint post_22a
  python tests/audit_sentinels.py --workers 1 --checkpoint post_22a

Exit code: 0 if all sentinels match the checkpoint's expectations,
1 if any drift is detected.

Cost: ~$0.04 per sentinel audit × 5 sentinels = ~$0.20 per run.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import pathlib
import sys
from dataclasses import dataclass

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import anthropic  # noqa: E402

from audit_stem_contradictions import (  # noqa: E402
    audit_question, load_api_key,
)


# ── Sentinel registry ───────────────────────────────────────

@dataclass(frozen=True)
class Sentinel:
    """A pinned-verdict canary for the audit."""
    name: str                           # short identifier
    chapter_path: pathlib.Path           # path under data/quiz/
    question_id: str                    # specific question to audit
    distractor_letter: str              # which option to assert against
    expected: dict[str, str]            # checkpoint -> expected class
    rationale: str                      # one-line reason this sentinel matters


_QUIZ_DIR = REPO_ROOT / "data" / "quiz"

# Empirical observation (2026-04-29): even at temperature=0, Sonnet's
# audit verdict on borderline cases is not perfectly deterministic
# across runs. Two consecutive runs of the same audit on PMET IV/DV B
# and SOCU race-frame produced different classifications (english_gap
# vs clean). Pinning sentinels on those cases would produce flaky
# tests. Soft_flag (Phase 20b) is the architectural answer to this —
# the auditor should be allowed to express uncertainty rather than
# being forced into a binary that flips run-to-run.
#
# For now, sentinels include only STABLE cases (verified consistent
# across multiple runs). Borderline cases are documented in
# `BORDERLINE_NOTES` for tracking but not asserted.

SENTINELS: list[Sentinel] = [
    # 1. BPSY postsynaptic-firing — real english_gap (stem prints "no
    #    measurable change in postsynaptic firing"; distractor says
    #    "exerting its own postsynaptic biological effect"). Stable
    #    across runs. Must remain english_gap — neither the schema
    #    sub-rule nor the structural classifier's Tier-B lexical match
    #    on (presynaptic, postsynaptic) should promote this. The stem
    #    only mentions 'postsynaptic' (not 'presynaptic'), so Tier-B
    #    finds no paired-concept structure; english_gap stands.
    Sentinel(
        name="bpsy_postsynaptic_firing",
        chapter_path=_QUIZ_DIR / "BPSY" /
            "the-language-of-the-brain-neurons-neurotransmitters-and-neur.json",
        question_id="QZ-BPSY-AP-D7-PHY-195-H-01",
        distractor_letter="D",
        expected={
            "baseline": "english_gap",
            "post_20a": "english_gap",
            "post_22a": "english_gap",
        },
        rationale=(
            "Mechanism over-specification (specific finding contradicted) "
            "with no labeled-pair structure. Both audit prompt and "
            "structural classifier must leave this as english_gap."
        ),
    ),

    # 2. CASS MMPI — universal-vs-split contradiction. Stable across
    #    runs. Tests that universal-quantifier signal survives any new
    #    addendum/clause AND survives the structural classifier's
    #    universal-quantifier guard precondition.
    Sentinel(
        name="cass_mmpi_universal",
        chapter_path=_QUIZ_DIR / "CASS" /
            "reading-the-profile-mmpi-2-development-validity-scales-and-c.json",
        question_id="QZ-CASS-AP-D8-PAS-011-X-01",
        distractor_letter="A",
        expected={
            "baseline": "english_gap",
            "post_20a": "english_gap",
            "post_22a": "english_gap",
        },
        rationale=(
            "'Throughout the entire protocol' (universal) contradicts "
            "stated split 'F normal, FB elevated'. Universal-quantifier "
            "precondition (prompt and structural classifier) must hold."
        ),
    ),

    # 3. PMET IV/DV inversion (A) — the core over-flag case the
    #    schema-labeling exception fixed. Stable as english_gap at
    #    baseline. POST_20A flipped via prompt-side sub-rule.
    #    POST_22A: still content_gap, now also caught by the
    #    deterministic Tier-B match on the canonical (IV, DV) pair.
    #    Belt-and-suspenders.
    Sentinel(
        name="pmet_iv_dv_inversion_A",
        chapter_path=_QUIZ_DIR / "PMET" /
            "the-architecture-of-association-how-classical-conditioning-b.json",
        question_id="QZ-PMET-AP-D1-RMS-01-1-M-01",
        distractor_letter="A",
        expected={
            "baseline": "english_gap",
            "post_20a": "content_gap",
            "post_22a": "content_gap",
        },
        rationale=(
            "Stem prints categorized factors + measured outcomes; "
            "distractor swaps IV/DV role. Tier-B lexical match on the "
            "canonical (IV, DV) pair should fire the structural "
            "override deterministically."
        ),
    ),

    # 4. SOCU/persuasion M-01 (C) — Phase 22a positive case. The
    #    distractor swaps which label (refutational/supportive)
    #    attaches to which behaviour. Brief D5-SOC-002-107651e6
    #    carries 'refutational_vs_supportive_defense' as a
    #    discriminator. Must override to content_gap at post_22a.
    Sentinel(
        name="socu_persuasion_m01_refutational_supportive_swap",
        chapter_path=_QUIZ_DIR / "SOCU" /
            "how-we-are-moved-persuasion-social-influence-and-compliance.json",
        question_id="QZ-SOCU-AP-D5-SOC-002-M-01",
        distractor_letter="C",
        expected={
            "baseline": "english_gap",
            "post_20a": "english_gap",  # social_process flavor not yet listed
            "post_22a": "content_gap",
        },
        rationale=(
            "Refutational↔supportive label swap with no universal "
            "quantifier. Tier-B lexical (refutational, supportive) — "
            "and Tier-A once briefs are loaded at audit time (Phase "
            "22c) — should fire structural override."
        ),
    ),
]


# Borderline cases NOT in the active sentinel set but worth tracking.
# These jitter between english_gap and clean across runs; soft_flag
# (Phase 20b) is the architectural answer.
BORDERLINE_NOTES = {
    "pmet_iv_dv_inversion_B": (
        "QZ-PMET-AP-D1-RMS-01-1-M-01:B — 'IVs classify all 5 variables; "
        "DVs describe statistical test family' jitters between english_gap "
        "(matches universal 'all 5 variables together') and clean (the "
        "off-topic 'statistical test family' makes it not directly "
        "rejectable). After 20b, expect soft_flag."
    ),
    "socu_race_frame_denial": (
        "QZ-SOCU-AP-D5-CLI-209-H-01:D — 'unrelated to racial group "
        "membership' jitters between english_gap (frame denial) and "
        "clean (just naming a different analytic frame). After 20b, "
        "expect soft_flag — auditor genuinely uncertain whether this is "
        "lexical or conceptual rejection."
    ),
}


# ── Runner ──────────────────────────────────────────────────

async def _audit_one(client, sentinel: Sentinel, semaphore) -> dict:
    """Audit a single sentinel question; return the classification
    of the target distractor + audit metadata."""
    if not sentinel.chapter_path.exists():
        return {
            "name": sentinel.name,
            "error": f"chapter not found: {sentinel.chapter_path}",
            "actual": None,
        }
    with open(sentinel.chapter_path, encoding="utf-8") as f:
        questions = json.load(f)
    target = next(
        (q for q in questions if q.get("question_id") == sentinel.question_id),
        None,
    )
    if target is None:
        return {
            "name": sentinel.name,
            "error": f"question not found: {sentinel.question_id}",
            "actual": None,
        }

    # Phase 21a: use n_passes=3 for sentinel stability. Borderline cases
    # jitter on single-pass; quorum dampens that.
    audit_result = await audit_question(client, target, semaphore, n_passes=3)
    if audit_result.get("error"):
        return {
            "name": sentinel.name,
            "error": audit_result["error"],
            "actual": None,
        }

    classifications = audit_result.get("classifications") or []
    target_class = next(
        (c.get("class") for c in classifications
         if c.get("letter") == sentinel.distractor_letter),
        None,
    )
    return {
        "name": sentinel.name,
        "actual": target_class,
        "error": None,
    }


async def run_sentinels(checkpoint: str, workers: int, api_key: str | None) -> int:
    """Run all sentinels at temp=0 against the live audit. Return
    exit code (0 if all match expectations, 1 if any drift)."""
    if api_key is None:
        api_key = load_api_key()
    client = anthropic.AsyncAnthropic(api_key=api_key)
    semaphore = asyncio.Semaphore(workers)

    print(f"Sentinel checkpoint: {checkpoint}")
    print(f"Sentinels: {len(SENTINELS)}")
    print()

    tasks = [_audit_one(client, s, semaphore) for s in SENTINELS]
    results = await asyncio.gather(*tasks)

    failures = []
    for sentinel, result in zip(SENTINELS, results):
        expected = sentinel.expected.get(checkpoint)
        if expected is None:
            print(f"  [skip] {sentinel.name}: no expectation at checkpoint '{checkpoint}'")
            continue
        actual = result.get("actual")
        err = result.get("error")
        if err:
            print(f"  [ERROR] {sentinel.name}: {err}")
            failures.append((sentinel, expected, actual, err))
            continue
        match = (actual == expected)
        marker = "[ok]   " if match else "[FAIL] "
        print(f"  {marker} {sentinel.name}: expected={expected}, actual={actual}")
        if not match:
            failures.append((sentinel, expected, actual, None))
            print(f"          why this matters: {sentinel.rationale}")

    print()
    if failures:
        print("=" * 70)
        print(f"FAILED: {len(failures)}/{len(SENTINELS)} sentinels drifted")
        print("=" * 70)
        for s, expected, actual, err in failures:
            print(f"\n  {s.name}")
            print(f"    chapter: {s.chapter_path.name}")
            print(f"    qid:     {s.question_id} letter={s.distractor_letter}")
            print(f"    expected: {expected}, actual: {actual}, err: {err}")
            print(f"    rationale: {s.rationale}")
        return 1

    print(f"All {len(SENTINELS)} sentinels match checkpoint '{checkpoint}'.")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint", choices=("baseline", "post_20a", "post_22a"),
        default="post_22a",
        help="Which checkpoint's expectations to assert against. "
             "Use 'baseline' before Phase 20a ships, 'post_20a' after "
             "the prompt-side schema-labeling clause, 'post_22a' after "
             "the deterministic structural classifier ships.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--api-key", default=None)
    args = parser.parse_args()
    rc = asyncio.run(run_sentinels(args.checkpoint, args.workers, args.api_key))
    sys.exit(rc)


if __name__ == "__main__":
    main()
