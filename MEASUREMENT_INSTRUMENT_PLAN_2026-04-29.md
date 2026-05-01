# Measurement Instrument Frame Shift — Execution Plan

## Context

Diagnosis (DIAGNOSIS_EXECUTION_2026-04-29.md) established that goliath generates **content artifacts** rather than **measurement instruments**. Every reinforcer through Phase 25 polishes how questions LOOK; none addresses whether they FUNCTION as discriminating measurement devices. The 4/10 ceiling is real and empirically measured (path-B pilot: 32% C-or-D rate, 5 factual errors / 50 questions, multiple keyed-answer ambiguities).

User has authorized full execution of the recommended phases plus exploratory programming of measurement instruments. This plan covers seven phases. Phases 26-30 are tactical (build now); Phase 31 is the long-horizon empirical loop (specification only).

The intended outcome: goliath rises from ~6/10 (post Phases 23-25) to ~7-8/10 with measurement-instrument structure + three new audit dimensions, plus a specification for the 9/10 work that requires PassEPPP integration.

## Phase 26a — measurement_target schema + post-hoc inference

**The structural slot.** Each question gains an optional `measurement_target` field that articulates:
- What competency it probes (1 sentence)
- What knowing-vs-non-knowing candidates differ on
- What each distractor's pick reveals about the candidate's knowledge state
- Predicted discrimination level

**Files to create:**
- `pipeline/measurement_target.py` — schema constants + validators + canonical example
- `scripts/diagnosis/infer_measurement_target.py` — runs Opus 4.7 over existing questions; produces measurement_target specs; saves to a sidecar JSON

**Validation:** infer targets for 20 stratified-sample questions; spot-check 5 manually for coherence.

**Cost:** ~$2-3 for sample inference.

## Phase 26b — target-driven generation prototype

**The frame shift in generative direction.** Demonstrate that goliath can take a measurement_target as INPUT and generate an item that probes it. This proves the architecture works in both directions (audit extracts; generation produces).

**Files to create:**
- `scripts/diagnosis/generate_from_target.py` — given (anchor, tier, measurement_target), calls Opus 4.7 with a target-driven prompt; produces the question; runs through goliath's audit + measurement-target validation

**Validation:** generate 4 items (one per tier) for ONE anchor where we've manually authored a measurement_target. Compare to standard generation on the same anchor — does target-driven produce items with stronger discrimination predictions?

**Cost:** ~$3 for the comparison.

## Phase 27 — factual-correctness audit pass

**Catches the 10% factual-error rate** found by path-B pilot. Production-quality audit.

**Files to create:**
- `pipeline/factual_correctness_rubric.py` — rubric + canonical examples
- `scripts/audit_factual_correctness.py` — Sonnet 4.6 pass; checks for factual errors in stem, correct answer text, correct answer explanation, distractor explanations
- `tests/test_factual_correctness_rubric.py` — unit tests

**Validation:** run on the 50-question path-B sample; verify it catches the 5 known factual errors; spot-check 10 audit-clean items for false positives.

**Cost:** ~$0.50 to validate.

## Phase 28 — ambiguity audit pass

**Catches the 16% scoring 2 on ambiguity** + the keyed-distractor-actually-correct cases (PMET X-01, PETH E-01).

**Files to create:**
- `pipeline/ambiguity_rubric.py`
- `scripts/audit_ambiguity.py` — Sonnet 4.6 pass; explicitly tries to argue for each option; flags items where multiple options are defensible
- `tests/test_ambiguity_rubric.py`

**Validation:** run on path-B sample; verify it catches the known ambiguity cases.

**Cost:** ~$0.50 to validate.

## Phase 29 — tier-fit audit pass

**Catches the 60% scope_creep + 8% scope_match failures.**

**Files to create:**
- `pipeline/tier_fit_rubric.py`
- `scripts/audit_tier_fit.py` — Sonnet 4.6 pass; rates whether the cognitive demand of selecting the correct answer matches the labeled Bloom's tier
- `tests/test_tier_fit_rubric.py`

**Validation:** run on path-B sample; spot-check tier mismatches.

**Cost:** ~$0.50 to validate.

## Phase 30 — self-critique production wiring (Phase 25b)

**Wires the existing `pipeline/self_critique` module into ship_readiness as a `--self-critique` flag.** Lets users run self-critique on flagged questions instead of (or before) `fix_question`.

**Files to modify:**
- `scripts/ship_readiness.py` — add `--self-critique` CLI flag; when set, run `self_critique_question` on flagged questions before/instead of fix_question; track per-chapter `self_critique_invocations` in manifest

**Validation:** smoke test on one chapter with the flag set.

**Cost:** ~$2 smoke test.

## Phase 31 — empirical feedback hook spec (defer execution)

**The path to 9/10.** Goliath consumes per-item performance data from PassEPPP and identifies low-discrimination items.

**Specification only this session:**
- Per-item logging schema in PassEPPP: question_id, attempt_id, candidate_anonymized_id, picked_letter, correct_letter, time_seconds, timestamp
- Aggregation pipeline: nightly batch computes per-item stats (p-value, point-biserial correlation, distractor pick distribution)
- Goliath integration: `scripts/import_item_performance.py` reads aggregated stats; flags items with point-biserial < 0.15 OR p-value < 0.10 OR p-value > 0.95 OR distractor pick distribution non-discriminating
- Feedback loop: low-discrimination items routed to review with empirical evidence; rewrite or replace

**Execution: separate session + PassEPPP-side changes (out of scope this session).**

## Execution order + cost

| Phase | Effort | API cost | Status |
|---|---|---|---|
| 26a | 2-3 hours | ~$2-3 | This session |
| 26b | 1-2 hours | ~$3 | This session |
| 27 | 2 hours | ~$0.50 | This session |
| 28 | 2 hours | ~$0.50 | This session |
| 29 | 2 hours | ~$0.50 | This session |
| 30 | 1-2 hours | ~$2 | This session if budget allows |
| 31 | spec only | $0 | This session (doc); execution later |

**Total this session estimate:** ~$10-15 API cost; engineering time fits if executed efficiently.

## Verification (overall)

After all phases, the test is whether running `ship_readiness` on a fresh cohort produces:
1. Per-question manifest entries with measurement_target attached (Phase 26a)
2. Measurement-target validation score per question (Phase 26b infrastructure)
3. Factual-correctness verdict per question (Phase 27)
4. Ambiguity verdict per question (Phase 28)
5. Tier-fit verdict per question (Phase 29)
6. Self-critique invocations tracked when flag set (Phase 30)

The corpus quality picture moves from "english_gap-clean + editorially-clean" (current) to "english_gap-clean + editorially-clean + factually-correct + scope-matched + unambiguous + measurement-target-validated."

## Out of scope (stretch goals for future sessions)

- Anchor → competency mapping (would require curating 1,666 anchors with competency claims)
- Item-set composition (multiple items per competency)
- Curriculum sequencing
- PassEPPP empirical feedback loop (Phase 31 execution)
- Fine-tuning on validated items
