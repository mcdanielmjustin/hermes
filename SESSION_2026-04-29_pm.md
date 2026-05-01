# Session — 2026-04-29 PM (Detector-Driven Architecture Pivot + Phase A1)

> Written so a fresh Claude session can pick up the work cold.
> Earlier same-day work documented in `SESSION_2026-04-29.md` (Phase 21 validation morning session). This document covers the evening pivot.

## Active state at session close

**Branch:** `main`. **Working tree:** modified.

**Modified (uncommitted):**
- `pipeline/orchestrator.py` — adds `self.detectors = create_detector_registry()` at `__init__`
- `scripts/audit_stem_contradictions.py` — replaces `_scan_question_for_eg` direct call with `registry.scan_for_phase("audit", question)`

**Created (uncommitted, untracked):**
- `pipeline/detectors/__init__.py` — `Detector` ABC + `DetectorSignal` dataclass + verdict/phase constants
- `pipeline/detectors/registry.py` — `DetectorRegistry` + `create_detector_registry()`
- `pipeline/detectors/english_gap.py` — wraps `english_gap_scanner.scan_question`
- `pipeline/detectors/schema_labeling.py` — wraps `schema_labeling_classifier`
- `pipeline/detectors/laterality.py` — wraps `LateralityIntegrityGate`
- `pipeline/detectors/universal_denial.py` — wraps `UniversalDenialGate`
- `tests/test_detector_registry_parity.py` — 11 parity tests, all passing
- `scripts/diagnosis/anchor_validation_e2e_100.py` — stratified 100-anchor E2E pipeline (cost: $24)
- `scripts/diagnosis/fix_and_export_e2e_100.py` — bad-question fix + export (cost: $9.46 + $5 lost on premature kill)
- Various `data/.diagnosis/measurement_targets/*.json` artifacts

**Pre-existing test failures unrelated to this work:** 2 (verified by `git stash` test). Both predate this session.

**Tasks (TaskList state at close):** all 10 Phase A1 tasks completed.

**Plan file:** `C:\Users\mcdan\.claude\plans\the-former-makes-sense-abstract-diffie.md` — Phase A1 marked DONE; A2–A7 with detailed implementation notes.

## Session arc (chronological)

This session opened with the user reporting their computer froze in the middle of work and asking me to recover the conversation locally. The arc that followed:

1. **Recovery.** Found the frozen session at `~/.claude/projects/C--Users-mcdan/f71e63b6-97ae-452d-b85a-0c41a58e409e.jsonl` (21 MB, last activity 16:35). It had been running an E2E validation on 3 stratified anchors through the measurement-instrument pipeline (Phase 26b target-driven generation + Phase 27/28/29 audits). Last action: ran `scripts/diagnosis/anchor_validation_e2e.py` on 3 anchors; results showed 1 case (CASS T4 D8-PAS-033 WISC age-band) with `dq major (factual=4 ambiguity=2 tier_fit=4)`. The frozen session ended with a menu of next-step options.

2. **Investigation of the D8 ambiguity case.** User asked why it scored so badly. Inspected the artifact at `data/.diagnosis/measurement_targets/e2e_validation_2026-04-29T23-34-33+00-00.json`. Two issues:
   - **Major (the killer):** Option A assigned WAIS-IV to the adolescent. WISC-V (6:0–16:11) and WAIS-IV (16:0–90:11) overlap at 16:0–16:11 — both valid for an adolescent in that band. Option A and keyed Option D were equally defensible.
   - **Minor:** "below the WISC-V floor by one month" was underspecified (could mean 5:11 within WPPSI-IV range OR younger than 2:6 below all instruments).

   Pattern: T4 questions in overlap-zone scenarios accidentally test "what's the most clinically common answer?" rather than the declared competency.

3. **Scaled the E2E validation to 100 stratified anchors.** Built `scripts/diagnosis/anchor_validation_e2e_100.py` with stratified sampling (per_cell=3 across 36 cells = 100 anchors), inline pipeline (infer target → generate from target → english_gap audit + diagnostic_quality audit), checkpointing every 10 anchors. Ran the full 100 in ~30 minutes for $23.81. Results:

   - **91% english_gap clean** (the prompt-driven ceiling — Phase 23+24+25 reinforcers active)
   - **55% dq clean / 35% minor / 8% major**
   - **T4 collapsed:** only 26% dq clean at T4, 22% major
   - Mean scores: factual 4.42, ambiguity **3.97** (lowest), tier_fit 4.58
   - Worst domains: SOCU (3/12 dq clean, 3 majors) and PTHE (5/10, 2 majors)
   - Best domains: BPSY (9/12, 0 majors), CASS, PMET, WDEV (all 0 majors)

   Artifact: `data/.diagnosis/measurement_targets/e2e_100_2026-04-30T00-22-14+00-00.json`.

4. **Fix + export attempt.** User asked to apply the fix pipeline to the 48 bad questions and export to Downloads. Built `scripts/diagnosis/fix_and_export_e2e_100.py` — for each bad question: re-audit eg → run `fix_question` → run `self_critique` → re-audit eg + dq → record before/after.

   First run **crashed on Unicode** (Windows cp1252 console can't encode `→` in print statements; lost ~$0.50–1 worth of API calls). Fixed with `sys.stdout.reconfigure(encoding="utf-8")` + replacing `→` with `->`.

   Second run **was killed by my premature `Stop-Process`** at 19 min based on a too-tight wakeup ("kill if past 17:55"). The kill cost ~$5 — Python died before flushing stdout or writing Downloads files. Lesson: I underestimated runtime; original e2e_100 was 30 min for 100 q, so 48 q should have been ~14 min, but the per-question pipeline has more LLM calls so actual was ~17–19 min.

   Third run (with `python -u` for unbuffered output + per-batch checkpointing) completed cleanly in ~50 min. Final cost $9.46.

5. **Fix outcomes — surprising finding.**
   - 6/48 (13%) now fully clean
   - 15/48 (31%) improved to minor
   - 12/48 (25%) still flagged
   - **15/48 (31%) still major — and dq_major INCREASED 8 → 15** (regression)

   **Mechanism of the regression:** `pipeline/self_critique.py`'s prompt is english_gap-focused. It rewrites distractors to remove lexical contradictions, but in doing so introduces NEW ambiguity that the dq audit catches. Surface fixes break dq invariants.

6. **User confronted the architectural ceiling.** "Is Claude incapable of overcoming this issue? Isn't there a planning document that you made that tells us what to do next?"

   I read `CONTINUATION_2026-04-29.md` (from the prior morning session) and found the answer was already mapped out at lines 53–66. The hierarchy of reinforcers (weakest → strongest): prompt → calibration → **deterministic structural classifier** → **brief-encoded contract** → schema-typed questions. The morning session had explicitly said: "prompts are the weakest reinforcer" and recommended option (3)+(4): "structural classifier for the un-briefed default path; brief contracts where they exist."

   I gave my honest pushback: Claude isn't incapable; *prompt-based reinforcement* is. The 91% english_gap floor is the prompt-driven ceiling. The remaining 9% needs deterministic detection. The user's frustration was well-founded: more prompt iterations were never going to break through.

7. **User proposed "agents with hardcoded skills + orchestrator."** I clarified the framing trap: if "skill" means LLM-with-specialized-prompt, you've just re-introduced the prompt-weakness one level deeper. If "skill" means deterministic Python function, that's exactly the architectural fix. Goliath already has `pipeline/agents.py` (12+ deterministic agents) and `pipeline/orchestrator.py` — extension, not rebuild.

8. **Planning conversation.** Three Explore agents in parallel mapped the existing detector landscape, orchestration flow, and brief structure. Key findings:
   - `pipeline/english_gap_scanner.py` (Phase 24) is **informational only** — its signals don't override LLM verdicts
   - `pipeline/schema_labeling_classifier.py` (Phase 22) DOES override
   - `pipeline/orchestrator.py` and `pipeline/agents.py` are load-bearing (not vestigial)
   - 3 clean insertion points for a unified registry: `orchestrator.py:678` (gen-time), `audit_stem_contradictions.py:791` (audit), `ship_readiness.py:713` (routing)
   - **Brief schema is partially built** — `discriminators` field defined in code but not populated on disk

9. **First plan draft included brief-discriminator backfill.** User pushed back: "we tried using briefs before and the results became more catastrophic the more we continued developing them. look at our activity journal."

   Read `SESSION_2026-04-28.md` D1 (line 1095): "Brief presence reduces gen pass rate by ~25pp at first attempt. Real, controlled, n=12. Mechanism: Opus reproduces concept_explanation richness in the stem (over-specification), creating english_gap traps." Phase 20-revert (2026-04-29) disabled brief content in generation. **Briefs were proven net-negative.**

   Revised plan: dropped brief work entirely. Phase A4 became "extend `LABEL_PAIRS` programmatically by mining the existing corpus" instead of "backfill brief discriminators." Phase A5 became "new structural classifier modules" instead of "brief contracts."

10. **User's second pushback: tier conditionality.** "For apply and analysis questions, reading comprehension becomes tricky naturally. We were supposed to have curated gates per Bloom's level."

    I largely agreed and pushed back on framing: english_gap is tier-blind by definition ("rejectable from stem alone, no domain knowledge needed"); what varies by tier is the *prior* on whether a lexical contradiction is true english_gap vs content_gap-disguised-as-english_gap. At T4, the prior shifts toward content_gap because reading-comprehension-as-test is the natural mode.

    Found the architectural home: `pipeline/distractor_policy.py:95-349` is a `(tier, source_type, stem_pattern)` cell matrix that's currently P0 (all `DEFAULT`). Added Phase A2.5: populate the cell matrix.

11. **Plan approved.** Eight phases (A1, A2, A2.5, A3, A4, A5, A6, A7), four pillars (Detector Registry, Orchestrator Integration, Pattern Coverage via Structural Classifiers (NOT briefs), Tier-Conditional Cell Matrix).

12. **Phase A1 implementation.** Built the unified detector abstraction with bit-identical behavior to today. 10 tasks, 11 new tests, 0 new regressions. Detail in §"Phase A1 — Implementation detail" below.

## Key discoveries

**D1. Prompts are the weakest reinforcer — empirically validated by today's E2E-100.**
After Phase 23+24+25 (the most recent prompt patches), english_gap clean rate is 91%. The remaining 9% won't yield to more prompt patches. This isn't a Claude limitation — it's an architectural ceiling for prompt-driven reinforcement. Deterministic detection that *overrides* the LLM verdict is the only mechanism that breaks through.

**D2. T4 ambiguity is systematic, not anecdotal.**
The morning session had ONE example (D8 WISC age-band overlap). E2E-100 at scale showed it's a pattern: T4 dq clean rate is 26% vs T1 at 73%. Mean ambiguity score is 3.97 (lowest of the three dimensions). T4 majors are at 22%, vs ~4% at lower tiers. The prior session's hypothesis about overlap zones was confirmed across many anchors.

**D3. Self_critique fix pipeline INCREASES dq_majors when run on flagged questions (regression).**
Today's fix-and-export run: pre-fix 8 dq_major / post-fix 15 dq_major. Mechanism: `pipeline/self_critique.py`'s prompt is english_gap-focused. When it rewrites distractors, it doesn't preserve dq invariants (factual correctness, ambiguity, tier-fit). Surface lexical fixes can introduce new overlap-zone phrasing or defensible-alternative ambiguity.

This is the Phase A6 "routed fixers" target: each fixer should fix only what its detector signature targets, leaving everything else alone.

**D4. Briefs at generation time degrade quality (re-confirmed from yesterday's controlled test).**
Yesterday's `SESSION_2026-04-28.md` D1: briefed 8/12 (67%) vs un-briefed 11/12 (92%), n=12, controlled. Mechanism: Opus reproduces brief richness in the stem, creating english_gap traps. Today the user re-emphasized this: "the more we continued developing them, the more catastrophic." The plan now explicitly forbids brief schema expansion / regeneration. Equivalent constraints become detectors.

**D5. Tier-conditional detector behavior is architecturally already provisioned in `distractor_policy.py` but not populated.**
The cell matrix at `pipeline/distractor_policy.py:95-349` is keyed by `(tier, source_type, stem_pattern)` and is currently P0 (every cell falls through to `DEFAULT`). The infrastructure exists; A2.5 populates it. T1/T2 cells: aggressive override (lexical contradiction is more likely true english_gap). T3/T4 cells: conservative override (prior shifts toward content_gap).

**D6. The `_FLAVORS_WITH_SCHEMA_LABELING` curated list is a degraded version of the cell-matrix architecture.**
`pipeline/audit_calibration.py:237-252` lists which flavors get the schema-labeling sub-rule prompt addendum. It's a curated lookup, varies by flavor but not by tier, and lives in prompt space (the weakest reinforcer). The systemic answer is to push this into the cell matrix (deterministic, queryable by `(tier, source_type, stem_pattern)`).

**D7. The detectors needed already exist as scattered modules; A1 is unification, not new code.**
- `pipeline/english_gap_scanner.py` (Phase 24): 5 regex signatures — universal_quantifier (0.85), laterality (0.75), numeric_ratio (0.80), stage_timing (0.65), and a fall-through. Returns `dict[letter, EnglishGapSignal]`.
- `pipeline/schema_labeling_classifier.py` (Phase 22): 3-tier classifier (Tier A brief discriminators conf=1.0; Tier B `LABEL_PAIRS` conf=0.5; universal-quantifier guard). Already produces overrides at audit time.
- `pipeline/gates.py:1734-1923`: `LateralityIntegrityGate` and `UniversalDenialGate` (regex gates that BLOCK at generation time).
- `pipeline/citation_patterns.py`: 5 regex patterns for researcher attributions (consumed by `AttributionGate`).
- `pipeline/distractor_policy.py`: cell matrix infrastructure (P0).

A1 wraps the four primary modules in one interface; the rest stay as gates for now.

**D8. The fix-and-export run had two failure modes worth remembering.**
- Windows cp1252 console can't encode Unicode arrows. Always reconfigure stdout or use ASCII alternatives. (Memory already has this lesson from generate_tables.py; I forgot it.)
- Background jobs piped through `tee` buffer Python's stdout. Even with `flush=True`, log files appear empty until process exits. **Don't kill background jobs based on tee output silence.** Use the checkpoint files as the real progress signal.

**D9. The kill that cost $5 was operator error, not a goliath issue.**
I had set up a wakeup with the user's instruction "kill if past 17:55." When 17:55 arrived, the python process was still running (~19 min in). The original e2e_100 had taken 30 min for 100 q; 48 bad questions had a deeper per-question pipeline (~5 LLM calls each vs the original ~4), so actual runtime was ~17 min. I underestimated. Resilience added in the retry: per-batch checkpoints + `python -u`.

## Architectural pivots made

**Pivot 1: From prompt-driven to deterministic-detector-driven reinforcement.**
The hierarchy from `CONTINUATION_2026-04-29.md` lines 53–66 became the operational principle. Detectors that fire deterministically OVERRIDE the LLM verdict. Prompt-side reinforcers (Phase 23 patches, Phase 21a quorum, Phase 21c cross-model) are downgraded to "belt-and-suspenders" — they remain but no longer carry the load.

**Pivot 2: Brief schema is FROZEN.**
After yesterday's controlled test (briefed 67% vs un-briefed 92%, n=12), no further brief expansion. New patterns get NEW DETECTOR MODULES, not new brief fields. `LABEL_PAIRS` is extended by mining the corpus, not by regenerating briefs. The plan's risk table explicitly forbids "let's also expand briefs" temptation under pressure.

**Pivot 3: Tier-conditional detector behavior centralized in `distractor_policy.py`.**
T1/T2 cells: aggressive override at conf ≥ 0.9 (the prior is true english_gap). T3/T4 cells: conservative override at higher thresholds (≥ 0.95) AND co-firing required (multiple detectors must agree). The cell matrix is the One Source of Truth for "where, when, with what confidence does each detector override the LLM."

**Pivot 4: Routed fixers replace generic `fix_question` for known signatures.**
Today's regression (8 → 15 dq_majors) is the reason. Each fixer touches only what its signature targets. `self_critique` becomes the fallback for unsignatured cases.

## Phase A1 — Implementation detail

**Goal (per plan):** Wrap existing detectors behind one `Detector` ABC. Behavior bit-identical to today.

**Files created:**

| Path | LOC | Role |
|---|---|---|
| `pipeline/detectors/__init__.py` | 153 | `Detector` ABC, `DetectorSignal` dataclass, verdict + phase constants |
| `pipeline/detectors/registry.py` | 158 | `DetectorRegistry` class, `create_detector_registry()` factory |
| `pipeline/detectors/english_gap.py` | 53 | Wraps `english_gap_scanner.scan_question` — emits ADVISORY signals (A2 promotes) |
| `pipeline/detectors/schema_labeling.py` | 110 | Wraps `classify_distractor` — emits OVERRIDE_TO:content_gap on fire |
| `pipeline/detectors/laterality.py` | 73 | Wraps `LateralityIntegrityGate.check` — emits BLOCK at generation |
| `pipeline/detectors/universal_denial.py` | 74 | Wraps `UniversalDenialGate.check` — emits BLOCK at generation |
| `tests/test_detector_registry_parity.py` | 270 | 11 parity + structure tests |

**Files modified:**

| Path | Change |
|---|---|
| `pipeline/orchestrator.py:138-145` | Added `self.detectors = create_detector_registry()` (lazy import inside `__init__`) |
| `scripts/audit_stem_contradictions.py:68-79` | Added imports for `PHASE_AUDIT` and `create_detector_registry`; added lazy `_get_audit_registry()` factory |
| `scripts/audit_stem_contradictions.py:792-815` | Replaced direct `_scan_question_for_eg(question)` with `registry.scan_for_phase(PHASE_AUDIT, question)` filtered to english_gap signals |

**Architectural invariants (A1):**
1. **`DetectorSignal` schema is the universal currency.** Every detector emits these. Fields:
   `detector_id`, `letter`, `fired`, `confidence`, `signature`, `verdict_action` ∈ {BLOCK, OVERRIDE_TO, ADVISORY}, `proposed_class` (only for OVERRIDE_TO), `reason`, `extra` dict.
2. **Phase tags route detectors to insertion points.** PHASE_GENERATION (gen-time gates), PHASE_AUDIT (audit-time deterministic), PHASE_AUDIT_LLM (audit-time LLM-backed, A7).
3. **Behavior bit-identical to pre-A1.** The english_gap registry path emits ADVISORY signals. The actual override at audit time still goes through `apply_schema_labeling_override` (we add the registry path *alongside* until A2 promotes).

**Test results:** 11/11 parity tests pass (TestEnglishGapParity, TestSchemaLabelingParity, TestLateralityParity, TestUniversalDenialParity, TestRegistryStructure). Full goliath test suite: 735/737 pass; 2 pre-existing failures unrelated to this work (verified via `git stash`):
- `test_gate_classification.py::test_every_gate_in_pipeline_is_classified` — `StemEliminableDistractorGate` not categorized in PREREQUISITES/CONTENT_GATES (gates.py maintenance task)
- `test_orchestrator.py::TestLoadAnchorContext::test_returns_brief_data_when_present` — brief loading test, broken on `main`

**Test invocation gotcha:** the test suite uses `import conftest` for sys.path. To run, `cd tests/ && python -m pytest`. Running `python -m pytest tests/` from repo root fails with `ModuleNotFoundError: No module named 'conftest'`.

**Key design decisions made during A1:**
1. **Detector wrappers are thin.** They translate underlying outputs into `DetectorSignal`s but do not duplicate logic. `EnglishGapDetector.scan` just calls `english_gap_scanner.scan_question` and reshapes results.
2. **The registry catches detector exceptions.** A buggy detector can't blow up an entire generation/audit run. Exceptions become synthetic ADVISORY signals on the manifest.
3. **Negative-fire signals are emitted.** `fired=False` advisories let manifests show the detector ran. Useful for telemetry and false-negative analysis.
4. **`apply_schema_labeling_override` stays as the application primitive.** A1 doesn't rip out the existing override application — the registry produces signals; the audit code applies them via the same helper. A2 will rewire.

## Cost roll-up

| Activity | Cost |
|---|---|
| E2E-100 stratified validation | $23.81 |
| Fix-and-export Round 1 (Unicode crash) | ~$0.50–1 lost |
| Fix-and-export Round 2 (premature kill) | ~$5 lost |
| Fix-and-export Round 3 (clean run) | $9.46 |
| Phase A1 implementation | $0 (refactor only) |
| **Total** | **~$39** |

## Where the session ended

User reviewed Phase A1 results (10 tasks complete, 11 parity tests passing, 0 new regressions). User then asked for activity log update + plan update with great detail per phase. This document is the activity log; the plan is at `C:\Users\mcdan\.claude\plans\the-former-makes-sense-abstract-diffie.md`.

**Phase A1 has not been committed.** Working tree is dirty. Recommend committing A1 as one logical unit before starting A2.

Suggested commit message:
```
Phase A1: unify scattered detectors behind registry

Wraps english_gap_scanner, schema_labeling_classifier, LateralityIntegrityGate,
and UniversalDenialGate behind one Detector ABC + DetectorSignal schema. Behavior
bit-identical to pre-A1 (ADVISORY signals; existing override path preserved).

- pipeline/detectors/{__init__,registry,english_gap,schema_labeling,laterality,universal_denial}.py
- pipeline/orchestrator.py: instantiate registry alongside gates
- scripts/audit_stem_contradictions.py: route audit-time scanner call through registry
- tests/test_detector_registry_parity.py: 11 tests verifying registry/direct-call parity
```

## Where to pick up next

**Next phase:** A2 — Promote `english_gap_scanner` from advisory to override on T1/T2 questions.

**Entry point for new session:** read this doc + the plan file (`C:\Users\mcdan\.claude\plans\the-former-makes-sense-abstract-diffie.md`) Phase A2 section. The plan now contains step-by-step implementation notes for A2 including:
- Exact file paths and line numbers
- Signal/verdict logic (when to emit `OVERRIDE_TO:english_gap` vs `ADVISORY`)
- The `apply_english_gap_override` helper to add at `scripts/audit_stem_contradictions.py:784-810`
- Manual-inspection requirement before merging (review every flipped verdict on E2E-100)

**Verification target after A2:** ≥94% english_gap clean on the E2E-100 fixture (T1/T2 lift). T3/T4 stay at current rate until A2.5 ships.

**Re-audit fixture:** `data/.diagnosis/measurement_targets/e2e_100_2026-04-30T00-22-14+00-00.json` — same questions, same seed, just re-run audit through the new override path.

## Files of interest (for resume)

| File | Why |
|---|---|
| `C:\Users\mcdan\.claude\plans\the-former-makes-sense-abstract-diffie.md` | The plan with detailed phase-by-phase implementation notes |
| `goliath/CONTINUATION_2026-04-29.md` | Prior session's diagnosis — the hierarchy of reinforcers (lines 53–66) |
| `goliath/SESSION_2026-04-28.md` | Yesterday's brief-controlled-test (D1, line 1095) — why briefs are out of scope |
| `goliath/SESSION_2026-04-29.md` | Morning Phase 21 validation session |
| `goliath/pipeline/detectors/` | A1 implementation — read `__init__.py` first |
| `goliath/tests/test_detector_registry_parity.py` | A1 acceptance tests |
| `goliath/data/.diagnosis/measurement_targets/e2e_100_2026-04-30T00-22-14+00-00.json` | E2E-100 fixture (verification corpus) |
| `goliath/data/.diagnosis/measurement_targets/e2e_100_checkpoint.json` | Post-run checkpoint |
| `goliath/scripts/audit_stem_contradictions.py:792-815` | Where the registry now plugs in at audit time |

## What this session did NOT touch

- Generation prompt content (Phase 20-revert preserved — `USE_BRIEF_FOR_GENERATION` stays default-off)
- `pipeline/audit_calibration.py:237-252` (`_FLAVORS_WITH_SCHEMA_LABELING`) — left intact as belt-and-suspenders
- The 151-question active corpus in `data/quiz/` — unchanged
- The measurement-instrument pipeline (`scripts/diagnosis/generate_from_target.py` etc.) — stays parallel to ship_readiness for now

## Phase A2 — Implementation detail (added after the activity-log write)

**Goal (per plan):** When the regex scanner fires at confidence ≥ 0.9 AND the LLM disagrees, the scanner verdict wins on T1/T2 questions only. T3/T4 deferred to A2.5.

**Files modified:**
- `pipeline/detectors/english_gap.py` — `EnglishGapDetector.scan()` now emits `verdict_action=OVERRIDE_TO` with `proposed_class="english_gap"` when the signal is fired AND signature ∈ {universal_quantifier, laterality, numeric_ratio} AND tier ∈ {1, 2}; ADVISORY otherwise. `OVERRIDE_ELIGIBLE_SIGNATURES` and `OVERRIDE_ELIGIBLE_TIERS` exported as module constants.
- `pipeline/english_gap_scanner.py` — new `apply_english_gap_override(question, classifications, eg_signals)` helper. Mirrors `apply_schema_labeling_override` pattern. Stamps: `original_class`, `structural_override="english_gap_scanner"`, `structural_override_confidence`, `structural_override_signature`, `structural_override_tier_gate="T1_T2"`.
- `scripts/audit_stem_contradictions.py` — registry call moved BEFORE schema_labeling override (so `eg_signals` is in scope); `apply_english_gap_override` called AFTER schema_labeling; `english_gap_override_count` added to audit return dict.
- `scripts/ship_readiness.py` — `_english_gap_override_count` aggregator; `english_gap_override_count` per chapter; `english_gap_overrides_total` + `chapters_with_english_gap_override` in corpus aggregate.

**Files created:**
- `scripts/diagnosis/reaudit_e2e100_for_phase_a2.py` — deterministic simulation that runs the english_gap scanner on the existing E2E-100 fixture, computes how many overrides A2 would fire (without re-running LLM audit). Cost: $0.
- `tests/test_english_gap_override.py` — 16 tests covering: detector promotion across tiers, override application, schema_labeling × english_gap interaction.

**Test results:** 16/16 A2 tests pass; full suite 751/753 (2 pre-existing failures unchanged).

### A2 empirical finding on E2E-100 (the punchline)

Running `reaudit_e2e100_for_phase_a2.py` on the existing 100-question fixture surfaced a critical signal:

**0 overrides fire at T1/T2.** The corpus's deterministic-detectable english_gap is all at T3/T4. Diagnostic output:

| Tier | Questions | Scanner fires | Already eg | False positives | A2 overrides |
|---|---|---|---|---|---|
| T1 | 26 | 0 | 0 | 0 | 0 |
| T2 | 26 | 0 | 0 | 0 | 0 |
| T3 | 25 | 1 | 1 | 0 | 0 (advisory only at T3) |
| T4 | 23 | 5 | 1 | 3 | 0 (advisory only at T4) |

The 3 T4 false positives are all on the same anchor (`QZ-BPSY-AP-D7-PHY-07-2-X-01`) — a left-stroke vignette where the patient is "right-handed". The scanner's laterality regex fires because the stem contains both "left" and "right", and the distractors mention "left ___ ". This is exactly the kind of surface-pattern noise that A2.5's conservative T4 thresholds (≥ 0.95 + co-firing required) are designed to filter out.

**What this means architecturally:**

1. **The audit (Sonnet quorum, multi-pass) is doing well at catching scanner-detectable cases.** 91% clean rate isn't hiding LLM blind spots on lexical patterns. The scanner agrees with the audit on the cases where audit said "english_gap." The remaining 9% the audit caught are the same patterns the scanner would catch.

2. **A2 ships architecture, not immediate quality lift.** The override mechanism + tier-conditional logic is in place and tested. But on this corpus, T1/T2 overrides count is zero. The plan's projected lift (91% → 94%) requires A2.5 + fix downstream.

3. **A2.5's conservative T4 design is empirically validated.** The right-handed-patient case demonstrates exactly the false-positive failure mode. A naive T4 override would have flipped 3 legitimate content_gap cases to english_gap. With A2.5's higher thresholds + co-firing, these would correctly stay content_gap.

4. **The remaining 9% needs detectors beyond regex.** A5's new structural classifiers (numeric_overlap, etc.) and A7's LLM-backed ambiguity detector are where the next quality lift will come from.

### Where A2 ends

- Phase A2 committed: `e9cfca4` "Phase A2: promote english_gap_scanner from advisory to override at T1/T2"
- Verification artifact at `data/.diagnosis/measurement_targets/a2_override_simulation_2026-04-30T02-47-50+00-00.json` captures the per-tier and per-signature breakdown.
- Working tree clean except for untracked prior-session docs (CONTINUATION_2026-04-29.md, DIAGNOSIS_EXECUTION_2026-04-29.md, REBUILD_HANDOFF.md, SESSION_2026-04-29.md) and untracked diagnostic scripts from earlier today.
- Next phases to pick up:
  1. **S1 (NEW, blocks A2.5) — Cross-cutting scanner improvement: handedness exception in `_check_laterality`.** ~2 hours, $0. Fixes the BPSY false-positive at the scanner level rather than relying on A2.5 co-firing to mask it. See plan's "Cross-cutting scanner improvements" section.
  2. **A2.5 — Tier-conditional cell matrix via `distractor_policy.py`.** Read `~/.claude/plans/the-former-makes-sense-abstract-diffie.md` Phase A2.5 section for step-by-step. The BPSY laterality false positive remains a regression test even after S1 (verify scanner doesn't fire post-S1; cell matrix shouldn't need to filter post-S1).

### Discoveries that adapted the plan (post-A2)

Three discoveries surfaced during A2 implementation that warranted plan changes:

**Plan adaptation 1 — Added cross-cutting scanner improvements section (S1).**
The `_check_laterality` regex doesn't distinguish anatomical laterality ("left hemisphere") from descriptive laterality ("right-handed"). The 3 false positives on the BPSY anchor are scanner-level noise, not a tier-conditional issue. Originally the plan relied on A2.5's co-firing to filter; now S1 fixes the root cause.

**Plan adaptation 2 — Added schema_labeling × english_gap interaction gotcha to A2.5 + A4.**
Universal-quantifier guard prevents the conflict for `universal_quantifier` signature, but laterality/numeric_ratio can still co-fire with schema_labeling Tier B on the same distractor. A2's "english_gap wins" rule could undo correct schema_labeling demotions. Empirical validation needed before A2.5 commits to its co-firing logic.

**Plan adaptation 3 — Verification trajectory revised.**
Original projection: A2 lifts E2E-100 to ≥94%. Empirical reality: 0 T1/T2 overrides, no lift on this corpus. Trajectory updated:
- A2 now: 91% (no change on this corpus)
- A2.5 now carries the first measurable lift: ≥94%
- End-state target adjusted from ≥98% post-A6 to ≥97% post-A6 (more conservative).

## S1 + A2.5 implementation detail (added after A2.5 ship)

### S1 — Scanner: handedness exception (commit `d60f147`)

Added `_HANDEDNESS_RE` matching `(right|left)[-\s](handed|handedness|hander|handedly|dominant)` and a `_strip_handedness()` helper. Inside `_check_laterality`, the stem text is preprocessed to remove handedness phrases before the laterality cross-check.

6 new tests in `tests/test_english_gap_scanner.py` cover: BPSY canonical case (no fire), true laterality flip post-strip (still fires), `left-dominant`, `right-handedness`, space-separated form, and the subtle case where `right-handed` is stripped but a separate `right hemisphere` reference must remain detectable.

Verified on E2E-100: scanner fires dropped from 6 → 3 (only the real universal_quantifier fires on T3/T4 remain; all 3 BPSY false positives gone).

### A2.5 — Tier-conditional cell matrix (commit `c04295b`)

Extended `Cell` dataclass with `override_thresholds: tuple[tuple[str, float], ...]`, `co_firing_required: bool`, `classification_prior: str | None`. Defined `DEFAULT_T1_T2` (threshold 0.75), `DEFAULT_T3` (0.85), `DEFAULT_T4` (0.95 + co_firing_required + classification_prior="content_gap"). `resolve()` falls back to tier-keyed default when no domain-specific cell matches.

`pipeline/detectors/english_gap.py` now consults the cell matrix. The override fires iff `signature in OVERRIDE_ELIGIBLE_SIGNATURES AND confidence >= cell_threshold`.

`apply_english_gap_override` stamping changed: `structural_override_tier_gate="T1_T2"` (A2) → `structural_override_tier=<int>` + `structural_override_cell_threshold=<float>` (A2.5). Carries full per-tier-cell semantics through to the manifest.

20 new cell-matrix tests; 5 A2 tests updated; 2 pre-existing distractor_policy tests updated for the new tier-default fallback.

**Empirical finding**: A2.5 simulation on E2E-100 produces 0 overrides (same as A2). The 3 post-S1 scanner fires (UQ on T3/T4) are all on distractors the audit already classified as english_gap. The architecture is sound; the lift will materialize as A3+ phases come online (gen-time mirror catches what audit misses; A4 extends LABEL_PAIRS for new patterns).

**Co-firing deferred**: `DEFAULT_T4.co_firing_required=True` is in the schema but has no enforcement effect yet. The current english_gap scanner returns ONE signature per (stem, distractor) pair; true co-firing (multiple signatures from one detector on the same letter) requires scanner refactoring. Staged work; A2.5 ships the schema.

### Where this work ends

Commits this session (in order):
- `0cf96f6` Phase A1 — unify detectors behind registry
- `e9cfca4` Phase A2 — promote english_gap_scanner override at T1/T2
- `c651b53` docs: A2 implementation + empirical finding
- `558c150` docs: three plan adaptations
- `d60f147` S1 — scanner handedness exception
- `c04295b` Phase A2.5 — tier-conditional cell matrix

Test totals: 781 passing (16 A2 + 6 S1 + 20 A2.5 added on top of A1's 11). 2 pre-existing failures unchanged throughout.

**Next phase to pick up**: A3 — Mirror detectors at generation-time. The cell-matrix infrastructure is in place; A3 reuses it to gate generation-time output before the audit runs. See plan Phase A3 step-by-step.

### Discoveries that adapted the plan (post-A2.5)

Three more discoveries surfaced during A2.5 implementation that warranted plan changes:

**Plan adaptation 4 — Domain-cell composition bug (tracked as S2).**
A2.5's `resolve()` returns the domain-specific cell directly when one matches. But existing BPSY/PETH cells (authored pre-A2.5) have empty `override_thresholds = ()`. So at every populated domain cell, detector promotion is silently disabled — even when the tier-default would have promoted. Curators didn't intend to opt out; the field just predates them. S2 fixes via composition: domain cell wins on `gate_action`/`correction_strategy`, tier-default wins on `override_thresholds`. ~1 hour fix; lands before A3.

**Plan adaptation 5 — Co-firing enforcement deferred (tracked as S3).**
`co_firing_required=True` on `DEFAULT_T4` has NO effect because `english_gap_scanner.scan_distractor()` returns the FIRST signature that fires. Co-firing within one detector requires scanner refactor (return `list[EnglishGapSignal]`). Staged as S3; ~3 hours work; A2.5 ships the schema, S3 enables it.

**Plan adaptation 6 — T3 threshold semantics clarified.**
`DEFAULT_T3.override_thresholds = (("english_gap_scanner", 0.85),)` means: at T3, only universal_quantifier (conf 0.85 = threshold) overrides. laterality (0.75) and numeric_ratio (0.80) stay advisory. So T3 isn't "conservative threshold across signatures" — it's effectively "UQ-only" promotion. Documented in plan to prevent misreading.

These plan adaptations are captured in the plan file's "Cross-cutting scanner improvements" section (S2, S3).

### S2 implementation (commit `f1a24dc`)

Added `_compose_with_tier_default(domain_cell, tier)` helper in `pipeline/distractor_policy.py`. `resolve()` now composes domain cells with tier-defaults:
- Domain cell wins on `gate_action`, `correction_strategy`, `note` (the original P1+ contract — preserved unchanged)
- Tier-default fills in `override_thresholds` when domain cell uses `()`
- `co_firing_required`: True wins (more conservative; domain cell can opt IN, can't opt OUT of tier-default)
- `classification_prior`: domain cell wins if set; else tier-default's prior fills in

Existing BPSY/PETH cells now correctly inherit detector behavior:
- BPSY T4 cells (4 cells) → threshold 0.95 + co_firing + content_gap prior (from DEFAULT_T4)
- PETH T3 cells (3 cells) → threshold 0.85 (from DEFAULT_T3)
- PETH T4 cells (5 cells) → threshold 0.95 + co_firing + content_gap prior
- BPSY/PETH T2 cells → threshold 0.75 (from DEFAULT_T1_T2)
- BPSY T3 cells → threshold 0.85

5 new tests in `tests/test_distractor_policy_cells.py`: 3 assertions on existing domain cells inheriting correctly, 1 explicit-override-respected case, 1 co_firing propagation, 1 classification_prior precedence.

**Test totals**: 786 passing (+5 from S2). 2 pre-existing failures unchanged throughout.

### A3 implementation (commit `0981e50`)

The english_gap detector now declares `phases = (PHASE_AUDIT, PHASE_GENERATION)`. After the gate loop in `pipeline/orchestrator.py:generate_and_validate`, when `GOLIATH_DETECTORS_AT_GEN=1` env flag is set, the registry runs at gen-time. Fired OVERRIDE_TO/BLOCK signals append to the `failures` list as `("detector:<id>", "<signature> on letter <X>: <reason>")`. The existing retry loop (max_attempts) handles re-prompting.

`pipeline/prompts.py:build_correction_prompt` extended with `_DETECTOR_SIGNATURE_GUIDANCE` map that routes `detector:*` failures to signature-specific guidance (universal_quantifier, laterality, numeric_ratio, stage_timing, schema_labeling). Each entry tells the LLM what to fix and why. Generic fallback for unknown signatures.

9 tests in `tests/test_orchestrator_detector_phase.py`: detector phases declaration, T1/T3/T4 verdict semantics at gen-time (matches audit-time per A2.5 cell-matrix), correction-prompt routing.

**Empirical batch validation deferred**. Plan called for a 20-anchor regen test with the env flag set ($20-50 LLM cost). On the current E2E-100 corpus, the scanner and audit already converge at T1/T2 (zero scanner-detectable cases the audit missed), so a 20-anchor regen would likely produce zero gen-time detector blocks. A wider sweep (100+ anchors) or a known-bad chapter would give cleaner validation.

A3 is therefore "code-complete and tested" but not yet "validated on a real generation batch." The infrastructure is in place behind the env flag for risk-free promotion to default-on once empirical data supports it.

**Test totals**: 795 passing (+9 from A3). 2 pre-existing failures unchanged throughout.

### A4 mining + empirical finding (commit `6873390`)

Built `scripts/diagnosis/mine_label_pairs.py` — deterministic regex scanner over corpus testable_facts/stems/anchor_content_summaries. Six paired-name patterns; auto-filters directional/generic/universal-quantifier noise. Output: `data/.diagnosis/label_pairs_candidates.json` (gitignored).

**Empirical finding**: 151 questions / 35 chapters yielded ONE candidate above freq>=3 (`criteria/predictors`, freq=4) — but inspection showed all 4 occurrences came from a single repeated suppressor-variable context, semantic shape is variable roles not label-swap. **REJECTED**.

Below-threshold pairs (freq=2): zero. The corpus has paired-name variety but insufficient repetition at this scale. Canonical-shape singletons (environmental/genetic, ANOVA/MANOVA, alpha/beta) each appeared once — strong signal of variety, weak signal of any specific pair.

**`LABEL_PAIRS` unchanged** (still 16 + 2 UPPERCASE). No tests added (no code change to schema_labeling_classifier).

A4's value at this scale is **infrastructure shipped**, not pairs added. The mining script is parameterized for re-running on larger corpora (32K-question target). Adding criteria/predictors on weak evidence would have been the wrong call — schema_labeling needs canonical paired-name concepts, not co-occurrence noise.

**Test totals unchanged**: 795 passing. A4 added no tests because no code changes to schema_labeling_classifier (the mining script lives in `scripts/diagnosis/` outside the test suite).

### A5 implementation (commit history continues)

Five new detector modules under `pipeline/detectors/`:

1. **numeric_overlap** — WISC age-band-style overlap detection. Tier-conditional: T2/T3 BLOCK, T4 ADVISORY. Hardcoded WPPSI/WISC/WAIS/WIAT range table.
2. **imperative_lead** — distractors starting with "Identify", "Predict", "Classify", etc. Tier-blind BLOCK.
3. **meta_evaluative** — stems with "best"/"most"/"correctly"/"option". Tier-blind BLOCK.
4. **lead_form_parallelism** — 4 options diverge in grammatical form. ADVISORY only (heuristic).
5. **defensible_alternative** — T4-only stub. ≥2 distractors share ≥3 content words with stem testable_fact. ADVISORY only.

24 new tests in `tests/test_a5_detectors.py`. Registry updated to instantiate all 5.

**Empirical scan on E2E-100 (deterministic, $0)**:

| Detector | Fires | Notes |
|---|---|---|
| lead_form_parallelism | 19 | All advisory; spread across all tiers; potentially noisy |
| defensible_alternative | 6 | T4 only (sensible — T4 ambiguity is the target) |
| meta_evaluative | 1 | T3, would have BLOCKED at gen-time |
| imperative_lead | 0 | Goliath's prompt already prevents this |
| numeric_overlap | 0 | D8 case past 240-char stem truncation in artifact |

**Honest read**: 1 real gen-time block fires on the existing corpus (meta_evaluative T3). Advisories are informational only. The detectors are guard rails for FUTURE generation, not surgery on the existing corpus. lead_form_parallelism's 19/100 fire rate suggests possible overcalibration — should be inspected before promoting to BLOCK.

**Test totals**: 819 passing (+24 from A5). 2 pre-existing failures unchanged.

### A6 implementation + verification (commit history continues)

Added `pipeline/fixers/` directory with:
- `__init__.py`: `Fixer` ABC, `FixerRegistry`, `create_fixer_registry()` factory
- `universal_quantifier_fixer.py`: drops UQ word deterministically; minimal Sonnet rewrite if remaining text too short. Invariant guard rejects rewrites that re-introduce a UQ.
- `laterality_fixer.py`: deterministic regex flip; no LLM.
- `schema_labeling_fixer.py`: deterministic pair-swap via detector's `pair_matched` extra. Skips ambiguous cases.
- `numeric_overlap_fixer.py`: T2/T3 only — adjusts stem age to mid-range. The only fixer authorized to modify the stem.

13 new tests in `tests/test_routed_fixers.py` — registry routing + deterministic paths + invariant preservation tests.

**A6 production wiring (ship_readiness fix dispatch) DEFERRED** — multi-day refactor. The verification script demonstrates value via standalone path.

### A6 verification (10-question subset, $0.99)

`scripts/diagnosis/verify_a6_routed_fixers.py` runs routed fixers + re-audit on 10 representative bad questions from today's fix-and-export. Used PRE-fix state (so routed fixers have something to act on; the previous attempt accidentally used POST-fix state).

| Metric | Pre-A6 baseline | Post-A6 |
|---|---|---|
| dq_major | 5/10 | 5/10 (**no regression**) |
| dq_minor | 5/10 | 3/10 |
| dq_clean | 0/10 | 1/10 |
| english_gap clean | 8/10 | 9/10 |
| **Routed fixers fired** | — | **0** |

**Empirical reading**: 0 routed fixers fired because the 10 questions' audit-level signatures didn't match A6's fixer registry. Most bad questions on this corpus are flagged for ambiguity/factual/tier_fit issues — exactly the dq dimensions A7 (LLM-backed) targets, not the eg-signature patterns A6 handles. A6 proved it doesn't introduce regressions; the lift on this corpus is ~zero because the targeted signatures are rare here.

**Test totals**: 832 passing (+13 from A6). 2 pre-existing failures unchanged throughout.

### Cost summary at A6 close

| Activity | Cost |
|---|---|
| E2E-100 stratified validation | $23.81 |
| Fix-and-export Round 1 (Unicode crash) | ~$0.50–1 lost |
| Fix-and-export Round 2 (premature kill) | ~$5 lost |
| Fix-and-export Round 3 (clean run) | $9.46 |
| A1-A5 implementation | $0 (refactor + deterministic detectors only) |
| A6 verification (10-question subset) | $0.99 |
| A7 verification (10-question subset) | $0.04 |
| **Total** | **~$41** |

### A7 implementation + verification (commit history continues)

Two LLM-backed detectors at PHASE_AUDIT_LLM:

- `pipeline/detectors/llm_ambiguity.py`: Sonnet-backed; asks "for each option, is it defensibly correct?" Returns one DetectorSignal per defensible-alternative letter. Always advisory until calibration.
- `pipeline/detectors/llm_fact_check.py`: Opus-backed, T4-only; verifies factual claims. Returns signal per factual error with claim + correction.

Registry gains `scan_for_phase_async()` that awaits async detectors. Sync `scan()` raises NotImplementedError on LLM detectors — forces async path.

14 new tests with mocked anthropic clients (signal shape, JSON parsing, T4 gating, async dispatch, error handling).

**A7 verification (10 T4 questions, $0.04)**:
- llm_ambiguity: 8 total fires across 6/10 questions
- llm_fact_check: 0 fires (Opus found no factual errors)
- Correlation with audit dq ambiguity:
  - 6/9 audit-flagged-ambiguous ALSO fired llm_ambiguity
  - 0/1 audit-clean ambiguous fired llm_ambiguity
- Pattern: agreement on flagged + agreement on clean → meaningful signal, no false positives

**A7 is the first detector this session to fire non-trivially on actual problem cases in the corpus.** Combined with A6's routed-fixer architecture, the chain is now:
1. Deterministic detectors (A1-A5) catch surface patterns
2. Cell matrix (A2.5/S2) tunes overrides per tier
3. LLM-backed detectors (A7) surface ambiguity that regex can't see
4. Routed fixers (A6) act on signature-matched flags
5. (Future): A7's ambiguity signals could route to a dedicated ambiguity fixer

**Test totals**: 846 passing (+14 from A7). 2 pre-existing failures unchanged.
