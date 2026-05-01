# Goliath Pipeline — Continuation Context

Use this document to continue development in a new Claude session. It covers the full architecture, what's built, what's remaining, and how everything connects.

## Project

Goliath is the quiz question generation pipeline for the PassEPPP EPPP exam prep platform. It generates 4-tier multiple-choice questions from 1,566 research-backed anchor points across 9 psychology domains.

**Repo:** `C:\Users\mcdan\goliath\` (GitHub: mcdanielmjustin/goliath)
**Latest commit:** `975550c` — "Add QuestionOrchestrator, anchor brief system, and pipeline v2 architecture"
**Uncommitted changes (2026-04-26):** Quality-rule layer added — InputSanitizerAgent, AttributionGate, OptionLengthBalanceGate, EPONYM_WHITELIST (~120 names), new prompt rules, pytest harness in `tests/`. See "Quality-Rule Layer" section below.

## Architecture Overview

```
Phase 0: Anchor Brief Generation (LLM, once per anchor, ~$0.50 each)
         → data/anchor_briefs/{DOMAIN}/{uid}.json
         Script: scripts/generate_anchor_briefs.py

Phase 1a: Chapter Context (once per chapter)
          → ConceptVocabAgent loads concept_vocab/{domain}/{chapter_id}.json

Phase 1b: Anchor Context (once per anchor)
          → AnchorBriefAgent loads brief, falls back to chapter vocab
          → InputSanitizerAgent strips researcher citations from core_claims
          → KeywordExtractorAgent extracts topic keywords

Phase 1c: Task Preparation (per anchor × tier × variant)
          → InputSanitizerAgent strips citations from verbatim_anchor + testable_fact
          → TestedConceptSelectorAgent (deterministic rotation)
          → DistractorPlannerAgent (pre-assigns 3 unique misconception_ids)
          → FlashcardTemplateAgent (pre-generates flashcard fronts)
          → MetadataAgent (builds question_id, UUID, all metadata — uses sanitized text)
          → QuestionAngleSelectorAgent (AVAILABLE: picks angle from brief by tier affinity)
          → build_user_prompt() with core_claims + angle + concept_vocab + distractor assignments + 3 testwise-defense rules

Phase 2: Creative Generation (1 LLM call via QuestionCreatorAgent)
Phase 3: Assembly (QuestionAssemblerAgent merges creative + metadata)
Phase 4: Validation (8 gates: Structure → ContentQuality → Consistency → AnchorGrounding → Attribution → OptionLengthBalance → DistractorMix → Uniqueness)
Phase 5: Smart Retry (feedback-driven correction prompt, not blind retry)
```

## Key Files

### Pipeline Package (`pipeline/`)
- **`__init__.py`** — Base classes (BaseAgent, BaseGate, AgentRegistry), shared constants (DOMAIN_CODES, STEM_PATTERNS, DISTRACTOR_MIX, CORRECT_POSITIONS, SECTION_TITLE_MAP, **EPONYM_WHITELIST**), utility functions.
- **`orchestrator.py`** — QuestionOrchestrator class. Single source of truth for the pipeline graph. Methods: `load_chapter_context()`, `load_anchor_context()`, `prepare_task()`, `generate_and_validate()`, `_load_domain_vocab()`. Has ASSIGNED_SKILLS (11) and AVAILABLE_SKILLS (1).
- **`agents.py`** — All agent classes. Phase 1: **InputSanitizerAgent**, ConceptVocabAgent, AnchorBriefAgent, QuestionAngleSelectorAgent, DistractorPlannerAgent, KeywordExtractorAgent, MetadataAgent, TestedConceptSelectorAgent, FlashcardTemplateAgent. Phase 2: QuestionCreatorAgent. Phase 3: QuestionAssemblerAgent.
- **`gates.py`** — 8 validation gates + `create_gate_pipeline()`. StructureGate, ContentQualityGate, ConsistencyGate, AnchorGroundingGate, **AttributionGate**, **OptionLengthBalanceGate**, DistractorMixGate, UniquenessGate.
- **`prompts.py`** — `build_system_prompt()` (focused/open modes, with `_mcq_quality_rules()` block enforcing the 3 testwise-defense rules), `build_user_prompt()` (with core_claims, question_angle, concept_vocab, distractor_plan, character), `build_correction_prompt()` (gate-specific feedback for retry — includes guidance for attribution + option_length_balance).
- **`names.py`** — 3,300+ names for deterministic character assignment. `get_character_assignment(question_id)`.

### Scripts (`scripts/`)
- **`generate_quiz_questions.py`** — Main generation script. Uses QuestionOrchestrator. Async with concurrent workers. `--domain`, `--all`, `--anchor`, `--difficulty`, `--count`, `--workers`, `--resume`, `--clean`, `--dry-run`.
- **`generate_anchor_briefs.py`** — Generates per-anchor analysis briefs using claude-opus-4-7. `--domain`, `--all`, `--anchor`, `--chapter`, `--resume`, `--dry-run`. Run BEFORE question generation.
- **`generate_concept_vocab.py`** — Generates chapter-level concept vocabulary. Superseded by anchor briefs for anchors that have briefs; still needed as fallback for anchors without briefs.
- **`batch_generate.py`** — Batch API workflow (prepare → submit → wait → collect). `cmd_prepare()` needs rewriting to use the orchestrator (plan exists at `.claude/plans/validated-watching-adleman.md`).
- **`audit_question_quality.py`** — Audits a generated quiz JSON for the 3 testwise-defense rules (attribution leakage, option-length balance, elaboration tells). Re-runnable on any batch. Writes a per-question table + summary stats. Imports `EPONYM_WHITELIST` from `pipeline` (single source of truth).
- **`sweep_corpus_for_names.py`** — Scans `anchor_points.csv`, `anchor_passages_v3`, and `data/anchor_briefs/` for all researcher-citation patterns. Tallies by frequency, flags whitelist hits/misses. Run before expanding EPONYM_WHITELIST.
- **`sync_csvs_from_onedrive.py`** — Helper for the Excel-edit workflow. Copies the three source CSVs from OneDrive → `csvs/` with row-count and byte-size deltas. `--dry-run` previews; `--force` skips overwrite prompt.
- **`shared_constants.py`** — DOMAIN_CODES (int keys: {1: "PMET", ..., 9: "PETH"}), CODE_TO_ID, DOMAIN_NAMES.

### Tests (`tests/`)
Pytest-discoverable, also runnable with stdlib unittest (`python -m unittest discover -s tests -v` from repo root).
- **`test_input_sanitizer.py`** — Citation stripping, whitelist exemption, edge cases, **Bandura regression test** for the `\\band\\b` fix.
- **`test_gates.py`** — AttributionGate (year, et al., according-to, possessive, institutional false-positive exemption) + OptionLengthBalanceGate (ratio mode, tell mode, custom thresholds).
- **`test_eponym_whitelist.py`** — Sanity checks: required names present, frozenset, no whitespace, no empty strings.
- **`conftest.py`** — Adds repo root + `scripts/` to `sys.path` so `pipeline` is importable. Both pytest and unittest pick it up.
- 47 tests, ~5 ms run time.

### Config
- **`config.py`** — All paths. `MASTER_CSV_DIR = REPO_ROOT / "csvs"` (source CSVs in repo); `ONEDRIVE_DISTRIBUTION_DIR` for the generated enrichment archive that stays in OneDrive.

### Data Taxonomy (3 categories)

| Category | Examples | Where it lives |
|---|---|---|
| **Source data** (canonical, low churn, drives generation) | `anchor_points.csv`, `anchor_passages_v3_pure_textbook_1081.csv`, `chapter_schema_v3.csv` | **`csvs/`** in repo (committed) |
| **Generated artifacts** (regeneratable from source + LLM) | Question JSONs, anchor briefs, concept vocab, logs | `data/` (gitignored) |
| **Distribution outputs** (large flat-form views for downstream consumers) | `enrichment_all_questions.csv` (86+ MB) | OneDrive `Master CSVs/` (`ONEDRIVE_DISTRIBUTION_DIR`) |

Excel-edit workflow: edit `.xlsx` in OneDrive → re-export to CSV → run `python scripts/sync_csvs_from_onedrive.py` → review `git diff csvs/` → commit.

### Source CSVs (`csvs/` — committed)
- **`anchor_points.csv`** (1.4 MB, 1,566 rows) — All anchors with verbatim_anchor (canonical) + testable_fact (context).
- **`anchor_passages_v3_pure_textbook_1081.csv`** (4.6 MB, 1,081 rows) — Textbook anchors with `passage` column.
- **`chapter_schema_v3.csv`** (1.0 MB, 1,091 rows) — Sister to passages CSV; per-anchor metadata (blooms, tiers, content_summary, topic).

### Data (`data/` — gitignored, generated output)
- **`anchor_briefs/{DOMAIN}/{uid}.json`** — Per-anchor analysis briefs. **Primary input path.** 25 generated so far (BPSY=4, CASS=5, CPAT=2, LDEV=2, PETH=5, PMET=2, PTHE=2, SOCU=1, WDEV=2). Each brief has core_claims, testable_fact, concepts, misconceptions, question_angles, chapter_num.
- **`concept_vocab/{DOMAIN}/{chapter_id}.json`** — Chapter-level vocabulary. **Fallback path** — used only when no anchor brief is available. Superseded by anchor_briefs for anchors with briefs. Currently only 1 file (BPSY/D7-Ch11.json); not actively expanded since briefs are the production path.
- **`domain_vocab/{DOMAIN}.json`** — Phase 7 curated/bootstrapped domain-level vocabulary pool. T1/T2 only — broadens permitted_vocabulary so distractors can share more terms with the correct option without violating Bloom's identity. Bootstrapped via `scripts/generate_domain_vocab.py`; `curated: false` flag tracks human-review status.
- **`quiz/{DOMAIN}/{chapter-slug}.json`** — Generated questions.

## Skill Architecture

### Assigned Skills (always fire, deterministic, $0)
| Skill | Agent | What it does |
|---|---|---|
| concept_vocab | ConceptVocabAgent | Loads chapter-level concepts/misconceptions (fallback path; superseded by anchor briefs for anchors with briefs) |
| anchor_brief | AnchorBriefAgent | Loads per-anchor brief (concepts, misconceptions, core_claims, question_angles) |
| input_sanitizer | InputSanitizerAgent | Strips "Author (YYYY):" researcher citations from verbatim_anchor / testable_fact / core_claims so the LLM never sees the names. Whitelisted eponyms preserved (year stripped). |
| keyword_extractor | KeywordExtractorAgent | Extracts topic keywords from content + concepts |
| metadata | MetadataAgent | Builds question_id, UUIDs, all deterministic metadata |
| tested_concept_selector | TestedConceptSelectorAgent | Picks tested concept by rotation across variants/tiers |
| concept_integration_planner | ConceptIntegrationPlannerAgent | T4 only: pre-assigns 2 concepts the question MUST integrate (synthesis-tier scaffold) |
| correct_answer_form_planner | CorrectAnswerFormPlannerAgent | Pre-assigns cognitive verb, option form, length, permitted concept set, and L3 vocabulary scaffold for the correct answer (T1/T2 broader brief-internal pool + Phase 7 domain pool) |
| anchor_cluster | AnchorClusterAgent | Tier-keyed sibling-anchor selection: 0 at T1/T2, 1 at T3, 2 at T4. Vocabulary headroom + cross-content substrate for application/synthesis tiers. |
| flashcard_template | FlashcardTemplateAgent | Pre-generates flashcard front templates |
| distractor_planner | DistractorPlannerAgent | Assigns distractor levels + 3 unique misconception_ids (prioritizes those involving tested concept) |

### Available Skills (fire conditionally)
| Skill | Agent | When it fires |
|---|---|---|
| question_angle_selector | QuestionAngleSelectorAgent | When anchor brief has question_angles. Maps angle types to tiers (definitional→T1, clinical_application→T3, mechanism→T4). |

## Anchor Brief Structure

```json
{
  "uid": "D7-PHY-021-b323a513",
  "domain_code": "BPSY",
  "has_passage": true,
  "core_claims": [
    "Nondeclarative (implicit) memory includes procedural/skill learning...",
    "Implicit memory is preserved in amnesic patients with hippocampal damage.",
    "Procedural learning depends on the basal ganglia, not the hippocampus."
  ],
  "concepts": [
    {"concept_id": "nondeclarative-memory-system", "label": "...", "description": "..."},
    {"concept_id": "basal-ganglia-procedural", "label": "...", "description": "..."}
  ],
  "misconceptions": [
    {"misconception_id": "hippocampus-vs-basal-ganglia-procedural", "label": "...", "type": "similar_store", "concepts_involved": ["hippocampal-declarative-system", "basal-ganglia-procedural"]}
  ],
  "question_angles": [
    {"type": "clinical_application", "description": "Present an amnesic patient who can learn a new motor skill..."},
    {"type": "neuroanatomical", "description": "Ask which brain structure mediates a specific implicit memory subtype..."}
  ]
}
```

## What's Remaining

### 1. Generate All Anchor Briefs (~$540)
```bash
cd scripts
python generate_anchor_briefs.py --all --resume
```
1,081 textbook anchors + 486 proprietary = 1,566 total. Only 1 brief exists so far. This is a prerequisite for the production run. Resume-safe.

### 2. 20-Question Test Batch
Run on anchor D7-PHY-021-b323a513 (BPSY, has brief) to verify:
- All 20 questions address the anchor's core claims
- 0 failures (duplicate misconceptions were previously a 50% failure rate, fixed by DistractorPlannerAgent)
- Question angles correctly routed by tier
```bash
python generate_quiz_questions.py --domain BPSY --anchor D7-PHY-021-b323a513 --workers 5 --clean
```

### 3. Rewrite batch_generate.py cmd_prepare()
Plan exists at `.claude/plans/validated-watching-adleman.md`. The `cmd_prepare()` function references deleted files and agents. It needs to use the QuestionOrchestrator instead. The submit/wait/collect steps are working fine — only cmd_prepare needs rewriting. The orchestrator's `prepare_task()` produces the exact task dict that cmd_submit expects.

### 4. Production Run
- 1,566 anchors × 4 tiers × 5 variants = ~31,320 questions (some may be filtered)
- Estimated cost: depends on model. Opus at ~$0.20/question = ~$6,264
- Use batch API (50% cost savings) via batch_generate.py after cmd_prepare is rewritten

## Important Technical Details

### Domain Codes
Integer keys in shared_constants.py: `{1: "PMET", 2: "LDEV", 3: "CPAT", 4: "PTHE", 5: "SOCU", 6: "WDEV", 7: "BPSY", 8: "CASS", 9: "PETH"}`. NEVER use string keys like "domain7".

### Anchor Brief Fallback
When an anchor has no brief, the pipeline falls back to chapter-level concept vocab. The `load_anchor_context()` method handles this automatically. Questions without briefs use "open" prompt mode (LLM generates concept/misconception IDs). Questions with briefs use "focused" mode (IDs pre-assigned).

### Feedback-Driven Retry
When a gate fails, `build_correction_prompt()` appends gate-specific guidance to the original prompt. The `_GATE_GUIDANCE` dict in `prompts.py` has pre-written guidance for each gate type. This replaces blind retry (same prompt sent again) with targeted correction.

### Temperature Deprecation
claude-opus-4-7 no longer accepts the `temperature` parameter (returns 400 error). Removed from all API calls. Only exception: `temperature=1` is required for extended thinking mode in batch_generate.py.

### Model
All generation uses `claude-opus-4-7`. Anchor brief generation also uses `claude-opus-4-7`.

### Running Scripts
Always run from `scripts/` directory:
```bash
cd C:\Users\mcdan\goliath\scripts
python generate_quiz_questions.py --domain BPSY --dry-run
```
The `sys.path` setup in each script assumes this working directory.

## Key Decisions Made

1. **Anchor briefs > chapter vocab expansion** — 5 anchor-specific concepts + 8 misconceptions are strictly better than 12 generic chapter-level misconceptions. The $540 investment produces reusable per-anchor intelligence.

2. **Assigned vs available skills** — Assigned skills (always fire) handle constraint satisfaction (picking unique misconceptions, rotating concepts). Available skills handle conditional intelligence (question angles only when brief exists). This is "best of both worlds" — deterministic reliability + creative flexibility.

3. **DistractorPlannerAgent pre-assigns misconception_ids** — The LLM was failing 50% of the time on "pick 3 unique items from a list." The hardcoded agent does this perfectly every time. The LLM shapes scenarios around assigned misconceptions rather than choosing them.

4. **Core claims in prompt, not in gates** — Checking if a question "addresses" a core claim requires semantic understanding. Gates are $0/instant so they can only check structural properties. The core claims constraint is enforced via the prompt ("MUST ADDRESS at least one"), and the AnchorGroundingGate checks the structural proxy (tested_concept from brief's concept list).

5. **Question angles are content guidance, stem patterns are format** — Both exist in the prompt. The stem pattern (e.g., "clinical_vignette") defines the question FORMAT. The question angle (e.g., "present an amnesic patient who can learn a new motor skill") defines the CONTENT APPROACH. They complement each other.

## Quality-Rule Layer (added 2026-04-26)

Three testwise-defense rules added to defeat documented MCQ heuristics that exam-takers exploit:

1. **No researcher attribution** — "Squire (2004)", "According to Smith", "Smith's framework", "Smith et al." are forbidden in stems/options/explanations. Eponyms on `EPONYM_WHITELIST` (Piaget, Pavlovian, Cannon-Bard, etc.) are exempt.
2. **Answer-length balance** — All 4 options within ~1.5x character count of each other; correct must not be >20% longer than longest distractor.
3. **Parallel construction** — Parens/semicolons/em-dashes/compound clauses must not cluster only on the correct answer.

### Architecture: layered enforcement

| Rule | Pre-LLM agent | Prompt rule | Validation gate |
|---|---|---|---|
| 1. Attribution | `InputSanitizerAgent` strips citations from anchor inputs | Yes — restated in `_mcq_quality_rules()` with full whitelist | `AttributionGate` (regex with whitelist exemption) |
| 2. Length balance | — | Yes | `OptionLengthBalanceGate` (ratio_max=1.7, tell_margin=1.2) |
| 3. Elaboration tells | — | Yes | None — prompt-only (defer until larger sample shows pattern) |

### EPONYM_WHITELIST (in `pipeline/__init__.py`)

~120 names organized by category (developmental, conditioning, social, multicultural, I-O/career, neuroanatomy, etc.) plus ~5 institutional terms (Psychology, Association, Society, Guidelines, Standards) to prevent false positives on document titles like "APA Guidelines for Forensic Psychology (2013)".

**Source:** `scripts/sweep_corpus_for_names.py` was run over the full corpus — 1,566 anchors + 1,081 passages + 1 brief — finding 836 distinct names across 2,079 citation instances. After whitelist expansion, coverage went from 9.8% → 27.8%. The remaining 72.2% are textbook authors (Berk, Cascio, Aguinis) and one-off citations that we WANT stripped.

**Adding a new eponym:** edit `EPONYM_WHITELIST` in `pipeline/__init__.py`, then run `python -m unittest discover -s tests -v` to confirm the eponym sanity test still passes.

### Audit baseline (from this anchor's 20-question batch, before vs. after the layer)

| Metric | Before | After |
|---|---|---|
| Attribution violations | 3/20 questions, 5 instances | **0/20** |
| Correct = longest option | 15/20 (75%) | 10/20 (50%) — chance is 25% |
| Correct >20% longer than all distractors | 5/20 | **0/20** |
| Mean max/min length ratio | 1.38 | 1.27 |
| Cost | $4.04 | $3.91 |
| Gate retries fired | n/a | 0 — prompt rules sufficient on first pass |

### Critical regression caught by tests (do not re-introduce)

The split pattern `(?:&|and|,)` matches "and" *inside* names like **Bandura**, splitting it into `["B", "ura"]` — neither whitelisted. Fix uses `\band\b` (word boundary). Test: `tests/test_input_sanitizer.py::TestBanduraRegression`.

### Citation detection coverage (added 2026-04-26)

Empirical scan of the source corpus surfaced four leak patterns the original gate missed:

| Pattern | Pre-fix corpus hits | Post-fix |
|---|---|---|
| Initialed citations (`"Smith, A. (2010)"`, `"Watson, J.B. & Rayner, R. (1920)"`) | 155 unhandled | All caught |
| Bare multi-author (`"Smith and Jones found..."` w/o year) | 15 unhandled | All caught |
| Non-ASCII names (`Latané`, `Köhler`) | 0 in current data | Preemptively handled |
| Mixed-whitelist multi-author (`"Smith and Bandura (2010)"`) | All-or-nothing strip | Best-of-breed: keep WL authors |

`pipeline/citation_patterns.py` is the single source of truth for the five attribution regexes (year, et_al, according_to, possessive, bare_multi). All four consumers (gate, sanitizer, audit, sweep) import from it.

### Whitelist policy

140 names in `EPONYM_WHITELIST`. Categories: developmental, conditioning/behaviorism, psychoanalytic/humanistic, CBT/clinical, family/clinical models, emotion/motivation, memory/cognition, linguistics/culture, social, multicultural, I-O/career, neuroanatomy, aging/lifespan, assessment, institutional. Tradeoffs documented in `pipeline/__init__.py` next to the constant.
