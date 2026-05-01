# Enrichment Question Generation -- Complete Specification

Compiled 2026-04-25 from 5 extraction documents. This is the single authoritative reference for the EPPP enrichment question generation project.

---

### Terminology: Tier vs Difficulty

| Term | Meaning | Example |
|------|---------|---------|
| **Tier** | Bloom's cognitive level a question is generated at. There are 4 tiers. | Tier 1 = Remember / Understand (ceiling), Tier 4 = Evaluate / Analyze (floor) |
| **Difficulty** | What the student selects (Easy/Medium/Hard/Expert). Determines the *mix* of tiers served in a session. | Student picks "Hard" → gets 5% Tier 1 + 15% Tier 2 + 55% Tier 3 + 25% Tier 4 |

Generation formula: **4 tiers × 5 variants × 1,566 anchors = 31,320 total questions**

---

## 1. Project Scope & Numbers

| Metric | Value | Source |
|--------|-------|--------|
| Total anchor points (source of truth) | 1,567 | `anchors_parsed.json` |
| In-book anchors (V3 schema) | 1,081 | `anchor_passages_v3` CSV |
| Unique anchors in master CSV | 1,509 | master CSV audit |
| Textbook chapters (mastery-page, source of truth) | 96 | content directory |
| Textbook chapters (PassEPPP-website, +33 expanded) | 129 | website content |
| V3 chapter schema | 121 chapters | design doc |
| Textbook subsections (h2+h3) | 2,676 | extraction |
| Questions generated locally (full metadata) | 27,213 | local JSON files |
| Questions in master CSV | 12,703 | `enrichment_all_questions.csv` (62 columns) |
| Questions uploaded to Supabase enrichment_questions | 9,977 (37% of local) | upload audit |
| Questions deployed (quiz_stats.json) | 13,048 | deployed stats |
| Gap: master CSV vs deployed | +1,971 undeployed | audit |
| Questions in old format (incompatible) | 7,445 | `all_questions.json` |
| Target total | ~35,689 | 15,019 existing + 20,670 new |
| New questions to generate | 20,670 | 4 tiers x 5 variants x 1,566 anchors minus existing |
| Classic flashcards | 2,936 | manifest |
| Adaptive flashcards (zero API cost) | 78,749 | extracted from quiz seeds |
| Total flashcard inventory | 81,685 | classic + adaptive |
| Misconception registry entries | 706 | `misconception_catalog_with_remediation.json` |
| Name pool for vignettes | 3,701 | generation pipeline |
| Mock exam questions (separate system) | 4,500 | questions table |

### Master CSV Schema (62 columns)

**Source:** `C:\Users\mcdan\OneDrive\Master CSVs\enrichment_all_questions.csv` (12,703 rows)

**Identity:** question_id, question_type, variant

**Taxonomy:** domain_code, domain_id, domain_name, chapter_num, chapter_title, chapter_file, section_title, chapter_uuid, anchor_uuid, anchor_label

**Difficulty:** difficulty_tier, difficulty_label, blooms_primary, blooms_secondary

**Question:** stem_pattern, source_type, question_stem, option_a, option_b, option_c, option_d, correct_answer

**Explanations:** explanation_a, explanation_b, explanation_c, explanation_d

**Concept:** tested_concept_id, tested_concept_label, knowledge_tested

**Anchor:** anchor_uid, anchor_point_id_v2, anchor_content_summary, testable_fact

**Flashcards:** flashcard_concept_front, flashcard_concept_back, flashcard_comparison_front, flashcard_comparison_back, flashcard_nuance_front, flashcard_nuance_back

**Distractor summary:** distractor_1_letter, distractor_1_level, distractor_1_misconception_type, distractor_2_letter, distractor_2_level, distractor_2_misconception_type, distractor_3_letter, distractor_3_level, distractor_3_misconception_type

**Distractor detail:** distractor_1_concept_id, distractor_1_misconception_id, distractor_1_confused_with, distractor_2_concept_id, distractor_2_misconception_id, distractor_2_confused_with, distractor_3_concept_id, distractor_3_misconception_id, distractor_3_confused_with

**Provenance:** generation_batch, generated_by

Supabase `enrichment_questions` table mirrors this schema.

**Note:** `anchor_content_summary` = canonical `verbatim_anchor`. Column names are legacy; data is current as of 2026-04-25.

### Distribution by Domain (Master CSV, 15,019 questions)

| Domain | Count | Easy | Medium | Hard |
|--------|-------|----|----|-----|
| CASS | 2,201 | 756 | 729 | 716 |
| PMET | 2,056 | 689 | 684 | 683 |
| BPSY | 1,827 | 602 | 636 | 589 |
| PETH | 1,726 | 581 | 592 | 553 |
| WDEV | 1,713 | 563 | 581 | 569 |
| PTHE | 1,520 | 523 | 509 | 488 |
| SOCU | 1,444 | 492 | 481 | 471 |
| CPAT | 1,416 | 489 | 472 | 455 |
| LDEV | 1,116 | 376 | 381 | 359 |
| **Total** | **15,019** | **5,071** | **5,065** | **4,883** |

Expert: 0 across all domains (never generated).

### Deployed vs Master Gap

| Domain | Delta (Master - Deployed) |
|--------|--------------------------|
| PMET | +9 |
| LDEV | +417 |
| CPAT | **-261** (deployed > master -- anomaly) |
| PTHE | +87 |
| SOCU | +182 |
| WDEV | +315 |
| BPSY | +520 |
| CASS | +119 |
| PETH | +583 |

---

## 2. Anchor Point System

### Canonical Anchor CSV Schema (10 columns)

`uid`, `anchor_point_id_v2`, `domain_num`, `domain_name`, `source`, `chapter_num`, `chapter_title`, `verbatim_anchor`, `old_anchor`, `testable_fact`

- **Canonical source (SINGLE SOURCE OF TRUTH):** `goliath/csvs/anchor_points.csv` (committed to repo as of 2026-04-26; mirrored from OneDrive Master CSVs via `scripts/sync_csvs_from_onedrive.py`)
- **1,566 data rows**, 100% populated on all content columns
- **Stale (DO NOT USE):** `all_anchors.csv` (archived to Desktop), old `anchor_points.csv` (8 columns, no testable_fact)
- `verbatim_anchor` now includes citations (e.g., "von Hippel, P.T. (2005)...")
- `old_anchor` preserves pre-citation text
- `testable_fact` = distilled, test-ready content (NEW as of 2026-04-24, all 1,566 rows populated)

### Content Columns Used for Generation

The generator uses exactly TWO content columns from this CSV:

| Column | Purpose | Populated |
|--------|---------|-----------|
| `verbatim_anchor` | Citation-bearing anchor text; provides domain context and source attribution | 1,566 / 1,566 (100%) |
| `testable_fact` | Distilled, test-ready fact; feeds directly into question stem generation | 1,566 / 1,566 (100%) |

**Both columns are fed to the generation prompt.** `verbatim_anchor` (with citations) provides grounding; `testable_fact` provides the precise testable claim.

### Anchor Coverage (V3 Schema, 1,081 in-book anchors)

| Domain | Code | In-Book | Total (1,567) | Coverage |
|--------|------|---------|---------------|----------|
| PMET | D1 | 139 | 193 | 72.0% |
| LDEV | D2 | 113 | 174 | 64.9% |
| CPAT | D3 | 87 | 134 | 64.9% |
| PTHE | D4 | 111 | 168 | 66.1% |
| SOCU | D5 | 120 | 153 | 78.4% |
| WDEV | D6 | 122 | 172 | 70.9% |
| BPSY | D7 | 127 | 192 | 66.1% |
| CASS | D8 | 134 | 190 | 70.5% |
| PETH | D9 | 128 | 191 | 67.0% |
| **Total** | | **1,081** | **1,567** | **69.0%** |

### Anchor Statistics (Master CSV)

- 1,509 unique anchors across 15,019 questions
- Questions per anchor: min=1, max=80, median=6, mean=10.0
- 57 anchors have zero questions (P0 generation priority)
- ~44% of anchors have only 1 variant per Bloom's level (repeat exposure risk)
- `analyze` level: 123 anchors, ALL single-variant

### Anchor Categories (for enrichment pipeline filtering)

| Category | Definition | Detection Signals |
|----------|-----------|------------------|
| `standard_concept` | Generic psychology concept any textbook covers | Definitional language, absence of named characters/scenarios |
| `scenario_vignette` | Named characters or clinical scenarios revealing exam questions | Dr./Mrs./Mr. [Name], specific person-situation combos |
| `exam_strategy` | Test-answer framing | "would be unacceptable", "acted ethically only if" |

**Ethics nuance:** Anchors stating rules = `standard_concept`. Anchors applying scenarios = `scenario_vignette`.

### How Anchor Data Flows into Prompts

1. Loaded from Master CSV
2. Passed as list of dicts to `build_user_prompt()`
3. Up to 5 anchors per subsection
4. Each includes: uid, verbatim_anchor (200 char limit), testable_fact (200 char limit)
5. Format: `- [{uid}] {verbatim}\n  Testable Fact: {testable}`
6. Instruction: "Use BOTH fields. verbatim_anchor = sourced claim with citation. testable_fact = specific knowledge student must demonstrate. Do NOT quote either field verbatim."

### Source Type Mapping

Source type is now determined per stem pattern via `STEM_SOURCE_TYPE` in `pipeline/__init__.py`.
Two types: `anchor_grounded` (test anchor directly) and `integrated` (synthesize anchor + passage).
Gradient: T1 mostly anchor-grounded, T4 all integrated. Overall ~45/55 anchor/integrated.

### Anchor Text Budget

| Setting | Per Anchor | Max Anchors | Total |
|---------|-----------|-------------|-------|
| Current | 200 chars | 5 | 1,000 chars |
| Recommended | 300 chars | 8 | ~2,400 chars |

---

## 3. Bloom's Tiers (Generation Levels)

### Tier Definitions

| Tier | Bloom's Primary (design target) | Bloom's Secondary (incidental) | EPPP Calibration |
|------|----------------------------------|-------------------------------|------------------|
| Tier 1 | Remember | Understand (ceiling) | Below EPPP |
| Tier 2 | Understand | Apply (ceiling) | Below EPPP |
| Tier 3 | Apply | Analyze (ceiling) | Approximate EPPP difficulty |
| Tier 4 | Evaluate | Analyze (floor) | Above EPPP |

### Three Independent Dimensions Per Tier

1. **Bloom's level** -- Tier 1=Remember to Tier 4=Evaluate
2. **Distractor complexity** -- Tier 1-2 get L1+L2+L3; Tier 3 gets L2+L3+L4; Tier 4 gets L3+L4+L4
3. **Model routing** -- `--fast-tiers` sends Tier 1-2 to Sonnet, Tier 3-4 to Opus

### Blueprint Tier Tags (Domain-Level)

| Tag | Domains | Blueprint Tier |
|-----|---------|---------------|
| FOUNDATION | D1, D2 | Remember, Understand |
| APPLICATION | D3, D4 | Apply, Analyze |
| CONTEXT | D5, D6, D7 | Analyze, Evaluate |
| CAPSTONE | D8, D9 | Evaluate, Create |

### Production Distribution (27,213 questions, 3 tiers only)

| Tier | Count | % |
|------|-------|---|
| Tier 1 (Remember) | 9,144 | 33.6% |
| Tier 2 (Understand) | 9,152 | 33.6% |
| Tier 3 (Apply) | 8,917 | 32.8% |
| Tier 4 (Evaluate) | 0 | 0% (dropped from production run, user decision) |

### Tier 4 Pool Exhaustion Risk

- Original generation target: 20/35/30/15 across Tier 1/2/3/4
- Expert mode (60% Expert) uses 12 of 15 available Expert questions per 20-question session = 80% of pool
- After 2 sessions, student sees repeats
- Future target if needed: 15/30/30/25

### Stem Length Constraints by Tier

| Tier | Bloom's (primary / secondary) | Sentences | Words | Description |
|------|-------------------------------|-----------|-------|-------------|
| Tier 1 (Remember) | Remember / Understand (ceiling) | 1-3 | ~20-50 | Direct questions, no scenario |
| Tier 2 (Understand) | Understand / Apply (ceiling) | 2-4 | ~40-70 | Brief context, one setup + question |
| Tier 3 (Apply) | Apply / Analyze (ceiling) | 4-7 | ~80-130 | Full scenario, named characters, multi-step |
| Tier 4 (Evaluate) | Evaluate / Analyze (floor) | 5-8 | ~100-150 | Complex scenario, competing info, 2+ factors |

**Hard cap:** 8 sentences / 150 words for all tiers.

### Tier 4 Specifications (Never Generated Before)

- **Distractor mix:** L3 + L4 + L4 (double partially-correct)
- **Key distinction from Tier 3:** Tier 3 = one right answer is clear; Tier 4 = two answers look defensible, evaluate which is BETTER
- **Stem requirements:** Must present genuine ambiguity; at least 2 options partially true; stem must provide enough info to discriminate; NOT harder by being longer -- harder by being more subtle
- **Explanation requirements:** Correct must say WHY more correct than alternatives; wrong L4 must acknowledge what's TRUE before explaining why not BEST; distinction must be clinically/empirically meaningful, not a technicality
- **Bloom's enforcement:** Acceptable verbs: differentiate, organize, attribute, determine which factor, judge, critique, defend, justify, assess validity. Forbidden: questions where one answer is obviously right.

### Bloom's Enforcement by Tier

| Tier | Acceptable | Forbidden | Guard |
|------|-----------|-----------|-------|
| Tier 1 (Remember) | define, identify, recognize, list, recall, explain, compare | "In this scenario..." | Named character doing something = reject |
| Tier 2 (Understand) | explain, compare, contrast, paraphrase, interpret, demonstrate | Pure recall | Must require minimal inference |
| Tier 3 (Apply) | demonstrate, use, solve, apply to scenario, differentiate | -- | All options must be plausible in SOME context |
| Tier 4 (Evaluate) | differentiate, attribute, determine which factor, judge, critique | Questions where one answer is obviously right | If first-year student can eliminate 2 options immediately, too easy |

All 4 tiers have `_blooms_stem_enforcement()` rules. Tier 1-2: prevent upward creep (scenarios, multi-step reasoning). Tier 3-4: prevent downward creep (anti-patterns, Named-Effect rules, Single-Fact Concept Guard, Two-Concept Integration Requirement for Tier 4).

---

## 4. Bloom's Taxonomy & Webb's DOK Mapping

### Bloom's Verb Reference (injected into every prompt)

- **remember:** define, identify, recognize, list, recall
- **apply:** demonstrate, use, solve, apply to a scenario
- **analyze:** differentiate, organize, attribute, determine which factor

### Bloom's Distribution (27,213 questions)

| Bloom's Primary | Count | % |
|-----------------|-------|---|
| Remember | 9,144 | 33.6% |
| Understand | 9,152 | 33.6% |
| Apply | 8,917 | 32.8% |

Each question also has `blooms_secondary` (one level up). Evaluate is now covered by Tier 4 (5 stem patterns — see Section 7). Create is excluded as it doesn't fit MCQ format.

### Master CSV Bloom's Distribution (15,019 questions)

| Level | Count |
|-------|-------|
| remember | 5,071 |
| understand | 5,065 |
| apply | 4,758 |
| analyze | 125 |

### Bloom's Enforcement Gap

- System prompt includes Bloom's verbs per level
- `validate_question` NEVER checks whether the stem actually uses appropriate-level verbs
- A Hard question could start with "Which of the following IS..." (remember-level) and pass validation
- **Recommendation:** Soft warning (not hard fail) when stem lacks difficulty-appropriate verbs

### Webb's DOK Mapping (V3 Chapters)

| DOK | Description | Where Used |
|-----|-------------|-----------|
| 1 | Recall | Early chapters in each domain |
| 2 | Skills/Concepts | Early-middle chapters |
| 3 | Strategic Thinking | Middle-late chapters |
| 4 | Extended Thinking | Capstone chapters |

Every domain starts at DOK 1-2 and ends at DOK 3-4. See Section 17 for per-chapter assignments.

---

## 5. Distractor Framework (L1-L4)

### Distractor Level Definitions

| Level | Name | Diagnostic Question | What It Probes |
|-------|------|---------------------|----------------|
| L1 | Cross-subdomain | "Did they study this chapter?" | "Do they know this topic exists?" |
| L2 | Same-subdomain | "Do they know distinctions within this topic?" | "Can they distinguish within this topic?" |
| L3 | Same-concept-family | "Can they discriminate closely related concepts?" | "Can they discriminate closely related concepts?" |
| L4 | Partially-correct | "Can they evaluate which is MOST correct?" | "Can they evaluate what's MOST correct?" |

As tier increases, distractors get closer to the correct answer. L1 drops out and L4 takes over.

### Distractor Mix Per Tier (FINAL -- Post-Scrutiny)

| Tier | Mix | Purpose |
|------|-----|---------|
| Tier 1 (Remember) | 1x L1, 1x L2, 1x L3 | Cross-subdomain + same-subdomain + same-concept |
| Tier 2 (Understand) | 1x L1, 1x L2, 1x L3 | Same spread as Tier 1 |
| Tier 3 (Apply) | 1x L2, 1x L3, 1x L4 | Drops L1, adds partially-correct |
| Tier 4 (Evaluate) | 1x L3, 2x L4 | All close-in: confused concepts + 2 partially-correct |

Source: `DISTRACTOR_MIX` at `__init__.py:47-52`. Enforcement: 100% hardcoded by assembler.

### Distractor Mix Per Tier (ORIGINAL -- Pre-Scrutiny, superseded)

| Tier | Mix | Note |
|------|-----|------|
| Tier 1 | 2x L1 + 1x L2 | No misconception probe |
| Tier 2 | 1x L1 + 1x L2 + 1x L3 | Balanced |
| Tier 3 | 1x L2 + 2x L3 | No L4 |
| Tier 4 | 1x L3 + 2x L4 | Expert |

### Rationale for Revised Mix

- Tier 1 original: 2 L1 distractors = diagnostically redundant (both cross-subdomain; picking either reveals the same thing: "they're lost"). Replacing one L1 with L3 detects common misconceptions even at easy level.
- Tier 3 original: no L4; actual EPPP is notorious for "all four options seem right" -- Tier 3 needs at least one L4.

### Distractor Enforcement

- Distractor levels: 100% hardcoded by assembler
- Types: Validated by ConsistencyGate
- Mix: Checked by DistractorMixGate (soft enforcement)

### Per-Distractor Fields

| Field | Type | Present On |
|-------|------|-----------|
| `distractor_level` | integer | wrong options (1-4) |
| `concept_id` | string | all options |
| `misconception_id` | string | wrong options |
| `misconception_type` | string | wrong options |
| `confused_with` | string | wrong options (concept_id of correct answer) |
| `explanation` | string | all options (200-300 words) |

---

## 6. Misconception Type System

### 6 Misconception Types

| Type | Definition | % of Production | Optimal Study Strategy |
|------|-----------|----------------|----------------------|
| similar_property | Shared surface features, different mechanism | 32.5% | Side-by-side comparison tables |
| partial_understanding | Almost right, missing one crucial qualifier | 27.7% | Case study practice, application drills |
| overgeneralization | Correct principle applied beyond its scope | 19.6% | Exception lists, boundary exercises |
| similar_name | Terminological confusion | 19.2% | Mnemonic devices, discrimination drills |
| opposite_direction | Reversed causal direction | 12.4% | Arrow diagrams, directional mnemonics |
| similar_store | Same mental shelf (chapter/category) | 2.1% | System diagrams, categorical maps |

Note: percentages sum to >100% because each question has 3 distractors.

### Master CSV Misconception Distribution (all distractors)

| Type | Count |
|------|-------|
| similar_property | 12,724 |
| partial_understanding | 10,004 |
| similar_name | 8,454 |
| overgeneralization | 7,381 |
| opposite_direction | 5,136 |
| similar_store | 1,351 |

### LEVEL_TYPE_AFFINITY

Source: `agents.py:158-163`

| Level | Preferred Misconception Types |
|-------|-------------------------------|
| L1 | similar_name, opposite_direction |
| L2 | similar_property, similar_store |
| L3-L4 | overgeneralization, partial_understanding |

**Status:** Fully enforced in vocab-backed (focused) mode. Dead code in open mode.

### Teaching Explanations Per Misconception Type

Each explanation addresses three things:
1. What the student was probably thinking
2. Why it fails here
3. The correction

Template: "You may have chosen this because [misconception_type pattern]: [misconception_label]. The key distinction: [targeted correction]."

Examples:
- **similar_name:** "These terms sound alike, but retroactive interference works backward in time while proactive works forward."
- **opposite_direction:** "This reverses the direction -- positive punishment ADDS an aversive stimulus, it doesn't remove a pleasant one."

Current state: prompt says "explain the SPECIFIC misconception" but doesn't enforce structure. Explanations are ad-hoc.

---

## 7. Stem Patterns (20 Tier-Aligned Patterns)

Replaces the original flat 5-pattern system (all Apply+) with tier-aligned pools.
Full pattern specifications in `STEM_PATTERN_SPEC.md`.

### Asymmetric Bloom's Mixing

Each tier has a **primary** Bloom's level (design target) and a **secondary** (permitted ceiling/floor).
Prompts optimize for the primary; the secondary is incidental, not a co-equal target.

| Tier | Primary | Secondary | Prompt verb |
|------|---------|-----------|-------------|
| 1 | **Remember** | Understand (ceiling) | "recall" |
| 2 | **Understand** | Apply (ceiling) | "comprehension" |
| 3 | **Apply** | Analyze (ceiling) | "application" |
| 4 | **Evaluate** | Analyze (floor) | "evaluation" |

### Variant-to-Pattern Mapping (Tier-Aware)

| Variant | Tier 1 (Remember) | Tier 2 (Understand) | Tier 3 (Apply) | Tier 4 (Evaluate) |
|---------|-------------------|---------------------|-----------------|-------------------|
| v1 | direct_definition | comparison | clinical_vignette* | contrast_prompt* |
| v2 | concept_identification | example_recognition | scenario_completion* | best_answer* |
| v3 | fact_recognition | simple_application | error_identification* | subtle_error |
| v4 | true_false_which | paraphrase | case_analysis | competing_evidence |
| v5 | feature_listing | categorization | mechanism_application | integration |

\* = existing pattern from original 5, reassigned to its natural tier.

### Pattern Cognitive Demands

**Tier 1 — Remember** (purely conceptual, no scenarios):
- `direct_definition` — "Which best defines [concept]?"
- `concept_identification` — "The term for [description] is:"
- `fact_recognition` — "According to [authority], [concept]:"
- `true_false_which` — "Which statement about [concept] is correct?"
- `feature_listing` — "Which is NOT a characteristic of [concept]?"

**Tier 2 — Understand** (comprehension, brief scenarios permitted):
- `comparison` — "How does [A] differ from [B]?"
- `example_recognition` — "Which best illustrates [concept]?"
- `simple_application` — "[Brief scenario]. This is an example of:"
- `paraphrase` — "Which statement best restates [concept]?"
- `categorization` — "Which would be classified as [category]?"

**Tier 3 — Apply** (full professional scenarios, existing patterns kept):
- `clinical_vignette` — Named clinician/client case → diagnosis/treatment/principle
- `scenario_completion` — Professional scenario → predict next step/outcome
- `error_identification` — Applied context with claims → find the error
- `case_analysis` — Detailed case → explain WHY (mechanism, not label)
- `mechanism_application` — Named principle + novel situation → predict outcome

**Tier 4 — Evaluate** (complex judgment, multi-concept required):
- `contrast_prompt` — Case where two concepts apply → which fits better in THIS context
- `best_answer` — All options contain truth → evaluate MOST correct
- `subtle_error` — Mostly-correct expert reasoning → detect nuanced flaw
- `competing_evidence` — Two defensible positions + context → which is better supported
- `integration` — Multiple concept areas → synthesize for integrated conclusion

### Legacy Production Counts (Original 5 Flat Patterns)

| Pattern | Local (27,213) | Master CSV (15,019) |
|---------|---------------|-------------------|
| clinical_vignette | 5,439 | 3,125 |
| scenario_completion | 5,348 | 2,992 |
| contrast_prompt | 5,501 | 2,879 |
| error_identification | 5,457 | 2,982 |
| best_answer | 5,468 | 3,041 |

Existing 15,019 questions keep their original stem_pattern values unchanged.

### 3 Rejected Stem Patterns

| Pattern | Domain | Rejection Reason |
|---------|--------|-----------------|
| research_interpretation | D1 | Exam-specific format, not quiz-appropriate |
| ethical_dilemma | D9 | Exam format, not learning quiz |
| developmental_sequence | D2 | Same reason |

User verbatim: "the missing patterns that the eppp heavily uses do not need to be forced. we are designing quizzes, not mock exams."

---

## 8. Difficulty Distribution Ratios

### Final Recommended Ratios (Analytics-Aware, Iteration 3)

| User Selects (Difficulty) | Tier 1 | Tier 2 | Tier 3 | Tier 4 |
|---------------------------|--------|--------|--------|--------|
| Easy | 60% | 25% | 10% | 5% |
| Medium | 10% | 55% | 25% | 10% |
| Hard | 5% | 15% | 55% | 25% |
| Expert | 5% | 10% | 25% | 60% |

### Currently Implemented (Tier 4 Restored 2026-04-25)

```
easy:   {tier1: 60, tier2: 25, tier3: 10, tier4: 5}
medium: {tier1: 10, tier2: 55, tier3: 25, tier4: 10}
hard:   {tier1: 5,  tier2: 15, tier3: 55, tier4: 25}
expert: {tier1: 5,  tier2: 10, tier3: 25, tier4: 60}
```

Source: `DIFFICULTY_DISTRIBUTION` at `__init__.py`.

### Previously Implemented (Tier 4 Suspended, superseded)

```
easy:   {tier1: 60, tier2: 30, tier3: 10}
medium: {tier1: 10, tier2: 55, tier3: 35}
hard:   {tier1: 5,  tier2: 15, tier3: 80}
```

### Ratio Evolution History

| Iteration | Changes | Problems |
|-----------|---------|----------|
| 1 (Learning-Optimized) | 0% Expert in Easy, 0% Easy in Expert | Blind spots for L4 and L1 misconception types |
| 2 (First Draft) | Added some Expert | Still had 0% Expert in Easy, 0% Easy in Expert |
| 3 (FINAL) | +5% Expert in Easy; +5% Easy in Expert; adjustments throughout | All cells have coverage |

### L-Level Exposure Per 20-Question Session (Final Ratios)

| Mode | L1 | L2 | L3 | L4 |
|------|----|----|----|----|
| Easy | 17 | 19 | 20 | 5 |
| Medium | 10 | 17 | 20 | 9 |
| Hard | 5 | 14 | 20 | 22 |
| Expert | 5 | 9 | 20 | 28 |

Every cell has 5+ encounters (minimum for Mislevy's evidence-centered design).

### Minimum-1 Floor Rule

For 10-question sessions, 5% rounds to 0. Implementation must enforce: at least 1 question from every non-zero tier. Example: 10-question Easy = 6×Tier1 + 2×Tier2 + 1×Tier3 + 1×Tier4 (not 6+3+1+0).

### Why Mix Tiers, Not Pure-Tier (6 reasons)

1. **Diagnostic precision** -- pure tier can't distinguish WHY a student fails
2. **Zone of Proximal Development (Vygotsky, 1978)** -- learning happens at boundary
3. **Self-efficacy maintenance (Bandura, 1977)** -- prevents failure streaks
4. **Retrieval practice variability (Bjork)** -- varying tier strengthens memory
5. **Floor/ceiling avoidance** -- pure tier = no variance for analytics
6. **Full diagnostic spectrum** -- every wrong answer generates misconception data at all levels

### Three Simultaneous Constraints

1. **Teaching:** 85% accuracy sweet spot (Wilson et al., 2019) -- Easy ~82%, Medium ~79%, Hard ~77%, Expert ~75%
2. **Testing:** Tier 3 never drops below 10% in any mode (guarantees EPPP-level exposure)
3. **Analytics:** Every session must generate data at all 4 distractor levels

### Distribution Design Principle

Asymmetric downward -- more below-level than above-level questions. Below-level = cheap diagnostic data + confidence. Above-level = expensive cognitive stretch.

---

## 9. Question Schema (Canonical Field List)

### Per-Question Fields (Local JSON)

```
question (top-level)
  question_id             string
  stem_pattern            string (20 values, tier-keyed — see Section 7)
  difficulty              string (easy|medium|hard|expert)
  difficulty_tier         integer (1-4)
  blooms_primary          string (remember|understand|apply|analyze|evaluate)
  blooms_secondary        string (understand|apply|analyze|evaluate)
  correct_answer_letter   string (A|B|C|D)
  tested_concept_id       string (e.g., "valproate-monitoring-hepatotoxicity")
  tested_concept_label    string
  knowledge_tested        string (prose, what the question tests)
  anchor_point_ids[]      array of strings
  section_h2              string
  subsection_h3           string
  section_uuid            string
  subsection_uuid         string
  options[]               array of 4 objects (see below)
  flashcard_seeds[]       array of 3 objects (see below)
```

### Correct Option Schema

```
option (correct)
  letter           string (A|B|C|D)
  text             string
  is_correct       boolean (true)
  concept_id       string (matches tested_concept_id)
  explanation      string (200-300 words, clinical facts, mechanisms, thresholds)
```

### Distractor Option Schema

```
option (distractor)
  letter               string
  text                 string
  is_correct           boolean (false)
  distractor_level     integer (1-4)
  concept_id           string (different from correct)
  misconception_id     string ("{correct_concept}-vs-{distractor_concept}")
  misconception_type   string (one of 6 types)
  confused_with        string (concept_id of correct answer)
  explanation          string (what student likely confused, actual correct fact)
```

### Flashcard Seed Schema (3 per question)

```
flashcard_seed
  type    string ("concept"|"comparison"|"nuance")
  front   string (57-121 chars observed)
  back    string (178-436 chars observed)
```

### Correct Answer Position Cycle (20 positions)

`B, C, A, D, C, A, D, B, C, A, D, A, C, B, D, C, B, D, A, B` (exactly 5A/5B/5C/5D)

---

## 10. Explanation & Teaching System

### Core Principle

Wrong answers are the primary teaching mechanism. Explanations for wrong answers are MORE important than the question itself.

### Correct Option Explanation

- Detailed prose explaining why correct
- Includes specific clinical facts, numerical thresholds, mechanisms of action
- Examples: therapeutic ranges ("0.6-1.2 mEq/L"), FDA categories, enzyme names, prevalences

### Distractor Option Explanation

- Explains what the student likely confused
- States the actual correct fact
- References which other concept the distractor is actually true for
- 200-300 words each

### Two-Pass Architecture

1. Opus generates the diagnostic scaffolding (stem, options, distractor metadata)
2. Sonnet writes the teaching explanations using that metadata

### Explanation Template (recommended, not yet enforced)

"You may have chosen this because [misconception_type pattern]: [misconception_label]. The key distinction: [targeted correction]."

---

## 11. Flashcard Seed System

### 3 Seed Types Per Question

| Seed Type | Purpose | Trigger Rule |
|-----------|---------|-------------|
| concept | Core factual concept | L1 errors trigger this |
| comparison | X vs Y -- what distinguishes them | L2 errors trigger this |
| nuance | Edge cases, when rule applies vs. breaks down | L3/L4 errors trigger this |

- 3 seeds per question (27,213 x 3 = ~81K seeds exist)
- Generated by LLM during question generation, embedded in question JSON

### Classic Flashcards

- **Total:** 2,936 (1,557 anchor-sourced + 1,379 subsection-sourced)
- 1 flashcard per anchor point; 1 per subsection with no anchor coverage
- **Style:** short front, concise back (~30-35 words), two-part format (definition + exam-relevant implication)
- **ID offset:** New IDs start at 100,001 (avoids collision with old 1-9,700+ range)

### Adaptive Flashcards

- **Total:** 78,749 (zero API cost -- extracted from quiz data)
- 26,754 concept + 24,878 comparison + 27,117 nuance
- 12,408 unique concepts covered
- 50% have anchor points (39,736)
- 100% have distractor diagnostics

### Flashcard Quality Validation

| Check | Current | Recommended |
|-------|---------|-------------|
| Front exists | Yes | >= 20 chars |
| Back exists | Yes | >= 40-50 chars |
| Comparison front | Not checked | Must contain "vs" or "distinguish" or "differ" |
| Nuance back | Not checked | >= 2 sentences |

### Flashcard-to-Failure Mapping

Student picks wrong answer -> misconception_id -> misconception_type -> confused_with -> stored in quiz_answers -> triggers remediation card (type based on distractor level) -> adaptive flashcard surfaces seeds -> SM-2 schedules review

---

## 12. SM-2 Remediation Algorithm

### SM-2 Parameters

- `ease_factor`: starts at 2.50 (standard SM-2 default)
- `interval_days` / `interval_minutes`
- `next_review_at` / `next_review`
- `status`: new -> learning -> review -> mastered
- `review_count`

### Spaced Repetition Schedule (Lightweight Path)

1 day -> 3 days -> 7 days -> 14 days (expanding interval)

### Supabase Fields

`flashcard_progress` table: `ease_factor`, `interval_minutes`, `next_review`, `review_count`, `updated_at`

### Remediation Cards Table

`quiz_remediation_cards`: SQL applied to Supabase with SM-2 + 3 RPCs:
- `get_remediation_cards_due`
- `review_remediation_card`
- `get_remediation_stats`

**Status:** Backend exists. No frontend review page built.

---

## 13. Generation Pipeline Architecture

### Core Pipeline Files

| File | Path |
|------|------|
| Main script | `C:\Users\mcdan\JustinQuestionsDatabase-2.0\scripts\generate_quiz_questions.py` |
| System prompts | `...\scripts\pipeline\prompts.py` |
| Config/constants | `...\scripts\pipeline\__init__.py` |
| Validation gates | `...\scripts\pipeline\gates.py` |
| Agents | `...\scripts\pipeline\agents.py` |
| Concept vocab generator | `...\scripts\generate_concept_vocab.py` |
| Build bundle | `...\scripts\build_quiz_bundle.py` |

### Prompt Architecture (8 variants)

| # | Mode | Description |
|---|------|-------------|
| 1 | `_build_system_prompt_focused()` | ~120 lines, vocab-backed |
| 2 | `_build_system_prompt_open()` | ~200 lines, no vocab |
| 3 | `_build_system_prompt_focused_slim()` | vocab-backed, no explanations (two-pass) |
| 4 | `_build_system_prompt_open_slim()` | no vocab, no explanations (two-pass) |
| 5 | `build_enrichment_system_prompt()` | Sonnet explanation pass (two-pass Phase 2b) |
| 6 | `build_enrichment_user_prompt()` | per-question skeleton with diagnostic metadata |

Dispatch: `build_system_prompt(difficulty_tier, mode="open", slim=False)`

### build_user_prompt() Signature

```python
def build_user_prompt(subsection, content, anchors, source_type, variant_num,
                      domain_name, difficulty_tier, concept_vocab=None,
                      character=None, target_position=None,
                      tested_concept=None):
```

### Content Snippet Sampling

- **Old (problematic):** `content[:2000]` -- biases toward early-introduced concepts
- **Current (fixed):** first 1200 + middle 800 + last 1000 chars (if >3000 total)

### Anti-Practice-Effect Rules

1. **Different cognitive task per variant:** 5 stem patterns per tier (20 total) enforce different mental operations
2. **Different facet per tier:** Tier 1=definition, Tier 2=key distinction, Tier 3=application, Tier 4=evaluation. `knowledge_tested` must differ across tiers for same anchor.
3. **Different surface details:** No repeated character names, settings, demographics, or scenario framing across 20 questions per anchor
4. **Different distractors:** Each variant's 3 wrong answers should pull from different misconception_ids
5. **Stem uniqueness check:** No two stems for same anchor should exceed 0.7 cosine similarity on first sentence

### Generation Sequencing (20,670 new questions)

| Priority | Description | Est. Questions |
|----------|-------------|---------------|
| P0 | 57 zero-question anchors (complete gaps) | 855 |
| P1 | Fill missing Easy-Hard slots (fewest Qs first) | ~12,000 |
| P2 | Generate all Expert questions (only after Easy-Hard solid) | 7,830 |

### Batch Sizes

- `--count 5`: generates 5 question variants per subsection call
- Batch API could cut costs 50% (deferred)

### Validation Gates

| Gate | What It Checks |
|------|----------------|
| ConsistencyGate | Misconception types validated |
| DistractorMixGate | Distractor level mix (soft enforcement) |
| (Missing) | Bloom's verb match -- NOT implemented |
| (Missing) | Stem pattern structure -- NOT validated |
| (Missing) | confused_with vs correct concept_id -- NOT validated |

### Position Randomization

20-position balanced cycle: 5A/5B/5C/5D. Prevents test-taking tricks.

### Name Bias Prevention

3,701-name pool + deterministic hashing eliminates LLM defaults.

### Permanent Failure Tracking

After 3 failed attempts per question_id -> add to `quiz_failures.json` and skip on future runs.

---

## 14. Model Routing & Cost Estimates

### Model Routing

- `--fast-tiers` flag: Tier 1-2 to Sonnet 4.6, Tier 3-4 to Opus 4.6
- Two-pass mode: Opus generates skeleton, Sonnet writes explanations
- User preference: Sonnet 4.6 for elaborations, Opus 4.6 for generation

### Cost Estimates -- Question Generation (~30K questions)

| Strategy | Cost |
|----------|------|
| All Opus | ~$4,000 |
| Opus + prompt caching | ~$3,100 |
| Sonnet Easy-Medium + Opus Hard-Expert | ~$2,400 |
| All Sonnet | ~$800 |
| All Sonnet + prompt caching | ~$650 |
| **Recommended: Sonnet Easy-Medium + Opus Hard-Expert, two-pass** | **~$1,200-1,500** |

### Cost Estimates -- 20,670 New Questions

| Approach | Cost |
|----------|------|
| All Sonnet + prompt caching (~$0.04/q) | ~$827 |
| All Opus + prompt caching (~$0.10/q) | ~$2,067 |
| **Recommended: Sonnet Easy-Medium + Opus Hard-Expert** | **~$1,200-1,500** |

### Per-Question Cost (Actual)

- Opus actual: ~$0.13/question (2,900 input + 1,200 output tokens)
- Original estimate: ~$0.01/question (was ~10x off)

### Prompt Caching

- System prompt: 2,100 tokens (sent with every question)
- Caching saves ~$840 on input tokens
- Implementation: one extra parameter (`cache_control`) on system message

### Other Cost Estimates

| Operation | Cost |
|-----------|------|
| Concept vocab generation (all domains) | $9-45 |
| Classic flashcard generation (2,936 cards, Opus) | ~$47 |
| Classic flashcard generation (batched) | ~$15-20 |
| Classic flashcard generation (Sonnet) | ~$5-8 |
| Adaptive flashcard extraction | $0 |
| Concept vocab pre-seeding (~1,493 subsections) | ~$30 |
| Anchor enrichment pipeline (all 9 domains) | ~$25 |
| Anchor enrichment pilot (PMET only) | ~$3.50 |

### V1 Test Batch (2026-04-24)

- 100 questions, Opus 4.6, $10.87, OLD anchors (no testable_fact)
- 5 anchor points tested across PMET, LDEV, CASS, WDEV, PETH

### V2 Test Batch (2026-04-24)

- 20 questions, Opus 4.6, NEW anchors with testable_fact
- 62 columns (current schema)
- Purpose: colleague quality review comparing V1 vs V2 anchor quality

---

## 15. Concept Vocabulary Pre-seeding

### The Problem

Each variant generated independently. No mechanism ensures variant 1's `concept_id: "sensory-memory-iconic-duration"` matches variant 3's `concept_id: "iconic-memory-sensory-duration"`.

### What Breaks Without It

- Per-concept mastery tracking sees 5 "different" concepts instead of 1
- Confusion matrix is fragmented
- Flashcard deduplication fails
- Entire error-driven analytics chain breaks silently

### Two Generation Modes

| Mode | Behavior |
|------|---------|
| Open | LLM invents concept_ids on the fly (current default) |
| Focused | concept_ids are canonical, pre-seeded by assembler |

### Pre-seeding Approach (Option A -- Most Practical)

Before generation, extract 3-5 key `concept_ids` per subsection from anchor point content + h3 title. Pass them in user prompt: "You MUST use concept_ids from this list: [...]"

Cost: ~$0.02/subsection x 1,493 subsections = ~$30. Insurance against $4,000 run producing fragmented IDs.

### Script

`generate_concept_vocab.py` exists but has never been run. Must run before generating more questions.

### Concept Vocabulary Integration in Prompts

- When `concept_vocab` provided with `has_vocab=True`, adds `## Canonical Concept Vocabulary (MUST USE THESE IDs)` section
- When `tested_concept` provided, adds `**TARGET CONCEPT TO TEST**` directive
- Lists all concept_ids and misconception_ids from pre-seeded vocabulary

### Rejected Alternatives

| Option | Assessment |
|--------|-----------|
| B. Post-generation normalization (clustering) | Extra pass, less deterministic |
| C. Two-pass generation (sequential) | Requires sequential generation, slower |

---

## 16. Domain Isolation Rules

### Domain Definitions and Content Restrictions

| Domain | Code | Focus | Content Restriction |
|--------|------|-------|-------------------|
| D1 | PMET | Statistical reasoning & validity | All methodology and measurement. CC/OC learning theory leads. |
| D2 | LDEV | Stage theories & critical periods | All normal developmental stages only |
| D3 | CPAT | Genetics & etiology | Pathology, etiology, diagnostic criteria ONLY. **ZERO treatment, ZERO drugs** |
| D4 | PTHE | Mechanisms of change | All therapy techniques and change mechanisms |
| D5 | SOCU | Identity & group dynamics | All social cognition, group processes, cultural identity |
| D6 | WDEV | Human performance & systems | All organizational/I-O psychology. **ZERO clinical content** |
| D7 | BPSY | Brain anatomy & neural circuits | All structure-function. **ZERO drug names** |
| D8 | CASS | Clinical interpretation & profiles | All test interpretation, assessment ethics (Standards 9-10), clinical profiles, forensic |
| D9 | PETH | Drug mechanisms & ethical decision-making | Ethics Standards 1-8 + **ALL pharmacology. Every drug name goes here** |

### Cross-Domain Notes

- D8 CASS has 9 disorder-specific anchors already assigned in source CSV (substance treatment, sexual dysfunction treatment) -- NOT moved by V3 schema
- D9 PETH has two interleaved strands: Ethics (80 anchors, Ch01-08) then Pharmacology (48 anchors, Ch09-15)
- **All pharmacology** routed exclusively to D9 -- no other domain discusses drugs
- No anchors moved between domains -- all 1,081 domain assignments from source CSV preserved
- Every domain_code/domain_name pair verified consistent -- zero mismatches

---

## 17. Chapter Schema (V3) -- 121 Chapters

### Global Summary

| Domain | Code | Anchors | Chapters | Avg/Ch |
|--------|------|---------|----------|--------|
| D1 Psychometrics & Research | PMET | 139 | 15 | 9.3 |
| D2 Lifespan Development | LDEV | 113 | 13 | 8.7 |
| D3 Clinical Psychopathology | CPAT | 87 | 9 | 9.7 |
| D4 Psychotherapy Models | PTHE | 111 | 13 | 8.5 |
| D5 Social & Cultural | SOCU | 120 | 13 | 9.2 |
| D6 Workforce Development | WDEV | 122 | 14 | 8.7 |
| D7 Biopsychology | BPSY | 127 | 14 | 9.1 |
| D8 Clinical Assessment | CASS | 134 | 15 | 8.9 |
| D9 Pharmacology & Ethics | PETH | 128 | 15 | 8.5 |
| **Total** | | **1,081** | **121** | **8.9** |

### V3 Design Constraints

- Max anchors per chapter: 10 (hard ceiling)
- Anchor floor per chapter: 5 (BPSY floor)
- Header cap: ~20 H2+H3 combined per chapter
- Word cap: ~5,000 words per chapter
- Chapter names tailored AFTER content is written

### D1: PMET -- Psychometrics & Research Methods (139 anchors / 15 chapters)

| Ch | Title | ~N | Bloom's | DOK |
|----|-------|----|---------|-----|
| 01 | How Associations Form: Classical Conditioning Foundations | 10 | Remember/Understand | 1-2 |
| 02 | When Learning Gets Complex: Higher-Order Conditioning, Blocking, and Extinction | 5 | Understand/Apply | 2 |
| 03 | Consequences That Shape Behavior: Reinforcement, Punishment, and Thorndike | 10 | Understand | 2 |
| 04 | Schedules, Shaping, and Stimulus Control | 10 | Apply | 2-3 |
| 05 | Applied Operant Principles: Discrimination, Two-Factor Theory, and Bridges to Practice | 2 | Apply/Analyze | 2-3 |
| 06 | Relationships in Data: Correlation and Bivariate Statistics | 10 | Understand/Apply | 2-3 |
| 07 | Prediction and Complexity: Regression and Multivariate Methods | 7 | Analyze | 3 |
| 08 | Testing Ideas: Sampling Distributions, Hypothesis Testing, and Power | 10 | Understand/Apply | 2-3 |
| 09 | Choosing the Right Test: t-Tests, ANOVA, and Post-Hoc Comparisons | 10 | Apply/Analyze | 2-3 |
| 10 | Advanced Tests: Chi-Square, MANOVA, and Clinical Significance | 9 | Analyze | 3 |
| 11 | Threats from Within and Without: Research Design and Validity | 10 | Analyze/Evaluate | 3-4 |
| 12 | Building Consistent Measures: Classical Test Theory and Reliability Foundations | 10 | Understand/Apply | 2-3 |
| 13 | Reliability in Practice: Methods, Agreement, and Factors | 9 | Apply/Analyze | 3 |
| 14 | Measuring What Matters: Content Validity, Construct Validity, and Factor Analysis | 10 | Analyze | 3 |
| 15 | Predicting Outcomes: Criterion Validity, Decision Accuracy, and Cross-Validation | 10 | Analyze/Evaluate | 3-4 |

**Prerequisite chains:**
- CC core -> generalization -> blocking/overshadowing -> extinction
- Thorndike -> reinforcement/punishment -> schedules -> stimulus control
- Bivariate correlation -> regression -> multivariate
- Sampling distribution -> hypothesis testing -> t-tests -> ANOVA
- CTT -> reliability methods -> SEM -> validity framework -> criterion prediction

### D2: LDEV -- Lifespan Development (113 anchors / 13 chapters)

| Ch | Title | ~N | Bloom's | DOK |
|----|-------|----|---------|-----|
| 01 | The Architecture of Development: Genes, Heritability, and Ecological Systems | 10 | Remember/Understand | 1-2 |
| 02 | Blueprints and Beginnings: Prenatal Development, Teratogens, and Genetic Disorders | 10 | Remember/Understand | 1-2 |
| 03 | Gene-Environment Interplay and Early Physical Development | 8 | Understand | 2 |
| 04 | The Sensorimotor World: Piaget's First Stage and Infant Cognition | 10 | Understand | 2 |
| 05 | Finding Words: Language Acquisition from Babbling to Grammar | 9 | Understand/Apply | 2 |
| 06 | Language in Context: Pragmatics, Bilingualism, and Vygotsky | 8 | Apply/Analyze | 2-3 |
| 07 | Bonds That Shape Us: Attachment, Temperament, and Early Emotion | 10 | Understand/Apply | 2 |
| 08 | Discovering the World: Preoperational Thought and Early Childhood | 9 | Apply | 2-3 |
| 09 | Building Competence: Concrete Operations, Peers, and Moral Reasoning | 9 | Apply/Analyze | 2-3 |
| 10 | Transformation: Puberty, Identity, and the Adolescent Mind | 9 | Analyze | 3 |
| 11 | Who We Become: Gender, Personality, and Identity Across the Lifespan | 8 | Analyze | 3 |
| 12 | The Adult Journey: Cognition, Relationships, and Generativity | 8 | Analyze/Evaluate | 3 |
| 13 | The Long Horizon: Aging, Wisdom, and the Final Chapter | 5 | Evaluate | 3-4 |

**Prerequisite chains:**
- Genetics -> heritability -> gene-environment -> epigenetics -> ecological systems
- Piaget core -> sensorimotor -> preoperational -> concrete -> formal
- Phonemes/morphemes -> babbling -> first words -> syntax -> pragmatics
- Bowlby -> Harlow -> Ainsworth -> attachment patterns -> internal working models
- Piaget moral -> Kohlberg moral (sequential)

### D3: CPAT -- Clinical Psychopathology (87 anchors / 9 chapters)

| Ch | Title | ~N | Bloom's | DOK |
|----|-------|----|---------|-----|
| 01 | Early-Onset Conditions: ADHD, Autism Spectrum, and Neurodevelopmental Disorders | 9 | Remember/Understand | 1-2 |
| 02 | Fear Without Cause: Anxiety Disorders, OCD, and the Anxiety Spectrum | 10 | Understand | 2 |
| 03 | When Darkness Descends: Major Depression, Persistent Depressive Disorder, and DMDD | 10 | Understand/Apply | 2-3 |
| 04 | The Pendulum Swings: Bipolar Spectrum, Suicide Risk, and Mood Disorder Integration | 10 | Apply/Analyze | 3 |
| 05 | Fractured Reality: Schizophrenia Spectrum and Psychotic Disorders | 8 | Analyze | 3 |
| 06 | Defiance and Destruction: Conduct Disorder, Impulse Control, and Antisocial Trajectories | 10 | Apply/Analyze | 3 |
| 07 | When the World Breaks In: Trauma, Dissociation, and Somatic Responses | 10 | Analyze | 3 |
| 08 | Enduring Patterns: Personality Disorders and Remaining Mood Concepts | 10 | Evaluate | 3-4 |
| 09 | The Remaining Spectrum: Eating, Sleep, Sexual, Substance, and Remaining Disorders | 10 | Evaluate | 3-4 |

**Prerequisite chains:**
- Separation anxiety -> specific phobia -> social anxiety -> agoraphobia/panic -> GAD
- MDD -> PDD -> DMDD; Bipolar I -> II -> cyclothymia
- Brief psychotic -> schizophreniform -> schizophrenia -> schizoaffective
- ODD -> CD -> ASPD developmental trajectory

### D4: PTHE -- Psychotherapy Models (111 anchors / 13 chapters)

| Ch | Title | ~N | Bloom's | DOK |
|----|-------|----|---------|-----|
| 01 | Through the Unconscious: Psychodynamic Foundations and Defense Mechanisms | 8 | Remember/Understand | 1-2 |
| 02 | Through the Relationship: Rogers, Person-Centered Therapy, and Humanistic Approaches | 8 | Understand | 2 |
| 03 | Through Classical Conditioning: Desensitization, Flooding, and Exposure | 8 | Apply | 2-3 |
| 04 | Through Operant Principles: Shaping, Token Economies, and Behavioral Activation | 8 | Apply | 2-3 |
| 05 | Through Thoughts: Beck's CBT, Cognitive Distortions, and Collaborative Empiricism | 9 | Apply/Analyze | 2-3 |
| 06 | Through Acceptance: ACT, MBCT, and Mindfulness-Based Approaches | 8 | Analyze | 3 |
| 07 | Through Motivation: Transtheoretical Model, MI, and SFBT | 10 | Apply/Analyze | 2-3 |
| 08 | Through Connection: IPT, Brief Psychodynamic, and Short-Term Approaches | 7 | Analyze | 3 |
| 09 | Through the System: Feedback Loops, Homeostasis, and Structural Family Therapy | 10 | Understand/Apply | 2-3 |
| 10 | Families in Motion: Strategic, Bowenian, Milan, and Narrative Approaches | 10 | Analyze | 3 |
| 11 | Healing Together: Couples Therapy, EFT, Group Process, and Evidence-Based Family Work | 10 | Analyze/Evaluate | 3 |
| 12 | Before the Crisis: Prevention, Consultation, and Community Intervention | 8 | Evaluate | 3-4 |
| 13 | What Works and for Whom: Common Factors, ESTs, and Psychotherapy Research | 7 | Evaluate | 4 |

**Prerequisite chains:**
- Defense mechanisms/transference -> all therapy models
- CC principles (D1) -> systematic desensitization -> exposure -> implosive therapy
- OC principles (D1) -> shaping -> token economies -> behavioral activation
- Beck's CBT -> ACT/MBCT (third-wave builds on traditional CBT)
- Systems theory -> ALL family therapy models
- TTM stages -> MI -> SFBT -> IPT

### D5: SOCU -- Social & Cultural Psychology (120 anchors / 13 chapters)

| Ch | Title | ~N | Bloom's | DOK |
|----|-------|----|---------|-----|
| 01 | How We Judge: Attribution Theory, Biases, and Causal Explanations | 7 | Remember/Understand | 1-2 |
| 02 | The Shortcuts We Take: Cognitive Heuristics and Impression Formation | 10 | Understand | 2 |
| 03 | What We Believe: Attitudes, Cognitive Dissonance, and Self-Perception | 10 | Understand/Apply | 2 |
| 04 | How We're Moved: Persuasion, Social Influence, and Compliance | 9 | Apply | 2-3 |
| 05 | The Power of the Situation: Conformity, Obedience, and Group Dynamics | 9 | Apply/Analyze | 2-3 |
| 06 | Why We Connect: Attraction, Love, Altruism, and Helping Behavior | 9 | Apply | 2-3 |
| 07 | Where We Break: Prejudice, Stereotypes, Aggression, and Discrimination | 9 | Analyze | 3 |
| 08 | Foundations of Culture: Etic, Emic, and Communication Across Cultures | 9 | Remember/Understand | 1-2 |
| 09 | Navigating Two Worlds: Acculturation, Minority Stress, and Worldview | 9 | Understand/Apply | 2-3 |
| 10 | The Wounds of Racism: Internalized Oppression, Microaggressions, and Bias | 9 | Analyze | 3 |
| 11 | Culturally Responsive Practice: Therapeutic Adaptations and Client Preferences | 9 | Apply/Analyze | 3 |
| 12 | Identity in Development: Cross, Helms, Atkinson, and Racial Identity Models | 10 | Analyze | 3 |
| 13 | Becoming Multicultural: Sue's Framework, Competence, and Remaining Attitudes | 10 | Analyze/Evaluate | 3-4 |

**Prerequisite chains:**
- Attribution -> self-serving bias -> ultimate attribution error
- Heuristics -> attitudes -> dissonance -> self-perception
- Attitudes -> ELM/persuasion -> compliance -> conformity
- Social identity -> in-group/out-group -> prejudice -> microaggressions
- Etic/emic -> acculturation -> worldview -> multicultural competence

### D6: WDEV -- Workforce Development & Leadership (122 anchors / 14 chapters)

| Ch | Title | ~N | Bloom's | DOK |
|----|-------|----|---------|-----|
| 01 | Understanding Organizations: Classical, Human Relations, and Systems Theory | 8 | Remember/Understand | 1-2 |
| 02 | Navigating Career Paths: Super, Dawis, and Career Decision-Making | 9 | Understand | 2 |
| 03 | Defining the Work: Job Analysis Methods and Foundations | 9 | Understand/Apply | 2 |
| 04 | Measuring Performance: Appraisal Methods and Rating Errors | 9 | Apply | 2-3 |
| 05 | Keeping It Fair: Adverse Impact, Criterion Issues, and Legal Foundations | 8 | Apply/Analyze | 3 |
| 06 | Finding the Right People: Selection Methods and Assessment Centers | 9 | Apply | 2-3 |
| 07 | Is the Test Worth It? Validity, Utility, and Selection Decision Models | 9 | Analyze | 3 |
| 08 | Optimizing Selection: Banding, Incremental Validity, and Advanced Problems | 8 | Analyze/Evaluate | 3-4 |
| 09 | What Drives Us: Content Theories of Motivation | 8 | Understand/Apply | 2 |
| 10 | The Mechanics of Motivation: Process Theories, Equity, and Goal Setting | 8 | Apply/Analyze | 2-3 |
| 11 | Leading People: Traits, Contingency, and Transformational Leadership | 8 | Apply/Analyze | 2-3 |
| 12 | Making Better Decisions: Bounded Rationality, Groupthink, and Decision Models | 8 | Analyze | 3 |
| 13 | Transforming Organizations: Change Models, OD Interventions, and Team Building | 10 | Analyze/Evaluate | 3-4 |
| 14 | Developing Talent: Training, Evaluation, and Workplace Well-Being | 10 | Evaluate | 3-4 |

**Prerequisite chains:**
- Job analysis -> ALL personnel functions (selection, performance, training)
- Performance appraisal -> criterion -> selection evaluation
- Content motivation theories -> process theories -> leadership
- Decision-making -> OD/change

### D7: BPSY -- Biopsychology (127 anchors / 14 chapters)

| Ch | Title | ~N | Bloom's | DOK |
|----|-------|----|---------|-----|
| 01 | The Language of the Brain: Neurons, Neurotransmitters, and Neural Signaling | 7 | Remember | 1 |
| 02 | Below the Cortex I: Thalamus, Hypothalamus, and the Limbic System | 10 | Remember/Understand | 1-2 |
| 03 | Below the Cortex II: Basal Ganglia, Brainstem, and Cerebellum | 10 | Understand | 2 |
| 04 | The Vital Infrastructure: Endocrine System, HPA Axis, and Neuroimaging | 6 | Understand | 2 |
| 05 | The Frontal Lobe: Executive Function, Motor Control, and Prefrontal Syndromes | 9 | Understand/Apply | 2 |
| 06 | Parietal and Temporal Lobes: Sensation, Language Areas, and the Agnosias | 9 | Apply | 2-3 |
| 07 | Language, Lateralization, and the Split Brain | 9 | Analyze | 3 |
| 08 | Memory Architecture: From Sensory Store to Long-Term Memory | 10 | Understand | 2 |
| 09 | How We Learn and Remember: Encoding, Retrieval, and Mnemonics | 10 | Apply | 2-3 |
| 10 | How Memory Fails: Forgetting, Interference, and False Memory | 9 | Analyze | 3 |
| 11 | The Biology of Memory and Sleep | 10 | Analyze | 3 |
| 12 | Sleep Across the Lifespan and Sensation/Perception | 7 | Apply/Analyze | 3 |
| 13 | Emotion, Arousal, and Stress: The Feeling Brain | 10 | Analyze | 3 |
| 14 | When the Brain Breaks: Neurocognitive and Neurological Disorders | 10 | Evaluate | 3-4 |

**Prerequisite chains:**
- Neurotransmitters -> all brain anatomy -> all pharmacology (D9)
- Subcortical (limbic -> basal ganglia -> brainstem) -> cortical
- Cortex lobes -> lateralization -> split-brain -> aphasias/agnosias
- Memory architecture -> encoding -> forgetting -> LTP/consolidation -> sleep
- Emotion theories -> neural substrates -> stress

### D8: CASS -- Clinical Assessment & Interpretation (134 anchors / 15 chapters)

| Ch | Title | ~N | Bloom's | DOK |
|----|-------|----|---------|-----|
| 01 | Assessment Ethics I: Standard 9 | 9 | Remember/Understand | 1-2 |
| 02 | Assessment Ethics II: Automated Scoring, Special Populations | 9 | Understand | 2 |
| 03 | Therapy Ethics: Standard 10 | 8 | Understand/Apply | 2 |
| 04 | Remaining Ethics and Score Interpretation Foundations | 7 | Apply | 2-3 |
| 05 | Law and Practice I: Licensure, Malpractice | 9 | Understand/Apply | 2-3 |
| 06 | Law and Practice II: Forensic Psychology, Competency | 9 | Analyze | 3 |
| 07 | Clinical Supervision: Models, Gatekeeping | 7 | Apply/Analyze | 3 |
| 08 | Measuring Intelligence: The Wechsler Scales | 9 | Understand/Apply | 2-3 |
| 09 | Beyond Wechsler: Stanford-Binet, CHC Theory | 9 | Apply/Analyze | 3 |
| 10 | Specialized Cognitive and Achievement Measures | 8 | Apply | 2-3 |
| 11 | The MMPI-2: Development, Scales, and Profile Interpretation | 9 | Apply/Analyze | 3 |
| 12 | Personality Assessment: Projective, Behavioral, and Self-Report | 9 | Analyze | 3 |
| 13 | Neuropsychological Assessment: Batteries, Screening | 9 | Analyze | 3 |
| 14 | Vocational Assessment and Interest Inventories | 8 | Apply | 2-3 |
| 15 | Disorder-Specific Treatment and Clinical Integration | 9 | Evaluate | 3-4 |

**Prerequisite chains:**
- Ethics framework -> all test-specific chapters
- Licensure/malpractice -> forensic applications
- Intelligence theory -> Wechsler -> Stanford-Binet -> other cognitive
- MMPI-2 development -> scales -> profile interpretation

### D9: PETH -- Pharmacology & Ethics (128 anchors / 15 chapters)

| Ch | Title | ~N | Bloom's | DOK |
|----|-------|----|---------|-----|
| 01 | Ethics Foundation: APA Principles, Code Structure | 9 | Remember/Understand | 1-2 |
| 02 | Standard 2: Competence Boundaries, Emergency Services | 9 | Understand/Apply | 2 |
| 03 | Standard 2 Applied: Scope of Practice, Supervision | 9 | Apply | 2-3 |
| 04 | Standard 3: Human Relations, Multiple Relationships | 10 | Understand/Apply | 2 |
| 05 | Standard 3 Applied: Conflicts of Interest, Boundaries | 10 | Apply/Analyze | 2-3 |
| 06 | Standard 4: Confidentiality, Privacy, Limits of Privilege | 10 | Understand/Apply | 2-3 |
| 07 | Standard 4 Applied: Minors, Records, Court Orders | 9 | Analyze | 3 |
| 08 | Standards 5-8: Advertising, Fees, Records, Research | 10 | Apply | 2-3 |
| 09 | The Chemistry of the Mind: Neurotransmitter Systems | 8 | Remember/Understand | 1-2 |
| 10 | Antidepressants I: SSRIs, SNRIs, and Serotonin Syndrome | 9 | Understand/Apply | 2-3 |
| 11 | Antidepressants II: TCAs, MAOIs | 8 | Apply | 2-3 |
| 12 | Antipsychotics: Conventional, Atypical, and Clozapine | 9 | Apply/Analyze | 3 |
| 13 | Mood Stabilizers, Anxiolytics, and Stimulants | 9 | Apply | 2-3 |
| 14 | Specialty Medications: Cognitive Enhancers, Substance Use Drugs | 8 | Analyze | 3 |
| 15 | Pharmacology in Practice: Disorder-Specific Prescribing | 10 | Evaluate | 3-4 |

**Two strands:** Ethics (80 anchors, Ch01-08) -> Pharmacology (48 anchors, Ch09-15)

**Prerequisite chains:**
- General Principles -> Standards 1-2 -> Standards 3-4 -> Standards 5-8
- Neurotransmitter review (from D7) -> pharmacokinetics -> specific drug classes
- SSRIs -> SNRIs -> TCAs -> MAOIs -> drug interactions
- Conventional antipsychotics -> atypical -> clozapine -> side effects

---

## 18. Pedagogical Framework

### V3 Writing Instructions

- Bloom's Taxonomy or Webb's DOK used strategically throughout each chapter
- Frontloading: core vocabulary and prerequisite concepts in opening chapters
- Scaffolding: each chapter builds complexity on prior knowledge
- Clustering: semantically related concepts grouped within same chapter
- Spiral Curriculum: key concepts introduced at basic level, revisited at increasing depth
- Progressive language: early chapters use simpler explanations, later assume developed vocabulary
- Strong chapter openers, transitions, closers
- Learning objectives + key takeaways per chapter
- Quality target: "better than Pearson level"

### Research Principles Embedded in Generation

| Principle | Implementation | Status |
|-----------|---------------|--------|
| Retrieval Practice (Roediger & Karpicke, 2006) | In prompts + code | Active |
| Error-Driven Learning (Metcalfe, 2017) | In prompts + code | Active |
| Desirable Difficulties (Bjork, 1994) | In prompts + code | Active |
| Transfer-Appropriate Processing (Morris et al., 1977) | 5 stem patterns | Active |
| Bloom's Taxonomy | Difficulty mapping + verbs | Active |
| Interleaving (Rohrer, 2007) | Stem rotation | Active |
| Dual Coding | Flashcard seeds | Seeds not surfaced |
| Elaborative Interrogation | Comparison flashcards | Seeds not surfaced |
| ZPD (Vygotsky, 1978) | Difficulty mixing | No adaptive adjustment |
| 85% Accuracy Rule (Wilson et al., 2019) | Informed ratios only | Not enforced |
| Self-Efficacy (Bandura, 1977) | Asymmetric mixing only | Not enforced |
| Spaced Repetition (Cepeda et al., 2006) | SM-2 in DB + localStorage | No review page |

### 4-Dimensional Student Failure Profile

`difficulty x bloom's x stem_pattern x misconception_type`

A student who gets hard + apply + clinical_vignette + overgeneralization wrong needs different remediation than easy + remember + best_answer + similar_name.

### Error-Driven Learning Data Flow

1. Student picks wrong answer
2. -> misconception_id, misconception_type, confused_with captured
3. -> stored in quiz_answers with time_to_answer_ms, bloom's, difficulty, stem_pattern
4. -> triggers remediation card (type based on distractor level)
5. -> adaptive flashcard surfaces concept/comparison/nuance seeds
6. -> SM-2 schedules review

---

## 19. Quiz Schema (quiz_attempts, quiz_answers)

### quiz_answers Table (23+ columns)

- selected_letter / selected_option_letter
- is_correct
- time_to_answer_ms
- difficulty_tier
- blooms_level
- stem_pattern
- misconception_id / selected_misconception_id
- distractor_level / selected_distractor_level
- concept_id_correct / correct_concept_id
- selected_concept_id
- subsection_uuid
- anchor_point_ids[] (array)
- domain_code
- chapter_file
- confidence_rating
- answer_changed
- initial_answer_letter
- question_variant
- chapter_section_uuid
- tested_concept_id

### quiz_attempts Table

- quiz_type (from migration 010)
- curved_distribution (from migration 010)
- chapter_section_uuid (from migration 010)
- selected_difficulty (recommended)
- chapter_file (recommended)
- time_total_ms (recommended)
- questions_answered (recommended)
- abandonment_point (recommended)

### Column Naming Conflicts (Migrations 001-007 vs sql/20)

| Old Name | sql/20 Name | Same Concept |
|----------|-------------|-------------|
| distractor_level | selected_distractor_level | Yes |
| misconception_id | selected_misconception_id | Yes |
| concept_id_selected | selected_concept_id | Yes |
| concept_id_correct | correct_concept_id | Yes |

### enrichment_questions Table

- Flat 62-column schema matching enrichment_all_questions.csv (no JSONB nesting)
- Primary key: question_id (TEXT)
- Upload script: JustinQuestionsDatabase-2.0/scripts/upload_quiz_questions.py
- Migration: PassEPPP-Database/migrations/013_enrichment_questions_table.sql

### Current Frontend Capture (inadequate)

```json
{
  "attempt_id": "...",
  "question_id": "...",
  "question_position": 0,
  "selected_options": ["a"],
  "is_correct": true,
  "answered_at": "..."
}
```

**The problem:** "You built the most sophisticated diagnostic distractor system -- then throw away the diagnostic data at the point of capture."

### Recommended Denormalized quiz_answers (write-time)

```json
{
  "attempt_id": "...",
  "question_id": "...",
  "selected_option_letter": "B",
  "is_correct": false,
  "difficulty_tier": 3,
  "blooms_level": "apply",
  "selected_distractor_level": 3,
  "selected_misconception_id": "iconic-vs-echoic-duration",
  "selected_misconception_type": "similar_store",
  "selected_concept_id": "sensory-memory-echoic-duration",
  "time_to_answer_ms": 14200,
  "answered_at": "..."
}
```

---

## 20. Analytics & Diagnostic Matrices

### Time-to-Answer Matrix

| | Fast | Slow |
|---|------|------|
| **Correct** | Mastered (automaticity) | Shaky (effortful retrieval) |
| **Wrong** | Careless error or guess | Genuine confusion |

80% correct in 8 seconds vs 80% correct in 45 seconds = very different mastery levels.

### Confidence Rating Matrix

| | Confident | Not Confident |
|---|-----------|---------------|
| **Correct** | Mastered | Lucky / shaky |
| **Wrong** | **MISCONCEPTION (highest priority)** | Knowledge gap (normal) |

Wrong + confident = actively held misconception. Combined with diagnostics enables: "You were confident that echoic memory lasts ~500ms, but that's iconic memory. You've made this confusion 3 times across 2 quizzes."

### Diagnostic Separation (enabled by mixing)

When Hard-mode student gets 60% wrong:
- Missed Easy questions? -> Foundation gap, redirect to review
- Nailed Easy-Medium but failed Expert? -> Discrimination gap, need more nuance work

### Readiness Signaling

5% minority-difficulty probes (Expert in Easy, Easy in Expert) function as high-information probes. Base rate makes unexpected success strongly diagnostic.

### IRT vs Current Diagnostic System

| Dimension | IRT | Current System |
|-----------|-----|---------------|
| Output | Single theta per domain | concept_id + misconception_type + confused_with + distractor_level + bloom's + stem_pattern + time + confidence |
| Purpose | Rank candidates for pass/fail | "What does this person need to learn, why, and fastest fix path?" |
| Requirement | Hundreds of students to calibrate | Works from session 1 |
| EPPP context | Progress bar | Targeted remediation |

### Empirical Difficulty Calibration (Future)

- Columns exist (`empirical_difficulty`, `discrimination_index`) but never populated
- Needs ~50 serves per question for meaningful statistics
- Would identify: difficulty reclassification, broken distractors, non-discriminating questions

### Confused_with Graph (Future)

- Every wrong option stores `confused_with` = concept_id of correct answer
- Creates directed confusion graph: "When tested on X, students confuse it with Y and Z"
- Currently stored in JSONB but never extracted or surfaced (P3)

---

## 21. Supabase Architecture

### Supabase Project IDs

| Purpose | Project ID | Key Type |
|---------|-----------|----------|
| Content DB | `hjfrqlfltqfhrwjchhip` | service_role (server-side only) |
| Main App DB | `ybvklratgtxzhafhycvd` | anon key (client-side) |

### Migration 010 -- Consolidated (1,283 lines)

| Part | Creates | Details |
|------|---------|---------|
| 1 | 4 new columns | quiz_type, curved_distribution, chapter_section_uuid on quiz_attempts; anchor_point_ids on quiz_answers |
| 2 | 2 RPCs | create_quiz_attempt_v2, record_quiz_answer_v2 |
| 3 | 5 analytics RPCs | blooms, source, distractor, difficulty progression, chapter mastery |
| 4 | 1 table + 2 RPCs | anchor_density, insert_anchor_density_batch, get_study_recommendations |
| 5 | 1 table + 5 RPCs | auto_flashcard_decks, struggle topics, flashcard assembly, deck CRUD |
| 6 | 2 tables + 5 RPCs | video_catalog, video_recommendations, catalog batch, push/update/history |
| 7 | 1 table + 2 RPCs | misconception_registry, batch insert, user misconception profile |
| 8 | 3 RPCs | insert_quiz_questions_batch, assemble_flashcard_decks_for_user, on_quiz_complete |

**Totals:** 5 new tables, 24 RPCs, 3 new columns. Wrapped in BEGIN/COMMIT.

**Prerequisites:** sql/20, sql/21, sql/22 already applied. textbook_sections table exists.

**Status:** Migrations 001-007 were NEVER applied to Supabase. All analytics, remediation, flashcard assembly, video, misconception systems are not live.

### RLS Changes (2026-04-24)

| Table | Before | After |
|-------|--------|-------|
| textbook_2 (121 rows) | RLS DISABLED | RLS ON, zero public policies (service_role only) |
| textbook_quizzes (1,666 rows) | RLS DISABLED | RLS ON, zero public policies (service_role only) |
| v1_section_hierarchy | ALL to anon | SELECT only (public read for flashcards.js) |
| flashcard_sections | ALL to anon + public INSERT/UPDATE | SELECT only |

### enrichment_questions Table (Current)

- Flat 62-column schema + `created_at`, `updated_at`
- Primary key: question_id (TEXT)
- 12,703 rows (master CSV), pending re-upload after column rename

### RPCs Used by Enrichment Quiz

- `create_quiz_attempt_v2()`
- `record_quiz_answer_v2()`
- `upsert_remediation_card()`
- `on_quiz_complete()`

### Frontend-to-enrichment_questions Connection

Frontend (`quiz-enrichment.html`) reads directly from `enrichment_questions` table via `sb.from('enrichment_questions')`.

- `quiz-enrichment.html` loads from `enrichment_questions` (flat columns, not JSONB)
- `quiz-questions.js` loads exam questions from `questions` table (separate system)

---

## 22. Current Pipeline State (as of 2026-04-24)

### What Has Been Done

1. RLS lockdown applied (textbook_2, textbook_quizzes now service_role only)
2. Variant backfill completed (all 15,019 rows have variant numbers in CSV and Supabase)
3. testable_fact integrated (new column in anchor CSV, wired into all generation prompts)
4. Canonical anchor source updated (`Master CSVs\anchor_points.csv`, 10 columns)
5. Old anchor files archived/deprecated
6. V1 (100q) and V2 (20q) test batches generated for colleague review
7. Generation priority order set: P0 (57 zeros) -> P1 (fill Easy-Hard) -> P2 (all Expert)
8. 5 stem patterns locked in (3 additional rejected)
9. Cost model chosen: Sonnet + caching for bulk, Opus for Hard-Expert
10. Content snippet bias fixed: first 1200 + middle 800 + last 1000 chars

### What Is Pending Colleague Review

- V1 test batch (100 questions, old anchors) vs V2 test batch (20 questions, new anchors with testable_fact)
- Full generation run waits on quality approval

### Files Updated (2026-04-24)

- `master/test_batch_generate.py` -- loads from Master CSV, includes testable_fact
- `master/test_batch_v2.py` -- written with new CSV
- `master/fill_gaps.py` -- user prompt aligned with v2 format
- `pipeline/prompts.py` -- build_user_prompt() now includes testable_fact

---

## 23. Pre-Generation Checklist

| Status | Item |
|--------|------|
| [ ] | Archive `all_anchors.csv` to Desktop as dated backup |
| [ ] | Verify `generate_concept_vocab.py` has been run for all anchors (~$30, prevents concept_id fragmentation) |
| [ ] | Resolve 16 verbatim mismatches between `all_anchors.csv` and `anchor_points.csv` (15 of 16 have zero questions) |
| [ ] | Update `DIFFICULTY_DISTRIBUTION` in `__init__.py` to restore Expert ratios |
| [ ] | Add Expert to `quiz-enrichment-settings.html` UI (currently disabled) |
| [ ] | Decide: two-pass mode vs full-prompt mode |
| [ ] | Fix ID collisions (27 subsections, 540 questions at risk) |
| [ ] | Fix encoding artifacts (8 subsections get zero book content) |
| [ ] | Add file logging (15-hour run printing to stdout is fragile) |
| [ ] | Add permanent failure tracking (3 fails -> skip) |
| [ ] | Rebuild bundle (`build_quiz_bundle.py` -- 1,971 gap persists) |

### Post-Generation Audit Checks

| Check | What It Measures |
|-------|-----------------|
| Near-duplicate detection | Compare stems across 5 variants (0.7 cosine threshold) |
| concept_id consistency | Cluster within subsection, flag divergence |
| Distractor mix compliance | Count exact L1/L2/L3/L4 hit rate |
| Explanation quality | Flag < 50 chars or containing anchor text fragments |
| Correct answer position | Verify A/B/C/D well-distributed |
| Flashcard seed diversity | Flag subsections where all 20 seeds are near-identical |

---

## 24. Known Issues & Technical Debt

### Blocking Issues

1. **Question ID collisions:** 27 subsections where duplicate h3 names produce identical slugs (e.g., "DSM-5 Diagnostic Criteria" under 6 different disorders). Fix: include h2 slug prefix. 540 questions at risk.
2. **Encoding artifacts:** 8 subsections get zero book content due to mojibake (r-squared -> rA-squared, eta-squared -> I-hat-A-squared, em dash -> a-euro-oe). Fix: BeautifulSoup or Unicode normalization.
3. **Frontend connection:** `quiz-enrichment.html` reads from `enrichment_questions` table (flat columns). Exam questions use separate `questions` table.

### Data Issues

4. **Upload gap:** 27,213 generated locally but only 15,019 in Supabase/master CSV
5. **Deployed bundles stale:** quiz_stats.json shows 13,048 vs master 15,019
6. **CPAT discrepancy:** Deployed has 261 MORE than master. Root cause unknown.
7. **Missing subsection info:** 2,582 questions missing both subsection_title AND section_id. Worst: WDEV (779), CASS (476), PMET (367), PTHE (291)
8. **anchor_point_ids stripped on upload:** upload script maps section/subsection titles but does NOT carry over anchor_point_ids

### Feature Gaps

9. **Expert never generated:** Full infrastructure exists but zero questions produced
10. **SM-2 review UI missing:** Cards load on remediation.html but users can't rate them
11. **Confidence rating not in UI:** Schema ready, radio group trivial, not implemented
12. **Adaptive question selection:** Still `ORDER BY RANDOM()`
13. **Flashcard seeds never surfaced:** 78,749 seeds exist but no schedule/UI
14. **Misconception -> study strategy mapping:** Designed, never surfaced to student
15. **Confused_with graph:** Stored in JSONB, never extracted
16. **quiz_review_queue (localStorage):** Written but never read
17. **Adaptive bridge:** Enrichment scores never read by flashcards.js

### Code Quality

18. **Bloom's verb enforcement gap:** validate_question never checks stem verbs
19. **Stem pattern not validated:** Request clinical_vignette but don't verify output actually contains clinical scenario
20. **confused_with not validated against correct answer:** inconsistent confusion matrix edges possible
21. **content_snippet_chars=0 not flagged:** Questions generated from anchors only with "(no content extracted)"
22. **No .gitignore for JustinQuestionsDatabase-2.0**

---

## 25. Rejected Alternatives (with reasons)

| What | Rejected By | Reason |
|------|-------------|--------|
| Pure-difficulty difficulty (no mixing) | Framework design | Can't diagnose WHY student fails; misses ZPD; causes failure streaks; no floor/ceiling avoidance; incomplete misconception coverage |
| 3 additional EPPP stem patterns (research_interpretation, ethical_dilemma, developmental_sequence) | User | "We are designing quizzes, not mock exams" -- would constrain LLM unnecessarily |
| 0% Expert in Easy mode | Analytics audit | Only 2 L4 distractors in 60 total; blind spot for overgeneralization/partial_understanding |
| 0% Easy in Expert mode | Analytics audit | Only 2 L1 distractors; advanced students can have foundational terminology confusions |
| Original Easy distractor mix (2xL1 + 1xL2) | Deep scrutiny | Diagnostically redundant; two cross-subdomain distractors reveal same information |
| Original Hard distractor mix (1xL2 + 2xL3, no L4) | Deep scrutiny | EPPP is notorious for "all four options seem right" -- Hard needs at least one L4 |
| Minimum information principle (1 fact per flashcard) | User decision | EPPP tests concept-implication pairs; keeping both mirrors exam format |
| Open mode concept_ids | Framework design | LLM invents IDs on the fly causing fragmentation; breaks all analytics silently |
| First-2000-chars truncation | Deep scrutiny | Biases questions toward early content; later content under-tested |
| Hard fail on Bloom's verb mismatch | Deep scrutiny | Soft warning preferred over hard fail |
| Generating new adaptive flashcards (API call) | Framework design | 78,749 seeds already exist; extraction (zero cost) beats generation |
| Post-generation normalization for concept_ids | Framework design | Extra pass, less deterministic; pre-seeding is cheaper and more reliable |
| Two-pass generation (sequential variants) | Framework design | Requires sequential generation, slower than pre-seeding |

---

## 26. Research Foundations

### Citations Explicitly in Generation Prompts

| Citation | Principle | Location |
|----------|-----------|---------|
| Roediger & Karpicke (2006) | Retrieval Practice | prompts.py:44,148 |
| Metcalfe (2017) | Error-Driven Learning | prompts.py:45,149 |
| Bjork (1994) | Desirable Difficulties | prompts.py:46,150 |
| Morris et al. (1977) | Transfer-Appropriate Processing | prompts.py:47,151 |

### Citations Referenced in Framework Design (Not in Prompts)

| Citation | Principle / Use |
|----------|----------------|
| Vygotsky (1978) | Zone of Proximal Development -- justifies difficulty mixing |
| Wilson et al. (2019) | 85% accuracy sweet spot -- learning optimized at ~85% correct |
| Bandura (1977) | Self-efficacy maintenance -- justifies asymmetric distribution |
| Seligman | Learned helplessness -- sustained failure leads to disengagement |
| Cepeda et al. (2006) | Spaced repetition -- 2-3x better long-term retention |
| Karpicke & Roediger (2008) | Spaced retrieval evidence |
| Rohrer (2007) | Interleaving -- referenced for stem rotation |
| Mislevy | Evidence-centered design -- need ~5+ encounters per type for reliable misconception profiling |

---

## 27. File Locations & Paths

### Core Pipeline

| File | Path |
|------|------|
| Main generation script | `C:\Users\mcdan\JustinQuestionsDatabase-2.0\scripts\generate_quiz_questions.py` |
| System prompts | `C:\Users\mcdan\JustinQuestionsDatabase-2.0\scripts\pipeline\prompts.py` |
| Config/constants | `C:\Users\mcdan\JustinQuestionsDatabase-2.0\scripts\pipeline\__init__.py` |
| Validation gates | `C:\Users\mcdan\JustinQuestionsDatabase-2.0\scripts\pipeline\gates.py` |
| Agents | `C:\Users\mcdan\JustinQuestionsDatabase-2.0\scripts\pipeline\agents.py` |
| Concept vocab generator | `C:\Users\mcdan\JustinQuestionsDatabase-2.0\scripts\generate_concept_vocab.py` |
| Build bundle | `C:\Users\mcdan\JustinQuestionsDatabase-2.0\scripts\build_quiz_bundle.py` |

### Data Files

| File | Path |
|------|------|
| Master CSV (12,703 questions, 57 cols) | `C:\Users\mcdan\OneDrive\Master CSVs\enrichment_all_questions.csv` |
| Master CSV backup (pre-anchor-update) | `...enrichment_all_questions.csv.backup_pre_anchor_update` |
| Legacy CSV (15,019 questions, 45 cols) | `C:\Users\mcdan\OneDrive\eppp-claude-context\master\enrichment_questions.csv` |
| Canonical anchor CSV (10 columns) | `goliath/csvs/anchor_points.csv` (in-repo since 2026-04-26) |
| Stale anchor (DO NOT USE) | `C:\Users\mcdan\OneDrive\eppp-claude-context\anchors\all_anchors.csv` |
| Deployed stats | `C:\Users\mcdan\PassEPPP-website\content\enrichment\quiz_stats.json` |
| Deployed bundles | `C:\Users\mcdan\PassEPPP-website\content\enrichment\{DOMAIN}_quiz.json` |
| Source JSONs | `C:\Users\mcdan\JustinQuestionsDatabase-2.0\data\quiz\{DOMAIN}\*.json` |

### Frontend

| File | Path |
|------|------|
| Settings page | `C:\Users\mcdan\PassEPPP-website\pages\quiz-enrichment-settings.html` |
| Exercise page | `C:\Users\mcdan\PassEPPP-website\pages\quiz-enrichment.html` |
| NextJS project | `C:\Users\mcdan\PassEPPP-NextJS\` |

### Flashcard Files

| File | Path | Size |
|------|------|------|
| Classic manifest | `JustinQuestionsDatabase-2.0/data/flashcards/flashcards_manifest.json` | 2.5 MB |
| Frontend bundle | `JustinQuestionsDatabase-2.0/data/flashcards/flashcards-data.js` | 1.2 MB |
| Adaptive cards | `JustinQuestionsDatabase-2.0/data/flashcards/adaptive_flashcards.json` | 259 MB |
| Adaptive by domain | `JustinQuestionsDatabase-2.0/data/flashcards/adaptive_flashcards_by_domain.json` | 269 MB |
| Live site data | `PassEPPP-website/js/flashcards-data.js` | 2.2 MB |

### Anchor Pipeline

| File | Path | Status |
|------|------|--------|
| Parsed anchors (source of truth) | `mastery-page/scripts/data/anchors_parsed.json` | 1,567 |
| Chapter map | `mastery-page/scripts/data/anchor_chapter_map.json` | 1,567 |
| Classified anchors | `mastery-page/scripts/data/anchors_classified.json` | Phase 1 output |
| Coverage audit | `mastery-page/scripts/data/coverage_audit.json` | Phase 2 output |
| Generated content | `mastery-page/scripts/data/anchor_content_generated.json` | Phase 3 output |

### SQL Migrations

| File | Purpose |
|------|---------|
| `PassEPPP-Database/migrations/010_consolidated_v2_systems.sql` | 5 tables + 24 RPCs (NOT applied) |
| `sql/20` through `sql/22` | Already applied |
| `sql/26-flashcard-sections-v2.sql` | 2,936 card mappings (365 KB) |

### Test Batch Files

| File | Path |
|------|------|
| V1 script | `C:\Users\mcdan\OneDrive\eppp-claude-context\master\test_batch_generate.py` |
| V1 output (100q) | `...master\test_batch_opus_2026-04-24.csv` |
| V2 script | `...master\test_batch_v2.py` |
| V2 fill script | `...master\fill_gaps.py` |
| V2 output (20q) | `...master\test_batch_v2_opus_2026-04-24.csv` |

### V3 Chapter Schema

| File | Path |
|------|------|
| Excel output | `C:\Users\Admin\PassEPPP-Textbook-2.0\chapter_schema_v3.xlsx` |
| Source CSV | `C:\Users\Admin\Downloads\anchor_passages_v3_pure_textbook_1081 - Anchor Passages V3.csv` |
| Build script | `C:\Users\Admin\PassEPPP-Textbook-2.0\build_chapter_schema.py` |

### Content Directories

| Purpose | Path |
|---------|------|
| Mastery page content (96 chapters, source of truth) | `C:\Users\mcdan\mastery-page\content\domain1-9\*.html` |
| PassEPPP website content (129 chapters) | `C:\Users\mcdan\PassEPPP-website\pages\mastery\content\` |
| Chat history | `C:\Users\mcdan\OneDrive\enrichment chat history original\` |
