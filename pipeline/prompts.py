"""
Prompt builders for the quiz question pipeline.

Two modes:
  - FOCUSED (vocab-backed): Smaller prompt, LLM generates only creative content.
    Misconception metadata is pre-assigned by DistractorPlannerAgent.
  - OPEN (no vocab): Larger prompt, LLM generates metadata fields too.
    Used when concept_vocab hasn't been generated yet.

Both modes strip letter/position/is_correct from LLM output
(the Assembler handles those deterministically).
"""

from . import (
    BLOOMS_BY_TIER, BLOOMS_PRIMARY, DIFFICULTY_LABELS, DISTRACTOR_MIX,
    CORRECT_POSITIONS, EPONYM_WHITELIST,
    get_blooms_verbs, get_stem_pattern,
)


def _mcq_quality_rules():
    """Returns the shared MCQ quality + testwise-defense block.

    Three testwise heuristics are addressed:
      1. Researcher attribution leakage (with whitelist for true eponyms)
      2. Answer-length balance (correct cannot be longest by wide margin)
      3. Parallel construction (elaboration markers cannot cluster on correct)
    """
    eponym_list = ", ".join(sorted(EPONYM_WHITELIST))
    return f"""
## Originality & MCQ Quality
- NEVER use verbatim text from anchor summaries in stems, options, or explanations.
- Create NOVEL scenarios with unique names, settings, demographics, presenting concerns.
- Options must be grammatically consistent with the stem.
- No absolute language unless testing genuine absolutes.

### Researcher Attribution (DO NOT name researchers)
NEVER attribute findings to named researchers in stems, options, or explanations.
This includes: "Squire (2004)", "According to Smith", "Smith and Jones found",
"Smith's research/framework/study/theory", "(Author, Year)", "Author et al."
The student should engage with the concept itself, not the citation history.

EXCEPTION — eponymous concepts where the name IS the standard label may be used
(e.g., "Piaget's sensorimotor stage", "Cannon-Bard theory", "Pavlovian conditioning").
Whitelist: {eponym_list}

If a researcher is not on this whitelist, write the concept descriptively without
naming any person, year, paper, or framework. Strip citations even when they
appear in the source material.

### Answer Length Balance (DO NOT pad the correct answer)
All four options must be similar in length — within ~1.5x character count of each
other. Do NOT add qualifiers, parentheticals, or extra detail to the correct
answer to make it "more complete" or "more accurate." Compress the correct
answer to roughly the same length as your distractors.

Failure pattern to defeat: students who do not know the answer learn that the
longest option is most often correct. Defeat this by writing all four options
to comparable length. If your correct answer is more than 20% longer than the
longest distractor, rewrite it shorter or expand the distractors.

### Parallel Construction (DO NOT cluster elaboration on the correct answer)
Match the structural complexity of the correct answer to the distractors. If
the correct answer contains a parenthetical clarification, a semicolon, an
em-dash, or a compound clause ("X (which means Y)"; "X; therefore Y";
"X — i.e., Y"; "X, but Y"), then at least ONE distractor must mirror that
structure. Elaboration markers must not cluster only on the correct answer.

Failure pattern to defeat: students learn to pick the option with the careful
qualification. Apply qualifiers, parentheticals, and compound structure
evenly across all four options or to none.

### Stem & Distractor Format Hygiene (DO NOT use "test-the-test" framing)
The downstream editorial audit rejects two systemic patterns. Both are
forbidden at generation time. NO exceptions, regardless of tier or flavor.

STEM rules — pose ONE direct question. The stem should be answerable in
the same way regardless of the multiple-choice format. The student is
reasoning about the CONCEPT, not selecting from "options".

  FORBIDDEN stem framings (test-the-test):
  - "Which option correctly identifies..."
  - "Which definition correctly identifies..."
  - "Which of the following best characterizes..."
  - "Which most accurately describes..."
  - "Which best identifies..."
  - "Which option correctly distinguishes..."
  - Any stem containing the words "correctly", "best", "most",
    "accurately", or "option" used to qualify the answer choice itself.

  PREFERRED direct framings:
  - "Which of the following defines X?"
  - "What distinguishes A from B?"
  - "X most directly results from which mechanism?"
  - "[stimulus]. Which conclusion follows?"
  - "How does Y relate to Z?"

  Reason: meta-evaluative modifiers ("correctly"/"best"/"most") cue the
  student to find the right answer rather than reasoning about the
  concept. The stem becomes a reading-comprehension exercise on the
  question text, not a probe of domain knowledge.

DISTRACTOR rules — options are ANSWERS, not INSTRUCTIONS. Distractors
must read as candidate answers to the stem's question.

  FORBIDDEN distractor lead forms (imperative-verb commands):
  - "Identify the X..."
  - "Classify Y..."
  - "Recognize Z..."
  - "Distinguish A from B..."
  - "Integrate X with Y..."
  - "Evaluate option (b) as superior"
  - Any distractor that begins with an imperative verb directing the
    test-taker to perform an action rather than stating the answer.

  PREFERRED distractor forms:
  - Noun phrases that ARE the answer ("Retrograde amnesia of episodic
    memory beyond five years")
  - Short declarative claims ("The compound has intrinsic activity at
    D2 receptors")
  - Clinical predictions framed as outcomes ("Hemiplegia ipsilateral
    to the lesion")

  All four options must share the same grammatical form (all noun
  phrases, OR all declarative claims, OR all clinical predictions —
  but never mixed). All four lead words must match in part-of-speech.
  A four-option block where three options begin with the same
  imperative verb and one breaks the pattern is also a structural
  defect; rewrite all four together.

### Stem Over-Specification Hygiene (DO NOT print facts your distractors will contradict)

When designing distractors, the stem MUST NOT print specific facts
(numbers, ratios, directions/laterality, named outcomes, stage-specific
timing, named durations, named symptom counts) that any distractor
lexically contradicts. If you find yourself writing such a fact into the
stem, you have a choice: redesign the distractor to be wrong via concept
knowledge (content_gap), or remove the specific fact from the stem.

The downstream audit will classify any distractor whose claim a student
can reject by re-reading the stem alone (no concept knowledge invoked)
as ENGLISH_GAP. That fails the question. Avoid this at the source by
keeping the stem ABOVE the level of the distractors' specific
contradictions.

  FORBIDDEN: stem says "rates approach a 2:1 female-to-male ratio by
  adulthood" + distractor says "stable 3:1 ratio from childhood onward"
  — student rejects on the number alone, no domain knowledge needed.

  PREFERRED: stem says "the well-established sex-difference pattern
  emerging in adolescence" + distractor names a different epidemiological
  pattern. Student must know the pattern to reject; concept_gap.

  FORBIDDEN: stem says "infarct in the RIGHT cerebral hemisphere" +
  distractor says "predict left-side hemiplegia from disruption of
  IPSILATERAL pyramidal pathways." Student rejects on "ipsilateral"
  contradicting the stem's named laterality.

  PREFERRED: stem says "infarct in the right cerebral hemisphere" +
  distractor says "predict bilateral hemiplegia from cortical
  disinhibition of brainstem motor pathways." Wrong by mechanism, not
  by laterality contradiction.

  FORBIDDEN: stem says "no measurable change in postsynaptic firing" +
  distractor says "compound exerts its own postsynaptic effect" — the
  stem's specific finding is the lexical contradiction.

  PREFERRED: stem says "compound binds the receptor at high affinity" +
  distractor says "compound has intrinsic activity that mimics the
  endogenous neurotransmitter." Wrong by pharmacology concept (intrinsic
  activity implies measurable effect); rejection requires knowing what
  intrinsic activity means.

DESIGN RULE: the stem provides the CONTEXT and the QUESTION. The
distractors are the WRONG CONCEPTS the student might pick. Keep the stem
ABOVE the level of the distractors' specifics — describe the situation
without revealing answer-distinguishing details that distractors will
then contradict.

DECISION TEST before committing the stem text: for each distractor you
plan to write, ask "does this distractor's claim lexically contradict
any specific fact (number, direction, named outcome) in my draft stem?"
If YES, you have an english_gap on your hands — redesign the distractor
OR strip the specific fact from the stem. NEVER ship the question with
the lexical contradiction intact.
"""


def _blooms_stem_enforcement(difficulty_tier, blooms_primary, blooms_secondary):
    """Bloom's × stem pattern enforcement for all tiers.

    Tier 1-2: prevent upward creep (questions too complex for the tier).
    Tier 3-4: prevent downward creep (questions too simple for the tier).
    """
    primary = BLOOMS_PRIMARY[difficulty_tier]

    if difficulty_tier == 1:
        return f"""
## Bloom's Enforcement — Tier 1 (Primary: REMEMBER)
This is a RECALL-level question. The student retrieves stored knowledge.

RULES:
- NO scenarios, cases, or vignettes. Test the concept directly.
- NO "apply to this situation" framing. If the student must reason about a scenario, the question is too complex.
- The correct answer must be determinable from a single recalled fact, definition, or feature.
- All 4 options should be factual statements, terms, or features — not analyses or interpretations.
- Distractors should be real concepts/facts that a student might confuse with the correct answer.
- Paraphrase source material — NEVER use verbatim textbook language in options.

ANTI-PATTERNS (reject if present):
- A stem longer than 2 sentences (indicates scenario creep)
- Options that require evaluating a scenario
- "In this case..." or "Given that..." framing (implies application)
- Options that are analyses or judgments rather than facts
"""

    if difficulty_tier == 2:
        return f"""
## Bloom's Enforcement — Tier 2 (Primary: UNDERSTAND)
This is a COMPREHENSION-level question. The student demonstrates understanding of concepts.

RULES:
- Brief scenarios are permitted (1-2 sentences max) but the cognitive demand is comprehension, not multi-step reasoning.
- The student should need ONE conceptual step to answer: recognize an example, identify a distinction, classify an item, or restate a concept.
- Do NOT require integrating multiple concepts, analyzing competing explanations, or evaluating nuanced claims.
- Do NOT require predicting outcomes of complex scenarios — that is Tier 3.
- Distractors should test adjacent confusions: similar concepts, reversed relationships, misclassifications.

ANTI-PATTERNS (reject if present):
- Multi-step reasoning required (apply principle THEN analyze outcome)
- "Which is the BEST/MOST appropriate" framing (implies evaluation)
- Scenarios longer than 2 sentences (indicates application-level complexity)
- Options that require weighing evidence or judging relative merit
- Integration of multiple concept areas
"""

    if difficulty_tier == 3:
        return f"""
## Bloom's Enforcement — Tier 3 (Primary: APPLY)
Every question MUST require genuine {blooms_primary}/{blooms_secondary}-level thinking regardless of stem pattern. A question answerable by recalling a single definition — regardless of scenario dressing — is NOT acceptable at Tier 3.

AVOID these anti-patterns:
- **clinical_vignette**: Do NOT ask the student to classify or label a case using a single definition (e.g., "this is called a false positive"). Instead, require multi-step reasoning or evaluation of competing interpretations.
- **scenario_completion**: Do NOT ask "which of these is the [term]?" Instead, require predicting outcomes, selecting actions, or evaluating trade-offs based on scenario-specific details.
- **error_identification**: NEVER present 4 abstract definitional statements and ask which is factually wrong — this is always recall-level. Instead: (1) anchor ALL claims within a specific scenario, case, or applied context, (2) the erroneous claim must involve a misapplication or flawed reasoning step with consequences within that scenario, and (3) at least one other claim must be plausible enough that the student must evaluate the interaction between two concepts to rule it out.
- **case_analysis**: Do NOT accept a single-concept label as the answer. The answer must explain WHY something is happening (mechanism), not just WHAT it is (diagnosis). The case must contain details supporting multiple plausible mechanisms.
- **mechanism_application**: Do NOT use textbook examples of the named principle. The novel situation must require genuine forward-reasoning, not pattern recognition.

If the topic is narrow, test APPLICATION of the concept in a novel context rather than testing whether the student can recall its definition.

IMPORTANT — option format at T3:
"Predict outcomes" / "select actions" / "evaluate trade-offs" describe what
the STEM asks for. Distractors must NOT begin with imperative verbs like
"Predict X", "Select Y", or "Evaluate Z" — those read as task instructions,
not answer choices. Format the prediction or action AS the answer:
  WRONG: "Predict hemiplegia ipsilateral to the lesion"
  RIGHT: "Hemiplegia ipsilateral to the lesion"
  WRONG: "Select foot-in-the-door escalation"
  RIGHT: "Foot-in-the-door escalation from minor to larger requests"
All four options must share the same form (all noun phrases, OR all
declarative claims) — never imperative-led commands.

## Named-Effect & Labeled-Phenomenon Concepts
When the target concept is a named effect, bias, phenomenon, or syndrome, simple pattern-matching from scenario to label is NOT sufficient at Tier 3. Use at least ONE of these strategies:
1. **Test consequences/predictions**: Ask what happens NEXT, not what the effect IS.
2. **Present atypical or ambiguous cases**: Multiple effects could plausibly apply; scenario details discriminate.
3. **Require mechanism over label**: Ask WHY, not WHAT it is called.
4. **Make identification necessary-but-insufficient**: Recognize the concept AND do something with it.
5. **Force discrimination between overlapping concepts**: Pair with most confusable neighbor as distractor.

## Single-Fact Concept Guard
When the core concept is a single quantitative fact (duration, capacity, threshold), the question MUST require USING that fact to predict an outcome or evaluate a decision — NOT merely identify or recall the value.
"""

    # difficulty_tier == 4
    return f"""
## Bloom's Enforcement — Tier 4 (Primary: EVALUATE)
Every question MUST require genuine evaluation-level thinking. The student must judge, critique, weigh evidence, or synthesize across concepts. A question answerable by applying a single concept to a single scenario — regardless of complexity — is NOT acceptable at Tier 4.

AVOID these anti-patterns:
- **contrast_prompt**: Do NOT ask "what distinguishes X from Y?" as a standalone definition question. Present a specific case where BOTH concepts seem to apply, and the student must analyze scenario details to determine which fits better.
- **best_answer**: All options MUST contain genuine truth. If three options are clearly wrong and one is clearly right, this is a standard MCQ, not a best-answer question. The specific context must drive which answer is "best."
- **subtle_error**: The reasoning presented must be at least 80% correct. The error must require deep understanding to detect — not a simple factual mistake. One option MUST be "The reasoning is sound."
- **competing_evidence**: BOTH positions must be genuinely defensible. No straw men. The context must be what tips the balance — remove the context and the answer should be indeterminate.
- **integration**: BOTH concept areas must be necessary to answer. If either concept alone is sufficient, add complexity or change the question. At least two distractors must represent partial integration (using only one concept).

IMPORTANT — option format at T4:
T4 questions weigh competing positions, but options are STILL ANSWERS,
not meta-instructions to evaluate them. Distractors must NOT begin with
"Evaluate option (a) as superior" / "Evaluate option (b) as superior"
or any other meta-evaluation wrapper. State each position AS a
declarative claim in the same grammatical form across all four:
  WRONG: "Evaluate option (a) as superior; stacking pro-attitudinal
          arguments raises confidence."
  RIGHT: "Stacking pro-attitudinal arguments raises confidence and
          blocks future doubt."
The competing-position framing belongs in the STEM (the scenario
presents a debate); each option is a candidate answer in declarative
form. All four options must share the same lead form — never use
"Evaluate X as superior" prefixes.

## Two-Concept Integration Requirement
At Tier 4, every question MUST integrate at least TWO distinct concepts so that no single definition is sufficient to answer correctly. The student should need to combine, compare, or chain multiple pieces of knowledge. Examples:
- Test a named bias AND how it interacts with a situational variable to produce a specific outcome
- Test a diagnostic criterion AND how it applies differently across two competing presentations
- Test echoic memory duration AND its implication for a specific assessment technique

If the question can be answered by recalling a single isolated fact or definition, add a second concept that the student must integrate to reach the correct answer.

## Named-Effect & Labeled-Phenomenon Concepts
When the target concept is a named effect, bias, phenomenon, or syndrome, simple pattern-matching from scenario to label is NOT sufficient at Tier 4. Use at least ONE of these strategies:
1. **Test consequences/predictions**: Ask what happens NEXT, not what the effect IS.
2. **Present atypical or ambiguous cases**: Multiple effects could plausibly apply; scenario details discriminate.
3. **Require mechanism over label**: Ask WHY, not WHAT it is called.
4. **Make identification necessary-but-insufficient**: Recognize the concept AND do something with it.
5. **Force discrimination between overlapping concepts**: Pair with most confusable neighbor as distractor.
"""


# ══════════════════════════════════════════════════════════════
# System Prompts
# ══════════════════════════════════════════════════════════════

def build_system_prompt(difficulty_tier, mode="open",
                          prompt_version="v2", flavor=None):
    """Build system prompt.

    Args:
        difficulty_tier: 1-4 Bloom's tier.
        mode: 'focused' (vocab-backed) or 'open' (free-form).
        prompt_version: 'v2' (default since Phase 18) or 'v1' (legacy,
            retained for cohort comparisons). v2 adds the Distractor
            Quality Framework section that mirrors the audit's 3-class
            scheme + flavor-aware
            preferred-wrongness-mode block).
        flavor: pedagogical flavor for the current anchor (e.g.,
            "mechanism", "framework"). Only used when prompt_version
            == "v2"; resolved by `pipeline.anchor_flavor.flavor_for_anchor`
            at task-build time. Ignored in v1.

    P6 v2 design notes:
      - The new section is inserted AFTER the existing "Distractor
        Design" block in both focused and open variants.
      - Mirrors `scripts/audit_stem_contradictions.py:PROMPT`'s
        3-class scheme (english_gap / content_gap / clean) so
        generation and audit share the same vocabulary.
      - Includes explicit over-correction warning + paired
        english_gap → content_gap rewrite examples per pattern.
      - Per-flavor preferred-wrongness-mode block (one block, not
        all 13 — keeps the prompt focused).
    """
    if mode == "focused":
        return _build_system_prompt_focused(difficulty_tier,
                                              prompt_version=prompt_version,
                                              flavor=flavor)
    return _build_system_prompt_open(difficulty_tier,
                                       prompt_version=prompt_version,
                                       flavor=flavor)


# ── P6 v2: Distractor Quality Framework section ──────────────────────
#
# Inserted into both system-prompt variants when prompt_version="v2".
# Mirrors the audit prompt's three-class scheme, includes paired
# english_gap → content_gap rewrite examples drawn from real failures
# observed in stress tests + Phase 16-pre sampling, includes an
# explicit over-correction warning, and dispatches the
# DOMAIN-PREFERRED WRONGNESS MODES block on `flavor`.

_FLAVOR_WRONGNESS_BLOCKS: dict[str, str] = {
    "mechanism": (
        "FLAVOR=mechanism (biopsychology / pharmacology mechanism content): "
        "prefer mechanism-inversion as the wrongness mode. Wrong "
        "neurotransmitter, wrong pathway, wrong region role, wrong "
        "direction-of-effect via mechanism. NOT lexical inversion of "
        "stated facts."
    ),
    "framework": (
        "FLAVOR=framework (ethics / legal framework application): "
        "prefer framework-misapplication as the wrongness mode. Wrong "
        "APA standard, wrong legal authority, wrong principle weighting "
        "for the situation. NOT contradicting case-fact details printed "
        "in the stem."
    ),
    "cognitive_process": (
        "FLAVOR=cognitive_process (learning / memory / attention): "
        "prefer process-stage confusion. Wrong encoding/storage/retrieval "
        "stage, wrong attention mechanism, wrong memory-system "
        "attribution. NOT contradicting outcome facts in the stem."
    ),
    "clinical_disease": (
        "FLAVOR=clinical_disease (neurocognitive disorders / brain "
        "pathology): prefer differential-diagnosis errors. Wrong disorder "
        "with overlapping features, wrong etiology category. NOT "
        "contradicting imaging findings or stated symptoms."
    ),
    "applied_cultural": (
        "FLAVOR=applied_cultural (racial / cultural identity / "
        "multicultural counseling): prefer stage-misattribution by "
        "ABILITY/BEHAVIOR (not by state-description). Prefer mis-naming "
        "the framework being applied. NOT contradicting described "
        "awareness/conflict states."
    ),
    "social_process": (
        "FLAVOR=social_process (group dynamics / social psychology): "
        "prefer process-misattribution. Wrong attribution-bias type, "
        "wrong group-process mechanism. NOT contradicting described "
        "behavioral outcomes."
    ),
    "test_psychometric": (
        "FLAVOR=test_psychometric (clinical assessment instruments): "
        "prefer scale-misattribution or pattern-misinterpretation. NOT "
        "contradicting scale numbers or score values printed in the stem."
    ),
    "statistical": (
        "FLAVOR=statistical (research methods / psychometric statistics): "
        "prefer test-conflation. Wrong statistical test for the design, "
        "wrong assumption violation, wrong effect-size interpretation. "
        "NOT contradicting numeric thresholds in the stem."
    ),
    "developmental_stage": (
        "FLAVOR=developmental_stage (lifespan development / stage theory): "
        "prefer stage-misattribution by AGE/ABILITY. NOT contradicting "
        "described age/behavior in the stem."
    ),
    "diagnostic_criterion": (
        "FLAVOR=diagnostic_criterion (DSM / clinical diagnosis): prefer "
        "differential-diagnosis or criterion-substitution errors. NOT "
        "contradicting symptom counts or duration thresholds printed "
        "in the stem."
    ),
    "therapeutic_modality": (
        "FLAVOR=therapeutic_modality (psychotherapy / intervention): "
        "prefer modality-misapplication or technique-substitution errors. "
        "NOT contradicting client-presentation facts."
    ),
    "selection_psychometric": (
        "FLAVOR=selection_psychometric (I/O / personnel selection): "
        "prefer wrong-test-for-purpose or validity-coefficient "
        "misinterpretation. NOT contradicting numeric coefficients."
    ),
    "generic": (
        "FLAVOR=generic: prefer concept-confusion. Wrong adjacent "
        "concept, wrong scope of application, wrong direction via "
        "mechanism. NOT lexical contradiction of stated facts."
    ),
}


def _distractor_quality_framework_section_v2(flavor: str | None = None) -> str:
    """Build the v2 'Distractor Quality Framework' section.

    Mirrors `scripts/audit_stem_contradictions.py:PROMPT`'s 3-class
    scheme. Inserted after the existing 'Distractor Design' block in
    both focused and open variants.
    """
    flavor_block = _FLAVOR_WRONGNESS_BLOCKS.get(
        flavor or "generic", _FLAVOR_WRONGNESS_BLOCKS["generic"]
    )

    return f"""## Distractor Quality Framework (CRITICAL)

Each distractor you write will be classified by a downstream auditor into one of three categories. Use the same classification scheme to DESIGN your distractors deliberately.

THE THREE CLASSES:

ENGLISH_GAP (FORBIDDEN — quality failure):
A student can reject the distractor by lexical comparison with the stem alone. The contradiction is in printed words, not concepts.
  Canonical example:
    Stem: "After bilateral hippocampal damage, Lester still recalls his wedding from a decade earlier."
    Distractor: "Retrograde amnesia erases ALL pre-injury memories."
    Why FORBIDDEN: "all" is contradicted by the wedding case in the stem. Student rejects without knowing what retrograde amnesia is.

CONTENT_GAP (PREFERRED — the workhorse):
A real contradiction exists, but recognizing it requires invoking concept knowledge. The distractor LOOKS plausible until you know what a technical term actually means.
  Canonical example:
    Stem: "compound binds receptor but produces no measurable postsynaptic activity on its own."
    Distractor: "compound has intrinsic activity that mimics the endogenous neurotransmitter."
    Why PREFERRED: rejecting requires knowing intrinsic activity ⇒ measurable postsynaptic effect. Tests concept knowledge directly.

CLEAN (PREFERRED — secondary):
A wrong-but-plausible alternative not directly contradicting any stem fact. Rejection requires applying concept knowledge.
  Canonical example:
    Stem: "CT confirms thromboembolic infarct in motor cortex."
    Distractor: "Predict hemiplegia from closed head trauma."
    Why PREFERRED: stem doesn't say "not closed head trauma"; rejecting requires knowing thromboembolic ≠ trauma etiology.

THE DESIGN DIRECTIVE:
Produce content_gap (preferred) or clean (acceptable). Never produce english_gap. Discriminating test:

  "Could a student who hasn't studied the concept reject this distractor by re-reading the stem alone?"
  - YES → english_gap. Redesign.
  - NO  → content_gap or clean. Proceed.

### CRITICAL: The over-correction failure mode

A naive reading of "avoid english_gap" leads to writing tangential distractors that ignore the stem entirely. That is ALSO bad — it collapses content_gap (the workhorse) into clean-only, losing the pedagogical bite of distractors that engage stem facts.

  EXAMPLE OF OVER-CORRECTION (also bad):
    Stem: "compound produces no measurable postsynaptic activity"
    Tangential distractor: "compound was first synthesized in 1985"
    Why bad: doesn't engage the stem's pharmacology question; doesn't probe any misconception the question is supposed to test.

  CORRECT MOVE: write distractors that ENGAGE the stem (touch its facts, share its vocabulary domain) but where rejection requires concept knowledge. The right distractor for the stem above:
    "compound has intrinsic activity that mimics neurotransmitter."
  This DOES contradict the stem ("no measurable activity" vs "intrinsic activity"), but the contradiction requires knowing what intrinsic activity means in receptor pharmacology — content_gap.

The distinction is WHERE the contradiction lives:
  - In printed words alone → english_gap (forbidden)
  - In concept knowledge → content_gap (preferred)

DO NOT avoid contradiction. AVOID lexically-visible contradiction.

### Pattern-specific paired examples (english_gap → content_gap rewrites)

These show how to convert tempting english_gap distractors into content_gap ones on the same content area. Use as design references.

[1] DIRECTIONAL/laterality terms:
  Stem: "Edgar arrived with sudden inability to move his LEFT arm and LEFT leg. CT confirms thromboembolic infarct in RIGHT cerebral hemisphere motor cortex."
  ❌ FORBIDDEN: "Predict bilateral hemiplegia from disruption of contralateral pyramidal pathways."
     (Lexical: "bilateral" contradicts stated unilateral.)
  ✅ PREFERRED: "Predict ipsilateral hemiplegia, with pyramidal fibers remaining ipsilateral to the damaged hemisphere."
     (Wrong by mechanism — pyramidal decussation makes ipsilateral wrong — but recognizing requires neuroanatomical knowledge.)

[2] STAGE-mental-state descriptions:
  Stem: "Lena seems puzzled when Dr. Jacobs raises the concept of White privilege."
  ❌ FORBIDDEN: "Lena is in immersion stage actively examining privilege."
     (Lexical: "actively examining" contradicts "puzzled.")
  ✅ PREFERRED: "Lena is in autonomy stage with internalized non-racist identity."
     (Wrong stage — but recognizing requires Helms-framework knowledge that autonomy = late-stage/developed, incompatible with described puzzlement.)

[3] LEXICAL-etymology:
  Stem: "Sundowning is a chronobiological phenomenon characterized by emergence of agitation in late afternoon and evening."
  ❌ FORBIDDEN: "Sundowning involves morning worsening of cognition in dementia patients."
     (The word "sundowning" lexically implies evening; "morning" contradicts before student even knows what sundowning means.)
  ✅ PREFERRED: "Sundowning is mediated by glymphatic clearance deficits during NREM sleep cycles."
     (Wrong mechanism — sundowning is circadian, not glymphatic — but rejecting requires neurobiology knowledge, not word reading.)

[4] STATED-finding contradiction:
  Stem: "MMPI-2 profile shows F within normal limits and FB markedly elevated."
  ❌ FORBIDDEN: "Pattern reflects parallel F and FB elevations suggesting random responding throughout."
     (Lexical: stem prints "F normal"; "parallel elevations" directly contradicts.)
  ✅ PREFERRED: "Pattern reflects content-based response shift in the latter half of the test, indicating genuine fatigue."
     (Engages the F/FB distinction at right level; rejecting requires knowing what FB measures vs the actual elevation pattern's clinical meaning.)

[5] SHARED-feature contradiction:
  Stem: "Prader-Willi and Angelman syndromes both involve deletions in the 15q11-q13 region of chromosome 15."
  ❌ FORBIDDEN: "Prader-Willi reflects a deletion at 7q11.23 whereas Angelman reflects 15q11-q13."
     (Lexical: stem says BOTH have 15q11-q13.)
  ✅ PREFERRED: "Prader-Willi reflects loss of MATERNAL imprinted expression while Angelman reflects loss of PATERNAL imprinted expression."
     (Preserves the shared deletion locus; the imprinting direction is reversed — recognizing requires knowing the parental-origin specificity for each syndrome.)

[6] SETTING/timeline contradiction:
  Stem: "Drake presents to the inpatient psychiatric unit with agitation, diaphoresis, fever 39.8°C, and clonus, three hours after a phenelzine dose."
  ❌ FORBIDDEN: "Recommend outpatient monitoring with daily check-ins pending symptom resolution."
     (Lexical: stem says inpatient; "outpatient" contradicts.)
  ✅ PREFERRED: "Recommend cyproheptadine and supportive care, with restart of phenelzine after 14 days."
     (Setting consistent — inpatient cyproheptadine is plausible — but rejecting requires knowing serotonin syndrome management contraindicates restarting the precipitating MAOI.)

### Domain-preferred wrongness mode (current anchor)

{flavor_block}

This is the PREFERRED shape WITHIN the content_gap and clean classes for this anchor's pedagogical flavor. It guides which wrongness mechanism to choose; it does NOT relax the english_gap prohibition.
"""


def _build_system_prompt_focused(difficulty_tier, prompt_version="v2", flavor=None):
    """~120 lines. Vocab-backed: LLM generates only creative content."""
    blooms_primary, blooms_secondary = BLOOMS_BY_TIER[difficulty_tier]
    primary_target = BLOOMS_PRIMARY[difficulty_tier]
    blooms_enforcement = _blooms_stem_enforcement(difficulty_tier, blooms_primary, blooms_secondary)

    return f"""You are an expert EPPP exam question author for PassEPPP.
Generate a multiple-choice question at difficulty tier {difficulty_tier} ({DIFFICULTY_LABELS[difficulty_tier]}).

## STEM-FACT INTEGRITY (READ FIRST — non-negotiable)

**No distractor may lexically contradict any fact stated in the stem.**

If the stem states a laterality, a numeric ratio, an age, a stage (childhood/adulthood, acute/chronic), or a named outcome — distractors must be wrong via MECHANISM or CONCEPT MISUNDERSTANDING, never by inverting the stated fact. A student who can reject your distractor by re-reading the stem alone signals a quality failure called "english_gap".

**STEM HYGIENE:**
- NEVER print numbers, ratios, named outcomes, lateralities, or stage timings that any distractor lexically contradicts.
- NEVER use meta-evaluative modifiers in the stem ("correctly", "best", "most", "option").
- NEVER lead distractors with imperative verbs. Distractors must be answer-form (noun phrases or declarative claims).
- All four option leads must share grammatical form.

**AMBIGUITY GUARD (especially T3/T4):** the keyed answer must be UNIQUELY correct under any reasonable interpretation. If a distractor is defensible under an alternative reading (overlap zones, framework choice, boundary cases), revise the stem to eliminate the alternative reading or revise the distractor.

## Pedagogical Framework
Error-driven learning: distractors are the PRIMARY teaching mechanism. Each distractor must map to a specific cognitive error a real test-taker might make. Tier {difficulty_tier} requires genuine cognitive effort at the {primary_target} level.

## EPPP Difficulty Calibration
- Easy (Tier 1): Below exam level — foundational knowledge (primary: recall)
- Medium (Tier 2): Approaching exam level — comprehension and simple application (primary: understand)
- **Hard (Tier 3): Approximate EPPP difficulty** — application to scenarios (primary: apply)
- Expert (Tier 4): Above exam level — critiquing, evaluating nuance (primary: evaluate)
You are generating Tier {difficulty_tier} ({DIFFICULTY_LABELS[difficulty_tier]}).

## Bloom's Level: {primary_target.upper()} (design target) / {blooms_secondary.upper()} (permitted ceiling/floor)
Optimize for {primary_target}-level verbs: {get_blooms_verbs(primary_target)}
{blooms_enforcement}
## Distractor Design
You must include exactly 3 wrong answers. Use concepts from the Canonical Vocabulary in the user prompt.
Each distractor must probe a DIFFERENT failure mode:
1. **Foundational probe** — "Do they know this concept?" → triggers concept flashcard
2. **Discrimination probe** — "Can they distinguish from nearest neighbor?" → triggers comparison flashcard
3. **Application probe** — "Do they understand when/how it applies?" → triggers nuance flashcard

**misconception_type** (exactly one per distractor):
- `similar_name` — Confusable names
- `similar_property` — Different concepts sharing surface feature
- `similar_store` — Same system, different modality/subtype
- `opposite_direction` — Reversed direction/effect
- `overgeneralization` — Rule applied beyond valid scope
- `partial_understanding` — Correct concept, wrong context/application

{_mcq_quality_rules()}
## Explanation Quality (Primary Teaching Mechanism)
- **Correct answer**: WHY it's correct in 2-3 sentences. Reference underlying principle.
- **Wrong answers**: Use this diagnostic structure — OPEN with why the student picked this, THEN correct:
  1. **Diagnose first**: Lead with a sentence addressing the student's likely reasoning (vary the phrasing):
    - `similar_name`: "You may have confused X with Y because their names sound similar. However, ..."
    - `similar_property`: "This is a common mix-up because X and Y both [shared feature]. The key difference is ..."
    - `similar_store`: "Since X and Y are both types of [system], it's easy to confuse them. However, ..."
    - `opposite_direction`: "It's common to reverse the direction here. Actually, ..."
    - `overgeneralization`: "While the rule about X is correct in [context], it doesn't extend to ..."
    - `partial_understanding`: "This is partially right — [true part]. However, in this specific case ..."
  2. **Correct**: Why this option fails and what's actually true (1-2 sentences).
  3. **Distinguish**: One key fact or rule that prevents this error.

## Flashcard Seeds
Generate 3 remediation flashcards. ALL seeds MUST be anchored to the tested_concept — not to a distractor or tangential detail:
- **concept**: Direct factual question about the TESTED CONCEPT → clear answer with 2-3 key details
- **comparison**: Compare the TESTED CONCEPT with the most confused concept from the distractors → "X vs Y — what distinguishes?" → parallel differences (3-4 lines)
- **nuance**: Edge case or exception about the TESTED CONCEPT — "When/how does this apply differently?" → 2-3 sentences
{(_distractor_quality_framework_section_v2(flavor) if prompt_version == "v2" else "")}
## Output Format
Return ONLY valid JSON (no markdown, no explanation):
{{
  "question_stem": "...",
  "tested_concept": {{
    "concept_id": "kebab-case-from-vocab",
    "concept_label": "Human-Readable Label",
    "knowledge_tested": "What the student must know to answer correctly"
  }},
  "correct_answer": {{
    "text": "Option text for the correct answer",
    "explanation": "2-3 sentence teaching explanation"
  }},
  "distractors": [
    {{
      "slot": 1,
      "text": "Wrong answer text",
      "explanation": "Teaching explanation addressing this specific error",
      "concept_id": "kebab-case-id-of-wrong-concept",
      "concept_label": "Label of the concept this distractor represents",
      "misconception_id": "concept-a-vs-concept-b",
      "misconception_label": "Human-readable confusion label",
      "misconception_type": "similar_name|similar_property|similar_store|opposite_direction|overgeneralization|partial_understanding"
    }},
    {{ "slot": 2, "text": "...", "explanation": "...", "concept_id": "...", "concept_label": "...", "misconception_id": "...", "misconception_label": "...", "misconception_type": "..." }},
    {{ "slot": 3, "text": "...", "explanation": "...", "concept_id": "...", "concept_label": "...", "misconception_id": "...", "misconception_label": "...", "misconception_type": "..." }}
  ],
  "flashcard_seeds": {{
    "concept": {{ "front": "...", "back": "..." }},
    "comparison": {{ "front": "...", "back": "..." }},
    "nuance": {{ "front": "...", "back": "..." }}
  }}
}}

Use concept_ids and misconception_ids from the Canonical Vocabulary provided in the user prompt.
The tested_concept.concept_id MUST match the correct answer's concept.
Ensure exactly 3 distractors with unique misconception_ids."""


def _build_system_prompt_open(difficulty_tier, prompt_version="v2", flavor=None):
    """~200 lines. No vocab: LLM generates concept/misconception IDs too."""
    blooms_primary, blooms_secondary = BLOOMS_BY_TIER[difficulty_tier]
    primary_target = BLOOMS_PRIMARY[difficulty_tier]
    mix = DISTRACTOR_MIX[difficulty_tier]
    mix_desc = ", ".join(f"{v}x {k}" for k, v in mix.items())
    blooms_enforcement = _blooms_stem_enforcement(difficulty_tier, blooms_primary, blooms_secondary)

    return f"""You are an expert EPPP exam question author for the PassEPPP adaptive learning platform.
You generate multiple-choice questions for chapter quizzes at difficulty tier {difficulty_tier} ({DIFFICULTY_LABELS[difficulty_tier]}).

## STEM-FACT INTEGRITY (READ FIRST — non-negotiable)

**No distractor may lexically contradict any fact stated in the stem.**

If the stem states a laterality (left/right, bilateral/unilateral), a numeric ratio (2:1), an age, a stage (childhood/adulthood, acute/chronic), or a named outcome — distractors must be wrong via MECHANISM or CONCEPT MISUNDERSTANDING, never by inverting the stated fact. A student who can reject your distractor by re-reading the stem alone (no domain knowledge needed) signals a quality failure called "english_gap".

**STEM HYGIENE — these prevent english_gap and editorial issues:**
- NEVER print numbers, ratios, named outcomes, lateralities, or stage timings that any distractor lexically contradicts. Either drop the specific fact from the stem, or ensure no distractor inverts it.
- NEVER use meta-evaluative modifiers in the stem ("correctly", "best", "most", "option", "which option").
- NEVER lead distractors with imperative verbs ("Identify", "Predict", "Classify", "Determine"). Distractors must be answer-form (noun phrases or declarative claims).
- All four option leads must share grammatical form (parallelism).

**AMBIGUITY GUARD (especially T3/T4):** the keyed answer must be UNIQUELY correct under any reasonable interpretation of the stem. If a distractor is defensible under an alternative reading (overlap zones, framework choice, boundary cases), revise the stem to eliminate the alternative reading or revise the distractor to make it clearly wrong.

## Pedagogical Framework
Error-driven learning: distractors are the PRIMARY teaching mechanism. Wrong answers + targeted feedback produce deeper learning than correct answers alone. Your distractors must each map to a specific cognitive error a real test-taker might make, and your explanation must diagnose that error before stating the correct concept.

## EPPP Context
- Easy (Tier 1): Below exam level — foundational recall (primary: remember)
- Medium (Tier 2): Approaching exam level — comprehension and simple application (primary: understand)
- **Hard (Tier 3): Approximate EPPP difficulty** — application to scenarios (primary: apply)
- Expert (Tier 4): Above exam level — evaluation and nuanced judgment (primary: evaluate)
You are generating Tier {difficulty_tier} ({DIFFICULTY_LABELS[difficulty_tier]}).

## Bloom's Level: {primary_target.upper()} (design target) / {blooms_secondary.upper()} (permitted ceiling/floor)
- Optimize for {primary_target}-level verbs (e.g., {get_blooms_verbs(primary_target)})
- Do NOT write questions above or below the target cognitive level for this tier
{blooms_enforcement}
## Distractor Design: Diagnostic Spread Rule
You must include exactly 3 wrong answers with this distractor-level mix: {mix_desc}

Distractor levels:
- L1 (Cross-subdomain): Wrong answer from a different subdomain. Tests: "Did they study this chapter?"
- L2 (Same-subdomain): Wrong answer from same subdomain, different concept. Tests: "Do they understand distinctions within this topic?"
- L3 (Same-concept-family): Common misconception or frequently confused concept. Tests: "Can they discriminate between closely related concepts?"
- L4 (Partially-correct): True statement that doesn't answer THIS question. Tests: "Can they evaluate which is MOST correct?"

Each distractor MUST probe a DIFFERENT failure mode:
1. **Foundational probe** → triggers `concept` flashcard
2. **Discrimination probe** → triggers `comparison` flashcard
3. **Application probe** → triggers `nuance` flashcard

## Concept & Misconception ID Consistency
**concept_id**: Kebab-case `{{subsystem}}-{{specific-concept}}` (e.g., `sensory-memory-iconic-duration`).
**misconception_id**: Format `{{concept-a}}-vs-{{concept-b}}`.
**confused_with**: Each distractor's `confused_with` MUST contain the correct answer's concept_id.
**misconception_type** (exactly one per distractor):
- `similar_name` — Confusable names
- `similar_property` — Different concepts sharing surface feature
- `similar_store` — Same system, different modality/subtype
- `opposite_direction` — Reversed direction/effect
- `overgeneralization` — Rule applied beyond valid scope
- `partial_understanding` — Correct concept, wrong context/application

{_mcq_quality_rules()}
## Explanation Quality (Error-Driven Learning)
- **Correct answer**: WHY it's correct in 2-3 sentences. Reference underlying principle.
- **Wrong answers**: Use this diagnostic structure — OPEN with why the student picked this, THEN correct:
  1. **Diagnose first**: Lead with a sentence addressing the student's likely reasoning (vary the phrasing):
    - `similar_name`: "You may have confused X with Y because their names sound similar. However, ..."
    - `similar_property`: "This is a common mix-up because X and Y both [shared feature]. The key difference is ..."
    - `similar_store`: "Since X and Y are both types of [system], it's easy to confuse them. However, ..."
    - `opposite_direction`: "It's common to reverse the direction here. Actually, ..."
    - `overgeneralization`: "While the rule about X is correct in [context], it doesn't extend to ..."
    - `partial_understanding`: "This is partially right — [true part]. However, in this specific case ..."
  2. **Correct**: Why this option fails and what's actually true (1-2 sentences).
  3. **Distinguish**: One key fact or rule that prevents this error.

## Flashcard Seeds
ALL seeds MUST be anchored to the tested_concept — not to a distractor or tangential detail:
- **concept**: Direct factual question about the TESTED CONCEPT → clear answer with 2-3 key details
- **comparison**: Compare the TESTED CONCEPT with the most confused concept from the distractors → "X vs Y — what distinguishes?" → parallel differences (3-4 lines)
- **nuance**: Edge case or exception about the TESTED CONCEPT — "When/how does this apply differently?" → 2-3 sentences

## Topic Keywords
3-5 specific psychology terms/concepts tested in this question. NOT structural labels.
{(_distractor_quality_framework_section_v2(flavor) if prompt_version == "v2" else "")}
## Output Format
Return ONLY valid JSON:
{{
  "question_stem": "...",
  "tested_concept": {{
    "concept_id": "kebab-case-id",
    "concept_label": "Human-readable label",
    "knowledge_tested": "What student must know"
  }},
  "correct_answer": {{
    "text": "...",
    "explanation": "..."
  }},
  "distractors": [
    {{
      "slot": 1,
      "text": "...",
      "explanation": "...",
      "concept_id": "kebab-case-id",
      "concept_label": "Human-readable label",
      "misconception_id": "concept-a-vs-concept-b",
      "misconception_label": "Human-readable confusion label",
      "misconception_type": "similar_name|similar_property|similar_store|opposite_direction|overgeneralization|partial_understanding"
    }},
    {{ "slot": 2, ... }},
    {{ "slot": 3, ... }}
  ],
  "flashcard_seeds": {{
    "concept": {{ "front": "...", "back": "..." }},
    "comparison": {{ "front": "...", "back": "..." }},
    "nuance": {{ "front": "...", "back": "..." }}
  }},
  "topic_keywords": ["specific psychology term 1", "term 2", "term 3"]
}}

IMPORTANT: distractor_level is assigned externally — do NOT include it in your output.
The tested_concept.concept_id MUST match the concept the correct answer tests.
Ensure exactly 3 distractors with unique misconception_ids."""




# ══════════════════════════════════════════════════════════════
# User Prompt
# ══════════════════════════════════════════════════════════════

def build_user_prompt(anchor_info, passage, anchor_data, source_type, variant_num,
                      domain_name, difficulty_tier, concept_vocab=None,
                      character=None, target_position=None,
                      tested_concept=None, distractor_plan=None,
                      core_claims=None, question_angle=None,
                      concept_integration=None, correct_answer_form=None):
    """Build user prompt for a single question generation call.

    Args:
        anchor_info: dict with chapter_title, anchor_id_v2, uid
        passage: textbook passage text (from anchor_passages_v3, or "" for proprietary anchors)
        anchor_data: list with one anchor dict (uid, verbatim_anchor, testable_fact)
        source_type: anchor_grounded or integrated
        variant_num: 1-5
        domain_name: human-readable domain name
        difficulty_tier: 1-4 (determines which stem pattern pool to draw from)
        concept_vocab: ConceptVocabAgent output (concepts, misconceptions)
        character: dict from get_character_assignment() (clinician_name, client_name, etc.)
        tested_concept: TestedConceptSelectorAgent output (concept_id, concept_label)
    """
    chapter_title = anchor_info.get("chapter_title", "")
    anchor_id_v2 = anchor_info.get("anchor_id_v2", "")

    if passage:
        if len(passage) <= 3000:
            passage_snippet = passage
        else:
            first = passage[:1200]
            mid_start = len(passage) // 2 - 400
            middle = passage[mid_start:mid_start + 800]
            last = passage[-1000:]
            passage_snippet = f"{first}\n\n[...]\n\n{middle}\n\n[...]\n\n{last}"
    else:
        passage_snippet = "(no textbook passage — proprietary/lecture anchor)"

    if anchor_data:
        a = anchor_data[0]
        uid = a.get('uid', a.get('id', ''))
        verbatim = a.get('verbatim_anchor', a.get('text', ''))
        testable = a.get('testable_fact', '')
        anchor_text = f"- [{uid}] {verbatim}"
        if testable:
            anchor_text += f"\n  Testable Fact: {testable}"
    else:
        anchor_text = "(no anchor data)"

    pattern_name, pattern_desc = get_stem_pattern(difficulty_tier, variant_num)
    if target_position is None:
        target_position = CORRECT_POSITIONS[(variant_num - 1) % len(CORRECT_POSITIONS)]

    source_instructions = {
        "anchor_grounded": "Anchor-grounded: Test the anchor concept directly. Use passage context for plausible distractors.",
        "integrated": "Integrated: Require synthesizing the anchor concept with broader passage content.",
    }

    character_section = ""
    if character:
        from .names import build_character_block
        character_section = "\n" + build_character_block(character, stem_pattern=pattern_name)

    concept_vocab_section = ""
    if concept_vocab and concept_vocab.get("has_vocab"):
        vocab_lines = [
            "",
            "## Canonical Concept Vocabulary (MUST USE THESE IDs)",
            "Use ONLY the following concept_ids for this anchor.",
            "",
        ]
        if tested_concept and tested_concept.get("has_tested_concept"):
            vocab_lines.append(
                f"**TARGET CONCEPT TO TEST**: `{tested_concept['concept_id']}` "
                f"— {tested_concept['concept_label']}"
            )
            vocab_lines.append(
                "Your question's tested_concept.concept_id MUST be this concept. "
                "Use other concepts from the list below as distractors."
            )
            vocab_lines.append("")
        concepts = concept_vocab.get("concepts", [])
        if concepts:
            vocab_lines.append("**Concepts:**")
            for c in concepts:
                vocab_lines.append(f"- `{c['concept_id']}` — {c['label']}")
            vocab_lines.append("")
        misconceptions = concept_vocab.get("misconceptions", [])
        if misconceptions:
            vocab_lines.append("**Misconceptions:**")
            for m in misconceptions:
                involved = ", ".join(f"`{x}`" for x in m.get("concepts_involved", []))
                vocab_lines.append(
                    f"- `{m['misconception_id']}` — {m['label']} "
                    f"(type: {m['type']}, concepts: {involved})"
                )
            vocab_lines.append("")
        concept_vocab_section = "\n".join(vocab_lines)

    core_claims_section = ""
    if core_claims:
        cc_lines = [
            "",
            "## Anchor Core Claims (MUST ADDRESS at least one)",
            "This question MUST test at least one of the following claims from this specific anchor.",
            "Do NOT drift to other topics in the chapter — stay grounded in this anchor's content.",
            "",
        ]
        for i, claim in enumerate(core_claims, 1):
            cc_lines.append(f"{i}. {claim}")
        cc_lines.append("")
        core_claims_section = "\n".join(cc_lines)

    question_angle_section = ""
    if question_angle and question_angle.get("has_angle"):
        question_angle_section = f"""
## Question Angle (recommended approach for this anchor + tier)
**Angle type**: {question_angle["angle_type"]}
**Guidance**: {question_angle["angle_description"]}

Use this angle to shape your question's content approach. The stem pattern above
defines the FORMAT; this angle defines the CONTENT FOCUS. Adapt the angle to fit
the required stem pattern — do not override the pattern.
"""

    answer_form_section = ""
    if correct_answer_form:
        # Hard scaffold for the correct answer's form. Mirrors the
        # DistractorPlanner pre-assignment that took distractor uniqueness
        # to ~100% compliance. Without this contract the LLM defaults to
        # bare-label T3 answers, "X because Y" option text, and
        # concept-bloated correct answers when the brief is concept-rich.
        verb = correct_answer_form.get("required_verb", "")
        verb_pool = correct_answer_form.get("verb_pool", []) or []
        option_form = correct_answer_form.get("option_form_constraint", "")
        option_length = correct_answer_form.get("option_length_constraint", "")
        permitted_labels = correct_answer_form.get("permitted_concept_labels", []) or []
        permitted_ids = correct_answer_form.get("permitted_concept_ids", []) or []
        max_concepts = correct_answer_form.get("max_concept_count", 1)

        verb_alt = ", ".join(f"`{v}`" for v in verb_pool) if verb_pool else ""
        if permitted_labels and permitted_ids:
            concept_lines = "\n".join(
                f"  - `{cid}` ({lbl})"
                for cid, lbl in zip(permitted_ids, permitted_labels)
            )
            concept_block = (
                f"- The correct answer's claim MAY reference ONLY these "
                f"concept(s) (max {max_concepts}):\n{concept_lines}\n"
                f"  Other brief concepts MUST NOT be woven into the correct "
                f"answer's claim. Distractors must address the SAME concept "
                f"set so scope is symmetric.\n"
            )
        else:
            concept_block = (
                f"- The correct answer should reference at most "
                f"{max_concepts} concept(s) — keep scope tight so distractors "
                f"can mirror it.\n"
            )

        # Plan N: tier-specific verb-at-start mandate. T3+ generations had
        # 30-50% per-question variance because the LLM placed the verb in
        # the stem and wrote a noun-phrase answer. Mandating the verb as
        # the FIRST WORD of the option text collapses the placement
        # ambiguity — same specificity-collapse pattern as the L3 vocab
        # scaffold (which took KeywordDistributionGate failures to 0).
        verb_at_start_clause = ""
        if difficulty_tier >= 3 and verb_pool:
            cap_pool = ", ".join(v.capitalize() for v in verb_pool)
            verb_at_start_clause = (
                f" Furthermore, ALL FOUR options' text fields MUST BEGIN "
                f"with the SAME verb as the very first word (capitalized "
                f"as a sentence opener). Pick ONE verb from this pool "
                f"and use it as the opener for the correct option AND "
                f"all three distractors: {cap_pool}. The verb identity "
                f"must be UNIFORM across all four options — students "
                f"must not be able to identify the correct option by "
                f"which verb it starts with. Bare-label noun-phrase "
                f"openers (e.g., 'Hemiplegia of the right side', "
                f"'Right-sided weakness') are forbidden in any option — "
                f"every option starts with the chosen verb."
            )

        # T3 (Apply) ONLY: the correct option must contain a MECHANISM
        # marker — a causal connective that links the prediction to its
        # cause. Empirically the LLM produces "Predict X as part of Y"
        # labeling-as-prediction ~40% of the time at T3 without explicit
        # mechanism guidance; the apply_identity gate then fires and
        # retry begins. Putting the requirement in the INITIAL prompt
        # (not just retry guidance) drops the labeling rate substantially
        # — same prevention-vs-validation move as Phase 7's "TWO distinct
        # vocab terms per distractor" rule. T1/T2/T4 unaffected; this
        # is curated to T3's Apply identity.
        mechanism_marker_clause = ""
        if difficulty_tier == 3:
            mechanism_marker_clause = (
                " The correct option MUST contain a CAUSAL ANCHOR — "
                "either a MECHANISM marker (for mechanism-rich content "
                "like biopsych) OR a CRITERION-APPLICATION marker (for "
                "criteria-driven content like DSM diagnoses, ethical "
                "thresholds, duration/age cutoffs).\n"
                "Mechanism markers: `from`, `via`, `through`, "
                "`reflecting`, `producing`, `mediated by`, `resulting "
                "from`, `with [poor/impaired/reduced/disrupted] X`, "
                "`by [verb-ing] Y`.\n"
                "Criterion-application markers: `based on [X]`, "
                "`given that [X]`, `satisfying/failing/exceeding the "
                "[criterion/threshold/cutoff]`, `for [meeting/failing/"
                "exceeding] [criterion-language]`.\n"
                "CRITICAL — `because`, `since`, `due to`, `owing to` "
                "are FORBIDDEN in option text (they are stripped to "
                "explanation, leaving the option as a bare label that "
                "fails apply-identity). When you want to express "
                "'X meets/fails Y because of Z', INSTEAD WRITE one of:\n"
                "  • `Determine X are met BASED ON Z`\n"
                "  • `Determine X are unmet, FAILING THE [criterion] "
                "of Z`\n"
                "  • `Determine X are met, SATISFYING THE [threshold] "
                "of Z`\n"
                "  • `Determine X are unmet GIVEN THAT Z`\n"
                "These phrasings carry the same meaning as `because Z` "
                "but survive the option-text contract.\n"
                "Forbidden labeling patterns (T2 cognition under a T3 "
                "stem): `as part of [the triad]`, `as the [X] "
                "component of [the triad]`, `alongside [the rest]`, "
                "`consistent with [the syndrome]`. Example clean "
                "(mechanism): `Predict mood fluctuations REFLECTING "
                "lost limbic regulation`. Example clean (criterion): "
                "`Determine ADHD criteria are met BASED ON age-9 "
                "onset and two-setting pervasiveness`. Example "
                "forbidden: `Predict mood fluctuations AS PART OF "
                "the frontal triad`. Apply-tier identity requires "
                "the answer to explain WHY/HOW (mechanism) or BY "
                "WHICH RULE (criterion), not WHICH category-label."
            )

        answer_form_section = (
            "\n## REQUIRED CORRECT-ANSWER FORM (hard contract)\n"
            "The correct answer is constrained on four dimensions. "
            "Violating any of them will fail validation.\n\n"
            f"- The correct answer's claim MUST express a `{verb}` "
            f"action (or another verb from this pool: {verb_alt}). "
            f"This verb MUST appear in the correct option's `text` field "
            f"itself — not merely in the question stem. Even when the "
            f"answer describes an outcome or syndrome, USE an action "
            f"verb: write 'Right-sided weakness DEVELOPS contralateral "
            f"to the lesion' (verb=develops), NOT 'Hemiplegia of the "
            f"right side' (bare-label noun phrase). Bare-label noun "
            f"phrases for clinical syndromes/conditions are the most "
            f"common T3 anti-pattern — they fail validation."
            f"{verb_at_start_clause}"
            f"{mechanism_marker_clause}\n"
            f"- {option_form}\n"
            f"- {option_length}\n"
            f"{concept_block}"
        )

    integration_section = ""
    if concept_integration and concept_integration.get("requires_integration"):
        # Hard scaffold for Tier 4: pre-assigns the 2 concepts the question
        # MUST integrate. Mirrors the DistractorPlanner pre-assignment that
        # eliminated the duplicate-misconception failure mode at scale.
        integration_section = (
            "\n## REQUIRED 2-Concept Integration (Tier 4)\n"
            "This question MUST require integrating BOTH of the following concepts. "
            "Neither concept ALONE should be sufficient to answer correctly. "
            "If a student knows only one of these, the correct answer must remain "
            "ambiguous to them.\n\n"
            f"- **Primary concept**: `{concept_integration.get('primary_concept_id', 'UNKNOWN')}` "
            f"({concept_integration.get('primary_concept_label', '')})\n"
            f"- **Secondary concept**: `{concept_integration.get('secondary_concept_id', 'UNKNOWN')}` "
            f"({concept_integration.get('secondary_concept_label', '')})\n\n"
            "Both concepts must be referenced (by name or by definitional content) "
            "in the question stem AND in the correct answer. Distractors should "
            "reflect partial integration — using only one of the two concepts, or "
            "applying both incorrectly. Single-concept-sufficient correct answers "
            "are the dominant T4 failure pattern and will be rejected.\n"
        )

    distractor_assignment_section = ""
    if distractor_plan and distractor_plan.get("mode") == "focused":
        da_lines = [
            "",
            "## Pre-Assigned Distractor Misconceptions (MUST USE — DO NOT CHANGE)",
            "Each distractor slot has a pre-assigned misconception_id. You MUST use exactly these IDs.",
            "",
        ]
        for slot in distractor_plan.get("slots", []):
            mid = slot.get("misconception_id")
            if mid:
                da_lines.append(
                    f"- **Slot {slot['slot']}** (L{slot['distractor_level']}): "
                    f"`{mid}` — {slot.get('misconception_label', '')} "
                    f"(type: {slot.get('misconception_type', '')})"
                )
        # Cross-planner alignment: surface the form planner's permitted concept
        # set as a scope constraint here. Without this, distractors target
        # only the tested concept's misconception while the correct answer
        # weaves N concepts (cap by tier), and ScopeMatchGate fires at T4.
        # Connecting the two planners' outputs at the prompt-rendering layer
        # is the smallest fix — DistractorPlannerAgent itself is unchanged.
        permitted_labels = (
            (correct_answer_form or {}).get("permitted_concept_labels", []) or []
        )
        if permitted_labels:
            scope_list = ", ".join(f"**{lbl}**" for lbl in permitted_labels)
            n_concepts = len(permitted_labels)
            # Plan O: tighten "MUST reference these concepts" (which the LLM
            # treats as set-of-options) to "MUST mention ALL of the
            # following concept names" with explicit count and example.
            # Eliminates the under-scoped distractor pattern where one
            # distractor mentions only one concept while correct mentions
            # all N — a scope_match resurgence we saw on regen v5.
            example_pair = (
                f"If the correct option mentions {permitted_labels[0]} AND "
                f"{permitted_labels[1] if len(permitted_labels) > 1 else permitted_labels[0]}, "
                f"distractor texts must mention BOTH — saying just "
                f"'{permitted_labels[0]}' is not enough."
            ) if n_concepts >= 2 else ""
            da_lines.append("")
            da_lines.append(
                "**Distractor scope MUST mirror the correct answer's concept set.** "
                f"Each distractor's text MUST mention ALL {n_concepts} of the "
                f"following concept names (or close synonyms): {scope_list}. "
                f"{example_pair} Distractors invoke the assigned misconception_id "
                "but must weave the broader concept scope into the surface text — "
                "not stay focused on a single concept. Asymmetric scope (correct "
                f"mentions all {n_concepts}, distractor mentions <{n_concepts}) "
                "is a testwise tell that students rule out by spotting the "
                "missing scope, not by reasoning about the concepts."
            )

        # L3 vocabulary scaffold: surface the technical terms extracted from
        # the brief's concept descriptions so distractors share vocabulary
        # with the correct answer. Closes the L1+L2-only gap that left
        # KeywordDistributionGate firing at ~30% on the D7-PHY-209 phase 2
        # batches — the LLM had no upfront list of "vocabulary that must
        # appear in distractors" and produced technical-term clusters that
        # the gate then rejected. Pre-assigning specific terms collapses
        # the LLM's "guess what vocabulary to share" decision the same way
        # DistractorPlanner pre-assigns misconception_ids.
        permitted_vocab = (
            (correct_answer_form or {}).get("permitted_vocabulary", []) or []
        )
        if permitted_vocab:
            vocab_list = ", ".join(f"`{v}`" for v in permitted_vocab)
            da_lines.append("")
            # Phase 7: at T1/T2 the pool is broader (concept descriptions
            # + brief-internal pool + curated domain pool, cap 12), and
            # the per-distractor minimum bumps from ONE to TWO so the
            # extra pool size translates into actual vocabulary spread.
            # T3/T4 keep the original "at least ONE" rule because their
            # pool is narrower (cap 8) and Bloom's identity already
            # demands concept-anchored distractors.
            min_terms = "TWO distinct" if difficulty_tier <= 2 else "ONE"
            da_lines.append(
                "**Required distractor vocabulary (DRAW FROM, not all required).** "
                "The correct option will likely use technical terms from the "
                "brief's concept descriptions. To prevent vocabulary-clustering "
                "tells (where students recognize unique technical words and pick "
                "correct without engaging the concept), each distractor's text "
                f"MUST contain at least {min_terms} of these terms: {vocab_list}. "
                "Use them in contexts that make the distractor wrong — distractors "
                "should engage the same vocabulary as the correct answer, just "
                "applied incorrectly."
            )

        # Layer A — Stem-fact integrity (Phase 9). Sonnet audit on
        # D7-PHY-076 flagged 10/18 questions where a distractor inverted
        # a fact stated in the stem (bilateral → "surviving contralateral";
        # "still rides a bicycle" → "loss of procedural skills"; "recalls
        # his wedding" → "all pre-injury memories erased"). The pattern
        # scales with tier (T1 0%, T2 40%, T3 75%, T4 100%) because richer
        # stems pack more facts to contradict and the LLM picks the cheapest
        # wrong claim by inverting a stated one. A distractor rejectable
        # by reading the stem alone defeats the pedagogy — students rule
        # it out by re-reading, not by content knowledge.
        da_lines.append("")
        da_lines.append(
            "**Distractors must NOT contradict facts already stated in the stem.** "
            "If the stem states laterality (e.g., 'bilateral damage'), a named "
            "subject's preserved abilities (e.g., 'still rides a bicycle', "
            "'recalls his wedding'), observed findings, or onset details, NO "
            "distractor may invert that fact (e.g., claim 'unilateral injury', "
            "'loss of cycling', 'all pre-injury memories erased'). Make each "
            "distractor wrong via the underlying MECHANISM the question tests "
            "— not by flipping a surface fact the student can re-read. A "
            "distractor that a student rejects by reading the stem alone (no "
            "content knowledge required) is a quality failure."
        )
        da_lines.append("")
        distractor_assignment_section = "\n".join(da_lines)

    return f"""Generate 1 quiz question for this anchor point.

DOMAIN: {domain_name}
CHAPTER: {chapter_title}
ANCHOR POINT: {anchor_id_v2}
SOURCE TYPE: {source_type}
VARIANT: {variant_num} of 5

## Variant Constraints
- **Required stem pattern**: {pattern_desc}
- **Target correct-answer position**: Place the correct option at letter {target_position}
- Make this question DISTINCT from other variants — different scenario, different tested aspect

## Textbook Passage (context for this anchor — never quote verbatim):
{passage_snippet}

## Anchor Point (verbatim_anchor + testable_fact — never quote verbatim):
{anchor_text}

## Source Type Instructions:
{source_instructions.get(source_type, source_instructions["anchor_grounded"])}
{core_claims_section}{question_angle_section}{integration_section}{answer_form_section}{concept_vocab_section}{distractor_assignment_section}{character_section}
Generate the question now. Return ONLY the JSON object."""


# ══════════════════════════════════════════════════════════════
# Correction Prompt (feedback-driven retry)
# ══════════════════════════════════════════════════════════════

import re as _re
_STEM_ELIMINABLE_STRATEGY_RE = _re.compile(r"^\[strategy=(\w+)\]\s*")


def _stem_eliminable_guidance(tier=None, reason=""):
    """Strategy-aware guidance for the stem_eliminable_distractor gate.

    The gate prepends `[strategy=<name>]` to its failure reason when it
    resolved a non-DEFAULT cell from pipeline.distractor_policy. This
    function reads that tag and returns the correction-strategy-specific
    guidance. Falls back to tier-based logic for backward compat with
    callers that don't provide a reason.
    """
    base = (
        "A distractor's claim is contradicted by a fact stated in the "
        "stem, so a student can reject it by reading alone — no content "
        "knowledge required. Rewrite so the distractor is wrong via the "
        "underlying concept the question tests."
    )

    strategy = None
    m = _STEM_ELIMINABLE_STRATEGY_RE.match(reason or "")
    if m:
        strategy = m.group(1)

    if strategy == "judgment_error":
        return base + " " + (
            "STRATEGY=JUDGMENT_ERROR (T4 EVALUATE): the stem is a rich "
            "vignette with multiple discriminating facts; eliminating ALL "
            "stem-contradiction is over-constraint and may be impossible. "
            "Aim instead for JUDGMENT ERRORS — the distractor should be "
            "the conclusion a student would defend IF they over-weighted "
            "ONE piece of evidence, applied the wrong inferential rule, "
            "or used a related-but-wrong framework. The distractor stays "
            "inside the stem's evidence space; the synthesis is what's "
            "wrong. Pattern: 'Conclude X via [framework that fits some "
            "stem facts but ignores the decisive ones]' rather than "
            "'Claim Y where Y is directly refuted by stem fact Z'."
        )
    if strategy == "rewrite_stem_to_observation":
        return base + " " + (
            "STRATEGY=REWRITE_STEM (STRUCTURAL CEILING): this question's "
            "stem prints the diagnostic criterion being tested (e.g., "
            "'no measurable activity' is the textbook definition of an "
            "antagonist). Distractors that vary on the criterion will "
            "necessarily collapse against the printed text. The fix is "
            "to REWRITE THE STEM so the criterion becomes a clinical "
            "observation requiring the concept to interpret — NOT to "
            "rewrite distractors. Replace the printed definition with a "
            "specific observed phenomenon (lab finding, response curve, "
            "patient behavior) that the student must classify using the "
            "concept. Maintain the Bloom's tier cognitive demand. "
            "Distractors stay; only the stem changes."
        )
    if strategy == "mechanism_inversion":
        return base + " " + (
            "STRATEGY=MECHANISM_INVERSION (T3 APPLY): shift wrongness from "
            "fact-inversion to mechanism-inversion. The distractor should "
            "predict the wrong outcome via an incorrect mechanism, not by "
            "contradicting a fact about the scenario. The student rejects "
            "the distractor because they understand the mechanism, not "
            "because they re-read the scenario."
        )
    if strategy == "framework_misapplication":
        return base + " " + (
            "STRATEGY=FRAMEWORK_MISAPPLICATION (ethics): the distractor "
            "should apply the WRONG framework correctly to the scenario "
            "(e.g., applies a research-ethics rule to a clinical-ethics "
            "case). The student rejects the distractor because they "
            "select the right framework for the question type."
        )

    # Fallback: tier-based branches preserve backward-compat with
    # callers that don't pass `reason` (older code paths or external
    # invocations).
    if tier == 4:
        return base + " " + (
            "AT T4 (EVALUATE): the stem is a rich vignette with multiple "
            "discriminating facts; eliminating ALL stem-contradiction is "
            "over-constraint and may be impossible. Aim instead for "
            "JUDGMENT ERRORS."
        )
    if tier == 3:
        return base + " " + (
            "AT T3 (APPLY): shift wrongness from fact-inversion to "
            "mechanism-inversion. The distractor should predict the "
            "wrong outcome via an incorrect mechanism, not by "
            "contradicting a fact about the scenario."
        )
    return base


_GATE_GUIDANCE = {
    "structure": (
        "Your JSON output was structurally invalid. Ensure your response contains: "
        "question_stem, tested_concept (with concept_id, concept_label, knowledge_tested), "
        "correct_answer (with text, explanation), exactly 3 distractors "
        "(each with slot, text, explanation, concept_id, concept_label, "
        "misconception_id, misconception_label, misconception_type), "
        "and flashcard_seeds (concept, comparison, nuance — each with front and back)."
    ),
    "content_quality": (
        "Your flashcard seeds or topic keywords were insufficient. Requirements: "
        "all 3 flashcard types (concept, comparison, nuance) must have front (20+ chars) "
        "and back (40+ chars). Include 3-5 specific psychology topic keywords."
    ),
    "consistency": (
        "Your output had diagnostic consistency issues. Ensure: "
        "tested_concept.concept_id matches the correct answer's concept, "
        "each distractor has a UNIQUE misconception_id, "
        "all distractors have concept_id, misconception_type, and confused_with fields."
    ),
    "anchor_grounding": (
        "Your question drifted from this anchor's content. "
        "Re-read the Anchor Core Claims section and ensure your question "
        "directly tests one of those specific claims. Do not test other "
        "chapter topics — stay focused on this anchor."
    ),
    "assembly": (
        "Your output could not be assembled into a valid question record. "
        "Common causes: missing tested_concept.concept_id, fewer than 3 distractors, "
        "or incorrect JSON structure. Follow the output format exactly."
    ),
    "attribution": (
        "Your question attributed findings to a non-whitelisted researcher. "
        "Rewrite the stem, options, and explanations WITHOUT naming any "
        "researcher, year, or paper. Do not write 'According to X', "
        "'X (YYYY)', 'X et al.', or 'X's research/framework/theory'. "
        "Refer to the concept descriptively instead. The whitelisted "
        "eponyms (Piaget, Cannon-Bard, Pavlovian, etc.) listed in your "
        "system prompt are the only personal names permitted."
    ),
    "option_length_balance": (
        "Your option lengths reveal the correct answer. Rewrite the four "
        "options so they are within ~1.5x character length of each other. "
        "If the correct answer is the longest, COMPRESS it — strip "
        "qualifiers, parentheticals, or extra clauses — until it matches "
        "the distractors. Do NOT pad the distractors with filler; tighten "
        "the correct answer instead. The four options must be similar in "
        "length AND in structural complexity (parens, semicolons, "
        "compound clauses applied evenly)."
    ),
    "blooms_cognitive_level": (
        "Your question violates the Bloom's-tier cognitive demand. "
        "For Tier 3 (apply): the correct answer must require an "
        "application/analysis step — predicting, distinguishing, "
        "determining, choosing, evaluating, or comparing — not merely "
        "identifying a definition or label even if the stem is dressed "
        "as a scenario. Rewrite the correct answer so it expresses the "
        "RESULT of applying the concept to the scenario. "
        "For Tier 4 (evaluate): the question must integrate at least 2 "
        "concepts from this anchor's brief. If the correct answer can be "
        "derived from one concept's definition alone, the question is too "
        "easy for T4. Rewrite the stem AND correct answer so BOTH concepts "
        "are required to answer — neither alone should suffice."
    ),
    "domain_expertise": (
        "Your correct answer can be reached without psychology expertise. "
        "Rewrite the correct answer + explanation to use at least 2 "
        "technical terms from this anchor's brief vocabulary (concept "
        "labels or testable_fact key terms). A T2+ question must require "
        "domain-specific knowledge — a smart layperson with general world "
        "knowledge should NOT be able to answer it from common sense. "
        "Replace generic phrasing with domain vocabulary that signals the "
        "specific psychology concept being tested."
    ),
    "scope_match": (
        "Your distractors have asymmetric scope. For a comparison or "
        "best-answer question, when the correct answer addresses 2 "
        "concepts (e.g., agonist AND antagonist), each distractor must "
        "ALSO address that scope. A distractor that mentions only one of "
        "the compared concepts is testwise-defective — students rule it "
        "out by spotting the missing scope, not by understanding the "
        "concepts. Rewrite under-scoped distractors so they engage with "
        "BOTH concepts that the stem compares, but do so incorrectly."
    ),
    "option_claim": (
        "Your option text fields contain reasoning words (because, since, "
        "due to) that turn the answer into a self-justifying claim. "
        "Option text should be the CLAIM only; put the justification in "
        "the explanation field. Convert 'X because Y' option text into "
        "option text='X', explanation='because Y'. Comparative connectives "
        "(whereas, but, in contrast) ARE permitted for comparison-stem "
        "patterns and should be kept. Only the causal markers need to "
        "move to the explanation field."
    ),
    "originality": (
        "Your question recycles too much verbatim text from the anchor "
        "source. Paraphrase the stem and options using novel sentence "
        "structure and synonyms. The source material is context for what "
        "to test, not text to copy. A test-taker who has READ the source "
        "must still have to reason — verbatim recycling lets them "
        "pattern-match the stem against memorized text. Rewrite the "
        "stem from scratch using the same concept relationships."
    ),
    "keyword_distribution": (
        "Your correct option contains content words that don't appear in "
        "the stem or any distractor — students will pick the answer by "
        "recognizing the unique technical vocabulary, not by engaging the "
        "concept. Rewrite the distractors so they use the SAME technical "
        "vocabulary as the correct option (in contexts that make the "
        "distractor wrong). If your correct uses a synonym (e.g., "
        "'unilateral' for 'one side'), at least one distractor must also "
        "use that synonym attached to a wrong claim. Vocabulary should be "
        "distributed across all four options, not concentrated in correct."
    ),
    "remember_identity": (
        "Your T1 (Remember) question violates Recognition-tier identity. "
        "T1 stems must be DIRECT — no scenario, no vignette, no named "
        "subject ('Dr. X', age N, 'the patient', 'the client'). T1 "
        "correct answers must be STATIC FORMS — definitions, labels, "
        "feature lists — NOT predictions starting with 'predict'/"
        "'determine'/'apply'/'choose'/'select'. Rewrite the stem as a "
        "direct question (e.g., 'Which of the following defines X?') "
        "and the correct option as a noun phrase or short declarative "
        "claim."
    ),
    "understand_identity": (
        "Your T2 (Understand) question violates Comprehension-tier "
        "identity. T2 allows brief context (1-2 sentences MAX); longer "
        "stems drift into T3 application territory. Also, T2 must NOT "
        "use evaluative framings ('MOST appropriate', 'critique', "
        "'defend the choice') — those are T4. Tighten the stem to ≤280 "
        "characters / ≤40 words, and rewrite as a direct interrogative. "
        "Acceptable comprehension framings: 'Which describes X?' / "
        "'Which classifies Y?' / 'What distinguishes A from B?' / "
        "'How does X relate to Y?'. AVOID meta-evaluative modifiers — "
        "no 'correctly', 'best', 'most', 'option' in the stem. AVOID "
        "imperative-verb leads on distractors — options must be answers, "
        "not commands ('Identify…'/'Classify…'/'Recognize…' are forbidden)."
    ),
    "evaluate_identity": (
        "Your T4 (Analyze/Evaluate) question violates Evaluate-tier "
        "identity. T4 requires (1) a COMPLEX STIMULUS — multi-part "
        "case (≥2 sentences), conjunctive complexity (whereas/however/"
        "although), OR competing-claim framing (argues/claims/the "
        "position/expert reasoning) — and (2) DEFENSIBLE DISTRACTORS "
        "with sufficient content overlap with the correct option "
        "(Jaccard ≥0.20). Rewrite: enrich the stem with a multi-part "
        "case or competing claims; rewrite obvious-wrong distractors as "
        "plausible-but-inferior alternatives that share the correct "
        "option's reasoning vocabulary."
    ),
    "apply_identity": (
        "Your T3 (Apply) question violates Apply-tier identity. Bloom's "
        "Apply requires THREE elements: (1) a NOVEL SCENARIO in the "
        "stem, (2) the correct option must CARRY OUT a prediction or "
        "determination (verb + ≥4 content words), and (3) the correct "
        "option must contain a CAUSAL ANCHOR — either a MECHANISM "
        "marker (`from`, `via`, `through`, `reflecting`, `producing`, "
        "`mediated by`, `with [poor/impaired/etc.] X`, `by [verb-ing] "
        "Y`) or a CRITERION-APPLICATION marker (`based on [X]`, "
        "`given that [X]`, `satisfying/failing/exceeding the "
        "[criterion/threshold/cutoff]`, `for [meeting/failing/"
        "exceeding] [criterion-language]`). Without one, 'Predict X "
        "as part of Y' or 'Determine criteria are unmet' is T2 "
        "labeling under a T3 stem shell. Rewrite: include scenario "
        "indicators in the stem; rewrite the correct option as "
        "'[Verb] [outcome/decision] [mechanism-or-criterion-marker] "
        "[cause/criterion]' (e.g., 'Predict mood fluctuations "
        "reflecting lost limbic regulation', 'Determine ADHD "
        "criteria are met based on age-9 onset', NOT 'Predict mood "
        "fluctuations as part of the triad')."
    ),
    "topic_realm": (
        "Your distractors stray off-topic — only the correct option "
        "engages the concept's topic realm. Students will pick correct "
        "because it's the only option using realm vocabulary, not "
        "because they reasoned about the application/synthesis. Rewrite "
        "all three distractors as plausible-but-incorrect claims about "
        "the SAME concept and topic. Each distractor must use the same "
        "concept-vocabulary as the correct option (the brief's concept "
        "descriptions and the stem's terms), but applied incorrectly — "
        "wrong direction, wrong mechanism, wrong scope, etc. Distractors "
        "that talk about a different concept or stay tangentially-related "
        "make the question test topic recognition, not the cognitive "
        "level the question is meant to assess."
    ),
    "stem_keyword_distribution": (
        "The correct option repeats specific terms from the stem that NO "
        "distractor uses. Students will match the stem's keywords to the "
        "option containing them — picking correct by keyword-spotting "
        "rather than by reasoning. For each stem keyword that appears in "
        "the correct option, AT LEAST ONE distractor must also use that "
        "term (in a context that makes the distractor wrong). Stem terms "
        "should be distributed across all four options, never confined "
        "to the correct one alone."
    ),
    "laterality_integrity": (
        "A distractor inverts a laterality fact stated in the stem (stem "
        "says 'bilateral' → distractor says 'unilateral', or vice versa). "
        "A student rejects this by reading the stem alone — no content "
        "knowledge required — defeating the pedagogy. Make the distractor "
        "wrong via the underlying MECHANISM the question tests, not by "
        "flipping a stem-stated laterality fact. If the stem says "
        "'bilateral hippocampal damage', no distractor may claim "
        "'unilateral injury' or 'surviving contralateral hippocampus'."
    ),
    "universal_denial": (
        "A distractor uses a universal quantifier ('all', 'every', 'no', "
        "'none', 'never', 'regardless of') paired with denial language "
        "('erased', 'lost', 'abolished', 'impaired') for a category the "
        "stem cites a specific counterexample for (e.g., stem: 'still "
        "recalls his wedding' → distractor: 'all pre-injury memories "
        "erased'). The student rejects this by reading the stem alone — "
        "no content knowledge required. Reframe the distractor to be "
        "wrong via the underlying mechanism, not by universally denying "
        "a category the stem already names a preserved instance of."
    ),
    "stem_eliminable_distractor": _stem_eliminable_guidance,
}


# ── Normalize _GATE_GUIDANCE entries to a uniform callable contract ──
#
# Entries can be authored as either:
#   - a static string (most gates have fixed guidance regardless of
#     tier/reason),
#   - a callable `(tier, reason) -> str` (the stem_eliminable_distractor
#     entry branches on the [strategy=...] tag in `reason`),
#   - a callable `(tier) -> str` (legacy signature, kept working via
#     try/except in earlier versions of build_correction_prompt).
#
# To eliminate the try/except dispatch in build_correction_prompt and
# make the contract uniform, we post-process the dict at module load
# time: every entry becomes a callable with signature
# `(tier, reason) -> str`. Static strings are wrapped; older
# single-argument callables get adapted; new (tier, reason) callables
# pass through unchanged.

def _wrap_static_guidance(text: str):
    """Wrap a static guidance string in the (tier, reason) signature."""
    def _wrapped(tier=None, reason=""):
        return text
    return _wrapped


def _adapt_callable_guidance(fn):
    """Adapt a callable guidance entry to the (tier, reason) signature.

    Tries (tier, reason) first; on TypeError (older single-argument
    callable), falls back to (tier,). The exception is caught only at
    adapter-construction's first call — after that the adapter caches
    which signature works and uses it directly.
    """
    cached_signature = [None]  # 'wide' or 'narrow'

    def _adapted(tier=None, reason=""):
        if cached_signature[0] == "wide":
            return fn(tier, reason)
        if cached_signature[0] == "narrow":
            return fn(tier)
        try:
            result = fn(tier, reason)
            cached_signature[0] = "wide"
            return result
        except TypeError:
            cached_signature[0] = "narrow"
            return fn(tier)

    return _adapted


def _normalize_gate_guidance(d: dict) -> None:
    """In-place normalize all entries in `d` to callables."""
    for key, value in list(d.items()):
        if callable(value):
            d[key] = _adapt_callable_guidance(value)
        elif isinstance(value, str):
            d[key] = _wrap_static_guidance(value)
        # Else: unknown type — leave as-is (will fail at call site,
        # which is correct behavior for a malformed entry).


_normalize_gate_guidance(_GATE_GUIDANCE)


# ── Phase A3: detector-driven gen-time correction guidance ──────
# When the detector registry fires at generation time (with
# GOLIATH_DETECTORS_AT_GEN=1), the orchestrator emits failures with
# names like "detector:english_gap_scanner". This second guidance map
# is keyed on the signature carried in the failure reason, not the
# gate name — each english_gap_scanner signature (universal_quantifier,
# laterality, numeric_ratio) has different remediation. When a failure
# matches one of these prefixes, build_correction_prompt routes here.

_DETECTOR_SIGNATURE_GUIDANCE: dict[str, str] = {
    "universal_quantifier": (
        "A distractor uses a universal quantifier (all / every / always / "
        "throughout / never / no / none / entire / any) that the stem "
        "directly contradicts via a specific stated case. Rewrite this "
        "distractor to express a wrong-but-plausible specific claim, NOT a "
        "universal denial. The distractor should be wrong because of a "
        "concept misunderstanding, not because the universal contradicts a "
        "stem-stated counterexample."
    ),
    "laterality": (
        "A distractor asserts the OPPOSITE laterality from the stem "
        "(bilateral vs unilateral, left vs right, ipsilateral vs "
        "contralateral). The student rejects this by reading the stem "
        "alone — no concept knowledge required. Rewrite the distractor to "
        "be wrong via the underlying mechanism the question tests, not by "
        "inverting a stem-stated laterality."
    ),
    "numeric_ratio": (
        "The stem prints a specific ratio and a distractor prints a "
        "different ratio that the stem directly contradicts. Either drop "
        "the specific ratio from the stem (so the distractor's number "
        "becomes wrong via concept reasoning, not via lexical "
        "contradiction) or rewrite the distractor to express a different "
        "kind of wrongness — wrong concept, wrong direction, wrong scope."
    ),
    "stage_timing": (
        "A distractor claims a developmental-stage opposite to the stem "
        "(childhood vs adulthood, prepubertal vs postpubertal). Rewrite "
        "the distractor so the stage match is correct but a different "
        "concept-level claim is wrong."
    ),
    "schema_labeling": (
        "A distractor swaps which label attaches to which member of a "
        "paired-named-concept (IV/DV, agonist/antagonist, encoding/"
        "retrieval). This is a legitimate test of label recognition and "
        "is NOT english_gap if the stem genuinely tests labeling — but "
        "if you intended a content distinction, rewrite the distractor "
        "to test the underlying mechanism instead."
    ),
}


def _detector_signature_from_reason(reason: str) -> str | None:
    """Extract the signature name from a detector failure reason of the
    form 'universal_quantifier on letter B: <details>'."""
    if not reason:
        return None
    head = reason.split(" on letter ", 1)[0].strip()
    if head in _DETECTOR_SIGNATURE_GUIDANCE:
        return head
    return None


def build_correction_prompt(original_prompt, failures, tier=None):
    """Build a corrected user prompt with targeted feedback for retry.

    failures: list of (gate_name, failure_reason) tuples. Multiple
    content-level gate failures get bundled into a single retry so the
    LLM sees and addresses all of them at once. Single-failure callers
    can pass a one-element list.

    tier: optional 1-4 difficulty tier. Some gate guidance entries are
    callables that branch on tier (e.g. stem_eliminable_distractor uses
    a different fix-strategy at T4 evaluate-tier than at T1 remember-
    tier).

    Instead of blind retry with the same prompt, this appends specific
    guidance about what went wrong so the LLM can fix the exact issues.
    """
    if not failures:
        return original_prompt

    sections = []
    for gate_name, reason in failures:
        # A3: detector-driven failures carry "detector:<id>" gate name and
        # the signature in the reason field. Route to detector-signature
        # guidance when that prefix matches.
        if isinstance(gate_name, str) and gate_name.startswith("detector:"):
            sig = _detector_signature_from_reason(reason)
            guidance = (
                _DETECTOR_SIGNATURE_GUIDANCE.get(sig)
                if sig else None
            )
            if guidance is None:
                guidance = f"Fix this detector finding: {reason}"
            sections.append(
                f"### {gate_name}\n**Failure:** {reason}\n\n**Guidance:** {guidance}"
            )
            continue

        entry = _GATE_GUIDANCE.get(gate_name)
        # All entries are normalized to callables at module load time
        # via _normalize_gate_guidance. Missing entries fall back to a
        # generic "fix this" message.
        if entry is None:
            guidance = f"Fix this issue: {reason}"
        else:
            guidance = entry(tier, reason)
        sections.append(
            f"### {gate_name}\n**Failure:** {reason}\n\n**Guidance:** {guidance}"
        )

    correction_block = "\n\n".join(sections)
    plural = "s" if len(failures) > 1 else ""

    return f"""{original_prompt}

## CORRECTION{plural.upper()} REQUIRED (previous attempt failed validation)
The previous attempt failed {len(failures)} validation gate{plural}. Address ALL of the following:

{correction_block}

Generate a corrected question now. Return ONLY the JSON object."""
