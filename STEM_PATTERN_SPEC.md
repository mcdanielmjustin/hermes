# Stem Pattern Specification — 20 Patterns Across 4 Bloom's Tiers

Designed 2026-04-25. Replaces the flat 5-pattern system (all Apply+) with tier-aligned pools.

**Formula unchanged:** 4 tiers x 5 variants x 1,566 anchors = 31,320 total questions.
Each variant maps 1:1 to a pattern within its tier. No double-assignment.

---

## Asymmetric Bloom's Mixing

Each tier targets a **primary** Bloom's level (the design target for all patterns in that tier)
and permits a **secondary** level as a ceiling or floor. The LLM prompt optimizes for the
primary level; the secondary is an incidental byproduct of the MCQ format, not a co-equal target.

| Tier | Primary (design target) | Secondary (incidental) | Prompt instruction |
|------|------------------------|----------------------|-------------------|
| 1 | **Remember** | Understand as ceiling | "Generate at **recall** level" |
| 2 | **Understand** | Apply as ceiling | "Generate at **comprehension** level" |
| 3 | **Apply** | Analyze as ceiling | "Generate at **application** level" |
| 4 | **Evaluate** | Analyze as floor | "Generate at **evaluation** level" |

Why mix at all: Bloom's has 6 levels, we have 4 tiers. Pure isolation forces dropping 2 levels.
MCQs inherently involve comprehension (reading options), so pure-Remember is impossible.
The levels are cumulative — you can't apply without understanding, can't analyze without applying.

Why asymmetric, not 50/50: Equal weighting ("Remember/Understand") makes both levels design
targets, producing patterns that could sit in either tier (the paraphrase problem). Asymmetric
mixing gives each tier a clear identity: "this tier tests X, with Y as permitted overhead."

---

## Mapping Table

| Variant | Tier 1 (primary: Remember) | Tier 2 (primary: Understand) | Tier 3 (primary: Apply) | Tier 4 (primary: Evaluate) |
|---------|---------------------------|------------------------------|------------------------|---------------------------|
| v1 | direct_definition | comparison | clinical_vignette* | contrast_prompt* |
| v2 | concept_identification | example_recognition | scenario_completion* | best_answer* |
| v3 | fact_recognition | simple_application | error_identification* | subtle_error |
| v4 | true_false_which | paraphrase | case_analysis | competing_evidence |
| v5 | feature_listing | categorization | mechanism_application | integration |

\* = existing pattern, reassigned to its natural tier

---

## Cross-Cutting Constraints

### Stem & Option Length by Tier

| Tier | Stem Length | Option Length | Scenario Depth |
|------|-----------|--------------|----------------|
| 1 | 1-2 sentences | 5-25 words | None — purely conceptual |
| 2 | 1-3 sentences | 8-30 words | Brief (1-2 sentence scenario max) |
| 3 | 2-5 sentences | 10-35 words | Full clinical/professional scenario |
| 4 | 3-7 sentences | 15-45 words | Complex multi-detail scenario |

### Character Injection by Pattern

| Rule | Patterns |
|------|----------|
| **None** | direct_definition, concept_identification, fact_recognition, true_false_which, feature_listing, comparison, paraphrase, categorization |
| **Optional** | example_recognition, simple_application, scenario_completion, error_identification, contrast_prompt, best_answer, mechanism_application |
| **Mandatory** | clinical_vignette, case_analysis, subtle_error, competing_evidence, integration |

### Bloom's Enforcement by Tier

| Tier | Direction | Primary target | Rule |
|------|-----------|---------------|------|
| 1 | Prevent upward creep | Remember | If answering requires analyzing a scenario or applying to a case, the question is too complex. Strip the scenario and test the concept directly. |
| 2 | Prevent upward creep | Understand | If answering requires multi-step reasoning, integrating multiple concepts, or evaluating competing claims, the question is too complex. Simplify to a single conceptual step. |
| 3 | Prevent downward creep | Apply | If answerable by recalling a single definition regardless of scenario dressing, the question is too simple. Require genuine application or analysis. |
| 4 | Prevent downward creep | Evaluate | Must integrate at least two concepts. Must require evaluation, not just identification. Single-concept questions are rejected regardless of difficulty of content. |

### Misconception Type Manifestation by Tier

The 6 misconception types (`similar_name`, `similar_property`, `similar_store`, `opposite_direction`, `overgeneralization`, `partial_understanding`) describe WHY a student picks the wrong answer. The types are stable across tiers — the same 6 categories apply everywhere. What changes is HOW each type manifests at different cognitive levels.

**Tier 1–2 (Remember/Understand):** Misconceptions are knowledge-level failures. The student retrieves the wrong fact, confuses two terms, or recalls an incomplete definition. The distractor exploits a storage or retrieval error.

**Tier 3 (Apply):** Misconceptions manifest as misapplication. The student's knowledge might be correct in isolation but they apply it to the wrong case, predict the wrong outcome, or miss a qualifying condition. The distractor exploits a transfer error.

**Tier 4 (Evaluate):** Misconceptions manifest as reasoning failures. The student's factual knowledge may be complete, but they accept flawed logic, weigh evidence incorrectly, or fail to synthesize across concept areas. The distractor exploits an evaluation error.

| Misconception Type | Tier 1–2 (knowledge error) | Tier 3 (application error) | Tier 4 (evaluation error) |
|---|---|---|---|
| **similar_name** | Picks wrong term/definition because names sound alike | Picks wrong diagnosis/treatment because labels are confusable | Accepts reasoning that invokes a similarly-named but wrong principle |
| **similar_property** | Picks wrong concept because it shares a surface feature | Picks wrong intervention because two approaches share a feature | Evaluates based on a shared property and misses the discriminating detail |
| **similar_store** | Picks wrong concept from the same mental category | Picks wrong approach from the same therapeutic/methodological family | Evaluates using the wrong framework from the same family of frameworks |
| **opposite_direction** | Reverses which concept has which property | Predicts the wrong outcome by reversing the direction of effect | Weighs evidence incorrectly by reversing which factor supports which position |
| **overgeneralization** | Applies a rule beyond its defined scope | Applies a treatment/principle to a case where an exception applies | Accepts reasoning that extends a valid principle past its boundary conditions |
| **partial_understanding** | Knows part of the definition or feature set | Applies the concept correctly but misses a qualifying condition | Evaluates reasoning as sound because they understand 80% and miss the subtle gap |

The key shift at Tier 4: at lower tiers, distractors exploit what the student **doesn't know**. At Tier 4, distractors exploit what the student **can't do with what they know** — flawed weighing, incomplete synthesis, uncritical acceptance of mostly-correct reasoning.

---

## Tier 1 — Remember/Understand

### 1.1 direct_definition

**Cognitive operation:** Recall — retrieve the definition of a technical term.

**Stem archetype:**
> "Which of the following best defines [CONCEPT]?"

The student sees a concept name and selects its correct definition from four options.

**Generation instruction:**
Present a psychology term and ask the student to select its correct definition. The correct definition must be accurate but NOT verbatim from source material — paraphrase in your own words. Distractors must be real definitions of related concepts from the same subdomain, not fabricated nonsense.

**Structural constraints:**
- Stem is 1 sentence naming the concept
- All 4 options are definitional statements (not terms)
- Options are parallel in structure and similar in length (+/-30%)
- No "all of the above" or "none of the above"

**Anti-patterns:**
- Verbatim textbook definitions (student pattern-matches instead of recalling)
- Distractor definitions from completely different domains (too easy to eliminate)
- Concepts so obscure that only one option is plausible
- Defining words in the stem that leak the answer ("Which defines stimulus generalization, where a response spreads...")

**Domain examples:**
- PMET: "Which of the following best defines statistical power?"
- CPAT: "Which of the following best defines anosognosia?"
- PTHE: "Which of the following best defines unconditional positive regard?"
- SOCU: "Which of the following best defines the fundamental attribution error?"
- BPSY: "Which of the following best defines anterograde amnesia?"
- CASS: "Which of the following best defines predictive validity?"
- LDEV: "Which of the following best defines object permanence?"
- WDEV: "Which of the following best defines organizational citizenship behavior?"
- PETH: "Which of the following best defines informed consent?"

**Distractor design:** All 3 distractors should be accurate definitions of other concepts — typically the most confusable neighbors. The student fails by misremembering which definition belongs to which term.

---

### 1.2 concept_identification

**Cognitive operation:** Recall — given a description, name the correct concept.

**Stem archetype:**
> "The term for [DESCRIPTION] is:"
> "[DESCRIPTION] is known as:"

Reverse of direct_definition. The student reads a description and selects the matching term.

**Generation instruction:**
Present a description or definition of a psychology concept and ask the student to identify the correct term. The description must be clear and unambiguous but must NOT use the term itself or obvious derivatives. All four options should be real psychology terms from the same domain area.

**Structural constraints:**
- Stem is 1-2 sentences describing a concept without naming it
- All 4 options are terms/names (not definitions)
- Options should be from the same subdomain or closely related areas
- The description must be sufficient to uniquely identify the concept

**Anti-patterns:**
- Description so specific that only one option could possibly fit (too easy)
- Description so vague that multiple options could legitimately fit (ambiguous)
- Including the answer term or a derivative in the stem
- Distractors from completely unrelated fields

**Domain examples:**
- LDEV: "The developmental process by which an infant forms a deep emotional bond with a primary caregiver, observable through proximity-seeking and separation distress, is known as:"
- PETH: "The ethical obligation requiring psychologists to practice only within the boundaries of their education, training, and supervised experience is referred to as:"
- WDEV: "The tendency for a single positive trait to influence the overall evaluation of an employee's performance is called:"
- BPSY: "The brain region at the medial temporal lobe that is critical for consolidating new explicit memories is the:"
- CASS: "A type of test reliability estimated by administering two equivalent forms of a test to the same group is known as:"

**Distractor design:** Options should be terms that a student might confuse with the correct answer — similar-sounding names, concepts from the same theory, or terms from the same chapter. Avoid random unrelated terms.

---

### 1.3 fact_recognition

**Cognitive operation:** Remember/Understand — recognize a correct factual claim about a concept.

**Stem archetype:**
> "According to [AUTHORITY], [CONCEPT]:"
> "[THEORY] states that:"
> "The DSM-5-TR criteria for [DISORDER] include:"

The student evaluates four factual statements and identifies the one that is accurate.

**Generation instruction:**
Ask the student to identify the correct factual statement about a specific concept, theory, or finding. Ground the question in an authoritative source (e.g., "According to Piaget," "The DSM-5-TR criteria for X include," "Bandura's social learning theory holds that"). The correct answer states an accurate fact. Distractors state plausible but incorrect facts — commonly misremembered details, switched numbers, confused attributions, reversed relationships.

**Structural constraints:**
- Stem names a specific authority, theory, or source
- All 4 options are factual claims (not definitions or terms)
- Options are parallel in grammar and length
- Distractors should be plausible misstatements, not absurd

**Anti-patterns:**
- Absurd or obviously wrong distractors
- Trivia with no clinical or conceptual significance (dates, page numbers)
- Wrong attribution in the stem itself (the authority named must be accurately connected to the topic)
- Absolute language ("always," "never") unless testing a genuine absolute

**Domain examples:**
- PMET: "According to Cohen's conventions, a medium effect size for Pearson's r is:"
- CPAT: "The DSM-5-TR requires the following duration criterion for a diagnosis of generalized anxiety disorder:"
- SOCU: "Milgram's obedience experiments found that the percentage of participants who administered the maximum shock was approximately:"
- BPSY: "The capacity of short-term memory, as proposed by Miller (1956), is approximately:"
- PETH: "Standard 4.05 of the APA Ethics Code addresses:"

**Distractor design:** Wrong facts should be the kind of errors students actually make — swapping the numbers for two related statistics, attributing a finding to the wrong researcher, reversing a direction of effect. Not random inventions.

---

### 1.4 true_false_which

**Cognitive operation:** Understand — discriminate between correct and incorrect statements about a topic.

**Stem archetype:**
> "Which statement about [CONCEPT] is correct?"
> "Which of the following is TRUE regarding [TOPIC]?"
> "Which statement about [CONCEPT] is INCORRECT?"

The student reads four statements about the same topic and determines which is accurate (or inaccurate).

**Generation instruction:**
Present four statements about the same concept and ask the student to identify which is correct (or incorrect). All statements should be about the same topic — this tests precision and depth of understanding, not breadth. Distractors should contain common misunderstandings: reversed direction, overgeneralization, confusion with a related concept, or partially correct claims.

**Structural constraints:**
- Stem names a specific concept or topic
- All 4 options are declarative statements about that SAME concept
- Can use positive ("Which is correct?") or negative ("Which is INCORRECT?") framing
- If negative framing, NOT/INCORRECT must be in ALL CAPS
- Options are parallel in structure

**Anti-patterns:**
- Mixing statements from unrelated topics (student just picks the on-topic one — too easy)
- Mini-paragraph options (keep each statement concise)
- One option obviously longer or more detailed than others (length bias)
- Double negatives

**Domain examples:**
- PTHE: "Which statement about systematic desensitization is correct?"
- LDEV: "Which of the following is TRUE regarding Erikson's stage of generativity vs. stagnation?"
- WDEV: "Which statement about Herzberg's two-factor theory is INCORRECT?"
- CASS: "Which of the following accurately describes the MMPI-3?"
- PMET: "Which statement about Type I error is correct?"

**Distractor design:** Each wrong statement should contain a specific, identifiable error — not be entirely wrong. The best distractors are 80% true with one crucial detail wrong (reversed direction, wrong scope, attributed to wrong mechanism).

---

### 1.5 feature_listing

**Cognitive operation:** Remember — identify features that do or do not belong to a concept.

**Stem archetype:**
> "Which of the following is NOT a characteristic of [CONCEPT]?"
> "All of the following are features of [CONCEPT] EXCEPT:"

The student identifies the outlier — the one feature that belongs to a different concept.

**Generation instruction:**
Test whether the student knows the complete feature set of a concept. Three options should be genuine features/characteristics of the target concept. One option (the correct answer) should be a feature of a related but different concept. The outlier must be plausible — it should SOUND like it could belong but actually describes a different, closely related concept.

**Structural constraints:**
- Stem names a specific concept and uses EXCEPT or NOT (capitalized)
- 3 options are true features of the named concept
- 1 option (correct answer) is a feature of a related concept
- The outlier should come from the nearest conceptual neighbor
- Options are all features/characteristics (parallel structure)

**Anti-patterns:**
- Outlier from a completely different domain (too easy to spot)
- Features so obscure that the student can only guess
- Outlier that is the longest or shortest option (format bias)
- "All of the above" as an option

**Domain examples:**
- CPAT: "All of the following are diagnostic features of PTSD EXCEPT:"
- BPSY: "Which of the following is NOT a function of the frontal lobe?"
- PMET: "All of the following are characteristics of a quasi-experimental design EXCEPT:"
- PTHE: "Which of the following is NOT a core condition in person-centered therapy?"
- CASS: "All of the following are subtests of the WAIS-IV EXCEPT:"

**Distractor design:** The 3 true features are straightforward (student should recognize them). The outlier is a feature of the closest neighboring concept — e.g., a PTSD question where the outlier is a feature of acute stress disorder, or a frontal lobe question where the outlier is a parietal lobe function.

---

## Tier 2 — Understand/Apply

### 2.1 comparison

**Cognitive operation:** Understand — articulate the key difference between two related concepts.

**Stem archetype:**
> "How does [CONCEPT A] differ from [CONCEPT B]?"
> "What is the key distinction between [A] and [B]?"

The student holds two related concepts in mind and identifies the genuine discriminating feature.

**Generation instruction:**
Ask the student to identify the key distinction between two genuinely confusable concepts. Both concepts should be from the same subdomain and share surface-level similarities. The correct answer identifies the actual discriminating feature. Distractors state differences that are wrong (reversed), irrelevant (true but not THE key distinction), or describe a difference involving a third concept.

**Structural constraints:**
- Stem names two specific concepts being compared
- Both concepts must be real and genuinely confusable
- All 4 options are statements about how the concepts differ
- The correct answer identifies the genuine discriminating feature
- Options are specific and testable (not vague)

**Anti-patterns:**
- Comparing concepts from different domains (too easy — no genuine confusion)
- Presenting both definitions side by side in the options (student matches instead of knowing)
- Vague comparisons ("they differ in their approach") — must be specific
- Comparing concepts that aren't actually confusable (no pedagogical value)

**Domain examples:**
- PMET: "How does internal validity differ from external validity?"
- CPAT: "What is the key distinction between obsessions and compulsions?"
- LDEV: "How does assimilation differ from accommodation in Piaget's theory?"
- SOCU: "What distinguishes the actor-observer bias from the fundamental attribution error?"
- BPSY: "How does Broca's aphasia differ from Wernicke's aphasia?"
- WDEV: "What is the key distinction between job enlargement and job enrichment?"

**Distractor design:** One distractor should reverse the direction of the true difference (swap which concept has which feature). One should state a true fact that isn't the key distinction. One should state a difference that actually applies to a third concept.

---

### 2.2 example_recognition

**Cognitive operation:** Understand/Apply — identify which concrete example illustrates a named concept.

**Stem archetype:**
> "Which of the following best illustrates [CONCEPT]?"
> "Which scenario is an example of [CONCEPT]?"

The student is given a concept and selects the scenario that demonstrates it.

**Generation instruction:**
Name a concept and present four brief scenarios (1-2 sentences each). One correctly illustrates the named concept. The other three illustrate related but different concepts — each distractor scenario should be a valid example of a real concept, not nonsense. The student must understand the concept well enough to recognize it in action.

**Structural constraints:**
- Stem names the concept being illustrated
- All 4 options are brief scenarios (1-2 sentences, max 40 words each)
- Each option illustrates a real concept (no fabricated situations)
- Scenarios should be realistic and domain-appropriate
- The correct scenario must unambiguously illustrate the named concept

**Anti-patterns:**
- Correct scenario is a thinly disguised restatement of the definition
- Scenarios that are ambiguous between two concepts
- Unrealistically extreme or cartoonish scenarios
- Reusing the concept name within the scenario text

**Domain examples:**
- PMET: "Which of the following best illustrates negative reinforcement?"
- PTHE: "Which scenario is an example of cognitive restructuring?"
- SOCU: "Which of the following best illustrates the bystander effect?"
- WDEV: "Which scenario demonstrates transformational leadership?"
- PETH: "Which of the following illustrates a dual relationship?"
- LDEV: "Which scenario best illustrates scaffolding?"

**Distractor design:** Each distractor scenario illustrates a concept from the same subdomain. The student who confuses negative reinforcement with punishment will pick the punishment scenario. The distractor concepts should be the ones most commonly confused with the target.

---

### 2.3 simple_application

**Cognitive operation:** Apply — identify the concept operating in a given brief scenario.

**Stem archetype:**
> "[Brief scenario]. This is an example of:"
> "[Brief scenario]. [Person]'s behavior demonstrates:"

The reverse of example_recognition. The student is given a scenario and identifies the concept.

**Generation instruction:**
Present a brief, concrete scenario (1-3 sentences) and ask the student to identify which psychological concept, principle, or phenomenon is being demonstrated. The scenario should clearly illustrate one concept without being a verbatim restatement of its definition. Use everyday language in the scenario, not technical jargon. Distractors should be concepts that share surface features with the scenario.

**Structural constraints:**
- Stem presents a scenario (1-3 sentences) followed by a question
- All 4 options are concept names or brief concept descriptions
- The scenario must clearly map to one concept (no ambiguity)
- Scenario uses everyday language, not technical jargon
- Distractors from the same conceptual neighborhood

**Anti-patterns:**
- Scenarios so long they become Tier 3 (keep under 3 sentences)
- Technical terms in the scenario that give away the answer
- Scenario that restates the definition instead of showing a real-world instance
- Distractors from completely different domains

**Domain examples:**
- PMET: "A child who was bitten by a German Shepherd now shows fear around all large dog breeds, but not small dogs. This is an example of:"
- LDEV: "A 4-year-old insists that a tall, narrow glass has 'more juice' than a short, wide glass, even after watching the juice poured from one to the other. This demonstrates:"
- BPSY: "After studying vocabulary right before sleep, Maria performs better the next morning than her classmate who studied at the same time but stayed awake afterward. This is best explained by:"
- CASS: "A manager consistently rates an employee high across all dimensions after being impressed by an initial presentation. This is an example of:"
- SOCU: "After buying an expensive car he can barely afford, David starts reading only positive reviews and avoids Consumer Reports. This demonstrates:"

**Distractor design:** Options should be concepts that the scenario's surface features could suggest. A negative reinforcement scenario should have "punishment," "extinction," and "negative punishment" as distractors — not "groupthink" or "self-actualization."

---

### 2.4 paraphrase

**Cognitive operation:** Understand — recognize a concept restated in genuinely different words.

**Stem archetype:**
> "Which statement best restates [CONCEPT/PRINCIPLE]?"
> "Another way to express [THEORY'S CLAIM] is:"

The student must recognize the same idea expressed in novel language.

**Generation instruction:**
Name a specific concept, principle, or theoretical claim and ask the student to select its most accurate paraphrase from four options. The correct option must convey the same meaning using genuinely different wording — not just synonym substitution. Distractors should be plausible paraphrases that subtly change the meaning: reversing a relationship, overstating a claim, narrowing the scope, or conflating with a related concept.

**Structural constraints:**
- Stem names a specific concept or quotes a principle
- All 4 options are restatements in different words
- The correct paraphrase preserves the full meaning
- Distractors subtly distort the meaning
- Options are similar in length and formality

**Anti-patterns:**
- Simple synonym swap (too easy — "happy" to "glad")
- Distractors that are obviously wrong (meaning completely reversed)
- Paraphrasing by just making it longer or shorter without changing structure
- Testing principles so obscure the student has no basis to paraphrase

**Domain examples:**
- SOCU: "Which statement best restates the concept of cognitive dissonance?"
- PTHE: "Another way to express Rogers' concept of 'conditions of worth' is:"
- PMET: "Which of the following best restates what it means to reject the null hypothesis?"
- BPSY: "Which statement best restates the principle of long-term potentiation?"
- WDEV: "Another way to express Vroom's expectancy theory is:"

**Distractor design:** Each distractor should be a plausible paraphrase that contains one subtle distortion. Example: for cognitive dissonance, a distractor might say "People feel uncomfortable when they notice others' beliefs conflict with their actions" (shifts from self to others). The distortions should be the kind that reveal whether the student truly understands vs. has a fuzzy approximation.

---

### 2.5 categorization

**Cognitive operation:** Apply — classify an item into its correct category within a real taxonomy.

**Stem archetype:**
> "Which of the following would be classified as [CATEGORY]?" (Category -> Item)
> "[ITEM] belongs to which of the following categories?" (Item -> Category)

The student applies a classification system to sort correctly.

**Generation instruction:**
Test the student's ability to correctly classify within a psychological taxonomy, diagnostic system, or theoretical framework. Use one of two formats: (A) present a category, ask which item belongs; (B) present an item, ask which category it belongs to. Categories must be from real systems (DSM diagnostic groups, research design types, test classifications, ethical standards, substance schedules, leadership theories, etc.). Test categories that are commonly confused.

**Structural constraints:**
- One of two formats: Category->Item or Item->Category
- Categories from a real classification system
- All 4 options at the same taxonomic level
- The classification must be unambiguous
- Test commonly confused categories

**Anti-patterns:**
- Category membership obvious from the name alone
- Mixed taxonomic levels (a specific test alongside "projective tests" as a category)
- Categories so broad that multiple options could be correct
- Trivia classifications with no clinical or conceptual significance

**Domain examples:**
- CPAT: "Which of the following disorders is classified as an anxiety disorder in the DSM-5-TR?"
- PMET: "A study that measures variables at a single point in time with no manipulation is classified as:"
- PETH: "A psychologist accepting a gift from a client would be addressed under which APA Ethics Standard?"
- CASS: "The Rorschach Inkblot Test would be classified as which type of assessment?"
- WDEV: "Maslow's hierarchy of needs is classified as which type of motivation theory?"
- BPSY: "Serotonin would be classified as which type of neurotransmitter?"

**Distractor design:** Options should be adjacent categories from the same taxonomy. For a DSM question, all options should be DSM diagnostic groups. For a research design question, all options should be design types. The student fails by misclassifying — putting the item in the wrong bin.

---

## Tier 3 — Apply/Analyze

### 3.1 clinical_vignette (existing — reassigned from flat list)

**Cognitive operation:** Apply/Analyze — apply clinical knowledge to identify a diagnosis, treatment, or principle in a realistic case.

**Stem archetype:**
> "[Named clinician] is working with [named client] who presents with [specific symptoms/behavior in a specific setting]. [Additional clinical detail]. Which of the following [diagnoses/treatments/explanations] is most appropriate?"

Multi-step: read the case, identify relevant features, match to knowledge, select the best answer.

**Generation instruction:**
Write a clinical vignette featuring a named professional and named client in a specific setting. Include enough clinical detail to support the correct answer AND to make at least one distractor plausible. The student must apply knowledge to a novel case, not just recognize a textbook description. Vary demographics, settings, and presenting concerns across questions.

**Structural constraints:**
- MANDATORY character injection (clinician + client names from name pool)
- Stem is 3-5 sentences with specific clinical details
- Setting must be named (private practice, school, hospital, community center)
- At least one detail must specifically support the correct answer
- At least one detail must make a distractor plausible
- Options are diagnoses, treatments, explanations, or appropriate next steps

**Anti-patterns (enforced in existing Bloom's rules):**
- Scenario that simply restates the correct answer's definition in narrative form
- Asking the student to classify/label using a single definition
- Case so clear-cut that only one option is remotely plausible
- Named-effect shortcut: matching scenario description to a single label is not sufficient at this tier

**Character injection:** Mandatory
**Stem length:** 3-5 sentences
**Option length:** 10-30 words each

---

### 3.2 scenario_completion (existing — reassigned from flat list)

**Cognitive operation:** Apply — predict the next step, outcome, or appropriate action in a professional scenario.

**Stem archetype:**
> "A [professional] is [doing X in context Y]. Based on [principle/finding], what would be the most appropriate next step?"
> "[Scenario with specific details]. What would most likely happen next?"

Forward-reasoning from scenario details and principles to actions or outcomes.

**Generation instruction:**
Present a professional (researcher, clinician, supervisor) in a specific situation. The student must predict what should happen next, what the outcome would be, or what the appropriate action is, based on established principles. Scenario details should be specific enough that general knowledge isn't sufficient — the details must drive the answer.

**Structural constraints:**
- Stem presents a specific professional scenario (2-4 sentences)
- Question asks about next steps, outcomes, or appropriate actions
- The correct answer follows logically from an established principle
- Distractors represent common misapplications or alternative approaches
- Scenario details must discriminate between options

**Anti-patterns (enforced in existing Bloom's rules):**
- "Which of these is the [term]?" — that's identification, not completion
- Requiring only recall to determine the next step
- Scenario-irrelevant details that don't affect the answer

**Character injection:** Optional (recommended for clinical scenarios)
**Stem length:** 2-4 sentences
**Option length:** 10-25 words each

---

### 3.3 error_identification (existing — reassigned from flat list)

**Cognitive operation:** Analyze — detect an error or flaw in professional reasoning or claims.

**Stem archetype:**
> "[Professional context with multiple claims]. Which of the following statements contains an error?"
> "[Applied scenario]. Identify the claim that is flawed."

The student analyzes each claim for accuracy within an applied context.

**Generation instruction:**
Write a question where one statement among four contains a substantive error. The error should be a genuine misapplication, wrong attribution, reversed relationship, or overgeneralization — not a trivial factual mistake. Anchor ALL claims within a specific scenario, case, or applied context. The erroneous claim must involve a misapplication or flawed reasoning step with real consequences.

**Structural constraints:**
- All claims anchored in a specific scenario or applied context
- 3 statements correct, 1 contains a substantive error
- The error must be consequential (would lead to wrong decisions)
- At least one correct statement must be plausible as an error (tests careful analysis)
- Error detectable through analysis, not just factual recall

**Anti-patterns (enforced in existing Bloom's rules):**
- NEVER present 4 abstract definitional statements and ask which is wrong — that is recall-level
- The erroneous claim must involve a misapplication or flawed reasoning step
- At least one correct claim must be plausible enough as an error that the student must evaluate interactions between concepts to rule it out

**Character injection:** Optional
**Stem length:** 2-4 sentences of context + 4 claims
**Option length:** 15-35 words each

---

### 3.4 case_analysis (new)

**Cognitive operation:** Analyze — identify the underlying mechanism or causal factor in a presented case.

**Stem archetype:**
> "[Detailed case]. What is the most likely explanation for [observed behavior/outcome]?"
> "[Case description]. Which underlying mechanism best accounts for [specific observation]?"

Different from clinical_vignette: clinical_vignette asks "WHAT does this person have / WHAT should you do?" Case_analysis asks "WHY is this happening?" The student must analyze mechanism, not label diagnosis.

**Generation instruction:**
Present a detailed scenario and ask the student to identify the underlying mechanism or causal explanation for a specific observation within that case. The case should contain details relevant to multiple possible mechanisms. The correct answer identifies the mechanism that best explains the SPECIFIC observation cited in the question. Distractors should be real mechanisms that plausibly could apply but don't fully account for the cited observation.

**Structural constraints:**
- Stem presents a detailed case (3-5 sentences) with a specific observation to explain
- Question asks about mechanism, cause, or explanation — NOT diagnosis or label
- At least two plausible mechanisms supported by some case details
- The cited observation must discriminate between mechanisms
- Options are mechanisms, principles, or causal explanations

**Anti-patterns:**
- "What does this person have?" (that's clinical_vignette)
- "What should you do?" (that's scenario_completion)
- Accepting a single-concept label as an answer — the answer must explain WHY
- Mechanism obvious from a single detail — require synthesizing multiple observations

**Domain examples:**
- BPSY: "[Patient with specific memory pattern after injury]. What is the most likely explanation for the selective preservation of procedural but not episodic memory?"
- PTHE: "[Client shows resistance pattern]. Which therapeutic mechanism best accounts for the client's increased symptom reporting after two sessions of cognitive restructuring?"
- SOCU: "[Group decision scenario]. Which underlying process best explains why the group arrived at a riskier decision than any individual member initially endorsed?"
- PMET: "[Research findings pattern]. Which methodological factor most likely accounts for the discrepancy between the experimental and control group results?"

**Character injection:** Mandatory
**Stem length:** 3-5 sentences
**Option length:** 15-35 words each

---

### 3.5 mechanism_application (new)

**Cognitive operation:** Apply/Analyze — apply a named principle to a novel situation and predict the outcome.

**Stem archetype:**
> "Based on [PRINCIPLE/THEORY], what would be the expected outcome if [NOVEL SITUATION]?"
> "If [PRINCIPLE] applies to [NEW CONTEXT], which result is most likely?"

Forward-reasoning: the student is told which principle applies and must predict what would happen in a new situation. Different from scenario_completion ("what should you do?") — this asks "what would happen?"

**Generation instruction:**
Name a specific principle, theory, or research finding and ask the student to predict what would happen when it's applied to a genuinely novel situation. The situation should NOT be a textbook example of the principle — it should require reasoning about how the principle operates in new circumstances. Distractors represent predictions based on different or misapplied principles.

**Structural constraints:**
- Stem explicitly names the principle/theory
- Stem describes a novel situation not found in standard coursework
- Question asks about predicted outcomes or expected results
- The correct answer follows logically from the named principle
- Distractors are predictions that follow from different principles

**Anti-patterns:**
- Textbook examples of the principle (student recognizes instead of reasons)
- Novel situation so far removed that the principle can't reasonably apply
- Naming the principle then describing its textbook scenario
- "What should you do?" — ask "what would happen?"

**Domain examples:**
- PMET: "Based on the principles of operant extinction, what would be the expected behavioral pattern if a slot machine that previously paid out on a variable-ratio schedule suddenly stopped all payouts?"
- SOCU: "According to social identity theory, what would most likely happen when members of a high-status group learn their performance ranking will be published alongside a rising lower-status outgroup?"
- LDEV: "Based on Vygotsky's zone of proximal development, what would be predicted if a child who solves 20-piece puzzles independently is given a 50-piece puzzle with no scaffolding?"
- BPSY: "Based on the encoding specificity principle, what would be expected if a student studies vocabulary in a noisy cafeteria but takes the test in a silent exam hall?"

**Character injection:** Optional
**Stem length:** 2-4 sentences
**Option length:** 15-30 words each

---

## Tier 4 — Analyze/Evaluate

### 4.1 contrast_prompt (existing — reassigned from flat list)

**Cognitive operation:** Analyze/Evaluate — distinguish two significantly overlapping concepts within a specific applied context.

**Stem archetype:**
> "[Case where two concepts could plausibly apply]. Which concept BEST applies to this specific situation, and what distinguishes it from [the alternative]?"

At Tier 4, this is NOT "define the difference between X and Y" (that's Tier 2 comparison). Here, the student analyzes a specific case where both concepts seem applicable and determines which one fits based on subtle discriminating features in the scenario.

**Generation instruction:**
Present a case or scenario where two significantly overlapping concepts both seem applicable. The student must analyze specific details to determine which concept is the better fit. The distinguishing feature should be subtle and context-dependent — not just a definitional difference. Distractors should include the "wrong" concept from the contrasted pair plus two other concepts sharing surface features.

**Structural constraints (from existing Bloom's enforcement):**
- MUST present a specific case or applied context
- Both concepts genuinely share features making discrimination hard
- Scenario contains specific details favoring one concept over the other
- The student must analyze the scenario to determine the answer
- At least one distractor should be the other concept from the contrasted pair

**Anti-patterns (from existing Bloom's enforcement):**
- Do NOT ask "what distinguishes X from Y?" as a standalone definition question
- Present a specific case and ask which concept applies TO THAT CASE and why
- Context-free contrast is Tier 2, not Tier 4

**Distractor design (misconception type manifestation at Tier 4):**
- **similar_property** — Student picks the overlapping concept because both share the feature prominent in the case. They see the shared property and stop analyzing the discriminating details.
- **partial_understanding** — Student knows one concept deeply but not the other. Defaults to the familiar concept rather than evaluating which fits the specific case.
- **overgeneralization** — Student learned "Concept A applies when X is present" and applies it here because X is present, ignoring contextual details that shift the answer to Concept B.
- **opposite_direction** — Student correctly identifies the two concepts in play but reverses which one fits this case vs. the other.

**Character injection:** Optional (recommended)
**Stem length:** 3-5 sentences
**Option length:** 15-35 words each

---

### 4.2 best_answer (existing — reassigned from flat list)

**Cognitive operation:** Evaluate — judge which option is MOST correct when all contain genuine truth.

**Stem archetype:**
> "Which BEST describes [complex situation]?"
> "All of the following are valid considerations. Which is the MOST important in this context?"

The student evaluates relative correctness, not just eliminates wrong answers.

**Generation instruction:**
Write a question where all four options contain genuine truth or valid reasoning. The correct answer is the MOST accurate, complete, or contextually appropriate — not the only correct one. The specific context in the stem must be what makes one answer "best." In a different context, a different option might be best.

**Structural constraints (from existing Bloom's enforcement):**
- All options partially valid or defensible
- The word "BEST" must appear in the stem (or "MOST appropriate/important")
- Specific context drives which answer is best
- Changing the context should plausibly change the best answer
- Options are substantively different approaches, not variations on one idea

**Anti-patterns (from existing Bloom's enforcement):**
- Three obviously wrong options with one clearly right (defeats the purpose)
- The answer would be the same regardless of context (not evaluation-level)
- Options that are all wrong except one (standard MCQ, not best-answer)

**Distractor design (misconception type manifestation at Tier 4):**
- **partial_understanding** — Student picks an option that's genuinely true but not the MOST correct. Their understanding is deep enough to recognize truth but not deep enough to evaluate relative correctness across options.
- **overgeneralization** — Student picks the option stating the broadest generalization rather than the one most calibrated to the specific context. Confuses "generally true" with "best answer here."
- **similar_property** — Two options both contain a true claim about a shared property. Student can't distinguish which is more complete or more applicable to this specific scenario.

**Character injection:** Optional
**Stem length:** 3-5 sentences
**Option length:** 20-40 words each

---

### 4.3 subtle_error (new)

**Cognitive operation:** Evaluate — detect a nuanced flaw in mostly-correct expert reasoning.

**Stem archetype:**
> "A [senior professional] states: '[paragraph of mostly-correct reasoning with one subtle error].' What, if anything, is problematic about this reasoning?"

Significantly harder than error_identification (Tier 3). The reasoning is sophisticated and mostly correct. The error is a subtle overextension, conflation, or logical flaw that would fool someone with surface-level knowledge.

**Generation instruction:**
Present a paragraph of professional reasoning (3-5 sentences) that is mostly correct but contains one subtle error. The error should be the kind an advanced student or practitioner might make — an overextension of a valid principle, a subtle conflation of two related mechanisms, or a logically valid-sounding conclusion that doesn't follow from the premises. The student must understand the concepts deeply enough to spot where expert reasoning goes wrong.

**Structural constraints:**
- Stem presents 3-5 sentences of professional reasoning from a named professional
- The reasoning must be mostly correct (at least 80% of claims are accurate)
- The error must be substantive (would lead to wrong conclusions or decisions)
- One option MUST be "The reasoning is sound; there is no error" (tests detection ability)
- Other distractors identify non-existent errors (claiming something is wrong that's actually right)
- The error requires evaluation to detect, not factual recall

**Anti-patterns:**
- Simple factual mistakes (wrong name, wrong year) — that's recall
- Error obvious to anyone who's read the chapter — requires deep understanding to spot
- Surrounding reasoning is also wrong (changes pattern to error_identification)
- Jargon-heavy reasoning that obscures rather than tests — language should be clear, logic should be subtle

**Domain examples:**
- PTHE: "A supervisor explains: 'Motivational interviewing works for substance use because it leverages cognitive dissonance — when clients hear themselves argue for change, the inconsistency between behavior and stated values creates discomfort that motivates action. Reflective listening amplifies this by mirroring change talk.' What, if anything, is problematic?"
  [The subtle error: MI uses the concept of discrepancy (from MI theory), not cognitive dissonance (from Festinger). They're related but theoretically distinct — MI's model specifically avoids creating dissonance and instead develops discrepancy through guided exploration.]

- BPSY: "A neuropsychologist notes: 'The patient's intact procedural learning despite severe anterograde amnesia is expected, since procedural memory is mediated by the basal ganglia and cerebellum, not the hippocampus. This dissociation confirms bilateral hippocampal damage, as unilateral damage would still allow some explicit memory formation through the contralateral structure.' What, if anything, is problematic?"
  [The subtle error: The second claim — unilateral hippocampal damage allowing explicit memory through the contralateral structure — overstates the case. Unilateral hippocampal damage typically does impair explicit memory, though less severely than bilateral. The intact contralateral hippocampus provides partial compensation, not normal function.]

- PETH: "A psychologist explains to a trainee: 'When a client reveals during therapy that they are planning to harm a specific person, Tarasoff establishes our duty to protect that individual. This duty overrides confidentiality under Standard 4.05, which permits disclosure when mandated by law. Since Tarasoff is federal law, it applies uniformly across all states.' What, if anything, is problematic?"
  [The subtle error: Tarasoff is a California Supreme Court ruling, not federal law. Duty-to-warn/protect laws vary significantly by state.]

**Distractor design (misconception type manifestation at Tier 4):**
- **partial_understanding** — Student doesn't know the concept deeply enough to spot the subtle overextension. They verify the individual facts (which are correct) but can't evaluate the logical connection between them.
- **overgeneralization** — The error IS an overgeneralization of a valid principle. The student knows the general rule but not its boundary conditions, so the flawed extension looks correct.
- **similar_name** — The reasoning invokes a principle by name that sounds correct but is actually a similarly-named but distinct principle. Student accepts the name match without checking the mechanism.
- **opposite_direction** — The reasoning reverses a causal direction or temporal sequence. Student knows the relationship exists but doesn't catch that the arrow points the wrong way.

**Character injection:** Mandatory (the reasoning must come from a named professional)
**Stem length:** 4-7 sentences (including quoted reasoning)
**Option length:** 20-40 words each

---

### 4.4 competing_evidence (new)

**Cognitive operation:** Evaluate — weigh two defensible positions and determine which is better supported in a specific context.

**Stem archetype:**
> "Both [Position A] and [Position B] have empirical support. Given [specific contextual details], which position is better supported in this case?"

The student evaluates competing claims that are BOTH legitimate. The context provides the tiebreaker. Neither position is simply wrong — both are right in different contexts.

**Generation instruction:**
Present two genuinely defensible positions on a clinical, theoretical, or methodological issue. Both must have real empirical support — do not create a straw man. Then provide specific contextual details (patient characteristics, setting constraints, research parameters, ethical considerations) that make one position more appropriate. The student must evaluate which is better supported GIVEN THAT SPECIFIC CONTEXT.

**Structural constraints:**
- Stem presents two named positions with brief evidence summaries
- Specific context provided that favors one position
- Correct answer identifies the better-supported position AND explains why the context matters
- One distractor is the other position without context-sensitive reasoning
- Other distractors offer flawed reasoning about why one position is better
- Both positions must be genuinely defensible (no straw man)

**Anti-patterns:**
- Straw man (one position is obviously weaker) — becomes Apply-level
- Context is irrelevant (answer would be the same regardless) — not evaluation
- False dichotomy where a third option is clearly better
- Requiring knowledge of specific study citations — test reasoning about evidence, not bibliography

**Domain examples:**
- PTHE: "Both CBT and psychodynamic therapy have demonstrated efficacy for depression in RCTs. A 22-year-old presents with moderate depression that began after her father's sudden death 6 months ago. She reports strong intellectual curiosity, no prior therapy, and describes her symptoms primarily in terms of 'not understanding why I can't move on.' Given this presentation, which approach is better supported?"
- PMET: "Both within-subjects and between-subjects designs could address this research question. The study involves a taste-perception test, a small sample (N=24), and concern about practice effects. Which design is better supported for this specific protocol?"
- PETH: "Both reporting the colleague's impairment to the licensing board and first attempting informal resolution are ethically defensible courses of action. Given that the colleague is a close friend, the impairment has not yet resulted in documented client harm, and the psychologist has not yet spoken directly with the colleague, which course of action is most consistent with the APA Ethics Code?"

**Distractor design (misconception type manifestation at Tier 4):**
- **overgeneralization** — Student picks the position that's generally correct without weighing the specific contextual factors that favor the alternative. Confuses "true in most cases" with "true in this case."
- **partial_understanding** — Student understands one position deeply but not the other. Defaults to the position they know rather than genuinely evaluating both against the context.
- **opposite_direction** — Student reverses which contextual factor supports which position. Correctly identifies the relevant factors but misassigns their directional weight.
- **similar_property** — Both positions share a surface-level rationale. Student can't differentiate because they focus on the shared property instead of the contextual discriminator.

**Character injection:** Recommended
**Stem length:** 4-6 sentences
**Option length:** 25-45 words each

---

### 4.5 integration (new)

**Cognitive operation:** Analyze/Evaluate — synthesize knowledge from multiple concept areas to reach a conclusion that neither concept alone supports.

**Stem archetype:**
> "[Complex case involving multiple concept areas]. Considering both [Area A finding] and [Area B principle], which integrated conclusion is best supported?"

The highest cognitive demand in the system. The student must hold multiple frameworks in mind and synthesize them. Neither concept alone is sufficient — the correct answer requires genuine integration.

**Generation instruction:**
Write a question requiring integration of knowledge from two or more distinct concept areas. The case should involve observations relevant to multiple frameworks. Neither concept alone should be sufficient to answer — the student must synthesize. The correct answer represents an integrated conclusion accounting for all relevant information. Distractors represent conclusions based on only one concept (incomplete integration) or incorrect synthesis.

**Structural constraints:**
- Stem explicitly references at least two distinct concept areas
- Scenario contains information relevant to both areas
- Correct answer requires genuine synthesis (both concepts needed)
- At least two distractors represent partial integration (using only one concept)
- One distractor represents plausible but incorrect integration
- The integration must be clinically or scientifically meaningful (not forced)

**Anti-patterns:**
- Mentioning two concepts but testing only one (fake integration)
- Forcing integration between unrelated areas (must be meaningful)
- One concept clearly more important — both must contribute to the answer
- Testing whether the student knows both concepts separately — test whether they can COMBINE them
- If either concept alone is sufficient to answer, the question needs more complexity

**Domain examples:**
- CPAT + BPSY: "A 68-year-old presents with progressive word-finding difficulties, preserved spatial reasoning, and recent-onset depressive symptoms. Neuroimaging shows left temporal lobe atrophy. Considering both the neurocognitive profile and the mood presentation, which integrated formulation best accounts for the full clinical picture?"
- PMET + CASS: "A researcher develops a new anxiety screening tool showing excellent internal consistency (alpha = .93) and strong criterion validity against the BAI, but normed exclusively on college students. Considering both the psychometric strengths and the standardization limitation, which conclusion about using this tool in a community mental health setting is best supported?"
- PTHE + SOCU: "A therapist notices that her Japanese-American client consistently defers to suggestions, avoids disagreement, and describes family obligations as more important than personal goals. Considering both the therapeutic alliance literature on collaboration and cultural psychology research on collectivism, which interpretation is most clinically sound?"
- PETH + CASS: "A forensic psychologist is asked to evaluate a non-English-speaking defendant's competency to stand trial using a cognitive assessment battery normed on English speakers. Considering both the ethical standards on assessment with diverse populations and the psychometric implications of cross-cultural test use, which course of action is most appropriate?"

**Distractor design (misconception type manifestation at Tier 4):**
- **partial_understanding** — Student knows each concept independently but can't combine them. Picks an answer that's correct for one concept area in isolation but wrong for the integrated conclusion. This is the signature Tier 4 manifestation: complete knowledge of parts, failure to synthesize.
- **overgeneralization** — Student extends one concept to cover both areas instead of genuinely integrating. "Concept A explains everything; I don't need Concept B." The distractor is the Concept-A-only conclusion.
- **similar_property** — Two concept areas share a surface feature. Student synthesizes based on the shared feature rather than the deeper interaction, producing a plausible but shallow integration.
- **similar_store** — Concepts from the same broad family (e.g., two therapeutic modalities, two assessment frameworks). Student applies the wrong member of the family when integrating across domains.

**Character injection:** Mandatory
**Stem length:** 4-7 sentences
**Option length:** 25-45 words each

---

## Implementation Notes

### Changes to `pipeline/__init__.py`

```python
# Replace flat STEM_PATTERNS with tier-keyed dict
STEM_PATTERNS = {
    1: [
        ("direct_definition",
         "Present a concept and ask the student to select its correct definition from four options"),
        ("concept_identification",
         "Present a description and ask the student to identify the correct term"),
        ("fact_recognition",
         "Ask the student to identify the correct factual statement, grounded in an authority"),
        ("true_false_which",
         "Present four statements about the same concept; student identifies which is correct/incorrect"),
        ("feature_listing",
         "Ask which feature does NOT belong to a concept (EXCEPT/NOT format)"),
    ],
    2: [
        ("comparison",
         "Ask the student to identify the key distinction between two confusable concepts"),
        ("example_recognition",
         "Name a concept and present four brief scenarios; student picks the correct illustration"),
        ("simple_application",
         "Present a brief scenario; student identifies the concept being demonstrated"),
        ("paraphrase",
         "Name a concept; student selects the most accurate restatement in different words"),
        ("categorization",
         "Ask the student to classify an item within a real taxonomy or classification system"),
    ],
    3: [
        ("clinical_vignette",
         "Clinical vignette with named professional/client, specific setting, presenting concern"),
        ("scenario_completion",
         "Professional scenario: predict next step, outcome, or appropriate action"),
        ("error_identification",
         "Identify a substantive error in professional claims anchored in an applied context"),
        ("case_analysis",
         "Analyze a case for the underlying mechanism or causal explanation (WHY, not WHAT)"),
        ("mechanism_application",
         "Apply a named principle to a novel situation and predict the outcome"),
    ],
    4: [
        ("contrast_prompt",
         "Distinguish overlapping concepts within a specific case where both seem to apply"),
        ("best_answer",
         "All options contain truth; evaluate which is MOST correct in this specific context"),
        ("subtle_error",
         "Detect a nuanced flaw in mostly-correct expert reasoning (80%+ of claims accurate)"),
        ("competing_evidence",
         "Weigh two genuinely defensible positions; context provides the tiebreaker"),
        ("integration",
         "Synthesize knowledge from 2+ concept areas; neither alone is sufficient to answer"),
    ],
}
```

### Changes to variant-to-pattern lookup

```python
# Old: pattern_name, pattern_desc = STEM_PATTERNS[(variant_num - 1) % len(STEM_PATTERNS)]
# New: tier-aware lookup
def get_stem_pattern(tier, variant_num):
    patterns = STEM_PATTERNS[tier]
    pattern_name, pattern_desc = patterns[(variant_num - 1) % len(patterns)]
    return pattern_name, pattern_desc
```

### Changes to `prompts.py`

The `_blooms_stem_enforcement()` function currently returns empty string for T1-T2. It needs tier-specific enforcement for all 4 tiers:

- **Tier 1-2 enforcement:** Prevent upward creep. If the LLM generates a scenario-based question at Tier 1, reject it. If it requires multi-step reasoning at Tier 2, reject it.
- **Tier 3-4 enforcement:** Already exists. Keep the existing anti-patterns, named-effect guard, single-fact guard, and two-concept integration requirement for Tier 4.

### Changes to `stem_pattern` field in schema

```
# Old enum:
stem_pattern: clinical_vignette | scenario_completion | contrast_prompt | error_identification | best_answer

# New enum (20 values):
stem_pattern: direct_definition | concept_identification | fact_recognition | true_false_which | feature_listing |
              comparison | example_recognition | simple_application | paraphrase | categorization |
              clinical_vignette | scenario_completion | error_identification | case_analysis | mechanism_application |
              contrast_prompt | best_answer | subtle_error | competing_evidence | integration
```

### Existing 15,019 questions

Existing questions keep their current stem_pattern values unchanged. All 5 existing patterns are retained in the new system — just assigned to their natural tiers. No migration needed.

### Validation gate updates

The StructureGate should validate that stem_pattern matches the expected tier:
- Tier 1 question with clinical_vignette stem_pattern → reject
- Tier 4 question with direct_definition stem_pattern → reject
