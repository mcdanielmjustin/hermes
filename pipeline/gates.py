"""
Validation gates for the quiz question pipeline (Phase 4).

All gates are hardcoded (instant, $0). They check the assembled
question record for structural integrity, content quality,
diagnostic consistency, and uniqueness.

With vocab-backed mode, metadata fields (distractor_level,
misconception_type, etc.) are hardcoded by the assembler and
CANNOT fail validation — eliminating the largest source of
retry-causing failures from the original pipeline.
"""

import re

from . import BaseGate, VALID_MISCONCEPTION_TYPES, DISTRACTOR_MIX
from .citation_patterns import find_attributions
from .stopwords import (
    BASE_STOP_LIGHT as _BASE_STOP_LIGHT,
    BASE_FULL as _BASE_FULL,
    GENERIC_DESCRIPTORS as _GENERIC_DESCRIPTORS,
)


def _canonical_terms_from_context(context):
    """T3/T4 canonical vocabulary set: curated domain pool + brief
    concept-description terms. The keyword gates use this to restrict
    the unique-to-correct count to legitimate technical-synonym tells
    rather than generic English descriptive asymmetry.

    Returns an empty set when context is missing the inputs — in which
    case the canonical filter is bypassed (defensive: never make the
    gate stricter from a missing dependency).
    """
    if not context:
        return set()
    canonical = set()
    domain_vocab = context.get("domain_vocab") or []
    for term in domain_vocab:
        if term:
            canonical.add(term.lower())
    brief_concept_terms = context.get("brief_concept_terms") or []
    for term in brief_concept_terms:
        if term:
            canonical.add(term.lower())
    return canonical


class StructureGate(BaseGate):
    """Checks JSON structure: 4 options, 1 correct, required fields."""
    name = "structure"

    def check(self, q, context=None):
        if not q or "_error" in q:
            return False, q.get("_error", "null question")
        if "question_stem" not in q:
            return False, "missing question_stem"
        if "options" not in q or len(q["options"]) != 4:
            return False, f"expected 4 options, got {len(q.get('options', []))}"

        correct_count = sum(1 for o in q["options"] if o.get("is_correct"))
        if correct_count != 1:
            return False, f"expected 1 correct answer, got {correct_count}"

        if "tested_concept" not in q:
            return False, "missing tested_concept"
        if not q["tested_concept"].get("concept_id"):
            return False, "tested_concept missing concept_id"

        return True, "ok"


class ContentQualityGate(BaseGate):
    """Checks content quality: flashcard seeds, topic keywords, stem length."""
    name = "content_quality"

    def check(self, q, context=None):
        # Flashcard seed quality
        if "flashcard_seeds" not in q:
            return False, "missing flashcard_seeds"
        seeds = q["flashcard_seeds"]
        for seed_type in ("concept", "comparison", "nuance"):
            if seed_type not in seeds:
                return False, f"missing flashcard_seeds.{seed_type}"
            seed = seeds[seed_type]
            if not seed.get("front") or not seed.get("back"):
                return False, f"flashcard_seeds.{seed_type} missing front or back"
            if len(seed["front"]) < 20:
                return False, f"flashcard_seeds.{seed_type}.front too short ({len(seed['front'])} chars)"
            if len(seed["back"]) < 40:
                return False, f"flashcard_seeds.{seed_type}.back too short ({len(seed['back'])} chars)"

        # Topic keywords
        kw = q.get("topic_keywords")
        if not kw or not isinstance(kw, list) or len(kw) < 3:
            return False, f"topic_keywords needs 3+ items, got {kw}"
        if not all(isinstance(k, str) and len(k) > 1 for k in kw):
            return False, "topic_keywords must be non-empty strings"

        return True, "ok"


class ConsistencyGate(BaseGate):
    """Checks diagnostic consistency: concept_id matches, distractor fields."""
    name = "consistency"

    def check(self, q, context=None):
        # Correct option must match tested_concept
        correct_opt = next((o for o in q["options"] if o.get("is_correct")), None)
        if not correct_opt:
            return False, "no correct option found"
        if not correct_opt.get("concept_id"):
            return False, f"correct answer ({correct_opt.get('letter')}) missing concept_id"
        if correct_opt["concept_id"] != q["tested_concept"]["concept_id"]:
            return False, (
                f"tested_concept.concept_id ({q['tested_concept']['concept_id']}) "
                f"!= correct option concept_id ({correct_opt['concept_id']})"
            )

        # Correct option must have null distractor fields
        for field in ("distractor_level", "confused_with", "misconception_id", "misconception_type"):
            if correct_opt.get(field) is not None:
                return False, f"correct answer ({correct_opt.get('letter')}) has non-null {field}"

        # Wrong options must have all diagnostic fields
        misconception_ids = []
        for o in q["options"]:
            if not o.get("is_correct"):
                if not o.get("misconception_id"):
                    return False, f"option {o.get('letter')} missing misconception_id"
                if not o.get("concept_id"):
                    return False, f"option {o.get('letter')} missing concept_id"
                if not o.get("confused_with"):
                    return False, f"option {o.get('letter')} missing confused_with"
                if not o.get("misconception_label"):
                    return False, f"option {o.get('letter')} missing misconception_label"
                mt = o.get("misconception_type")
                if mt not in VALID_MISCONCEPTION_TYPES:
                    return False, f"option {o.get('letter')} invalid misconception_type: {mt}"
                dl = o.get("distractor_level")
                if not isinstance(dl, int) or dl not in (1, 2, 3, 4):
                    return False, f"option {o.get('letter')} invalid distractor_level: {dl}"
                misconception_ids.append(o["misconception_id"])

        # Diagnostic spread: unique misconception_ids
        if len(misconception_ids) != len(set(misconception_ids)):
            return False, f"duplicate misconception_ids: {misconception_ids}"

        return True, "ok"


class DistractorMixGate(BaseGate):
    """Warns (does not fail) if distractor-level distribution doesn't match tier.

    This is a soft gate — close-enough is acceptable. In vocab-backed mode
    this is always correct since the planner assigns levels deterministically.
    """
    name = "distractor_mix"
    is_prerequisite = False  # soft (always returns ok); needs to follow content gates

    def check(self, q, context=None):
        tier = q.get("difficulty_tier")
        if not tier or tier not in DISTRACTOR_MIX:
            return True, "ok"

        expected = DISTRACTOR_MIX[tier]
        actual = {}
        for o in q["options"]:
            if not o.get("is_correct"):
                dl = o.get("distractor_level")
                if isinstance(dl, int):
                    key = f"L{dl}"
                    actual[key] = actual.get(key, 0) + 1

        if actual != expected:
            # Warn but don't fail
            print(f"    \u26a0 distractor mix: expected {expected}, got {actual}")

        return True, "ok"


class AnchorGroundingGate(BaseGate):
    """Checks that the tested concept comes from the anchor brief's concept list."""
    name = "anchor_grounding"
    is_prerequisite = False  # content-level — collect alongside other content failures

    def check(self, q, context=None):
        if context is None:
            return True, "ok"
        anchor_concept_ids = context.get("anchor_concept_ids")
        if not anchor_concept_ids:
            return True, "ok"

        tested_id = q.get("tested_concept", {}).get("concept_id", "")
        if tested_id in anchor_concept_ids:
            return True, "ok"

        return False, f"tested_concept '{tested_id}' not in anchor brief concepts: {anchor_concept_ids}"


class UniquenessGate(BaseGate):
    """Checks question_id not already in the output file."""
    name = "uniqueness"

    def check(self, q, context=None):
        if context is None:
            return True, "ok"
        existing_ids = context.get("existing_ids", set())
        qid = q.get("question_id")
        if qid and qid in existing_ids:
            return False, f"duplicate question_id: {qid}"
        return True, "ok"


class OptionLengthBalanceGate(BaseGate):
    """Fails if option lengths reveal the correct answer.

    Two failure modes (either triggers a failure):
      A. Spread:    max/min character ratio across all 4 options exceeds ratio_max.
      B. Tell:      correct option is longer than the longest distractor by
                    more than tell_margin (e.g., 20%).

    Both are well-documented testwise heuristics: students learn that
    "longest answer = correct" and "the most carefully qualified option is
    correct." Threshold tuned from baseline audit (15/20 questions had
    correct = longest before this gate).
    """
    name = "option_length_balance"
    is_prerequisite = False  # content-level — collect alongside other content failures

    def __init__(self, ratio_max=1.7, tell_margin=1.2):
        self.ratio_max = ratio_max
        self.tell_margin = tell_margin

    def check(self, q, context=None):
        opts = q.get("options", [])
        if len(opts) != 4:
            return True, "ok"  # StructureGate owns this

        lengths = {o["letter"]: len(o.get("text", "")) for o in opts}
        if not all(lengths.values()):
            return True, "ok"

        mn = min(lengths.values())
        mx = max(lengths.values())
        ratio = mx / mn

        correct = next((o for o in opts if o.get("is_correct")), None)
        if not correct:
            return True, "ok"
        c_len = lengths[correct["letter"]]
        d_lens = [v for k, v in lengths.items() if k != correct["letter"]]
        d_max = max(d_lens)

        if ratio > self.ratio_max:
            return False, (
                f"option length ratio {ratio:.2f} exceeds {self.ratio_max} "
                f"(min={mn}, max={mx}, lengths={lengths})"
            )

        if c_len > d_max * self.tell_margin:
            return False, (
                f"correct option ({correct['letter']}, {c_len} chars) is longer "
                f"than max distractor ({d_max} chars) by >{int((self.tell_margin-1)*100)}% "
                f"(lengths={lengths})"
            )

        return True, "ok"


class AttributionGate(BaseGate):
    """Fails if the question attributes findings to non-whitelisted researchers.

    Catches patterns the InputSanitizerAgent missed (LLM may invent attributions
    even when the prompt input was clean). Whitelist exemption preserves
    legitimate eponyms (Piaget's stages, Pavlovian conditioning, etc.).

    Uses the canonical patterns from citation_patterns — same regexes used
    by the audit and sweep scripts, so all three tools agree on what counts
    as an attribution.
    """
    name = "attribution"
    is_prerequisite = False  # content-level — collect alongside other content failures

    @staticmethod
    def _scan(text):
        """Return [(matched_str, kind), ...] for non-whitelisted attributions."""
        return [(m, k) for m, _, k, wl in find_attributions(text) if not wl]

    def check(self, q, context=None):
        sources = {
            "stem":            q.get("question_stem", ""),
            "tested_concept":  q.get("tested_concept", {}).get("knowledge_tested", ""),
        }
        for o in q.get("options", []):
            sources[f"opt_{o['letter']}_text"] = o.get("text", "")
            sources[f"opt_{o['letter']}_explanation"] = o.get("explanation", "")

        violations = []
        for where, text in sources.items():
            for matched, kind in self._scan(text):
                violations.append(f"{where}:{kind}:'{matched}'")
                if len(violations) >= 3:
                    break
            if len(violations) >= 3:
                break

        if violations:
            return False, "researcher attribution detected — " + "; ".join(violations)
        return True, "ok"


class BloomsCognitiveLevelGate(BaseGate):
    """Catches T3/T4 questions that violate the explicit Bloom's anti-patterns
    in `_blooms_stem_enforcement()`.

    Empirical audit found 50% of T3/T4 questions violated the prompt's stated
    rules — T4 questions answerable by single-concept recall, T3 questions
    that test definition identification despite scenario dressing. The prompt
    rules exist; this gate codifies them as post-generation enforcement.

    T3 path (Bloom's primary: apply): correct answer must have at least one
    application/analysis indicator — a verb that requires going beyond
    definition recall, OR comparison/contrastive structure, OR a specific
    consequence-prediction phrasing. A bare definitional statement at T3
    is the documented anti-pattern and fails the gate.

    T4 path (Bloom's primary: evaluate): the question must integrate at least
    2 concepts from the brief — operationalized as: stem + correct answer
    combined must reference ≥2 distinct concept_ids OR ≥2 distinct concept
    labels from the brief's concept list. The "single-concept-sufficient"
    failure pattern is the dominant T4 violation.

    is_prerequisite=False — collected with other content-gates.
    """
    name = "blooms_cognitive_level"
    is_prerequisite = False

    # T3 application/analysis indicator verbs and phrases. The presence of
    # any one of these in the correct answer signals genuine apply-level
    # cognitive demand rather than definitional recall.
    # NB: this set MUST cover every verb the CorrectAnswerFormPlanner-
    # Agent injects into T3 options via _VERB_POOL[3]. Drift between
    # the form planner's permitted verbs and this gate's recognized
    # verbs surfaces as a "Tier 3 anti-pattern" failure on questions
    # the form planner explicitly permitted (e.g., "Infer left-
    # hemisphere lesion..." reads as bare-definitional to the gate
    # while the planner picked "infer" as a legal application verb).
    _T3_APPLY_INDICATORS = frozenset({
        # Application verbs (mirror form planner's T3 _VERB_POOL)
        "predict", "predicts", "predicted", "predicting",
        "apply", "applies", "applied", "applying",
        "determine", "determines", "determined", "determining",
        "decide", "decides", "decided", "deciding",
        "choose", "chooses", "chose", "choosing",
        "select", "selects", "selected", "selecting",
        "infer", "infers", "inferred", "inferring",
        "recommend", "recommends", "recommended", "recommending",
        "evaluate", "evaluates", "evaluated", "evaluating",
        "assess", "assesses", "assessed", "assessing",
        # Analysis verbs
        "distinguish", "distinguishes", "distinguished", "distinguishing",
        "differentiate", "differentiates", "differentiated", "differentiating",
        "compare", "compares", "compared", "comparing",
        "contrast", "contrasts", "contrasted", "contrasting",
        # Outcome/consequence words signal forward-reasoning
        "result", "results", "outcome", "outcomes", "consequence",
        "leads", "lead", "produce", "produces", "produced",
        "increase", "decrease", "improve", "worsen",
        # Comparative connectives indicate parallel reasoning structure
        "whereas", "however", "rather than", "instead",
    })

    @staticmethod
    def _content_words(text):
        """Lowercase 4+ char words from text."""
        if not text:
            return set()
        return set(re.findall(r"\b[a-zà-öø-ÿ]{4,}\b", text.lower()))

    def _t3_check(self, q):
        correct = next((o for o in q["options"] if o.get("is_correct")), None)
        if not correct:
            return True, "ok"
        ct = (correct.get("text") or "").lower()
        # Match any apply/analyze indicator. Word-boundary on each.
        for indicator in self._T3_APPLY_INDICATORS:
            # Multi-word indicators ("rather than") use 'in' substring; single
            # words use word boundary
            if " " in indicator:
                if indicator in ct:
                    return True, "ok"
            else:
                if re.search(r"\b" + re.escape(indicator) + r"\b", ct):
                    return True, "ok"
        return False, (
            "T3 correct answer reads as a bare definitional statement; no "
            "application/analysis verbs (predict, distinguish, evaluate, "
            "apply, determine, etc.) or comparative structure detected. "
            "Tier 3 anti-pattern: students should APPLY the concept, not "
            "identify its label."
        )

    def _t4_check(self, q, context):
        """Stem + correct answer combined must reference ≥2 distinct
        concept_ids or labels from the brief."""
        if context is None:
            return True, "ok"
        anchor_concept_ids = context.get("anchor_concept_ids")
        anchor_concept_labels = context.get("anchor_concept_labels")
        # If we don't know the brief's concepts, can't check. Pass.
        if not anchor_concept_ids and not anchor_concept_labels:
            return True, "ok"

        stem = q.get("question_stem", "") or ""
        correct = next((o for o in q["options"] if o.get("is_correct")), None)
        correct_text = (correct.get("text") or "") if correct else ""
        correct_expl = (correct.get("explanation") or "") if correct else ""
        haystack = (stem + " " + correct_text + " " + correct_expl).lower()

        # Count distinct concept references. concept_id slug-tokens (kebab-case
        # parts ≥4 chars) count when at least 1 token appears. Labels also count.
        concept_hits = set()
        for cid in (anchor_concept_ids or []):
            tokens = [t for t in cid.lower().split("-") if len(t) >= 4]
            if not tokens:
                continue
            if any(t in haystack for t in tokens):
                concept_hits.add(cid)
        for cid, label in (anchor_concept_labels or {}).items():
            label_words = [w for w in re.findall(
                r"\b[a-zà-öø-ÿ]{4,}\b", (label or "").lower())]
            if label_words and any(w in haystack for w in label_words):
                concept_hits.add(cid)

        if len(concept_hits) < 2:
            return False, (
                f"T4 question integrates only {len(concept_hits)} brief "
                f"concept(s); evaluation-level questions require ≥2-concept "
                f"integration. Hits: {sorted(concept_hits)}. Rewrite stem "
                f"and correct answer to require BOTH concepts simultaneously."
            )
        return True, "ok"

    def check(self, q, context=None):
        tier = q.get("difficulty_tier")
        if len(q.get("options", [])) != 4:
            return True, "ok"  # StructureGate owns this
        if tier == 3:
            return self._t3_check(q)
        if tier == 4:
            return self._t4_check(q, context)
        # T1/T2 not gated by this — Bloom's anti-patterns there are different
        return True, "ok"


class ApplyIdentityGate(BaseGate):
    """T3 (Apply) identity gate.

    Bloom's Apply = "carry out a procedure in a given situation." Three
    structural requirements all must hold:
      1. A given situation — the stem must contain a novel scenario
         (named subject, clinical context, age + role, "patient" /
         "client" framing, "given X" / "after Y" contextual setup).
      2. Carry out / use — the correct option must contain a forward-
         action application verb followed by enough content words to
         constitute a meaningful prediction (verb + ≥4 content words),
         not a verb tagged onto a bare label.
      3. Mechanism connection — the correct option must contain a
         mechanism/causal marker (from / via / through / reflecting /
         producing / mediated by / with [impaired] X / by [verb-ing] Y)
         that links the predicted outcome to its underlying cause.
         Without this, "Predict X as part of the triad" passes the
         word-count check but is T2 labeling cognition under a T3 stem
         shell. Empirical: 3/5 T3 questions in the D7-PHY-058 batch
         drifted into this labeling-as-prediction pattern despite
         passing checks 1 and 2.

    Other tiers exempt:
      • T1 (Recognize) — definitional answers are correct by design.
      • T2 (Understand) — comprehension-level questions may not have
        a scenario; that's allowed.
      • T4 (Analyze/Evaluate) — has its own integration check via
        BloomsCognitiveLevelGate (≥2-concept integration). Apply
        identity is T3-specific.

    Architectural rationale: T3 is empirically the failure tier (67%
    pass at T3 vs 100% at T1/T2/T4 in recent calibrations). The
    failures cluster around Apply identity violations: stems without
    scenarios (find-error patterns), or correct options that are bare
    labels with verbs tagged on. This gate codifies Apply identity at
    validation time, complementing the form-planner-side prevention
    (trimmed STEM_PATTERNS[3] + trimmed _VERB_POOL[3]).

    is_prerequisite=False — collected with other content gates.
    """
    name = "apply_identity"
    is_prerequisite = False

    _STOP = _BASE_FULL

    # Stem-novelty markers. At least one must be present for the stem
    # to qualify as "a given situation" rather than a recall prompt.
    _SCENARIO_PATTERNS = [
        # Title + name (Dr. X, Ms. Y, Prof. Z, etc.)
        r"\b(Dr|Mr|Ms|Mrs|Miss|Prof|Professor)\.?\s+[A-Z][a-z]+",
        # Age markers
        r"\b\d{1,3}-year-old\b",
        r"\bage\s+\d{1,3}\b",
        # Clinical / scenario keywords
        r"\b(patient|client|case|scenario|vignette|resident|clinician)\b",
        # Clinical-action stem opens
        r"\b(presents|evaluates|examines|reports|reviews|encounters|"
        r"admits|discharged|consults|interviewing|seeing|assessing)\b",
        # "Given X" / "After Y" framing
        r"\b(given|following|after|during)\s+(a|an|the|that)\b",
        # Imaging / diagnostic context
        r"\b(imaging|MRI|CT|fMRI|EEG|examination)\b",
    ]

    # Application verbs accepted as the forward-action marker. Must
    # mirror the form planner's T3 _VERB_POOL[3] so verbs the planner
    # actually injects are recognized here.
    _APPLICATION_VERBS = frozenset({
        "predict", "predicts", "predicted", "predicting",
        "determine", "determines", "determined", "determining",
        "apply", "applies", "applied", "applying",
        "choose", "chooses", "chose", "choosing",
        "select", "selects", "selected", "selecting",
    })

    # Min content-words AFTER the application verb in the correct
    # option's text. Verb + ≥4 substance words = a meaningful
    # prediction. Calibrated against D7-PHY-058/D7-PHY-209 clean
    # questions (most have 6-10 post-verb content words; bare-label
    # failures had ≤2).
    _MIN_POST_VERB_CONTENT_WORDS = 4

    # Mechanism-presence check. The "verb + ≥4 content words" rule
    # passes "Predict mood swings as part of the frontal triad" (a
    # T2-flavored labeling-as-prediction) the same as it passes
    # "Predict mood swings reflecting lost limbic regulation" (genuine
    # apply). Empirically, the LLM produces both at T3; the labeling
    # form is T2 cognition wearing a T3 stem shell. To distinguish
    # them, require the correct option to contain at least one
    # MECHANISM/CAUSAL marker — a connective that links the prediction
    # to a reason rather than a taxonomy slot.
    #
    # Excludes "because" / "since" / "due to" / "owing to" — those are
    # OptionClaimGate's reasoning markers (forbidden in option.text),
    # so they can't be required here. The accepted markers are
    # PREPOSITIONS / GERUNDS / CAUSAL VERB FORMS that fit the noun-
    # phrase shape OptionClaimGate enforces.
    # Phase 10 — accepts mechanism markers OR criterion-application
    # markers. Some content domains (DSM-5/ICD-11 diagnostic criteria,
    # ethical-procedure standards, age/duration thresholds) are
    # fundamentally criterion-driven: T3 cognition is "apply this
    # threshold to this case" rather than "predict via biological
    # mechanism." For these, the criterion IS the causal anchor — the
    # determination follows from the rule.
    #
    # Empirical motivation: CPAT D3-PPA-034 (ADHD/autism criteria)
    # lost 5/5 T3 questions in Phase 9 cross-anchor validation, all
    # to mechanism-marker absence despite producing valid criterion-
    # application reasoning ("based on age-12 cutoff", "for failing
    # the 6-month persistence threshold"). Adding these phrasings as
    # accepted markers recovers the criteria-driven path without
    # weakening the mechanism-rich path.
    _MECHANISM_MARKERS = re.compile(
        r"\b("
        # Causal prepositions (not in OptionClaim's _REASONING_MARKERS)
        r"from|via|through|"
        # Causal verb forms — not reasoning conjunctions, but causal
        # participles or passive constructions
        r"reflecting|reflects|producing|produces|caused\s+by|"
        r"mediated\s+by|driven\s+by|resulting\s+from|stemming\s+from|"
        r"manifesting\s+(as|in)|"
        # By-verbing causal construction (e.g., "by disrupting X")
        r"by\s+\w+ing|"
        # Mechanism-state descriptors
        r"with\s+(poor|impaired|reduced|enhanced|disrupted|intact|"
        r"lost|preserved|restored|abolished|spared|severed)\s+\w+|"
        # Phase 10 — criterion-application markers for criteria-driven
        # content. Causal/conditional connectives that fit the noun-
        # phrase form OptionClaimGate enforces (no 'because/since/due
        # to') and link the determination to its underlying criterion
        # or threshold.
        r"based\s+on|given\s+that|"
        # Verb + criterion-noun within ~30 chars (allows phrasings
        # like 'satisfying the age-12 onset threshold')
        r"(?:satisfies|satisfying|meets|meeting|fails|failing|"
        r"exceeds|exceeding)\s+[^.?!]{0,30}?"
        r"(?:criterion|criteria|threshold|cutoff|requirement|standard)|"
        r"for\s+(?:meeting|failing|satisfying|exceeding)"
        r")\b",
        re.IGNORECASE,
    )

    @classmethod
    def _has_scenario(cls, stem):
        for pattern in cls._SCENARIO_PATTERNS:
            if re.search(pattern, stem, re.IGNORECASE):
                return True
        return False

    @classmethod
    def _has_application_substance(cls, text):
        """Find any application verb in option text; require enough
        content words follow it to constitute a meaningful prediction.

        Returns False if the text contains no application verb, OR if
        every application verb's following content (after stop-word
        removal and 5+ char filter) has fewer than _MIN_POST_VERB_
        CONTENT_WORDS.
        """
        if not text:
            return False
        text_lower = text.lower()
        # Search for any application verb. Loop because multiple verbs
        # may appear (rare); accept the question if ANY of them is
        # followed by enough content.
        for verb in cls._APPLICATION_VERBS:
            for m in re.finditer(r"\b" + re.escape(verb) + r"\b", text_lower):
                tail = text[m.end():]
                tail_words = re.findall(
                    r"\b[a-zà-öø-ÿ]{5,}\b", tail.lower()
                )
                content_words = [w for w in tail_words if w not in cls._STOP]
                if len(content_words) >= cls._MIN_POST_VERB_CONTENT_WORDS:
                    return True
        return False

    def check(self, q, context=None):
        if q.get("difficulty_tier") != 3:
            return True, "ok"  # T1/T2/T4 exempt

        stem = q.get("question_stem", "") or ""
        if not self._has_scenario(stem):
            return False, (
                "T3 (Apply) stem lacks a novel scenario; reads as "
                "definitional/recall. Apply identity requires a given "
                "situation the student carries the concept into. Add "
                "a named subject (Dr./Mr./Ms.), clinical context "
                "(patient/client/case), age, or scenario framing "
                "(given/following/after) to the stem."
            )

        correct = next(
            (o for o in q.get("options", []) if o.get("is_correct")),
            None,
        )
        if not correct:
            return True, "ok"

        correct_text = correct.get("text", "")
        if not self._has_application_substance(correct_text):
            return False, (
                "T3 (Apply) correct option lacks application substance: "
                "the application verb is followed by too few content "
                "words. Apply identity requires verb + meaningful "
                "prediction (e.g., 'Predict bilateral hemiplegia from "
                "decussation in the medulla'), not a verb tagged onto "
                "a bare label (e.g., 'Predict hemiplegia.'). Expand "
                "the correct option to specify the predicted outcome "
                "with at least 4 content words after the verb."
            )

        # Mechanism-presence check. A correct option that names the
        # outcome but lacks a causal/mechanism connective is doing T2
        # labeling cognition under a T3 stem shell. Phase 10 broadens
        # the accepted markers to include criterion-application
        # connectives ('based on', 'given that', 'satisfying/failing
        # the threshold/criterion/cutoff') for criteria-driven content
        # where the criterion IS the causal anchor.
        if not self._MECHANISM_MARKERS.search(correct_text):
            return False, (
                "T3 (Apply) correct option lacks a causal anchor — "
                "either a MECHANISM marker (e.g., 'from', 'via', "
                "'through', 'reflecting', 'producing', 'mediated by', "
                "'with [impaired/reduced] X', 'by [verb-ing] Y') OR a "
                "CRITERION-APPLICATION marker (e.g., 'based on [X]', "
                "'given that [X]', 'satisfying the threshold', "
                "'failing the criterion', 'for exceeding the cutoff'). "
                "The current correct answer names the outcome/decision "
                "but does not link it to a mechanism or criterion — "
                "this is T2 labeling cognition under a T3 stem shell. "
                "Apply identity requires the correct answer to connect "
                "the determination to its underlying mechanism (for "
                "mechanism-rich content like biopsych) OR criterion "
                "(for criteria-driven content like DSM diagnoses, "
                "ethical thresholds), not just identify it as a "
                "category member."
            )

        return True, "ok"


class RememberIdentityGate(BaseGate):
    """T1 (Remember/Recognize) identity gate.

    Bloom's Remember = retrieve a stored fact, definition, or feature.
    Curated structure (inverse of ApplyIdentityGate's positive scenario
    check):

      1. Stem must NOT contain scenario indicators. T1 stems are direct
         questions about facts/definitions, not vignettes. A T1 stem
         that names a clinical subject (Dr. X, age N, "the patient")
         has drifted toward T2/T3 territory.
      2. Correct option must NOT start with forward-action verbs
         (predict/determine/apply/choose/select). T1 correct answers
         are static forms — definitions, labels, feature lists — not
         predictions.

    The patterns that catch scenario drift are TIGHTER than ApplyIdentity
    Gate's positive check: only specific scenario indicators count
    (title+name, age, "the patient/client", clinical action verbs).
    Generic medical references (e.g., "stroke patients" plural, "in
    clinical practice") do NOT fire — those can legitimately appear
    in T1 stems describing a class of phenomena.

    is_prerequisite=False — collected with other content gates.
    """
    name = "remember_identity"
    is_prerequisite = False

    # TIGHTER scenario patterns than ApplyIdentityGate. These match
    # SPECIFIC individuals/cases — not generic medical context.
    _SPECIFIC_SCENARIO_PATTERNS = [
        # Title + name (Dr. X, Ms. Y, Mr. Z, Prof. Z)
        r"\b(Dr|Mr|Ms|Mrs|Miss|Prof|Professor)\.?\s+[A-Z][a-z]+",
        # Age compounds
        r"\b\d{1,3}-year-old\b",
        r"\bage\s+\d{1,3}\b",
        # Definite-article references to a SPECIFIC person/case
        # ("the patient" but not "patients" or "patients commonly")
        r"\bthe\s+(patient|client|case|resident|child|adolescent)\b",
        r"\bthis\s+(patient|client|case|scenario|vignette|individual)\b",
        # Clinical action verbs (someone doing something specific)
        r"\b(presents|evaluates|examines|reports|admits|interviewing|"
        r"discharged|consulting|seeing\s+a)\b",
    ]

    # Forward-action verbs that don't belong at T1. T1's own verbs
    # (identify/recognize/name/define/label) are allowed; only T3's
    # apply verbs are flagged here.
    _FORWARD_ACTION_VERBS = frozenset({
        "predict", "predicts", "predicted", "predicting",
        "determine", "determines", "determined", "determining",
        "apply", "applies", "applied", "applying",
        "choose", "chooses", "chose", "choosing",
        "select", "selects", "selected", "selecting",
        # Forward-action phrases LLMs sometimes substitute
        "infer", "infers", "anticipate", "anticipates",
    })

    @classmethod
    def _has_specific_scenario(cls, stem):
        for pattern in cls._SPECIFIC_SCENARIO_PATTERNS:
            if re.search(pattern, stem, re.IGNORECASE):
                return True
        return False

    @classmethod
    def _starts_with_forward_action(cls, text):
        if not text:
            return False
        words = text.strip().split()
        if not words:
            return False
        first = words[0].rstrip(",.:;").lower()
        return first in cls._FORWARD_ACTION_VERBS

    def check(self, q, context=None):
        if q.get("difficulty_tier") != 1:
            return True, "ok"  # T2/T3/T4 exempt

        stem = q.get("question_stem", "") or ""
        if self._has_specific_scenario(stem):
            return False, (
                "T1 (Remember) stem contains a specific scenario indicator "
                "(named subject, age marker, 'the patient/client', or a "
                "clinical-action verb). T1 questions test retrieval of a "
                "fact/definition; vignette stems push the cognitive demand "
                "into T2/T3 territory. Rewrite as a direct question."
            )

        correct = next(
            (o for o in q.get("options", []) if o.get("is_correct")),
            None,
        )
        if correct and self._starts_with_forward_action(correct.get("text", "")):
            return False, (
                "T1 (Remember) correct option starts with a forward-action "
                "verb (predict/determine/apply/choose/select/infer). T1 "
                "answers are static forms (definitions, labels, features), "
                "not predictions. Rewrite the correct option as a noun "
                "phrase or short declarative claim."
            )
        return True, "ok"


class UnderstandIdentityGate(BaseGate):
    """T2 (Understand) identity gate.

    Bloom's Understand = comprehend a concept (explain, classify,
    paraphrase, recognize an example). Curated structure:

      1. Stem brevity cap. T2 allows brief context (1-2 sentences max,
         project's prompt rule), but vignettes longer than that drift
         into T3 application territory. Cap: ≤280 chars AND ≤40 words.
      2. Non-evaluative framing. T4-specific framings ("MOST appropriate",
         "MOST defensible", "critique", "defend the choice") signal
         Evaluate cognition, not Understand. Note: "best illustrates"
         and "most accurate restatement" are LEGITIMATE T2 framings
         (per STEM_PATTERNS[2]) and do not fire.

    is_prerequisite=False — collected with other content gates.
    """
    name = "understand_identity"
    is_prerequisite = False

    _MAX_STEM_CHARS = 280
    _MAX_STEM_WORDS = 40

    # Specific T4-Evaluate framings that should NOT appear in T2 stems.
    # Crafted tightly so legitimate T2 patterns aren't false-flagged:
    # "best illustrates" and "most accurate restatement" are valid
    # T2 stems (from STEM_PATTERNS[2]) and don't match these.
    _EVALUATIVE_FRAMING = re.compile(
        r"\b(most\s+appropriate|most\s+defensible|most\s+effective|"
        r"critique|defend\s+(this|your|the)\s+(choice|position|claim)|"
        r"weigh\s+(the|these)\s+(competing|defensible))\b",
        re.IGNORECASE,
    )

    def check(self, q, context=None):
        if q.get("difficulty_tier") != 2:
            return True, "ok"  # T1/T3/T4 exempt

        stem = q.get("question_stem", "") or ""

        # Check 1: brevity
        if len(stem) > self._MAX_STEM_CHARS:
            return False, (
                f"T2 (Understand) stem is too long ({len(stem)} chars > "
                f"{self._MAX_STEM_CHARS}). T2 allows brief context "
                f"(1-2 sentences max); longer vignettes drift into T3 "
                f"application territory. Tighten the stem."
            )
        word_count = len(stem.split())
        if word_count > self._MAX_STEM_WORDS:
            return False, (
                f"T2 (Understand) stem too long ({word_count} words > "
                f"{self._MAX_STEM_WORDS}). T2 allows 1-2 sentences max."
            )

        # Check 2: no evaluative framing
        m = self._EVALUATIVE_FRAMING.search(stem)
        if m:
            return False, (
                f"T2 (Understand) stem uses evaluative framing "
                f"({m.group()!r}). 'MOST appropriate' / 'critique' / "
                f"'defend' are T4 (Analyze/Evaluate) markers; they ask "
                f"for judgment among defensibles, not comprehension. "
                f"Rewrite using comprehension framings: 'Which describes' "
                f"/ 'Which best illustrates' / 'Which classifies'."
            )
        return True, "ok"


class EvaluateIdentityGate(BaseGate):
    """T4 (Analyze/Evaluate) identity gate.

    Bloom's Analyze + Evaluate = reason ABOUT a complex stimulus —
    compare relationships, judge value, defend choices. Curated
    structure SUPPLEMENTING BloomsCognitiveLevelGate's _t4_check
    (which enforces the integration subtype via ≥2-concept reference):

      Complex stimulus check. T4 stems require either multiple
      sentences (≥2 sentence-ending punctuation marks), conjunctive
      complexity (whereas/however/yet/although markers), OR
      competing-claim/expert-reasoning markers (argues/claims/the
      position/in expert reasoning). A simple one-sentence stem
      doesn't carry enough complexity for Evaluate cognition.

    Note: an earlier draft of this gate also enforced "defensible
    distractors" via Jaccard overlap with the correct option. That
    check was REMOVED after empirical false positives — misconception-
    based distractors at T4 legitimately introduce the misconception's
    vocabulary (e.g., "reframe as depression" on a frontal-lobe-triad
    question), which lowers direct vocab overlap with correct even
    though the distractor IS engaging the stem's realm. TopicRealmGate
    already detects truly off-topic distractors via realm overlap;
    duplicating the check here added noise without signal.

    is_prerequisite=False — collected with other content gates.
    """
    name = "evaluate_identity"
    is_prerequisite = False

    _CONJUNCTIVE_COMPLEXITY = re.compile(
        r"\b(whereas|however|yet|although|nevertheless|even\s+though|"
        r"despite|in\s+contrast)\b",
        re.IGNORECASE,
    )
    _COMPETING_CLAIM_MARKERS = re.compile(
        r"\b(argues|claims|argued|the\s+position|asserts|maintains|"
        r"insists|defends|expert\s+reasoning|the\s+resident\s+reasons|"
        r"competing\s+(claims|positions|evidence))\b",
        re.IGNORECASE,
    )

    @classmethod
    def _has_complex_stimulus(cls, stem):
        # Multiple sentences (≥2 sentence-ending marks)
        sentence_ends = sum(stem.count(m) for m in (".", "?", "!"))
        if sentence_ends >= 2:
            return True
        # Conjunctive complexity within a single sentence
        if cls._CONJUNCTIVE_COMPLEXITY.search(stem):
            return True
        # Competing-claim / expert-reasoning framing
        if cls._COMPETING_CLAIM_MARKERS.search(stem):
            return True
        return False

    def check(self, q, context=None):
        if q.get("difficulty_tier") != 4:
            return True, "ok"  # T1/T2/T3 exempt

        stem = q.get("question_stem", "") or ""
        if not self._has_complex_stimulus(stem):
            return False, (
                "T4 (Analyze/Evaluate) stem lacks a complex-stimulus "
                "marker. T4 cognition requires reasoning ABOUT something "
                "complex — multi-part case (≥2 sentences), conjunctive "
                "complexity (whereas/however/although), or competing-"
                "claim framing (argues/claims/the position/expert "
                "reasoning). Simple one-sentence stems push the question "
                "into T2/T3 territory. Add a multi-part case, contrast, "
                "or expert claim to evaluate."
            )

        return True, "ok"


class DomainExpertiseGate(BaseGate):
    """Catches the lay-person solvability creep at T2+.

    Empirical audit found 60-80% of T2+ questions could be answered by a
    smart non-psychologist with general world knowledge. The prompt rules
    require domain expertise; this gate codifies a concrete heuristic: the
    correct answer + explanation combined must contain ≥2 technical terms
    from the brief's concept labels or testable_fact. Single-technical-term
    correct answers are too easy to pattern-match.

    Hard fail at T3+, soft warn at T2 (and not gated at T1, where
    foundational recall is the design goal).

    is_prerequisite=False.
    """
    name = "domain_expertise"
    is_prerequisite = False

    @staticmethod
    def _technical_term_set(context):
        """Build a set of significant content words from the brief's concept
        labels and testable_fact. These are the domain-specific vocabulary
        that signals psychology-knowledge use."""
        terms = set()
        if not context:
            return terms
        for label in (context.get("anchor_concept_labels") or {}).values():
            for w in re.findall(r"\b[a-zà-öø-ÿ]{5,}\b", (label or "").lower()):
                terms.add(w)
        for w in re.findall(r"\b[a-zà-öø-ÿ]{5,}\b",
                            (context.get("anchor_testable_fact") or "").lower()):
            terms.add(w)
        # Strip common-but-non-technical words that creep into labels
        common = {
            "system", "process", "concept", "general", "model", "theory",
            "approach", "method", "function", "factor",
        }
        return terms - common

    def check(self, q, context=None):
        tier = q.get("difficulty_tier")
        if tier is None or tier < 2:
            return True, "ok"
        terms = self._technical_term_set(context)
        if not terms:
            # No brief vocabulary to test against — can't check.
            return True, "ok"

        correct = next((o for o in q.get("options", []) if o.get("is_correct")), None)
        if not correct:
            return True, "ok"
        text = (correct.get("text", "") + " " + correct.get("explanation", "")).lower()
        hits = [t for t in terms if t in text]
        if len(hits) >= 2:
            return True, "ok"
        # Tier 2 gets a soft warn (still passes); Tier 3+ hard fail.
        if tier == 2:
            # ASCII-safe to avoid Windows console encoding errors when
            # the test runner's stdout isn't UTF-8 configured.
            try:
                print(f"    [warn] domain_expertise: T2 correct answer has only "
                      f"{len(hits)} technical term(s) (hits={hits})")
            except Exception:
                pass
            return True, "ok"
        return False, (
            f"T{tier} correct answer contains only {len(hits)} technical "
            f"term(s) from the brief's vocabulary (hits={hits}). A T{tier} "
            f"question must require domain expertise — rewrite so the correct "
            f"answer + explanation use at least 2 brief-vocabulary terms "
            f"(concept labels or testable_fact key terms)."
        )


class ScopeMatchGate(BaseGate):
    """For comparison/contrast/best_answer stems, distractors must address
    the same concepts the correct answer addresses.

    User's colleague flagged: "if a question discusses both agonist and
    antagonist in a comparison, some answer choices only address one or
    the other." When the correct answer compares X-vs-Y, a one-sided
    distractor is testwise-defective — students can rule it out by spotting
    the missing scope alone, not by understanding the concepts.

    Heuristic: detect concept references in correct answer using the brief's
    concept labels. Each distractor must reference at least the same number
    of brief concepts (within tolerance), or fail.

    Only fires for stem patterns where symmetric scope is structurally
    required: comparison, contrast_prompt, best_answer.

    is_prerequisite=False (collected with other content gates).
    """
    name = "scope_match"
    is_prerequisite = False

    SCOPE_REQUIRED_PATTERNS = frozenset({
        "comparison", "contrast_prompt", "best_answer",
    })

    @staticmethod
    def _concept_hits(text, concept_labels):
        """Count distinct brief concepts referenced in `text`.

        Each concept counts if at least one significant word from its
        label (4+ chars, lowercase) appears in the text.
        """
        if not text or not concept_labels:
            return set()
        text_lower = text.lower()
        hits = set()
        for cid, label in concept_labels.items():
            words = [w for w in re.findall(
                r"\b[a-zà-öø-ÿ]{4,}\b", (label or "").lower())]
            if words and any(w in text_lower for w in words):
                hits.add(cid)
        return hits

    def check(self, q, context=None):
        if q.get("stem_pattern") not in self.SCOPE_REQUIRED_PATTERNS:
            return True, "ok"
        if context is None:
            return True, "ok"
        labels = context.get("anchor_concept_labels")
        if not labels:
            return True, "ok"
        if len(q.get("options", [])) != 4:
            return True, "ok"  # StructureGate owns this

        correct = next((o for o in q["options"] if o.get("is_correct")), None)
        if not correct:
            return True, "ok"
        correct_hits = self._concept_hits(correct.get("text", ""), labels)
        # If the correct answer itself doesn't reference 2+ concepts, the
        # stem-pattern requirement is moot — pass.
        if len(correct_hits) < 2:
            return True, "ok"

        # Symmetric scope rule:
        #   • If correct addresses exactly 2 concepts (the typical comparison
        #     case), distractors must address BOTH — no one-sided distractors.
        #     This is the dominant failure mode: dropping the second concept
        #     creates a testwise tell.
        #   • If correct addresses 3+ concepts, allow a tolerance of 1
        #     (distractors may legitimately drop one of three+ to test a
        #     specific misconception about the dropped concept's role).
        ch = len(correct_hits)
        threshold = ch if ch <= 2 else ch - 1
        weak = []
        for o in q["options"]:
            if o.get("is_correct"):
                continue
            d_hits = self._concept_hits(o.get("text", ""), labels)
            if len(d_hits) < threshold:
                weak.append(
                    f"{o['letter']}({len(d_hits)} concept(s): {sorted(d_hits)})"
                )
        if weak:
            return False, (
                f"comparison/best-answer stem pattern requires symmetric "
                f"scope: correct answer references {len(correct_hits)} "
                f"brief concepts, but distractor(s) lag: {'; '.join(weak)}. "
                f"Rewrite under-scoped distractors to address all the "
                f"concepts the correct answer compares."
            )
        return True, "ok"


class OptionClaimGate(BaseGate):
    """Option `text` fields must be CLAIMS, not justifications.

    User's colleague flagged: "the answer choice itself explained why it
    wasn't the correct answer (before submission), written in the answer
    choice text." When option text contains reasoning words ("because",
    "since", "due to"), the LLM is folding the explanation into the
    answer text — spoon-feeding the student a matching cue rather than
    requiring genuine reasoning. The output format separates `text` from
    `explanation` for exactly this reason.

    Per-pattern exemption: comparison/contrast/case_analysis stems
    legitimately use parallel comparative connectives ("whereas", "but",
    "in contrast", "however") inside option text. These are NOT flagged.

    Only "because/since/due to" causal-justification words are flagged
    when they appear in option text.

    is_prerequisite=False.
    """
    name = "option_claim"
    is_prerequisite = False

    # Causal/reasoning markers that turn an option into a self-justification.
    # Word boundaries ensure we don't catch partial matches like "however"
    # inside "whoever".
    _REASONING_MARKERS = (
        r"\bbecause\b",
        r"\bsince\b",  # NOTE: "ever since" / "since 1985" are rare in MCQ; accept some risk
        r"\bdue to\b",
        r"\bowing to\b",
        r"\bfor this reason\b",
        r"\bthis is correct\b",
        r"\bthis is wrong\b",
        r"\bthis is incorrect\b",
        r"\bin order to\b",  # rationale connector
        r"\bso that\b",  # purpose connector
    )
    _REASONING_RE = re.compile("|".join(_REASONING_MARKERS), re.IGNORECASE)

    # Stem patterns where comparative connectives are structurally expected.
    # The reasoning markers above are NOT comparative and stay banned even
    # in these patterns. This exemption set documents the design intent.
    _COMPARATIVE_PATTERNS = frozenset({
        "comparison", "contrast_prompt", "case_analysis", "best_answer",
    })

    def check(self, q, context=None):
        violations = []
        for o in q.get("options", []):
            text = o.get("text", "")
            if not text:
                continue
            for m in self._REASONING_RE.finditer(text):
                marker = m.group(0).lower()
                role = "correct" if o.get("is_correct") else "distractor"
                violations.append(
                    f"option {o['letter']} ({role}) has reasoning marker "
                    f"'{marker}': {text[:80]}"
                )
                break  # one violation per option is enough to report
            if len(violations) >= 3:
                break
        if violations:
            return False, (
                "option text fields contain reasoning markers (because, "
                "since, due to). The text is the CLAIM; the explanation "
                "field is for justification. Move 'X because Y' content "
                "into the explanation: option text should state 'X' and "
                "the explanation should state 'because Y'. Hits: "
                + "; ".join(violations)
            )
        return True, "ok"


class OriginalityGate(BaseGate):
    """Catches verbatim copying from anchor source material.

    Prompt rule explicitly forbids "verbatim text from anchor summaries
    in stems, options, or explanations" but with no gate, compliance
    depends on LLM discretion.

    Heuristic: 5-gram (5-word) overlap between question_stem + option
    texts and the anchor_content_summaries field. Hard fail at >50%
    overlap (clear copying), soft warn at 30-50% (borderline paraphrase).
    Stop-words ignored for the n-gram comparison so common phrases like
    "in the absence of" don't inflate overlap.

    Reuses _jaccard pattern from pipeline/concept_clustering for
    consistency.

    is_prerequisite=False.
    """
    name = "originality"
    is_prerequisite = False

    # Minimal stop set — n-gram comparison wants broad coverage of
    # content tokens, so only function words are stripped here.
    _STOP = _BASE_STOP_LIGHT

    def __init__(self, hard_threshold=0.5, soft_threshold=0.3, ngram_size=5):
        self.hard_threshold = hard_threshold
        self.soft_threshold = soft_threshold
        self.ngram_size = ngram_size

    @staticmethod
    def _content_tokens(text, stop):
        if not text:
            return []
        return [w for w in re.findall(r"\b[a-zà-öø-ÿ]+\b", text.lower())
                if w not in stop]

    def _ngrams(self, tokens):
        n = self.ngram_size
        if len(tokens) < n:
            return set()
        return {tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1)}

    def check(self, q, context=None):
        # Source: try the anchor_content_summaries on the question record
        # first (set by metadata agent), fall back to context if missing.
        summaries = q.get("anchor_content_summaries", []) or []
        source_text = " ".join(s for s in summaries if s)
        if not source_text and context:
            source_text = context.get("anchor_testable_fact", "") or ""
        if not source_text:
            return True, "ok"

        source_ngrams = self._ngrams(self._content_tokens(source_text, self._STOP))
        if not source_ngrams:
            return True, "ok"

        # Concatenate question content the LLM produced
        produced_parts = [q.get("question_stem", "")]
        for o in q.get("options", []):
            produced_parts.append(o.get("text", ""))
            produced_parts.append(o.get("explanation", ""))
        produced_text = " ".join(p for p in produced_parts if p)
        produced_ngrams = self._ngrams(self._content_tokens(produced_text, self._STOP))
        if not produced_ngrams:
            return True, "ok"

        overlap = source_ngrams & produced_ngrams
        ratio = len(overlap) / len(source_ngrams)

        if ratio >= self.hard_threshold:
            sample = sorted(" ".join(g) for g in list(overlap)[:3])
            return False, (
                f"verbatim copying from anchor source: {len(overlap)} of "
                f"{len(source_ngrams)} source 5-grams reused ({ratio:.0%}). "
                f"Sample reused phrases: {sample}. Paraphrase the question "
                f"using novel sentence structure."
            )
        if ratio >= self.soft_threshold:
            try:
                print(f"    [warn] originality: {ratio:.0%} 5-gram overlap "
                      f"with source (threshold {self.soft_threshold:.0%})")
            except Exception:
                pass
        return True, "ok"


class KeywordDistributionGate(BaseGate):
    """Catches the synonym-uniqueness testwise tell.

    Empirical pattern (D7-PHY-209 audit, 5/17 = 29% of non-EXCEPT
    questions): the LLM gives the correct option rich technical
    vocabulary (e.g., "unilateral / contralateral / hemiplegia") while
    distractors use simpler vocabulary or describe different topics. A
    student who recognizes the term-equivalences picks the correct
    without engaging the concept — pure English comprehension, not
    content knowledge.

    Heuristic: count content words (4+ chars, lowercase, not stop-words)
    that appear in the correct option's text but NOT in the question
    stem and NOT in any distractor's text. If ≥3 such words, fail.
    Substring matching handles plural variants ("antagonists" matches
    "antagonist").

    EXCEPT-pattern exemption. EXCEPT/feature_listing questions
    structurally make the correct option the off-topic one — vocabulary
    divergence is by design. Skip the check when stem matches an EXCEPT
    pattern OR stem_pattern is feature_listing.

    is_prerequisite=False — collected with other content gates.
    """
    name = "keyword_distribution"
    is_prerequisite = False

    # Stop-word set: function words + modifiers + cognitive verbs that
    # the form planner injects. Centralized in pipeline/stopwords.py so
    # this set, the form planner's _VOCAB_STOP, and audit_subtle_tells.py
    # don't drift from each other.
    _STOP = _BASE_FULL

    _EXCEPT_RE = re.compile(
        r"\b(except|all of the following|are true except)\b",
        re.IGNORECASE,
    )

    # Stem patterns where vocabulary divergence is structural, not a tell.
    _EXEMPT_PATTERNS = frozenset({"feature_listing"})

    # Bloom's-identity-keyed thresholds. T1/T2 (Recognize/Understand)
    # naturally exhibit vocabulary-correct alignment because vocabulary
    # IS the test at those tiers — a T1 distractor contrasting "frontal
    # lobe" with "parietal lobe" is a valid recognition distinction, not
    # a tell. T3/T4 (Apply / Analyze-Evaluate) keep the strict threshold because
    # vocabulary divergence at those tiers IS the tell — the LLM uses
    # technical terminology as a shortcut for application reasoning.
    # Default 3 (strict) for missing/unknown tier so malformed questions
    # never bypass the gate.
    _THRESHOLD_BY_TIER = {1: 5, 2: 4, 3: 3, 4: 3}
    _DEFAULT_THRESHOLD = 3
    # Tier-keyed minimum word length. T1/T2 keep the original 5-char
    # floor because vocabulary IS the test at those tiers and short
    # technical words (agonist, frontal, lesion) need to be visible to
    # the gate. T3/T4 raise to 7 chars because at apply/evaluate
    # tiers, generic English descriptors at 5-6 chars (rapid, triad,
    # weakness) drove false positives — the genuine technical-synonym
    # tells the gate was designed to catch (hemiplegia, contralateral,
    # decussation) all comfortably exceed 7 chars.
    _MIN_WORD_LEN_BY_TIER = {1: 5, 2: 5, 3: 7, 4: 7}
    _DEFAULT_MIN_WORD_LEN = 5

    @classmethod
    def _content_words(cls, text, min_word_len=None):
        """Lowercase content words from text — N+ chars, no stop-words.

        Word length floor filters out short modifiers/descriptors. The
        floor is tier-aware via min_word_len arg (default 5 = T1/T2;
        7 at T3/T4 to filter generic English from the unique-word count).
        """
        if not text:
            return set()
        floor = min_word_len if min_word_len is not None else cls._DEFAULT_MIN_WORD_LEN
        pattern = r"\b[a-zà-öø-ÿ]{" + str(floor) + r",}\b"
        return {
            w for w in re.findall(pattern, text.lower())
            if w not in cls._STOP
        }

    @classmethod
    def _word_appears_in(cls, word, texts):
        """True if `word` appears as substring in any of the texts.

        Substring match handles plurals ("antagonist" inside "antagonists")
        and morphological variants without needing a stemmer.
        """
        for t in texts:
            if word in (t or "").lower():
                return True
        return False

    def check(self, q, context=None):
        # EXCEPT-pattern exemption
        stem = q.get("question_stem", "") or ""
        stem_pattern = q.get("stem_pattern", "")
        if stem_pattern in self._EXEMPT_PATTERNS:
            return True, "ok"
        if self._EXCEPT_RE.search(stem):
            return True, "ok"

        options = q.get("options", []) or []
        correct = next((o for o in options if o.get("is_correct")), None)
        if not correct:
            return True, "ok"
        distractors = [o for o in options if not o.get("is_correct")]
        if not distractors:
            return True, "ok"

        # Tier-aware extraction: T3/T4 use a 7-char min length to drop
        # generic English short words (rapid, triad, weakness) that
        # aren't technical-synonym tells.
        tier = q.get("difficulty_tier")
        min_len = self._MIN_WORD_LEN_BY_TIER.get(
            tier, self._DEFAULT_MIN_WORD_LEN
        )
        correct_words = self._content_words(
            correct.get("text", ""), min_word_len=min_len
        )
        if not correct_words:
            return True, "ok"

        # Build the haystack of texts where a "shared" word may appear:
        # the stem itself plus every distractor's text.
        haystack_texts = [stem] + [d.get("text", "") for d in distractors]

        unique_to_correct = sorted(
            w for w in correct_words
            if not self._word_appears_in(w, haystack_texts)
        )

        # T3/T4 architectural fix: count only canonical technical
        # vocabulary as potential tells. The gate's original concern
        # was technical-synonym divergence (hemiplegia, contralateral,
        # decussation) — not descriptive English asymmetry between
        # options, which is question content. Restrict the unique-word
        # count to (a) words in the curated domain pool and/or (b)
        # words extracted from the brief's concept descriptions, with
        # generic clinical descriptors excluded outright. T1/T2 keep
        # the broader count because vocabulary IS the test there.
        if tier in (3, 4):
            unique_to_correct = [
                w for w in unique_to_correct
                if w not in _GENERIC_DESCRIPTORS
            ]
            canonical = _canonical_terms_from_context(context)
            if canonical:
                unique_to_correct = [
                    w for w in unique_to_correct if w in canonical
                ]

        threshold = self._THRESHOLD_BY_TIER.get(tier, self._DEFAULT_THRESHOLD)
        if len(unique_to_correct) >= threshold:
            sample = unique_to_correct[:5]
            return False, (
                f"correct option contains {len(unique_to_correct)} content "
                f"word(s) not present in stem or any distractor: {sample}. "
                f"Vocabulary divergence is a testwise tell — students pick "
                f"the answer by recognizing the unique technical word. "
                f"Spread the vocabulary so distractors use the same terms "
                f"in contexts that make them wrong."
            )
        return True, "ok"


class StemKeywordDistributionGate(BaseGate):
    """Catches the stem-keyword-in-correct-only tell.

    Empirical pattern (D7-PHY-209 phase 2 v4 audit, 1/18 questions):
    a stem mentioning specific terms (e.g., 'ipsilateral / lesion / loss')
    has only the correct option repeating those terms — distractors
    avoid them. Students match the stem's keywords to the option that
    repeats them, picking correct without engaging the concept.

    Tighter than KeywordDistributionGate: this gate looks specifically
    at words that BOTH appear in the stem AND appear in the correct
    option, then requires those to also appear in at least one distractor.
    KeywordDistributionGate catches all unique-to-correct vocabulary;
    this catches the subset that's specifically a stem-keyword-match tell.

    EXCEPT-pattern exemption (same as KeywordDistributionGate): vocabulary
    divergence is structural in EXCEPT/feature_listing.

    is_prerequisite=False — collected with other content gates.
    """
    name = "stem_keyword_distribution"
    is_prerequisite = False

    # Same stop-word and content-word logic as KeywordDistributionGate so
    # the two gates "see" the same vocabulary. Drift would have one gate
    # endorsing what the other rejects.
    _STOP = KeywordDistributionGate._STOP
    # Tier-aware min word length mirrors KeywordDistributionGate so the
    # two gates "see" the same vocabulary at each tier.
    _MIN_WORD_LEN_BY_TIER = {1: 5, 2: 5, 3: 7, 4: 7}
    _DEFAULT_MIN_WORD_LEN = 5
    # Bloom's-identity-keyed thresholds. T1/T2 (Recognize/Understand)
    # legitimately echo stem keywords because vocabulary IS the test —
    # a recognition stem asking about "frontal lobe" naturally has
    # "frontal lobe" in the correct answer. T3/T4 keep tighter
    # (threshold 2) because stem-keyword echo at apply/evaluate is
    # the substituting-recognition-for-reasoning tell. Default 2
    # (strict) for missing/unknown tier.
    _THRESHOLD_BY_TIER = {1: 3, 2: 3, 3: 2, 4: 2}
    _DEFAULT_THRESHOLD = 2

    _EXCEPT_RE = KeywordDistributionGate._EXCEPT_RE
    _EXEMPT_PATTERNS = KeywordDistributionGate._EXEMPT_PATTERNS

    @classmethod
    def _content_words(cls, text, min_word_len=None):
        if not text:
            return set()
        floor = min_word_len if min_word_len is not None else cls._DEFAULT_MIN_WORD_LEN
        pattern = r"\b[a-zà-öø-ÿ]{" + str(floor) + r",}\b"
        return {
            w for w in re.findall(pattern, text.lower())
            if w not in cls._STOP
        }

    @staticmethod
    def _word_appears_in(word, texts):
        for t in texts:
            if word in (t or "").lower():
                return True
        return False

    def check(self, q, context=None):
        stem = q.get("question_stem", "") or ""
        stem_pattern = q.get("stem_pattern", "")
        if stem_pattern in self._EXEMPT_PATTERNS:
            return True, "ok"
        if self._EXCEPT_RE.search(stem):
            return True, "ok"

        options = q.get("options", []) or []
        correct = next((o for o in options if o.get("is_correct")), None)
        if not correct:
            return True, "ok"
        distractors = [o for o in options if not o.get("is_correct")]
        if not distractors:
            return True, "ok"

        # Find words that the stem and correct share — these are the
        # "stem keywords used by the correct option." Tier-aware min
        # length filters generic English at T3/T4.
        tier = q.get("difficulty_tier")
        min_len = self._MIN_WORD_LEN_BY_TIER.get(
            tier, self._DEFAULT_MIN_WORD_LEN
        )
        stem_words = self._content_words(stem, min_word_len=min_len)
        if not stem_words:
            return True, "ok"
        correct_text_lower = (correct.get("text", "") or "").lower()
        stem_keywords_in_correct = {
            w for w in stem_words if w in correct_text_lower
        }
        if not stem_keywords_in_correct:
            return True, "ok"

        # Of those shared stem-and-correct keywords, find ones missing
        # from every distractor.
        distractor_texts = [d.get("text", "") for d in distractors]
        leaked = sorted(
            w for w in stem_keywords_in_correct
            if not self._word_appears_in(w, distractor_texts)
        )

        # T3/T4 architectural fix (mirrors KeywordDistributionGate):
        # exclude generic English descriptors AND restrict to canonical
        # vocabulary so the gate counts technical-synonym tells only,
        # not descriptive-English asymmetry between options.
        if tier in (3, 4):
            leaked = [w for w in leaked if w not in _GENERIC_DESCRIPTORS]
            canonical = _canonical_terms_from_context(context)
            if canonical:
                leaked = [w for w in leaked if w in canonical]

        threshold = self._THRESHOLD_BY_TIER.get(tier, self._DEFAULT_THRESHOLD)
        if len(leaked) >= threshold:
            return False, (
                f"correct option repeats {len(leaked)} stem keyword(s) that "
                f"NO distractor uses: {leaked[:5]}. Students match the "
                f"stem's keywords to the option containing them, picking "
                f"correct without engaging the concept. Distribute these "
                f"terms across distractors (in contexts that make the "
                f"distractor wrong) so keyword-matching alone doesn't "
                f"identify the answer."
            )
        return True, "ok"


class TopicRealmGate(BaseGate):
    """Catches the topic-isolation tell: only the correct option engages
    the topic realm while distractors stray off-topic.

    The architectural concern: at apply/evaluate tiers, all four options
    must be plausible-but-different claims about the SAME topic. A pattern
    where 3 distractors are tangentially-related-but-off-topic and only
    correct is squarely on-topic tests TOPIC RECOGNITION, not the
    cognitive level the question is meant to assess. The student picks
    correct because it's the only option using realm vocabulary, not
    because they reasoned about the application/synthesis.

    Topic realm = (concept-description content words) ∪ (stem content
    words). Anchor-specific. The curated domain pool is intentionally
    NOT in the realm — that's too broad ("any biopsychology word"); the
    realm needs to be "what THIS specific question is testing."

    For each option: overlap = |content_words ∩ realm| / |content_words|.
    Fire when correct's overlap exceeds the mean of distractors' overlap
    by more than a tier-keyed gap threshold.

    Tier thresholds: stricter at T3/T4 because application / analyze-evaluate
    demands all 4 options engage the same concept's realm. T1/T2 looser
    because recognition-style distractors can legitimately be other
    concepts (e.g., "frontal lobe" vs "parietal lobe" at T1 is a valid
    recognition distinction even though their realms only partly overlap).

    is_prerequisite=False — collected with other content gates.

    Replaces KeywordDistributionGate's role at the apply/evaluate tiers.
    KW gate flagged on word-count uniqueness (correct uses N words found
    nowhere else); this gate flags on topic-realm engagement asymmetry.
    """
    name = "topic_realm"
    is_prerequisite = False

    _STOP = _BASE_FULL
    _MIN_WORD_LEN = 5
    # Below this floor on either side, we lack enough signal to compare
    # overlaps reliably — bypass to avoid false positives.
    _MIN_REALM_SIZE = 3
    _MIN_OPTION_WORDS = 5

    # Tier-keyed gap thresholds. The gap between correct's realm overlap
    # and the mean of distractor overlaps must exceed this to trigger.
    # Calibrated on the assumption that legitimate distractors at T3/T4
    # engage the same concept's realm as correct (gap ~0.0-0.2), while
    # off-topic distractors create a much wider gap (0.4+).
    _GAP_BY_TIER = {1: 0.50, 2: 0.40, 3: 0.30, 4: 0.25}
    _DEFAULT_GAP = 0.25  # strictest by default — never bypass on missing tier

    # Absolute floor on distractor engagement. The gate only fires when
    # distractor mean overlap is BELOW this — i.e., distractors are
    # clearly absent from the topic realm. The user's framing was
    # "only one answer uses keywords in the topic realm." If distractor
    # mean is at/above 0.5, distractors ARE in the realm (just at lower
    # density than an exceptionally on-topic correct option). Empirical:
    # D7-PHY-209 H-05 fired with distractor mean 0.52 but all 4 options
    # used hemiplegia/pyramidal/decussation — pure false positive on a
    # clean question. D7-PHY-058 H-03 fired with distractor mean 0.45
    # and distractors A/B introduced amnesia/depression — true positive.
    # Floor of 0.5 separates the two cases empirically.
    _MAX_DISTRACTOR_OVERLAP_FOR_FIRE = 0.5

    _EXCEPT_RE = re.compile(
        r"\b(except|all of the following|are true except)\b",
        re.IGNORECASE,
    )
    _EXEMPT_PATTERNS = frozenset({"feature_listing"})

    @classmethod
    def _content_words(cls, text):
        if not text:
            return set()
        pattern = r"\b[a-zà-öø-ÿ]{" + str(cls._MIN_WORD_LEN) + r",}\b"
        return {
            w for w in re.findall(pattern, text.lower())
            if w not in cls._STOP
        }

    @classmethod
    def _build_realm(cls, context, stem):
        """Topic realm for this question.

        Pulls from concept-description vocabulary (anchor-specific
        canonical terms) and the stem content words (this question's
        actual subject). Domain pool intentionally excluded — too broad.
        """
        realm = set()
        if context:
            for term in context.get("brief_concept_terms", []) or []:
                if term:
                    realm.add(term.lower())
        realm |= cls._content_words(stem)
        return realm

    def check(self, q, context=None):
        # EXCEPT-pattern exemption: correct is intentionally the off-
        # topic one in those questions, so realm engagement asymmetry
        # is structural, not a tell.
        stem = q.get("question_stem", "") or ""
        stem_pattern = q.get("stem_pattern", "")
        if stem_pattern in self._EXEMPT_PATTERNS:
            return True, "ok"
        if self._EXCEPT_RE.search(stem):
            return True, "ok"

        options = q.get("options", []) or []
        correct = next((o for o in options if o.get("is_correct")), None)
        if not correct:
            return True, "ok"
        distractors = [o for o in options if not o.get("is_correct")]
        if len(distractors) < 3:
            return True, "ok"

        realm = self._build_realm(context, stem)
        if len(realm) < self._MIN_REALM_SIZE:
            # Insufficient realm signal — cannot reliably evaluate.
            return True, "ok"

        correct_words = self._content_words(correct.get("text", ""))
        if len(correct_words) < self._MIN_OPTION_WORDS:
            # Correct option too sparse to compare. Skip.
            return True, "ok"
        correct_overlap = (
            len(correct_words & realm) / len(correct_words)
        )

        # Per-distractor overlap; skip distractors with too few content
        # words to keep the mean from being noisy.
        d_overlaps = []
        for d in distractors:
            d_words = self._content_words(d.get("text", ""))
            if len(d_words) < self._MIN_OPTION_WORDS:
                continue
            d_overlaps.append(len(d_words & realm) / len(d_words))
        if len(d_overlaps) < 3:
            return True, "ok"

        mean_d_overlap = sum(d_overlaps) / len(d_overlaps)
        gap = correct_overlap - mean_d_overlap

        # Absolute distractor-engagement floor. If distractors clearly
        # ARE in the realm (mean ≥ 0.5), the question doesn't have the
        # "only one answer in topic realm" pattern even if correct is
        # exceptionally dense. Bypass to avoid false positives.
        if mean_d_overlap >= self._MAX_DISTRACTOR_OVERLAP_FOR_FIRE:
            return True, "ok"

        tier = q.get("difficulty_tier")
        threshold = self._GAP_BY_TIER.get(tier, self._DEFAULT_GAP)
        if gap >= threshold:
            return False, (
                f"correct option engages the topic realm "
                f"({correct_overlap:.2f}) much more than distractors do "
                f"on average ({mean_d_overlap:.2f}); gap {gap:.2f} >= "
                f"threshold {threshold:.2f}. Distractors should be "
                f"plausible-but-incorrect claims about the SAME topic, "
                f"not off-topic options. Rewrite distractors to engage "
                f"the same concept vocabulary as the correct option, "
                f"applied incorrectly."
            )
        return True, "ok"


class LateralityIntegrityGate(BaseGate):
    """Catches stem-eliminable laterality inversions in distractors.

    Pattern (from D7-PHY-076 audit, post-Layer-A):
        Stem: 'bilateral hippocampal damage'
        Distractor: 'unilateral hippocampal injury'

    The student rejects this by reading the stem alone — no domain
    knowledge needed. This is a quality failure (the misconception
    target was meant to be a content distinction, not a re-readable
    surface fact).

    Heuristic: if stem asserts a laterality EXCLUSIVELY (contains
    'bilateral' but not 'unilateral', or vice versa), distractors
    must not assert the inverted laterality. If the stem references
    both lateralities (e.g., comparing them) or neither, bypass.

    Narrow by design — catches only the canonical bilateral↔unilateral
    flip. Semantic inversions like 'destroyed both' → 'surviving
    contralateral one' need the Sonnet audit (Layer D).
    """
    name = "laterality_integrity"
    is_prerequisite = False

    _BILATERAL_RE = re.compile(r"\bbilateral(?:ly)?\b", re.IGNORECASE)
    _UNILATERAL_RE = re.compile(r"\bunilateral(?:ly)?\b", re.IGNORECASE)

    def check(self, q, context=None):
        stem = q.get("question_stem", "") or ""
        stem_has_bi = bool(self._BILATERAL_RE.search(stem))
        stem_has_uni = bool(self._UNILATERAL_RE.search(stem))

        # Bypass when stem has neither or both — no exclusive laterality
        # fact to invert.
        if stem_has_bi == stem_has_uni:
            return True, "ok"

        if stem_has_bi:
            forbidden_re = self._UNILATERAL_RE
            stated, inverted = "bilateral", "unilateral"
        else:
            forbidden_re = self._BILATERAL_RE
            stated, inverted = "unilateral", "bilateral"

        for opt in q.get("options", []) or []:
            if opt.get("is_correct"):
                continue
            text = opt.get("text", "") or ""
            if forbidden_re.search(text):
                letter = opt.get("letter", "?")
                return False, (
                    f"distractor {letter} asserts '{inverted}' laterality "
                    f"while the stem states '{stated}' damage/injury. A "
                    f"student rejects this by reading the stem alone (no "
                    f"content knowledge required) — quality failure. "
                    f"Make the distractor wrong via the underlying "
                    f"mechanism the question tests, not by inverting "
                    f"a stem-stated laterality fact. Hit: option "
                    f"{letter}: {text[:140]}"
                )
        return True, "ok"


class UniversalDenialGate(BaseGate):
    """Catches universal-quantifier denials that contradict stem-stated
    counterexamples.

    Pattern (from D7-PHY-076 audit, multiple T4 cases):
        Stem: 'still recalls his wedding from a decade earlier'
        Distractor: 'Retrograde amnesia erases ALL pre-injury memories'

        Stem: 'vividly recounts her wedding fifteen years ago'
        Distractor: 'hippocampal injury erases all remote declarative
                     memories'

    Heuristic: stem contains a preservation marker (still / intact /
    preserved / spared / vividly / can still / recalls / retains) AND
    a distractor contains a universal quantifier (all / every / no /
    none / never / regardless of / completely / entirely / globally)
    paired with a denial verb (erase / lose / abolish / impair / wipe
    / destroy / forget) within ~80 chars.

    To reduce false positives, require ≥1 shared content word (≥6 chars,
    not in stopword set) between the stem and the universal-denial
    distractor — without overlap, the universal could be about a
    different category than what the stem preserves.

    Narrow heuristic — Sonnet audit (Layer D) catches the broader
    semantic versions this regex misses.
    """
    name = "universal_denial"
    is_prerequisite = False

    _PRESERVATION_RE = re.compile(
        r"\b(still|intact|preserved?|spared?|vividly|"
        r"can\s+still|demonstrates?\s+(intact|preserved|normal)|"
        r"shows?\s+(intact|preserved|normal)|"
        r"recalls?|recounts?|retains?|remains?\s+intact)\b",
        re.IGNORECASE,
    )

    # Universal-quantifier and denial-verb patterns. Both must appear
    # in the distractor within ~80 chars of each other in EITHER order
    # (canonical audit cases include both orderings:
    #   "ALL pre-injury memories ERASED" — quantifier-first
    #   "ERASING ALL remote declarative memories" — denial-verb-first
    # ).
    _UNIVERSAL_RE = re.compile(
        r"\b(all|every|no|none|never|regardless\s+of|"
        r"completely|entirely|globally?)\b",
        re.IGNORECASE,
    )
    _DENIAL_VERB_RE = re.compile(
        r"\b(eras(?:e[ds]?|ing)|los[est]+|abolish(?:es|ed|ing)?|"
        r"impair(?:s|ed|ing)?|loss|wipe[ds]?|wiping|"
        r"destroy(?:s|ed|ing)?|forgotten|forgets?|forget|"
        r"abolishing)\b",
        re.IGNORECASE,
    )
    _MAX_PROXIMITY_CHARS = 80

    _STOP = _BASE_FULL
    _MIN_WORD_LEN = 6

    @classmethod
    def _content_words(cls, text):
        if not text:
            return set()
        pattern = r"\b[a-zà-öø-ÿ]{" + str(cls._MIN_WORD_LEN) + r",}\b"
        return {
            w for w in re.findall(pattern, text.lower())
            if w not in cls._STOP
        }

    def check(self, q, context=None):
        stem = q.get("question_stem", "") or ""
        if not self._PRESERVATION_RE.search(stem):
            return True, "ok"

        stem_words = self._content_words(stem)
        if not stem_words:
            return True, "ok"

        for opt in q.get("options", []) or []:
            if opt.get("is_correct"):
                continue
            text = opt.get("text", "") or ""

            # Find universal quantifier + denial-verb within proximity
            # in either order. Both must be present and within
            # _MAX_PROXIMITY_CHARS of each other to count as a single
            # universal-denial claim (rather than two unrelated phrases).
            uni_matches = list(self._UNIVERSAL_RE.finditer(text))
            den_matches = list(self._DENIAL_VERB_RE.finditer(text))
            paired = None
            for u in uni_matches:
                for d in den_matches:
                    # Distance between match starts (order-agnostic)
                    distance = abs(u.start() - d.start())
                    if distance <= self._MAX_PROXIMITY_CHARS:
                        # Use whichever match starts first as the
                        # representative span for the error message
                        first = u if u.start() < d.start() else d
                        last = d if u.start() < d.start() else u
                        paired = (first, last)
                        break
                if paired:
                    break
            if not paired:
                continue

            distractor_words = self._content_words(text)
            shared = stem_words & distractor_words
            if not shared:
                continue
            letter = opt.get("letter", "?")
            first, last = paired
            span = text[first.start():last.end()]
            return False, (
                f"distractor {letter} uses a universal-quantifier denial "
                f"('{span[:60]}...') paired with category words "
                f"({sorted(shared)[:3]}) that the stem describes as "
                f"preserved/intact/spared. A student rejects this by "
                f"reading the stem alone (no content knowledge required) "
                f"— quality failure. Reframe the distractor to be wrong "
                f"via the underlying mechanism, not by universally "
                f"denying a category the stem cites a counterexample for."
            )
        return True, "ok"


class StemEliminableDistractorGate(BaseGate):
    """Layer-C semantic check for stem-eliminable distractors.

    Promotes the offline Sonnet audit (scripts/audit_stem_contradictions)
    to a generation-time gate. Catches semantic versions of the
    contradiction pattern that the regex layers (LateralityIntegrity,
    UniversalDenial) miss — paraphrase contradictions, named
    counterexamples, etc. Sits after the cheap regex gates so questions
    that fail those don't pay the Sonnet cost.

    Wiring: requires `client` (anthropic.AsyncAnthropic) and
    `audit_token_acc` (list) in gate_context. Without either, the gate
    bypasses with ok=True (defensive: never stricter from a missing
    dependency — Layer D offline audit still runs as a backstop).

    Cost: ~$0.004/question on questions that reach this gate.
    """
    name = "stem_eliminable_distractor"
    is_prerequisite = False

    MODEL_ID = "claude-sonnet-4-6"

    async def check(self, q, context=None):
        if not context:
            return True, "ok"
        client = context.get("client")
        if client is None:
            return True, "ok"

        # Lazy import keeps the audit script the single source of truth
        # for the prompt + JSON parser without forcing a top-of-file
        # dependency on scripts/.
        from scripts.audit_stem_contradictions import (
            PROMPT, parse_response,
        )

        stem = q.get("question_stem", "") or ""
        options = q.get("options", []) or []
        if not stem or not options:
            return True, "ok"

        # ── P1 dispatch: resolve the policy cell, branch on action ──
        # The matrix at pipeline/distractor_policy.py keys on
        # (tier, source_type, stem_pattern) per the P-1 finding that
        # patterns are the empirical content-type signal. Skip cells
        # short-circuit before the Sonnet call (cost win + correctness:
        # the cell explicitly says "this question type's pedagogy
        # tolerates apparent stem-contradiction"). Permissive cells
        # call Sonnet but with a relaxed prompt that recognizes
        # judgment-error and synthesis-error distractors as legitimate.
        from .distractor_policy import resolve
        # Pick the misconception_type of the first non-correct option
        # as a representative for the question's distractor mix.
        # Per-distractor decisions are P3 work.
        misc_type = None
        for o in options:
            if not o.get("is_correct"):
                misc_type = o.get("misconception_type")
                if misc_type:
                    break
        cell = resolve(
            tier=q.get("difficulty_tier"),
            domain_code=context.get("domain_code"),
            pedagogical_content_type=context.get("pedagogical_content_type"),
            misconception_type=misc_type,
            source_type=context.get("source_type"),
            stem_pattern=context.get("stem_pattern"),
        )

        # Skip cells: pedagogy makes "stem-eliminable" tautological
        # (e.g., concept-definition stems must restate the criterion).
        # Don't burn a Sonnet call on these.
        if cell.gate_action == "skip":
            return True, f"policy=skip:{cell.note}"

        options_block = "\n".join(
            f"  {o.get('letter', '?')} "
            f"{'[CORRECT]' if o.get('is_correct') else '[distractor]'}: "
            f"{o.get('text', '')}"
            for o in options
        )

        # Build a cell-aware prompt. The audit script's PROMPT now
        # accepts a `mode_addendum` placeholder that varies by
        # gate_action: strict gets the full content-knowledge threshold
        # (default); permissive adds language recognizing judgment- and
        # synthesis-errors as legitimate distractors at evaluate-tier.
        mode_addendum = _build_mode_addendum(cell)
        prompt = PROMPT.format(
            stem=stem,
            options_block=options_block,
            mode_addendum=mode_addendum,
        )

        try:
            response = await client.messages.create(
                model=self.MODEL_ID,
                max_tokens=2048,
                temperature=0,
                messages=[{"role": "user", "content": prompt}],
            )
        except Exception:
            return True, "ok"

        acc = context.get("audit_token_acc")
        if isinstance(acc, list):
            acc.append((response.usage.input_tokens,
                        response.usage.output_tokens))

        text = response.content[0].text if response.content else ""
        parsed = parse_response(text)
        if not parsed:
            return True, "ok"

        # ── 3-class consumption ────────────────────────────────
        # New audit prompt outputs `classifications` per distractor:
        #   english_gap → quality failure; flag
        #   content_gap → testing content knowledge; PASS (do not flag)
        #   clean       → no contradiction; PASS
        # Backward-compat: also derive flagged from `flagged_distractors`
        # if Sonnet emitted the legacy shape.
        classifications = parsed.get("classifications") or []
        if classifications:
            from .quality_taxonomy import ENGLISH_GAP
            english_gaps = [
                c for c in classifications
                if c.get("class") == ENGLISH_GAP
            ]
            if not english_gaps:
                return True, "ok"
            flagged_for_msg = [
                {
                    "letter": c.get("letter", "?"),
                    "contradicted_stem_fact": c.get("contradicted_stem_fact", ""),
                    "explanation": c.get("explanation", ""),
                }
                for c in english_gaps
            ]
        else:
            # Legacy path — old prompt format
            flagged_for_msg = parsed.get("flagged_distractors") or []
            if not flagged_for_msg:
                return True, "ok"

        parts = []
        for f in flagged_for_msg:
            letter = f.get("letter", "?")
            fact = f.get("contradicted_stem_fact", "")
            why = f.get("explanation", "")
            parts.append(
                f"distractor {letter}: english_gap — claim contradicted by "
                f"stem fact '{fact}'. {why}. Rewrite so the distractor "
                f"requires CONTENT knowledge to reject, not stem-reading."
            )
        # Prepend strategy tag so build_correction_prompt's guidance
        # branches on the cell's correction_strategy. Tag syntax matches
        # _STEM_ELIMINABLE_STRATEGY_RE in pipeline.prompts.
        body = " | ".join(parts)
        return False, f"[strategy={cell.correction_strategy}] {body}"


def _build_mode_addendum(cell) -> str:
    """Build a calibration hint for the 3-class audit prompt based on
    the cell's gate_action and note.

    Under the 3-class audit, the gate doesn't override Sonnet's
    decision — Sonnet classifies each distractor and we flag only
    `english_gap`. The mode_addendum biases Sonnet toward correct
    classification at cells where the structural pattern is known to
    invite false-positive english_gap calls.

    `permissive` and `audit-with-hint` cells emit calibration text
    explaining what kind of distractors are typical at this cell,
    helping Sonnet pick content_gap when content knowledge is the
    test medium. `strict` cells emit no addendum (default).
    `framework_aware` adds ethics-specific calibration. `skip` is
    handled before this function is called.
    """
    if cell.gate_action in ("permissive", "audit"):
        # Default calibration for cells where contradictions tend to
        # be content_gap. The cell's note (if any) is appended for
        # cell-specific calibration.
        base = (
            "\n\nCELL CALIBRATION HINT: this question is at a tier/pattern "
            "where the structural form often produces distractors that "
            "lexically appear to contradict the stem but actually require "
            "concept knowledge to reject. Be especially careful before "
            "classifying as english_gap; consider whether recognizing the "
            "contradiction requires invoking what a technical term means "
            "(e.g., understanding what 'intrinsic activity' or 'agonist' "
            "imply at the receptor level). When in doubt between english_"
            "gap and content_gap, prefer content_gap."
        )
        if cell.note:
            base += f" Cell context: {cell.note}."
        return base
    if cell.gate_action == "framework_aware":
        base = (
            "\n\nFRAMEWORK CALIBRATION HINT: this question tests application "
            "of an ethical or legal framework. Distractors that apply the "
            "WRONG framework correctly are CONTENT_GAP, not english_gap. "
            "Distractors that mis-state a framework's content (e.g., "
            "misquote what an APA Standard says) are CONTENT_GAP — "
            "rejecting requires knowing the framework's actual content. "
            "Only classify as english_gap when the contradiction is with "
            "a specific stated fact in the stem (e.g., a named subject's "
            "documented action) AND independent of framework content."
        )
        if cell.note:
            base += f" Cell context: {cell.note}."
        return base
    # strict: no addendum; default audit behavior is appropriate
    return ""


# ── Gate pipeline convenience ─────────────────────────────────

def create_gate_pipeline():
    """Returns the ordered list of validation gates.

    KeywordDistributionGate is intentionally NOT registered here. Its
    target (synonym uniqueness in correct) was an over-broad heuristic
    that fired on legitimate vocabulary variety. TopicRealmGate replaces
    it with a Jaccard-based realm-engagement check that targets the
    actual concern: distractors straying off-topic. The KW gate class
    remains in this module for future revival or reference.
    """
    return [
        StructureGate(),
        ContentQualityGate(),
        ConsistencyGate(),
        AnchorGroundingGate(),
        AttributionGate(),
        OptionLengthBalanceGate(),
        BloomsCognitiveLevelGate(),
        RememberIdentityGate(),
        UnderstandIdentityGate(),
        ApplyIdentityGate(),
        EvaluateIdentityGate(),
        DomainExpertiseGate(),
        ScopeMatchGate(),
        OptionClaimGate(),
        OriginalityGate(),
        TopicRealmGate(),
        StemKeywordDistributionGate(),
        LateralityIntegrityGate(),
        UniversalDenialGate(),
        StemEliminableDistractorGate(),
        DistractorMixGate(),
        UniquenessGate(),
    ]
