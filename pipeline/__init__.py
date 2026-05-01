"""
Quiz Question Pipeline — Agent Architecture

Phases:
  1. Preparation  — hardcoded agents (instant, $0)
  2. Generation   — focused LLM call (smaller prompt/output)
  3. Assembly     — hardcoded merge (instant, $0)
  4. Validation   — gate checks (instant, $0)
  5. Smart Retry  — re-run Phase 2 only, prep data cached
"""

import re
import uuid

# ── Shared Constants ──────────────────────────────────────────

PASSEPPP_UUID_NS = uuid.UUID("e7a1c3d4-b5f6-4890-abcd-ef1234567890")

# Re-export from shared_constants (single source of truth for domain names)
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from shared_constants import DOMAIN_CODES, CODE_TO_ID, DOMAIN_NAMES  # noqa: E402

DIFFICULTY_LABELS = {1: "easy", 2: "medium", 3: "hard", 4: "expert"}
DIFFICULTY_LETTERS = {1: "E", 2: "M", 3: "H", 4: "X"}

# Asymmetric Bloom's mixing: primary = design target, secondary = permitted ceiling/floor.
# Prompts optimize for primary; secondary is incidental, not a co-equal target.
BLOOMS_BY_TIER = {
    1: ("remember", "understand"),      # primary: remember
    2: ("understand", "apply"),         # primary: understand
    3: ("apply", "analyze"),            # primary: apply
    4: ("evaluate", "analyze"),         # primary: evaluate, analyze is floor
}

BLOOMS_PRIMARY = {1: "remember", 2: "understand", 3: "apply", 4: "evaluate"}

DISTRACTOR_MIX = {
    1: {"L1": 1, "L2": 1, "L3": 1},
    2: {"L1": 1, "L2": 1, "L3": 1},
    3: {"L2": 1, "L3": 1, "L4": 1},
    4: {"L3": 1, "L4": 2},
}

# Source type per (tier, pattern): anchor_grounded = question built from the
# verbatim research anchor; integrated = requires both anchor AND passage.
# Enrichment uses anchor_grounded + integrated only (fundamentals owns passage-only).
# Gradient: anchor-dominant at T1 (recall), integrated-dominant at T3-T4 (apply/evaluate).
STEM_SOURCE_TYPE = {
    (1, "direct_definition"):      "anchor_grounded",
    (1, "concept_identification"): "anchor_grounded",
    (1, "fact_recognition"):       "anchor_grounded",
    (1, "true_false_which"):       "anchor_grounded",
    (1, "feature_listing"):        "integrated",
    (2, "comparison"):             "integrated",
    (2, "example_recognition"):    "anchor_grounded",
    (2, "simple_application"):     "integrated",
    (2, "paraphrase"):             "anchor_grounded",
    (2, "categorization"):         "anchor_grounded",
    (3, "clinical_vignette"):      "integrated",
    (3, "scenario_completion"):    "integrated",
    (3, "error_identification"):   "anchor_grounded",
    (3, "case_analysis"):          "integrated",
    (3, "mechanism_application"):  "anchor_grounded",
    (4, "contrast_prompt"):        "integrated",
    (4, "best_answer"):            "integrated",
    (4, "subtle_error"):           "integrated",
    (4, "competing_evidence"):     "integrated",
    (4, "integration"):            "integrated",
}


def get_source_type(tier, pattern_name):
    """Get source type for a tier + pattern combination."""
    return STEM_SOURCE_TYPE[(tier, pattern_name)]

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
         "Present a brief scenario (1-3 sentences); student identifies the concept being demonstrated"),
        ("paraphrase",
         "Name a concept; student selects the most accurate restatement in different words"),
        ("categorization",
         "Ask the student to classify an item within a real taxonomy or classification system"),
    ],
    3: [
        # T3 (Apply) — every stem MUST present a novel scenario the
        # student applies a concept to. The error_identification pattern
        # was removed because empirically it produced "find the wrong
        # claim" exercises where distractors became alternative diagnoses
        # rather than alternative applications of the tested concept,
        # violating Apply identity.
        ("clinical_vignette",
         "Clinical vignette with named professional/client and specific setting. "
         "The question MUST ask the student to PREDICT the most likely outcome "
         "OR SELECT the next clinical step / intervention. Each option's text "
         "MUST be an outcome statement or action (e.g., 'Receptor activation "
         "will decrease at D2 sites'), NOT a concept label (e.g., 'Compound X "
         "is an antagonist'). Bare-label answers fail validation."),
        ("scenario_completion",
         "Professional scenario in an applied context. The question MUST ask "
         "the student to PREDICT the next step, outcome, or consequence. Each "
         "option's text MUST be a prediction or action statement, NOT a "
         "concept label or definition. The student should APPLY the concept "
         "to derive the outcome, not recognize the concept's name."),
        ("case_analysis",
         "Analyze a case for the underlying mechanism or causal explanation "
         "(WHY, not WHAT). The correct answer must articulate the mechanism "
         "(e.g., 'Lacks intrinsic activity, whereas Y produces partial "
         "activation') — comparative or mechanistic structure required."),
        ("mechanism_application",
         "Apply a named principle to a novel situation and PREDICT the "
         "outcome. The correct answer must be the predicted outcome itself "
         "(e.g., 'Synaptic transmission will be enhanced'), not the name of "
         "the principle being applied."),
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

def get_stem_pattern(tier, variant_num):
    """Get (pattern_name, pattern_desc) for a tier and variant number (1-5)."""
    patterns = STEM_PATTERNS[tier]
    return patterns[(variant_num - 1) % len(patterns)]

CORRECT_POSITIONS = [
    "B", "C", "A", "D", "C",   # topic-group 1
    "A", "D", "B", "C", "A",   # topic-group 2
    "D", "A", "C", "B", "D",   # topic-group 3
    "C", "B", "D", "A", "B",   # topic-group 4
]  # 20 positions: exactly 5A, 5B, 5C, 5D per full cycle

VALID_MISCONCEPTION_TYPES = {
    "similar_name", "similar_property", "similar_store",
    "opposite_direction", "overgeneralization", "partial_understanding",
}

# ── Eponym Whitelist ──────────────────────────────────────────
# Personal names that are inseparable from the concept itself
# (Piaget's stages, Cannon-Bard theory, Ribot's law, etc.) and are
# therefore exempt from the no-researcher-attribution rule. Used by:
#   • prompts.py (positive guidance: "you may use these names")
#   • InputSanitizerAgent (do not strip these from anchor data)
#   • AttributionGate (skip whitelisted names when flagging stems/options)
#
# DESIGN POLICY (intentional tradeoffs):
# 1. Name-based, NOT person-based. "Berry" is whitelisted because John W.
#    Berry's acculturation model is heavily cited in EPPP. If a different
#    Berry (e.g., a textbook author) appears, they're also exempted. This
#    over-permissiveness is accepted in exchange for true-positive eponym
#    handling — disambiguating people would require context-aware NLP that
#    isn't justified by the small false-positive surface.
# 2. Common surnames (Sue, James, Singer, Bem, Holland, Cross, Beck) are
#    whitelisted only because they are heavily-cited concept eponyms in
#    EPPP literature (Sue's microaggressions, Holland's RIASEC, etc.).
#    Their occurrence as first names or character names in clinical
#    vignettes does NOT trigger the AttributionGate (no citation pattern
#    around them), so there's no false-positive risk in vignettes.
# 3. Sanitizer keeps eponym, drops year. So "Piaget (1936)" → "Piaget";
#    non-whitelisted "Smith (2010)" → "" (whole citation removed).
# 4. Multi-author handling is all-or-nothing: "Bandura and Walters (1963)"
#    strips entirely because Walters isn't whitelisted. Predictable but
#    sometimes loses a legitimate eponym reference.
# 5. To add a name: edit this set, run `python -m unittest discover -s tests`
#    (test_eponym_whitelist.py verifies required names are present).

EPONYM_WHITELIST = frozenset({
    # Developmental
    "Piaget", "Vygotsky", "Bowlby", "Ainsworth", "Erikson", "Kohlberg",
    "Kübler-Ross", "Bronfenbrenner", "Baumrind", "Baltes",
    "Thomas", "Chess",  # temperament types (Thomas & Chess)
    # Conditioning / behaviorism
    "Pavlov", "Pavlovian", "Skinner", "Skinnerian", "Thorndike",
    "Tolman", "Honzik", "Premack",
    "Watson", "Rescorla", "Herrnstein", "Kamin", "Terrace",
    # Psychoanalytic / humanistic
    "Freud", "Freudian", "Jung", "Jungian", "Adler", "Adlerian",
    "Rogers", "Maslow", "Horney",
    # CBT / clinical
    "Beck", "Ellis", "Bandura",
    "Jacobson",  # clinical significance, behavioral activation
    # Family / clinical models
    "Minuchin", "Lewinsohn", "Lazarus", "Caplan", "Moffitt", "Troiden",
    "Epston",  # narrative therapy (with M. White)
    "Garber",  # hopelessness theory of depression
    "Truax",   # clinical significance / outcome research
    "Elkind",  # adolescent egocentrism, imaginary audience, personal fable
    "Sadker",  # gender bias in education (Myra & David Sadker)
    "Barker",  # ecological psychology, behavior settings (Roger Barker)
    # Emotion / motivation
    "Cannon", "Bard", "James", "Lange", "Yerkes", "Dodson", "Schachter",
    "Singer", "Selye", "Ekman",
    # Memory / cognition
    "Ribot", "Ebbinghaus", "Hebb", "Hebbian", "Atkinson", "Shiffrin",
    "Tulving", "Baddeley", "Kahneman", "Tversky", "Simon",
    "Brown", "Peterson",  # Brown-Peterson task
    # Linguistics / culture
    "Sapir", "Whorf", "Chomsky",
    # Social
    "Asch", "Milgram", "Zimbardo", "Festinger", "Heider", "Kelley",
    "Allport", "Cattell", "Eysenck",
    "Sherif", "Janis", "Latane", "Latané", "Darley", "Berscheid",
    "Aronson", "Hovland", "Fiske", "Correll", "Rosenthal", "Ajzen",
    "Nisbett", "Bem",
    "Deutsch", "Gerard",  # informational vs normative social influence
    # Multicultural
    "Sue", "Cross", "Berry", "Helms", "Ridley",
    # I-O / career / vocational
    "Holland", "Lofquist", "Dawis", "Super", "Tiedeman",
    "Vroom", "Herzberg", "Fiedler", "Blanchard", "Kirkpatrick",
    "Schmidt", "Hunter",  # validity generalization (Schmidt & Hunter)
    "Campbell",  # research design / Campbell's threats to validity
    # Neuroanatomy (eponymous structures and conditions)
    "Broca", "Wernicke", "Korsakoff", "Alzheimer", "Parkinson",
    "Huntington", "Brodmann", "Geschwind", "Dement", "Kandel",
    "Moruzzi", "Magoun",  # reticular activating system
    # Aging / lifespan
    "Rowe", "Kahn",  # successful aging model (Rowe & Kahn)
    # Assessment instruments / authors
    "Stanford", "Binet", "Wechsler", "Rorschach", "Murray",
    # Other
    "Gestalt",
    # Institutional / document terms — prevents false positives on
    # "Forensic Psychology (2013)", "American Psychological Association
    # (2017)", etc. These look like person names to the regex but are
    # organizations or document titles that legitimately accompany years.
    "Psychology", "Association", "Society", "Guidelines", "Standards",
})

# ── Chapter Display Name Mapping ──────────────────────────────
# Maps (domain_code, chapter_slug) → QUIZ_DOMAINS subdomain name.
# The frontend filters questions by section_title, which must match
# QUIZ_DOMAINS subdomain names exactly. Pipeline chapter_titles from
# CSVs are sometimes shorter than the full subdomain names.
# Key = "DOMAIN:chapter-slug", Value = full QUIZ_DOMAINS subdomain name.

SECTION_TITLE_MAP = {
    # Domain 1: PMET
    "PMET:variables-scales": "Variables, Scales & the Language of Data",
    "PMET:classical-conditioning": "How Organisms Learn — Classical Conditioning",
    "PMET:operant-conditioning": "Shaping Behavior — Operant Conditioning",
    "PMET:correlation-regression": "Relationships in Data — Correlation & Regression",
    "PMET:inferential-statistics": "From Samples to Populations — Inferential Statistics",
    "PMET:research-designs": "Blueprints for Discovery — Research Designs",
    "PMET:research-validity": "Threats & Safeguards — Research Validity",
    "PMET:test-reliability": "Consistency of Measurement — Test Reliability",
    "PMET:content-construct-validity": "Measuring What Matters — Content & Construct Validity",
    "PMET:criterion-related-validity": "Predicting Outcomes — Criterion-Related Validity",
    # Domain 2: LDEV
    "LDEV:architecture-of-development": "The Architecture of Development",
    "LDEV:conception-to-first-breath": "From Conception to First Breath",
    "LDEV:the-awakening": "The Awakening — Infancy & Toddlerhood",
    "LDEV:discovering-the-world": "Discovering the World — Early Childhood",
    "LDEV:building-competence": "Building Competence — Middle Childhood",
    "LDEV:transformation": "Transformation — Adolescence",
    "LDEV:the-long-horizon": "The Long Horizon — Adulthood & Aging",
    # Domain 3: CPAT
    "CPAT:ch01-neurodevelopmental": "Ch 1: Neurodevelopmental Disorders",
    "CPAT:ch02-schizophrenia-psychotic": "Ch 2: Schizophrenia Spectrum & Psychotic Disorders",
    "CPAT:ch03-bipolar": "Ch 3: Bipolar and Related Disorders",
    "CPAT:ch04-depressive": "Ch 4: Depressive Disorders",
    "CPAT:ch05-anxiety": "Ch 5: Anxiety Disorders",
    "CPAT:ch06-obsessive-compulsive": "Ch 6: Obsessive-Compulsive and Related Disorders",
    "CPAT:ch07-trauma-stressor": "Ch 7: Trauma- and Stressor-Related Disorders",
    "CPAT:ch08-dissociative": "Ch 8: Dissociative Disorders",
    "CPAT:ch09-somatic": "Ch 9: Somatic Symptom and Related Disorders",
    "CPAT:ch10-feeding-eating": "Ch 10: Feeding and Eating Disorders",
    "CPAT:ch11-elimination": "Ch 11: Elimination Disorders",
    "CPAT:ch12-sleep-wake": "Ch 12: Sleep-Wake Disorders",
    "CPAT:ch13-sexual-dysfunctions": "Ch 13: Sexual Dysfunctions",
    "CPAT:ch14-gender-dysphoria": "Ch 14: Gender Dysphoria",
    "CPAT:ch15-disruptive-impulse-conduct": "Ch 15: Disruptive, Impulse-Control, and Conduct Disorders",
    "CPAT:ch16-substance-addictive": "Ch 16: Substance-Related and Addictive Disorders",
    "CPAT:ch17-personality": "Ch 17: Personality Disorders",
    "CPAT:ch18-paraphilic": "Ch 18: Paraphilic Disorders",
    # Domain 4: PTHE
    "PTHE:insight": "Through the Unconscious — Insight, Defense & the Depths of the Psyche",
    "PTHE:cognition": "Through Thoughts — Cognitive Restructuring, Rational Disputation & Skills Training",
    "PTHE:conditioning": "Through Conditioning — Classical Extinction, Counterconditioning & Operant Principles",
    "PTHE:acceptance": "Through Acceptance — Mindfulness, Defusion & Existential Presence",
    "PTHE:relationship": "Through the Relationship — Person-Centered Conditions, Gestalt Awareness & Human Needs",
    "PTHE:motivation": "Through Motivation — Stages of Change, Solution Focus & Interpersonal Process",
    "PTHE:systems": "Through the System — Family Structure, Communication Patterns & Group Process",
    "PTHE:evidence": "Through Evidence — Outcome Research, Prevention & Professional Practice",
    # Domain 5: SOCU
    "SOCU:cultural-roots-identity-self": "Where You Come From — Cultural Roots, Identity & Self",
    "SOCU:shortcuts-attribution-biases": "The Shortcuts We Take — How the Mind Judges People",
    "SOCU:attitudes-dissonance-behavior": "What We Believe — Attitudes, Dissonance & Behavior",
    "SOCU:persuasion-conformity-resistance": "How We're Moved — Persuasion, Conformity & Resistance",
    "SOCU:attraction-altruism-helping": "Why We Connect — Attraction, Altruism & Helping",
    "SOCU:prejudice-discrimination-conflict": "Where We Break — Prejudice, Discrimination & Conflict",
    "SOCU:group-dynamics-collective": "When We're Together — Group Dynamics & Collective Behavior",
    "SOCU:multicultural-clinical-practice": "Across the Divide — Multicultural Clinical Practice",
    # Domain 6: WDEV
    "WDEV:organizational-theories": "Organizational Theories",
    "WDEV:job-analysis-performance-assessment": "Job Analysis and Performance Assessment",
    "WDEV:employee-selection-techniques": "Employee Selection - Techniques",
    "WDEV:employee-selection-evaluation": "Employee Selection - Evaluation of Techniques",
    "WDEV:training-methods-evaluation": "Training Methods and Evaluation",
    "WDEV:career-choice-development": "Career Choice and Development",
    "WDEV:theories-of-motivation": "Theories of Motivation",
    "WDEV:satisfaction-commitment-stress": "Satisfaction, Commitment, and Stress",
    "WDEV:organizational-leadership": "Organizational Leadership",
    "WDEV:organizational-decision-making": "Organizational Decision-Making",
    "WDEV:organizational-change-development": "Organizational Change and Development",
    # Domain 7: BPSY
    "BPSY:brain-structure-lateralization": "Brain Structure & Lateralization",
    "BPSY:cerebral-cortex": "Brain Structure & Lateralization",
    "BPSY:subcortical": "Brain Structure & Lateralization",
    "BPSY:learning-encoding": "Learning Processes & Encoding Strategies",
    "BPSY:memory-architecture": "Memory Architecture & Working Memory",
    "BPSY:retrieval-forgetting": "Retrieval, Forgetting & Memory Distortion",
    "BPSY:memory-forgetting": "Retrieval, Forgetting & Memory Distortion",
    "BPSY:neuroscience-memory": "Neuroscience of Memory & Amnesia",
    "BPSY:sleep-architecture": "Sleep Architecture, Dreaming & the Lifespan",
    "BPSY:memory-sleep": "Sleep Architecture, Dreaming & the Lifespan",
    "BPSY:neurons-neurotransmitters": "Neurons, Neurotransmitters & Neural Signaling",
    "BPSY:emotion-arousal": "Emotion Theories, Arousal & the Autonomic Nervous System",
    "BPSY:emotion-stress": "Emotion Theories, Arousal & the Autonomic Nervous System",
    "BPSY:sensation-perception": "Sensation, Perception & Psychophysics",
    "BPSY:neurodegenerative-movement": "Neurodegenerative & Movement Disorders",
    "BPSY:cerebrovascular-injury": "Cerebrovascular Disease, Brain Injury & Seizure Disorders",
    "BPSY:neurological-disorders": "Cerebrovascular Disease, Brain Injury & Seizure Disorders",
    "BPSY:endocrine-neuroimaging": "Endocrine Conditions, Neuroimaging & Diagnostics",
    "BPSY:neurocognitive": "Neurocognitive Disorders",
    # Domain 8: CASS
    "CASS:intelligence-cognitive-assessment": "Intelligence & Cognitive Assessment: Wechsler, Stanford-Binet & Beyond",
    "CASS:personality-assessment-methods": "Personality Assessment: Objective & Projective Methods",
    "CASS:neuropsych-screening-measurement": "Neuropsychological Screening & Clinical Measurement",
    "CASS:vocational-interest-inventories": "Vocational Assessment & Interest Inventories",
    "CASS:test-score-psychometrics": "Test Score Interpretation & Psychometric Concepts",
    "CASS:assessment-ethics-fairness": "Assessment Ethics: Test Fairness, Cultural Competence & Test Security",
    "CASS:therapeutic-relationships-boundaries": "Therapeutic Relationships: Boundaries, Dual Relationships & Termination",
    "CASS:legal-forensic-psychology": "Legal & Forensic Psychology",
    "CASS:supervision-training-standards": "Clinical Supervision, Training & Professional Standards",
    "CASS:ebt-anxiety-trauma-ocd": "Evidence-Based Treatments: Anxiety, Trauma & OCD",
    "CASS:ebt-mood-personality-eating": "Evidence-Based Treatments: Mood, Personality & Eating Disorders",
    "CASS:ebt-substance-developmental-behavioral": "Evidence-Based Treatments: Substance Use, Developmental & Behavioral Disorders",
    # Domain 9: PETH
    "PETH:ethics-overview-general-principles": "Ethics Code Overview & General Principles",
    "PETH:standard-1-resolving-ethical-issues": "Standard 1: Resolving Ethical Issues",
    "PETH:standard-2-competence": "Standard 2: Competence",
    "PETH:standard-3-human-relations": "Standard 3: Human Relations",
    "PETH:standard-4-privacy-confidentiality": "Standard 4: Privacy and Confidentiality",
    "PETH:standard-5-advertising": "Standard 5: Advertising and Other Public Statements",
    "PETH:standard-6-record-keeping-fees": "Standard 6: Record Keeping and Fees",
    "PETH:standard-7-education-training": "Standard 7: Education and Training",
    "PETH:standard-8-research-publication": "Standard 8: Research and Publication",
    "PETH:standard-9-assessment": "Standard 9: Assessment",
    "PETH:standard-10-therapy": "Standard 10: Therapy",
    "PETH:antidepressants": "Antidepressants",
    "PETH:antipsychotics": "Antipsychotics",
    "PETH:anxiolytics-sedatives": "Anxiolytics and Sedatives",
    "PETH:mood-stabilizers": "Mood Stabilizers",
    "PETH:stimulants-adhd": "Stimulants and ADHD Medications",
    "PETH:professional-practice-issues": "Professional Practice Issues",
}


def get_section_title(domain_code, chapter_slug, fallback_title=None):
    """Resolve the QUIZ_DOMAINS section_title for a chapter.

    Returns the mapped name (matching frontend QUIZ_DOMAINS), or
    falls back to the chapter_title from the CSV.
    """
    key = f"{domain_code}:{chapter_slug}"
    return SECTION_TITLE_MAP.get(key, fallback_title or chapter_slug)


# ── Mojibake Repair ──────────────────────────────────────────
# Claude sometimes outputs UTF-8 bytes misinterpreted as CP-1252,
# producing sequences like â€" instead of —. Fix the common ones.

_MOJIBAKE_TABLE = {
    "\u00e2\u20ac\u201d": "\u2014",   # — em dash
    "\u00e2\u20ac\u201c": "\u2013",   # – en dash
    "\u00e2\u20ac\u2122": "\u2019",   # ' right single quote
    "\u00e2\u20ac\u02dc": "\u2018",   # ' left single quote
    "\u00e2\u20ac\u0153": "\u201c",   # \u201c left double quote
    "\u00e2\u20ac\u009d": "\u201d",   # \u201d right double quote
    "\u00c2\u00a0": "\u00a0",         # non-breaking space
}


def fix_mojibake(text):
    """Repair common UTF-8→CP-1252 mojibake in a string."""
    if not text or not isinstance(text, str):
        return text
    for bad, good in _MOJIBAKE_TABLE.items():
        if bad in text:
            text = text.replace(bad, good)
    return text


def fix_mojibake_deep(obj):
    """Recursively fix mojibake in all string values of a dict/list."""
    if isinstance(obj, str):
        return fix_mojibake(obj)
    if isinstance(obj, dict):
        return {k: fix_mojibake_deep(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [fix_mojibake_deep(item) for item in obj]
    return obj


# ── Utilities ─────────────────────────────────────────────────

def slugify(text):
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def get_blooms_verbs(level):
    verbs = {
        "remember": "define, identify, recognize, list, recall",
        "understand": "explain, compare, contrast, paraphrase, interpret",
        "apply": "demonstrate, use, solve, apply to a scenario",
        "analyze": "differentiate, organize, attribute, determine which factor",
        "evaluate": "judge, critique, defend, justify, assess validity",
    }
    return verbs.get(level, "")


# ── Base Classes ──────────────────────────────────────────────

class BaseAgent:
    """Base class for all pipeline agents."""
    name = "base"
    agent_type = "hardcoded"  # "hardcoded" or "llm"

    def validate_input(self, data):
        """Check inputs before execution. Returns (ok, reason)."""
        return True, "ok"

    def execute(self, data):
        """Main execution logic. Override in subclass."""
        raise NotImplementedError(f"{self.name}.execute()")

    def validate_output(self, result):
        """Check outputs meet quality requirements. Returns (ok, reason)."""
        return True, "ok"

    def run(self, data):
        """Full pipeline: validate_input -> execute -> validate_output."""
        ok, reason = self.validate_input(data)
        if not ok:
            return {"_error": f"input: {reason}"}
        result = self.execute(data)
        if result is None:
            return {"_error": "execute returned None"}
        ok, reason = self.validate_output(result)
        if not ok:
            return {"_error": f"output: {reason}", "_partial": result}
        return result


class BaseGate:
    """Base class for validation gates.

    is_prerequisite controls how the orchestrator handles a failure here:
      • True  (default) — short-circuit. Validation stops at the first
                          prerequisite failure; downstream gates don't
                          run (they may depend on this gate's invariant
                          and would crash or produce noise).
      • False           — collect. The orchestrator runs all non-
                          prerequisite gates and gathers every failure
                          for a combined correction prompt. Use for
                          content-level checks (attribution, length
                          balance) that can be addressed alongside
                          other content failures in a single retry.
    """
    name = "base_gate"
    is_prerequisite = True

    def check(self, question, context=None):
        """Returns (ok, reason)."""
        raise NotImplementedError


class AgentRegistry:
    """Maps agent names to classes for discovery and instantiation."""
    _agents = {}

    @classmethod
    def register(cls, name):
        def decorator(agent_cls):
            cls._agents[name] = agent_cls
            return agent_cls
        return decorator

    @classmethod
    def get(cls, name):
        return cls._agents.get(name)

    @classmethod
    def create(cls, name, **kwargs):
        agent_cls = cls._agents.get(name)
        if not agent_cls:
            raise KeyError(f"Unknown agent: {name}")
        return agent_cls(**kwargs)

    @classmethod
    def list_agents(cls):
        return list(cls._agents.keys())
