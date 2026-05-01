"""Single source of truth for stop-word sets used across gates, agents,
and audit scripts.

Five named sets compose into the canonical exports below. Adding a word
to one named set automatically propagates to every consumer importing
the corresponding combined export — eliminating drift between gates and
the form-planner / audit script / brief-vocab extractor.

NB: ``KeywordExtractorAgent.STOPWORDS`` is intentionally NOT consolidated
here. That set serves a different purpose (topic-keyword extraction, not
testwise-tell detection) and includes terms like "many" / "often" /
"called" that we deliberately keep visible to the gates.
"""
from __future__ import annotations


# ── Named building blocks ────────────────────────────────────

# Minimal function words — pronouns, articles, copulas. Used by
# OriginalityGate where we want to compare n-gram content broadly.
BASE_STOP_LIGHT = frozenset({
    "the", "a", "an", "of", "to", "in", "on", "at", "for", "with",
    "and", "or", "but", "is", "are", "was", "were", "be", "been",
    "by", "as", "from", "that", "this", "these", "those", "it",
    "its", "his", "her", "their", "they", "we", "you", "i",
})

# Function words proper — auxiliaries, common adverbs, prepositions
# beyond the basics. Used by every consumer except OriginalityGate.
_FUNCTION_WORD_EXTENSIONS = frozenset({
    "being", "he", "she", "have", "has", "had", "do", "does", "did",
    "will", "would", "could", "should", "may", "might", "must", "can",
    "not", "no", "yes", "if", "than", "then", "so", "such", "what",
    "which", "who", "when", "where", "how", "why", "also", "all",
    "any", "each", "every", "some", "most", "more", "less", "very",
    "into", "onto", "out", "off", "up", "down", "over", "under",
    "between", "about", "above", "below", "after", "before", "during",
    "while", "since", "until", "because", "due", "owing", "without",
    "within", "through",
})

BASE_STOP = BASE_STOP_LIGHT | _FUNCTION_WORD_EXTENSIONS

# Common content modifiers that aren't testwise-tell vocabulary.
# Used by the form-planner's vocab extractor and brief-vocab helper —
# these words are too generic to count as "technical synonym tells".
MODIFIERS = frozenset({
    "side", "type", "form", "kind", "case", "step", "way", "part",
    "level", "result", "effect", "rate", "time", "first", "second",
})

# Cognitive verbs the form planner injects into prompts (T1-T4 verb
# pools). The KW gate strips them so they don't dominate unique-word
# counts; the form planner strips them so they don't appear in
# permitted_vocabulary; the audit script strips them too.
COGNITIVE_VERBS = frozenset({
    "predict", "determine", "evaluate", "select", "choose", "infer",
    "apply", "distinguish", "identify", "recognize", "describe",
    "classify", "characterize", "integrate", "synthesize", "justify",
    "reconcile", "weigh",
})


# ── Canonical combined exports ───────────────────────────────

# Form-planner _VOCAB_STOP and KeywordDistributionGate _STOP both want
# the full set: function words + modifiers + cognitive verbs.
BASE_FULL = BASE_STOP | MODIFIERS | COGNITIVE_VERBS

# Brief-vocab extractor wants function words + modifiers but NOT
# cognitive verbs (the verbs aren't injected into briefs by the form
# planner — they're prompt-time additions).
BASE_AND_MODIFIERS = BASE_STOP | MODIFIERS

# Audit script wants function words + cognitive verbs (no modifier
# stripping — auditors want to see modifiers in the analysis).
BASE_AND_COGNITIVE = BASE_STOP | COGNITIVE_VERBS


# ── Generic clinical/psychology descriptors ──────────────────────
# Common English words that read as psychology vocabulary but aren't
# testwise-tell technical vocabulary. Used by the keyword gates at T3+
# to filter out generic descriptors from the unique-to-correct count.
# At T1/T2 these stay countable (vocabulary IS the test); at T3/T4
# the test is application reasoning, so descriptive English asymmetry
# between correct and distractors is question content, not a tell.
#
# Empirical motivation: D7-PHY-058 calibration showed T3 questions
# failing the keyword gate with 3 unique-to-correct words where 2 of
# the 3 were generic English (e.g., "rapid", "triad", "reflecting"
# alongside one canonical term like "lability"). Filtering generic
# English from the count drops those failures while preserving real
# technical-synonym tells (hemiplegia, contralateral, decussation).
#
# Inclusion criterion: a word is "generic" if it's plausible plain-
# English psychology prose (clinical descriptors, common modifiers)
# AND not in domain_vocab pools as canonical vocabulary.
GENERIC_DESCRIPTORS = frozenset({
    # Clinical/symptom descriptors that aren't anchor-specific concepts
    "abrupt", "acute", "chronic", "consistent", "fluctuating",
    "fluctuations", "gradual", "impaired", "intermittent", "ongoing",
    "rapid", "severe", "sudden", "transient", "weakness",
    # Cognitive/behavioral process verbs (not technical concepts)
    "engages", "execution", "initiated", "passively", "prompted",
    "prompting", "rarely", "reflecting", "responds", "starts",
    "unprompted",
    # Generic nouns clinical-flavored but not testable concepts
    "alongside", "claim", "component", "context", "criterion",
    "factor", "feature", "instance", "occurrence", "presence",
    "process", "scenario", "setting", "situation", "triad",
})
