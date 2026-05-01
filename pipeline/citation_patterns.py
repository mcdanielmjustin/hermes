"""Shared citation-detection patterns.

Single source of truth for the five researcher-attribution patterns used
across the pipeline. Keep these in sync everywhere or risk drift between
what the AttributionGate flags and what the audit/sweep reports.

Consumers:
  • pipeline/gates.py        — AttributionGate (validation)
  • pipeline/agents.py       — InputSanitizerAgent (input scrubbing)
  • scripts/audit_question_quality.py    — post-hoc QA on generated batches
  • scripts/sweep_corpus_for_names.py    — corpus-wide citation tally

Patterns

  CITATION_RE          "Squire (2004)" / "Smith & Jones (1985)" /
                       "Watson, J.B. & Rayner, R. (1920)" /
                       "Smith, Jones, and Brown (2010)" /
                       "Latané (1968)"
  ETAL_RE              "Smith et al." (no year)
  ACCORDING_TO_RE      "According to Smith" / "Per Smith"
  POSSESSIVE_RES_RE    "Smith's research/study/framework/theory/etc."
  BARE_MULTI_AUTHOR_RE "Smith and Jones found/showed/proposed/..." (no year)

Coverage decisions

  • Unicode names — `[a-zà-öø-ÿ]` covers Latin-1 diacritics (Latané,
    Köhler, Müller, Søren). Python's re lacks `\\p{Ll}`; explicit ranges
    are the next-cleanest option.
  • Initialed citations — "Smith, A.", "Watson, J.B." form is common
    in psychology literature (155 hits in this corpus). Initials are
    matched, then stripped before whitelist check.
  • Bare multi-author — two-name + attribution-verb only. Single-name
    bare ("Smith showed...") collides with clinical-vignette names too
    often to detect safely.
  • Multi-author split — word-boundary on "and" prevents matching the
    substring "and" inside "Bandura". Oxford comma "X, Y, and Z" is
    recognized as a single citation group.
"""
import re

from . import EPONYM_WHITELIST


# Unicode-aware name pattern: ASCII capital + Latin-1 lowercase range.
# Covers Latané, Köhler, Müller without breaking ASCII names.
_NAME = r"[A-Z][a-zà-öø-ÿ]+"

# Optional initials trailing a surname: "Smith, A." / "Watson, J.B." / "A. B."
_INITIALS = r"(?:,?\s*[A-Z]\.(?:\s*[A-Z]\.)*)?"

# A single author = surname + optional initials + optional " et al."
_AUTHOR = _NAME + _INITIALS + r"(?:\s+et\s+al\.?)?"

# Multi-author connector. Order in alternation matters — longer-form
# alternatives (",\\s*and\\b", ",\\s*&") must come before shorter overlapping
# ones (",") so the regex engine commits to the longer match when both apply.
#   "Smith and Jones"           → ` and `
#   "Smith & Jones"             → ` & `
#   "Smith, Jones, and Brown"   → `, ` then `, and ` (oxford)
#   "Smith, Jones, & Brown"     → `, ` then `, & ` (APA standard)
#   "Cannon-Bard"               → `-` (compound eponym)
_CONNECTOR = r"\s*(?:&|\band\b|,\s*and\b|,\s*&|,|-)\s*"

# "Squire (2004)" / "Atkinson & Shiffrin (1968)" / "Smith, Jones, and Brown (2010)"
# / "Watson, J.B. & Rayner, R. (1920)" / "Latané (1968)" / "Smith (2010a)"
CITATION_RE = re.compile(
    r"\b(?P<name>"
    + _AUTHOR
    + r"(?:" + _CONNECTOR + _AUTHOR + r")*"
    + r")\s*\(\d{4}[a-z]?\)"
)

ETAL_RE = re.compile(r"\b(?P<name>" + _NAME + r")\s+et\s+al\.")

ACCORDING_TO_RE = re.compile(
    r"(?:According to|Per)\s+(?P<name>" + _NAME + r"(?:\s+et\s+al\.?)?)"
)

POSSESSIVE_RES_RE = re.compile(
    r"\b(?P<name>" + _NAME + r")(?:'s|s')\s+"
    r"(?:research|study|studies|finding|findings|framework|theory|"
    r"model|hypothesis|paradigm|experiment|experiments|work|paper|review)\b"
)

# Bare two-author attribution: "Smith and Jones found", "Cannon & Bard proposed".
# Single-name bare attribution ("Smith found") is NOT detected — too many false
# positives with clinical vignette names. Three-author oxford "Smith, Jones, and
# Brown found" is also not caught (the comma form is too noisy without a year);
# accept the gap.
ATTRIBUTION_VERBS = (
    "found", "finds", "showed", "shown", "shows",
    "demonstrated", "demonstrates", "reported", "reports",
    "observed", "observes", "discovered", "discovers",
    "proposed", "proposes", "argued", "argues",
    "suggested", "suggests", "claimed", "claims",
    "investigated", "investigates", "examined", "examines",
    "developed", "develops", "hypothesized", "hypothesizes",
    "theorized", "theorizes", "defined", "defines",
    "concluded", "concludes",
)
_VERBS_GROUP = r"(?:" + "|".join(ATTRIBUTION_VERBS) + r")"

# Bare multi-author with optional oxford comma:
#   "Smith and Jones found"
#   "Smith & Jones demonstrated"
#   "Smith, Jones, and Brown proposed"
#   "Smith, Jones, Brown, and Davis showed"
BARE_MULTI_AUTHOR_RE = re.compile(
    r"\b(?P<name>" + _NAME +
    r"(?:,\s*" + _NAME + r")*"          # optional intermediate authors with commas
    r"\s*,?\s*(?:and|&)\s+" + _NAME + r")"  # final connector (with optional oxford ",")
    r"\s+" + _VERBS_GROUP + r"\b"
)

# Bare single-author attribution: "Smith found that...", "Pavlov demonstrated that...",
# "Smith hypothesized that...". Kept narrow on purpose — single-name detection in
# clinical-vignette text is fundamentally ambiguous, so we only fire on tight,
# research-leaning verb phrases that almost never appear in patient narratives.
#
# Multi-layer false-positive suppression:
#   1. Verb phrases require a complementizer ("found that") to filter out
#      "Smith found her wallet" and similar non-research uses.
#   2. Solo verbs are limited to ones rarely used outside research
#      ("hypothesized", "postulated", "theorized", "formulated").
#   3. Title prefix lookbehind rejects "Dr. Smith / Mr. Jones / etc.".
#   4. Post-filter (in find_attributions) skips names preceded by another
#      capitalized word (likely first name in vignette: "Maria Smith found...").
#   5. Post-filter (in find_attributions) skips NON_NAME_BLOCKLIST tokens
#      ("Results showed that", "It demonstrated that", "Aplysia showed that").
# Verb-phrase list deliberately trimmed for low clinical-context overlap.
# Excluded: "found that", "reported that", "observed that", "noted that",
# "showed that" — all common in patient narratives ("Smith reported that
# anxiety persisted", "the patient observed that..."). The remaining
# verbs are research-leaning enough that legitimate vignette characters
# rarely act as their subject. Solo verbs (hypothesized, postulated,
# theorized, formulated) are virtually never used by clinical characters.
_RESEARCH_VERB_PHRASES = (
    "shown that",
    "demonstrated that",
    "argued that", "concluded that", "proposed that",
    "suggested that", "claimed that", "discovered that",
    "hypothesized", "postulated", "theorized", "formulated",
)
# Each entry is a fixed-width negative lookbehind; the regex engine
# rejects the match if the surname is preceded by any of these titles.
# Includes both abbreviated ("Dr. ") and spelled-out ("Doctor ") forms
# the LLM might use in formal vignettes.
_TITLES = (
    "Dr. ", "Dr ", "Mr. ", "Mr ", "Ms. ", "Ms ",
    "Mrs. ", "Mrs ", "Prof. ", "Prof ", "Rev. ", "Rev ",
    "Sir ", "Doctor ", "Professor ", "Sister ", "Father ",
)
_TITLE_LOOKBEHIND = "".join(f"(?<!{re.escape(t)})" for t in _TITLES)

BARE_SINGLE_AUTHOR_RE = re.compile(
    _TITLE_LOOKBEHIND
    + r"\b(?P<name>" + _NAME + r")\s+"
    + r"(?:" + "|".join(re.escape(p) for p in _RESEARCH_VERB_PHRASES) + r")\b"
)

# Tokens that match the name regex but are not person names. Used to
# suppress false positives in bare_multi and bare_single detection.
NON_NAME_BLOCKLIST = frozenset({
    # Pronouns / determiners / sentence-starters
    "It", "This", "That", "These", "Those", "They", "Such",
    "Some", "Most", "Many", "Both", "All", "The", "An",
    "He", "She", "We", "Us",
    "When", "If", "While", "Although", "However", "Therefore",
    "Thus", "Hence", "Note", "See", "For",
    # Two-letter capitalized tokens that match _NAME but are conjunctions /
    # prepositions / common short words at sentence start.
    "As", "At", "Be", "By", "Do", "Go", "Is", "In", "Of",
    "On", "Or", "So", "To", "Up", "Am",
    # Generic research nouns
    "Results", "Findings", "Studies", "Research", "Data",
    "Evidence", "Analysis", "Theory", "Models", "Method",
    "Methods", "Hypothesis", "Approaches", "Studies", "Tests",
    # Clinical role nouns — common subjects in vignette/case-note prose.
    "Patient", "Client", "Subject", "Participant",
    "Therapist", "Counselor", "Clinician", "Doctor",
    "Person", "Individual", "Practitioner",
    # Disorders / clinical concepts (look like proper nouns when capitalized)
    "Anxiety", "Depression", "Schizophrenia", "Bipolar", "Autism",
    "Trauma", "Phobia", "Mania", "Psychosis", "Aphasia", "Dementia",
    # Cognitive / affective concepts
    "Memory", "Attention", "Cognition", "Perception", "Sensation",
    "Learning", "Reasoning", "Intelligence", "Emotion", "Mood",
    "Affect", "Stress", "Arousal",
    # Personality / social concepts
    "Personality", "Temperament", "Identity", "Self",
    "Conformity", "Compliance", "Obedience", "Aggression",
    "Altruism", "Prejudice", "Stereotype", "Discrimination",
    # Methods / psychometrics
    "Validity", "Reliability", "Correlation", "Regression",
    "Variance",
    # Anatomy
    "Hippocampus", "Amygdala", "Cortex", "Thalamus",
    "Hypothalamus", "Cerebellum",
    # Behavior / processes
    "Behavior", "Behaviour", "Conditioning", "Reinforcement",
    "Extinction", "Motivation", "Therapy", "Treatment",
    "Intervention", "Counseling",
    # Animal models that look name-like
    "Aplysia", "Drosophila",
})

# Internal helpers for split_authors. Hyphen is a split-token too so
# "Cannon-Bard" splits into ["Cannon", "Bard"] for the parts whitelist
# check; whole-name lookup happens first so "Kübler-Ross" stays atomic.
_MULTI_AUTHOR_SPLIT = re.compile(r"\s*(?:&|\band\b|,|-)\s*")
_INITIALS_STRIP = re.compile(r",?\s*[A-Z]\.(?:\s*[A-Z]\.)*")
_ETAL_STRIP = re.compile(r"\s+et\s+al\.?")


def split_authors(name):
    """Split a multi-author capture into individual surnames.

    Strips "et al." and "A." / "J.B." style initials before splitting.
    Returns surnames in original order; empty list for empty input.
    """
    if not name:
        return []
    bare = _ETAL_STRIP.sub("", name.strip())
    bare = _INITIALS_STRIP.sub("", bare)
    return [p.strip() for p in _MULTI_AUTHOR_SPLIT.split(bare) if p.strip()]


def is_whitelisted(name):
    """True iff the name is whitelisted as a whole OR every split part is.

    Whole-name check first so hyphenated atomic eponyms ("Kübler-Ross")
    keep their identity. Falls back to per-part check for compound
    eponyms ("Cannon-Bard" → both parts in whitelist).
    """
    if not name:
        return False
    if name in EPONYM_WHITELIST:
        return True
    parts = split_authors(name)
    return bool(parts) and all(p in EPONYM_WHITELIST for p in parts)


# Detection order shapes the AttributionGate's "first 3 violations" cap in
# error messages. Year/initialed citations are most informative, so first.
_PATTERNS = (
    ("year",         CITATION_RE),
    ("according_to", ACCORDING_TO_RE),
    ("et_al",        ETAL_RE),
    ("possessive",   POSSESSIVE_RES_RE),
    ("bare_multi",   BARE_MULTI_AUTHOR_RE),
    ("bare_single",  BARE_SINGLE_AUTHOR_RE),
)

# Patterns where false-positive suppression via NON_NAME_BLOCKLIST and
# preceding-word inspection is necessary. Year/et-al/according-to/possessive
# patterns have surrounding cues (parens, "et al.", "According to", "'s
# research") that already filter to actual citations, so they don't need it.
_BARE_KINDS = frozenset({"bare_single", "bare_multi"})


def _previous_token(text, start):
    """Return the word immediately before `start`, stripped of punctuation."""
    before = text[max(0, start - 30):start].rstrip()
    if not before:
        return None
    last = before.split()[-1]
    return last.rstrip(",.;:!?") or None


def _bare_match_is_likely_attribution(text, match, name):
    """Filter false positives for bare_single/bare_multi.

    Skip if any of these are true:
      • Any author surname is in NON_NAME_BLOCKLIST (e.g., "Results",
        "Anxiety", "Aplysia").
      • For bare_single: the surname is preceded by another capitalized
        word (likely first-name in a vignette: "Maria Smith found that").
    """
    parts = split_authors(name)
    if any(p in NON_NAME_BLOCKLIST for p in parts):
        return False
    prev = _previous_token(text, match.start("name"))
    if prev and re.fullmatch(_NAME, prev):
        return False
    return True


def find_attributions(text):
    """Yield (matched_str, name, kind, whitelisted) for every attribution.

    kind is one of: 'year', 'et_al', 'according_to', 'possessive',
    'bare_multi', 'bare_single'. Useful for both validation gates (filter
    to non-whitelisted) and audits (count all matches by kind).
    """
    if not text or not isinstance(text, str):
        return
    for kind, pattern in _PATTERNS:
        for m in pattern.finditer(text):
            name = m.group("name")
            if kind in _BARE_KINDS and not _bare_match_is_likely_attribution(text, m, name):
                continue
            yield (m.group(0), name, kind, is_whitelisted(name))
