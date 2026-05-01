"""Heuristic grounding check for anchor briefs.

Each concept in a brief should have textual support in the anchor's source
material (verbatim_anchor + passage). When the LLM invents concepts that
don't appear in the source, the brief drifts from what the anchor actually
testifies — and the question pipeline ends up generating questions about
content the student wasn't asked to learn.

This is a soft signal, not a hard gate. The LLM legitimately rephrases and
abstracts; not every concept word will appear verbatim in the source. The
threshold is a coverage ratio — if too many key terms are absent, flag the
concept for review.

Used by scripts/generate_anchor_briefs.py post-generation. Issues are
printed as warnings; they don't block save.
"""
import re

# Words that match the keyword extraction pattern but are too generic to
# be informative for grounding. Adding to the stop list reduces false
# negatives (concepts wrongly flagged as ungrounded).
_STOP = frozenset({
    "with", "from", "this", "that", "these", "those", "their",
    "have", "been", "were", "will", "would", "could", "should",
    "system", "process", "approach", "concept", "general",
    "such", "some", "many", "more", "most", "other", "another",
    "each", "every", "also", "into", "onto", "upon",
    "when", "where", "what", "which", "while",
})


def _extract_keywords(text):
    """Return lowercase 4+ char words from text, minus generic stop words."""
    if not text:
        return set()
    words = re.findall(r"\b[a-zà-öø-ÿ]{4,}\b", text.lower())
    return set(words) - _STOP


def validate_concept_grounding(concept, source_text, threshold=0.5):
    """Check that concept's significant terms appear in source text.

    Returns (ok, coverage, missing).
      coverage: 0.0–1.0 fraction of keywords found in source.
      missing: sorted list of keywords not found.

    Keywords come from the concept's ID (kebab-case, splits on '-') AND
    its human-readable label. Keeping both broadens the matching so a
    concept named `nondeclarative-memory-system` with label "Implicit
    Memory" can match either set of words in the source.
    """
    keywords = set()

    cid = concept.get("concept_id") or ""
    for w in cid.lower().split("-"):
        if len(w) >= 4:
            keywords.add(w)

    keywords |= _extract_keywords(concept.get("label", ""))
    keywords -= _STOP

    if not keywords:
        return True, 1.0, []

    source_lower = source_text.lower() if source_text else ""
    found = {k for k in keywords if k in source_lower}
    missing = sorted(keywords - found)
    coverage = len(found) / len(keywords)

    return coverage >= threshold, coverage, missing


def validate_brief_grounding(brief, source_text, threshold=0.5):
    """Check every concept in the brief. Returns list of issues (empty = clean).

    source_text should be the concatenation of verbatim_anchor + passage —
    everything the LLM had as factual ground truth when generating.
    """
    issues = []
    for concept in brief.get("concepts", []):
        ok, coverage, missing = validate_concept_grounding(
            concept, source_text, threshold
        )
        if not ok:
            issues.append({
                "concept_id": concept.get("concept_id"),
                "label": concept.get("label"),
                "coverage": round(coverage, 2),
                "missing_terms": missing,
            })
    return issues
