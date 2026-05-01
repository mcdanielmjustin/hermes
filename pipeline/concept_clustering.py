"""Fuzzy clustering of canonical concept_ids that may refer to the same thing.

Phase 0's ConceptRegistry intentionally only deduplicates on EXACT concept_id
or normalized-label match. Anything fuzzier was deferred here, where
clustering happens post-hoc with a manual review step before merges are
applied.

The reason: character-level fuzzy matchers confuse opposite concepts that
share most of their letters. "Agonist" and "Antagonist" overlap on 7 of
10 chars (string ratio ~82%) but are deliberately distinct. Same for
"hyperthymia" vs "hypothymia", "anterograde" vs "retrograde", etc.
Auto-merging those is catastrophic; surfacing them for review is bad UX.

The similarity used here is WORD-LEVEL JACCARD across slug tokens, label
words, and description content. Word-level avoids the substring trap:
{"agonist"} vs {"antagonist"} have empty intersection (Jaccard 0), even
though their characters share 70%.

Score components
  • slug_jaccard  — Jaccard on the concept_id's hyphen-split tokens.
    "nondeclarative-memory-system" vs "nondeclarative-memory" → 2/3 ≈ 0.67
  • label_jaccard — Jaccard on lowercased label words.
    "Agonist" vs "Receptor Agonist" → {agonist} vs {receptor, agonist} → 0.5
  • desc_jaccard  — Jaccard on description content words (4+ chars), minus stop words.

Combined score is a weighted average. Default threshold (0.5) is the
operating point where genuine fragmentation surfaces and near-twin
opposites stay below the line.
"""
import re

# Words too generic to discriminate concepts; drop from description Jaccard.
_DESC_STOP = frozenset({
    "this", "that", "these", "those", "with", "from", "have", "been",
    "system", "process", "approach", "concept", "general", "specific",
    "such", "some", "many", "more", "most", "other", "another",
    "into", "onto", "upon", "when", "where", "what", "which", "while",
    "form", "type", "kind", "rate", "size", "part",
})


def _slug_tokens(concept_id):
    """Split a kebab-case concept_id into its meaningful tokens."""
    if not concept_id:
        return set()
    return {t for t in concept_id.lower().split("-") if len(t) >= 3}


def _description_words(text):
    """Lowercase 4+ char content words from a description string."""
    if not text:
        return set()
    words = set(re.findall(r"\b[a-zà-öø-ÿ]{4,}\b", text.lower()))
    return words - _DESC_STOP


def _jaccard(a, b):
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _label_words(label):
    """Lowercase 3+ char words from a label."""
    if not label:
        return set()
    return set(re.findall(r"\b[a-zà-öø-ÿ]{3,}\b", label.lower()))


def similarity(concept_a, concept_b,
               weight_slug=0.5, weight_label=0.3, weight_desc=0.2):
    """Compute a 0-1 similarity score between two concept registry entries.

    Each entry is a dict with keys: label, description, plus a concept_id
    field for slug comparison.

    Uses word-level Jaccard for label similarity to avoid the SequenceMatcher
    trap where "Agonist" and "Antagonist" score artificially high due to
    shared character substrings.
    """
    slug_a = _slug_tokens(concept_a.get("concept_id", ""))
    slug_b = _slug_tokens(concept_b.get("concept_id", ""))
    slug_score = _jaccard(slug_a, slug_b)

    label_score = _jaccard(
        _label_words(concept_a.get("label", "")),
        _label_words(concept_b.get("label", "")),
    )

    desc_score = _jaccard(
        _description_words(concept_a.get("description", "")),
        _description_words(concept_b.get("description", "")),
    )

    return (weight_slug * slug_score
            + weight_label * label_score
            + weight_desc * desc_score)


def find_merge_candidates(registry_concepts, threshold=0.7):
    """Find concept_id pairs likely referring to the same concept.

    registry_concepts: dict mapping canonical_id -> {label, description, ...}
    threshold: minimum similarity score (0-1) to surface as a candidate.

    Returns a list of dicts sorted by score descending:
      [{"id_a": ..., "id_b": ..., "score": ..., "label_a": ..., "label_b": ..., ...}, ...]

    Each pair appears once (not symmetric duplicates). Self-pairs excluded.
    Returns empty list when no candidates clear the threshold.
    """
    if not registry_concepts:
        return []

    items = [(cid, dict(meta, concept_id=cid))
             for cid, meta in registry_concepts.items()]

    candidates = []
    for i in range(len(items)):
        cid_a, ca = items[i]
        for j in range(i + 1, len(items)):
            cid_b, cb = items[j]
            score = similarity(ca, cb)
            if score >= threshold:
                candidates.append({
                    "id_a": cid_a,
                    "id_b": cid_b,
                    "score": round(score, 3),
                    "label_a": ca.get("label", ""),
                    "label_b": cb.get("label", ""),
                })

    candidates.sort(key=lambda c: -c["score"])
    return candidates
