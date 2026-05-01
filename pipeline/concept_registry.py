"""Canonical concept_id registry for cross-brief consistency.

When the brief generator produces concepts, this module deduplicates against
previously-registered concepts using exact concept_id match, alias mapping,
and normalized-label match. Without this, scaling to 1,500+ briefs produces
fragmented IDs ("agonist" vs "receptor-agonist" vs "ligand-agonist") that
break the question pipeline's cross-anchor concept reuse.

Fuzzy matching (edit distance, Jaccard on words) is deliberately omitted from
the registration path — too risky for false merges of similar but distinct
concepts (e.g., "agonist" vs "antagonist" share most letters). The Phase 3
cross-brief consistency pass is where fuzzy clustering happens, with manual
review.

Storage: data/concept_registry.json — committed to repo via gitignore
exception so concept IDs are stable across runs and machines.
"""
import json
import pathlib
import re
from datetime import datetime, timezone


def _normalize_label(label):
    """Lowercase, drop non-alphanumeric, collapse whitespace.

    "Hippocampal Declarative Memory System" → "hippocampal declarative memory system"
    "Cannon-Bard Theory" → "cannon bard theory"
    """
    if not label:
        return ""
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", label.lower()).split())


class ConceptRegistry:
    """Single source of truth for concept_id ↔ label across all briefs.

    Public API:
        registry = ConceptRegistry(path)
        canonical_id, is_new = registry.lookup_or_register(
            concept_id, label, description, brief_uid)
        registry.save()
    """

    def __init__(self, path):
        self.path = pathlib.Path(path)
        self.data = self._load()

    def _load(self):
        if self.path.exists():
            return json.loads(self.path.read_text(encoding="utf-8"))
        return {
            "version": 1,
            "concepts": {},      # canonical_id -> {label, description, first_seen, appears_in, created_at}
            "label_index": {},   # normalized_label -> canonical_id
            "aliases": {},       # alias_id -> canonical_id (LLM proposed a different ID for the same concept)
        }

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps(self.data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def lookup_or_register(self, concept_id, label, description, brief_uid):
        """Return (canonical_id, is_new).

        Resolution order:
          1. Exact concept_id match → existing canonical, record appearance.
          2. Known alias → existing canonical, record appearance.
          3. Same normalized label → existing canonical (record alias too).
          4. None of the above → register new canonical.
        """
        # 1. Exact ID
        if concept_id in self.data["concepts"]:
            self._record_appearance(concept_id, brief_uid)
            return concept_id, False

        # 2. Known alias
        if concept_id in self.data["aliases"]:
            canonical = self.data["aliases"][concept_id]
            self._record_appearance(canonical, brief_uid)
            return canonical, False

        # 3. Same normalized label
        label_key = _normalize_label(label)
        if label_key and label_key in self.data["label_index"]:
            canonical = self.data["label_index"][label_key]
            self.data["aliases"][concept_id] = canonical
            self._record_appearance(canonical, brief_uid)
            return canonical, False

        # 4. New canonical entry
        self.data["concepts"][concept_id] = {
            "label": label,
            "description": description,
            "first_seen": brief_uid,
            "appears_in": [brief_uid],
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        if label_key:
            self.data["label_index"][label_key] = concept_id
        return concept_id, True

    def _record_appearance(self, concept_id, brief_uid):
        if not brief_uid:
            return
        entry = self.data["concepts"].get(concept_id)
        if entry and brief_uid not in entry["appears_in"]:
            entry["appears_in"].append(brief_uid)

    def stats(self):
        return {
            "total_concepts": len(self.data["concepts"]),
            "total_aliases": len(self.data["aliases"]),
        }


def canonicalize_brief(brief, registry):
    """Update brief's concept_ids to canonical forms; remap misconceptions too.

    Returns (n_new, n_aliased) — how many concepts were registered fresh and
    how many got remapped to an existing canonical ID.
    """
    n_new = 0
    n_aliased = 0
    id_remap = {}  # old_id -> canonical_id (only when they differ)
    brief_uid = brief.get("uid", "")

    for concept in brief.get("concepts", []):
        old_id = concept.get("concept_id", "")
        if not old_id:
            continue
        canonical_id, is_new = registry.lookup_or_register(
            old_id,
            concept.get("label", ""),
            concept.get("description", ""),
            brief_uid,
        )
        if is_new:
            n_new += 1
        elif canonical_id != old_id:
            n_aliased += 1
            id_remap[old_id] = canonical_id
            concept["concept_id"] = canonical_id

    # Remap misconceptions' concepts_involved to use canonical IDs.
    for m in brief.get("misconceptions", []):
        if "concepts_involved" in m:
            m["concepts_involved"] = [
                id_remap.get(c, c) for c in m["concepts_involved"]
            ]

    return n_new, n_aliased
