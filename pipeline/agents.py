"""
Pipeline agents — Phase 1 (preparation), Phase 2 (creative), Phase 3 (assembly).

Phase 1 agents are hardcoded (instant, $0, 100% deterministic).
Phase 2 agent is the single LLM call (async, focused prompt).
Phase 3 agent merges creative output with hardcoded metadata.
"""

import re
import json
import uuid
import asyncio
from datetime import datetime, timezone

from . import (
    BaseAgent, AgentRegistry, PASSEPPP_UUID_NS,
    BLOOMS_BY_TIER, DIFFICULTY_LABELS, DIFFICULTY_LETTERS,
    DISTRACTOR_MIX, VALID_MISCONCEPTION_TYPES, EPONYM_WHITELIST,
    slugify, fix_mojibake_deep,
)
from .citation_patterns import CITATION_RE, split_authors
from .stopwords import (
    BASE_AND_MODIFIERS as _BASE_AND_MODIFIERS,
    BASE_FULL as _BASE_FULL,
)


# ══════════════════════════════════════════════════════════════
# PHASE 1: Preparation Agents (hardcoded, instant, $0)
# ══════════════════════════════════════════════════════════════


@AgentRegistry.register("input_sanitizer")
class InputSanitizerAgent(BaseAgent):
    """Strips researcher citations from anchor input data before prompt assembly.

    Brief.verbatim_anchor sometimes opens with "Squire (2004):" style
    citations. Without sanitization the LLM echoes those names into stems
    and explanations, violating the no-attribution rule. Whitelisted
    eponyms (Piaget, Cannon-Bard, etc.) are kept without the year; non-
    whitelisted names are removed entirely.
    """
    name = "input_sanitizer"

    # CITATION_RE comes from citation_patterns — same regex the AttributionGate
    # uses, so input scrubbing and output validation can't drift.
    _LEADING_PUNCT_RE = re.compile(r"^[\s:\-—,;]+")
    # Collapse only horizontal whitespace runs (spaces/tabs). \s{2,} would
    # also collapse newlines, mangling multi-line briefs if they're ever
    # introduced. Briefs are currently single-line per claim; this is
    # forward-proofing.
    _DOUBLE_SPACE_RE = re.compile(r"[ \t]{2,}")

    @staticmethod
    def _strip_citation(match):
        """Best-of-breed: keep whitelisted authors, strip non-whitelisted.

        'Cannon and Bard (1929)'        → 'Cannon and Bard' (all WL → preserve formatting)
        'Watson, J.B. & Rayner, R. (1920)' → 'Watson' (only Watson WL; initials dropped)
        'Bandura and Walters (1963)'    → 'Bandura' (only Bandura WL)
        'Smith and Jones (1985)'        → '' (none WL → strip everything)
        'Squire (2004)'                 → '' (Squire not WL → strip)
        """
        name_part = match.group("name")
        parts = split_authors(name_part)
        kept = [p for p in parts if p in EPONYM_WHITELIST]
        if not kept:
            return ""
        if len(kept) == len(parts):
            # All authors whitelisted — preserve original formatting,
            # initials, ordering ("Cannon and Bard" stays as written).
            return name_part
        # Mixed: rebuild with whitelisted only, in original order.
        return " & ".join(kept) if len(kept) > 1 else kept[0]

    @classmethod
    def sanitize(cls, text):
        if not text or not isinstance(text, str):
            return text
        text = CITATION_RE.sub(cls._strip_citation, text)
        text = cls._LEADING_PUNCT_RE.sub("", text)
        text = cls._DOUBLE_SPACE_RE.sub(" ", text).strip()
        return text

    def execute(self, data):
        verbatim = data.get("verbatim_anchor", "")
        testable = data.get("testable_fact", "")
        core_claims = data.get("core_claims", []) or []
        return {
            "verbatim_anchor": self.sanitize(verbatim),
            "testable_fact": self.sanitize(testable),
            "core_claims": [self.sanitize(c) for c in core_claims],
        }


@AgentRegistry.register("concept_vocab")
class ConceptVocabAgent(BaseAgent):
    """Loads pre-generated concept vocabulary for a chapter.

    Skill: JSON file lookup from concept_vocab/{domain}/{chapter_id}.json.
    """
    name = "concept_vocab"

    def execute(self, data):
        vocab_dir = data["concept_vocab_dir"]
        domain_code = data["domain_code"]
        chapter_id = data["chapter_id"]

        vocab_path = vocab_dir / domain_code / f"{chapter_id}.json"
        if not vocab_path.exists():
            return {"has_vocab": False, "concepts": [], "misconceptions": []}
        try:
            with open(vocab_path, encoding="utf-8") as f:
                vocab_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return {"has_vocab": False, "concepts": [], "misconceptions": []}

        concepts = vocab_data.get("concepts", [])
        misconceptions = vocab_data.get("misconceptions", [])
        if not concepts:
            return {"has_vocab": False, "concepts": [], "misconceptions": []}

        return {
            "has_vocab": True,
            "concepts": concepts,
            "misconceptions": misconceptions,
        }


@AgentRegistry.register("anchor_brief")
class AnchorBriefAgent(BaseAgent):
    """Loads pre-generated anchor analysis brief for a single anchor.

    Skill: JSON file lookup from anchor_briefs/{domain}/{uid}.json.
    When available, provides anchor-specific concepts, misconceptions,
    core_claims, and question_angles — replacing chapter-level vocab
    as the primary intelligence source.
    """
    name = "anchor_brief"

    def execute(self, data):
        briefs_dir = data["anchor_briefs_dir"]
        domain_code = data["domain_code"]
        uid = data["uid"]

        brief_path = briefs_dir / domain_code / f"{uid}.json"
        empty_brief = {
            "has_brief": False,
            "core_claims": [],
            "concepts": [],
            "misconceptions": [],
            "question_angles": [],
            # Phase 20c additions: structured fields for stem-rewrite
            # templates (and any future generation-time augmentation).
            # Default empty strings/lists so legacy briefs (without these
            # fields) load cleanly.
            "concept_explanation": "",
            "discriminators": [],
        }
        if not brief_path.exists():
            return empty_brief
        try:
            with open(brief_path, encoding="utf-8") as f:
                brief_data = json.load(f)
        except (json.JSONDecodeError, OSError):
            return empty_brief

        core_claims = brief_data.get("core_claims", [])
        concepts = brief_data.get("concepts", [])
        misconceptions = brief_data.get("misconceptions", [])
        question_angles = brief_data.get("question_angles", [])
        concept_explanation = brief_data.get("concept_explanation", "")
        discriminators = brief_data.get("discriminators", [])

        if not concepts:
            return {"has_brief": False, "core_claims": core_claims, "concepts": [],
                    "misconceptions": [], "question_angles": question_angles,
                    "concept_explanation": concept_explanation,
                    "discriminators": discriminators,
                    "chapter_num": brief_data.get("chapter_num", "")}

        return {
            "has_brief": True,
            "core_claims": core_claims,
            "concepts": concepts,
            "misconceptions": misconceptions,
            "question_angles": question_angles,
            # Phase 20c: irreducible mechanism/principle the question
            # must force the student to invoke + cognitive dimensions
            # the concept can be probed along. Used by Phase 20d
            # stem-rewrite templates. NOT included in vocabulary
            # extraction (extract_brief_vocabulary_terms below) — these
            # are structural metadata, not lexical content.
            "concept_explanation": concept_explanation,
            "discriminators": discriminators,
            # chapter_num is needed by load_anchor_context to feed
            # AnchorClusterAgent's same-chapter priority selection. Without
            # it, cluster_anchors falls through to same-domain fallback
            # for every anchor — the same-chapter priority never fires.
            "chapter_num": brief_data.get("chapter_num", ""),
        }


# Shared vocabulary stop-set used by both AnchorClusterAgent's selection
# (Option 5: text-level vocabulary diversity) and the L3 scaffold's term
# extraction. Single source of truth — drift between selection and L3
# extraction would mean the cluster agent picks anchors based on different
# vocabulary than what actually feeds the scaffold.
# Stop-word set for brief vocabulary extraction — function words +
# modifiers, no cognitive verbs (those aren't injected into briefs).
# Centralized in pipeline/stopwords.py to prevent drift across consumers.
_BRIEF_VOCAB_STOP = _BASE_AND_MODIFIERS


def extract_brief_vocabulary_terms(brief, max_terms=None):
    """Extract ≥6-char content-vocabulary terms from a brief's text fields.

    Used by AnchorClusterAgent (Option 5: vocabulary-diversity selection)
    and by the CorrectAnswerFormPlannerAgent's L3 scaffold extraction —
    single source of truth so cluster selection and L3 scaffold operate
    on the same term-set.

    Reads from concept descriptions, core_claims, and testable_fact —
    the same text fields the L3 scaffold sources from. Returns a SET of
    deduplicated lowercase terms.

    `brief` is a dict with the standard brief schema (concepts list,
    core_claims list, testable_fact string).
    """
    if not brief:
        return set()
    text_sources = []
    for c in brief.get("concepts", []) or []:
        desc = c.get("description", "") if isinstance(c, dict) else ""
        if desc:
            text_sources.append(desc)
    text_sources.extend(c for c in (brief.get("core_claims", []) or []) if c)
    tf = brief.get("testable_fact", "") or ""
    if tf:
        text_sources.append(tf)
    if not text_sources:
        return set()
    joined = " ".join(text_sources).lower()
    words = re.findall(r"\b[a-zà-öø-ÿ]{6,}\b", joined)
    terms = {w for w in words if w not in _BRIEF_VOCAB_STOP}
    if max_terms is not None and len(terms) > max_terms:
        # Sort by length desc to keep most-technical terms when capping
        terms = set(sorted(terms, key=len, reverse=True)[:max_terms])
    return terms


@AgentRegistry.register("anchor_cluster")
class AnchorClusterAgent(BaseAgent):
    """Selects 1-2 sibling anchors from the same domain to provide
    supplementary vocabulary and (at T3+) cross-content integration
    substrate. Always-on Phase 4 baseline diversification.

    The primary anchor remains primary — testable_fact, tested concept,
    distractor misconceptions, stem framing all come from primary. Cluster
    anchors contribute ONLY:
      • Vocabulary headroom for the L3 scaffold (richer term pool that
        distractors can spread across, defeating synonym_uniqueness tells)
      • Optional integration concepts at T4 (evaluate/analyze substrate)

    Bloom's-tier scaling:
      • T1/T2: 1 cluster anchor (vocabulary supplement only — recognition
        and understanding stay anchored to primary)
      • T3/T4: 2 cluster anchors (application / analyze-evaluate structurally
        require cross-content scenarios — clustering provides them)

    Selection priority:
      1. Same chapter (highest — same textbook section)
      2. Same domain + ≥1 shared concept_id (related neighbors)
      3. Same domain (broadest fallback)

    Deterministic rotation by primary UID hash for run-to-run consistency.
    """
    name = "anchor_cluster"

    # Bloom's-identity invariant: T1 (Recognize) and T2 (Understand) test
    # ONE concept from ONE anchor — that's the cognitive identity of those
    # tiers. Importing sibling-anchor content (even just vocabulary) dilutes
    # what the question is supposed to be about. Clustering is reserved for
    # T3 (Apply) and T4 (Analyze/Evaluate) where cross-content scaffolding IS the
    # cognitive substrate of the tier.
    _CLUSTER_COUNT_BY_TIER = {1: 0, 2: 0, 3: 1, 4: 2}

    def execute(self, data):
        primary_uid = data.get("primary_uid", "")
        domain_code = data.get("domain_code", "")
        primary_chapter = data.get("primary_chapter_num", "") or ""
        primary_concepts = data.get("primary_concepts", []) or []
        tier = data.get("tier", 1)
        briefs_dir = data.get("anchor_briefs_dir")
        # Option 5: vocabulary-level diversity selection. The agent now
        # compares EXTRACTED VOCABULARY TERMS between primary and each
        # candidate sibling, not just concept_id sets. This is the same
        # vocabulary the L3 scaffold sources from — selection uses the
        # actual L3 source as the diversity signal.
        primary_core_claims = data.get("primary_core_claims", []) or []
        primary_testable_fact = data.get("primary_testable_fact", "") or ""

        # Default to 0 for unknown tiers — safer than silently introducing
        # cluster content at an unexpected tier (which would risk violating
        # the Bloom's-identity invariant for that tier).
        n_target = self._CLUSTER_COUNT_BY_TIER.get(tier, 0)

        if not briefs_dir or not primary_uid or not domain_code:
            return {"cluster_anchors": []}

        domain_dir = briefs_dir / domain_code if hasattr(briefs_dir, "__truediv__") \
            else __import__("pathlib").Path(briefs_dir) / domain_code
        if not domain_dir.exists():
            # T3+ silently violates the Bloom's-identity invariant
            # (cross-content scaffolding required) when this happens.
            # Surface it so the caller can fix the data layout.
            if n_target > 0:
                print(
                    f"  [warn] anchor_cluster: domain dir missing for "
                    f"{domain_code} ({domain_dir}); T{tier} expects "
                    f"{n_target} cluster anchor(s) but got 0."
                )
            return {"cluster_anchors": []}

        primary_concept_ids = {
            c.get("concept_id", "") for c in primary_concepts if c.get("concept_id")
        }

        # Load all sibling briefs in the same domain
        siblings = []
        for brief_path in sorted(domain_dir.glob("*.json")):
            uid = brief_path.stem
            if uid == primary_uid:
                continue
            try:
                with open(brief_path, encoding="utf-8") as f:
                    brief = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue
            sibling_concept_ids = {
                c.get("concept_id", "")
                for c in brief.get("concepts", []) if c.get("concept_id")
            }
            siblings.append({
                "uid": uid,
                "brief": brief,
                "chapter_num": brief.get("chapter_num", "") or "",
                "concept_ids": sibling_concept_ids,
            })

        if not siblings:
            if n_target > 0:
                print(
                    f"  [warn] anchor_cluster: no sibling briefs in "
                    f"{domain_code} for primary {primary_uid}; T{tier} "
                    f"expects {n_target} cluster anchor(s) but got 0."
                )
            return {"cluster_anchors": []}

        # Tier 1: same chapter (only when primary has a non-empty chapter_num)
        same_chapter = [
            s for s in siblings
            if primary_chapter and s["chapter_num"] == primary_chapter
        ]
        # Tier 2: same domain + VOCABULARY-DIVERSE (Option 5).
        #
        # Compares actual extracted vocabulary terms (concept descriptions
        # + core_claims + testable_fact) between primary and each sibling.
        # This is the same source the L3 scaffold pulls from — the
        # selection signal IS the L3 source, not a proxy.
        #
        # Why this is stronger than Option 4 (concept_id Jaccard):
        # CASS calibration showed siblings with disjoint concept_ids
        # ("informed-consent" vs "dual-relationship") still shared heavy
        # ETHICS-vocabulary terms (professional, practice, client,
        # boundary, obligation). Concept-id selection thought they were
        # diverse; vocabulary selection sees they're not. Option 5 uses
        # the actual L3 source for diversity scoring.
        def _jaccard_distance(a, b):
            if not a and not b:
                return 1.0  # both empty → treat as fully different (rare)
            union = a | b
            if not union:
                return 1.0
            return 1 - (len(a & b) / len(union))

        # Build a synthetic primary "brief" for vocabulary extraction
        # using the data the orchestrator passes in.
        primary_brief = {
            "concepts": primary_concepts,
            "core_claims": primary_core_claims,
            "testable_fact": primary_testable_fact,
        }
        primary_terms = extract_brief_vocabulary_terms(primary_brief)

        # Compute vocabulary distance for each sibling. Cache the
        # extracted terms on the sibling dict for later inspection.
        for s in siblings:
            s["vocab_terms"] = extract_brief_vocabulary_terms(s["brief"])
            s["vocab_distance"] = _jaccard_distance(s["vocab_terms"], primary_terms)

        disjoint_vocab = [
            s for s in siblings
            if s not in same_chapter and s["vocab_distance"] >= 0.5
        ]
        # Tier 3: same domain (broadest fallback for cases where no sibling
        # has sufficient vocabulary diversity — e.g., a narrow domain where
        # every brief is written in the same terminology).
        rest = [
            s for s in siblings
            if s not in same_chapter and s not in disjoint_vocab
        ]

        if not (same_chapter or disjoint_vocab or rest):
            return {"cluster_anchors": []}

        # Deterministic rotation by primary UID hash, BUT always exhaust
        # higher-priority groups before falling to lower priority ones.
        # Earlier implementation rotated across the flat candidate list,
        # which could skip same-chapter siblings in favor of arbitrary
        # same-domain siblings if the rotation offset landed past the
        # higher-priority group. Priority is priority — rotation only
        # decides WHICH same-chapter sibling, not whether to use a
        # same-chapter sibling at all.
        seed = sum(ord(c) for c in primary_uid) if primary_uid else 0
        selected = []
        for group in (same_chapter, disjoint_vocab, rest):
            if len(selected) >= n_target:
                break
            if not group:
                continue
            start = seed % len(group)
            needed = n_target - len(selected)
            for i in range(min(needed, len(group))):
                candidate = group[(start + i) % len(group)]
                if candidate not in selected:
                    selected.append(candidate)

        return {
            "cluster_anchors": [
                {
                    "uid": s["uid"],
                    "concepts": s["brief"].get("concepts", []),
                    "core_claims": s["brief"].get("core_claims", []),
                    "testable_fact": s["brief"].get("testable_fact", ""),
                    "selection_tier": (
                        "same_chapter" if s in same_chapter
                        else "disjoint_vocab" if s in disjoint_vocab
                        else "same_domain"
                    ),
                    # Surface the vocab distance for audit traceability —
                    # operators can inspect why a particular cluster was
                    # selected (or why a candidate was rejected).
                    "vocab_distance": s.get("vocab_distance", 0.0),
                }
                for s in selected
            ],
        }


@AgentRegistry.register("distractor_planner")
class DistractorPlannerAgent(BaseAgent):
    """Pre-assigns distractor levels and misconception_ids from tier rules + concept vocabulary.

    Skill: deterministic level assignment from DISTRACTOR_MIX, plus unique
    misconception_id selection from concept vocab (prioritizing misconceptions
    that involve the tested concept). Eliminates duplicate-misconception errors.
    """
    name = "distractor_planner"

    def execute(self, data):
        tier = data["tier"]
        variant = data.get("variant", 1)
        misconceptions = data.get("misconceptions", [])
        tested_concept_id = data.get("tested_concept_id")
        # Cross-planner alignment: when the form planner has restricted the
        # correct answer to a permitted concept set, distractors should target
        # misconceptions tied to ONE OF those concepts — otherwise distractors
        # introduce off-topic content that students eliminate by topic, not by
        # reasoning. Calibration finding (D7-PHY-209 phase 2 v4): one
        # surviving question (M-02) had a distractor about "embolism vs
        # thrombosis" on a stem about "contralateral motor control" because
        # the misconception pool wasn't filtered by permitted concepts.
        permitted_concept_ids = data.get("permitted_concept_ids", []) or []

        mix = DISTRACTOR_MIX[tier]
        levels_needed = []
        for level_key, count in mix.items():
            level_num = int(level_key.lstrip("L"))
            levels_needed.extend([level_num] * count)

        if not misconceptions or len(misconceptions) < 3:
            slots = [
                {"slot": i + 1, "distractor_level": level, "mode": "open"}
                for i, level in enumerate(levels_needed)
            ]
            return {"slots": slots, "mode": "open"}

        # Filter to misconceptions tied to permitted concepts when the form
        # planner has restricted scope. Falls back to the unfiltered pool if
        # filtering would leave fewer than 3 candidates (preserves existing
        # behavior when the form planner doesn't constrain).
        if permitted_concept_ids:
            permitted_set = set(permitted_concept_ids)
            filtered = [
                m for m in misconceptions
                if permitted_set & set(m.get("concepts_involved", []) or [])
            ]
            if len(filtered) >= 3:
                misconceptions = filtered

        # Split into primary (involves tested concept) and secondary
        primary = []
        secondary = []
        for m in misconceptions:
            if tested_concept_id and tested_concept_id in m.get("concepts_involved", []):
                primary.append(m)
            else:
                secondary.append(m)

        # Rotate starting position by variant+tier so each combo gets different misconceptions
        offset = (variant - 1) * 3 + (tier - 1)
        pool = primary + secondary
        selected = []
        for i in range(3):
            idx = (offset + i) % len(pool)
            candidate = pool[idx]
            while candidate["misconception_id"] in [s["misconception_id"] for s in selected]:
                idx = (idx + 1) % len(pool)
                candidate = pool[idx]
                if idx == (offset + i) % len(pool):
                    break
            selected.append(candidate)

        slots = []
        for i, level in enumerate(levels_needed):
            m = selected[i] if i < len(selected) else None
            slot = {"slot": i + 1, "distractor_level": level, "mode": "focused"}
            if m:
                slot["misconception_id"] = m["misconception_id"]
                slot["misconception_label"] = m.get("label", "")
                slot["misconception_type"] = m.get("type", "")
                slot["concepts_involved"] = m.get("concepts_involved", [])
            slots.append(slot)

        return {"slots": slots, "mode": "focused"}


@AgentRegistry.register("keyword_extractor")
class KeywordExtractorAgent(BaseAgent):
    """Extracts topic keywords from content text and concept vocabulary.

    Skill: concept-label extraction (vocab mode) or frequency-based
    extraction with psychology term boosting (open mode).
    NEW agent — replaces LLM-generated topic_keywords.
    """
    name = "keyword_extractor"

    STOPWORDS = frozenset({
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "shall", "can", "must", "need", "and",
        "but", "or", "nor", "not", "so", "yet", "both", "either", "neither",
        "each", "every", "all", "any", "few", "more", "most", "other", "some",
        "such", "no", "only", "own", "same", "than", "too", "very", "just",
        "because", "as", "until", "while", "of", "at", "by", "for", "with",
        "about", "against", "between", "through", "during", "before", "after",
        "above", "below", "to", "from", "up", "down", "in", "out", "on",
        "off", "over", "under", "again", "further", "then", "once", "here",
        "there", "when", "where", "why", "how", "this", "that", "these",
        "those", "it", "its", "they", "them", "their", "we", "our", "he",
        "his", "she", "her", "me", "my", "which", "who", "whom", "what",
        "also", "however", "although", "including", "within", "many", "often",
        "well", "much", "one", "two", "three", "like", "new", "used", "first",
        "even", "way", "may", "called", "also", "known", "such", "refers",
        "involves", "specific", "related", "different", "important", "example",
        "include", "includes", "typically", "common", "based", "associated",
    })

    PSYCH_BOOSTS = frozenset({
        "conditioning", "reinforcement", "extinction", "memory", "cognition",
        "attachment", "development", "disorder", "therapy", "assessment",
        "intelligence", "personality", "motivation", "emotion", "perception",
        "neurotransmitter", "syndrome", "psychotherapy", "diagnostic", "validity",
        "reliability", "correlation", "regression", "hypothesis", "variable",
        "stimulus", "response", "behavior", "learning", "consciousness",
        "anxiety", "depression", "schizophrenia", "bipolar", "trauma",
        "neuropsychological", "cognitive", "behavioral", "psychodynamic",
        "humanistic", "biological", "cortex", "hippocampus", "amygdala",
        "serotonin", "dopamine", "norepinephrine", "acetylcholine",
    })

    def execute(self, data):
        content = data.get("content", "")
        concepts = data.get("concepts", [])

        # Vocab-backed: use concept labels directly
        if concepts and len(concepts) >= 3:
            return {"topic_keywords": [c["label"] for c in concepts[:5]]}

        # Open: frequency-based extraction
        if not content or len(content) < 20:
            return {"topic_keywords": []}

        # Extract bigrams first (more specific)
        content_lower = content.lower()
        bigrams = {}
        for match in re.finditer(r'\b([a-z]{3,})\s+([a-z]{3,})\b', content_lower):
            w1, w2 = match.group(1), match.group(2)
            if w1 not in self.STOPWORDS and w2 not in self.STOPWORDS:
                bg = f"{w1} {w2}"
                bigrams[bg] = bigrams.get(bg, 0) + 1

        # Single words with boosting
        words = re.findall(r'\b[a-z]{4,}\b', content_lower)
        word_freq = {}
        for w in words:
            if w not in self.STOPWORDS:
                word_freq[w] = word_freq.get(w, 0) + 1
        for w in word_freq:
            if w in self.PSYCH_BOOSTS:
                word_freq[w] *= 3

        # Combine: bigrams first, then boosted single words
        ranked_bg = sorted(bigrams.items(), key=lambda x: -x[1])
        keywords = [bg for bg, _ in ranked_bg[:3]]

        if len(keywords) < 5:
            ranked_words = sorted(word_freq.items(), key=lambda x: -x[1])
            for w, _ in ranked_words:
                if w not in " ".join(keywords):
                    keywords.append(w)
                if len(keywords) >= 5:
                    break

        return {"topic_keywords": keywords[:5]}


@AgentRegistry.register("metadata")
class MetadataAgent(BaseAgent):
    """Builds all deterministic metadata fields for a question.

    Accepts anchor-level data from CSV iteration. Generates
    question_id from anchor_point_id_v2 + tier + variant.
    """
    name = "metadata"

    def execute(self, data):
        domain_code = data["domain_code"]
        domain_id = data["domain_id"]
        domain_name = data["domain_name"]
        chapter_title = data["chapter_title"]
        tier = data["tier"]
        variant = data["variant"]
        source_type = data["source_type"]
        stem_pattern = data["stem_pattern"]
        batch_id = data["batch_id"]
        content_chars = data.get("content_chars", 0)
        section_title = data.get("section_title")

        anchor_uid = data["anchor_uid"]
        anchor_id_v2 = data["anchor_id_v2"]
        verbatim_anchor = data.get("verbatim_anchor", "")
        testable_fact = data.get("testable_fact", "")

        chapter_slug = slugify(chapter_title)

        question_id = f"QZ-{domain_code}-{anchor_id_v2}-{DIFFICULTY_LETTERS[tier]}-{variant:02d}"

        chapter_uuid = str(uuid.uuid5(
            PASSEPPP_UUID_NS,
            f"{domain_code}:{chapter_title}",
        ))
        anchor_uuid = str(uuid.uuid5(
            PASSEPPP_UUID_NS,
            f"{domain_code}:{anchor_uid}",
        ))

        return {
            "question_id": question_id,
            "meta_base": {
                "question_id": question_id,
                "domain_id": domain_id,
                "domain_code": domain_code,
                "domain_name": domain_name,
                "chapter_file": f"{chapter_slug}.html",
                "chapter_title": chapter_title,
                "section_title": section_title or chapter_title,
                "chapter_uuid": chapter_uuid,
                "anchor_uuid": anchor_uuid,
                "anchor_label": verbatim_anchor[:120],
                "difficulty_tier": tier,
                "difficulty_label": DIFFICULTY_LABELS[tier],
                "blooms_primary": BLOOMS_BY_TIER[tier][0],
                "blooms_secondary": BLOOMS_BY_TIER[tier][1],
                "source_type": source_type,
                "stem_pattern": stem_pattern,
                "anchor_uids": [anchor_uid],
                "anchor_point_ids_v2": [anchor_id_v2],
                "anchor_content_summaries": [verbatim_anchor],
                "testable_fact": testable_fact,
                "variant": variant,
                "batch_id": batch_id,
                "content_snippet_chars": content_chars,
            },
        }


@AgentRegistry.register("tested_concept_selector")
class TestedConceptSelectorAgent(BaseAgent):
    """Pre-selects which concept from vocab is the tested concept.

    Skill: deterministic rotation across variants AND tiers to ensure
    maximum coverage of all concepts in a chapter. Eliminates LLM
    concept_id/concept_label generation (the LLM only writes knowledge_tested).
    NEW agent — previously LLM picked freely.
    """
    name = "tested_concept_selector"

    def execute(self, data):
        concepts = data.get("concepts", [])
        variant = data.get("variant", 1)
        tier = data.get("tier", 1)

        if not concepts:
            return {"has_tested_concept": False}

        # Rotate: variant×4 + tier gives unique selection per variant/tier combo
        idx = ((variant - 1) * 4 + (tier - 1)) % len(concepts)
        selected = concepts[idx]

        return {
            "has_tested_concept": True,
            "concept_id": selected["concept_id"],
            "concept_label": selected["label"],
        }


@AgentRegistry.register("concept_integration_planner")
class ConceptIntegrationPlannerAgent(BaseAgent):
    """Pre-selects the 2 concepts a Tier 4 question MUST integrate.

    The dominant Tier 4 failure pattern (50% rate in audit) is
    "single-concept-sufficient" — questions where one concept's definition
    alone answers it, despite the prompt's explicit 2-concept-integration
    requirement. The fix mirrors DistractorPlannerAgent's strategy: pre-
    assign the constraint upfront so the LLM's task collapses from
    "decide which concepts to integrate" to "write a question that
    integrates these specific two concepts."

    Selection: rotate deterministically across (variant, tier). The
    primary integration concept is the same as TestedConceptSelectorAgent's
    pick (so distractor-plan and integration-plan stay aligned). The
    secondary integration concept is the next concept in rotation that
    isn't the primary.

    Fires only at tier=4. T1-T3 are not gated by 2-concept integration in
    Bloom's enforcement.
    """
    name = "concept_integration_planner"

    def execute(self, data):
        tier = data.get("tier", 1)
        if tier != 4:
            return {"requires_integration": False}

        concepts = data.get("concepts", [])
        if len(concepts) < 2:
            # T4 invariant violation: analyze/evaluate tier without ≥2 concepts
            # would silently produce a question with no integration
            # scaffold (the most common T4 failure mode). Surface it so
            # the caller can surface a brief-quality issue rather than
            # let the LLM generate a single-concept-sufficient T4.
            uid = data.get("primary_uid") or data.get("uid") or "<unknown>"
            print(
                f"  [warn] concept_integration: T4 anchor {uid} has only "
                f"{len(concepts)} concept(s); integration scaffold disabled. "
                "Brief should expose 2+ concepts at T4."
            )
            return {"requires_integration": False}

        variant = data.get("variant", 1)
        primary_idx = ((variant - 1) * 4 + (tier - 1)) % len(concepts)
        # Secondary picks the next concept that isn't the primary.
        secondary_idx = (primary_idx + 1 + (variant - 1)) % len(concepts)
        if secondary_idx == primary_idx:
            secondary_idx = (primary_idx + 1) % len(concepts)

        primary = concepts[primary_idx]
        secondary = concepts[secondary_idx]
        return {
            "requires_integration": True,
            "primary_concept_id": primary["concept_id"],
            "primary_concept_label": primary["label"],
            "secondary_concept_id": secondary["concept_id"],
            "secondary_concept_label": secondary["label"],
        }


@AgentRegistry.register("correct_answer_form_planner")
class CorrectAnswerFormPlannerAgent(BaseAgent):
    """Pre-assigns the form of the correct answer: cognitive verb, option
    text shape, and the permitted concept set.

    Mirrors DistractorPlannerAgent's strategy but for the correct answer.
    Empirical pattern (D7-PHY-195 calibration, 11/20 pass rate without
    this scaffold):
      • Bloom's T3 "bare label" answers — 4/5 T3 questions failed
        because the LLM defaulted to "Compound X is the antagonist"
        instead of "predict/distinguish/evaluate" framing.
      • option_claim "X because Y" — 6/9 failures contained reasoning
        markers in option text, because that's the natural form of a
        complete answer in training data.
      • scope_match concept-bloat — when the brief lists 5+ concepts,
        the LLM weaves all of them into the correct answer while
        distractors stay at 2-3, breaking symmetric scope.

    Pre-assigning each constraint as a hard contract collapses the LLM's
    degrees of freedom upfront. The downstream gates remain as a safety
    net but stop firing at scale (same pattern as distractor uniqueness).

    Skill: deterministic rotation across (variant, tier).
      • Verb pool by tier (T1=recognition, T2=description, T3=application,
        T4=analyze/evaluate); rotation by variant for coverage.
      • Option-form constraint is static.
      • Concept cap by tier; permitted concepts seed from the tested
        concept (so distractor-plan and answer-plan stay aligned), then
        rotate through remaining concepts up to the cap.
    """
    name = "correct_answer_form_planner"

    # Tier-keyed cognitive verb pools. The LLM is required to use one of
    # these in the correct answer's claim. Verbs match Bloom's level for
    # the tier — T3 uses application/analysis verbs that the
    # BloomsCognitiveLevelGate also looks for.
    # T3 (Apply) verb pool is restricted to clean forward-action verbs
    # only. Removed: "evaluate" / "distinguish" (analysis-flavored, T4
    # territory) and "infer" (analysis-flavored AND empirically the
    # verb most associated with reasoning-marker failures: "Infer X,
    # since Y" patterns are natural for "infer" but invite option_claim
    # gate fires). T3 should be distinctly forward-looking — predict,
    # determine, apply, choose, select.
    _VERB_POOL = {
        1: ["identify", "recognize", "name", "define", "label"],
        2: ["identify", "describe", "classify", "characterize", "distinguish"],
        3: ["predict", "determine", "apply", "choose", "select"],
        4: ["integrate", "synthesize", "evaluate", "justify", "reconcile",
            "weigh"],
    }

    # Cap on how many brief concepts the correct answer may reference.
    # When briefs are concept-rich (5+ concepts on D7-PHY-195), the LLM
    # otherwise weaves them all into the correct answer, breaking the
    # symmetric-scope rule that distractors must mirror.
    _CONCEPT_CAP = {1: 1, 2: 1, 3: 2, 4: 3}

    _OPTION_FORM = (
        "Each option's `text` field MUST be a noun phrase or short "
        "declarative claim. Reasoning markers (because, since, due to) "
        "MUST NOT appear in the `text` field — they belong in the "
        "`explanation` field. Comparative connectives (whereas, but, "
        "in contrast) ARE permitted for comparison-pattern stems. "
        "Any technical term (≥6 characters) used in the correct option's "
        "text MUST also appear in at least one distractor's text — using "
        "the same vocabulary in a context that makes the distractor wrong. "
        "Vocabulary divergence between correct and distractors is a "
        "testwise tell: students recognize the unique technical word and "
        "pick it without engaging the concept."
    )

    # Length scaffold. Calibration showed prompt-only length rules get
    # ignored ~25% of the time (option_length_balance violations on D7-PHY-195
    # at A:53 B:133 type extremes). Pre-assigning a target range as a hard
    # contract closes the multi-layer pattern (prompt + scaffold + gate)
    # the same way distractor uniqueness reached ~100% compliance.
    _OPTION_LENGTH = (
        "Each option's `text` field MUST be 50-100 characters. All four "
        "options MUST be within ±25 characters of each other. The correct "
        "option's length MUST be ≤ the median distractor's length — the "
        "correct answer must NEVER be the longest of the four options. "
        "If anything, lean slightly shorter than the distractors. "
        "(Calibration showed prompt-only ±20% wording lets the LLM sit "
        "at the high end and still produce 47-53% strict-longest correct.)"
    )

    # Stop-words for key-term extraction. Aligned with KeywordDistribution
    # Gate's _STOP set so the scaffold and the gate "see" the same content
    # vocabulary. If they drift, the gate would flag terms the scaffold
    # told the LLM to ignore, and vice-versa.
    # Aliased to the canonical full stop set so this and
    # KeywordDistributionGate._STOP can never drift apart.
    _VOCAB_STOP = _BASE_FULL

    @classmethod
    def _extract_key_terms(cls, description, max_terms=6):
        """Extract technical content terms (≥6 chars) from a concept's
        description, sorted by length descending.

        Length-based ranking prioritizes the most distinctive technical
        terms (e.g., 'contralateral' over 'lesion'). Returns at most
        max_terms unique words to keep the distractor-vocabulary
        scaffold from over-constraining the LLM.
        """
        if not description:
            return []
        words = re.findall(r"\b[a-zà-öø-ÿ]{6,}\b", description.lower())
        seen = set()
        unique = []
        for w in words:
            if w in cls._VOCAB_STOP or w in seen:
                continue
            seen.add(w)
            unique.append(w)
        # Sort by length descending — longer = more technical typically.
        unique.sort(key=len, reverse=True)
        return unique[:max_terms]

    @classmethod
    def _extract_cluster_terms(cls, cluster_anchors, max_terms_per_anchor=4):
        """Phase 4a: extract vocabulary terms from cluster anchors' brief
        content (concepts' descriptions + core_claims + testable_fact).

        Cluster anchors expand the L3 vocabulary pool on narrow-domain
        primary anchors. Calibration showed BPSY agonist/antagonist
        concept descriptions yield only 5-8 technical terms — too few
        to spread across all 4 distractors, causing
        KeywordDistributionGate to fire. Cluster anchors triple the pool
        on average, giving the LLM headroom to redistribute vocabulary.

        Returns a deduplicated, length-sorted list of terms across all
        cluster anchors. Per-anchor cap prevents one verbose cluster from
        dominating.
        """
        if not cluster_anchors:
            return []
        seen = set()
        all_terms = []
        for cluster in cluster_anchors:
            cluster_text_sources = []
            for c in cluster.get("concepts", []) or []:
                cluster_text_sources.append(c.get("description", ""))
            cluster_text_sources.extend(cluster.get("core_claims", []) or [])
            cluster_text_sources.append(cluster.get("testable_fact", "") or "")
            joined = " ".join(t for t in cluster_text_sources if t)
            terms = cls._extract_key_terms(joined, max_terms=max_terms_per_anchor)
            for t in terms:
                if t not in seen:
                    seen.add(t)
                    all_terms.append(t)
        # Re-sort the merged pool by length so the most technical terms
        # bubble to the top of the cap.
        all_terms.sort(key=len, reverse=True)
        return all_terms

    def execute(self, data):
        # Coerce tier defensively: a string "3" or None silently fell
        # back to T1 verbs in the prior _VERB_POOL.get(tier, ...) call,
        # producing a T3 question with recognition verbs (Bloom's
        # contract violation). Coerce-or-default-1 makes the fallback
        # explicit and consistent.
        raw_tier = data.get("tier")
        try:
            tier = int(raw_tier) if raw_tier is not None else 1
        except (TypeError, ValueError):
            tier = 1
        variant = data.get("variant", 1)
        concepts = data.get("concepts", []) or []
        primary_concept_id = data.get("primary_concept_id")

        pool = self._VERB_POOL.get(tier, self._VERB_POOL[1])
        verb = pool[(variant - 1) % len(pool)]

        cap = self._CONCEPT_CAP.get(tier, 1)

        permitted_ids = []
        permitted_labels = []
        # L3 scaffold: extract key technical terms from each permitted
        # concept's description so the distractor section can require the
        # LLM to use them. Closes the L1+L2-only gap that left
        # KeywordDistributionGate firing at 4/14 on D7-PHY-209 phase 2 v3.
        # Without this, the LLM had no upfront list of "vocabulary that
        # must appear in distractors" — it had to guess.
        permitted_vocabulary = []
        if concepts:
            id_to_concept = {c["concept_id"]: c for c in concepts}
            id_to_label = {
                c["concept_id"]: c.get("label", "") for c in concepts
            }
            # Anchor on the tested concept so the answer-plan stays
            # aligned with the distractor-plan (which keys off the tested
            # concept's misconceptions).
            ordered_ids = []
            if primary_concept_id and primary_concept_id in id_to_label:
                ordered_ids.append(primary_concept_id)
            for c in concepts:
                cid = c["concept_id"]
                if cid not in ordered_ids:
                    ordered_ids.append(cid)
            permitted_ids = ordered_ids[:cap]
            permitted_labels = [id_to_label[i] for i in permitted_ids]

            # Extract key terms from each permitted concept's description.
            # Primary terms come first (highest priority).
            seen = set()
            for cid in permitted_ids:
                desc = (id_to_concept.get(cid, {}) or {}).get("description", "")
                for term in self._extract_key_terms(desc, max_terms=4):
                    if term not in seen:
                        seen.add(term)
                        permitted_vocabulary.append(term)

            # Phase 6: T1/T2 broader brief-internal pool. At recognition
            # and understand tiers, vocabulary IS the test — but Bloom's
            # invariant denies T1/T2 cluster anchors, so the only place
            # to grow the term pool is the primary brief itself. Pull
            # from core_claims and testable_fact (still brief-internal,
            # no external content imported). T3/T4 skip this — cluster
            # anchors handle their diversification path.
            if tier <= 2:
                pcc = data.get("primary_core_claims", []) or []
                ptf = data.get("primary_testable_fact", "") or ""
                brief_text = " ".join([ptf, *pcc])
                for term in self._extract_key_terms(brief_text, max_terms=6):
                    if term not in seen:
                        seen.add(term)
                        permitted_vocabulary.append(term)

            # Phase 4a: append cluster anchor vocabulary AFTER primary terms.
            # On narrow-domain primaries (BPSY agonist/antagonist), primary
            # alone yields ~5 terms; cluster anchors triple the pool. The
            # cap still applies — but with a larger source pool, the cap is
            # easier to fill with truly technical terms.
            cluster_anchors = data.get("cluster_anchors", []) or []
            cluster_terms = self._extract_cluster_terms(cluster_anchors,
                                                         max_terms_per_anchor=4)
            for term in cluster_terms:
                if term not in seen:
                    seen.add(term)
                    permitted_vocabulary.append(term)

            # Phase 7: T1/T2 also pull from the curated domain vocabulary
            # pool (priority 3, after concept descriptions and brief-
            # internal pool). Bloom's-compliant: vocabulary only, no
            # concepts imported. T3/T4 skip — cluster anchors are their
            # diversification path.
            if tier <= 2:
                domain_vocab = data.get("domain_vocab", []) or []
                for term in domain_vocab:
                    term_l = (term or "").lower()
                    if not term_l or len(term_l) < 6:
                        continue
                    if term_l not in seen:
                        seen.add(term_l)
                        permitted_vocabulary.append(term_l)

            # Tier-keyed cap: T1/T2 raised to 12 to give the wider pool
            # landing room (domain pool is priority 3 — cap eats domain
            # pool first if total exceeds). T3/T4 stay at 8 — strict cap
            # at apply/evaluate where vocab matching is the anti-pattern.
            cap_total = 12 if tier <= 2 else 8
            permitted_vocabulary = permitted_vocabulary[:cap_total]

        return {
            "required_verb": verb,
            "verb_pool": pool,
            "option_form_constraint": self._OPTION_FORM,
            "option_length_constraint": self._OPTION_LENGTH,
            "permitted_concept_ids": permitted_ids,
            "permitted_concept_labels": permitted_labels,
            "permitted_vocabulary": permitted_vocabulary,
            "max_concept_count": cap,
        }


@AgentRegistry.register("question_angle_selector")
class QuestionAngleSelectorAgent(BaseAgent):
    """Selects a question angle from the anchor brief by tier affinity.

    Available skill: fires only when anchor brief provides question_angles.
    Maps angle types to tiers with affinity scores, then selects the
    best-fit angle for the current tier/variant combination. Different
    variants rotate through top-scored angles for diversity.
    """
    name = "question_angle_selector"

    TIER_AFFINITY = {
        1: {"definitional": 3, "comparison": 2, "exception": 1,
            "clinical_application": 0, "neuroanatomical": 1, "mechanism": 0},
        2: {"comparison": 3, "definitional": 2, "exception": 2,
            "neuroanatomical": 1, "clinical_application": 1, "mechanism": 1},
        3: {"clinical_application": 3, "neuroanatomical": 3, "mechanism": 2,
            "exception": 2, "comparison": 1, "definitional": 0},
        4: {"mechanism": 3, "exception": 3, "comparison": 2,
            "clinical_application": 2, "neuroanatomical": 1, "definitional": 0},
    }

    def execute(self, data):
        angles = data.get("question_angles", [])
        tier = data.get("tier", 1)
        variant = data.get("variant", 1)

        if not angles:
            return {"has_angle": False}

        affinity = self.TIER_AFFINITY.get(tier, self.TIER_AFFINITY[1])

        scored = []
        for i, angle in enumerate(angles):
            score = affinity.get(angle.get("type", ""), 0)
            scored.append((score, i, angle))

        scored.sort(key=lambda x: (-x[0], x[1]))

        top_score = scored[0][0]
        top_angles = [s for s in scored if s[0] == top_score]

        if len(top_angles) >= variant:
            selected = top_angles[(variant - 1) % len(top_angles)][2]
        else:
            selected = scored[(variant - 1) % len(scored)][2]

        return {
            "has_angle": True,
            "angle_type": selected["type"],
            "angle_description": selected["description"],
        }


@AgentRegistry.register("flashcard_template")
class FlashcardTemplateAgent(BaseAgent):
    """Pre-generates flashcard fronts using templates per card type.

    Skill: template-based front generation tailored to the tested concept
    and its most common misconception. The LLM only writes backs (50%
    reduction in flashcard output tokens).
    NEW agent — previously LLM generated both fronts and backs.
    """
    name = "flashcard_template"

    def execute(self, data):
        tested_label = data.get("tested_concept_label", "this concept")
        misconceptions = data.get("misconceptions", [])

        # Concept card: direct factual question
        concept_front = (
            f"What is {tested_label}? "
            f"What are its key characteristics and functions?"
        )

        # Comparison card: contrast with most similar concept
        comparison_front = None
        if misconceptions:
            m = misconceptions[0]
            involved = m.get("concepts_involved", [])
            if len(involved) >= 2:
                a = involved[0].replace("-", " ").title()
                b = involved[1].replace("-", " ").title()
                comparison_front = (
                    f"{a} vs. {b} \u2014 "
                    f"what are the key differences between these concepts?"
                )
        if not comparison_front:
            comparison_front = (
                f"How does {tested_label} differ from "
                f"its most commonly confused concept?"
            )

        # Nuance card: contextual application
        nuance_front = (
            f"In what clinical or applied contexts does {tested_label} "
            f"function differently or have important exceptions?"
        )

        return {
            "concept_front": concept_front,
            "comparison_front": comparison_front,
            "nuance_front": nuance_front,
        }


# ══════════════════════════════════════════════════════════════
# PHASE 2: Creative Generation Agent (LLM, async)
# ══════════════════════════════════════════════════════════════

@AgentRegistry.register("question_creator")
class QuestionCreatorAgent(BaseAgent):
    """Calls Claude API to generate creative question content.

    Supports:
    - Prompt caching (90% discount on system prompt after first call per tier)
    - Model routing (Sonnet for Tier 1/2, Opus for Tier 3/4)
    - Focused output (no metadata fields — assembler handles those)
    """
    name = "question_creator"
    agent_type = "llm"

    # Per-call timeout (seconds). Empirically, Opus 4.7 generates ~2500
    # tokens in 30-60s. 120s gives 2x safety margin for queue spikes.
    # Without this, occasional Anthropic API hangs leave the orchestrator
    # waiting indefinitely — calibration regens lost 4-9 questions to
    # this on D7-PHY-209 (16/20 saved, 4 stuck mid-call). Timeout failure
    # is caught by the existing retry-on-exception path below.
    _API_TIMEOUT_SEC = 120

    async def async_execute(self, client, system_prompt, user_prompt,
                            model="claude-opus-4-7", use_cache=True,
                            max_retries=2, max_tokens=2500):
        """Async API call with prompt caching and model routing.

        Returns (creative_dict, api_meta). On failure: creative_dict is None.
        """
        # Build system content with cache control
        if use_cache:
            system_content = [{
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }]
        else:
            system_content = system_prompt

        for attempt in range(max_retries + 1):
            try:
                ts_before = datetime.now(timezone.utc)
                response = await asyncio.wait_for(
                    client.messages.create(
                        model=model,
                        max_tokens=max_tokens,
                        system=system_content,
                        messages=[{"role": "user", "content": user_prompt}],
                    ),
                    timeout=self._API_TIMEOUT_SEC,
                )
                ts_after = datetime.now(timezone.utc)
                text = response.content[0].text.strip()

                # Strip markdown code fences if present
                if text.startswith("```"):
                    text = re.sub(r"^```(?:json)?\s*", "", text)
                    text = re.sub(r"\s*```$", "", text)

                api_meta = {
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "retries": attempt,
                    "model_id": response.model,
                    "timestamp_utc": ts_before.isoformat(),
                    "latency_ms": int((ts_after - ts_before).total_seconds() * 1000),
                }

                # Check for cache usage in response
                if hasattr(response, 'usage'):
                    if hasattr(response.usage, 'cache_creation_input_tokens'):
                        api_meta["cache_creation_tokens"] = response.usage.cache_creation_input_tokens
                    if hasattr(response.usage, 'cache_read_input_tokens'):
                        api_meta["cache_read_tokens"] = response.usage.cache_read_input_tokens

                parsed = json.loads(text)
                # Fix UTF-8→CP-1252 mojibake (â€" → — etc.) in all text fields
                parsed = fix_mojibake_deep(parsed)
                return parsed, api_meta

            except json.JSONDecodeError as e:
                if attempt < max_retries:
                    await asyncio.sleep(2)
                else:
                    return None, {"retries": attempt, "error": f"json_parse: {e}"}
            except Exception as e:
                if attempt < max_retries:
                    wait = 10 * (attempt + 1)
                    await asyncio.sleep(wait)
                else:
                    return None, {"retries": attempt, "error": f"api: {e}"}


# ══════════════════════════════════════════════════════════════
# PHASE 3: Assembly Agent (hardcoded, instant, $0)
# ══════════════════════════════════════════════════════════════

# Causal-reasoning markers in option text. Mirrors OptionClaimGate's
# _REASONING_MARKERS so the assembler's auto-strip and the gate's
# detection can't drift. Calibration showed the LLM produces "X because Y"
# in option text on ~25% of generations even with explicit prompt rules
# forbidding it — natural training-data form gravity wins. The auto-strip
# moves the offending clause from option.text into option.explanation
# before validation, so the violation never reaches the gate.
_OPTION_CLAUSE_RE = re.compile(
    r"\s*[,;:.\-—]?\s*(?P<marker>"
    r"\bbecause\b"
    r"|\bsince\b"
    r"|\bdue to\b"
    r"|\bowing to\b"
    r"|\bfor this reason\b"
    r"|\bin order to\b"
    r"|\bso that\b"
    # Self-referential metacommentary — must mirror OptionClaimGate's
    # detection set (gates.py:_REASONING_MARKERS). When LLM dresses up
    # an option as "X — this is correct", the gate flags it; without
    # corresponding strip coverage, the violation reaches validation
    # instead of being neutralized in assembly.
    r"|\bthis is correct\b"
    r"|\bthis is wrong\b"
    r"|\bthis is incorrect\b"
    r")",
    re.IGNORECASE,
)


# Min absolute length the stripped claim must retain to be a meaningful
# option text. Tuned up from 4 → 20 chars after calibration showed strips
# producing stub options like "Yes, " or "True." (length-balance regression).
_MIN_STRIPPED_HEAD_LEN = 20

# Min relative length: stripped head must be at least this fraction of the
# longest sibling option's ORIGINAL length. Prevents creating a single
# 25-char stub among 80-char siblings.
_MIN_STRIPPED_RELATIVE_LEN = 0.55


def split_off_reasoning_clause(text, max_sibling_len=0):
    """Split option text into (claim, reasoning_clause) at the first
    causal-reasoning marker.

    Returns (text, "") unchanged if no marker found, OR if stripping would
    create an option that's too short to be a meaningful claim — either
    absolutely (< 20 chars) or relative to its siblings (< 55% of the
    longest sibling's original length). When the strip would create
    length imbalance, the original text is preserved so the
    OptionClaimGate fires and a feedback retry kicks in instead.

    Args:
        text: option text to split
        max_sibling_len: longest length among ORIGINAL sibling option
            texts (i.e., before any sibling was stripped). Pass 0 to
            disable the relative-length guard.
    """
    if not text:
        return text, ""
    m = _OPTION_CLAUSE_RE.search(text)
    if not m:
        return text, ""
    head = text[:m.start()].rstrip(" ,;:.-—")
    if len(head) < _MIN_STRIPPED_HEAD_LEN:
        return text, ""
    if max_sibling_len > 0 and len(head) < _MIN_STRIPPED_RELATIVE_LEN * max_sibling_len:
        return text, ""
    clause = text[m.start():].lstrip(" ,;:.-—")
    return head, clause


def merge_reasoning_into_explanation(explanation, clause):
    """Append a stripped reasoning clause to the explanation field.

    Skips the merge if the clause already appears in the explanation
    (avoiding duplication). Capitalizes the first letter when the clause
    becomes a standalone sentence.
    """
    if not clause:
        return explanation
    if not explanation:
        return clause[0].upper() + clause[1:] if clause else explanation
    if clause.lower() in explanation.lower():
        return explanation
    sep = " " if explanation.endswith((".", "!", "?")) else ". "
    return f"{explanation}{sep}{clause[0].upper() + clause[1:]}"


def auto_strip_reasoning_clauses(options):
    """Two-pass batch auto-strip across a set of options.

    Mutates the option dicts in-place: where a reasoning marker is
    found, the head replaces option["text"] and the clause is appended
    to option["explanation"].

    Why two-pass instead of looping `split_off_reasoning_clause`:
    when multiple options share the "X because Y" pattern (a common
    LLM artifact when the form-planner injects an action verb plus
    explicit reasoning), each individual single-pass strip's relative-
    length guard compared a stripped head against the longest UN-
    stripped sibling — guaranteed-shorter siblings refuse to strip
    even though stripping ALL of them would balance the lengths.

    Two-pass logic:
      1. Identify all options whose text contains a reasoning marker.
      2. Compute the relative-length floor against (a) the longest
         unstripped sibling original or (b) the longest stripped head,
         whichever is larger — i.e., what each option will look like
         after the batch strip completes.
      3. Strip each marker-bearing option whose head clears both the
         absolute and relative floors.

    The result: when ALL options share the pattern, all get stripped
    uniformly (post-strip lengths similar). When ONLY ONE option has
    the pattern, the strip is still skipped if it would create a
    sibling-length imbalance — preserving the original intent.
    """
    # Pass 1: identify marker-bearing options.
    candidates = []  # list of (idx, head, clause) for options with markers
    for i, o in enumerate(options):
        text = o.get("text", "")
        m = _OPTION_CLAUSE_RE.search(text)
        if not m:
            continue
        head = text[:m.start()].rstrip(" ,;:.-—")
        clause = text[m.start():].lstrip(" ,;:.-—")
        candidates.append((i, head, clause))

    if not candidates:
        return  # nothing to strip

    # Pass 2: compute the sibling-length reference. For unstripped
    # options we keep their original text; for stripped options we'd
    # produce the head. The reference for the relative-length guard
    # is the max across BOTH (whichever ends up longer at the end of
    # the batch strip).
    stripped_idx = {idx for idx, _, _ in candidates}
    unstripped_max = max(
        (len(options[i].get("text", ""))
         for i in range(len(options)) if i not in stripped_idx),
        default=0,
    )
    head_max = max(len(head) for _, head, _ in candidates)
    sibling_max = max(unstripped_max, head_max)

    # Pass 3: apply strips that clear both guards.
    for idx, head, clause in candidates:
        if len(head) < _MIN_STRIPPED_HEAD_LEN:
            continue
        if sibling_max > 0 and len(head) < _MIN_STRIPPED_RELATIVE_LEN * sibling_max:
            continue
        options[idx]["text"] = head
        options[idx]["explanation"] = merge_reasoning_into_explanation(
            options[idx].get("explanation", ""), clause
        )


@AgentRegistry.register("question_assembler")
class QuestionAssemblerAgent(BaseAgent):
    """Merges Phase 2 creative output with Phase 1 metadata.

    Produces the EXACT same output schema as the original
    build_question_record(). Every analytics datastring is preserved.

    Skill: deterministic JSON merge — letter assignment, distractor
    metadata attachment, confused_with resolution, topic_keywords.
    """
    name = "question_assembler"

    def execute(self, data):
        creative = data["creative"]
        plan = data["distractor_plan"]          # DistractorPlannerAgent output
        meta = data["meta_base"]                # MetadataAgent output
        topic_keywords = data["topic_keywords"] # KeywordExtractorAgent output
        target_position = data["target_position"]
        generation_metadata = data.get("generation_metadata", {})
        # Pre-assigned by TestedConceptSelectorAgent (vocab-backed)
        pre_tested = data.get("pre_tested_concept")
        # Pre-assigned by FlashcardTemplateAgent (vocab-backed)
        flashcard_fronts = data.get("flashcard_fronts")

        # ── Resolve tested_concept ────────────────────────────
        # Priority: pre-assigned (hardcoded) > LLM output
        if pre_tested and pre_tested.get("has_tested_concept"):
            tested_concept = {
                "concept_id": pre_tested["concept_id"],
                "concept_label": pre_tested["concept_label"],
                # knowledge_tested always from LLM (creative content)
                "knowledge_tested": (
                    creative.get("knowledge_tested")
                    or creative.get("tested_concept", {}).get("knowledge_tested", "")
                ),
            }
        else:
            tested_concept = creative.get("tested_concept")
            if not tested_concept or not tested_concept.get("concept_id"):
                raise ValueError("LLM output missing tested_concept with concept_id")

        letters = ["A", "B", "C", "D"]
        correct_idx = letters.index(target_position)

        # ── Correct option ────────────────────────────────────
        correct_option = {
            "letter": target_position,
            "text": creative["correct_answer"]["text"],
            "is_correct": True,
            "explanation": creative["correct_answer"].get("explanation", ""),
            "concept_id": tested_concept["concept_id"],
            "concept_label": tested_concept["concept_label"],
            "distractor_level": None,
            "confused_with": None,
            "misconception_id": None,
            "misconception_label": None,
            "misconception_type": None,
        }

        # ── Distractor options ────────────────────────────────
        distractor_letters = [l for i, l in enumerate(letters) if i != correct_idx]
        distractor_options = []

        # Guard: the LLM must produce exactly 3 distractors and the plan
        # must have exactly 3 slots. A length mismatch would IndexError
        # below — surface a meaningful error instead so the orchestrator
        # can route this through the retry path with a useful failure
        # reason instead of crashing the whole batch.
        creative_distractors = creative.get("distractors") or []
        plan_slots = plan.get("slots") or []
        n_letters = len(distractor_letters)
        if len(creative_distractors) < n_letters or len(plan_slots) < n_letters:
            raise ValueError(
                f"distractor count mismatch: expected {n_letters} "
                f"(got {len(creative_distractors)} from LLM and "
                f"{len(plan_slots)} from planner). "
                "LLM must return exactly 3 distractors; planner must "
                "produce exactly 3 slots."
            )

        for i, letter in enumerate(distractor_letters):
            slot = plan_slots[i]
            dist_creative = creative_distractors[i]

            option = {
                "letter": letter,
                "text": dist_creative["text"],
                "is_correct": False,
                "explanation": dist_creative.get("explanation", ""),
                "distractor_level": slot["distractor_level"],
                "confused_with": tested_concept["concept_id"],
            }

            option["concept_id"] = dist_creative.get("concept_id")
            option["concept_label"] = dist_creative.get("concept_label")
            # Pre-assigned misconception_id from plan takes priority over LLM output
            option["misconception_id"] = slot.get("misconception_id") or dist_creative.get("misconception_id")
            option["misconception_label"] = slot.get("misconception_label") or dist_creative.get("misconception_label")
            option["misconception_type"] = slot.get("misconception_type") or dist_creative.get("misconception_type")

            distractor_options.append(option)

        # ── Assemble in letter order ──────────────────────────
        all_options = [correct_option] + distractor_options
        all_options.sort(key=lambda o: o["letter"])

        # ── Auto-strip causal-reasoning clauses from option.text ─
        # Deterministic safety net for OptionClaimGate. The form-planner
        # contract tells the LLM not to write "X because Y" in option
        # text, but training-data gravity wins ~25% of the time. Moving
        # the offending clause into the explanation field before
        # validation eliminates the violation without retry cost.
        # Two-pass batch strip handles the case where multiple options
        # share the "X because Y" pattern — single-pass refused to strip
        # any of them because each individual strip's head was shorter
        # than the unstripped siblings; batch strip evaluates the relative-
        # length guard against post-strip siblings so uniform strips pass.
        auto_strip_reasoning_clauses(all_options)

        # ── Flashcard seeds: merge hardcoded fronts + LLM backs
        raw_seeds = creative.get("flashcard_seeds", {})
        # Handle both full format and backs-only format
        flashcard_seeds = {}
        for card_type in ("concept", "comparison", "nuance"):
            seed = raw_seeds.get(card_type, {})
            if isinstance(seed, dict):
                back = seed.get("back", seed.get(card_type, ""))
                front = seed.get("front", "")
            elif isinstance(seed, str):
                back = seed
                front = ""
            else:
                back = ""
                front = ""
            # Always prefer LLM-generated fronts — they're question-specific
            # and higher quality than hardcoded templates. Fall back to
            # template fronts only if the LLM didn't produce a front.
            if not front:
                front_key = f"{card_type}_front"
                if flashcard_fronts and flashcard_fronts.get(front_key):
                    front = flashcard_fronts[front_key]
            flashcard_seeds[card_type] = {"front": front, "back": back}

        # Always prefer LLM-generated keywords; fall back to hardcoded
        kw = creative.get("topic_keywords", []) or topic_keywords

        # ── Final record ──────────────────────────────────────
        return {
            "question_id": meta["question_id"],
            "question_type": "single_choice",
            "domain_id": meta["domain_id"],
            "domain_code": meta["domain_code"],
            "domain_name": meta["domain_name"],
            "chapter_file": meta["chapter_file"],
            "chapter_title": meta["chapter_title"],
            "section_title": meta.get("section_title") or meta["chapter_title"],
            "chapter_uuid": meta.get("chapter_uuid"),
            "anchor_uuid": meta.get("anchor_uuid"),
            "anchor_label": meta.get("anchor_label", ""),
            "difficulty_tier": meta["difficulty_tier"],
            "difficulty_label": meta["difficulty_label"],
            "blooms_primary": meta["blooms_primary"],
            "blooms_secondary": meta["blooms_secondary"],
            "source_type": meta["source_type"],
            "stem_pattern": meta["stem_pattern"],
            "correct_answer_letter": target_position,
            "anchor_uids": meta.get("anchor_uids", []),
            "anchor_point_ids_v2": meta.get("anchor_point_ids_v2", []),
            "anchor_content_summaries": meta.get("anchor_content_summaries", []),
            "testable_fact": meta.get("testable_fact", ""),
            "tested_concept": tested_concept,
            "question_stem": creative["question_stem"],
            "options": all_options,
            "flashcard_seeds": flashcard_seeds,
            "topic_keywords": kw,
            "book_reference": {
                "chapter": meta["chapter_title"],
                "section": meta.get("section_title") or meta["chapter_title"],
                "anchor": meta.get("anchor_label", ""),
            },
            "content_snippet_chars": meta.get("content_snippet_chars", 0),
            "variant": meta["variant"],
            "generation_batch": meta["batch_id"],
            "generated_by": generation_metadata.get("model_id", "claude-opus-4-7"),
            "generation_metadata": generation_metadata,
        }


