"""
QuestionOrchestrator — Single source of truth for the pipeline graph.

Coordinates all pipeline phases with two skill categories:
  ASSIGNED  — Always fire. Deterministic, $0, instant.
  AVAILABLE — Fire conditionally based on anchor data / context.
              Add intelligence without breaking the pipeline when
              their input data is missing.

Both generate_quiz_questions.py and batch_generate.py use this class
instead of manually coordinating agents.
"""

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

# Phase 20-revert (2026-04-29): brief content is no longer delivered
# to the generation prompt by default.
#
# Empirical basis (today's controlled tests + 9-anchor batch):
#   - briefed generation: ~50-67% pass rate (varied by batch)
#   - un-briefed generation: ~86% pass rate (n=72 sample)
#   - briefed + stem-hygiene mitigation: 45% (TESTED — actively
#     hurt vs un-mitigated briefed)
# At 32K-question scale, un-briefed saves ~$8K + simpler pipeline.
#
# Briefs REMAIN authored on disk and are still consumed by the
# rescue pipeline (rescue_chapter_stems.py reads concept_explanation
# for stem-rewrite templates). Only the GENERATION path skips brief
# content delivery.
#
# To re-enable for testing/experiments, set the env var:
#   export GOLIATH_USE_BRIEF_FOR_GENERATION=1
USE_BRIEF_FOR_GENERATION: bool = (
    os.getenv("GOLIATH_USE_BRIEF_FOR_GENERATION", "0").lower()
    in ("1", "true", "yes")
)

from . import (
from pipeline.api_client import create_client
    CORRECT_POSITIONS, get_source_type, get_stem_pattern,
)
from .agents import (
    AnchorBriefAgent,
    AnchorClusterAgent,
    ConceptIntegrationPlannerAgent,
    ConceptVocabAgent,
    CorrectAnswerFormPlannerAgent,
    DistractorPlannerAgent,
    InputSanitizerAgent,
    KeywordExtractorAgent,
    MetadataAgent,
    TestedConceptSelectorAgent,
    FlashcardTemplateAgent,
    QuestionAngleSelectorAgent,
    QuestionCreatorAgent,
    QuestionAssemblerAgent,
    extract_brief_vocabulary_terms,
)


def _extract_brief_concept_terms(concepts):
    """Extract canonical vocabulary terms from a brief's concept list.

    Used to populate `brief_concept_terms` on the task so the T3+
    keyword gates can restrict their unique-to-correct count to
    canonical technical vocabulary (the same terms the L3 scaffold
    extracts from concept descriptions).
    """
    if not concepts:
        return []
    fake_brief = {"concepts": concepts}
    return sorted(extract_brief_vocabulary_terms(fake_brief))
from .prompts import build_system_prompt, build_user_prompt, build_correction_prompt
from .names import get_character_assignment
from .gates import create_gate_pipeline


# Module-level cache so parallel orchestrator instances (e.g., one per
# worker in a multi-process generation run) don't each re-read the same
# vocabulary JSON. Keyed by (domain_vocab_dir, domain_code). Reads are
# deterministic so the second-write-wins race on cache mutation is
# benign — no lock needed.
_DOMAIN_VOCAB_CACHE: dict[tuple, list] = {}


class QuestionOrchestrator:
    """Coordinates the full quiz question generation pipeline.

    Phases:
      1a. Chapter context   — concept vocab (once per chapter)
      1b. Anchor context    — anchor brief + fallback resolution (once per anchor)
      1c. Task preparation  — per-variant agents + prompt building
      2.  Creative generation (LLM call)
      3.  Assembly          — merge creative output with metadata
      4.  Validation        — gate pipeline
      5.  Smart retry       — feedback-driven correction prompt
    """

    ASSIGNED_SKILLS = {
        "concept_vocab": "Load chapter-level concept vocabulary",
        "anchor_brief": "Load per-anchor analysis brief",
        "input_sanitizer": "Strip researcher citations from anchor inputs",
        "keyword_extractor": "Extract topic keywords from content + concepts",
        "metadata": "Build deterministic question metadata",
        "tested_concept_selector": "Pre-select tested concept by rotation",
        "concept_integration_planner": "Pre-assign 2 concepts for T4 integration scaffold",
        "correct_answer_form_planner": "Pre-assign cognitive verb, option form, and permitted concept set for the correct answer",
        "anchor_cluster": "Select 1-2 sibling anchors for vocabulary headroom + T3+ integration substrate",
        "flashcard_template": "Pre-generate flashcard front templates",
        "distractor_planner": "Pre-assign distractor levels + misconceptions",
    }

    AVAILABLE_SKILLS = {
        "question_angle_selector": "Select question angle from brief by tier affinity",
    }

    def __init__(self):
        self.concept_vocab_agent = ConceptVocabAgent()
        self.anchor_brief_agent = AnchorBriefAgent()
        self.input_sanitizer = InputSanitizerAgent()
        self.keyword_extractor = KeywordExtractorAgent()
        self.metadata_agent = MetadataAgent()
        self.tested_concept_selector = TestedConceptSelectorAgent()
        self.concept_integration_planner = ConceptIntegrationPlannerAgent()
        self.correct_answer_form_planner = CorrectAnswerFormPlannerAgent()
        self.anchor_cluster_agent = AnchorClusterAgent()
        self.flashcard_template = FlashcardTemplateAgent()
        self.distractor_planner = DistractorPlannerAgent()

        self.question_angle_selector = QuestionAngleSelectorAgent()

        self.creator = QuestionCreatorAgent()
        self.assembler = QuestionAssemblerAgent()

        self.gates = create_gate_pipeline()

        # Phase A1: detector registry alongside (not replacing) gates.
        # Detectors are read-only at gen-time in A1 (gates still drive
        # the failure decision). Phase A3 wires detector BLOCK signals
        # into the gate-loop equivalent path.
        from pipeline.detectors.registry import create_detector_registry
        self.detectors = create_detector_registry()

    def info(self):
        n_assigned = len(self.ASSIGNED_SKILLS)
        n_available = len(self.AVAILABLE_SKILLS)
        n_gates = len(self.gates)
        return (
            f"{n_assigned} assigned + {n_available} available skills, "
            f"1 LLM agent, 1 assembler, {n_gates} gates"
        )

    def _load_domain_vocab(self, domain_vocab_dir, domain_code):
        """Phase 7: load curated/bootstrapped domain vocabulary pool.

        Reads {domain_vocab_dir}/{domain_code}.json once per (dir, code)
        pair, caches the term list at module scope so parallel
        orchestrator instances share the load. Missing file → empty list
        (pipeline degrades cleanly to current behavior at T1/T2).

        The pool gives T1/T2 distractors broader vocabulary to share with
        the correct option without violating Bloom's identity (vocabulary-
        only import, no concepts). Cluster anchors handle T3/T4.
        """
        if domain_vocab_dir is None:
            return []
        cache_key = (str(domain_vocab_dir), domain_code)
        if cache_key in _DOMAIN_VOCAB_CACHE:
            return _DOMAIN_VOCAB_CACHE[cache_key]

        path = Path(domain_vocab_dir) / f"{domain_code}.json"
        terms: list = []
        if path.exists():
            try:
                with open(path, encoding="utf-8") as f:
                    data = json.load(f)
                terms = list(data.get("vocabulary", []) or [])
            except (OSError, json.JSONDecodeError):
                terms = []
        _DOMAIN_VOCAB_CACHE[cache_key] = terms
        return terms

    # ══════════════════════════════════════════════════════════════
    # Phase 1a: Chapter Context
    # ══════════════════════════════════════════════════════════════

    def load_chapter_context(self, concept_vocab_dir, domain_code, chapter_id):
        """Load chapter-level concept vocabulary. Call once per chapter."""
        return self.concept_vocab_agent.run({
            "concept_vocab_dir": concept_vocab_dir,
            "domain_code": domain_code,
            "chapter_id": chapter_id,
        })

    # ══════════════════════════════════════════════════════════════
    # Phase 1b: Anchor Context
    # ══════════════════════════════════════════════════════════════

    def load_anchor_context(self, anchor_briefs_dir, domain_code, uid,
                            chapter_vocab, passage_text=""):
        """Load anchor-level context with fallback to chapter vocab.

        Returns dict with: has_brief, active_concepts, active_misconceptions,
        core_claims, question_angles, active_vocab, anchor_has_vocab,
        topic_keywords, cluster_anchors.
        """
        brief_result = self.anchor_brief_agent.run({
            "anchor_briefs_dir": anchor_briefs_dir,
            "domain_code": domain_code,
            "uid": uid,
        })

        # Phase 20-revert: gate brief-content delivery to the
        # generation prompt on USE_BRIEF_FOR_GENERATION. The brief
        # is still loaded (so concept_explanation + discriminators
        # below flow to downstream stem-rewrite consumers); only the
        # generation-path fields fall through to chapter vocab when
        # the constant is False.
        use_brief_for_gen = (
            brief_result.get("has_brief") and USE_BRIEF_FOR_GENERATION
        )
        if use_brief_for_gen:
            active_concepts = brief_result["concepts"]
            active_misconceptions = brief_result["misconceptions"]
            # Strip researcher citations ("Squire (2004):") from core claims
            # so they don't leak into the prompt and bias the LLM.
            core_claims = [
                self.input_sanitizer.sanitize(c)
                for c in brief_result["core_claims"]
            ]
            question_angles = brief_result["question_angles"]
            anchor_has_vocab = True
            active_vocab = {
                "has_vocab": True,
                "concepts": active_concepts,
                "misconceptions": active_misconceptions,
            }
        else:
            # Default path post-Phase-20-revert: chapter vocab fallback.
            # Runs even when brief exists on disk if
            # USE_BRIEF_FOR_GENERATION is False.
            ch_concepts = chapter_vocab.get("concepts", [])
            ch_misconceptions = chapter_vocab.get("misconceptions", [])
            active_concepts = ch_concepts
            active_misconceptions = ch_misconceptions
            core_claims = []
            question_angles = []
            anchor_has_vocab = chapter_vocab.get("has_vocab", False)
            active_vocab = chapter_vocab

        # Phase 20c: structured fields for stem-rewrite templates. Loaded
        # whether or not the brief has concepts (concept_explanation can
        # be authored independently). Default empty for legacy briefs.
        concept_explanation = brief_result.get("concept_explanation", "") or ""
        discriminators = brief_result.get("discriminators", []) or []

        keyword_result = self.keyword_extractor.run({
            "content": passage_text[:3000] if passage_text else "",
            "concepts": active_concepts,
        })

        # Cluster anchors are NOT loaded here — they're loaded in prepare_task
        # because the cluster count is tier-keyed (T1/T2 = 0 clusters, T3 = 1,
        # T4 = 2). Loading here would either over-cluster T1/T2 (violating
        # anchor identity at recognition/description tiers) or require slicing
        # downstream. Instead, prepare_task calls the cluster agent with the
        # current task's tier so the count is correct from the start.

        return {
            "has_brief": brief_result.get("has_brief", False),
            "active_concepts": active_concepts,
            "active_misconceptions": active_misconceptions,
            "core_claims": core_claims,
            "question_angles": question_angles,
            # Phase 20c: passed through for downstream stem-rewrite
            # consumers (rescue pipeline, future generation-time
            # template fills). Empty strings/lists for legacy briefs.
            "concept_explanation": concept_explanation,
            "discriminators": discriminators,
            "active_vocab": active_vocab,
            "anchor_has_vocab": anchor_has_vocab,
            "topic_keywords": keyword_result.get("topic_keywords", []),
            # The next two fields are passed through for prepare_task's
            # tier-keyed cluster loading at T3/T4. T1/T2 won't use them
            # because the cluster agent returns [] at those tiers.
            "primary_chapter_num": brief_result.get("chapter_num", "")
                                    if brief_result.get("has_brief") else "",
            "anchor_briefs_dir": anchor_briefs_dir,
            "domain_code": domain_code,
            "primary_uid": uid,
        }

    # ══════════════════════════════════════════════════════════════
    # Phase 1c: Per-Variant Task Preparation
    # ══════════════════════════════════════════════════════════════

    def prepare_task(self, anchor, anchor_context, tier, variant,
                     domain_code, domain_id, domain_name,
                     chapter_title, section_title, batch_id,
                     anchor_idx=0, total_tiers_count=4, variants_per_tier=5,
                     domain_vocab_dir=None, prompt_version="v2"):
        """Run all assigned + applicable available skills for one question.

        Returns a task dict ready for generate_and_validate().

        Phase 7: when `domain_vocab_dir` is provided, the curated/
        bootstrapped domain vocabulary pool is loaded once per domain and
        passed to the form planner; at T1/T2 it broadens permitted
        vocabulary so distractors have richer cross-content terms to
        share with the correct option (Bloom's-compliant — vocabulary
        only, no concepts imported).
        """
        uid = anchor["uid"]
        id_v2 = anchor.get("anchor_point_id_v2", "")
        passage = anchor.get("passage", "")
        # ── Assigned: input_sanitizer ─────────────────────────────
        # Strip "Author (YYYY):" citations from anchor text fields so the
        # LLM never sees researcher names it would otherwise echo into stems.
        verbatim = self.input_sanitizer.sanitize(anchor.get("verbatim_anchor", ""))
        testable_fact = self.input_sanitizer.sanitize(anchor.get("testable_fact", ""))
        content_chars = min(len(passage.strip()), 3000) if passage else 0

        active_concepts = anchor_context["active_concepts"]
        active_misconceptions = anchor_context["active_misconceptions"]
        anchor_has_vocab = anchor_context["anchor_has_vocab"]

        # ── Assigned: tested_concept_selector ────────────────────
        tested_concept_result = self.tested_concept_selector.run({
            "concepts": active_concepts,
            "variant": variant,
            "tier": tier,
        })

        # ── Assigned: concept_integration_planner ────────────────
        # Tier 4 only — pre-assigns 2 concepts the question MUST integrate.
        # Mirrors DistractorPlannerAgent's scaffolding strategy: collapse the
        # LLM's "decide which concepts to integrate" decision so the prompt
        # carries an explicit constraint instead of a hopeful instruction.
        integration_result = self.concept_integration_planner.run({
            "tier": tier,
            "variant": variant,
            "concepts": active_concepts,
        })

        # ── Assigned: anchor_cluster ─────────────────────────────
        # Bloom's-identity-keyed clustering: T1/T2 get 0 cluster anchors
        # (recognition/description tests are anchored to ONE brief by
        # definition); T3 gets 1 (application requires cross-content);
        # T4 gets 2 (analyze/evaluate IS cross-content integration). The agent
        # returns [] at T1/T2 — invariant enforced in the agent's
        # _CLUSTER_COUNT_BY_TIER table.
        #
        # Option 5: pass primary's core_claims and testable_fact too so
        # the cluster agent can compare extracted vocabulary terms
        # (the actual L3 source) when scoring sibling diversity.
        cluster_result = self.anchor_cluster_agent.run({
            "anchor_briefs_dir": anchor_context.get("anchor_briefs_dir"),
            "primary_uid": anchor_context.get("primary_uid", uid),
            "domain_code": anchor_context.get("domain_code", domain_code),
            "primary_chapter_num": anchor_context.get("primary_chapter_num", ""),
            "primary_concepts": active_concepts,
            "primary_core_claims": anchor_context.get("core_claims", []),
            "primary_testable_fact": testable_fact,
            "tier": tier,
        })
        cluster_anchors = cluster_result.get("cluster_anchors", []) or []

        # ── Assigned: correct_answer_form_planner ────────────────
        # Pre-assigns the cognitive verb, option text shape, and permitted
        # concept set for the correct answer — the architectural counterpart
        # to DistractorPlanner. Without this scaffold, the LLM defaulted to
        # bare-label answers (T3), self-justifying option text, and
        # concept-bloat correct answers.
        # Phase 7: load curated domain vocab pool. Empty list when
        # domain_vocab_dir is not configured or the JSON file is missing.
        domain_vocab = self._load_domain_vocab(domain_vocab_dir, domain_code)

        answer_form_result = self.correct_answer_form_planner.run({
            "tier": tier,
            "variant": variant,
            "concepts": active_concepts,
            "primary_concept_id": tested_concept_result.get("concept_id"),
            # T1/T2 receive [] here (per Bloom's invariant); T3/T4 receive
            # 1-2 cluster anchor briefs whose vocabulary supplements the
            # primary brief's term pool in the L3 scaffold.
            "cluster_anchors": cluster_anchors,
            # Phase 6: T1/T2 also pull from these for the broader brief-
            # internal vocab pool (T3/T4 ignore — cluster anchors are
            # their diversification path).
            "primary_core_claims": anchor_context.get("core_claims", []),
            "primary_testable_fact": testable_fact,
            # Phase 7: T1/T2 broaden further from the curated domain pool.
            # T3/T4 ignore — cluster anchors are their path. Empty list
            # when the file is missing → degrades to Phase 6 behavior.
            "domain_vocab": domain_vocab,
        })

        # ── Assigned: distractor_planner ─────────────────────────
        # Pass form planner's permitted_concept_ids so the misconception pool
        # gets filtered to those tied to in-scope concepts. Without this, a
        # misconception about an off-topic concept could be assigned to a
        # distractor — students would eliminate it by recognizing the topic
        # mismatch instead of by reasoning about the tested concept.
        plan_result = self.distractor_planner.run({
            "tier": tier,
            "variant": variant,
            "misconceptions": active_misconceptions,
            "tested_concept_id": tested_concept_result.get("concept_id"),
            "permitted_concept_ids": answer_form_result.get(
                "permitted_concept_ids", []
            ),
        })

        # ── Assigned: flashcard_template ─────────────────────────
        flashcard_result = self.flashcard_template.run({
            "tested_concept_label": (
                tested_concept_result.get("concept_label", "this concept")
                if tested_concept_result.get("has_tested_concept")
                else "this concept"
            ),
            "misconceptions": active_misconceptions,
        })

        # ── Assigned: metadata ───────────────────────────────────
        pattern_name, _ = get_stem_pattern(tier, variant)
        source_type = get_source_type(tier, pattern_name)
        if not passage and source_type != "anchor_grounded":
            source_type = "anchor_grounded"

        pos_idx = (
            anchor_idx * total_tiers_count * variants_per_tier
            + (tier - 1) * variants_per_tier
            + (variant - 1)
        ) % len(CORRECT_POSITIONS)
        target_position = CORRECT_POSITIONS[pos_idx]

        metadata_result = self.metadata_agent.run({
            "domain_code": domain_code,
            "domain_id": domain_id,
            "domain_name": domain_name,
            "chapter_title": chapter_title,
            "chapter_num": anchor.get("chapter_num", ""),
            "section_title": section_title,
            "tier": tier,
            "variant": variant,
            "source_type": source_type,
            "stem_pattern": pattern_name,
            "anchor_uid": uid,
            "anchor_id_v2": id_v2,
            "verbatim_anchor": verbatim,
            "testable_fact": testable_fact,
            "batch_id": batch_id,
            "content_chars": content_chars,
        })

        question_id = metadata_result["question_id"]

        # ── Available: question_angle_selector ───────────────────
        question_angle = None
        if anchor_context.get("question_angles"):
            angle_result = self.question_angle_selector.run({
                "question_angles": anchor_context["question_angles"],
                "tier": tier,
                "variant": variant,
            })
            if angle_result.get("has_angle"):
                question_angle = angle_result

        # ── Build prompts ────────────────────────────────────────
        prompt_mode = "focused" if anchor_has_vocab else "open"
        # P6 v2: derive pedagogical flavor from anchor UID + domain_code
        # (mechanism / framework / cognitive_process / etc.). Used by
        # the v2 system prompt to select the matching preferred-
        # wrongness-mode block. v1 ignores flavor entirely.
        from .anchor_flavor import flavor_for_anchor
        flavor = flavor_for_anchor(uid, domain_code)
        system_prompt = build_system_prompt(
            tier, mode=prompt_mode,
            prompt_version=prompt_version, flavor=flavor,
        )

        character = get_character_assignment(question_id)

        anchor_info = {
            "chapter_title": chapter_title,
            "anchor_id_v2": id_v2,
            "uid": uid,
        }
        anchor_data = [{
            "uid": uid, "id_v2": id_v2,
            "verbatim_anchor": verbatim, "testable_fact": testable_fact,
            "text": verbatim, "id": id_v2,
        }]
        sub_content = passage.strip()[:3000] if passage else ""

        user_prompt = build_user_prompt(
            anchor_info, sub_content, anchor_data,
            source_type, variant, domain_name,
            difficulty_tier=tier,
            concept_vocab=anchor_context["active_vocab"] if anchor_has_vocab else None,
            character=character,
            target_position=target_position,
            tested_concept=tested_concept_result,
            distractor_plan=plan_result,
            core_claims=anchor_context["core_claims"],
            question_angle=question_angle,
            concept_integration=integration_result,
            correct_answer_form=answer_form_result,
        )

        return {
            "question_id": question_id,
            "checkpoint_key": question_id,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "tier": tier,
            "target_position": target_position,
            "has_concept_vocab": anchor_has_vocab,
            "distractor_plan": plan_result,
            "anchor_concept_ids": (
                [c["concept_id"] for c in active_concepts]
                if anchor_context.get("has_brief") else None
            ),
            # Used by BloomsCognitiveLevelGate (T4 path) and DomainExpertiseGate
            # to verify the question integrates real brief vocabulary.
            "anchor_concept_labels": (
                {c["concept_id"]: c.get("label", "") for c in active_concepts}
                if anchor_context.get("has_brief") else None
            ),
            "anchor_testable_fact": testable_fact,
            "meta_base": metadata_result["meta_base"],
            "model": "claude-opus-4-7",
            "max_tokens": 2500,
            # P6 v2 reproducibility: record which prompt version + flavor
            # produced this question so audits can A/B-compare cohorts.
            "prompt_version": prompt_version,
            "flavor": flavor,
            "pre_tested_concept": tested_concept_result,
            "flashcard_fronts": flashcard_result,
            "topic_keywords": anchor_context["topic_keywords"],
            "anchor_brief": {
                "uid": uid,
                "has_brief": anchor_context.get("has_brief", False),
                "core_claims": anchor_context.get("core_claims", []),
                "concepts": active_concepts,
                "misconceptions": active_misconceptions,
                "question_angles": anchor_context.get("question_angles", []),
                # Phase 20c: structured fields for stem-rewrite templates
                # and any future generation-time template-fills. Empty
                # for legacy briefs.
                "concept_explanation": anchor_context.get("concept_explanation", ""),
                "discriminators": anchor_context.get("discriminators", []),
            },
            # Expose form-planner output and clusters on the task for
            # introspection and downstream tooling. Auditors and future
            # retry interventions read these without re-running planners.
            "correct_answer_form": answer_form_result,
            "cluster_anchors": cluster_anchors,
            # T3+ keyword gates' canonical-vocabulary filter consumes
            # these via gate_context: domain_vocab is the curated pool
            # (T1/T2 also use it for permitted_vocabulary); brief_
            # concept_terms is the per-anchor concept-description vocab
            # extracted via the same helper as L3 scaffolding.
            "domain_vocab": domain_vocab,
            "brief_concept_terms": _extract_brief_concept_terms(active_concepts),
        }

    # ══════════════════════════════════════════════════════════════
    # Phase 2–5: Generate, Assemble, Validate, Retry
    # ══════════════════════════════════════════════════════════════

    async def generate_and_validate(self, client, task,
                                    gate_context=None, max_attempts=2):
        """Run LLM generation through validation with smart retry.

        Args:
            client: anthropic.AsyncAnthropic instance
            task: dict from prepare_task()
            gate_context: extra context for gates (e.g. existing_ids)
            max_attempts: max generation attempts (1 original + N-1 retries)

        Returns:
            (assembled_dict, total_tokens_in, total_tokens_out) on success
            (None, total_tokens_in, total_tokens_out) on failure
            Also returns failure_reason as 4th element on failure.
        """
        if gate_context is None:
            gate_context = {}
        if task.get("anchor_concept_ids"):
            gate_context["anchor_concept_ids"] = task["anchor_concept_ids"]
        if task.get("anchor_concept_labels"):
            gate_context["anchor_concept_labels"] = task["anchor_concept_labels"]
        if task.get("anchor_testable_fact"):
            gate_context["anchor_testable_fact"] = task["anchor_testable_fact"]
        # T3+ keyword gates use these to restrict the unique-to-correct
        # count to canonical technical vocabulary (curated domain pool +
        # brief concept-description terms). Without this, generic English
        # descriptors trigger false positives at apply/evaluate tiers.
        if task.get("domain_vocab"):
            gate_context["domain_vocab"] = task["domain_vocab"]
        if task.get("brief_concept_terms"):
            gate_context["brief_concept_terms"] = task["brief_concept_terms"]
        # ── Distractor policy matrix inputs ───────────────────
        # These signals feed pipeline.distractor_policy.resolve() so
        # the StemEliminableDistractorGate dispatches on the matrix
        # cell. domain_code / source_type / stem_pattern live on
        # task["meta_base"] (set by MetadataAgent), not at task top level.
        meta_base = task.get("meta_base") or {}
        if meta_base.get("domain_code"):
            gate_context["domain_code"] = meta_base["domain_code"]
        if meta_base.get("stem_pattern"):
            gate_context["stem_pattern"] = meta_base["stem_pattern"]
        if meta_base.get("source_type"):
            gate_context["source_type"] = meta_base["source_type"]
        # pedagogical_content_type lives on the anchor brief once P2
        # adds it; until then default to "unknown" so resolve() falls
        # through to DEFAULT and behavior is unchanged.
        anchor_brief = task.get("anchor_brief") or {}
        gate_context["pedagogical_content_type"] = (
            anchor_brief.get("pedagogical_content_type") or "unknown"
        )

        correction_prompt = None
        total_in = 0
        total_out = 0
        # Carries the most recent successfully-assembled (but gate-
        # rejected) question across attempts. Returned to the caller on
        # total failure so diagnostic tooling can inspect what was
        # generated, not just the failure reason.
        last_failed_assembly = None
        reason = ""

        for attempt in range(max_attempts):
            # ── Phase 2: Creative generation ─────────────────────
            user_prompt = correction_prompt or task["user_prompt"]
            # Use new client interface
            if hasattr(client, 'generate'):
                # New OpenAI-compatible client
                creative, api_meta = await client.generate(
                    task["system_prompt"],
                    user_prompt,
                    max_tokens=task.get("max_tokens", 2500),
                )
            else:
                # Legacy Anthropic client via QuestionCreatorAgent
                creative, api_meta = await self.creator.async_execute(
                    client, task["system_prompt"], user_prompt,
                    model=task.get("model", "claude-opus-4-7"),
                    use_cache=True,
                    max_tokens=task.get("max_tokens", 2500),
                )

            if api_meta:
                total_in += api_meta.get("prompt_tokens", 0)
                total_out += api_meta.get("completion_tokens", 0)

            if not creative:
                reason = api_meta.get("error", "null creative output") if api_meta else "no response"
                if attempt < max_attempts - 1:
                    await asyncio.sleep(2)
                continue

            # ── Phase 3: Assembly ────────────────────────────────
            generation_metadata = {
                "timestamp_utc": api_meta.get("timestamp_utc", datetime.now(timezone.utc).isoformat()),
                "prompt_tokens": api_meta.get("prompt_tokens", 0),
                "completion_tokens": api_meta.get("completion_tokens", 0),
                "retries": api_meta.get("retries", 0),
                "model_id": api_meta.get("model_id", "claude-opus-4-7"),
                "latency_ms": api_meta.get("latency_ms", 0),
                "has_concept_vocab": task["has_concept_vocab"],
                "correction_retry": attempt > 0,
                # P6 v2 reproducibility: which prompt version + flavor
                # produced this question. Lets audits filter cohorts
                # for A/B comparison (filter saved questions by
                # generation_metadata.prompt_version).
                "prompt_version": task.get("prompt_version", "v2"),
                "flavor": task.get("flavor"),
            }

            try:
                assembled = self.assembler.run({
                    "creative": creative,
                    "distractor_plan": task["distractor_plan"],
                    "meta_base": task["meta_base"],
                    "topic_keywords": task["topic_keywords"],
                    "target_position": task["target_position"],
                    "generation_metadata": generation_metadata,
                    "pre_tested_concept": task.get("pre_tested_concept"),
                    "flashcard_fronts": task.get("flashcard_fronts"),
                })
            except Exception as e:
                assembled = {"_error": f"assembly crash: {e}"}

            if "_error" in assembled:
                reason = assembled["_error"]
                if attempt < max_attempts - 1:
                    correction_prompt = build_correction_prompt(
                        task["user_prompt"], [("assembly", reason)],
                        tier=task.get("tier"),
                    )
                    await asyncio.sleep(2)
                continue

            # ── Phase 4: Validation ──────────────────────────────
            # Prerequisite gates short-circuit on first failure (downstream
            # would crash on missing structure). Content-level gates
            # (is_prerequisite=False) are collected so a single retry can
            # address attribution + length-balance + grounding together.
            # Async gates (e.g. stem_eliminable_distractor) return a
            # coroutine; orchestrator awaits transparently. The audit
            # token accumulator collects (in, out) tuples from any gate
            # that makes API calls, so they roll into the question's
            # total before return.
            gate_context["client"] = client
            gate_context["audit_token_acc"] = []
            failures = []
            for gate in self.gates:
                try:
                    result = gate.check(assembled, gate_context)
                    if asyncio.iscoroutine(result):
                        result = await result
                    gate_ok, gate_reason = result
                except Exception as e:
                    gate_ok, gate_reason = False, f"gate crashed: {e}"
                if not gate_ok:
                    failures.append((gate.name, gate_reason))
                    if getattr(gate, "is_prerequisite", True):
                        break

            for in_tok, out_tok in gate_context["audit_token_acc"]:
                total_in += in_tok
                total_out += out_tok

            # ── Phase A3: deterministic detector pass at gen-time ─
            # Mirrors the audit-time detector pass (same registry, same
            # threshold logic from `pipeline/distractor_policy.py`'s cell
            # matrix). When a detector emits OVERRIDE_TO at gen-time
            # (i.e. the signal would change the audit verdict if the
            # question shipped), treat it as a gate failure and re-prompt
            # with a targeted correction. The question would have been
            # flagged at audit anyway; catching it here saves a Sonnet
            # round-trip + downstream fix loop.
            #
            # Behind GOLIATH_DETECTORS_AT_GEN env flag (default off) for
            # measurement before default-on. The existing max_attempts
            # loop caps retries, so latency overhead is bounded.
            if os.environ.get("GOLIATH_DETECTORS_AT_GEN") == "1":
                # Lazy import to avoid loading detectors module unless flag is set.
                from pipeline.detectors import (
                    PHASE_GENERATION, VERDICT_BLOCK, VERDICT_OVERRIDE_TO,
                )
                detector_context = {
                    "tier": task.get("tier"),
                    "source_type": gate_context.get("source_type"),
                    "stem_pattern": gate_context.get("stem_pattern"),
                    "domain_code": gate_context.get("domain_code"),
                }
                # Ensure the assembled question carries the cell-resolution
                # inputs the detector reads (tier from task, others from gate
                # context). Detectors look at question.get("difficulty_tier")
                # and question.get("source_type") etc.
                if "difficulty_tier" not in assembled and task.get("tier"):
                    assembled = dict(assembled)
                    assembled["difficulty_tier"] = task.get("tier")
                for k in ("source_type", "stem_pattern", "domain_code"):
                    if k not in assembled and gate_context.get(k):
                        if not isinstance(assembled, dict):
                            continue
                        assembled = dict(assembled)
                        assembled[k] = gate_context.get(k)

                try:
                    detector_signals = self.detectors.scan_for_phase(
                        PHASE_GENERATION, assembled, context=detector_context,
                    )
                except Exception as e:
                    detector_signals = []
                    failures.append((
                        "detector:registry_crashed",
                        f"{type(e).__name__}: {e}",
                    ))

                for sig in detector_signals:
                    if not sig.fired:
                        continue
                    if sig.verdict_action not in (VERDICT_BLOCK, VERDICT_OVERRIDE_TO):
                        continue
                    failures.append((
                        f"detector:{sig.detector_id}",
                        f"{sig.signature or 'fired'} on letter {sig.letter}: {sig.reason}",
                    ))

            if not failures:
                return assembled, total_in, total_out, None

            reason = "; ".join(f"{g}: {r}" for g, r in failures)
            # Keep the most recent failed assembly available so the
            # caller can inspect it diagnostically. Without this, gate
            # fires that survive both attempts produce no inspectable
            # content — the prompt's payload is gone, and the LLM's
            # response is gone with it.
            last_failed_assembly = assembled

            # ── Phase 5: Smart retry ─────────────────────────────
            if attempt < max_attempts - 1:
                correction_prompt = build_correction_prompt(
                    task["user_prompt"], failures,
                    tier=task.get("tier"),
                )
                await asyncio.sleep(2)

        # On total failure, return the last assembled question (if any
        # was produced) so the caller can opt to log it for diagnostics.
        # The `result is None` contract for the caller is preserved by
        # tagging the returned record with `_failed_validation: True`.
        if last_failed_assembly is not None:
            last_failed_assembly = dict(last_failed_assembly)
            last_failed_assembly["_failed_validation"] = True
            last_failed_assembly["_failure_reason"] = reason
        return last_failed_assembly, total_in, total_out, reason
