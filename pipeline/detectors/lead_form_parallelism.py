"""lead_form_parallelism detector — Phase A5d.

Detects when the four options of an MCQ don't share grammatical lead
form. Inconsistent lead forms ("Dopamine" / "The patient shows..." /
"Increased activation" / "When stress is present...") create stylistic
asymmetry that test-takers can sometimes exploit — the option whose
form differs is signaled as either correct or incorrect by its
prominence.

Heuristic: classify each option's first content token by a coarse
shape (verb-ing, noun, definite article, gerund, ...). If 4 options
have ≥2 distinct shapes, fire.

Tier-blind: parallelism is a style invariant at every tier.

Verdict: VERDICT_ADVISORY on fire (low-confidence regex heuristic;
not authoritative enough to BLOCK at gen-time without false-positive
risk on legitimate clinical prose). Confidence 0.6.
"""
from __future__ import annotations

import re

from . import (
    Detector,
    DetectorSignal,
    PHASE_AUDIT,
    PHASE_GENERATION,
    VERDICT_ADVISORY,
)


_DEFINITE_ARTICLE_RE = re.compile(r"^(the|a|an)\s+", re.IGNORECASE)


def _classify_lead_shape(text: str) -> str:
    """Return a coarse shape label for the option's first content token.

    Shapes:
      - "DEF_ART_NP"   — starts with "The/A/An" (definite-article noun phrase)
      - "VERB_ING"     — starts with -ing verb
      - "VERB_PAST"    — starts with -ed verb (heuristic; many adjectives end -ed too)
      - "VERB_BASE"    — starts with a recognized base imperative-like verb
                         (see imperative_lead detector); rare since that
                         detector blocks them, but kept for completeness.
      - "WHEN_IF"      — starts with subordinating conjunction (when, if,
                         after, during, because)
      - "PROPER_NP"    — starts with a capitalized non-The word (proper noun)
      - "NP"           — default: anything that doesn't match above
    """
    if not text:
        return "EMPTY"
    stripped = text.strip()
    if not stripped:
        return "EMPTY"

    if _DEFINITE_ARTICLE_RE.match(stripped):
        return "DEF_ART_NP"

    first_word_match = re.match(r"^(\w+)", stripped)
    if not first_word_match:
        return "NP"
    first_word = first_word_match.group(1)
    lower = first_word.lower()

    # Subordinating conjunctions / temporal prefix
    if lower in {"when", "if", "after", "during", "because", "since",
                 "while", "though", "although", "until"}:
        return "WHEN_IF"

    # -ing verb (gerund or progressive participle)
    if lower.endswith("ing") and len(lower) > 4:
        return "VERB_ING"

    # -ed verb (past tense; could also be adjective — heuristic)
    if lower.endswith("ed") and len(lower) > 3:
        return "VERB_PAST"

    # Capitalized non-article = proper noun (Dopamine, GABA, Wernicke)
    if first_word[0].isupper():
        return "PROPER_NP"

    return "NP"


class LeadFormParallelismDetector(Detector):
    """Fires when the four options' lead shapes diverge.

    Threshold: ≥2 distinct shapes across the 4 options. Two shapes is
    the minimum signal — three or four diverse shapes is a stronger
    parallelism violation.
    """

    detector_id = "lead_form_parallelism"
    phases = (PHASE_GENERATION, PHASE_AUDIT)

    def scan(
        self,
        question: dict,
        context: dict | None = None,
    ) -> list[DetectorSignal]:
        question = question or {}
        options = question.get("options") or []
        if len(options) < 2:
            return [DetectorSignal(
                detector_id=self.detector_id,
                letter=None,
                fired=False,
                confidence=0.0,
                signature=None,
                verdict_action=VERDICT_ADVISORY,
                reason="too few options to assess parallelism",
            )]

        shapes = {}
        for opt in options:
            letter = opt.get("letter", "?")
            text = opt.get("text", "") or ""
            shapes[letter] = _classify_lead_shape(text)

        unique_shapes = set(shapes.values())
        if len(unique_shapes) < 2:
            return [DetectorSignal(
                detector_id=self.detector_id,
                letter=None,
                fired=False,
                confidence=1.0,
                signature=None,
                verdict_action=VERDICT_ADVISORY,
                reason=f"all options share lead shape: {next(iter(unique_shapes))}",
            )]

        # Fire on divergence. Letter=None (whole-question scope; no single
        # option is the issue — the SET diverges).
        return [DetectorSignal(
            detector_id=self.detector_id,
            letter=None,
            fired=True,
            confidence=0.6,
            signature="lead_form_divergence",
            verdict_action=VERDICT_ADVISORY,
            reason=(
                f"options have {len(unique_shapes)} distinct lead shapes: "
                f"{dict(shapes)}"
            ),
            extra={"shapes": dict(shapes)},
        )]


__all__ = ["LeadFormParallelismDetector"]
