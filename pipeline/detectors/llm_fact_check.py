"""llm_fact_check detector — Phase A7.

Opus-backed factual-correctness verifier for T4 questions only.

Why T4 only: factual errors at lower tiers are caught by the existing
diagnostic_quality factual_correctness audit (Sonnet, single-pass,
~$0.005/q). T4 vignettes have richer clinical detail and require
deeper specialty knowledge that Opus is meaningfully better at
verifying. The cost differential ($0.05 Opus vs $0.005 Sonnet) is
justified only at the highest-stakes tier where the existing audit
has the lowest factual_correctness scores.

Cost: ~$0.05/q on T4 only. At a 25% T4 ratio, amortized
~$0.012/audited-question.
"""
from __future__ import annotations

import json

from . import (
    Detector,
    DetectorSignal,
    PHASE_AUDIT_LLM,
    VERDICT_ADVISORY,
)


_OPUS_MODEL_ID = "claude-opus-4-7"

_PROMPT = """You are a domain expert fact-checking a multiple-choice question for FACTUAL CORRECTNESS. The question is at Bloom's Tier 4 (Evaluate) — clinically rich, with specific factual claims in stem and options.

For each option, verify the factual claims it makes against established knowledge in the domain. Flag options whose factual claims are INCORRECT or MISLEADING — even if the option is intended as a distractor (a wrong CONCEPT), distractors should not contain wrong FACTS that confuse the test-taker about the correct answer's basis.

CRITICAL RULES:

1. The keyed answer's claims must be factually correct. Flag if not.
2. Distractor claims should be wrong on the CONCEPT being tested, not on tangential facts. Flag distractors whose tangential facts are also wrong (this confuses the discrimination).
3. Be specific about which claim is wrong and why. Cite the correct fact.

QUESTION:

Domain: {domain}
Tier: {tier}

STEM:
{stem}

OPTIONS:
{options_block}

OUTPUT — single JSON object (no preamble, no markdown):

{{
  "factual_errors": [
    {{"letter": "X", "claim": "<the wrong claim quoted>", "correction": "<the correct fact>"}},
    ...one entry per option with a factual error; empty list if all factually correct...
  ]
}}"""


def _build_options_block(options) -> str:
    lines = []
    for o in options or []:
        marker = "[CORRECT]" if o.get("is_correct") else "[distractor]"
        lines.append(f"  {o.get('letter','?')} {marker}: {o.get('text','')}")
    return "\n".join(lines)


def _parse_response(text: str) -> dict | None:
    if not text:
        return None
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").lstrip("json").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


class LlmFactCheckDetector(Detector):
    """Opus-backed factual-correctness detector. T4 only.

    Async — must be called via the registry's async path."""

    detector_id = "llm_fact_check"
    phases = (PHASE_AUDIT_LLM,)

    def __init__(self, client=None, semaphore=None) -> None:
        self._client = client
        self._semaphore = semaphore

    def scan(self, question, context=None):
        raise NotImplementedError(
            "LlmFactCheckDetector requires async invocation"
        )

    async def async_scan(
        self,
        question: dict,
        context: dict | None = None,
    ) -> list[DetectorSignal]:
        question = question or {}
        tier = question.get("difficulty_tier")

        # T4 only — defense in depth even though phase tag controls dispatch.
        if tier != 4:
            return [DetectorSignal(
                detector_id=self.detector_id,
                letter=None,
                fired=False,
                confidence=0.0,
                signature=None,
                verdict_action=VERDICT_ADVISORY,
                reason=f"llm_fact_check: T4 only (got T{tier})",
            )]

        client = (context or {}).get("client") or self._client
        semaphore = (context or {}).get("semaphore") or self._semaphore
        if client is None or semaphore is None:
            return [DetectorSignal(
                detector_id=self.detector_id,
                letter=None,
                fired=False,
                confidence=0.0,
                signature=None,
                verdict_action=VERDICT_ADVISORY,
                reason="llm_fact_check: no client / semaphore in context",
            )]

        prompt = _PROMPT.format(
            domain=question.get("domain_code") or "?",
            tier=tier,
            stem=question.get("question_stem", "") or "",
            options_block=_build_options_block(question.get("options")),
        )

        async with semaphore:
            try:
                response = await client.messages.create(
                    model=_OPUS_MODEL_ID,
                    max_tokens=1024,
                    temperature=0,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text if response.content else ""
                usage = {
                    "input_tokens": response.usage.input_tokens,
                    "output_tokens": response.usage.output_tokens,
                }
            except Exception as e:
                return [DetectorSignal(
                    detector_id=self.detector_id,
                    letter=None,
                    fired=False,
                    confidence=0.0,
                    signature=None,
                    verdict_action=VERDICT_ADVISORY,
                    reason=f"llm_fact_check: api_error: {type(e).__name__}: {e}",
                )]

        parsed = _parse_response(text)
        if not parsed:
            return [DetectorSignal(
                detector_id=self.detector_id,
                letter=None,
                fired=False,
                confidence=0.0,
                signature=None,
                verdict_action=VERDICT_ADVISORY,
                reason="llm_fact_check: parse_failed",
                extra={"raw_response": text[:200], "usage": usage},
            )]

        errors = parsed.get("factual_errors") or []
        if not isinstance(errors, list):
            errors = []

        signals: list[DetectorSignal] = []
        if not errors:
            signals.append(DetectorSignal(
                detector_id=self.detector_id,
                letter=None,
                fired=False,
                confidence=1.0,
                signature=None,
                verdict_action=VERDICT_ADVISORY,
                reason="llm_fact_check: no factual errors found",
                extra={"usage": usage},
            ))
            return signals

        for entry in errors:
            if not isinstance(entry, dict):
                continue
            letter = entry.get("letter")
            claim = entry.get("claim") or ""
            correction = entry.get("correction") or ""
            signals.append(DetectorSignal(
                detector_id=self.detector_id,
                letter=letter,
                fired=True,
                confidence=0.7,  # higher confidence than ambiguity (Opus + T4)
                signature="llm_fact_check",
                verdict_action=VERDICT_ADVISORY,
                reason=f"factual error: '{claim[:80]}' — correction: {correction[:80]}",
                extra={"claim": claim, "correction": correction, "usage": usage},
            ))
        return signals


__all__ = ["LlmFactCheckDetector"]
