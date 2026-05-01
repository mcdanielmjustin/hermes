"""llm_ambiguity detector — Phase A7.

Sonnet-backed detector that asks "for each option, is it defensibly
correct under any reasonable interpretation of the stem?" Surfaces
ambiguity that regex can't see — the WISC age-band-style overlap zones,
DSM diagnostic-criterion edge cases, framework-application boundary
cases.

Runs at PHASE_AUDIT_LLM, AFTER deterministic detectors pass. This
ordering ensures regex verdicts (which are higher-confidence) take
precedence; LLM-backed signals only surface when the deterministic
layer left the question unflagged.

Verdict: VERDICT_ADVISORY initially (always advisory until calibration
data shows correlation >= 0.85 with existing dq audit ambiguity flag).
Then later iterations may promote to OVERRIDE_TO via the cell matrix.

Cost: ~$0.005-0.01 per question (Sonnet single-pass with constrained
JSON output).
"""
from __future__ import annotations

import json

from . import (
    Detector,
    DetectorSignal,
    PHASE_AUDIT_LLM,
    VERDICT_ADVISORY,
)


_SONNET_MODEL_ID = "claude-sonnet-4-6"

_PROMPT = """You are a psychometrics auditor checking a multiple-choice question for AMBIGUITY — cases where multiple options could be defended as correct under a reasonable interpretation of the stem.

Your job: identify each option that a knowledgeable test-taker could DEFEND as correct, given any plausible reading of the stem (clinical edge cases, overlap zones, alternative theoretical frameworks). The keyed answer is given; you flag DISTRACTORS that are also defensibly correct.

CRITICAL RULES:

1. Be CONSERVATIVE. Flag only options where a competent candidate could mount a SPECIFIC argument for correctness — not options that are merely plausible-sounding.

2. The keyed answer is correct by stipulation. You're checking whether OTHER options are ALSO defensibly correct.

3. State the alternative reading concretely. "Could be defended because of clinical-judgment override" is not specific enough. Specific: "WAIS-IV is defensibly correct because the age 16:0 falls within both WISC-V (6:0–16:11) and WAIS-IV (16:0–90:11) ranges."

QUESTION:

Domain: {domain}
Tier: {tier}

STEM:
{stem}

OPTIONS:
{options_block}

OUTPUT — single JSON object (no preamble, no markdown):

{{
  "defensible_alternatives": [
    {{"letter": "X", "argument": "<1-2 sentences naming the specific alternative interpretation that makes X defensible>"}},
    ...one entry per defensibly-correct distractor; empty list if no ambiguity...
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
    # Strip code-fence markers if present.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`").lstrip("json").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return None


class LlmAmbiguityDetector(Detector):
    """Sonnet-backed ambiguity detector. Async — must be called via the
    registry's `scan_for_phase_async` method (sync `scan` raises)."""

    detector_id = "llm_ambiguity"
    phases = (PHASE_AUDIT_LLM,)

    def __init__(self, client=None, semaphore=None) -> None:
        # Allow injection for testing; production calls pass via context.
        self._client = client
        self._semaphore = semaphore

    def scan(
        self,
        question: dict,
        context: dict | None = None,
    ) -> list[DetectorSignal]:
        """Sync scan is unsupported for LLM-backed detectors. Use
        `await detector.async_scan(...)` via the registry's async path."""
        raise NotImplementedError(
            "LlmAmbiguityDetector requires async invocation via "
            "DetectorRegistry.scan_for_phase_async (PHASE_AUDIT_LLM)"
        )

    async def async_scan(
        self,
        question: dict,
        context: dict | None = None,
    ) -> list[DetectorSignal]:
        question = question or {}
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
                reason="llm_ambiguity: no client / semaphore in context",
            )]

        prompt = _PROMPT.format(
            domain=question.get("domain_code") or "?",
            tier=question.get("difficulty_tier") or "?",
            stem=question.get("question_stem", "") or "",
            options_block=_build_options_block(question.get("options")),
        )

        async with semaphore:
            try:
                response = await client.messages.create(
                    model=_SONNET_MODEL_ID,
                    max_tokens=512,
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
                    reason=f"llm_ambiguity: api_error: {type(e).__name__}: {e}",
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
                reason="llm_ambiguity: parse_failed",
                extra={"raw_response": text[:200], "usage": usage},
            )]

        defensible = parsed.get("defensible_alternatives") or []
        if not isinstance(defensible, list):
            defensible = []

        signals: list[DetectorSignal] = []
        if not defensible:
            signals.append(DetectorSignal(
                detector_id=self.detector_id,
                letter=None,
                fired=False,
                confidence=1.0,
                signature=None,
                verdict_action=VERDICT_ADVISORY,
                reason="llm_ambiguity: no defensible alternatives found",
                extra={"usage": usage},
            ))
            return signals

        for entry in defensible:
            if not isinstance(entry, dict):
                continue
            letter = entry.get("letter")
            argument = entry.get("argument") or ""
            signals.append(DetectorSignal(
                detector_id=self.detector_id,
                letter=letter,
                fired=True,
                confidence=0.6,  # initially low — calibrate against dq audit
                signature="llm_ambiguity",
                verdict_action=VERDICT_ADVISORY,
                reason=argument[:200],
                extra={"argument": argument, "usage": usage},
            ))
        return signals


__all__ = ["LlmAmbiguityDetector"]
