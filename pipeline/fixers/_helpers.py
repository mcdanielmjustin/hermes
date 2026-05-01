"""Shared helpers for fixer modules — Phase B4 condensation.

Extracted from per-fixer JSON-parsing duplicates that appeared in:
  - ambiguity_fixer
  - universal_quantifier_fixer
  - numeric_ratio_fixer
  - (and the LLM-backed detectors llm_ambiguity, llm_fact_check)

Single source of truth for the LLM-response parsing pattern.
"""
from __future__ import annotations

import json


def parse_fixer_json(text: str | None) -> dict | None:
    """Parse an LLM's JSON response, tolerating code-fence wrapping.

    Returns the parsed dict on success, or None on:
      - Empty/None input
      - Malformed JSON
      - Non-dict top-level value

    Tolerates these wrappers (common LLM output decorations):
      - Triple backticks (```)
      - "json" language tag after backticks (```json)
      - Leading/trailing whitespace
    """
    if not text:
        return None
    cleaned = text.strip()
    if not cleaned:
        return None

    # Strip code-fence markers if present.
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        # After stripping backticks, "json" might prefix the JSON body.
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        parsed = json.loads(cleaned)
    except (json.JSONDecodeError, ValueError):
        return None

    if not isinstance(parsed, dict):
        return None
    return parsed


__all__ = ["parse_fixer_json"]
