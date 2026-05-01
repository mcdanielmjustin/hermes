"""Unit tests for `pipeline.self_critique`.

Phase 25 production self-critique. Tests cover:
  - Merge preserves is_correct from original (the structural is_correct
    safeguard from Test 1's false-positive lesson)
  - Merge preserves slot/concept_id/misconception_id fields the critique
    response doesn't include
  - revised=false path returns original unchanged
  - Parse failures return original unchanged with errors recorded
  - API errors return original unchanged
  - opus_cost calculation
"""
from __future__ import annotations

import asyncio
import pathlib
import sys
import types

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pipeline.self_critique import (  # noqa: E402
    self_critique_question, opus_cost, OPUS_MODEL_ID,
    SELF_CRITIQUE_PROMPT,
)


# ── Fake API client ─────────────────────────────────────────

class _FakeUsage:
    def __init__(self, input_tokens=100, output_tokens=200):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens


class _FakeContent:
    def __init__(self, text):
        self.text = text


class _FakeResponse:
    def __init__(self, text, input_tokens=100, output_tokens=200):
        self.content = [_FakeContent(text)]
        self.usage = _FakeUsage(input_tokens, output_tokens)


class _FakeMessages:
    def __init__(self, response_text=None, exception=None,
                 input_tokens=100, output_tokens=200):
        self._response_text = response_text
        self._exception = exception
        self._input = input_tokens
        self._output = output_tokens
        self.calls = []

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        if self._exception is not None:
            raise self._exception
        return _FakeResponse(self._response_text, self._input, self._output)


class _FakeClient:
    def __init__(self, response_text=None, exception=None,
                 input_tokens=100, output_tokens=200):
        self.messages = _FakeMessages(
            response_text, exception, input_tokens, output_tokens,
        )


def _q(stem="Sample stem.", correct_letter="B"):
    """Build a synthetic question with 4 options, one correct."""
    options = []
    for letter in ("A", "B", "C", "D"):
        options.append({
            "letter": letter,
            "text": f"option {letter} text",
            "is_correct": (letter == correct_letter),
            "explanation": f"explanation {letter}",
            "slot": ord(letter) - ord("A"),
            "concept_id": f"concept-{letter.lower()}",
            "misconception_id": f"misc-{letter.lower()}" if letter != correct_letter else None,
            "misconception_type": "similar_property" if letter != correct_letter else None,
        })
    return {
        "question_id": "TEST-Q1",
        "difficulty_tier": 2,
        "question_stem": stem,
        "options": options,
    }


def _run(coro):
    return asyncio.run(coro)


# ── Merge preserves is_correct ─────────────────────────────

def test_merge_preserves_is_correct_even_if_critique_swaps():
    """The structural is_correct safeguard: even if Opus tries to mark
    a different option correct in its response, the original mapping
    wins. This is the lesson from Test 1's false-positive incident."""
    response = """{
      "self_audit": [
        {"letter": "A", "class": "english_gap", "reason": "test"},
        {"letter": "C", "class": "content_gap", "reason": "test"},
        {"letter": "D", "class": "clean", "reason": "test"}
      ],
      "revised": true,
      "rationale": "test revision",
      "question": {
        "question_stem": "Revised stem.",
        "options": [
          {"letter": "A", "text": "new A", "is_correct": true, "explanation": "new"},
          {"letter": "B", "text": "new B", "is_correct": false, "explanation": "new"},
          {"letter": "C", "text": "new C", "is_correct": false, "explanation": "new"},
          {"letter": "D", "text": "new D", "is_correct": false, "explanation": "new"}
        ]
      }
    }"""
    client = _FakeClient(response_text=response)
    semaphore = asyncio.Semaphore(1)
    q = _q(correct_letter="B")  # Original: B is correct
    result = _run(self_critique_question(client, q, semaphore))

    assert result["patched"] is True
    out = result["question"]
    # Despite Opus saying A is correct in its response, the merge MUST
    # use the original is_correct mapping: B remains correct, A remains
    # incorrect.
    by_letter = {o["letter"]: o for o in out["options"]}
    assert by_letter["A"]["is_correct"] is False
    assert by_letter["B"]["is_correct"] is True
    assert by_letter["C"]["is_correct"] is False
    assert by_letter["D"]["is_correct"] is False
    # Text DID update to the revised version
    assert by_letter["A"]["text"] == "new A"
    assert by_letter["B"]["text"] == "new B"


def test_merge_preserves_metadata_fields_critique_omits():
    """slot, concept_id, misconception_id should carry over from the
    original since the critique doesn't include them."""
    response = """{
      "self_audit": [
        {"letter": "A", "class": "english_gap", "reason": "test"},
        {"letter": "C", "class": "clean", "reason": "test"},
        {"letter": "D", "class": "clean", "reason": "test"}
      ],
      "revised": true,
      "rationale": "test",
      "question": {
        "question_stem": "Revised.",
        "options": [
          {"letter": "A", "text": "new A", "is_correct": false, "explanation": "new"},
          {"letter": "B", "text": "new B", "is_correct": true, "explanation": "new"},
          {"letter": "C", "text": "new C", "is_correct": false, "explanation": "new"},
          {"letter": "D", "text": "new D", "is_correct": false, "explanation": "new"}
        ]
      }
    }"""
    client = _FakeClient(response_text=response)
    semaphore = asyncio.Semaphore(1)
    q = _q()
    result = _run(self_critique_question(client, q, semaphore))

    out = result["question"]
    by_letter = {o["letter"]: o for o in out["options"]}
    # Original metadata preserved
    assert by_letter["A"]["slot"] == 0
    assert by_letter["A"]["concept_id"] == "concept-a"
    assert by_letter["A"]["misconception_id"] == "misc-a"
    assert by_letter["A"]["misconception_type"] == "similar_property"
    # Correct option metadata also preserved
    assert by_letter["B"]["slot"] == 1
    assert by_letter["B"]["concept_id"] == "concept-b"


def test_revised_false_returns_original_unchanged():
    """If the critique decides no revision is needed, the original
    question is returned unchanged."""
    response = """{
      "self_audit": [
        {"letter": "A", "class": "content_gap", "reason": "test"},
        {"letter": "C", "class": "clean", "reason": "test"},
        {"letter": "D", "class": "clean", "reason": "test"}
      ],
      "revised": false,
      "rationale": "no changes needed",
      "question": {
        "question_stem": "Sample stem.",
        "options": []
      }
    }"""
    client = _FakeClient(response_text=response)
    semaphore = asyncio.Semaphore(1)
    q = _q()
    result = _run(self_critique_question(client, q, semaphore))

    assert result["patched"] is False
    assert result["question"] == q  # Returned unchanged
    assert result["rationale"] == "no changes needed"
    assert result["errors"] == []


def test_stem_revision_lands():
    """When revised=true, the stem is updated."""
    response = """{
      "self_audit": [],
      "revised": true,
      "rationale": "rewrote stem",
      "question": {
        "question_stem": "Brand new stem after revision.",
        "options": [
          {"letter": "A", "text": "kept A", "is_correct": false, "explanation": "kept"},
          {"letter": "B", "text": "kept B", "is_correct": true, "explanation": "kept"},
          {"letter": "C", "text": "kept C", "is_correct": false, "explanation": "kept"},
          {"letter": "D", "text": "kept D", "is_correct": false, "explanation": "kept"}
        ]
      }
    }"""
    client = _FakeClient(response_text=response)
    semaphore = asyncio.Semaphore(1)
    q = _q(stem="Original stem.")
    result = _run(self_critique_question(client, q, semaphore))

    assert result["patched"] is True
    assert result["question"]["question_stem"] == "Brand new stem after revision."


# ── Failure modes return original unchanged ────────────────

def test_api_error_returns_original():
    client = _FakeClient(exception=RuntimeError("API down"))
    semaphore = asyncio.Semaphore(1)
    q = _q()
    result = _run(self_critique_question(client, q, semaphore))

    assert result["patched"] is False
    assert result["question"] == q
    assert any("api_error" in e for e in result["errors"])


def test_parse_failure_returns_original():
    client = _FakeClient(response_text="not valid json")
    semaphore = asyncio.Semaphore(1)
    q = _q()
    result = _run(self_critique_question(client, q, semaphore))

    assert result["patched"] is False
    assert result["question"] == q
    assert "parse_failed" in result["errors"]


def test_no_options_short_circuits():
    client = _FakeClient(response_text="{}")
    semaphore = asyncio.Semaphore(1)
    q = {"question_id": "TEST", "options": []}
    result = _run(self_critique_question(client, q, semaphore))

    assert result["patched"] is False
    assert result["errors"] == ["no_options"]
    # API never called
    assert len(client.messages.calls) == 0


# ── Cost calculation ──────────────────────────────────────

def test_opus_cost_basic():
    cost = opus_cost({"input_tokens": 1_000_000, "output_tokens": 1_000_000})
    # Opus 4.7: $15/M in + $75/M out
    assert abs(cost - 90.0) < 0.001


def test_opus_cost_typical_call():
    """Typical self-critique call: ~5K input, ~500 output."""
    cost = opus_cost({"input_tokens": 5000, "output_tokens": 500})
    expected = 5000 / 1e6 * 15.0 + 500 / 1e6 * 75.0  # 0.075 + 0.0375 = 0.1125
    assert abs(cost - expected) < 0.0001
    # Should be ~$0.005 per typical question
    assert 0.005 < cost < 0.15  # generous range; just a sanity bound


def test_opus_cost_zero_usage():
    assert opus_cost({"input_tokens": 0, "output_tokens": 0}) == 0.0
    assert opus_cost({}) == 0.0


# ── Sanity on prompt content ──────────────────────────────

def test_prompt_has_preservation_block():
    """The ABSOLUTE PRESERVATION RULES block must be in the prompt —
    this is what (combined with the merge logic) prevents the Test 1
    is_correct-swap incident."""
    assert "ABSOLUTE PRESERVATION RULES" in SELF_CRITIQUE_PROMPT
    assert "MUST remain `is_correct: true`" in SELF_CRITIQUE_PROMPT
    assert "NEVER swap which option is marked correct" in SELF_CRITIQUE_PROMPT


def test_prompt_includes_audit_rubric():
    """The audit's exact rubric (the four classes) must be in the
    prompt so Opus is critiquing against the same standard the audit
    uses."""
    assert "ENGLISH_GAP" in SELF_CRITIQUE_PROMPT
    assert "CONTENT_GAP" in SELF_CRITIQUE_PROMPT
    assert "CLEAN" in SELF_CRITIQUE_PROMPT
    assert "SOFT_FLAG" in SELF_CRITIQUE_PROMPT
    # The Lester/wedding canonical example
    assert "Lester Nichols" in SELF_CRITIQUE_PROMPT


def test_model_id_is_opus_4_7():
    assert OPUS_MODEL_ID == "claude-opus-4-7"


# ── Standalone runner ──────────────────────────────────────

if __name__ == "__main__":
    import inspect
    funcs = [f for n, f in globals().items()
             if n.startswith("test_") and inspect.isfunction(f)]
    failures = []
    for f in funcs:
        try:
            f()
            print(f"PASS {f.__name__}")
        except AssertionError as e:
            failures.append((f.__name__, str(e)))
            print(f"FAIL {f.__name__}: {e}")
        except Exception as e:
            failures.append((f.__name__, f"{type(e).__name__}: {e}"))
            print(f"ERROR {f.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(funcs) - len(failures)}/{len(funcs)} passed")
    sys.exit(1 if failures else 0)
