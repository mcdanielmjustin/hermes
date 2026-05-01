"""Unit tests for ship_readiness._reconcile_cross_model.

Phase 21c reconciliation logic: pure function, no API calls. Tests cover:
- Per-distractor agreement/disagreement scenarios
- Conservative english_gap preservation on disagreement
- Soft_flag escalation on non-english_gap disagreement
- Failure-mode fallback (one side errored, both errored)
- Multi-question chapters with mixed verdicts
- flagged_distractors re-derivation from reconciled classifications
"""
from __future__ import annotations

import sys
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from ship_readiness import _reconcile_cross_model


def _result(qid, classifications, error=None, model_id="sonnet"):
    """Build an audit-result dict in the shape audit_question returns."""
    r = {
        "question_id": qid,
        "classifications": classifications,
        "flagged_distractors": [],
        "usage": {"input_tokens": 100, "output_tokens": 20},
        "error": error,
        "model_id": model_id,
    }
    return r


def _classify(letter, cls, **extras):
    e = {"letter": letter, "class": cls, "distractor_text": f"d-{letter}",
         "explanation": f"why-{cls}"}
    if cls in ("english_gap", "content_gap"):
        e["contradicted_stem_fact"] = f"fact-{cls}"
    if cls == "soft_flag":
        e["ambiguous_between"] = extras.get("ambiguous_between",
                                             ["english_gap", "clean"])
    return e


# ── Agreement ────────────────────────────────────────────────

def test_full_agreement_keeps_sonnet_verdict():
    s = _result("Q1", [_classify("A", "english_gap"), _classify("B", "clean")])
    h = _result("Q1", [_classify("A", "english_gap"), _classify("B", "clean")],
                model_id="haiku")
    out = _reconcile_cross_model([s], [h])
    assert len(out) == 1
    classes = {c["letter"]: c for c in out[0]["classifications"]}
    assert classes["A"]["class"] == "english_gap"
    assert classes["B"]["class"] == "clean"
    assert out[0]["cross_model_disagreement_count"] == 0


def test_agreement_no_disagreement_marker():
    s = _result("Q1", [_classify("A", "content_gap")])
    h = _result("Q1", [_classify("A", "content_gap")], model_id="haiku")
    out = _reconcile_cross_model([s], [h])
    assert "cross_model_disagreement" not in out[0]["classifications"][0]


# ── english_gap conservatism ─────────────────────────────────

def test_disagreement_sonnet_eg_haiku_clean_keeps_eg():
    s = _result("Q1", [_classify("A", "english_gap")])
    h = _result("Q1", [_classify("A", "clean")], model_id="haiku")
    out = _reconcile_cross_model([s], [h])
    a = out[0]["classifications"][0]
    assert a["class"] == "english_gap"
    assert a["cross_model_disagreement"] is True
    assert a["cross_model_other_class"] == "clean"
    assert out[0]["cross_model_disagreement_count"] == 1


def test_disagreement_sonnet_clean_haiku_eg_promotes_to_eg():
    """Conservative: Haiku catches english_gap, escalate even though
    Sonnet said clean. Use Haiku's entry (carries contradicted_stem_fact)."""
    s = _result("Q1", [_classify("A", "clean")])
    h = _result("Q1", [_classify("A", "english_gap")], model_id="haiku")
    out = _reconcile_cross_model([s], [h])
    a = out[0]["classifications"][0]
    assert a["class"] == "english_gap"
    assert a["cross_model_disagreement"] is True
    assert a["cross_model_other_class"] == "clean"
    # english_gap entries should carry contradicted_stem_fact
    assert "contradicted_stem_fact" in a


def test_disagreement_sonnet_eg_haiku_content_gap_keeps_eg():
    s = _result("Q1", [_classify("A", "english_gap")])
    h = _result("Q1", [_classify("A", "content_gap")], model_id="haiku")
    out = _reconcile_cross_model([s], [h])
    a = out[0]["classifications"][0]
    assert a["class"] == "english_gap"
    assert a["cross_model_disagreement"] is True


# ── Non-english_gap disagreement → soft_flag ─────────────────

def test_disagreement_content_gap_vs_clean_becomes_soft_flag():
    s = _result("Q1", [_classify("A", "content_gap")])
    h = _result("Q1", [_classify("A", "clean")], model_id="haiku")
    out = _reconcile_cross_model([s], [h])
    a = out[0]["classifications"][0]
    assert a["class"] == "soft_flag"
    assert a["cross_model_disagreement"] is True
    assert sorted(a["ambiguous_between"]) == ["clean", "content_gap"]
    # contradicted_stem_fact must be dropped on soft_flag
    assert "contradicted_stem_fact" not in a


def test_disagreement_existing_soft_flag_stays_soft_flag():
    """If Sonnet says soft_flag and Haiku says clean, escalate stays
    soft_flag (a non-eg disagreement). english_gap is the trigger for
    conservative override; soft_flag/clean is just borderline."""
    s = _result("Q1", [_classify("A", "soft_flag",
                                   ambiguous_between=["english_gap", "clean"])])
    h = _result("Q1", [_classify("A", "clean")], model_id="haiku")
    out = _reconcile_cross_model([s], [h])
    a = out[0]["classifications"][0]
    assert a["class"] == "soft_flag"
    assert a["cross_model_disagreement"] is True


# ── Failure-mode fallbacks ──────────────────────────────────

def test_sonnet_failed_defers_to_haiku():
    s = _result("Q1", [], error="json_parse_failed")
    h = _result("Q1", [_classify("A", "english_gap")], model_id="haiku")
    out = _reconcile_cross_model([s], [h])
    assert len(out[0]["classifications"]) == 1
    assert out[0]["classifications"][0]["class"] == "english_gap"
    assert out[0]["cross_model_disagreement_count"] == 0
    assert out[0]["cross_model_deferred_to"] == "haiku"


def test_haiku_failed_defers_to_sonnet():
    s = _result("Q1", [_classify("A", "english_gap")])
    h = _result("Q1", [], error="rate_limited", model_id="haiku")
    out = _reconcile_cross_model([s], [h])
    assert len(out[0]["classifications"]) == 1
    assert out[0]["classifications"][0]["class"] == "english_gap"
    assert out[0]["cross_model_disagreement_count"] == 0
    assert out[0]["cross_model_deferred_to"] == "sonnet_haiku_failed"


def test_both_failed_defers_to_sonnet_passthrough():
    s = _result("Q1", [], error="json_parse_failed")
    h = _result("Q1", [], error="rate_limited", model_id="haiku")
    out = _reconcile_cross_model([s], [h])
    assert out[0]["error"] == "json_parse_failed"
    assert out[0]["cross_model_disagreement_count"] == 0


# ── Letter coverage edge cases ──────────────────────────────

def test_letter_missing_in_haiku_keeps_sonnet():
    s = _result("Q1", [_classify("A", "english_gap"),
                       _classify("B", "content_gap")])
    h = _result("Q1", [_classify("A", "english_gap")], model_id="haiku")
    # B is in Sonnet only. Should pass through unchanged.
    out = _reconcile_cross_model([s], [h])
    classes = {c["letter"]: c for c in out[0]["classifications"]}
    assert classes["B"]["class"] == "content_gap"
    assert "cross_model_disagreement" not in classes["B"]


# ── Multi-question chapter ──────────────────────────────────

def test_multi_question_chapter_mixed_agreement():
    s = [
        _result("Q1", [_classify("A", "english_gap"), _classify("B", "clean")]),
        _result("Q2", [_classify("A", "content_gap")]),
    ]
    h = [
        _result("Q1", [_classify("A", "english_gap"),
                       _classify("B", "content_gap")], model_id="haiku"),
        _result("Q2", [_classify("A", "content_gap")], model_id="haiku"),
    ]
    out = _reconcile_cross_model(s, h)
    # Q1 A agreed, B disagreed (clean vs content_gap → soft_flag)
    q1_classes = {c["letter"]: c for c in out[0]["classifications"]}
    assert q1_classes["A"]["class"] == "english_gap"
    assert q1_classes["B"]["class"] == "soft_flag"
    assert out[0]["cross_model_disagreement_count"] == 1
    # Q2 A agreed
    assert out[1]["classifications"][0]["class"] == "content_gap"
    assert out[1]["cross_model_disagreement_count"] == 0


# ── flagged_distractors re-derivation ───────────────────────

def test_flagged_distractors_redrived_after_promotion():
    """When Sonnet said clean and Haiku said english_gap, the reconciled
    classification is english_gap. The flagged_distractors list (used by
    fix logic) must be re-derived from the reconciled classifications,
    not the original Sonnet output."""
    s = _result("Q1", [_classify("A", "clean")])
    h = _result("Q1", [_classify("A", "english_gap")], model_id="haiku")
    out = _reconcile_cross_model([s], [h])
    flagged = out[0]["flagged_distractors"]
    assert len(flagged) == 1
    assert flagged[0]["letter"] == "A"


def test_flagged_distractors_empty_after_demotion_to_soft_flag():
    """content_gap vs clean disagreement becomes soft_flag — soft_flag
    doesn't count as flagged. flagged_distractors should be empty."""
    s = _result("Q1", [_classify("A", "content_gap")])
    h = _result("Q1", [_classify("A", "clean")], model_id="haiku")
    out = _reconcile_cross_model([s], [h])
    assert out[0]["flagged_distractors"] == []


# ── Phase 22b: structural override early-skip ───────────────

def test_structural_override_on_both_sides_no_disagreement():
    """When the schema-labeling structural override fires on both
    Sonnet and Haiku, both classifications land on content_gap by
    construction — no disagreement counted, no soft_flag escalation.
    """
    s_entry = _classify("A", "content_gap")
    s_entry["structural_override"] = "schema_labeling"
    s_entry["original_class"] = "english_gap"
    h_entry = _classify("A", "content_gap")
    h_entry["structural_override"] = "schema_labeling"
    h_entry["original_class"] = "english_gap"
    s = _result("Q1", [s_entry])
    h = _result("Q1", [h_entry], model_id="haiku")
    out = _reconcile_cross_model([s], [h])
    assert out[0]["classifications"][0]["class"] == "content_gap"
    assert out[0]["cross_model_disagreement_count"] == 0


def test_structural_override_on_sonnet_only_wins_no_disagreement():
    """Sonnet's structural override demoted english_gap → content_gap;
    Haiku saw clean (no override fired). The deterministic verdict is
    authoritative — Sonnet's overridden entry is used and no
    cross-model disagreement is counted."""
    s_entry = _classify("A", "content_gap")
    s_entry["structural_override"] = "schema_labeling"
    s_entry["original_class"] = "english_gap"
    s = _result("Q1", [s_entry])
    h = _result("Q1", [_classify("A", "clean")], model_id="haiku")
    out = _reconcile_cross_model([s], [h])
    assert out[0]["classifications"][0]["class"] == "content_gap"
    assert out[0]["classifications"][0].get("structural_override") == "schema_labeling"
    assert out[0]["cross_model_disagreement_count"] == 0


def test_structural_override_on_haiku_only_wins_no_disagreement():
    """Symmetric case: Haiku's override fired (Haiku saw english_gap and
    structurally demoted), Sonnet didn't fire. Use Haiku's overridden
    entry; do not count as disagreement."""
    h_entry = _classify("A", "content_gap")
    h_entry["structural_override"] = "schema_labeling"
    h_entry["original_class"] = "english_gap"
    s = _result("Q1", [_classify("A", "clean")])
    h = _result("Q1", [h_entry], model_id="haiku")
    out = _reconcile_cross_model([s], [h])
    assert out[0]["classifications"][0]["class"] == "content_gap"
    assert out[0]["classifications"][0].get("structural_override") == "schema_labeling"
    assert out[0]["cross_model_disagreement_count"] == 0


def test_no_override_disagreement_still_escalates():
    """Regression check: when neither side has a structural override,
    the existing disagreement logic still fires (content_gap vs clean
    becomes soft_flag, matching prior behavior)."""
    s = _result("Q1", [_classify("A", "content_gap")])
    h = _result("Q1", [_classify("A", "clean")], model_id="haiku")
    out = _reconcile_cross_model([s], [h])
    assert out[0]["classifications"][0]["class"] == "soft_flag"
    assert out[0]["cross_model_disagreement_count"] == 1


# ── Disagreement count totals ──────────────────────────────

def test_multiple_disagreements_summed():
    s = _result("Q1", [_classify("A", "english_gap"),
                       _classify("B", "content_gap"),
                       _classify("C", "clean")])
    h = _result("Q1", [_classify("A", "clean"),
                       _classify("B", "clean"),
                       _classify("C", "clean")], model_id="haiku")
    out = _reconcile_cross_model([s], [h])
    # A disagreed (eg vs clean → eg+marker)
    # B disagreed (cg vs clean → soft_flag)
    # C agreed
    assert out[0]["cross_model_disagreement_count"] == 2


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
