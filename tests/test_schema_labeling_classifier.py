"""Unit tests for ``pipeline.schema_labeling_classifier``.

Phase 22a coverage:
- Tier A (brief-boosted) positive and negative cases
- Tier B (lexical) positive and negative cases
- Universal-quantifier guard — each token blocks independently
- ``labeled_pair_discriminators`` parser tolerance
- ``apply_schema_labeling_override`` walks classifications correctly,
  preserves non-english_gap entries, stamps trace metadata, removes
  ``contradicted_stem_fact`` on overridden entries.
"""
from __future__ import annotations

import pathlib
import sys

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pipeline.schema_labeling_classifier import (  # noqa: E402
    LABEL_PAIRS, UNIVERSAL_QUANTIFIERS,
    SchemaLabelingSignal,
    apply_schema_labeling_override,
    classify_distractor,
    labeled_pair_discriminators,
)


# ── Tier A (brief-boosted) ──────────────────────────────────

def test_tier_a_fires_with_brief_pair_in_stem_and_distractor():
    """SOCU/persuasion M-01 minimal repro: brief carries
    'refutational_vs_supportive_defense'; stem mentions both labels;
    distractor swaps them."""
    sig = classify_distractor(
        stem=("Inoculation theory contrasts a refutational defense "
              "with a supportive defense; only the former rehearses "
              "counterargument refutation."),
        distractor_text=("a supportive defense, in which the student "
                         "practices countering weakened opposing "
                         "arguments"),
        discriminators=["refutational_vs_supportive_defense"],
    )
    assert sig.fired is True
    assert sig.confidence == 1.0
    assert sig.brief_boosted is True
    assert sig.pair_matched == ("refutational", "supportive")
    assert sig.reason.startswith("tier_a_brief")


def test_tier_a_fires_for_agonist_antagonist_brief():
    """BPSY anchor D7-PHY-195 carries 'agonist_vs_antagonist_definition'."""
    sig = classify_distractor(
        stem=("An agonist binds a receptor to alter its state, while "
              "an antagonist produces no intrinsic effect on its own."),
        distractor_text="It is an agonist that blocks neurotransmitter binding.",
        discriminators=["agonist_vs_antagonist_definition",
                        "intrinsic_activity_yes_vs_no"],
    )
    assert sig.fired is True
    assert sig.brief_boosted is True
    assert sig.pair_matched == ("agonist", "antagonist")


def test_tier_a_does_not_fire_when_only_one_label_in_stem():
    """Brief carries a pair, but stem only mentions one of them — no
    paired-concept structure, so no schema-labeling pattern."""
    sig = classify_distractor(
        stem=("An antagonist produces no intrinsic effect on its own "
              "at the receptor."),
        distractor_text="The drug is an agonist that activates the receptor.",
        discriminators=["agonist_vs_antagonist_definition"],
    )
    assert sig.fired is False
    assert sig.brief_boosted is False  # no Tier-A match
    assert sig.confidence == 0.0


def test_tier_a_does_not_fire_when_distractor_mentions_neither():
    sig = classify_distractor(
        stem=("Refutational and supportive defenses differ in mechanism."),
        distractor_text="The technique relies on cognitive dissonance reduction.",
        discriminators=["refutational_vs_supportive_defense"],
    )
    assert sig.fired is False


def test_tier_a_with_non_pair_discriminators_falls_through():
    """Brief has only single-axis discriminators (no `_vs_`). Falls
    through to Tier B which may still fire on canonical pairs."""
    sig = classify_distractor(
        stem=("In a between-subjects design or a within-subjects design, "
              "researchers manipulate exposure differently."),
        distractor_text="A within-subjects design wastes statistical power.",
        discriminators=["intentionality_level", "substage_identification"],
    )
    # No Tier A (no _vs_); Tier B fires on between/within.
    assert sig.fired is True
    assert sig.confidence == 0.5
    assert sig.brief_boosted is False
    assert sig.pair_matched == ("between", "within")


# ── Tier B (lexical) ────────────────────────────────────────

def test_tier_b_fires_on_canonical_pair_no_brief():
    """Stem has both labels in proximity; distractor mentions one;
    no brief discriminators."""
    sig = classify_distractor(
        stem=("The encoding stage transforms input into memory traces; "
              "the retrieval stage accesses stored content."),
        distractor_text="Failure occurred in the encoding stage rather than retrieval.",
        discriminators=None,
    )
    assert sig.fired is True
    assert sig.confidence == 0.5
    assert sig.brief_boosted is False
    assert sig.pair_matched == ("encoding", "retrieval")


def test_tier_b_does_not_fire_when_pair_not_in_proximity():
    """Both labels appear but separated by far more than the proximity
    window. Two unrelated mentions don't constitute paired-concept
    structure."""
    long_filler = " filler" * 200  # ~1400 chars
    sig = classify_distractor(
        stem=f"Discussion of agonists.{long_filler} Note about antagonists.",
        distractor_text="An agonist blocks the receptor.",
        discriminators=None,
    )
    assert sig.fired is False


def test_tier_b_uppercase_iv_dv_fires():
    """IV/DV require uppercase to avoid false positives on 'give', 'live'.
    Stem with explicit IV/DV labels should fire."""
    sig = classify_distractor(
        stem=("In this design, the IV is the dosage condition and the "
              "DV is the reaction time."),
        distractor_text="The IV is the reaction time, classified into three groups.",
        discriminators=None,
    )
    assert sig.fired is True
    assert sig.pair_matched == ("IV", "DV")


def test_tier_b_does_not_fire_on_lowercase_iv_in_unrelated_words():
    """'give' and 'arrive' contain 'iv' as substring but should not
    trigger because UPPERCASE_PAIRS requires literal IV/DV casing and
    word boundaries."""
    sig = classify_distractor(
        stem="The participant gave the drink to drive arrivals.",
        distractor_text="They later returned to give it back to passengers.",
        discriminators=None,
    )
    assert sig.fired is False


# ── Universal-quantifier guard ──────────────────────────────

def test_universal_quantifier_blocks_tier_a():
    """Lester/wedding canonical: brief discriminator present, both labels
    present, BUT distractor uses 'all' which is a lexical universal."""
    sig = classify_distractor(
        stem=("After bilateral hippocampal damage, Lester recalls his "
              "wedding from a decade earlier. Encoding and retrieval "
              "both occur in this region."),
        distractor_text=("Retrograde amnesia erases all pre-injury "
                         "encoding and retrieval traces."),
        discriminators=["encoding_vs_retrieval"],
    )
    assert sig.fired is False
    assert sig.universal_quantifier_blocked is True
    # Trace records that the signal was found but blocked
    assert sig.confidence == 1.0
    assert sig.reason.startswith("tier_a_blocked_by_universal_quantifier")


def test_universal_quantifier_blocks_tier_b():
    sig = classify_distractor(
        stem=("Encoding and retrieval are sequential stages in memory."),
        distractor_text=("Encoding fails in every patient regardless of "
                         "retrieval strategy."),
        discriminators=None,
    )
    assert sig.fired is False
    assert sig.universal_quantifier_blocked is True


def test_each_universal_quantifier_blocks_independently():
    """Run a Tier-B-positive baseline against each universal token in
    UNIVERSAL_QUANTIFIERS. Each should block the override."""
    base_stem = ("Refutational and supportive defenses differ in "
                 "mechanism of resistance-building.")
    for token in UNIVERSAL_QUANTIFIERS:
        distractor = (f"A refutational defense {token} works through "
                      f"counterargument practice across cohorts.")
        sig = classify_distractor(
            stem=base_stem,
            distractor_text=distractor,
            discriminators=None,
        )
        assert sig.universal_quantifier_blocked is True, (
            f"quantifier '{token}' did not block as expected"
        )
        assert sig.fired is False


def test_no_universal_quantifier_does_not_block():
    sig = classify_distractor(
        stem="Refutational and supportive defenses differ in mechanism.",
        distractor_text="A supportive defense rehearses counterarguments.",
        discriminators=None,
    )
    # Tier B fires (refutational/supportive in stem and distractor),
    # quantifier guard does not block.
    assert sig.universal_quantifier_blocked is False
    assert sig.fired is True


# ── Real corpus regression cases ────────────────────────────

def test_bpsy_postsynaptic_specific_finding_does_not_fire():
    """Real over-specification case from corpus: stem prints 'no measurable
    postsynaptic activity'; distractor contradicts. No labeled-pair shape.
    Override must NOT fire."""
    sig = classify_distractor(
        stem=("A novel compound binds D2 receptors but produces no "
              "measurable change in postsynaptic firing on its own."),
        distractor_text=("The compound exerts its own postsynaptic "
                         "biological effect distinct from agonism."),
        discriminators=["intrinsic_activity_yes_vs_no"],
    )
    assert sig.fired is False


def test_socu_race_frame_universal_blocks():
    """Real SOCU race-frame denial: 'unrelated to racial group membership'
    is a frame denial. The distractor uses no labeled-pair structure;
    the universal-quantifier guard is a defense-in-depth safeguard."""
    sig = classify_distractor(
        stem=("Leon prefers lighter-skinned colleagues and has felt "
              "ashamed of his appearance since adolescence."),
        distractor_text=("Predict generalized self-esteem patterns "
                         "unrelated to any racial group membership."),
        discriminators=None,
    )
    assert sig.fired is False


# ── labeled_pair_discriminators parser ──────────────────────

def test_parser_handles_plain_vs_form():
    pairs = labeled_pair_discriminators([
        "agonist_vs_antagonist",
        "encoding_vs_retrieval",
    ])
    assert ("agonist", "antagonist") in pairs
    assert ("encoding", "retrieval") in pairs


def test_parser_handles_suffixed_forms():
    """`X_vs_Y_definition`, `X_vs_Y_defense` etc. should parse as `(X, Y)`
    when X and Y are single tokens (no internal underscores).

    Phase 22a scope: single-token labels only. Multi-word labels like
    'pre_exposure_vs_post_exposure_timing' are silently dropped — the
    parser cannot disambiguate where the label ends and the dimension
    suffix begins. Multi-word support is a Phase 22d concern."""
    pairs = labeled_pair_discriminators([
        "agonist_vs_antagonist_definition",
        "refutational_vs_supportive_defense",
        "pre_exposure_vs_post_exposure_timing",  # dropped
    ])
    labels = {p[0] for p in pairs} | {p[1] for p in pairs}
    assert "agonist" in labels and "antagonist" in labels
    assert "refutational" in labels and "supportive" in labels
    # Multi-word case dropped by design; should not appear.
    assert not any("exposure" in label for label in labels)


def test_parser_drops_non_pair_shapes():
    pairs = labeled_pair_discriminators([
        "intentionality_level",
        "substage_identification",
        "agonist_vs_antagonist",
    ])
    # Only the one labeled-pair survives.
    assert len(pairs) == 1
    assert pairs[0] == ("agonist", "antagonist")


def test_parser_handles_none_and_empty():
    assert labeled_pair_discriminators(None) == []
    assert labeled_pair_discriminators([]) == []
    assert labeled_pair_discriminators(["", "  "]) == []


def test_parser_skips_non_strings():
    assert labeled_pair_discriminators([None, 42, "agonist_vs_antagonist"]) == [
        ("agonist", "antagonist")
    ]


def test_parser_dedup_loose_ok():
    """Identical pairs may appear twice; parser does not deduplicate
    but the override consumer doesn't care (first match wins)."""
    pairs = labeled_pair_discriminators([
        "agonist_vs_antagonist",
        "agonist_vs_antagonist_definition",
    ])
    assert len(pairs) == 2
    assert pairs[0] == ("agonist", "antagonist")
    assert pairs[1] == ("agonist", "antagonist")


# ── apply_schema_labeling_override ──────────────────────────

def _eg_classification(letter: str, text: str,
                       fact: str = "stem fact") -> dict:
    return {
        "letter": letter,
        "class": "english_gap",
        "distractor_text": text,
        "contradicted_stem_fact": fact,
        "explanation": "explanation text",
    }


def _question(stem: str) -> dict:
    return {"question_id": "TEST-Q", "question_stem": stem}


def test_override_demotes_english_gap_when_signal_fires():
    q = _question(
        "The encoding stage and retrieval stage of memory differ in mechanism."
    )
    classifications = [
        _eg_classification("A",
                           "encoding errors block retrieval despite intact storage"),
    ]
    out, n = apply_schema_labeling_override(q, classifications)
    assert n == 1
    assert out[0]["class"] == "content_gap"
    assert out[0]["structural_override"] == "schema_labeling"
    assert out[0]["original_class"] == "english_gap"
    assert out[0]["structural_override_confidence"] == 0.5
    assert "contradicted_stem_fact" not in out[0]


def test_override_preserves_non_english_gap_entries():
    q = _question("Encoding and retrieval differ.")
    classifications = [
        _eg_classification("A", "encoding swaps retrieval"),
        {"letter": "B", "class": "content_gap", "distractor_text": "x"},
        {"letter": "C", "class": "clean", "distractor_text": "y"},
        {"letter": "D", "class": "soft_flag", "distractor_text": "z"},
    ]
    out, n = apply_schema_labeling_override(q, classifications)
    assert n == 1
    assert out[1]["class"] == "content_gap"
    assert out[2]["class"] == "clean"
    assert out[3]["class"] == "soft_flag"
    # Non-overridden entries should not gain override metadata
    for entry in out[1:]:
        assert "structural_override" not in entry


def test_override_does_not_fire_when_signal_negative():
    q = _question("The stem describes an antagonist's mechanism.")
    classifications = [
        _eg_classification("A", "an agonist blocks neurotransmitter binding"),
        # 'antagonist' is in stem but 'agonist' isn't; only one label
        # in stem → no Tier-B match.
    ]
    out, n = apply_schema_labeling_override(q, classifications)
    assert n == 0
    assert out[0]["class"] == "english_gap"
    assert "structural_override" not in out[0]


def test_override_blocks_on_universal_quantifier():
    """Lester/wedding-style: universal quantifier blocks override
    even when paired labels otherwise match."""
    q = _question(
        "Encoding and retrieval are sequential stages in memory."
    )
    classifications = [
        _eg_classification("A",
                           "all encoding traces decay throughout retrieval"),
    ]
    out, n = apply_schema_labeling_override(q, classifications)
    assert n == 0
    assert out[0]["class"] == "english_gap"


def test_override_uses_brief_discriminators_for_tier_a():
    q = _question(
        "Refutational and supportive defenses both target persuasion."
    )
    classifications = [
        _eg_classification("A", "a supportive defense rehearses refutation"),
    ]
    out, n = apply_schema_labeling_override(
        q, classifications,
        discriminators=["refutational_vs_supportive_defense"],
    )
    assert n == 1
    assert out[0]["class"] == "content_gap"
    assert out[0]["structural_override_confidence"] == 1.0
    assert "tier_a_brief" in out[0]["structural_override_reason"]


def test_override_returns_empty_input_unchanged():
    out, n = apply_schema_labeling_override(_question("anything"), [])
    assert out == []
    assert n == 0


def test_override_does_not_mutate_input():
    q = _question("Encoding and retrieval differ.")
    classifications = [
        _eg_classification("A", "encoding swaps retrieval"),
    ]
    snapshot = dict(classifications[0])
    apply_schema_labeling_override(q, classifications)
    # Original entry should still be unchanged
    assert classifications[0] == snapshot


def test_override_count_matches_overridden_entries():
    q = _question(
        "Refutational vs supportive defenses; encoding vs retrieval."
    )
    classifications = [
        _eg_classification("A", "a supportive defense rehearses refutation"),
        _eg_classification("B", "encoding errors swap retrieval order"),
        _eg_classification("C", "an unrelated tangential claim"),
    ]
    out, n = apply_schema_labeling_override(
        q, classifications,
        discriminators=["refutational_vs_supportive_defense"],
    )
    # A overridden via Tier A, B overridden via Tier B, C left english_gap
    assert n == 2
    assert out[0]["class"] == "content_gap"
    assert out[1]["class"] == "content_gap"
    assert out[2]["class"] == "english_gap"


# ── Sanity checks on constants ──────────────────────────────

def test_label_pairs_are_well_formed():
    """Every pair has two distinct lowercase strings."""
    for a, b in LABEL_PAIRS:
        assert isinstance(a, str) and isinstance(b, str)
        assert a and b
        assert a != b
        assert a == a.lower() and b == b.lower()


def test_universal_quantifiers_unique_lowercase():
    s = set(UNIVERSAL_QUANTIFIERS)
    assert len(s) == len(UNIVERSAL_QUANTIFIERS)
    assert all(t == t.lower() for t in UNIVERSAL_QUANTIFIERS)


# ── Standalone runner ───────────────────────────────────────

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
