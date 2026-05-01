"""Unit tests for ship_readiness brief-discriminator loading
(``_load_brief_discriminators``, ``_attach_discriminators``).

Phase 22c covers the brief-loading plumbing that threads discriminators
from on-disk anchor briefs into ``audit_question`` for Tier-A
schema-labeling detection. The classifier itself is unit-tested in
``test_schema_labeling_classifier.py``; this file tests the loader,
parser robustness, and the no-mutation contract on question records.
"""
from __future__ import annotations

import json
import pathlib
import sys
import tempfile

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import ship_readiness as sr  # noqa: E402


# ── _load_brief_discriminators ──────────────────────────────

def test_loader_returns_empty_dict_for_missing_dir():
    out = sr._load_brief_discriminators(REPO_ROOT / "_does_not_exist_")
    assert out == {}


def test_loader_skips_briefs_with_no_discriminators():
    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp)
        # Brief A: has discriminators
        (d / "a.json").write_text(json.dumps({
            "uid": "AAA", "discriminators": ["agonist_vs_antagonist"],
        }))
        # Brief B: empty discriminators list
        (d / "b.json").write_text(json.dumps({
            "uid": "BBB", "discriminators": [],
        }))
        # Brief C: missing field entirely
        (d / "c.json").write_text(json.dumps({"uid": "CCC"}))
        # Brief D: missing uid
        (d / "d.json").write_text(json.dumps({
            "discriminators": ["x_vs_y"],
        }))
        idx = sr._load_brief_discriminators(d)
        assert idx == {"AAA": ["agonist_vs_antagonist"]}


def test_loader_handles_invalid_json_gracefully():
    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp)
        (d / "good.json").write_text(json.dumps({
            "uid": "GOOD", "discriminators": ["a_vs_b"],
        }))
        (d / "bad.json").write_text("not valid json {{{")
        idx = sr._load_brief_discriminators(d)
        assert idx == {"GOOD": ["a_vs_b"]}


def test_loader_filters_non_string_discriminators():
    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp)
        (d / "x.json").write_text(json.dumps({
            "uid": "XXX",
            "discriminators": ["good_vs_bad", None, 42, "", "  ", "x_vs_y"],
        }))
        idx = sr._load_brief_discriminators(d)
        assert idx == {"XXX": ["good_vs_bad", "x_vs_y"]}


def test_loader_recurses_into_subdirectories():
    with tempfile.TemporaryDirectory() as tmp:
        d = pathlib.Path(tmp)
        (d / "BPSY").mkdir()
        (d / "SOCU").mkdir()
        (d / "BPSY" / "x.json").write_text(json.dumps({
            "uid": "B-1", "discriminators": ["agonist_vs_antagonist"],
        }))
        (d / "SOCU" / "y.json").write_text(json.dumps({
            "uid": "S-1", "discriminators": ["refutational_vs_supportive"],
        }))
        idx = sr._load_brief_discriminators(d)
        assert idx == {
            "B-1": ["agonist_vs_antagonist"],
            "S-1": ["refutational_vs_supportive"],
        }


def test_loader_works_on_real_corpus_briefs():
    """Smoke check against the live ``data/anchor_briefs/`` directory."""
    real = REPO_ROOT / "data" / "anchor_briefs"
    if not real.exists():
        return  # repo state without briefs is OK; skip
    idx = sr._load_brief_discriminators(real)
    # We expect at least the SOCU/persuasion brief
    assert "D5-SOC-002-107651e6" in idx
    discs = idx["D5-SOC-002-107651e6"]
    assert any("refutational" in d and "supportive" in d for d in discs)


# ── _attach_discriminators ──────────────────────────────────

def test_attach_returns_new_list_with_discriminators():
    qs = [
        {"question_id": "Q1", "anchor_uids": ["UID-A"]},
        {"question_id": "Q2", "anchor_uids": ["UID-B"]},
    ]
    idx = {"UID-A": ["a_vs_b"]}
    out = sr._attach_discriminators(qs, idx)
    assert len(out) == 2
    assert out[0]["_discriminators"] == ["a_vs_b"]
    assert "_discriminators" not in out[1]


def test_attach_does_not_mutate_input_questions():
    qs = [{"question_id": "Q1", "anchor_uids": ["UID-A"]}]
    snapshot = dict(qs[0])
    idx = {"UID-A": ["a_vs_b"]}
    sr._attach_discriminators(qs, idx)
    # Original input is untouched — _discriminators was added to the
    # COPY in the returned list, not the original dict.
    assert qs[0] == snapshot
    assert "_discriminators" not in qs[0]


def test_attach_handles_question_with_no_anchor_uids():
    qs = [
        {"question_id": "Q1", "anchor_uids": []},
        {"question_id": "Q2"},  # no anchor_uids field at all
    ]
    idx = {"UID-A": ["x_vs_y"]}
    out = sr._attach_discriminators(qs, idx)
    assert "_discriminators" not in out[0]
    assert "_discriminators" not in out[1]


def test_attach_uses_first_anchor_uid_only():
    qs = [{"question_id": "Q1",
           "anchor_uids": ["UID-A", "UID-B"]}]
    idx = {"UID-B": ["b_vs_x"], "UID-A": ["a_vs_x"]}
    out = sr._attach_discriminators(qs, idx)
    # Primary uid is anchor_uids[0] = UID-A
    assert out[0]["_discriminators"] == ["a_vs_x"]


def test_attach_with_empty_index_returns_copy_without_mutations():
    qs = [{"question_id": "Q1", "anchor_uids": ["UID-A"]}]
    out = sr._attach_discriminators(qs, {})
    assert "_discriminators" not in out[0]
    assert out is not qs


def test_attach_with_none_index_returns_copy():
    qs = [{"question_id": "Q1", "anchor_uids": ["UID-A"]}]
    out = sr._attach_discriminators(qs, None)
    assert out is not qs
    assert "_discriminators" not in out[0]


def test_attach_empty_questions():
    assert sr._attach_discriminators([], {"UID-A": ["a_vs_b"]}) == []
    assert sr._attach_discriminators(None, {"UID-A": ["a_vs_b"]}) == []


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
