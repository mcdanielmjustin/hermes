"""Phase A4 — Mine the existing corpus for paired-name concept patterns.

Scans `data/quiz/*/*.json` for explicit paired-name forms in testable_facts
and stems, surfacing candidates that the schema_labeling_classifier's
`LABEL_PAIRS` list (Tier B, conf=0.5) currently misses.

Mining patterns (kept conservative — these are the canonical forms a
two-named-concept stem uses):

  - "X vs Y" / "X vs. Y"
  - "X versus Y"
  - "X (rather than Y)" / "X but not Y"
  - "differentiate X from Y" / "distinguish X from Y"
  - "between X and Y"

Filters applied at the mining layer (rejected automatically):

  - Same-pair (X == Y after lowercase)
  - Either side ≤ 2 chars (noise)
  - Either side already in `UNIVERSAL_QUANTIFIERS` from the english_gap
    scanner (prevents overlap with the universal-quantifier signature)
  - Either side a directional descriptor (more, less, higher, lower,
    increase, decrease, etc.) — these are caught by other signatures
    and are NOT schema_labeling

The OUTPUT is a curation file (`data/.diagnosis/label_pairs_candidates.json`).
A human reviews this list before any pair is added to LABEL_PAIRS — pairs
that are domain-canonical (e.g., agonist/antagonist) get accepted; pairs
that are descriptive variation get rejected.

Cost: $0 (deterministic regex; no LLM).
"""
from __future__ import annotations

import glob
import json
import pathlib
import re
import sys
from collections import Counter, defaultdict

try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except (AttributeError, OSError):
    pass

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from pipeline.schema_labeling_classifier import (
    LABEL_PAIRS,
    UPPERCASE_PAIRS,
    UNIVERSAL_QUANTIFIERS,
)

QUIZ_DIR = REPO_ROOT / "data" / "quiz"
OUT_DIR = REPO_ROOT / "data" / ".diagnosis"
OUT_PATH = OUT_DIR / "label_pairs_candidates.json"

# Mining patterns. Each captures group(1)=X, group(2)=Y. Word boundaries
# on each side avoid partial-token matches.
PATTERNS: tuple[tuple[str, re.Pattern], ...] = (
    ("x_vs_y", re.compile(r"\b([a-z][a-z0-9_-]+)\s+vs\.?\s+([a-z][a-z0-9_-]+)\b", re.IGNORECASE)),
    ("x_versus_y", re.compile(r"\b([a-z][a-z0-9_-]+)\s+versus\s+([a-z][a-z0-9_-]+)\b", re.IGNORECASE)),
    ("x_rather_than_y", re.compile(r"\b([a-z][a-z0-9_-]+)\s+\(rather\s+than\s+([a-z][a-z0-9_-]+)\)", re.IGNORECASE)),
    ("differentiate_x_from_y", re.compile(r"\bdifferentiate\s+([a-z][a-z0-9_-]+)\s+from\s+([a-z][a-z0-9_-]+)\b", re.IGNORECASE)),
    ("distinguish_x_from_y", re.compile(r"\bdistinguish\s+([a-z][a-z0-9_-]+)\s+from\s+([a-z][a-z0-9_-]+)\b", re.IGNORECASE)),
    ("between_x_and_y", re.compile(r"\bbetween\s+([a-z][a-z0-9_-]+)\s+and\s+([a-z][a-z0-9_-]+)\b", re.IGNORECASE)),
)

# Directional / quantitative tokens that look like paired names but are
# actually directional contradictions caught by other signatures (the
# english_gap scanner's _DIRECTION_PAIRS, etc.). Reject pre-curation.
DIRECTIONAL_NOISE: frozenset[str] = frozenset({
    "more", "less", "higher", "lower", "greater", "smaller",
    "increase", "decrease", "increased", "decreased",
    "rise", "fall", "rises", "falls", "rising", "falling",
    "elevated", "reduced", "above", "below",
    "before", "after", "early", "late", "first", "second",
    "best", "worst", "good", "bad", "better", "worse",
})

# Tokens that are too generic to form a paired-name concept on their own.
GENERIC_NOISE: frozenset[str] = frozenset({
    "this", "that", "these", "those", "the", "a", "an",
    "one", "two", "many", "few", "all", "some", "none",
    "any", "every", "such", "other", "another",
    "not", "no", "yes", "if", "when", "where", "why", "how",
    # Common English connectors that aren't paired-name candidates:
    "type", "kind", "form", "way", "method", "approach",
    "person", "people", "patient", "client", "subject",
    "is", "be", "have", "do", "can", "will", "may", "should",
    "are", "was", "were", "been", "being",
})

EXISTING_PAIRS_SET: frozenset[tuple[str, str]] = frozenset(
    tuple(sorted(p)) for p in LABEL_PAIRS
) | frozenset(
    tuple(sorted([a.lower(), b.lower()])) for a, b in UPPERCASE_PAIRS
)

UNIVERSAL_SET: frozenset[str] = frozenset(t.lower() for t in UNIVERSAL_QUANTIFIERS)

MIN_FREQ = 3


def _normalize_token(t: str) -> str:
    return (t or "").strip().lower()


def _reject_reason(a: str, b: str) -> str | None:
    """Return None to accept the candidate, or a string reason to reject."""
    if a == b:
        return "same_token"
    if len(a) < 3 or len(b) < 3:
        return "too_short"
    if a in DIRECTIONAL_NOISE or b in DIRECTIONAL_NOISE:
        return "directional_noise"
    if a in GENERIC_NOISE or b in GENERIC_NOISE:
        return "generic_noise"
    if a in UNIVERSAL_SET or b in UNIVERSAL_SET:
        return "universal_quantifier"
    if tuple(sorted([a, b])) in EXISTING_PAIRS_SET:
        return "already_in_LABEL_PAIRS"
    return None


def _scan_text(text: str) -> list[tuple[tuple[str, str], str]]:
    """Returns list of ((a, b), pattern_name) tuples for pairs matched
    in the text. Order-preserving on first occurrence per pattern."""
    out: list[tuple[tuple[str, str], str]] = []
    for pattern_name, pat in PATTERNS:
        for match in pat.finditer(text or ""):
            a = _normalize_token(match.group(1))
            b = _normalize_token(match.group(2))
            if not a or not b:
                continue
            out.append(((a, b), pattern_name))
    return out


def main():
    files = sorted(glob.glob(str(QUIZ_DIR / "*" / "*.json")))
    files = [f for f in files
             if not f.endswith(".audit.json")
             and ".diagnostic_quality" not in f]
    print(f"Scanning {len(files)} chapter files in data/quiz/...")

    candidates: Counter = Counter()
    pattern_counts: dict[tuple[str, str], Counter] = defaultdict(Counter)
    examples: dict[tuple[str, str], list[str]] = defaultdict(list)
    rejections: Counter = Counter()

    n_questions = 0
    for path in files:
        try:
            with open(path, encoding="utf-8") as fh:
                data = json.load(fh)
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(data, list):
            continue

        for q in data:
            n_questions += 1
            text_blob = " ".join([
                str(q.get("testable_fact") or ""),
                str(q.get("question_stem") or ""),
                # also scan the anchor's testable fact summary if present
                " ".join(str(s) for s in (q.get("anchor_content_summaries") or [])),
            ])
            for (a, b), pat_name in _scan_text(text_blob):
                pair = tuple(sorted([a, b]))
                reason = _reject_reason(a, b)
                if reason:
                    rejections[reason] += 1
                    continue
                candidates[pair] += 1
                pattern_counts[pair][pat_name] += 1
                if len(examples[pair]) < 3:
                    snippet = text_blob[
                        max(0, text_blob.lower().find(a) - 50):
                        text_blob.lower().find(a) + len(a) + 100
                    ]
                    examples[pair].append(snippet.strip())

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Build the curation list — pairs with freq >= MIN_FREQ, sorted by frequency desc.
    surfaced: list[dict] = []
    for pair, freq in candidates.most_common():
        if freq < MIN_FREQ:
            break
        surfaced.append({
            "pair": list(pair),
            "frequency": int(freq),
            "patterns": dict(pattern_counts[pair]),
            "examples": examples[pair][:3],
        })

    out_payload = {
        "scanned_questions": n_questions,
        "scanned_chapter_files": len(files),
        "min_frequency": MIN_FREQ,
        "existing_pair_count": len(EXISTING_PAIRS_SET),
        "rejections_summary": dict(rejections.most_common()),
        "surfaced_candidates": surfaced,
        "all_candidates_below_threshold": [
            {"pair": list(p), "frequency": int(f)}
            for p, f in candidates.most_common()
            if f < MIN_FREQ and f >= 2  # show 2-frequency too for inspection
        ],
    }
    with open(OUT_PATH, "w", encoding="utf-8") as fh:
        json.dump(out_payload, fh, indent=2, ensure_ascii=False)

    print()
    print("=" * 70)
    print(f"Mining results — {n_questions} questions across {len(files)} files")
    print("=" * 70)
    print(f"Existing LABEL_PAIRS + UPPERCASE_PAIRS: {len(EXISTING_PAIRS_SET)}")
    print(f"Rejections (auto-filtered):")
    for reason, count in rejections.most_common():
        print(f"  {reason:25s} {count:4d}")
    print()
    print(f"Surfaced candidates (freq >= {MIN_FREQ}): {len(surfaced)}")
    for c in surfaced[:30]:
        pat_summary = "/".join(f"{p}={n}" for p, n in c["patterns"].items())
        print(f"  freq={c['frequency']:3d}  {c['pair'][0]:20s}  /  "
              f"{c['pair'][1]:20s}  ({pat_summary})")
    if len(surfaced) > 30:
        print(f"  ... and {len(surfaced) - 30} more")
    print()
    print(f"Output: {OUT_PATH}")
    print()
    print("Next: review candidates manually. Accept canonical paired-name")
    print("concepts (e.g., agonist/antagonist). Reject pairs that are domain")
    print("descriptors, generic word juxtapositions, or already covered by")
    print("other signatures.")


if __name__ == "__main__":
    main()
