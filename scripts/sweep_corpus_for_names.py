"""Sweep the source corpus for researcher names that will end up in questions.

Sources scanned:
  1. anchor_points.csv  (verbatim_anchor + testable_fact for all 1,566 anchors)
  2. anchor_passages_v3 (textbook passages for 1,081 textbook anchors)
  3. anchor_briefs/     (whatever briefs have been generated so far)

For each citation pattern (Name (YYYY) | Name et al. | According to Name |
Name's research/framework/etc.), tally per-name frequency and split by source.
Flag whether each name is currently in EPONYM_WHITELIST.

The output tells us which non-whitelisted names appear often enough that
production will hit gate violations or sanitizer over-strips. Use the data
to decide whether to expand the whitelist or accept stripping.

Usage:
  python sweep_corpus_for_names.py
  python sweep_corpus_for_names.py --top 50
"""
import argparse
import csv
import json
import pathlib
import re
import sys
from collections import Counter, defaultdict

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

from config import ANCHOR_POINTS_CSV, ANCHOR_PASSAGES_CSV, ANCHOR_BRIEFS_DIR
from pipeline import EPONYM_WHITELIST
from pipeline.citation_patterns import find_attributions, split_authors


def extract_names(text):
    """Return [(name_token, kind), ...] for every citation in `text`.

    Multi-author groups are expanded into individual names so each author's
    frequency is tracked independently. Uses the canonical patterns from
    pipeline.citation_patterns so the sweep stays in sync with what the
    AttributionGate actually flags.
    """
    if not text:
        return []
    found = []
    for matched, name, kind, _whitelisted in find_attributions(text):
        if kind in ("year", "according_to"):
            # Multi-author capture; split into individual names.
            for p in split_authors(name):
                found.append((p, kind))
        else:
            # et_al / possessive only ever capture a single name.
            found.append((name, kind))
    return found


def sweep_csv(csv_path, fields):
    """Yield (source_label, name, kind) for each citation in the named columns."""
    if not csv_path.exists():
        print(f"  (missing) {csv_path}")
        return
    n_rows = 0
    n_hits = 0
    with open(csv_path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            n_rows += 1
            for fld in fields:
                text = row.get(fld, "")
                for name, kind in extract_names(text):
                    n_hits += 1
                    yield (fld, name, kind)
    print(f"  scanned {n_rows} rows of {csv_path.name} ({n_hits} citation hits across {fields})")


def sweep_briefs(briefs_dir):
    """Yield (source_label, name, kind) from every brief JSON."""
    if not briefs_dir.exists():
        print(f"  (missing) {briefs_dir}")
        return
    n_briefs = 0
    n_hits = 0
    for brief_path in briefs_dir.rglob("*.json"):
        n_briefs += 1
        try:
            data = json.loads(brief_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        texts = [
            ("brief.verbatim_anchor", data.get("verbatim_anchor", "")),
            ("brief.testable_fact", data.get("testable_fact", "")),
            *[("brief.core_claims", c) for c in data.get("core_claims", []) or []],
            *[("brief.concepts.description", c.get("description", ""))
              for c in data.get("concepts", []) or []],
            *[("brief.misconceptions.label", m.get("label", ""))
              for m in data.get("misconceptions", []) or []],
            *[("brief.question_angles.description", a.get("description", ""))
              for a in data.get("question_angles", []) or []],
        ]
        for src, t in texts:
            for name, kind in extract_names(t):
                n_hits += 1
                yield (src, name, kind)
    print(f"  scanned {n_briefs} briefs ({n_hits} citation hits)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--top", type=int, default=40,
                        help="Show top N names (default 40)")
    args = parser.parse_args()

    print("=== Source sweep ===")
    print()
    print("anchor_points.csv:")
    anchor_hits = list(sweep_csv(ANCHOR_POINTS_CSV, ["verbatim_anchor", "testable_fact"]))
    print()
    print("anchor_passages_v3:")
    passage_hits = list(sweep_csv(ANCHOR_PASSAGES_CSV, ["passage"]))
    print()
    print("anchor_briefs/:")
    brief_hits = list(sweep_briefs(ANCHOR_BRIEFS_DIR))
    print()

    # Aggregate
    by_name_source = defaultdict(lambda: defaultdict(int))
    by_name_kind = defaultdict(lambda: defaultdict(int))
    total_by_name = Counter()

    for src, name, kind in anchor_hits + passage_hits + brief_hits:
        # Categorize source bucket
        if src in ("verbatim_anchor", "testable_fact"):
            bucket = "anchor_csv"
        elif src == "passage":
            bucket = "passage"
        else:
            bucket = "brief"
        by_name_source[name][bucket] += 1
        by_name_kind[name][kind] += 1
        total_by_name[name] += 1

    n_total_hits = sum(total_by_name.values())
    n_distinct = len(total_by_name)
    n_whitelisted_names = sum(1 for n in total_by_name if n in EPONYM_WHITELIST)
    n_whitelisted_hits = sum(c for n, c in total_by_name.items() if n in EPONYM_WHITELIST)
    n_unlisted_hits = n_total_hits - n_whitelisted_hits

    print(f"=== Summary ===")
    print(f"Total citation instances: {n_total_hits}")
    print(f"Distinct names found: {n_distinct}")
    print(f"  Already whitelisted: {n_whitelisted_names} names "
          f"({n_whitelisted_hits} instances, {100*n_whitelisted_hits/max(n_total_hits,1):.1f}%)")
    print(f"  NOT whitelisted: {n_distinct - n_whitelisted_names} names "
          f"({n_unlisted_hits} instances, {100*n_unlisted_hits/max(n_total_hits,1):.1f}%)")
    print()

    print(f"=== Top {args.top} names by total citation frequency ===")
    print(f"{'name':22s} {'total':>5} {'anchor':>7} {'passage':>8} {'brief':>6}  "
          f"{'year':>5} {'etal':>5} {'accT':>5} {'poss':>5}  WL")
    print("-" * 95)
    for name, total in total_by_name.most_common(args.top):
        srcs = by_name_source[name]
        kinds = by_name_kind[name]
        wl = "Y" if name in EPONYM_WHITELIST else "-"
        print(f"{name:22s} {total:>5} "
              f"{srcs.get('anchor_csv', 0):>7} {srcs.get('passage', 0):>8} {srcs.get('brief', 0):>6}  "
              f"{kinds.get('year', 0):>5} {kinds.get('et_al', 0):>5} "
              f"{kinds.get('according_to', 0):>5} {kinds.get('possessive', 0):>5}   {wl}")

    print()
    print(f"=== Top non-whitelisted names (candidates for whitelist expansion) ===")
    unlisted = [(n, c) for n, c in total_by_name.most_common() if n not in EPONYM_WHITELIST]
    print(f"(showing top {min(args.top, len(unlisted))} of {len(unlisted)})")
    print(f"{'name':22s} {'total':>5} {'anchor':>7} {'passage':>8} {'brief':>6}")
    print("-" * 60)
    for name, total in unlisted[:args.top]:
        srcs = by_name_source[name]
        print(f"{name:22s} {total:>5} "
              f"{srcs.get('anchor_csv', 0):>7} {srcs.get('passage', 0):>8} {srcs.get('brief', 0):>6}")


if __name__ == "__main__":
    main()
