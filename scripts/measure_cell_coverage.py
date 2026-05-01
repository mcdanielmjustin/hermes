"""measure_cell_coverage.py — P-1 cell coverage analysis.

Walks data/quiz/{DOMAIN}/*.json and emits two CSVs that describe the
distribution of saved questions across the dimensions of the
distractor-policy matrix:

  1. cell_distribution.csv — per (tier, domain_code, source_type,
     stem_pattern) tuple: question count.
     This is the proxy for `pedagogical_content_type` until P2 adds
     classification — `(source_type, stem_pattern)` reasonably
     approximates content_type for the existing corpus.

  2. misconception_distribution.csv — per (tier, domain_code,
     misconception_type) tuple: distractor count.
     Surfaces the misconception_type signal that gates currently
     ignore.

Together they let us focus P1 matrix authoring on the live cells
(empirically populated) rather than the full theoretical 864-cell
space.

Usage:
  python scripts/measure_cell_coverage.py
  python scripts/measure_cell_coverage.py --out reports/

Outputs (default to logs/cell_coverage_<utc-date>/):
  cell_distribution.csv
  misconception_distribution.csv
  summary.txt — top-line counts and the live-cell list
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
from collections import Counter
from datetime import datetime, timezone


# Files to ignore — sibling artifacts from earlier iterations.
SKIP_NAME_TOKENS = (
    "audit", "fixed", "bak", "backup",
    "pre_layer", "layer_a", "manifest", "stats",
)


def is_content_file(p: pathlib.Path) -> bool:
    """Return True for files that hold real saved questions."""
    if p.suffix != ".json":
        return False
    name = p.name.lower()
    return not any(tok in name for tok in SKIP_NAME_TOKENS)


def load_questions(p: pathlib.Path) -> list[dict]:
    """Load a chapter or batch JSON; tolerate list-shape and
    {questions: [...]} wrappers."""
    try:
        d = json.load(p.open(encoding="utf-8"))
    except Exception:
        return []
    if isinstance(d, list):
        return d
    if isinstance(d, dict) and "questions" in d:
        return d["questions"]
    return []


def collect(quiz_dir: pathlib.Path):
    """Walk quiz_dir, accumulate cell + misconception distributions."""
    cell_counter: Counter = Counter()
    misc_counter: Counter = Counter()
    total_questions = 0
    files_seen = 0

    for p in sorted(quiz_dir.rglob("*.json")):
        if not is_content_file(p):
            continue
        questions = load_questions(p)
        if not questions:
            continue
        files_seen += 1
        for q in questions:
            # Skip quarantined / failed-validation records
            if q.get("_failed_validation") or "_error" in q:
                continue
            tier = q.get("difficulty_tier")
            domain = q.get("domain_code") or "unknown"
            source = q.get("source_type") or "unknown"
            pattern = q.get("stem_pattern") or "unknown"
            if tier is None:
                continue

            cell_counter[(tier, domain, source, pattern)] += 1
            total_questions += 1

            for o in q.get("options", []) or []:
                if o.get("is_correct"):
                    continue
                misc = o.get("misconception_type") or "unset"
                misc_counter[(tier, domain, misc)] += 1

    return cell_counter, misc_counter, total_questions, files_seen


def write_csvs(out_dir: pathlib.Path,
               cell_counter: Counter, misc_counter: Counter):
    out_dir.mkdir(parents=True, exist_ok=True)

    cell_path = out_dir / "cell_distribution.csv"
    with cell_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tier", "domain_code", "source_type",
                    "stem_pattern", "question_count"])
        for (tier, dom, src, pat), n in sorted(
            cell_counter.items(), key=lambda x: (-x[1], x[0])
        ):
            w.writerow([tier, dom, src, pat, n])

    misc_path = out_dir / "misconception_distribution.csv"
    with misc_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["tier", "domain_code", "misconception_type",
                    "distractor_count"])
        for (tier, dom, mt), n in sorted(
            misc_counter.items(), key=lambda x: (-x[1], x[0])
        ):
            w.writerow([tier, dom, mt, n])

    return cell_path, misc_path


def write_summary(out_dir: pathlib.Path,
                  cell_counter: Counter, misc_counter: Counter,
                  total_q: int, files_seen: int):
    summary = []
    summary.append("=" * 72)
    summary.append(
        f"Cell coverage analysis — {datetime.now(timezone.utc).isoformat(timespec='seconds')}"
    )
    summary.append("=" * 72)
    summary.append("")
    summary.append(f"Files scanned: {files_seen}")
    summary.append(f"Questions counted: {total_q}")
    summary.append("")

    # Live cell summary
    summary.append("Active cells (tier, domain, source_type, stem_pattern):")
    summary.append(f"  populated: {len(cell_counter)}")
    summary.append(f"  theoretical max: 4 × 9 × 2 × ~20 = ~1440")
    summary.append("")

    # By tier
    by_tier: Counter = Counter()
    for (tier, *_), n in cell_counter.items():
        by_tier[tier] += n
    summary.append("By tier:")
    for tier in (1, 2, 3, 4):
        summary.append(f"  T{tier}: {by_tier[tier]}")
    summary.append("")

    # By domain
    by_domain: Counter = Counter()
    for (_, dom, *_), n in cell_counter.items():
        by_domain[dom] += n
    summary.append("By domain:")
    for dom, n in sorted(by_domain.items(), key=lambda x: -x[1]):
        summary.append(f"  {dom}: {n}")
    summary.append("")

    # Top 20 cells
    summary.append("Top 20 cells by question count:")
    summary.append(
        f"  {'tier':<5}{'domain':<8}{'source_type':<20}"
        f"{'stem_pattern':<28}count"
    )
    for (tier, dom, src, pat), n in sorted(
        cell_counter.items(), key=lambda x: -x[1]
    )[:20]:
        summary.append(
            f"  {tier:<5}{dom:<8}{src:<20}{pat:<28}{n}"
        )
    summary.append("")

    # Misconception distribution top 15
    summary.append("Top 15 misconception cells (tier, domain, type):")
    summary.append(
        f"  {'tier':<5}{'domain':<8}{'misconception_type':<28}count"
    )
    for (tier, dom, mt), n in sorted(
        misc_counter.items(), key=lambda x: -x[1]
    )[:15]:
        summary.append(
            f"  {tier:<5}{dom:<8}{mt:<28}{n}"
        )
    summary.append("")

    # Cells observed per (tier, domain) — gives a sense of where
    # P1 matrix authoring needs to focus
    by_tier_domain_cells: dict = {}
    for (tier, dom, src, pat), n in cell_counter.items():
        by_tier_domain_cells.setdefault((tier, dom), set()).add(
            (src, pat)
        )
    summary.append("Distinct (source, pattern) cells per (tier, domain):")
    for (tier, dom), cells in sorted(
        by_tier_domain_cells.items(), key=lambda x: (x[0][1], x[0][0])
    ):
        summary.append(f"  T{tier} {dom}: {len(cells)} cells")
    summary.append("")

    summary_text = "\n".join(summary)
    out_path = out_dir / "summary.txt"
    out_path.write_text(summary_text, encoding="utf-8")
    return out_path, summary_text


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument(
        "--quiz-dir", default="data/quiz",
        help="Root of saved quiz JSONs (default: data/quiz)",
    )
    parser.add_argument(
        "--out", default=None,
        help="Output directory (default: logs/cell_coverage_<UTC-date>)",
    )
    args = parser.parse_args()

    quiz_dir = pathlib.Path(args.quiz_dir)
    if not quiz_dir.exists():
        print(f"ERROR: quiz dir not found: {quiz_dir}", file=sys.stderr)
        sys.exit(1)

    out_dir = pathlib.Path(args.out) if args.out else pathlib.Path(
        "logs"
    ) / f"cell_coverage_{datetime.now(timezone.utc).strftime('%Y%m%d')}"

    cell_counter, misc_counter, total_q, files_seen = collect(quiz_dir)
    cell_path, misc_path = write_csvs(out_dir, cell_counter, misc_counter)
    summary_path, summary_text = write_summary(
        out_dir, cell_counter, misc_counter, total_q, files_seen
    )

    print(summary_text)
    print()
    print(f"Wrote: {cell_path}")
    print(f"Wrote: {misc_path}")
    print(f"Wrote: {summary_path}")


if __name__ == "__main__":
    main()
