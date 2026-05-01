"""Find concept_id clusters that likely refer to the same concept.

Reads the canonical concept registry (data/concept_registry.json) and
emits merge candidates to data/concept_merge_proposals.json for human
review. The accept/reject decisions are made offline; an --apply step
(future work) will rewrite the registry + affected briefs after review.

Why this is post-hoc instead of in-line in the brief generator: fuzzy
matching is too risky for false merges of similar-but-distinct concepts
("agonist" vs "antagonist", "anterograde" vs "retrograde"). Surfacing
proposals for review keeps the human in the loop on real merges while
catching genuine fragmentation ("agonist" vs "receptor-agonist").

Usage:
  python cross_brief_consistency.py
  python cross_brief_consistency.py --threshold 0.6   # more permissive
  python cross_brief_consistency.py --threshold 0.8   # stricter
"""
import argparse
import json
import pathlib
import sys
from datetime import datetime, timezone

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

from config import DATA_DIR
from pipeline.concept_clustering import find_merge_candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--threshold", type=float, default=0.5,
        help="Minimum similarity score to surface as a candidate (default 0.5; "
             "word-Jaccard scoring puts genuine fragmentation around 0.5–0.7 "
             "and near-twin opposites near 0.0–0.2)",
    )
    parser.add_argument(
        "--registry", type=pathlib.Path,
        default=DATA_DIR / "concept_registry.json",
        help="Path to the canonical concept registry",
    )
    parser.add_argument(
        "--output", type=pathlib.Path,
        default=DATA_DIR / "concept_merge_proposals.json",
        help="Where to write the proposals JSON",
    )
    args = parser.parse_args()

    if not args.registry.exists():
        sys.exit(f"Registry not found: {args.registry}")

    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    concepts = registry.get("concepts", {})

    print(f"Registry: {args.registry}")
    print(f"  Canonical concepts: {len(concepts)}")
    print(f"  Threshold:          {args.threshold}")
    print()

    candidates = find_merge_candidates(concepts, threshold=args.threshold)

    print(f"Merge candidates: {len(candidates)}")
    if candidates:
        print()
        print("Top candidates:")
        for c in candidates[:25]:
            print(f"  score={c['score']:.3f}  "
                  f"{c['id_a']!r:40s} ↔ {c['id_b']!r}")
            print(f"           label_a: {c['label_a']!r}")
            print(f"           label_b: {c['label_b']!r}")

    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "registry_path": str(args.registry),
        "threshold": args.threshold,
        "n_concepts_scanned": len(concepts),
        "candidates": candidates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print()
    print(f"Proposals written to {args.output}")
    if candidates:
        print()
        print("Next step: review the proposals JSON. Each entry is a "
              "candidate merge between two concept_ids that likely refer to "
              "the same concept. Accept the real fragmentation cases; "
              "reject the false-positive twins (e.g., agonist vs antagonist).")


if __name__ == "__main__":
    main()
