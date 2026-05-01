"""Seed concept_registry.json from existing anchor briefs.

Run once to populate the registry from whatever briefs exist on disk. Going
forward, generate_anchor_briefs.py will keep the registry in sync as new
briefs are produced.

Idempotent — running again with no new briefs is a no-op (already-registered
concepts increment their appears_in list but no new entries are created).

Usage:
  python seed_concept_registry.py
"""
import json
import pathlib
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

from config import ANCHOR_BRIEFS_DIR, DATA_DIR
from pipeline.concept_registry import ConceptRegistry, canonicalize_brief


def main():
    registry_path = DATA_DIR / "concept_registry.json"
    print(f"Registry path: {registry_path}")
    print(f"Briefs dir:    {ANCHOR_BRIEFS_DIR}")
    print()

    registry = ConceptRegistry(registry_path)
    starting_count = registry.stats()["total_concepts"]
    print(f"Existing canonical concepts: {starting_count}")
    print()

    if not ANCHOR_BRIEFS_DIR.exists():
        print(f"No briefs dir at {ANCHOR_BRIEFS_DIR} — nothing to seed.")
        registry.save()
        return

    n_briefs = 0
    n_concepts_added_total = 0
    n_aliased_total = 0

    for brief_path in sorted(ANCHOR_BRIEFS_DIR.rglob("*.json")):
        try:
            brief = json.loads(brief_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  SKIP {brief_path.name}: {e}")
            continue

        n_briefs += 1
        n_new, n_aliased = canonicalize_brief(brief, registry)
        n_concepts_added_total += n_new
        n_aliased_total += n_aliased
        rel = brief_path.relative_to(ANCHOR_BRIEFS_DIR)
        print(f"  {str(rel):50s}  +{n_new} new, {n_aliased} aliased")

    registry.save()

    stats = registry.stats()
    print()
    print(f"Scanned {n_briefs} brief(s).")
    print(f"  Total canonical concepts: {stats['total_concepts']} "
          f"(was {starting_count}, +{stats['total_concepts'] - starting_count})")
    print(f"  Total aliases:            {stats['total_aliases']}")
    print(f"  Concepts added this run:  {n_concepts_added_total}")
    print(f"  Aliases registered:       {n_aliased_total}")
    print()
    print(f"Registry saved to {registry_path}")


if __name__ == "__main__":
    main()
