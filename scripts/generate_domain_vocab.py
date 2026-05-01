"""Bootstrap script: extract candidate domain vocabulary pools.

Walks data/anchor_briefs/{DOMAIN}/*.json (and data/concept_vocab/{DOMAIN}/
*.json when available) for every domain, extracts ≥6-char content
vocabulary terms via the same helper the L3 scaffold uses
(extract_brief_vocabulary_terms), frequency-counts, sorts, and writes
data/domain_vocab/{DOMAIN_CODE}.json.

The output is a CANDIDATE pool. The `curated: false` flag marks each file
as "auto-bootstrapped, not human-reviewed yet." Hand-curate by editing
the file and flipping `curated: true`. The pipeline reads the file
regardless of the flag — it just tracks review status.

Usage:
    python scripts/generate_domain_vocab.py            # all 9 domains
    python scripts/generate_domain_vocab.py --domain BPSY    # one domain
    python scripts/generate_domain_vocab.py --dry-run        # print, don't write

Phase 7 of the multi-layer pedagogy enforcement architecture.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# sys.path bootstrap so this script runs from anywhere.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.agents import extract_brief_vocabulary_terms  # noqa: E402

# Single source of truth — DOMAIN_CODES/NAMES live in shared_constants.
# (Drift between this script and shared_constants previously wrote non-
# canonical names into the JSONs — fixed by importing.)
sys.path.insert(0, str(ROOT / "scripts"))
from shared_constants import ALL_CODES as DOMAIN_CODES  # noqa: E402
from shared_constants import DOMAIN_NAMES  # noqa: E402


def _load_briefs(domain_dir: Path) -> list[dict]:
    """Load all *.json files from a domain's anchor_briefs directory."""
    if not domain_dir.exists():
        return []
    briefs = []
    for path in sorted(domain_dir.glob("*.json")):
        try:
            with open(path, encoding="utf-8") as f:
                briefs.append(json.load(f))
        except (OSError, json.JSONDecodeError) as e:
            print(f"  ! skip {path.name}: {e}", file=sys.stderr)
    return briefs


def _load_concept_vocabs(domain_dir: Path) -> list[dict]:
    """Load concept_vocab chapter files. Each looks like a brief
    (concepts list with descriptions) so extract_brief_vocabulary_terms
    works on them too."""
    return _load_briefs(domain_dir)


def _term_frequencies(briefs: list[dict]) -> Counter:
    """Count term occurrences across briefs. A term appearing in N briefs
    is more domain-canonical than one in 1 brief."""
    counts: Counter = Counter()
    for b in briefs:
        terms = extract_brief_vocabulary_terms(b)
        for t in terms:
            counts[t] += 1
    return counts


def build_domain_vocab(domain_code: str, repo_root: Path) -> dict:
    """Build the domain vocab JSON dict for a single domain."""
    briefs_dir = repo_root / "data" / "anchor_briefs" / domain_code
    concept_vocab_dir = repo_root / "data" / "concept_vocab" / domain_code

    briefs = _load_briefs(briefs_dir)
    concept_vocabs = _load_concept_vocabs(concept_vocab_dir)
    sources = briefs + concept_vocabs

    counts = _term_frequencies(sources)
    # Sort: frequency desc, then alphabetical for tie-break determinism.
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    vocabulary = [term for term, _ in ordered]

    return {
        "domain_code": domain_code,
        "domain_name": DOMAIN_NAMES.get(domain_code, domain_code),
        "curated": False,
        "version": 1,
        "source_summary": {
            "anchor_briefs": len(briefs),
            "concept_vocab_files": len(concept_vocabs),
            "unique_terms": len(vocabulary),
        },
        "vocabulary": vocabulary,
    }


def write_domain_vocab(domain_code: str, repo_root: Path,
                       dry_run: bool = False) -> Path:
    out_dir = repo_root / "data" / "domain_vocab"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{domain_code}.json"

    payload = build_domain_vocab(domain_code, repo_root)

    if dry_run:
        print(f"--- {domain_code} ---")
        print(f"  briefs: {payload['source_summary']['anchor_briefs']}")
        print(f"  concept_vocab files: {payload['source_summary']['concept_vocab_files']}")
        print(f"  unique terms: {payload['source_summary']['unique_terms']}")
        if payload["vocabulary"]:
            sample = payload["vocabulary"][:15]
            print(f"  top 15: {sample}")
        return out_path

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"  wrote {out_path.name}: "
          f"{payload['source_summary']['unique_terms']} terms "
          f"from {payload['source_summary']['anchor_briefs']} briefs "
          f"+ {payload['source_summary']['concept_vocab_files']} concept_vocab files")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--domain", choices=DOMAIN_CODES,
                        help="Generate for one domain only (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print summary, do not write files")
    args = parser.parse_args()

    targets = [args.domain] if args.domain else DOMAIN_CODES
    print(f"Generating domain vocab for: {', '.join(targets)}")
    for code in targets:
        write_domain_vocab(code, ROOT, dry_run=args.dry_run)
    print("Done.")


if __name__ == "__main__":
    main()
