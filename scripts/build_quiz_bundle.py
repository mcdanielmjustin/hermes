"""
build_quiz_bundle.py

Reads quiz JSON files from data/quiz_shippable/{DOMAIN}/*.json — the
ship-ready output of scripts/ship_readiness.py — merges them per domain,
and writes to PassEPPP-website/content/enrichment/:
  1. {DOMAIN}_quiz.json  (per-domain question arrays)
  2. quiz_stats.json     (manifest with counts per domain/tier)
  3. quiz_data.js        (JS bundle for file:// protocol)

The bundle script REFUSES to run if data/quiz_shippable/manifest.json
is missing, because that means ship_readiness.py has never been run
(or its output was deleted) — bundling raw generation would skip the
english_gap quality gate.

Pipeline:
  generate_quiz_questions.py      → data/quiz/
  ship_readiness.py               → data/quiz_shippable/  (with manifest)
  build_quiz_bundle.py (this)     → PassEPPP-website/content/enrichment/

Run after ship_readiness completes:
  python scripts/ship_readiness.py
  python scripts/build_quiz_bundle.py

Override the source directory (e.g., to bundle from a different audit
output) with --source-dir.
"""

import argparse
import json
import sys
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

from config import QUIZ_SHIPPABLE_DIR, ENRICHMENT_BUNDLE_DIR as OUTPUT_DIR
from shared_constants import DOMAIN_NAMES, ALL_CODES
DOMAINS = sorted(ALL_CODES)

MANIFEST_FILENAME = "manifest.json"


def _load_manifest(source_dir: pathlib.Path) -> dict:
    """Load the ship_readiness manifest. Returns the parsed dict on
    success; exits with a helpful error if missing or malformed."""
    manifest_path = source_dir / MANIFEST_FILENAME
    if not manifest_path.exists():
        print(
            f"ERROR: ship-readiness manifest not found at {manifest_path}\n"
            f"\n"
            f"The bundle script requires that scripts/ship_readiness.py be\n"
            f"run first. That script audits chapters for english_gap\n"
            f"distractors, optionally auto-fixes them, and emits ready\n"
            f"chapters into {source_dir}.\n"
            f"\n"
            f"To proceed:\n"
            f"  python scripts/ship_readiness.py\n"
            f"  python scripts/build_quiz_bundle.py\n"
            f"\n"
            f"To bypass the gate (NOT recommended for production):\n"
            f"  python scripts/build_quiz_bundle.py --source-dir data/quiz "
            f"--no-require-manifest\n",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        with open(manifest_path, encoding="utf-8") as fh:
            return json.load(fh)
    except (json.JSONDecodeError, OSError) as e:
        print(
            f"ERROR: could not read ship-readiness manifest at {manifest_path}: {e}",
            file=sys.stderr,
        )
        sys.exit(1)


def build(source_dir: pathlib.Path, output_dir: pathlib.Path,
          require_manifest: bool = True, allow_partial: bool = False) -> int:
    """Build the bundle. Returns exit code (0 = success, 2 = partial
    bundle requires --allow-partial).

    The bundle script writes to OUTPUT_DIR (default: PassEPPP-website
    enrichment dir) but accepts an override for safe smoke-testing.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if require_manifest:
        manifest = _load_manifest(source_dir)
        summary = manifest.get("summary") or {}
        review_count = summary.get("chapters_review", 0)
        ready_count = summary.get("chapters_ready", 0)
        print(
            f"Ship-readiness manifest: {ready_count} ready, "
            f"{review_count} review, "
            f"{summary.get('english_gap_remaining', '?')} english_gap "
            f"remaining in shipped corpus"
        )
        if review_count > 0 and not allow_partial:
            print(
                f"\nERROR: {review_count} chapter(s) flagged for human review.\n"
                f"Bundling now would ship a partial corpus and skip flagged content.\n"
                f"Inspect data/quiz_review/ and either (a) re-run ship_readiness with\n"
                f"auto-fix to converge, or (b) pass --allow-partial to ship just the\n"
                f"ready chapters anyway.",
                file=sys.stderr,
            )
            return 2
        if review_count > 0 and allow_partial:
            print(
                f"  WARNING: {review_count} chapter(s) flagged for human review "
                f"are NOT in the bundle (--allow-partial set).",
            )

    stats = {}
    js_entries = []
    total_questions = 0

    for code in DOMAINS:
        domain_dir = source_dir / code
        questions = []

        if domain_dir.is_dir():
            for fp in sorted(domain_dir.glob("*.json")):
                # Skip audit/fix sidecars and manifests — only chapter
                # files contain the question list. (The bypass mode
                # `--source-dir data/quiz` exposes these sidecars.)
                if any(t in fp.name for t in
                       ("audit", "fixed", "bak", "backup",
                        "_layer_", "_pre_", "manifest")):
                    continue
                with open(fp, encoding="utf-8") as f:
                    chapter_qs = json.load(f)
                if not isinstance(chapter_qs, list):
                    continue
                questions.extend(chapter_qs)

        if not questions:
            print(f"  {code}: no questions, skipping")
            continue

        tier_counts = {1: 0, 2: 0, 3: 0, 4: 0}
        chapters = set()
        for q in questions:
            tier_counts[q.get("difficulty_tier", 0)] += 1
            chapters.add(q.get("chapter_file", "unknown"))

        # Write per-domain JSON
        dst = output_dir / f"{code}_quiz.json"
        with open(dst, "w", encoding="utf-8") as f:
            json.dump(questions, f, indent=2, ensure_ascii=False)
            f.write("\n")

        # JS bundle entry
        js_entries.append(f'  "{code}": {json.dumps(questions, ensure_ascii=False)}')

        stats[code] = {
            "domain_name": DOMAIN_NAMES.get(code, code),
            "total": len(questions),
            "chapters": len(chapters),
            "by_tier": tier_counts,
        }

        total_questions += len(questions)
        print(f"  {code}: {len(questions)} questions ({len(chapters)} chapters) "
              f"T1:{tier_counts[1]} T2:{tier_counts[2]} T3:{tier_counts[3]} T4:{tier_counts[4]}")

    # Write manifest
    out_manifest = output_dir / "quiz_stats.json"
    with open(out_manifest, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
        f.write("\n")
    print(f"\n  Manifest: {out_manifest.name}")

    # Write JS bundle (for file:// protocol compatibility)
    js_dst = output_dir / "quiz_data.js"
    with open(js_dst, "w", encoding="utf-8") as f:
        f.write("window.__QUIZ_DATA = {\n")
        f.write(",\n".join(js_entries))
        f.write("\n};\n")
    print(f"  Bundle: {js_dst.name} ({js_dst.stat().st_size:,} bytes)")

    print(f"\n  Total: {total_questions} questions across {len(stats)} domains")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--source-dir", default=str(QUIZ_SHIPPABLE_DIR),
        help=f"Source directory (default: {QUIZ_SHIPPABLE_DIR}). "
             "Override to bypass the ship-readiness gate (not recommended).",
    )
    parser.add_argument(
        "--output-dir", default=str(OUTPUT_DIR),
        help=f"Where to write the bundle (default: {OUTPUT_DIR} — "
             "production PassEPPP enrichment dir). Override to "
             "smoke-test bundle changes without clobbering production.",
    )
    parser.add_argument(
        "--no-require-manifest", action="store_true",
        help="Skip the ship-readiness manifest check. Use only when "
             "intentionally bypassing the gate (e.g., --source-dir data/quiz "
             "--no-require-manifest).",
    )
    parser.add_argument(
        "--allow-partial", action="store_true",
        help="Build the bundle even if the manifest reports chapters in "
             "review. Default behavior exits 2 to prevent silently shipping "
             "a partial corpus.",
    )
    args = parser.parse_args()
    src = pathlib.Path(args.source_dir).resolve()
    out = pathlib.Path(args.output_dir).resolve()
    rc = build(
        src, out,
        require_manifest=not args.no_require_manifest,
        allow_partial=args.allow_partial,
    )
    sys.exit(rc)
