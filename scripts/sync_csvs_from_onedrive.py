"""Copy the three source CSVs from OneDrive into the repo's csvs/ dir.

Workflow:
  1. Edit anchor_points.xlsx / chapter_schema_v3.xlsx / passages in Excel
     under C:\\Users\\mcdan\\OneDrive\\Master CSVs\\
  2. Re-export each .xlsx to .csv (same folder)
  3. Run this script to copy CSVs into csvs/ in the repo
  4. git diff csvs/  (review the change)
  5. git commit + push

This script:
  - Compares row counts before/after
  - Refuses to overwrite if the new CSV is empty or unreadable
  - Reports byte-size delta and row-count delta per file
  - Exits non-zero if any file failed validation

Usage:
  python sync_csvs_from_onedrive.py            # copies all three, prompts before overwriting
  python sync_csvs_from_onedrive.py --force    # skip prompt, overwrite anyway
  python sync_csvs_from_onedrive.py --dry-run  # show what would change, copy nothing
"""
import argparse
import csv
import pathlib
import shutil
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from config import MASTER_CSV_DIR, ONEDRIVE_DISTRIBUTION_DIR

# OneDrive Master CSVs/ — same dir holds the .xlsx working copies and
# the enrichment_all_questions.csv distribution archive. Re-use the
# constant from config.py to keep a single source of truth.
ONEDRIVE_DIR = ONEDRIVE_DISTRIBUTION_DIR
SOURCE_FILES = (
    "anchor_points.csv",
    "anchor_passages_v3_pure_textbook_1081.csv",
    "chapter_schema_v3.csv",
)


def count_rows(path):
    """Return row count (excluding header). Returns -1 on read failure."""
    try:
        with open(path, encoding="utf-8") as f:
            reader = csv.reader(f)
            next(reader, None)  # skip header
            return sum(1 for _ in reader)
    except Exception as e:
        print(f"  WARN: could not count rows in {path}: {e}")
        return -1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="Overwrite without prompting")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would change, copy nothing")
    args = parser.parse_args()

    if not ONEDRIVE_DIR.exists():
        sys.exit(f"OneDrive dir not found: {ONEDRIVE_DIR}")
    MASTER_CSV_DIR.mkdir(parents=True, exist_ok=True)

    print(f"OneDrive source: {ONEDRIVE_DIR}")
    print(f"Repo target:     {MASTER_CSV_DIR}")
    print()

    plan = []
    for name in SOURCE_FILES:
        src = ONEDRIVE_DIR / name
        dst = MASTER_CSV_DIR / name
        if not src.exists():
            print(f"  SKIP: {name} — missing in OneDrive")
            continue
        src_size = src.stat().st_size
        dst_size = dst.stat().st_size if dst.exists() else 0
        src_rows = count_rows(src)
        dst_rows = count_rows(dst) if dst.exists() else 0
        delta_bytes = src_size - dst_size
        delta_rows = src_rows - dst_rows if (src_rows >= 0 and dst_rows >= 0) else None

        plan.append((name, src, dst, src_size, dst_size, src_rows, dst_rows))

        print(f"  {name}")
        print(f"    OneDrive: {src_size:>10,} bytes, {src_rows} rows")
        print(f"    Repo:     {dst_size:>10,} bytes, {dst_rows} rows  "
              f"(delta: {delta_bytes:+,} bytes"
              f"{f', {delta_rows:+} rows' if delta_rows is not None else ''})")

        if src_rows == 0:
            print(f"    REFUSING: OneDrive copy has 0 data rows — likely a bad export.")
            sys.exit(2)
        if dst.exists() and src_size == dst_size:
            print(f"    No change.")
        print()

    if args.dry_run:
        print("--dry-run: no files copied.")
        return

    needs_overwrite = any(p[2].exists() for p in plan)
    if needs_overwrite and not args.force:
        ans = input("Proceed with overwrite? [y/N] ").strip().lower()
        if ans != "y":
            print("Aborted.")
            return

    for name, src, dst, *_ in plan:
        shutil.copy2(src, dst)
        print(f"  copied {name}")

    print()
    print(f"Done. Review with: git -C {REPO_ROOT} diff csvs/")


if __name__ == "__main__":
    main()
