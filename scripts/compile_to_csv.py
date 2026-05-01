"""
compile_to_csv.py — Flatten pipeline JSON output into the 58-column master CSV.

Reads all quiz JSON files from QUIZ_DIR, flattens nested structures
(options[], tested_concept{}, flashcard_seeds{}, distractor metadata),
joins anchor CSV data (chapter_num, testable_fact), and writes to
ENRICHMENT_CSV.

Modes:
  --full       Rewrite the CSV from scratch (all JSON files)
  --append     Append only new questions (skip existing question_ids)

Usage:
  python compile_to_csv.py --full
  python compile_to_csv.py --append
  python compile_to_csv.py --full --source-dir path/to/quiz/jsons
"""

import csv, json, argparse, sys
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

from config import QUIZ_DIR, ENRICHMENT_CSV, ANCHOR_POINTS_CSV
from shared_constants import CODE_TO_ID

CSV_COLUMNS = [
    "question_id", "question_type",
    "domain_code", "domain_id", "domain_name",
    "chapter_num", "chapter_title", "chapter_file",
    "section_title", "chapter_uuid", "anchor_uuid", "anchor_label",
    "difficulty_tier", "difficulty_label",
    "blooms_primary", "blooms_secondary",
    "stem_pattern", "source_type", "variant",
    "question_stem",
    "option_a", "option_b", "option_c", "option_d",
    "correct_answer",
    "explanation_a", "explanation_b", "explanation_c", "explanation_d",
    "tested_concept_id", "tested_concept_label", "knowledge_tested",
    "anchor_uid", "anchor_point_id_v2",
    "anchor_content_summary",
    "flashcard_concept_front", "flashcard_concept_back",
    "flashcard_comparison_front", "flashcard_comparison_back",
    "flashcard_nuance_front", "flashcard_nuance_back",
    "distractor_1_letter", "distractor_1_level", "distractor_1_misconception_type",
    "distractor_2_letter", "distractor_2_level", "distractor_2_misconception_type",
    "distractor_3_letter", "distractor_3_level", "distractor_3_misconception_type",
    "distractor_1_concept_id", "distractor_1_misconception_id", "distractor_1_confused_with",
    "distractor_2_concept_id", "distractor_2_misconception_id", "distractor_2_confused_with",
    "distractor_3_concept_id", "distractor_3_misconception_id", "distractor_3_confused_with",
    "generation_batch", "generated_by", "testable_fact",
]


def load_anchor_lookup(path):
    """Load anchor_points.csv into uid-keyed lookup for chapter_num + testable_fact."""
    lookup = {}
    if not path.exists():
        return lookup
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            lookup[row["uid"]] = row
    return lookup


def flatten_question(q, anchor_lookup):
    """Flatten one pipeline JSON question into a 62-column dict."""
    row = {}

    row["question_id"] = q.get("question_id", "")
    row["question_type"] = q.get("question_type", "single_choice")

    row["domain_code"] = q.get("domain_code", "")
    row["domain_id"] = str(CODE_TO_ID.get(row["domain_code"], ""))
    row["domain_name"] = q.get("domain_name", "")

    row["chapter_title"] = q.get("chapter_title", "")
    row["chapter_file"] = q.get("chapter_file", "")
    row["section_title"] = q.get("section_title", "")
    row["chapter_uuid"] = q.get("chapter_uuid", "")
    row["anchor_uuid"] = q.get("anchor_uuid", "")
    row["anchor_label"] = q.get("anchor_label", "")
    row["difficulty_tier"] = str(q.get("difficulty_tier", ""))
    row["difficulty_label"] = q.get("difficulty_label", "")
    row["blooms_primary"] = q.get("blooms_primary", "")
    row["blooms_secondary"] = q.get("blooms_secondary", "")
    row["stem_pattern"] = q.get("stem_pattern", "")
    row["source_type"] = q.get("source_type", "")
    row["variant"] = str(q.get("variant", ""))
    row["question_stem"] = q.get("question_stem", "")
    row["correct_answer"] = q.get("correct_answer_letter", "")

    # Flatten options
    options = q.get("options", [])
    option_map = {o["letter"]: o for o in options if "letter" in o}

    for letter in ["A", "B", "C", "D"]:
        opt = option_map.get(letter, {})
        row[f"option_{letter.lower()}"] = opt.get("text", "")
        row[f"explanation_{letter.lower()}"] = opt.get("explanation", "")

    # Flatten tested_concept
    tc = q.get("tested_concept", {}) or {}
    row["tested_concept_id"] = tc.get("concept_id", "")
    row["tested_concept_label"] = tc.get("concept_label", "")
    row["knowledge_tested"] = tc.get("knowledge_tested", "")

    # Flatten anchor arrays to scalar (first element)
    uids = q.get("anchor_uids", [])
    ids_v2 = q.get("anchor_point_ids_v2", [])
    summaries = q.get("anchor_content_summaries", [])
    row["anchor_uid"] = uids[0] if uids else ""
    row["anchor_point_id_v2"] = ids_v2[0] if ids_v2 else ""
    row["anchor_content_summary"] = summaries[0] if summaries else ""

    # Join anchor CSV data.
    #
    # By design (2026-04-26): the analytics CSV mixes two presentations of
    # the same anchor on purpose.
    #   • anchor_label / anchor_content_summary  ← from question JSON
    #     (sanitized via InputSanitizerAgent at generation time — these
    #     are the CANONICAL student-facing presentations, no citations)
    #   • testable_fact                          ← joined fresh from anchor CSV
    #     (raw original text, kept for ANALYTICS CONTEXT — citations preserved
    #     so analysts can trace anchor provenance)
    # Do NOT "fix" this divergence; testable_fact is contextual, not canonical.
    uid = row["anchor_uid"]
    anchor = anchor_lookup.get(uid, {})
    row["chapter_num"] = anchor.get("chapter_num", "")
    row["testable_fact"] = anchor.get("testable_fact", "")

    # Flatten flashcard_seeds
    seeds = q.get("flashcard_seeds", {}) or {}
    for card_type in ("concept", "comparison", "nuance"):
        seed = seeds.get(card_type, {}) or {}
        if isinstance(seed, str):
            row[f"flashcard_{card_type}_front"] = ""
            row[f"flashcard_{card_type}_back"] = seed
        else:
            row[f"flashcard_{card_type}_front"] = seed.get("front", "")
            row[f"flashcard_{card_type}_back"] = seed.get("back", "")

    # Flatten distractor metadata
    distractors = [o for o in options if not o.get("is_correct", False)]
    for i, dist in enumerate(distractors[:3], start=1):
        row[f"distractor_{i}_letter"] = dist.get("letter", "")
        row[f"distractor_{i}_level"] = str(dist.get("distractor_level", ""))
        row[f"distractor_{i}_misconception_type"] = dist.get("misconception_type", "")
        row[f"distractor_{i}_concept_id"] = dist.get("concept_id", "")
        row[f"distractor_{i}_misconception_id"] = dist.get("misconception_id", "")
        row[f"distractor_{i}_confused_with"] = dist.get("confused_with", "")

    # Fill missing distractor slots
    for i in range(len(distractors) + 1, 4):
        for suffix in ("letter", "level", "misconception_type",
                        "concept_id", "misconception_id", "confused_with"):
            row[f"distractor_{i}_{suffix}"] = ""

    row["generation_batch"] = q.get("generation_batch", "")
    row["generated_by"] = q.get("generated_by", "")

    return row


def load_json_questions(source_dir):
    """Load all question JSON files from source_dir/{DOMAIN}/*.json."""
    questions = []
    if not source_dir.exists():
        return questions
    for domain_dir in sorted(source_dir.iterdir()):
        if not domain_dir.is_dir():
            continue
        for json_file in sorted(domain_dir.glob("*.json")):
            if json_file.suffix != ".json":
                continue
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as e:
                print(f"  WARN: skipping {json_file}: {e}")
                continue
            if isinstance(data, list):
                questions.extend(data)
            elif isinstance(data, dict):
                questions.extend(data.values())
    return questions


def main():
    parser = argparse.ArgumentParser(description="Compile pipeline JSON to 58-column CSV")
    parser.add_argument("--full", action="store_true", help="Rewrite CSV from scratch")
    parser.add_argument("--append", action="store_true", help="Append new questions only")
    parser.add_argument("--source-dir", type=str, help="Override JSON source directory")
    args = parser.parse_args()

    if not args.full and not args.append:
        parser.error("Specify --full or --append")
    if args.full and args.append:
        parser.error("--full and --append are mutually exclusive")

    source_dir = Path(args.source_dir) if args.source_dir else QUIZ_DIR
    print(f"  Source: {source_dir}")
    print(f"  Target: {ENRICHMENT_CSV}")

    anchor_lookup = load_anchor_lookup(ANCHOR_POINTS_CSV)
    print(f"  Anchor lookup: {len(anchor_lookup)} entries")

    questions = load_json_questions(source_dir)
    print(f"  Loaded {len(questions)} questions from JSON")

    if not questions:
        print("  No questions found. Nothing to write.")
        return

    existing_ids = set()
    existing_rows = []
    if args.append and ENRICHMENT_CSV.exists():
        with open(ENRICHMENT_CSV, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_rows = list(reader)
            existing_ids = {r["question_id"] for r in existing_rows}
        print(f"  Existing CSV: {len(existing_rows)} rows, {len(existing_ids)} unique IDs")

    new_rows = []
    skipped = 0
    for q in questions:
        qid = q.get("question_id", "")
        if args.append and qid in existing_ids:
            skipped += 1
            continue
        flat = flatten_question(q, anchor_lookup)
        new_rows.append(flat)

    if args.append:
        print(f"  New questions: {len(new_rows)}, skipped (existing): {skipped}")
        all_rows = existing_rows + new_rows
    else:
        all_rows = new_rows

    ENRICHMENT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(ENRICHMENT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    print(f"  Written {len(all_rows)} rows × {len(CSV_COLUMNS)} columns to {ENRICHMENT_CSV.name}")

    # Summary by domain
    domain_counts = defaultdict(int)
    for r in (new_rows if args.append else all_rows):
        domain_counts[r.get("domain_code", "?")] += 1
    for code in sorted(domain_counts):
        print(f"    {code}: {domain_counts[code]}")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
