"""
generate_concept_vocab.py

Pre-generates canonical concept_id and misconception_id vocabularies
for each chapter. This ensures consistent IDs across all questions
(5 variants x 4 tiers) generated per chapter.

Run BEFORE generate_quiz_questions.py.

Data sources:
  - chapter_schema_v3.csv  (chapter structure + anchor metadata)
  - anchor_passages_v3.csv (passage text + verbatim anchors)

Usage:
  python generate_concept_vocab.py --domain BPSY
  python generate_concept_vocab.py --domain BPSY --chapter D7-Ch03
  python generate_concept_vocab.py --all
  python generate_concept_vocab.py --all --resume
  python generate_concept_vocab.py --all --dry-run
  python generate_concept_vocab.py --all --model sonnet  # cheaper for this task

Output:
  data/concept_vocab/{DOMAIN_CODE}/{chapter_id}.json
"""

import json, pathlib, argparse, time, sys, os, re, csv
from collections import defaultdict
import anthropic

# ── Paths (centralized in config.py) ──────────────────────────
SCRIPT_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from config import (
    CONCEPT_VOCAB_DIR,
    CHAPTER_SCHEMA_CSV, ANCHOR_PASSAGES_CSV,
    CONCEPT_VOCAB_CHECKPOINT as CHECKPOINT_FILE,
)

# ── Domain codes (from shared source of truth) ────────────────
from shared_constants import DOMAIN_CODES, CODE_TO_ID, DOMAIN_NAMES

# chapter_schema_v3.csv: rows 0-10 are summary, data starts at row 11
SCHEMA_DATA_START = 11

MODEL_MAP = {
    "opus": "claude-opus-4-7",
    "sonnet": "claude-sonnet-4-6",
}

SYSTEM_PROMPT = """You are building a concept vocabulary for the PassEPPP EPPP exam preparation platform.

Your task: for a given textbook chapter, generate the canonical set of concept_ids and
misconception_ids that ALL quiz questions about this chapter must use. This ensures
consistency across all questions generated per chapter.

## concept_id Rules
- Format: `{subsystem}-{specific-concept}` (kebab-case)
- Examples: `sensory-memory-iconic-duration`, `operant-conditioning-positive-reinforcement`
- Be specific enough to distinguish from neighboring concepts
- Be general enough that all questions about this concept use the SAME id
- Include the parent system/category as a prefix for disambiguation

## misconception_id Rules
- Format: `{concept-a}-vs-{concept-b}` (kebab-case)
- Examples: `iconic-vs-echoic-duration`, `positive-reinforcement-vs-negative-reinforcement`
- Represent REAL confusions that EPPP students commonly make
- Each misconception must involve 2+ concept_ids from your concept list

## misconception_type Values (exactly one per misconception)
- `similar_name` — Concepts with confusable names (retroactive vs proactive)
- `similar_property` — Different concepts sharing a surface feature (STM vs sensory duration)
- `similar_store` — Same system, different modality/subtype (iconic vs echoic)
- `opposite_direction` — Getting the direction/effect backwards (positive punishment)
- `overgeneralization` — Applying a rule beyond its valid scope
- `partial_understanding` — Correct concept, wrong context/application

## Output Format
Return ONLY valid JSON (no markdown, no explanation):
{
  "concepts": [
    {
      "concept_id": "kebab-case-id",
      "label": "Human-readable label (2-5 words)",
      "description": "What this concept covers (1 sentence)"
    }
  ],
  "misconceptions": [
    {
      "misconception_id": "concept-a-vs-concept-b",
      "label": "Human-readable confusion label",
      "type": "similar_name|similar_property|similar_store|opposite_direction|overgeneralization|partial_understanding",
      "concepts_involved": ["concept-id-1", "concept-id-2"]
    }
  ]
}

Generate 5-8 concepts and 3-5 misconceptions."""


# ── Data loading ──────────────────────────────────────────────

def load_chapter_schema():
    """Load chapter_schema_v3.csv, return dict keyed on (domain_code, chapter_id)."""
    chapters = defaultdict(lambda: {"anchors": [], "chapter_title": "", "chapter_id": ""})
    with open(CHAPTER_SCHEMA_CSV, encoding="utf-8") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i < SCHEMA_DATA_START:
                continue
            if len(row) < 14:
                continue
            chapter_id = row[0].strip()
            chapter_title = row[1].strip()
            uid = row[5].strip()
            content_summary = row[8].strip() if len(row) > 8 else ""
            topic = row[9].strip() if len(row) > 9 else ""
            testable_fact = row[12].strip() if len(row) > 12 else ""
            domain_code = row[13].strip()

            key = (domain_code, chapter_id)
            chapters[key]["chapter_id"] = chapter_id
            chapters[key]["chapter_title"] = chapter_title
            chapters[key]["domain_code"] = domain_code
            chapters[key]["anchors"].append({
                "uid": uid,
                "content_summary": content_summary,
                "topic": topic,
                "testable_fact": testable_fact,
            })
    return dict(chapters)


def load_passages():
    """Load anchor_passages_v3.csv, return dict keyed on uid."""
    passages = {}
    with open(ANCHOR_PASSAGES_CSV, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if len(row) < 13:
                continue
            uid = row[0].strip()
            passages[uid] = {
                "verbatim_anchor": row[9].strip() if len(row) > 9 else "",
                "testable_fact": row[11].strip() if len(row) > 11 else "",
                "passage": row[12].strip() if len(row) > 12 else "",
            }
    return passages


# ── Helpers ────────────────────────────────────────────────────

def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        return set(load_json(CHECKPOINT_FILE))
    return set()


def save_checkpoint(done_keys):
    save_json(CHECKPOINT_FILE, sorted(done_keys))


def build_user_prompt(chapter_id, chapter_title, domain_name, anchors, passages):
    """Build the prompt for concept vocabulary generation."""
    anchor_lines = []
    for a in anchors:
        uid = a["uid"]
        p = passages.get(uid, {})
        verbatim = p.get("verbatim_anchor", "")
        testable = a.get("testable_fact") or p.get("testable_fact", "")
        passage = p.get("passage", "")
        snippet = passage[:400] if passage else ""

        parts = [f"- [{uid}]"]
        if verbatim:
            parts.append(f"  Anchor: {verbatim[:200]}")
        if testable:
            parts.append(f"  Testable: {testable[:200]}")
        if snippet:
            parts.append(f"  Passage: {snippet}...")
        anchor_lines.append("\n".join(parts))

    anchor_block = "\n\n".join(anchor_lines) if anchor_lines else "(no anchors)"

    return f"""Generate the canonical concept vocabulary for this EPPP textbook chapter.

DOMAIN: {domain_name}
CHAPTER: {chapter_id} — {chapter_title}
ANCHOR COUNT: {len(anchors)}

## Anchor Points:
{anchor_block}

Generate the vocabulary now. Return ONLY the JSON object."""


def generate_vocab(client, user_prompt, model_id, max_retries=2):
    """Call Claude API for concept vocabulary generation."""
    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(
                model=model_id,
                max_tokens=2000,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
            result = json.loads(text)
            tokens = response.usage.input_tokens + response.usage.output_tokens
            return result, tokens
        except json.JSONDecodeError as e:
            if attempt < max_retries:
                print(f"    JSON parse error, retrying ({attempt + 1}/{max_retries})...")
                time.sleep(2)
            else:
                print(f"    FAILED to parse JSON: {e}")
                return None, 0
        except anthropic.APIError as e:
            if attempt < max_retries:
                wait = 10 * (attempt + 1)
                print(f"    API error: {e}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    API FAILED: {e}")
                return None, 0


def validate_vocab(vocab):
    """Validate the generated concept vocabulary structure."""
    if not vocab or not isinstance(vocab, dict):
        return False, "not a dict"
    concepts = vocab.get("concepts", [])
    if not concepts or len(concepts) < 3:
        return False, f"need 3+ concepts, got {len(concepts)}"
    for c in concepts:
        if not c.get("concept_id") or not c.get("label"):
            return False, f"concept missing concept_id or label: {c}"
    misconceptions = vocab.get("misconceptions", [])
    if not misconceptions or len(misconceptions) < 2:
        return False, f"need 2+ misconceptions, got {len(misconceptions)}"
    for m in misconceptions:
        if not m.get("misconception_id") or not m.get("type"):
            return False, f"misconception missing id or type: {m}"
        min_concepts = 1 if m.get("type") == "opposite_direction" else 2
        if not m.get("concepts_involved") or len(m["concepts_involved"]) < min_concepts:
            return False, f"misconception needs {min_concepts}+ concepts_involved: {m}"
    return True, "ok"


# ── Main ───────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate canonical concept vocabularies for quiz question generation"
    )
    parser.add_argument("--domain", type=str, help="Domain code (e.g., BPSY)")
    parser.add_argument("--all", action="store_true", help="Process all 9 domains")
    parser.add_argument("--chapter", type=str, help="Chapter ID (e.g., D7-Ch03)")
    parser.add_argument("--resume", action="store_true", help="Skip already-generated chapters")
    parser.add_argument("--dry-run", action="store_true", help="Preview without API calls")
    parser.add_argument("--model", type=str, default="opus",
                        choices=["opus", "sonnet"],
                        help="Model to use (default: opus). Sonnet is cheaper for this task.")
    parser.add_argument("--api-key", type=str, help="Override API key")
    args = parser.parse_args()

    if not args.domain and not args.all:
        parser.error("Specify --domain CODE or --all")

    model_id = MODEL_MAP[args.model]

    if args.all:
        target_domains = set(DOMAIN_CODES.values())
    else:
        args.domain = args.domain.upper()
        if args.domain not in CODE_TO_ID:
            parser.error(f"Unknown domain: {args.domain}")
        target_domains = {args.domain}

    # Load master CSVs
    print("Loading chapter_schema_v3.csv...")
    chapters = load_chapter_schema()
    print(f"  {len(chapters)} chapters loaded")

    print("Loading anchor_passages_v3.csv...")
    passages = load_passages()
    print(f"  {len(passages)} passages loaded")

    done_keys = load_checkpoint() if args.resume else set()

    # Setup API client
    if not args.dry_run:
        api_key = None
        if args.api_key:
            api_key = args.api_key
        elif os.environ.get("ANTHROPIC_API_KEY"):
            api_key = os.environ["ANTHROPIC_API_KEY"]
        else:
            for p in [pathlib.Path(".env"), pathlib.Path.home() / ".env"]:
                if p.exists():
                    for line in p.read_text(encoding="utf-8").splitlines():
                        if line.startswith("ANTHROPIC_API_KEY="):
                            api_key = line.split("=", 1)[1].strip().strip("\"'")
                            break
        if not api_key:
            print("ERROR: No API key. Set ANTHROPIC_API_KEY or pass --api-key.")
            sys.exit(1)
        client = anthropic.Anthropic(api_key=api_key)
    else:
        client = None

    total_generated = 0
    total_skipped = 0
    total_failed = 0
    total_tokens = 0

    print(f"\nConcept Vocabulary Generator")
    print(f"  Model: {model_id}")
    print(f"  Domains: {', '.join(sorted(target_domains))}")
    print(f"  Resume: {args.resume}")
    print()

    # Group chapters by domain
    by_domain = defaultdict(list)
    for (domain_code, chapter_id), chapter_data in chapters.items():
        if domain_code in target_domains:
            by_domain[domain_code].append((chapter_id, chapter_data))

    for domain_code in sorted(by_domain.keys()):
        domain_name = DOMAIN_NAMES[domain_code]
        domain_chapters = sorted(by_domain[domain_code], key=lambda x: x[0])

        if args.chapter:
            domain_chapters = [(cid, cd) for cid, cd in domain_chapters if cid == args.chapter]
            if not domain_chapters:
                continue

        print(f"\n{'='*60}")
        print(f"  {domain_code} — {domain_name} ({len(domain_chapters)} chapters)")
        print(f"{'='*60}")

        for chapter_id, chapter_data in domain_chapters:
            chapter_title = chapter_data["chapter_title"]
            anchors = chapter_data["anchors"]
            checkpoint_key = f"{domain_code}:{chapter_id}"

            if checkpoint_key in done_keys:
                total_skipped += 1
                continue

            vocab_path = CONCEPT_VOCAB_DIR / domain_code / f"{chapter_id}.json"
            if vocab_path.exists() and args.resume:
                done_keys.add(checkpoint_key)
                total_skipped += 1
                continue

            if args.dry_run:
                matched = sum(1 for a in anchors if a["uid"] in passages)
                print(f"  [DRY-RUN] {chapter_id} | {len(anchors)} anchors | {matched} with passages")
                total_skipped += 1
                continue

            user_prompt = build_user_prompt(
                chapter_id, chapter_title, domain_name, anchors, passages,
            )

            print(f"  {chapter_id} ({len(anchors)} anchors)...", end=" ", flush=True)
            vocab, tokens = generate_vocab(client, user_prompt, model_id)

            total_tokens += tokens

            if vocab is None:
                print("FAILED")
                total_failed += 1
                continue

            valid, reason = validate_vocab(vocab)
            if not valid:
                print(f"INVALID ({reason})")
                total_failed += 1
                continue

            output = {
                "chapter_id": chapter_id,
                "chapter_title": chapter_title,
                "domain_code": domain_code,
                "domain_name": domain_name,
                "anchor_count": len(anchors),
                "concepts": vocab["concepts"],
                "misconceptions": vocab["misconceptions"],
            }
            save_json(vocab_path, output)

            done_keys.add(checkpoint_key)
            save_checkpoint(done_keys)

            n_concepts = len(vocab.get("concepts", []))
            n_misconceptions = len(vocab.get("misconceptions", []))
            print(f"OK ({n_concepts}C, {n_misconceptions}M)")
            total_generated += 1

            time.sleep(1.0)

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Generated: {total_generated} chapter vocabularies")
    print(f"  Skipped: {total_skipped}")
    print(f"  Failed: {total_failed}")
    print(f"  Total tokens: {total_tokens:,}")
    if total_tokens:
        if "opus" in model_id:
            cost = total_tokens * 45 / 1_000_000
        else:
            cost = total_tokens * 9 / 1_000_000
        print(f"  Estimated cost: ${cost:.2f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
