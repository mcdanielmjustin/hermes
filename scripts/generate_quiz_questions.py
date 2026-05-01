"""
generate_quiz_questions.py — Anchor-Based Agent Pipeline

Generates 4-tier quiz questions by iterating over anchor points from
anchor_points.csv (1,566 anchors). Textbook passage context is loaded
from anchor_passages_v3 (1,081 in-book anchors with passage column).
Proprietary/lecture anchors (486) use verbatim_anchor + testable_fact only.

Pipeline phases:
  Phase 1: Preparation agents (hardcoded, instant, $0)
  Phase 2: Creative generation (1 focused LLM call)
  Phase 3: Assembly (hardcoded merge of creative + metadata)
  Phase 4: Validation gates (hardcoded checks)
  Phase 5: Smart retry (re-run Phase 2 only, prep data cached)

Usage:
  python generate_quiz_questions.py --domain BPSY --all-anchors
  python generate_quiz_questions.py --domain BPSY --chapter "Brain Structure"
  python generate_quiz_questions.py --domain BPSY --anchor D7-PHY-025-af701e64
  python generate_quiz_questions.py --all --difficulty 3 --resume --workers 10
  python generate_quiz_questions.py --domain BPSY --dry-run

Options:
  --domain CODE      Single domain (PMET, LDEV, CPAT, PTHE, SOCU, WDEV, BPSY, CASS, PETH)
  --all              Process all 9 domains
  --chapter SUBSTR   Filter anchors to chapters matching this substring
  --anchor UID       Single anchor UID (e.g., D7-PHY-025-af701e64)
  --difficulty N     Single tier (1-4), or omit for all tiers
  --count N          Variants per anchor per tier (default: 5)
  --workers N        Concurrent API workers (default: 5)
  --resume           Skip already-generated anchor/tier/variant combos
  --dry-run          Preview without API calls
  --api-key KEY      Override API key
"""

import csv, json, pathlib, argparse, time, sys, os, asyncio
from datetime import datetime, timezone
from collections import defaultdict
import anthropic
from pipeline.api_client import create_client

# ── Ensure pipeline package is importable ─────────────────────
SCRIPT_DIR = pathlib.Path(__file__).parent
REPO_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPT_DIR))

from pipeline import (
    DOMAIN_CODES, CODE_TO_ID, DOMAIN_NAMES,
    DIFFICULTY_LABELS, slugify, get_section_title,
)
from pipeline.orchestrator import QuestionOrchestrator

# ── Paths (centralized in config.py) ──────────────────────────
from config import (
    DATA_DIR, QUIZ_DIR, LOG_DIR,
    CONCEPT_VOCAB_DIR, ANCHOR_BRIEFS_DIR, DOMAIN_VOCAB_DIR,
    ANCHOR_POINTS_CSV, ANCHOR_PASSAGES_CSV,
    QUIZ_CHECKPOINT as CHECKPOINT_FILE,
)


# ── Utility functions ─────────────────────────────────────────

def load_api_key(args_key, provider="nous"):
    """Load API key from args, env, or .env file."""
    if args_key:
        return args_key
    
    # Try provider-specific env var first
    env_var = "NOUS_API_KEY" if provider == "nous" else "ANTHROPIC_API_KEY"
    if os.environ.get(env_var):
        return os.environ[env_var]
    
    # Fallback to .env file
    for p in [pathlib.Path(".env"), pathlib.Path.home() / ".env"]:
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.startswith(f"{env_var}="):
                    return line.split("=", 1)[1].strip().strip("\"'")
    
    raise RuntimeError(f"No API key found. Set {env_var} or pass --api-key.")


def load_json(path, encoding="utf-8"):
    with open(path, encoding=encoding) as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        return set(load_json(CHECKPOINT_FILE))
    return set()


def save_checkpoint(done_keys):
    """Save checkpoint with merge-and-atomic-rename semantics.

    Concurrent invocations of this script (e.g., parallel `--anchor X`
    runs) used to race-overwrite the checkpoint, losing other processes'
    progress. This implementation:

    1. Re-reads the current on-disk checkpoint just before writing.
    2. Merges it into the in-memory `done_keys` (union).
    3. Writes to a temp file in the same directory.
    4. Atomically renames the temp file over the real path.

    Race window narrowed from "entire process lifetime" to "the few
    microseconds between read-and-merge and rename." Two processes can
    still race at the very last step, but the loser's data is already
    in the winner's file (because both saw it during merge). True
    process-level safety would need an OS file-lock; this is a
    reasonable compromise that fits the existing pattern.
    """
    import os
    import tempfile

    # Re-read on-disk state and merge — preserves keys written by
    # concurrent processes since this process's last load_checkpoint().
    on_disk = set()
    if CHECKPOINT_FILE.exists():
        try:
            on_disk = set(load_json(CHECKPOINT_FILE))
        except Exception:
            on_disk = set()
    merged = sorted(set(done_keys) | on_disk)

    CHECKPOINT_FILE.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(
        dir=str(CHECKPOINT_FILE.parent),
        prefix=".checkpoint.", suffix=".tmp",
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(merged, fh, indent=2, ensure_ascii=False)
        # Atomic on POSIX; on Windows, os.replace overwrites the target
        # atomically too (unlike rename which fails if target exists).
        os.replace(tmp_path, str(CHECKPOINT_FILE))
    except Exception:
        # If anything goes wrong, clean up the temp file rather than
        # leaving it as litter in the scripts/ dir.
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def load_anchor_csv(path):
    """Load anchor_points.csv into a dict keyed by uid."""
    anchors = {}
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            anchors[row["uid"]] = row
    return anchors


def load_passage_csv(path):
    """Load anchor_passages_v3 CSV into a dict keyed by uid."""
    passages = {}
    if not path.exists():
        return passages
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            passages[row["uid"]] = row
    return passages


def setup_file_logger(batch_id):
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"{batch_id}.log"
    log_file = open(log_path, "a", encoding="utf-8")
    return log_file, log_path


def log_event(log_file, event_type, data):
    if log_file is None:
        return
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "event": event_type,
        **data,
    }
    log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    log_file.flush()


# ── Shared state for concurrent generation ────────────────────

class GenerationState:
    """Lock-protected shared state for concurrent question generation."""

    def __init__(self, done_keys, log_file):
        self.lock = asyncio.Lock()
        self.done_keys = done_keys
        self.log_file = log_file
        self.total_generated = 0
        self.total_failed = 0
        self.total_skipped = 0
        self.total_tokens_in = 0
        self.total_tokens_out = 0
        self._existing = []
        self._existing_ids = set()
        self._output_path = None

    def set_chapter(self, existing, existing_ids, output_path):
        self._existing = existing
        self._existing_ids = existing_ids
        self._output_path = output_path


# ── Phase 2-5: Async question task processor ──────────────────

async def process_question_task(semaphore, client, task, state, orch):
    """Generate, assemble, validate (with smart retry), and save one question."""
    qid = task["question_id"]

    async with semaphore:
        gate_context = {"existing_ids": state._existing_ids}
        result, tokens_in, tokens_out, failure_reason = await orch.generate_and_validate(
            client, task, gate_context=gate_context,
        )

    async with state.lock:
        state.total_tokens_in += tokens_in
        state.total_tokens_out += tokens_out

        # generate_and_validate returns either:
        #   - successful assembly: {... no _failed_validation key}
        #   - failed assembly with diagnostic content: {_failed_validation: True}
        #   - None: failed before any assembly produced (e.g., LLM error)
        if result is None or result.get("_failed_validation"):
            print(f"    ✗ {qid} — {failure_reason}")
            state.total_failed += 1
            failure_log = {
                "question_id": qid,
                "reason": failure_reason,
                "attempts": 2,
            }
            # If we have the failed assembly, capture stem + option text
            # in the log so diagnostic tools can inspect what was
            # actually generated, not just the failure reason.
            if result is not None:
                failure_log["question_stem"] = result.get("question_stem", "")
                failure_log["options"] = [
                    {
                        "letter": o.get("letter"),
                        "is_correct": o.get("is_correct", False),
                        "text": o.get("text", ""),
                    }
                    for o in result.get("options", [])
                ]
            log_event(state.log_file, "validation_fail", failure_log)
            return

        # Attach a snapshot of the brief that drove this question (concepts,
        # misconceptions, core_claims, question_angles) for downstream
        # inspection and audit.
        if "anchor_brief" in task:
            result["anchor_brief"] = task["anchor_brief"]

        state._existing.append(result)
        state._existing_ids.add(qid)
        state.total_generated += 1

        save_json(state._output_path, state._existing)
        state.done_keys.add(task["checkpoint_key"])
        save_checkpoint(state.done_keys)

        concept_id = result.get("tested_concept", {}).get("concept_id", "?")
        print(f"    ✓ {qid} ({concept_id})")

        log_event(state.log_file, "question_generated", {
            "question_id": qid,
            "concept_id": concept_id,
            "tokens": tokens_in + tokens_out,
        })


# ── Main ──────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser(description="Generate quiz questions (anchor-based pipeline)")
    parser.add_argument("--domain", type=str, help="Domain code (e.g., BPSY)")
    parser.add_argument("--all", action="store_true", help="Process all 9 domains")
    parser.add_argument("--chapter", type=str, help="Filter to anchors whose chapter_title contains this substring")
    parser.add_argument("--anchor", type=str, help="Single anchor UID (e.g., D7-PHY-025-af701e64)")
    parser.add_argument("--difficulty", type=int, choices=[1, 2, 3, 4], help="Single difficulty tier")
    parser.add_argument("--count", type=int, default=5, help="Variants per anchor per tier (default: 5)")
    parser.add_argument("--workers", type=int, default=5, help="Concurrent API workers (default: 5)")
    parser.add_argument("--resume", action="store_true", help="Skip already-generated combos")
    parser.add_argument("--clean", action="store_true", help="Delete existing output before generating")
    parser.add_argument("--dry-run", action="store_true", help="Preview without API calls")
    parser.add_argument("--api-key", type=str, help="Override API key")
        parser.add_argument("--provider", type=str, default="nous", 
                        choices=["nous", "anthropic"], 
                        help="API provider (nous for Qwen, anthropic for Claude)")
    parser.add_argument("--base-url", type=str, default=None,
                        help="API base URL (default: Nous inference API)")
    parser.add_argument("--model", type=str, default=None,
                        help="Model to use (default: qwen/qwen3.5-plus-02-15 for nous, claude-opus-4-7 for anthropic)")
parser.add_argument(
        "--prompt-version", type=str, choices=["v1", "v2"], default="v2",
        help="Generation system-prompt version. v2 (default, post-Phase-18) "
             "adds the Distractor Quality Framework section that mirrors "
             "the audit's 3-class scheme — empirical english_gap rate ~1%% "
             "vs ~4%% under v1. v1 is retained for cohort comparisons.",
    )
    args = parser.parse_args()

    if not args.domain and not args.all:
        parser.error("Specify --domain CODE or --all")
    if args.clean and args.resume:
        parser.error("--clean and --resume are mutually exclusive")

    # ── Resolve domains and tiers ─────────────────────────────
    if args.all:
        domains = list(DOMAIN_CODES.values())
    else:
        args.domain = args.domain.upper()
        if args.domain not in CODE_TO_ID:
            parser.error(f"Unknown domain: {args.domain}")
        domains = [args.domain]

    tiers = [args.difficulty] if args.difficulty else [3, 2, 1, 4]

    # ── Load anchor data from CSVs ───────────────────────────
    if not ANCHOR_POINTS_CSV.exists():
        print(f"ERROR: {ANCHOR_POINTS_CSV} not found.")
        sys.exit(1)
    all_anchors = load_anchor_csv(ANCHOR_POINTS_CSV)
    passages = load_passage_csv(ANCHOR_PASSAGES_CSV)

    # Merge passage text into anchor data
    for uid, anchor in all_anchors.items():
        p = passages.get(uid)
        if p:
            anchor["passage"] = p.get("passage", "")
            if not anchor.get("domain_code") and p.get("domain_code"):
                anchor["domain_code"] = p["domain_code"]
        else:
            anchor["passage"] = ""

    textbook_count = sum(1 for a in all_anchors.values() if a["passage"])
    print(f"  Loaded {len(all_anchors)} anchors ({textbook_count} with textbook passage)")

    # Single-anchor mode
    if args.anchor:
        if args.anchor not in all_anchors:
            print(f"ERROR: Anchor UID '{args.anchor}' not found in {ANCHOR_POINTS_CSV.name}")
            sys.exit(1)
        anchor_filter = {args.anchor}
    else:
        anchor_filter = None

    done_keys = load_checkpoint() if args.resume else set()

    # ── Initialize orchestrator ─────────────────────────────────
    orch = QuestionOrchestrator()
    features = ["prompt-cache", "anchor-based", "anchor-briefs", "orchestrator"]
    print(f"  Pipeline: {orch.info()}")
    print(f"  Features: {', '.join(features)}")

    # ── Initialize async client ───────────────────────────────
    if not args.dry_run:
        api_key = load_api_key(args.api_key, provider=args.provider)
        client = create_client(
            provider=args.provider,
            api_key=api_key,
            base_url=args.base_url,
            model=args.model,
        )
    else:
        client = None

    batch_id = time.strftime("%Y-%m-%d") + "-quiz"

    # ── Logging ───────────────────────────────────────────────
    log_file = None
    log_path = None
    if not args.dry_run:
        log_file, log_path = setup_file_logger(batch_id)
        log_event(log_file, "run_start", {
            "domains": domains,
            "tiers": tiers,
            "count": args.count,
            "workers": args.workers,
            "resume": args.resume,
            "pipeline": "orchestrator_v2",
        })

    state = GenerationState(done_keys, log_file)
    semaphore = asyncio.Semaphore(args.workers)

    # ══════════════════════════════════════════════════════════
    # Main generation loop — iterate over anchors
    # ══════════════════════════════════════════════════════════

    for domain_code in domains:
        domain_id = CODE_TO_ID[domain_code]
        domain_name = DOMAIN_NAMES[domain_code]

        # Filter anchors for this domain
        domain_anchors = [
            a for a in all_anchors.values()
            if str(a.get("domain_num", "")) == str(domain_id)
        ]

        if anchor_filter:
            domain_anchors = [a for a in domain_anchors if a["uid"] in anchor_filter]
        if args.chapter:
            domain_anchors = [
                a for a in domain_anchors
                if args.chapter.lower() in a.get("chapter_title", "").lower()
            ]

        if not domain_anchors:
            print(f"\n[{domain_code}] No matching anchors. Skipping.")
            continue

        # Group anchors by chapter_title for output files
        chapters = defaultdict(list)
        for a in domain_anchors:
            chapters[a.get("chapter_title", "unknown")].append(a)

        textbook_in_domain = sum(1 for a in domain_anchors if a.get("passage"))
        print(f"\n{'='*60}")
        print(f"  Domain: {domain_code} — {domain_name}")
        print(f"  Anchors: {len(domain_anchors)} ({textbook_in_domain} with passage)")
        print(f"  Chapters: {len(chapters)}")
        print(f"{'='*60}")

        for chapter_title, chapter_anchors in sorted(chapters.items()):
            chapter_slug = slugify(chapter_title)

            # Resolve section_title for frontend display
            section_title = get_section_title(domain_code, chapter_slug, chapter_title)

            # Prepare output file
            quiz_domain_dir = QUIZ_DIR / domain_code
            quiz_domain_dir.mkdir(parents=True, exist_ok=True)
            output_path = quiz_domain_dir / f"{chapter_slug}.json"

            # ── Clean mode ────────────────────────────────────
            if args.clean and not args.dry_run:
                if output_path.exists():
                    old_count = len(load_json(output_path))
                    output_path.unlink()
                    print(f"  --clean: removed {output_path.name} ({old_count} old questions)")
                if CHECKPOINT_FILE.exists():
                    all_cp = set(load_json(CHECKPOINT_FILE))
                    prefix = f"QZ-{domain_code}-"
                    purged = {k for k in all_cp if k.startswith(prefix) and chapter_slug in k}
                    if purged:
                        save_checkpoint(all_cp - purged)
                        print(f"  --clean: purged {len(purged)} checkpoint entries")

            existing = []
            existing_ids = set()
            if output_path.exists():
                existing = load_json(output_path)
                existing_ids = {r.get("question_id") for r in existing if r.get("question_id")}

            # Load concept vocab for this chapter (once per chapter)
            chapter_id = chapter_anchors[0].get("chapter_num", "")
            chapter_vocab = orch.load_chapter_context(CONCEPT_VOCAB_DIR, domain_code, chapter_id)
            has_vocab = chapter_vocab.get("has_vocab", False)
            ch_concepts = chapter_vocab.get("concepts", [])
            ch_misconceptions = chapter_vocab.get("misconceptions", [])

            vocab_label = f"{len(ch_concepts)}C/{len(ch_misconceptions)}M" if has_vocab else "no vocab"
            print(f"\n  Chapter: {chapter_title} ({len(chapter_anchors)} anchors, {vocab_label})")

            # ── Phase 1: Collect tasks ────────────────────────

            chapter_tasks = []

            for anchor_idx, anchor in enumerate(chapter_anchors):
                uid = anchor["uid"]
                passage = anchor.get("passage", "")

                anchor_ctx = orch.load_anchor_context(
                    ANCHOR_BRIEFS_DIR, domain_code, uid,
                    chapter_vocab, passage_text=passage,
                )

                for tier in tiers:
                    for variant in range(1, args.count + 1):
                        task = orch.prepare_task(
                            anchor, anchor_ctx, tier, variant,
                            domain_code, domain_id, domain_name,
                            chapter_title, section_title, batch_id,
                            anchor_idx=anchor_idx,
                            total_tiers_count=len(tiers),
                            variants_per_tier=args.count,
                            # Phase 7: domain vocab pool source dir.
                            domain_vocab_dir=DOMAIN_VOCAB_DIR,
                            # P6: pass through the v2 prompt version flag.
                            prompt_version=args.prompt_version,
                        )

                        question_id = task["question_id"]

                        if question_id in done_keys:
                            state.total_skipped += 1
                            continue

                        if question_id in existing_ids:
                            state.total_skipped += 1
                            continue

                        if args.dry_run:
                            mb = task["meta_base"]
                            model_label = "opus" if task["model"] == "claude-opus-4-7" else "sonnet"
                            has_passage = "passage" if passage else "no-passage"
                            print(f"    [DRY-RUN] {question_id} | {DIFFICULTY_LABELS[tier]} | {mb['source_type']} | {mb['stem_pattern']} | {model_label} | {has_passage}")
                            state.total_skipped += 1
                            continue

                        chapter_tasks.append(task)

            # ── Dispatch tasks concurrently ───────────────────
            if chapter_tasks:
                state.set_chapter(existing, existing_ids, output_path)
                if args.workers > 1:
                    print(f"    Dispatching {len(chapter_tasks)} questions ({args.workers} workers)...")
                await asyncio.gather(*[
                    process_question_task(semaphore, client, t, state, orch)
                    for t in chapter_tasks
                ])

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  DONE (orchestrator pipeline v2)")
    print(f"  Generated: {state.total_generated}")
    print(f"  Skipped: {state.total_skipped}")
    print(f"  Failed: {state.total_failed}")
    if state.total_tokens_in or state.total_tokens_out:
        opus_cost = (state.total_tokens_in * 15 / 1_000_000) + (state.total_tokens_out * 75 / 1_000_000)
        print(f"  Opus tokens: {state.total_tokens_in:,} in + {state.total_tokens_out:,} out (${opus_cost:.2f})")
        print(f"  Estimated cost: ${opus_cost:.2f}")
    if args.workers > 1:
        print(f"  Workers: {args.workers}")
    print(f"{'='*60}")

    log_event(log_file, "run_complete", {
        "generated": state.total_generated,
        "skipped": state.total_skipped,
        "failed": state.total_failed,
        "tokens_in": state.total_tokens_in,
        "tokens_out": state.total_tokens_out,
        "pipeline": "orchestrator_v2",
    })

    if log_file:
        log_file.close()
        print(f"  Log: {log_path}")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    asyncio.run(main())
