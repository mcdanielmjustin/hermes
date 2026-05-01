"""
batch_generate.py — Anthropic Message Batches API adapter (50% cost reduction)

Uses the Batch API to process quiz questions in bulk. Same pipeline agents
as generate_quiz_questions.py, but requests are submitted as async batches
(up to 10,000 per batch) processed within 24 hours.

Workflow:
  prepare → submit → [wait 1-24h] → collect

Subcommands:
  prepare   Run all 8 prep agents, build prompts, save manifest
  submit    Submit manifest to Batch API
  status    Check batch processing status
  collect   Download results, assemble, validate, save

Usage:
  python batch_generate.py prepare --all
  python batch_generate.py submit
  python batch_generate.py status
  python batch_generate.py collect
"""

import json, pathlib, argparse, time, sys, os, re, csv
from datetime import datetime, timezone
from collections import defaultdict

# ── Ensure pipeline package is importable ──────────────────
SCRIPT_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))

from pipeline import (
    DOMAIN_CODES, CODE_TO_ID, DOMAIN_NAMES,
    get_source_type,
    CORRECT_POSITIONS, slugify, get_section_title,
    get_stem_pattern, fix_mojibake_deep,
)
from pipeline.agents import (
    ConceptVocabAgent,
    DistractorPlannerAgent, KeywordExtractorAgent, MetadataAgent,
    TestedConceptSelectorAgent, FlashcardTemplateAgent,
    QuestionAssemblerAgent,
)
from pipeline.prompts import (
    build_system_prompt, build_user_prompt,
)
from pipeline.names import get_character_assignment
from pipeline.gates import create_gate_pipeline

# ── Paths (centralized in config.py) ──────────────────────
from config import (
    REPO_ROOT, DATA_DIR, QUIZ_DIR, BATCH_DIR,
    CONCEPT_VOCAB_DIR,
    ANCHOR_POINTS_CSV, ANCHOR_PASSAGES_CSV,
    BATCH_MANIFEST as MANIFEST_FILE,
)

MAX_BATCH_SIZE = 10_000  # Anthropic per-batch limit
MAX_CUSTOM_ID_LEN = 64   # Anthropic custom_id character limit
DEFAULT_THINK_BUDGET = 10_000  # tokens for extended thinking


# ── Utilities ──────────────────────────────────────────────

def load_json(path, encoding="utf-8"):
    with open(path, encoding=encoding) as f:
        return json.load(f)


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_api_key(args_key=None):
    if args_key:
        return args_key
    if os.environ.get("ANTHROPIC_API_KEY"):
        return os.environ["ANTHROPIC_API_KEY"]
    for p in [pathlib.Path(".env"), pathlib.Path.home() / ".env"]:
        if p.exists():
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.startswith("ANTHROPIC_API_KEY="):
                    return line.split("=", 1)[1].strip().strip("\"'")
    raise RuntimeError("No API key found. Set ANTHROPIC_API_KEY or pass --api-key.")


def make_short_id(question_id):
    """Create a batch-API-safe custom_id (max 64 chars).

    If the question_id fits, use it directly for readability.
    Otherwise, truncate and append a 6-char hash suffix to stay unique.
    """
    if len(question_id) <= MAX_CUSTOM_ID_LEN:
        return question_id
    # Keep as much of the ID as possible, append hash for uniqueness
    import hashlib
    suffix = hashlib.md5(question_id.encode()).hexdigest()[:6]
    return question_id[:MAX_CUSTOM_ID_LEN - 7] + "_" + suffix


def build_id_map(question_ids):
    """Build bidirectional mapping: short_id <-> question_id."""
    short_to_qid = {}
    for qid in question_ids:
        sid = make_short_id(qid)
        short_to_qid[sid] = qid
    return short_to_qid


def extract_text_block(msg):
    """Extract the text content from a message, skipping any thinking blocks."""
    for block in msg.content:
        if block.type == "text":
            return block.text.strip()
    return ""


def get_batch_client(api_key=None):
    """Create Anthropic client and verify batches API is available."""
    import anthropic
    client = anthropic.Anthropic(api_key=load_api_key(api_key))
    if not hasattr(client, "messages") or not hasattr(client.messages, "batches"):
        print("ERROR: Batch API not available in your anthropic SDK version.")
        print("  Run: pip install --upgrade anthropic")
        sys.exit(1)
    return client


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


# ══════════════════════════════════════════════════════════════
# PREPARE — Run prep agents, build batch manifest
# ══════════════════════════════════════════════════════════════

def cmd_prepare(args):
    """Run prep agents for every anchor × tier × variant, save manifest."""

    if MANIFEST_FILE.exists():
        existing = load_json(MANIFEST_FILE)
        phase = existing.get("phase", "prepared")
        if phase not in ("prepared", "complete"):
            print(f"ERROR: Active manifest in phase '{phase}'.")
            print(f"  Finish the current batch first, or delete {MANIFEST_FILE}.")
            sys.exit(1)

    # ── Resolve domains ────────────────────────────────────
    if args.all:
        domains = list(DOMAIN_CODES.values())
    else:
        code = args.domain.upper()
        if code not in CODE_TO_ID:
            print(f"ERROR: Unknown domain: {code}")
            sys.exit(1)
        domains = [code]

    tiers = args.difficulty
    count = args.count

    # ── Load anchor data from CSVs ─────────────────────────
    if not ANCHOR_POINTS_CSV.exists():
        print(f"ERROR: {ANCHOR_POINTS_CSV} not found.")
        sys.exit(1)
    all_anchors = load_anchor_csv(ANCHOR_POINTS_CSV)
    passages = load_passage_csv(ANCHOR_PASSAGES_CSV)

    for uid, anchor in all_anchors.items():
        p = passages.get(uid)
        if p:
            anchor["passage"] = p.get("passage", "")
            if not anchor.get("domain_code") and p.get("domain_code"):
                anchor["domain_code"] = p["domain_code"]
        else:
            anchor["passage"] = ""

    textbook_count = sum(1 for a in all_anchors.values() if a["passage"])
    print(f"  Loaded {len(all_anchors)} anchors ({textbook_count} with passage)")

    # ── Resume: collect existing question IDs ──────────────
    existing_ids = set()
    if args.resume:
        for dc in domains:
            quiz_dir = QUIZ_DIR / dc
            if quiz_dir.exists():
                for f in quiz_dir.glob("*.json"):
                    try:
                        for q in load_json(f):
                            qid = q.get("question_id")
                            if qid:
                                existing_ids.add(qid)
                    except (json.JSONDecodeError, OSError):
                        pass
        if existing_ids:
            print(f"  Resume: {len(existing_ids):,} existing questions found")

    # ── Initialize prep agents ─────────────────────────────
    concept_vocab_agent = ConceptVocabAgent()
    distractor_planner = DistractorPlannerAgent()
    keyword_extractor = KeywordExtractorAgent()
    metadata_agent = MetadataAgent()
    tested_concept_selector = TestedConceptSelectorAgent()
    flashcard_template = FlashcardTemplateAgent()

    batch_id = time.strftime("%Y-%m-%d") + "-batch"

    system_prompts = {}
    tasks = {}
    skipped = 0

    print(f"\n  Preparing batch manifest...")
    print(f"  Domains: {len(domains)} ({', '.join(domains[:5])}{'...' if len(domains) > 5 else ''})")
    print(f"  Tiers: {tiers} | Variants: {count}")
    print()

    for domain_code in domains:
        domain_id = CODE_TO_ID[domain_code]
        domain_name = DOMAIN_NAMES[domain_code]

        domain_anchors = [
            a for a in all_anchors.values()
            if str(a.get("domain_num", "")) == str(domain_id)
        ]
        if args.chapter:
            domain_anchors = [
                a for a in domain_anchors
                if args.chapter.lower() in a.get("chapter_title", "").lower()
            ]
        if not domain_anchors:
            continue

        chapters = defaultdict(list)
        for a in domain_anchors:
            chapters[a.get("chapter_title", "unknown")].append(a)

        domain_task_count = 0

        for chapter_title, chapter_anchors in sorted(chapters.items()):
            chapter_slug = slugify(chapter_title)
            section_title = get_section_title(domain_code, chapter_slug, chapter_title)

            # Load concept vocab once per chapter
            chapter_id = chapter_anchors[0].get("chapter_num", "")
            vocab_result = concept_vocab_agent.run({
                "concept_vocab_dir": CONCEPT_VOCAB_DIR,
                "domain_code": domain_code,
                "chapter_id": chapter_id,
            })
            has_vocab = vocab_result.get("has_vocab", False)
            prompt_mode = "focused" if has_vocab else "open"
            ch_concepts = vocab_result.get("concepts", [])
            ch_misconceptions = vocab_result.get("misconceptions", [])

            for anchor_idx, anchor in enumerate(chapter_anchors):
                uid = anchor["uid"]
                id_v2 = anchor.get("anchor_point_id_v2", "")
                passage = anchor.get("passage", "")
                verbatim = anchor.get("verbatim_anchor", "")
                testable_fact = anchor.get("testable_fact", "")

                sub_content = passage.strip()[:3000] if passage else ""
                content_chars = len(sub_content)

                anchor_data = [{
                    "uid": uid, "id_v2": id_v2,
                    "verbatim_anchor": verbatim, "testable_fact": testable_fact,
                    "text": verbatim, "id": id_v2,
                }]

                keyword_result = keyword_extractor.run({
                    "content": sub_content,
                    "concepts": ch_concepts,
                })
                topic_keywords = keyword_result.get("topic_keywords", [])

                anchor_info = {
                    "chapter_title": chapter_title,
                    "anchor_id_v2": id_v2,
                    "uid": uid,
                }

                for tier in tiers:
                    sp_key = f"T{tier}_{prompt_mode}"
                    if sp_key not in system_prompts:
                        system_prompts[sp_key] = build_system_prompt(
                            tier, mode=prompt_mode,
                        )

                    for variant in range(1, count + 1):
                        tested_concept_result = tested_concept_selector.run({
                            "concepts": ch_concepts,
                            "variant": variant, "tier": tier,
                        })

                        plan_result = distractor_planner.run({
                            "tier": tier, "variant": variant,
                            "misconceptions": ch_misconceptions,
                            "tested_concept_id": tested_concept_result.get("concept_id"),
                        })

                        flashcard_result = flashcard_template.run({
                            "tested_concept_label": (
                                tested_concept_result.get("concept_label", "this concept")
                                if tested_concept_result.get("has_tested_concept")
                                else "this concept"
                            ),
                            "misconceptions": ch_misconceptions,
                        })

                        pattern_name, _ = get_stem_pattern(tier, variant)
                        source_type = get_source_type(tier, pattern_name)
                        if not passage and source_type != "anchor_grounded":
                            source_type = "anchor_grounded"

                        pos_idx = (anchor_idx * len(tiers) * count + (tier - 1) * count + (variant - 1)) % len(CORRECT_POSITIONS)
                        target_position = CORRECT_POSITIONS[pos_idx]

                        model = "claude-opus-4-7"

                        metadata_result = metadata_agent.run({
                            "domain_code": domain_code,
                            "domain_id": domain_id,
                            "domain_name": domain_name,
                            "chapter_title": chapter_title,
                            "section_title": section_title,
                            "tier": tier,
                            "variant": variant,
                            "source_type": source_type,
                            "stem_pattern": pattern_name,
                            "anchor_uid": uid,
                            "anchor_id_v2": id_v2,
                            "verbatim_anchor": verbatim,
                            "testable_fact": testable_fact,
                            "batch_id": batch_id,
                            "content_chars": content_chars,
                        })

                        question_id = metadata_result["question_id"]

                        if question_id in existing_ids:
                            skipped += 1
                            continue

                        character = get_character_assignment(question_id)

                        user_prompt = build_user_prompt(
                            anchor_info, sub_content, anchor_data,
                            source_type, variant, domain_name,
                            difficulty_tier=tier,
                            concept_vocab=vocab_result if has_vocab else None,
                            character=character,
                            target_position=target_position,
                            tested_concept=tested_concept_result,
                            distractor_plan=plan_result,
                        )

                        tasks[question_id] = {
                            "system_prompt_key": sp_key,
                            "user_prompt": user_prompt,
                            "model": model,
                            "max_tokens": 2500,
                            "tier": tier,
                            "target_position": target_position,
                            "has_concept_vocab": has_vocab,
                            "distractor_plan": plan_result,
                            "meta_base": metadata_result["meta_base"],
                            "pre_tested_concept": tested_concept_result,
                            "flashcard_fronts": flashcard_result,
                            "topic_keywords": topic_keywords,
                            "output_domain": domain_code,
                            "output_chapter": chapter_slug,
                        }
                        domain_task_count += 1

        print(f"  [{domain_code}] {domain_task_count:,} tasks")

    # ── Save manifest ──────────────────────────────────────
    manifest = {
        "version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": {
            "tiers": tiers,
            "count": count,
            "domains": domains,
            "ultrathink": getattr(args, "ultrathink", False),
            "think_budget": getattr(args, "think_budget", DEFAULT_THINK_BUDGET),
        },
        "system_prompts": system_prompts,
        "tasks": tasks,
        "opus_batches": [],
        "phase": "prepared",
    }

    save_json(MANIFEST_FILE, manifest)

    num_batches = max(1, (len(tasks) + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE)

    # Cost estimate (batch = 50% of standard per-token pricing)
    # Opus batch: $7.50/$37.50 per MTok in/out; Sonnet batch: $1.50/$7.50
    ultrathink = getattr(args, "ultrathink", False)
    think_budget_cfg = getattr(args, "think_budget", DEFAULT_THINK_BUDGET)
    avg_in = 4000
    avg_out = 2500
    avg_think = think_budget_cfg * 0.7 if ultrathink else 0  # ~70% utilization
    opus_cost = len(tasks) * (avg_in * 7.5 + (avg_out + avg_think) * 37.5) / 1_000_000
    total = opus_cost
    sonnet_cost = 0
    print(f"\n{'='*60}")
    print(f"  Manifest: {MANIFEST_FILE}")
    print(f"  Tasks: {len(tasks):,}")
    print(f"  Skipped (resume): {skipped:,}")
    print(f"  System prompts: {len(system_prompts)} unique")
    print(f"  Opus batches needed: {num_batches}")
    if ultrathink:
        print(f"  Extended thinking: {think_budget_cfg:,} token budget")
    print(f"  Est. cost: ${total:,.0f}")
    print(f"\n  Next: python batch_generate.py submit")
    print(f"{'='*60}")


# ══════════════════════════════════════════════════════════════
# SUBMIT — Send tasks to Batch API
# ══════════════════════════════════════════════════════════════

def cmd_submit(args):
    """Submit tasks to the Anthropic Message Batches API."""
    if not MANIFEST_FILE.exists():
        print("ERROR: No manifest found. Run 'prepare' first.")
        sys.exit(1)

    manifest = load_json(MANIFEST_FILE)
    _submit_opus(manifest, args)


def _submit_opus(manifest, args):
    if manifest["phase"] != "prepared":
        if manifest["phase"] == "opus_submitted":
            print("Opus batches already submitted. Use 'status' to check or 'collect' when ready.")
        else:
            print(f"ERROR: Phase is '{manifest['phase']}', expected 'prepared'.")
        sys.exit(1)

    client = get_batch_client(args.api_key)
    tasks = manifest["tasks"]
    system_prompts = manifest["system_prompts"]

    # Build short_id mapping (custom_id max 64 chars)
    id_map = build_id_map(tasks.keys())  # short_id -> question_id
    reverse_map = {v: k for k, v in id_map.items()}  # question_id -> short_id

    # Extended thinking config
    ultrathink = manifest.get("config", {}).get("ultrathink", False)
    think_budget = manifest.get("config", {}).get("think_budget", DEFAULT_THINK_BUDGET)

    # Build request list
    requests = []
    for qid, task in tasks.items():
        params = {
            "model": task["model"],
            "max_tokens": task["max_tokens"],
            "system": system_prompts[task["system_prompt_key"]],
            "messages": [{"role": "user", "content": task["user_prompt"]}],
        }
        if ultrathink:
            params["thinking"] = {"type": "enabled", "budget_tokens": think_budget}
            params["temperature"] = 1  # required for extended thinking
            params["max_tokens"] = think_budget + task["max_tokens"]
        requests.append({"custom_id": reverse_map[qid], "params": params})

    num_batches = max(1, (len(requests) + MAX_BATCH_SIZE - 1) // MAX_BATCH_SIZE)
    print(f"\n  Submitting {len(requests):,} requests in {num_batches} batch(es)...\n")

    # Save ID mapping to manifest for collect
    manifest["opus_id_map"] = id_map

    # Resume: skip batches already submitted
    batches = manifest.get("opus_batches", [])
    already_submitted = sum(b["count"] for b in batches)
    remaining = requests[already_submitted:]

    for i in range(0, len(remaining), MAX_BATCH_SIZE):
        chunk = remaining[i:i + MAX_BATCH_SIZE]
        batch_num = len(batches) + 1
        print(f"  Batch {batch_num}: {len(chunk):,} requests...", end=" ", flush=True)

        try:
            batch = client.messages.batches.create(requests=chunk)
        except Exception as e:
            print(f"FAILED")
            print(f"    Error: {e}")
            print(f"    {len(batches)} batch(es) submitted so far. Re-run submit to resume.")
            manifest["opus_batches"] = batches
            if batches:
                manifest["phase"] = "opus_submitted"
            save_json(MANIFEST_FILE, manifest)
            sys.exit(1)

        batches.append({
            "batch_id": batch.id,
            "count": len(chunk),
            "submitted_at": datetime.now(timezone.utc).isoformat(),
        })
        print(f"OK  {batch.id}")

        # Save after each for resume safety
        manifest["opus_batches"] = batches
        manifest["phase"] = "opus_submitted"
        save_json(MANIFEST_FILE, manifest)

    print(f"\n  All {len(batches)} batch(es) submitted.")
    print(f"  Next: python batch_generate.py status")



# ══════════════════════════════════════════════════════════════
# STATUS — Check batch processing progress
# ══════════════════════════════════════════════════════════════

def cmd_status(args):
    """Check the status of submitted batches."""
    if not MANIFEST_FILE.exists():
        print("No manifest found. Run 'prepare' first.")
        return

    manifest = load_json(MANIFEST_FILE)
    phase = manifest["phase"]
    config = manifest["config"]

    print(f"\n  Phase: {phase}")
    print(f"  Tasks: {len(manifest['tasks']):,}")
    print(f"  Config: tiers={config['tiers']}, count={config['count']}")

    if phase == "prepared":
        print("\n  Ready to submit. Run: python batch_generate.py submit")
        return

    if phase == "complete":
        print("\n  All done.")
        return

    client = get_batch_client(args.api_key)

    def show_batches(label, batch_list):
        if not batch_list:
            return True
        print(f"\n  {label} ({len(batch_list)}):")
        all_done = True
        total_succeeded = 0
        total_errored = 0
        total_requests = 0
        for i, info in enumerate(batch_list):
            batch = client.messages.batches.retrieve(info["batch_id"])
            rc = batch.request_counts
            n = rc.processing + rc.succeeded + rc.errored + rc.canceled + rc.expired
            pct = (rc.succeeded / n * 100) if n > 0 else 0
            print(f"    [{i+1}] {info['batch_id']}")
            print(f"        {batch.processing_status}  {rc.succeeded}/{n} ({pct:.0f}%)  "
                  f"errors={rc.errored}  in-flight={rc.processing}")
            if batch.processing_status != "ended":
                all_done = False
            total_succeeded += rc.succeeded
            total_errored += rc.errored
            total_requests += n
        if len(batch_list) > 1:
            print(f"    Total: {total_succeeded}/{total_requests} succeeded, {total_errored} errors")
        return all_done

    opus_done = show_batches("Opus Batches", manifest.get("opus_batches", []))

    if phase == "opus_submitted" and opus_done:
        print(f"\n  Ready! Run: python batch_generate.py collect")


# ══════════════════════════════════════════════════════════════
# COLLECT — Download results, assemble, validate, save
# ══════════════════════════════════════════════════════════════

def cmd_collect(args):
    """Collect batch results, assemble questions, validate, and save."""
    if not MANIFEST_FILE.exists():
        print("ERROR: No manifest found.")
        sys.exit(1)

    manifest = load_json(MANIFEST_FILE)
    _collect_opus(manifest, args)


def _collect_opus(manifest, args):
    if manifest["phase"] != "opus_submitted":
        print(f"ERROR: Phase is '{manifest['phase']}', expected 'opus_submitted'.")
        sys.exit(1)

    client = get_batch_client(args.api_key)

    # Verify all batches are done
    for info in manifest["opus_batches"]:
        batch = client.messages.batches.retrieve(info["batch_id"])
        if batch.processing_status != "ended":
            rc = batch.request_counts
            print(f"  Batch {info['batch_id']}: {batch.processing_status} ({rc.processing} in-flight)")
            print(f"  Not ready yet. Use 'status' to check progress.")
            return

    print("  All Opus batches complete. Collecting results...\n")

    tasks = manifest["tasks"]
    assembler = QuestionAssemblerAgent()
    gates = create_gate_pipeline()

    # Reverse-map short IDs back to question IDs
    opus_id_map = manifest.get("opus_id_map", {})  # short_id -> question_id

    # ── Stream results from all batches ────────────────────
    results = {}     # qid -> creative dict
    api_metas = {}   # qid -> token counts
    errors = []

    for info in manifest["opus_batches"]:
        bid = info["batch_id"]
        print(f"  Streaming {bid}...", end=" ", flush=True)
        count = 0
        for result in client.messages.batches.results(bid):
            short_id = result.custom_id
            qid = opus_id_map.get(short_id, short_id)  # fallback to raw ID
            if result.result.type == "succeeded":
                msg = result.result.message
                text = extract_text_block(msg)

                # Strip markdown fences
                if text.startswith("```"):
                    text = re.sub(r"^```(?:json)?\s*", "", text)
                    text = re.sub(r"\s*```$", "", text)

                try:
                    creative = json.loads(text)
                    creative = fix_mojibake_deep(creative)
                    results[qid] = creative
                    api_metas[qid] = {
                        "prompt_tokens": msg.usage.input_tokens,
                        "completion_tokens": msg.usage.output_tokens,
                        "model_id": msg.model,
                    }
                    count += 1
                except json.JSONDecodeError as e:
                    errors.append({"question_id": qid, "error": f"json_parse: {e}"})
            else:
                errors.append({"question_id": qid, "error": f"batch_{result.result.type}"})
        print(f"{count} OK")

    print(f"\n  Total: {len(results):,} succeeded, {len(errors)} errors")

    # ── Assemble + validate ────────────────────────────────
    assembled_by_file = {}   # "DOMAIN/chapter-slug" -> [questions]
    stats = {"assembled": 0, "gate_fail": 0, "tokens_in": 0, "tokens_out": 0}

    for qid, creative in results.items():
        task = tasks.get(qid)
        if not task:
            errors.append({"question_id": qid, "error": "not in manifest"})
            continue

        meta = api_metas.get(qid, {})
        stats["tokens_in"] += meta.get("prompt_tokens", 0)
        stats["tokens_out"] += meta.get("completion_tokens", 0)

        gen_meta = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "prompt_tokens": meta.get("prompt_tokens", 0),
            "completion_tokens": meta.get("completion_tokens", 0),
            "retries": 0,
            "model_id": meta.get("model_id", task["model"]),
            "latency_ms": 0,
            "has_concept_vocab": task["has_concept_vocab"],
            "batch_mode": True,
        }

        # Phase 3: Assembly
        try:
            assembled = assembler.run({
                "creative": creative,
                "distractor_plan": task["distractor_plan"],
                "meta_base": task["meta_base"],
                "topic_keywords": task["topic_keywords"],
                "target_position": task["target_position"],
                "generation_metadata": gen_meta,
                "pre_tested_concept": task.get("pre_tested_concept"),
                "flashcard_fronts": task.get("flashcard_fronts"),
            })
        except Exception as e:
            errors.append({"question_id": qid, "error": f"assembly: {e}"})
            continue

        if "_error" in assembled:
            errors.append({"question_id": qid, "error": assembled["_error"]})
            continue

        # Phase 4: Validation gates (uniqueness checked at save time)
        skip_gates = {"uniqueness"}

        valid = True
        for gate in gates:
            if gate.name in skip_gates:
                continue
            ok, reason = gate.check(assembled, {})
            if not ok:
                valid = False
                errors.append({"question_id": qid, "error": f"gate.{gate.name}: {reason}"})
                stats["gate_fail"] += 1
                break

        if not valid:
            continue

        # Group by output file
        output_key = f"{task['output_domain']}/{task['output_chapter']}"
        if output_key not in assembled_by_file:
            assembled_by_file[output_key] = []
        assembled_by_file[output_key].append(assembled)
        stats["assembled"] += 1

    # ── Save to output files ───────────────────────────────
    saved_total = 0
    for output_key, questions in sorted(assembled_by_file.items()):
        domain, chapter = output_key.split("/", 1)
        path = QUIZ_DIR / domain / f"{chapter}.json"
        path.parent.mkdir(parents=True, exist_ok=True)

        existing = []
        existing_ids = set()
        if path.exists():
            try:
                existing = load_json(path)
                existing_ids = {q["question_id"] for q in existing}
            except (json.JSONDecodeError, OSError):
                pass

        new_qs = [q for q in questions if q["question_id"] not in existing_ids]
        if new_qs:
            existing.extend(new_qs)
            save_json(path, existing)
            saved_total += len(new_qs)
            print(f"    {domain}/{chapter}: +{len(new_qs)} (total {len(existing)})")

    # ── Update manifest ────────────────────────────────────
    manifest["phase"] = "complete"

    if errors:
        save_json(BATCH_DIR / "opus_errors.json", errors)

    save_json(MANIFEST_FILE, manifest)

    # ── Report ─────────────────────────────────────────────
    opus_cost = (stats["tokens_in"] * 7.5 + stats["tokens_out"] * 37.5) / 1_000_000

    print(f"\n{'='*60}")
    print(f"  Opus Collection Complete")
    print(f"  Assembled: {stats['assembled']:,}")
    print(f"  Saved to disk: {saved_total:,}")
    print(f"  Gate failures: {stats['gate_fail']}")
    print(f"  Other errors: {len(errors) - stats['gate_fail']}")
    print(f"  Tokens: {stats['tokens_in']:,} in + {stats['tokens_out']:,} out")
    print(f"  Batch cost: ${opus_cost:,.2f} (50% discount applied)")
    print(f"  Done!")
    print(f"{'='*60}")



# ══════════════════════════════════════════════════════════════
# CLI Entry Point
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Batch API adapter for quiz generation (50%% cost reduction)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Workflow:
  1. prepare --all          Run prep agents, build manifest
  2. submit                 Send Opus generation batch
  3. status                 Poll until complete (1-24h)
  4. collect                Download, assemble, validate, save
        """,
    )
    sub = parser.add_subparsers(dest="command")

    # prepare
    p = sub.add_parser("prepare", help="Run prep agents, build manifest")
    p.add_argument("--domain", type=str, help="Single domain code (e.g., BPSY)")
    p.add_argument("--all", action="store_true", help="All 9 domains")
    p.add_argument("--chapter", type=str, help="Single chapter slug")
    p.add_argument("--difficulty", type=int, nargs="+", default=[1, 2, 3],
                   help="Tiers to generate (default: 1 2 3)")
    p.add_argument("--count", type=int, default=5,
                   help="Variants per topic/tier (default: 5)")
    p.add_argument("--resume", action="store_true",
                   help="Skip questions already in output files")
    p.add_argument("--ultrathink", action="store_true",
                   help="Enable extended thinking for Opus generation")
    p.add_argument("--think-budget", type=int, default=DEFAULT_THINK_BUDGET,
                   help=f"Thinking token budget (default: {DEFAULT_THINK_BUDGET:,})")
    p.add_argument("--api-key", type=str)

    # submit
    s = sub.add_parser("submit", help="Submit to Batch API")
    s.add_argument("--api-key", type=str)

    # status
    st = sub.add_parser("status", help="Check batch processing status")
    st.add_argument("--api-key", type=str)

    # collect
    c = sub.add_parser("collect", help="Download results, assemble, save")
    c.add_argument("--api-key", type=str)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "prepare":
        if not args.domain and not args.all:
            print("ERROR: Specify --domain CODE or --all")
            sys.exit(1)
        cmd_prepare(args)
    elif args.command == "submit":
        cmd_submit(args)
    elif args.command == "status":
        cmd_status(args)
    elif args.command == "collect":
        cmd_collect(args)


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
