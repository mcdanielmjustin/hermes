"""
generate_anchor_briefs.py

Pre-generates per-anchor analysis briefs that guide question generation.
Each brief contains core claims, anchor-specific misconceptions, related
concepts, and question angles — replacing chapter-level concept vocab
as the primary source of intelligence for the generation pipeline.

Run BEFORE generate_quiz_questions.py.

Data sources:
  - anchor_points.csv       (all 1,566 anchors)
  - anchor_passages_v3.csv  (textbook passages for 1,081 anchors)
  - concept_vocab/          (chapter-level vocab, used as seed context)

Usage:
  python generate_anchor_briefs.py --domain BPSY
  python generate_anchor_briefs.py --domain BPSY --anchor D7-PHY-021-b323a513
  python generate_anchor_briefs.py --all
  python generate_anchor_briefs.py --all --resume
  python generate_anchor_briefs.py --all --dry-run

Output:
  data/anchor_briefs/{DOMAIN_CODE}/{uid}.json
"""

import json, pathlib, argparse, time, sys, os, re, csv
from collections import defaultdict
import anthropic

# ── Paths ─────────────────────────────────────────────────────
SCRIPT_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
from config import (
    ANCHOR_BRIEFS_DIR, CONCEPT_VOCAB_DIR, DATA_DIR,
    ANCHOR_POINTS_CSV, ANCHOR_PASSAGES_CSV,
)

from shared_constants import DOMAIN_CODES, CODE_TO_ID, DOMAIN_NAMES
from pipeline.concept_registry import ConceptRegistry, canonicalize_brief
from pipeline.brief_grounding import validate_brief_grounding
from pipeline.brief_pool_adequacy import validate_pool_adequacy

MODEL_ID = "claude-opus-4-7"

SYSTEM_PROMPT = """You are building an anchor analysis brief for the PassEPPP EPPP exam preparation platform.

Your task: for a given anchor point (a specific testable fact from the EPPP exam), produce a structured brief that will guide quiz question generation. This brief ensures every question generated for this anchor addresses the anchor's actual content.

## Input
You will receive:
- The anchor's verbatim text and testable fact
- A textbook passage (if available)
- The chapter's concept vocabulary (concepts and misconceptions) for context

## Output Requirements

### core_claims (3-6 claims)
The essential factual claims this anchor makes. Every question generated from this anchor MUST address at least one of these claims. Be specific and testable.

### concepts (3-5 concepts)
Concepts directly relevant to THIS anchor (not the whole chapter). Format: concept_id (kebab-case), label, description. Include the primary concept being tested and 2-4 closely related concepts that could serve as distractors.

### misconceptions (5-8 misconceptions)
Student confusions specific to THIS anchor's content. These must be confusions students actually make about the claims in this anchor — not generic chapter-level confusions.
- Format: misconception_id (kebab-case), label, type, concepts_involved
- misconception_type values: similar_name, similar_property, similar_store, opposite_direction, overgeneralization, partial_understanding

CRITICAL — type ↔ concepts_involved consistency:

The most common author error is mismatching the misconception's type with the number of concepts listed. Apply this rule strictly:

* similar_name / similar_property / similar_store → MUST have 2+ concepts_involved.
  These types MEAN "concept X confused with concept Y" — they are inherently comparative. If your label includes phrases like "Confusing X with Y", "Misattributing X to Y", "X vs Y", "Conflating X and Y", or "Mistaking X for Y", then concepts_involved MUST list BOTH the X concept and the Y concept.
  misconception_id should follow `concept-a-vs-concept-b` format.
  Examples of CORRECT pairings:
    label: "Confusing the Realistic and Investigative types"
    type: similar_property
    concepts_involved: ["holland-realistic-type", "holland-investigative-type"]   ← both listed

    label: "Misattributing actor-observer effect to Jones and Davis (correspondent inference) rather than Jones and Nisbett"
    type: similar_name
    concepts_involved: ["actor-observer-effect", "correspondent-inference"]   ← BOTH listed
  Common BUG to avoid: writing similar_property with only 1 concept_id. If you can't list 2 concepts, your TYPE is wrong — switch to partial_understanding or overgeneralization.

* opposite_direction / overgeneralization / partial_understanding → 1+ concepts_involved allowed.
  These describe a misconception about ONE concept's properties, scope, or direction. The student misunderstands a single concept (its scope, its direction, or some aspect of it).
  Examples:
    label: "Believing Klinefelter syndrome has normal fertility"
    type: partial_understanding
    concepts_involved: ["klinefelter-syndrome"]   ← single concept, valid

    label: "Believing the actor-observer effect applies only to negative outcomes"
    type: overgeneralization
    concepts_involved: ["actor-observer-effect"]   ← single concept, valid

Final self-check before returning JSON: for every misconception, if type starts with "similar_", concepts_involved.length must be >= 2. If you have an "X-vs-Y" misconception_id but only one concept listed, you have a bug — fix it.

### question_angles (4-6 angles)
Different ways to write questions about this anchor. Each angle should test the anchor from a different perspective. Include the angle type and a brief description.
Angle types: definitional, clinical_application, neuroanatomical, comparison, mechanism, exception

### concept_explanation (one paragraph, ~3-6 sentences)
The IRREDUCIBLE IDEA the question must force the student to actually run — the mechanism / framework / criterion / process that, if the student doesn't have it, they cannot reach the correct answer. This is distinct from `testable_fact` (a declarative claim) and `tested_concept` (a concept identifier). It captures the cognitive operation the student must perform.

Examples (one per flavor):
- mechanism: "An antagonist has zero intrinsic activity at its receptor — it binds without producing a biological effect on its own. Its observable effect appears only as a reduction in the action of an agonist or endogenous neurotransmitter, achieved through receptor blockade, synthesis inhibition, or release inhibition."
- framework: "APA Standard 9.09 governs psychologists' use of automated scoring services: they retain professional responsibility for the appropriate selection, interpretation, and use of test scoring services, regardless of the service's automation level."
- criterion: "DSM-5 schizoaffective disorder requires (a) a major mood episode (manic or depressive) concurrent with Criterion A psychotic symptoms, AND (b) at least 2 weeks of psychotic symptoms in the absence of a major mood episode during the lifetime course."
- statistical: "The standard error of the mean (SEM) is the standard deviation of the sampling distribution of the mean. SEM = σ/√n, so SEM increases when σ (population variability) increases and decreases when n (sample size) increases."

A good `concept_explanation`:
- Names the named entities the student must know (drugs, standards, criteria, statistics).
- States the conditional logic that makes the concept testable from multiple angles.
- Does NOT just paraphrase `testable_fact` — that's a different field for a different role.

### discriminators (3-5 cognitive dimensions)
Snake_case labels for the variables a question could ask the student to RESOLVE. Each discriminator names a SINGLE axis on which the concept can be probed; templates for stem-rewrite use these to know what to OMIT from the stem (so the student must invoke the concept to choose).

Examples:
- mechanism flavor: ["intrinsic_activity_yes_vs_no", "locus_pre_vs_post", "direction_of_change", "pathway_specificity"]
- framework flavor: ["scope_of_authority", "exception_recognition", "principle_priority", "responsibility_attribution"]
- criterion flavor: ["symptom_count_threshold", "duration_requirement", "exclusion_criterion", "concurrent_vs_sequential"]
- statistical flavor: ["assumption_violation", "design_role_attribution", "effect_size_magnitude", "sample_size_dependence"]

Choose 3-5 discriminators per anchor that span different cognitive operations. Repeating the same axis under different names is wasteful; missing a major axis (e.g., direction-of-change for a mechanism concept) is a gap.

## Output Format
Return ONLY valid JSON (no markdown, no explanation):
{
  "core_claims": [
    "Specific testable claim from this anchor"
  ],
  "concepts": [
    {
      "concept_id": "kebab-case-id",
      "label": "Human-readable label",
      "description": "One sentence"
    }
  ],
  "misconceptions": [
    {
      "misconception_id": "concept-a-vs-concept-b",
      "label": "Human-readable confusion label",
      "type": "similar_name|similar_property|similar_store|opposite_direction|overgeneralization|partial_understanding",
      "concepts_involved": ["concept-id-1", "concept-id-2"]
    }
  ],
  "question_angles": [
    {
      "type": "clinical_application|definitional|neuroanatomical|comparison|mechanism|exception",
      "description": "Brief description of this question angle"
    }
  ],
  "concept_explanation": "The irreducible idea — one paragraph (3-6 sentences) capturing the mechanism / framework / criterion / process the question must force the student to run.",
  "discriminators": ["snake_case_axis_1", "snake_case_axis_2", "snake_case_axis_3"]
}"""


REVIEW_SYSTEM_PROMPT = """You are reviewing an anchor analysis brief for the PassEPPP EPPP exam preparation platform. Your job is to spot LLM hallucinations and revise the brief so every concept, misconception, claim, and angle is tightly grounded in the source material.

You will receive:
- The anchor's source material (verbatim text + textbook passage)
- A draft brief that was generated by another LLM call

## Review checklist

1. **Concept grounding** — Each concept's description must correspond to content in the source material. Concepts that drift from this anchor's actual content (generic chapter knowledge that the passage doesn't support, fabricated mechanisms, subtopics outside the anchor's scope) should be replaced with grounded concepts derived from the source.

2. **Misconception plausibility** — Each misconception must be a confusion students plausibly make about THIS anchor's content. Generic chapter-level confusions or invented errors should be replaced with anchor-specific misconceptions. Each misconception must reference at least 1 concept from the concepts list (2+ preferred; 1 allowed only for opposite_direction type).

3. **Core claims essentialism** — Each core_claim must be a testable claim the anchor actually makes — not peripheral details from the passage and not generalizations beyond what the anchor states. Drop or rewrite weak claims.

4. **Question angle relevance** — Each angle should test something the anchor actually testifies. Drop irrelevant or generic angles.

5. **Internal consistency** — Misconceptions must reference concepts that exist in the concepts list. concept_ids and misconception_ids must be kebab-case strings.

6. **concept_explanation precision** — The concept_explanation must capture the IRREDUCIBLE idea the student must invoke to answer correctly. Verify it: (a) is distinct from `testable_fact` (which is the declarative surface claim — the explanation is the underlying mechanism/framework/criterion/process), (b) names the named entities the student must know, (c) states the conditional logic that makes the concept testable from multiple angles. Drop or revise if it just paraphrases the fact.

7. **discriminators meaningfulness** — Each discriminator must name a SINGLE cognitive axis the concept can be probed along. Verify: (a) labels are snake_case, (b) each axis is genuinely distinct (no redundancy), (c) axes cover the major cognitive operations the concept supports (e.g., for a mechanism concept, both direction and locus are typical; for a criterion concept, both threshold and exclusion are typical). Drop redundant axes; add missing major axes.

## Decision

If the brief is already correct: return it unchanged.
If revisions are needed: return the revised brief in the SAME JSON shape.

Preserve the original brief's structure exactly. Do not add or remove top-level keys. Do not add commentary outside the JSON.

## Output format

Return ONLY valid JSON matching this shape:
{
  "core_claims": [...],
  "concepts": [...],
  "misconceptions": [...],
  "question_angles": [...],
  "concept_explanation": "...",
  "discriminators": [...]
}"""


def build_review_prompt(brief, verbatim, testable, passage_text):
    """Build user prompt for the review pass."""
    if passage_text:
        passage_section = f"\n## Textbook Passage\n{passage_text}\n"
    else:
        passage_section = "\n(No textbook passage available for this anchor.)\n"

    brief_for_review = {
        "core_claims": brief.get("core_claims", []),
        "concepts": brief.get("concepts", []),
        "misconceptions": brief.get("misconceptions", []),
        "question_angles": brief.get("question_angles", []),
        # Phase 20c additions: review pass also evaluates these fields.
        "concept_explanation": brief.get("concept_explanation", ""),
        "discriminators": brief.get("discriminators", []),
    }

    return f"""Review this anchor brief.

## Source Material

Verbatim: {verbatim}
Testable Fact: {testable}
{passage_section}
## Draft Brief
{json.dumps(brief_for_review, indent=2, ensure_ascii=False)}

Apply the review checklist. Return ONLY the JSON brief (revised if changes are warranted, unchanged if already correct)."""


def review_brief(client, brief, verbatim, testable, passage_text, max_retries=2):
    """Run the second-pass review/revise on a structurally valid brief.

    Returns (reviewed_brief, tokens). On failure returns (brief, 0) so the
    pipeline keeps the original draft rather than dropping the anchor.
    """
    user_prompt = build_review_prompt(brief, verbatim, testable, passage_text)

    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(
                model=MODEL_ID,
                # 4000 (vs prior 2500) — Phase 20c added concept_explanation
                # (~500-1000 chars) and discriminators (~100-200 chars), plus
                # the brief content may be expanded by the review pass. The
                # 2500 limit was empirically truncating complex briefs (DSM-5
                # schizoaffective hit JSON parse failure at ~char 6800).
                max_tokens=4000,
                system=REVIEW_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = re.sub(r"^```(?:json)?\s*", "", text)
                text = re.sub(r"\s*```$", "", text)
            reviewed = json.loads(text)
            tokens = response.usage.input_tokens + response.usage.output_tokens
            return reviewed, tokens
        except json.JSONDecodeError as e:
            if attempt < max_retries:
                print(f"    review JSON parse error, retrying ({attempt + 1}/{max_retries})...")
                time.sleep(2)
            else:
                print(f"    review JSON unparseable: {e} — keeping draft")
                return brief, 0
        except anthropic.APIError as e:
            if attempt < max_retries:
                wait = 10 * (attempt + 1)
                print(f"    review API error: {e}. Waiting {wait}s...")
                time.sleep(wait)
            else:
                print(f"    review API failed: {e} — keeping draft")
                return brief, 0


def briefs_meaningfully_differ(original, reviewed):
    """Return True if the review produced material changes worth logging."""
    keys = ("core_claims", "concepts", "misconceptions", "question_angles",
            "concept_explanation", "discriminators")
    for k in keys:
        if json.dumps(original.get(k, []), sort_keys=True) != \
           json.dumps(reviewed.get(k, []), sort_keys=True):
            return True
    return False


# ── Data loading ──────────────────────────────────────────────

def load_anchors():
    anchors = {}
    with open(ANCHOR_POINTS_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            anchors[row["uid"]] = row
    return anchors


def load_passages():
    passages = {}
    with open(ANCHOR_PASSAGES_CSV, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            passages[row["uid"]] = row
    return passages


def load_chapter_vocab(domain_code, chapter_id):
    vocab_path = CONCEPT_VOCAB_DIR / domain_code / f"{chapter_id}.json"
    if not vocab_path.exists():
        return None
    try:
        with open(vocab_path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


# ── Helpers ───────────────────────────────────────────────────

def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


CHECKPOINT_FILE = pathlib.Path(__file__).parent / "anchor_briefs_checkpoint.json"


def load_checkpoint():
    if CHECKPOINT_FILE.exists():
        return set(load_json(CHECKPOINT_FILE))
    return set()


def save_checkpoint(done_keys):
    save_json(CHECKPOINT_FILE, sorted(done_keys))


def build_user_prompt(anchor, passage_data, chapter_vocab):
    uid = anchor["uid"]
    verbatim = anchor.get("verbatim_anchor", "")
    testable = anchor.get("testable_fact", "")
    chapter_title = anchor.get("chapter_title", "")
    domain_name = anchor.get("domain_name", "")

    passage = ""
    if passage_data:
        passage = passage_data.get("passage", "")

    passage_section = ""
    if passage:
        snippet = passage[:2000] if len(passage) <= 2000 else passage[:1000] + "\n\n[...]\n\n" + passage[-800:]
        passage_section = f"""
## Textbook Passage (context for this anchor):
{snippet}
"""

    vocab_section = ""
    if chapter_vocab:
        concepts = chapter_vocab.get("concepts", [])
        misconceptions = chapter_vocab.get("misconceptions", [])
        if concepts or misconceptions:
            lines = ["\n## Chapter-Level Concept Vocabulary (for context — your output should be MORE SPECIFIC to this anchor):"]
            if concepts:
                lines.append("Concepts: " + ", ".join(f"`{c['concept_id']}`" for c in concepts))
            if misconceptions:
                lines.append("Misconceptions: " + ", ".join(f"`{m['misconception_id']}`" for m in misconceptions))
            vocab_section = "\n".join(lines)

    return f"""Analyze this anchor point and generate an anchor brief.

DOMAIN: {domain_name}
CHAPTER: {chapter_title}
ANCHOR UID: {uid}

## Anchor Point:
Verbatim: {verbatim}
Testable Fact: {testable}
{passage_section}{vocab_section}

Generate the anchor brief now. Return ONLY the JSON object."""


def generate_brief(client, user_prompt, max_retries=2):
    for attempt in range(max_retries + 1):
        try:
            response = client.messages.create(
                model=MODEL_ID,
                # 4000 — see review-pass note on the same parameter
                max_tokens=4000,
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


def validate_brief(brief):
    if not brief or not isinstance(brief, dict):
        return False, "not a dict"

    claims = brief.get("core_claims", [])
    if not claims or len(claims) < 2:
        return False, f"need 2+ core_claims, got {len(claims)}"

    concepts = brief.get("concepts", [])
    if not concepts or len(concepts) < 2:
        return False, f"need 2+ concepts, got {len(concepts)}"
    for c in concepts:
        if not c.get("concept_id") or not c.get("label"):
            return False, f"concept missing id or label: {c}"

    misconceptions = brief.get("misconceptions", [])
    if not misconceptions or len(misconceptions) < 3:
        return False, f"need 3+ misconceptions, got {len(misconceptions)}"
    # Misconception types split into two semantic categories:
    #   • Inherently-comparative (need 2+ concepts): similar_name, similar_property,
    #     similar_store. These are "X confused with Y" patterns; a misconception
    #     of this type with only 1 concept is structurally incomplete.
    #   • Single-concept-permitted (1+ allowed): opposite_direction,
    #     overgeneralization, partial_understanding. These can legitimately
    #     describe a misconception about ONE concept's properties (e.g.,
    #     "believing Klinefelter syndrome has normal fertility" is about
    #     Klinefelter alone — partial_understanding of one concept).
    SINGLE_CONCEPT_OK_TYPES = {
        "opposite_direction", "overgeneralization", "partial_understanding",
    }
    for m in misconceptions:
        if not m.get("misconception_id") or not m.get("type"):
            return False, f"misconception missing id or type: {m}"
        min_concepts = 1 if m.get("type") in SINGLE_CONCEPT_OK_TYPES else 2
        if not m.get("concepts_involved") or len(m["concepts_involved"]) < min_concepts:
            return False, f"misconception needs {min_concepts}+ concepts_involved: {m}"

    angles = brief.get("question_angles", [])
    if not angles or len(angles) < 3:
        return False, f"need 3+ question_angles, got {len(angles)}"

    return True, "ok"


# ── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Generate per-anchor analysis briefs for quiz question generation"
    )
    parser.add_argument("--domain", type=str, help="Domain code (e.g., BPSY)")
    parser.add_argument("--all", action="store_true", help="Process all 9 domains")
    parser.add_argument("--anchor", type=str, help="Single anchor UID")
    parser.add_argument("--chapter", type=str, help="Filter to chapter (e.g., D7-Ch11)")
    parser.add_argument("--resume", action="store_true", help="Skip already-generated anchors")
    parser.add_argument("--dry-run", action="store_true", help="Preview without API calls")
    parser.add_argument("--api-key", type=str, help="Override API key")
    parser.add_argument("--no-review", action="store_true",
                        help="Skip the second-pass review (faster + cheaper, less rigorous)")
    args = parser.parse_args()

    if not args.domain and not args.all and not args.anchor:
        parser.error("Specify --domain CODE, --all, or --anchor UID")

    # Load data
    print("Loading anchor_points.csv...")
    all_anchors = load_anchors()
    print(f"  {len(all_anchors)} anchors loaded")

    print("Loading anchor_passages_v3.csv...")
    passages = load_passages()
    print(f"  {len(passages)} passages loaded")

    # Concept registry — keeps concept_ids stable across briefs so the
    # question pipeline can rely on cross-anchor concept reuse.
    registry_path = DATA_DIR / "concept_registry.json"
    registry = ConceptRegistry(registry_path)
    print(f"Loading concept registry from {registry_path.name}...")
    print(f"  {registry.stats()['total_concepts']} canonical concepts, "
          f"{registry.stats()['total_aliases']} aliases")

    # Single anchor mode
    if args.anchor:
        if args.anchor not in all_anchors:
            print(f"ERROR: Anchor {args.anchor} not found.")
            sys.exit(1)
        anchor = all_anchors[args.anchor]
        target_uids = {args.anchor}
        domain_code = anchor.get("domain_code", "")
        if not domain_code:
            # Derive from domain_num
            domain_num = anchor.get("domain_num", "")
            domain_code = DOMAIN_CODES.get(int(domain_num), "") if domain_num else ""
        target_domains = {domain_code} if domain_code else set(DOMAIN_CODES.values())
    else:
        target_uids = None
        if args.all:
            target_domains = set(DOMAIN_CODES.values())
        else:
            code = args.domain.upper()
            if code not in CODE_TO_ID:
                parser.error(f"Unknown domain: {code}")
            target_domains = {code}

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

    print(f"\nAnchor Brief Generator")
    print(f"  Model: {MODEL_ID}")
    print(f"  Domains: {', '.join(sorted(target_domains))}")
    print(f"  Resume: {args.resume}")
    print()

    # Group anchors by domain
    by_domain = defaultdict(list)
    for uid, anchor in all_anchors.items():
        domain_num = anchor.get("domain_num", "")
        domain_code = DOMAIN_CODES.get(int(domain_num), "") if domain_num else ""
        if domain_code in target_domains:
            if target_uids and uid not in target_uids:
                continue
            if args.chapter and anchor.get("chapter_num", "") != args.chapter:
                continue
            anchor["domain_code"] = domain_code
            by_domain[domain_code].append(anchor)

    for domain_code in sorted(by_domain.keys()):
        domain_name = DOMAIN_NAMES[domain_code]
        domain_anchors = sorted(by_domain[domain_code], key=lambda a: a["uid"])

        print(f"\n{'='*60}")
        print(f"  {domain_code} — {domain_name} ({len(domain_anchors)} anchors)")
        print(f"{'='*60}")

        for anchor in domain_anchors:
            uid = anchor["uid"]
            chapter_id = anchor.get("chapter_num", "")

            if uid in done_keys:
                total_skipped += 1
                continue

            brief_path = ANCHOR_BRIEFS_DIR / domain_code / f"{uid}.json"
            if brief_path.exists() and args.resume:
                done_keys.add(uid)
                total_skipped += 1
                continue

            if args.dry_run:
                has_passage = uid in passages
                print(f"  [DRY-RUN] {uid} | ch={chapter_id} | passage={'yes' if has_passage else 'no'}")
                total_skipped += 1
                continue

            passage_data = passages.get(uid)
            chapter_vocab = load_chapter_vocab(domain_code, chapter_id) if chapter_id else None

            user_prompt = build_user_prompt(anchor, passage_data, chapter_vocab)

            print(f"  {uid}...", end=" ", flush=True)
            brief, tokens = generate_brief(client, user_prompt)

            total_tokens += tokens

            if brief is None:
                print("FAILED")
                total_failed += 1
                continue

            valid, reason = validate_brief(brief)
            if not valid:
                print(f"INVALID ({reason})")
                total_failed += 1
                continue

            # Second-pass review unless explicitly skipped. The reviewer
            # critiques the draft against source material and returns either
            # the same brief (if good) or a revised version. Failures fall
            # back to the draft rather than dropping the anchor.
            review_changed = False
            if not args.no_review:
                verbatim = anchor.get("verbatim_anchor", "")
                testable = anchor.get("testable_fact", "")
                passage_text = passages.get(uid, {}).get("passage", "")
                reviewed, review_tokens = review_brief(
                    client, brief, verbatim, testable, passage_text,
                )
                total_tokens += review_tokens
                if reviewed and reviewed is not brief:
                    # Re-validate structure — review could have malformed it.
                    rvalid, rreason = validate_brief(reviewed)
                    if rvalid:
                        review_changed = briefs_meaningfully_differ(brief, reviewed)
                        brief = reviewed
                    else:
                        print(f"    review produced invalid structure ({rreason}) "
                              f"— keeping draft")

            output = {
                "uid": uid,
                "anchor_point_id_v2": anchor.get("anchor_point_id_v2", ""),
                "domain_code": domain_code,
                "domain_name": domain_name,
                "chapter_num": chapter_id,
                "chapter_title": anchor.get("chapter_title", ""),
                "verbatim_anchor": anchor.get("verbatim_anchor", ""),
                "testable_fact": anchor.get("testable_fact", ""),
                "has_passage": uid in passages,
                "core_claims": brief["core_claims"],
                "concepts": brief["concepts"],
                "misconceptions": brief["misconceptions"],
                "question_angles": brief["question_angles"],
                # Phase 20c additions: structured fields for stem-rewrite
                # templates. Defaults to "" / [] if the LLM omitted them
                # (older brief outputs predate these fields).
                "concept_explanation": brief.get("concept_explanation", ""),
                "discriminators": brief.get("discriminators", []),
            }

            # Canonicalize concept_ids against the cross-brief registry.
            # Modifies output in-place: concepts get canonical IDs, and any
            # misconception's concepts_involved list is remapped to match.
            n_new, n_aliased = canonicalize_brief(output, registry)

            # Grounding check — concept descriptions should have keyword
            # overlap with the anchor's source text. Issues are warnings,
            # not failures: legitimate abstraction can produce concept
            # labels not literally present in the source.
            source_text = (
                output.get("verbatim_anchor", "") + " "
                + passages.get(uid, {}).get("passage", "")
            )
            grounding_issues = validate_brief_grounding(output, source_text)

            # Pool adequacy check — does the misconception pool meet the
            # DistractorPlannerAgent's contract for 20-question rotation?
            # Soft warnings only; doesn't block save.
            pool_issues = validate_pool_adequacy(output)

            save_json(brief_path, output)

            done_keys.add(uid)
            save_checkpoint(done_keys)

            nc = len(brief["concepts"])
            nm = len(brief["misconceptions"])
            na = len(brief["question_angles"])
            ncl = len(brief["core_claims"])
            registry_note = f", registry: +{n_new} new, {n_aliased} aliased" if (n_new or n_aliased) else ""
            review_note = " [reviewed+revised]" if review_changed else ""
            print(f"OK ({ncl} claims, {nc}C, {nm}M, {na} angles{registry_note}){review_note}")
            for issue in grounding_issues:
                print(f"    ⚠ grounding: {issue['concept_id']} "
                      f"coverage={issue['coverage']:.2f} "
                      f"missing={issue['missing_terms']}")
            for issue in pool_issues:
                ctx = ""
                if "concept_id" in issue:
                    ctx = f" [{issue['concept_id']}]"
                elif "concepts" in issue:
                    ctx = f" {issue['concepts']}"
                print(f"    ⚠ pool[{issue['type']}]{ctx}: {issue['detail']}")
            total_generated += 1

            time.sleep(1.0)

    # Persist registry so subsequent brief runs see all concepts produced
    # so far. Saving even when 0 generated is fine — appearance counts on
    # existing concepts may have been updated.
    registry.save()

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Generated: {total_generated} anchor briefs")
    print(f"  Skipped: {total_skipped}")
    print(f"  Failed: {total_failed}")
    print(f"  Total tokens: {total_tokens:,}")
    if total_tokens:
        cost = total_tokens * 45 / 1_000_000
        print(f"  Estimated cost: ${cost:.2f}")
    final_stats = registry.stats()
    print(f"  Registry: {final_stats['total_concepts']} canonical concepts, "
          f"{final_stats['total_aliases']} aliases")
    print(f"{'='*60}")


if __name__ == "__main__":
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8")
    main()
