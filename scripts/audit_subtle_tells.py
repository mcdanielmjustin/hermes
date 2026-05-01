"""Audit subtle testwise tells beyond what existing gates catch.

Five dimensions per user request:
  1. Correct not substantially longer than distractors (within ±15% of median)
  2. "EXCEPT/NOT" stems: correct should be ONE plausibly-wrong-but-related claim,
     not the only odd-topic option in a sea of look-alike distractors
  3. Term-mismatch tell: stem mentions concept X, distractors only differ by
     containing concept Y (opposite/contrast term) — defeats by keyword match
  4. Synonym uniqueness: a content word appears ONLY in the correct option
     (not in stem or distractors) — student picks the synonym, not the answer
  5. Keyword concentration: topic keywords from the question's `topic_keywords`
     field cluster in the correct option

Reads questions from a single JSON file (the chapter file written by
generate_quiz_questions.py). Outputs per-question findings and aggregate
counts.
"""
import argparse
import json
import re
import statistics
import sys
from collections import Counter
from pathlib import Path

# Single source of truth for the stop-word set lives in pipeline/stopwords.py
# (BASE_AND_COGNITIVE = function words + cognitive verbs, no modifiers).
# Importing prevents drift between this auditor and the gates it informs.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline.stopwords import BASE_AND_COGNITIVE as STOP  # noqa: E402


def content_words(text):
    """Extract lowercase 4+ char content words, no stop-words."""
    if not text:
        return []
    return [w for w in re.findall(r"\b[a-zà-öø-ÿ]{4,}\b", (text or "").lower())
            if w not in STOP]


def jaccard(a, b):
    if not a or not b:
        return 0.0
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if sa | sb else 0.0


# ── Concern 1: Correct not substantially longer ──────────────────

def check_length_bias(q):
    options = q["options"]
    correct = next(o for o in options if o.get("is_correct"))
    distractors = [o for o in options if not o.get("is_correct")]
    cor_len = len(correct["text"])
    dist_lens = [len(o["text"]) for o in distractors]
    if not dist_lens:
        return None
    median_d = statistics.median(dist_lens)
    max_d = max(dist_lens)
    # Flags
    is_strict_longest = cor_len > max_d
    pct_over_median = (cor_len - median_d) / median_d * 100 if median_d else 0
    return {
        "cor_len": cor_len,
        "median_d": median_d,
        "max_d": max_d,
        "pct_over_median": round(pct_over_median, 1),
        "is_strict_longest": is_strict_longest,
        "flag": is_strict_longest or pct_over_median > 15,
    }


# ── Concern 2: EXCEPT/NOT odd-one-out tell ──────────────────────

EXCEPT_PATTERNS = re.compile(
    r"\b(except|not|all of the following|all of the above except|"
    r"are true except|EXCEPT|NOT)\b",
    re.IGNORECASE,
)


def check_except_oddness(q):
    stem = q.get("question_stem", "")
    if not EXCEPT_PATTERNS.search(stem):
        return None
    options = q["options"]
    correct = next(o for o in options if o.get("is_correct"))
    distractors = [o for o in options if not o.get("is_correct")]
    # Compute pairwise Jaccard similarity between distractors
    dwords = [content_words(d["text"]) for d in distractors]
    if len(dwords) < 2:
        return None
    sims = []
    for i in range(len(dwords)):
        for j in range(i + 1, len(dwords)):
            sims.append(jaccard(dwords[i], dwords[j]))
    mean_dist_sim = statistics.mean(sims) if sims else 0.0
    # Compute mean similarity of correct to distractors
    cwords = content_words(correct["text"])
    cor_sims = [jaccard(cwords, dw) for dw in dwords]
    mean_cor_sim = statistics.mean(cor_sims) if cor_sims else 0.0
    # Flag if distractors are very similar to each other but correct is far
    flag = mean_dist_sim - mean_cor_sim > 0.20 and mean_dist_sim > 0.30
    return {
        "stem_has_except": True,
        "mean_dist_to_dist": round(mean_dist_sim, 2),
        "mean_correct_to_dist": round(mean_cor_sim, 2),
        "gap": round(mean_dist_sim - mean_cor_sim, 2),
        "flag": flag,
    }


# ── Concern 3: Term-mismatch tell ────────────────────────────────

# Concept-pair antonyms common in EPPP/biopsych content. If the stem mentions
# one and distractors swap to the other, the answer is keyword-match.
ANTONYM_PAIRS = [
    ("agonist", "antagonist"),
    ("ipsilateral", "contralateral"),
    ("excitatory", "inhibitory"),
    ("activate", "block"),
    ("activate", "inhibit"),
    ("increase", "decrease"),
    ("upregulate", "downregulate"),
    ("agonism", "antagonism"),
    ("synthesis", "degradation"),
    ("afferent", "efferent"),
    ("sympathetic", "parasympathetic"),
    ("dorsal", "ventral"),
    ("anterior", "posterior"),
    ("medial", "lateral"),
    ("proximal", "distal"),
]


def check_term_mismatch(q):
    stem = q.get("question_stem", "").lower()
    options = q["options"]
    correct = next(o for o in options if o.get("is_correct"))
    distractors = [o for o in options if not o.get("is_correct")]
    findings = []
    for term_a, term_b in ANTONYM_PAIRS:
        # Case 1: stem has term_a; correct has term_a; ALL distractors have term_b
        if re.search(rf"\b{term_a}\b", stem):
            cor_has_a = bool(re.search(rf"\b{term_a}\b", correct["text"].lower()))
            dist_only_b = all(
                re.search(rf"\b{term_b}\b", d["text"].lower())
                and not re.search(rf"\b{term_a}\b", d["text"].lower())
                for d in distractors
            )
            if cor_has_a and dist_only_b:
                findings.append(f"stem+correct have '{term_a}'; ALL distractors have '{term_b}'")
        if re.search(rf"\b{term_b}\b", stem):
            cor_has_b = bool(re.search(rf"\b{term_b}\b", correct["text"].lower()))
            dist_only_a = all(
                re.search(rf"\b{term_a}\b", d["text"].lower())
                and not re.search(rf"\b{term_b}\b", d["text"].lower())
                for d in distractors
            )
            if cor_has_b and dist_only_a:
                findings.append(f"stem+correct have '{term_b}'; ALL distractors have '{term_a}'")
    return {"findings": findings, "flag": bool(findings)}


# ── Concern 4: Synonym uniqueness ────────────────────────────────

def check_synonym_uniqueness(q):
    """Words that appear only in correct (not in stem, not in any distractor)."""
    stem_words = set(content_words(q.get("question_stem", "")))
    options = q["options"]
    correct = next(o for o in options if o.get("is_correct"))
    distractors = [o for o in options if not o.get("is_correct")]
    correct_words = set(content_words(correct["text"]))
    distractor_words = set()
    for d in distractors:
        distractor_words.update(content_words(d["text"]))
    unique_to_correct = correct_words - stem_words - distractor_words
    # Filter to words that look like content/synonym candidates (4+ chars,
    # not too generic). Most short common words already filtered by STOP.
    unique = sorted(w for w in unique_to_correct if len(w) >= 5)
    flag = len(unique) >= 3
    return {
        "unique_words_in_correct": unique[:8],
        "count": len(unique),
        "flag": flag,
    }


# ── Concern 5: Keyword concentration in correct ──────────────────

def check_keyword_concentration(q):
    keywords = q.get("topic_keywords", []) or []
    if not keywords:
        return None
    options = q["options"]
    correct = next(o for o in options if o.get("is_correct"))
    distractors = [o for o in options if not o.get("is_correct")]
    cor_text = correct["text"].lower()
    dist_texts = [d["text"].lower() for d in distractors]
    cor_hits = sum(1 for kw in keywords if kw.lower() in cor_text)
    dist_hits_per = [
        sum(1 for kw in keywords if kw.lower() in dt) for dt in dist_texts
    ]
    avg_dist_hits = statistics.mean(dist_hits_per) if dist_hits_per else 0
    # Flag when correct has many keyword hits and distractors have ~none
    flag = cor_hits >= 2 and avg_dist_hits < 0.5
    return {
        "n_keywords": len(keywords),
        "cor_hits": cor_hits,
        "dist_hits_avg": round(avg_dist_hits, 1),
        "dist_hits_per": dist_hits_per,
        "flag": flag,
    }


# ── Concern 6: Stem keyword appears only in correct ──────────────

def check_stem_keyword_in_correct_only(q):
    """Stem keywords (4+ char content words from the stem) must be
    distributed — students shouldn't pick the option that contains the
    stem's key term while distractors avoid it.

    Skip EXCEPT-pattern questions (vocabulary divergence is structural).
    """
    stem = q.get("question_stem", "")
    if EXCEPT_PATTERNS.search(stem):
        return None
    options = q["options"]
    correct = next(o for o in options if o.get("is_correct"))
    distractors = [o for o in options if not o.get("is_correct")]

    stem_words = set(content_words(stem))
    if not stem_words:
        return None
    cor_text = correct["text"].lower()
    dist_texts_lower = [d["text"].lower() for d in distractors]

    # Stem keywords that appear in correct but NOT in any distractor
    in_correct_only = []
    for w in stem_words:
        if w in cor_text and not any(w in dt for dt in dist_texts_lower):
            in_correct_only.append(w)

    flag = len(in_correct_only) >= 2
    return {
        "stem_keywords_in_correct_only": sorted(in_correct_only)[:5],
        "count": len(in_correct_only),
        "flag": flag,
    }


# ── Concern 7: Option self-references "correct/wrong/incorrect" ───

_SELF_REFERENCE_RE = re.compile(
    r"\b(this is (correct|wrong|incorrect|the answer|right)|"
    r"the (correct|right|wrong) answer|"
    r"is the answer|"
    r"this option is)\b",
    re.IGNORECASE,
)


def check_option_self_reference(q):
    """Option text must not contain phrases that announce its own
    correctness (e.g., 'This is correct', 'is the answer'). These leak
    the answer key by appearing only in the correct option, or signal
    metacommentary that doesn't belong in answer claims.
    """
    options = q["options"]
    hits = []
    for o in options:
        text = o.get("text", "")
        if not text:
            continue
        for m in _SELF_REFERENCE_RE.finditer(text):
            role = "correct" if o.get("is_correct") else "distractor"
            hits.append(f"option {o['letter']} ({role}): {m.group(0)!r}")
    return {"hits": hits, "flag": bool(hits)}


# ── Concern 8: Option relevance to stem (via inter-option topic) ──

def check_option_relevance(q):
    """Each option should be on the same topic as its siblings — measured
    by shared content vocabulary. An option whose content words overlap
    with NONE of its siblings is off-topic and trivially eliminable
    without engaging the question.

    Stem-word overlap is too strict (penalizes natural paraphrasing where
    the option uses concrete vocabulary while the stem uses abstract
    framing). Inter-option overlap captures real off-topic outliers
    while accepting paraphrased relevance.

    Skip EXCEPT-pattern questions where vocabulary divergence is
    structural (the correct option IS the off-topic one by design).
    """
    stem = q.get("question_stem", "")
    if EXCEPT_PATTERNS.search(stem):
        return None
    options = q["options"]
    if len(options) < 2:
        return None

    opt_words = [content_words(o.get("text", "")) for o in options]
    irrelevant = []
    for i, o in enumerate(options):
        cw = set(opt_words[i])
        if not cw:
            continue  # empty option handled elsewhere
        # Check overlap with EVERY other option's content words
        overlaps_with_any = False
        for j, other_words in enumerate(opt_words):
            if i == j:
                continue
            if cw & set(other_words):
                overlaps_with_any = True
                break
        if not overlaps_with_any:
            role = "correct" if o.get("is_correct") else "distractor"
            irrelevant.append(f"option {o['letter']} ({role}, no sibling overlap)")
    flag = bool(irrelevant)
    return {"irrelevant": irrelevant, "flag": flag}


# ── Concern 9: Each option provides a full answer ───────────────

def check_option_fullness(q):
    """Each option should be a substantive claim, not a single word or
    fragment. Heuristic: at least 4 content words (4+ chars) in the
    option text. Length scaffold already enforces 50-100 chars, but
    this catches options that are technically long but content-poor
    (e.g., 'A type of pathological disorder of unknown origin' has
    ~10 words but only 'pathological / disorder / unknown / origin'
    are content words — passes; 'It does not happen here at all' is
    only 'happen / here' — fails).
    """
    options = q["options"]
    fragments = []
    for o in options:
        text = o.get("text", "")
        if not text:
            continue
        cw = content_words(text)
        if len(cw) < 4:
            role = "correct" if o.get("is_correct") else "distractor"
            fragments.append(
                f"option {o['letter']} ({role}, {len(cw)} content words): "
                f"{text[:60]}"
            )
    flag = bool(fragments)
    return {"fragments": fragments, "flag": flag}


# ── Main ──────────────────────────────────────────────────────────

def audit(qs):
    """Run all checks, return per-question findings + aggregate counts."""
    results = []
    aggregate = Counter()
    for q in qs:
        qid = q["question_id"]
        flags = []
        details = {}

        # 1. Length bias
        c1 = check_length_bias(q)
        if c1:
            details["length"] = c1
            if c1["flag"]:
                flags.append("length_bias")

        # 2. EXCEPT odd-one-out
        c2 = check_except_oddness(q)
        if c2:
            details["except"] = c2
            if c2["flag"]:
                flags.append("except_odd_one_out")

        # 3. Term-mismatch
        c3 = check_term_mismatch(q)
        if c3:
            details["term_mismatch"] = c3
            if c3["flag"]:
                flags.append("term_mismatch")

        # 4. Synonym uniqueness
        c4 = check_synonym_uniqueness(q)
        if c4:
            details["synonym"] = c4
            if c4["flag"]:
                flags.append("synonym_uniqueness")

        # 5. Keyword concentration
        c5 = check_keyword_concentration(q)
        if c5:
            details["keyword_conc"] = c5
            if c5["flag"]:
                flags.append("keyword_concentration")

        # 6. Stem keyword appears only in correct
        c6 = check_stem_keyword_in_correct_only(q)
        if c6:
            details["stem_keyword"] = c6
            if c6["flag"]:
                flags.append("stem_keyword_in_correct_only")

        # 7. Option self-reference
        c7 = check_option_self_reference(q)
        if c7:
            details["self_ref"] = c7
            if c7["flag"]:
                flags.append("option_self_reference")

        # 8. Option relevance
        c8 = check_option_relevance(q)
        if c8:
            details["relevance"] = c8
            if c8["flag"]:
                flags.append("irrelevant_option")

        # 9. Option fullness
        c9 = check_option_fullness(q)
        if c9:
            details["fullness"] = c9
            if c9["flag"]:
                flags.append("option_fragment")

        for f in flags:
            aggregate[f] += 1
        if flags:
            aggregate["any_flag"] += 1
        results.append({"qid": qid, "flags": flags, "details": details})

    return results, aggregate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("path", help="Path to chapter quiz JSON")
    p.add_argument("--verbose", action="store_true",
                   help="Print details for every question, not just flagged")
    args = p.parse_args()

    qs = json.loads(Path(args.path).read_text(encoding="utf-8"))
    print(f"=== AUDIT: {len(qs)} questions ===\n")

    results, aggregate = audit(qs)

    # Per-question flagged
    flagged = [r for r in results if r["flags"]]
    print(f"--- Flagged questions: {len(flagged)}/{len(results)} ---\n")
    for r in flagged if not args.verbose else results:
        qid_short = r["qid"].replace("QZ-BPSY-AP-", "").replace("D7-PHY-209-", "209-")
        flag_str = ", ".join(r["flags"]) if r["flags"] else "(no flags)"
        print(f"  {qid_short}: {flag_str}")
        d = r["details"]
        if "length" in d and d["length"]["flag"]:
            l = d["length"]
            tag = "STRICT-LONGEST" if l["is_strict_longest"] else "OVER-MEDIAN"
            print(f"    length: correct={l['cor_len']}, median_dist={l['median_d']}, "
                  f"+{l['pct_over_median']}% [{tag}]")
        if "except" in d and d["except"]["flag"]:
            e = d["except"]
            print(f"    except-odd: dist-pair-sim={e['mean_dist_to_dist']}, "
                  f"correct-to-dist-sim={e['mean_correct_to_dist']}, "
                  f"gap={e['gap']}")
        if "term_mismatch" in d and d["term_mismatch"]["flag"]:
            for f in d["term_mismatch"]["findings"]:
                print(f"    term-mismatch: {f}")
        if "synonym" in d and d["synonym"]["flag"]:
            s = d["synonym"]
            print(f"    synonym-only-in-correct ({s['count']}): {s['unique_words_in_correct']}")
        if "keyword_conc" in d and d["keyword_conc"]["flag"]:
            k = d["keyword_conc"]
            print(f"    keyword-conc: {k['cor_hits']} in correct, "
                  f"{k['dist_hits_avg']} avg in distractors (per-distractor: {k['dist_hits_per']})")
        if "stem_keyword" in d and d["stem_keyword"]["flag"]:
            sk = d["stem_keyword"]
            print(f"    stem-keyword-in-correct-only ({sk['count']}): "
                  f"{sk['stem_keywords_in_correct_only']}")
        if "self_ref" in d and d["self_ref"]["flag"]:
            for h in d["self_ref"]["hits"]:
                print(f"    self-reference: {h}")
        if "relevance" in d and d["relevance"]["flag"]:
            print(f"    irrelevant options: {d['relevance']['irrelevant']}")
        if "fullness" in d and d["fullness"]["flag"]:
            for f in d["fullness"]["fragments"]:
                print(f"    fragment: {f}")

    # Aggregate
    print(f"\n--- Aggregate counts ---")
    print(f"  Length-bias flag:               {aggregate['length_bias']}/{len(results)}")
    print(f"  EXCEPT odd-one-out:              {aggregate['except_odd_one_out']}/{len(results)}")
    print(f"  Term-mismatch tell:              {aggregate['term_mismatch']}/{len(results)}")
    print(f"  Synonym uniqueness tell:         {aggregate['synonym_uniqueness']}/{len(results)}")
    print(f"  Keyword concentration:           {aggregate['keyword_concentration']}/{len(results)}")
    print(f"  Stem-keyword-in-correct-only:    {aggregate['stem_keyword_in_correct_only']}/{len(results)}")
    print(f"  Option self-reference:           {aggregate['option_self_reference']}/{len(results)}")
    print(f"  Irrelevant options:              {aggregate['irrelevant_option']}/{len(results)}")
    print(f"  Option fragments (full answer):  {aggregate['option_fragment']}/{len(results)}")
    print(f"  Any flag:                        {aggregate['any_flag']}/{len(results)}")


if __name__ == "__main__":
    main()
