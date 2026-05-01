"""Audit generated questions for the three new quality parameters.

  1. Researcher attribution leakage (e.g., "Squire (2004)", "according to X")
  2. Answer-length balance (correct answer not the longest by a wide margin)
  3. Elaboration tells (parens/semicolons/etc. clustering only on correct answer)

Usage:
  python audit_question_quality.py <path-to-quiz.json>
"""
import argparse
import json
import pathlib
import re
import statistics
import sys

# Use the canonical citation patterns and whitelist from the pipeline
# so audit results agree with what the AttributionGate actually flags.
# Earlier the audit had its own copies that drifted out of sync.
SCRIPT_DIR = pathlib.Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR.parent))
sys.path.insert(0, str(SCRIPT_DIR))
from pipeline.citation_patterns import find_attributions  # noqa: E402

ELAB_PATTERNS = {
    "parens":     re.compile(r"\([^)]+\)"),
    "semicolons": re.compile(r";"),
    "em_dashes":  re.compile(r"—|--"),
    "ie_eg":      re.compile(r"\b(?:i\.?e\.?|e\.?g\.?)\b", re.IGNORECASE),
}


def count_elab(text: str) -> dict:
    return {k: len(p.findall(text)) for k, p in ELAB_PATTERNS.items()}


def total_elab(text: str) -> int:
    return sum(count_elab(text).values())


def audit(qs):
    n = len(qs)
    print(f"=== AUDIT: {n} questions ===\n")

    # ---- Rule 1: Researcher attribution ----
    print("--- Rule 1: Researcher Attribution ---")
    violations, whitelisted = [], []
    for q in qs:
        sources = {
            "stem": q.get("question_stem", ""),
            "tested_kn": q.get("tested_concept", {}).get("knowledge_tested", ""),
        }
        for o in q.get("options", []):
            sources[f"opt_{o['letter']}_text"] = o.get("text", "")
            sources[f"opt_{o['letter']}_expl"] = o.get("explanation", "")
        for where, text in sources.items():
            for matched, name, kind, wl in find_attributions(text):
                entry = (q["question_id"], where, kind, matched, name)
                (whitelisted if wl else violations).append(entry)

    qids_violating = {e[0] for e in violations}
    print(f"Questions with >=1 non-whitelisted violation: {len(qids_violating)}/{n}")
    print(f"Total violation instances: {len(violations)}")
    print(f"Total whitelisted (exempt) instances: {len(whitelisted)}")

    # Group violations by kind
    from collections import Counter
    kind_counts = Counter(e[2] for e in violations)
    print(f"Violations by kind: {dict(kind_counts)}")

    if violations:
        print("\nViolation examples (first 15):")
        for qid, where, kind, matched, name in violations[:15]:
            print(f"  {qid} | {where:22s} | {kind:20s} | '{matched}'")
    if whitelisted:
        print(f"\nWhitelisted matches (first 5 of {len(whitelisted)}):")
        for qid, where, kind, matched, name in whitelisted[:5]:
            print(f"  {qid} | {where:22s} | '{matched}'")
    print()

    # ---- Rule 2: Length balance ----
    print("--- Rule 2: Answer Length Balance ---")
    print(f"{'question_id':40s} {'CORR':>5} "
          f"{'A_ch':>5} {'B_ch':>5} {'C_ch':>5} {'D_ch':>5} "
          f"{'min':>4} {'max':>4} {'mx/mn':>6} {'corr/avgdist':>13} {'longest':>9}")
    ratios = []
    correct_is_longest = 0
    correct_is_longest_by_20 = 0
    cd_ratios = []
    over_thresh = {1.3: 0, 1.5: 0, 1.8: 0, 2.0: 0}
    for q in qs:
        opts = sorted(q["options"], key=lambda o: o["letter"])
        lens = {o["letter"]: len(o["text"]) for o in opts}
        correct = q["correct_answer_letter"]
        correct_len = lens[correct]
        all_lens = list(lens.values())
        mn, mx = min(all_lens), max(all_lens)
        ratio = mx / mn if mn else float("inf")
        ratios.append(ratio)
        is_longest = correct_len == mx
        correct_is_longest += int(is_longest)
        dist_lens = [v for k, v in lens.items() if k != correct]
        avg_dist = statistics.mean(dist_lens)
        cd_ratio = correct_len / avg_dist if avg_dist else float("inf")
        cd_ratios.append(cd_ratio)
        if correct_len > max(dist_lens) * 1.2:
            correct_is_longest_by_20 += 1
        for t in over_thresh:
            if ratio > t:
                over_thresh[t] += 1
        marker = "*" if is_longest else " "
        print(f"{q['question_id']:40s} {correct:>5} "
              f"{lens['A']:>5} {lens['B']:>5} {lens['C']:>5} {lens['D']:>5} "
              f"{mn:>4} {mx:>4} {ratio:>6.2f} {cd_ratio:>13.2f}    {marker:>5}")
    print()
    print(f"Mean max/min ratio: {statistics.mean(ratios):.2f}")
    print(f"Median max/min ratio: {statistics.median(ratios):.2f}")
    print(f"Max max/min ratio observed: {max(ratios):.2f}")
    for t, c in over_thresh.items():
        print(f"Questions with max/min > {t}: {c}/{n} ({100*c/n:.0f}%)")
    print(f"Correct = longest: {correct_is_longest}/{n} (chance baseline: {n*0.25:.1f})")
    print(f"Correct longer than ALL distractors by >=20%: {correct_is_longest_by_20}/{n}")
    print(f"Mean correct/avg-distractor ratio: {statistics.mean(cd_ratios):.2f}")
    print()

    # ---- Rule 3: Elaboration tells ----
    print("--- Rule 3: Elaboration Tells (parens / semicolons / em-dashes / i.e./e.g.) ---")
    print(f"{'question_id':40s} {'c_paren':>8} {'dmax_p':>7} {'c_semi':>7} {'dmax_s':>7} "
          f"{'c_emd':>6} {'dmax_em':>8} {'c_total':>8} {'dmax_t':>7} {'flag':>6}")
    correct_more_than_max_dist = 0
    correct_more_than_avg_dist = 0
    paren_only_correct = 0
    semi_only_correct = 0
    emdash_only_correct = 0
    flag_strict = 0
    for q in qs:
        opts = q["options"]
        correct = q["correct_answer_letter"]
        c_text = next(o["text"] for o in opts if o["letter"] == correct)
        d_texts = [o["text"] for o in opts if o["letter"] != correct]

        c_marks = count_elab(c_text)
        d_marks_list = [count_elab(t) for t in d_texts]

        c_total = sum(c_marks.values())
        d_totals = [sum(d.values()) for d in d_marks_list]
        d_avg = statistics.mean(d_totals)
        d_max = max(d_totals)

        if c_total > d_max:
            correct_more_than_max_dist += 1
        if c_total > d_avg:
            correct_more_than_avg_dist += 1

        c_p, c_s, c_e = c_marks["parens"], c_marks["semicolons"], c_marks["em_dashes"]
        d_p_max = max(d["parens"] for d in d_marks_list)
        d_s_max = max(d["semicolons"] for d in d_marks_list)
        d_e_max = max(d["em_dashes"] for d in d_marks_list)

        if c_p >= 1 and d_p_max == 0:
            paren_only_correct += 1
        if c_s >= 1 and d_s_max == 0:
            semi_only_correct += 1
        if c_e >= 1 and d_e_max == 0:
            emdash_only_correct += 1

        # Strict flag = correct has marker AND no distractor has any marker
        is_strict_flag = (c_total >= 2 and d_max == 0)
        if is_strict_flag:
            flag_strict += 1

        print(f"{q['question_id']:40s} {c_p:>8} {d_p_max:>7} {c_s:>7} {d_s_max:>7} "
              f"{c_e:>6} {d_e_max:>8} {c_total:>8} {d_max:>7}    {'!!' if is_strict_flag else '':>4}")
    print()
    print(f"Correct has more total markers than MAX distractor: {correct_more_than_max_dist}/{n}")
    print(f"Correct has more total markers than AVG distractor: {correct_more_than_avg_dist}/{n}")
    print(f"Parens >=1 ONLY in correct (all distractors 0): {paren_only_correct}/{n}")
    print(f"Semicolons >=1 ONLY in correct: {semi_only_correct}/{n}")
    print(f"Em-dashes >=1 ONLY in correct: {emdash_only_correct}/{n}")
    print(f"STRICT flag (correct has >=2 markers AND distractors 0): {flag_strict}/{n}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path", nargs="?",
        help=("Path to a quiz JSON file. If omitted, the most recently "
              "modified file under data/quiz/<DOMAIN>/*.json is used."),
    )
    args = parser.parse_args()
    if args.path:
        p = pathlib.Path(args.path)
    else:
        # Auto-discover: find the most recently modified quiz JSON anywhere
        # under data/quiz/. Removes the prior hardcoded user-specific
        # default that broke on other systems.
        repo_root = pathlib.Path(__file__).resolve().parent.parent
        candidates = sorted(
            (repo_root / "data" / "quiz").rglob("*.json"),
            key=lambda f: f.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            parser.error(
                "No quiz JSON files found under data/quiz/. "
                "Generate questions first or pass an explicit path."
            )
        p = candidates[0]
        print(f"(no path given — auto-selected most recent: {p})")
    qs = json.loads(p.read_text(encoding="utf-8"))
    audit(qs)


if __name__ == "__main__":
    main()
