"""Validate that a brief's misconception pool is structurally adequate
for the DistractorPlannerAgent.

The DistractorPlanner picks 3 unique misconceptions per question, prefers
those involving the tested concept (the "primary" pool), and rotates by
(variant, tier) for diversity across the 20 questions per anchor. For
this to produce good distractors, the brief's pool needs:

  • ≥3 total misconceptions (hard minimum already enforced in
    validate_brief, repeated here for completeness).
  • Each concept covered by ≥1 misconception (zero primary misconceptions
    means questions on that concept rely entirely on the secondary pool,
    losing the prioritization the planner is designed for).
  • Misconceptions reference real concepts in the brief's concepts list
    (no orphan refs — those break DistractorPlanner's primary/secondary
    split silently).
  • Type diversity: if the pool has ≥4 misconceptions, at least 2 distinct
    types so students don't see monotone wrong-answer feedback.

Issues are warnings, not failures: the save proceeds either way. The
brief generator prints them so authors can spot adequacy problems before
~31,320 questions are produced from a thin pool.
"""


def validate_pool_adequacy(brief):
    """Run pool adequacy checks. Returns a list of issue dicts (empty = clean).

    Each issue dict has at minimum: type, detail. Optional: concept_id,
    concepts (for orphan refs), or other context-specific fields.
    """
    issues = []

    concepts = brief.get("concepts", []) or []
    misconceptions = brief.get("misconceptions", []) or []
    concept_ids = {c.get("concept_id") for c in concepts if c.get("concept_id")}

    # 1. Total pool minimum
    if len(misconceptions) < 3:
        issues.append({
            "type": "pool_too_small",
            "detail": f"need 3+ misconceptions for distractor planner, "
                      f"got {len(misconceptions)}",
        })

    # 2. Per-concept primary coverage
    for concept_id in sorted(concept_ids):
        primary_count = sum(
            1 for m in misconceptions
            if concept_id in (m.get("concepts_involved") or [])
        )
        if primary_count == 0:
            issues.append({
                "type": "no_primary_misconceptions",
                "concept_id": concept_id,
                "detail": (
                    "no misconceptions reference this concept; questions "
                    "testing it will fall back to the secondary pool"
                ),
            })

    # 3. Orphan concept references in misconceptions
    referenced = set()
    for m in misconceptions:
        for c in (m.get("concepts_involved") or []):
            referenced.add(c)
    orphans = referenced - concept_ids
    if orphans:
        issues.append({
            "type": "orphan_concept_references",
            "concepts": sorted(orphans),
            "detail": (
                "misconceptions reference concept_ids not present in the "
                "brief's concepts list — DistractorPlanner's primary "
                "matching will silently miss these"
            ),
        })

    # 4. Type diversity for non-trivial pools
    types_used = {m.get("type") for m in misconceptions if m.get("type")}
    if len(misconceptions) >= 4 and len(types_used) < 2:
        issues.append({
            "type": "low_type_diversity",
            "detail": (
                f"only {len(types_used)} distinct misconception type(s) in "
                f"pool of {len(misconceptions)}; students see monotone "
                f"wrong-answer feedback patterns"
            ),
        })

    return issues
