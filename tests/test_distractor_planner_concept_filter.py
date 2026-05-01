"""Tests for DistractorPlannerAgent's permitted_concept_ids filter.

Cross-planner alignment fix: when the form planner restricts the correct
answer to a permitted concept set, the distractor planner should filter
its misconception pool to those tied to in-scope concepts. Without this,
the planner could pick a misconception about an off-topic concept,
leaving the LLM no choice but to write an off-topic distractor.

Calibration finding (D7-PHY-209 phase 2 v4): question M-02 had distractor
B about "embolism vs thrombosis" on a stem about "contralateral motor
control" because the misconception pool wasn't filtered.
"""
import unittest

import conftest  # noqa: F401  — sets sys.path

from pipeline.agents import DistractorPlannerAgent


def _misc(mid, concepts, label="lbl", mtype="similar_property"):
    return {
        "misconception_id": mid,
        "label": label,
        "type": mtype,
        "concepts_involved": concepts,
    }


class TestPermittedConceptFilter(unittest.TestCase):
    def setUp(self):
        self.agent = DistractorPlannerAgent()

    def test_filters_out_off_topic_misconceptions(self):
        # Tested concept: contralateral-motor-control
        # Permitted set: {contralateral-motor-control, hemiplegia}
        # Pool: 4 in-scope + 1 off-topic
        misconceptions = [
            _misc("m1", ["contralateral-motor-control"]),
            _misc("m2", ["hemiplegia"]),
            _misc("m3", ["contralateral-motor-control", "hemiplegia"]),
            _misc("m4", ["contralateral-motor-control"]),
            _misc("off-topic", ["embolism-vs-thrombosis"]),  # OFF-TOPIC
        ]
        out = self.agent.execute({
            "tier": 3, "variant": 1,
            "misconceptions": misconceptions,
            "tested_concept_id": "contralateral-motor-control",
            "permitted_concept_ids": ["contralateral-motor-control", "hemiplegia"],
        })
        # Should be in focused mode
        self.assertEqual(out["mode"], "focused")
        # The off-topic misconception must NOT appear in any slot
        assigned_ids = {s.get("misconception_id") for s in out["slots"]}
        self.assertNotIn("off-topic", assigned_ids,
                         "off-topic misconception was selected despite filter")

    def test_falls_back_when_filter_too_aggressive(self):
        # If filtering would leave fewer than 3 misconceptions, fall back
        # to the unfiltered pool — better an off-topic distractor than
        # an empty slot.
        misconceptions = [
            _misc("in-scope", ["hemiplegia"]),  # only 1 in-scope
            _misc("off1", ["other"]),
            _misc("off2", ["yet-another"]),
            _misc("off3", ["unrelated"]),
        ]
        out = self.agent.execute({
            "tier": 3, "variant": 1,
            "misconceptions": misconceptions,
            "tested_concept_id": "hemiplegia",
            "permitted_concept_ids": ["hemiplegia"],
        })
        # Should still produce focused mode (≥3 misconceptions in pool)
        self.assertEqual(out["mode"], "focused")
        assigned_ids = {s.get("misconception_id") for s in out["slots"]}
        self.assertEqual(len(assigned_ids), 3,
                         "should still fill 3 slots from fallback pool")

    def test_no_permitted_ids_leaves_pool_unchanged(self):
        # Backward compatibility: when permitted_concept_ids is missing or
        # empty, use the entire misconception pool (existing behavior).
        misconceptions = [
            _misc(f"m{i}", ["something"]) for i in range(5)
        ]
        out_none = self.agent.execute({
            "tier": 3, "variant": 1,
            "misconceptions": misconceptions,
            "tested_concept_id": "something",
        })
        out_empty = self.agent.execute({
            "tier": 3, "variant": 1,
            "misconceptions": misconceptions,
            "tested_concept_id": "something",
            "permitted_concept_ids": [],
        })
        # Both should produce focused-mode plans — proves the filter
        # doesn't fire when permitted_ids is missing/empty.
        self.assertEqual(out_none["mode"], "focused")
        self.assertEqual(out_empty["mode"], "focused")

    def test_all_off_topic_falls_back_gracefully(self):
        # Pathological case: NO misconceptions involve permitted concepts.
        # Filter would leave 0 → falls back to unfiltered pool. Better
        # than crashing or returning open mode.
        misconceptions = [
            _misc(f"m{i}", ["off-topic-concept"]) for i in range(4)
        ]
        out = self.agent.execute({
            "tier": 3, "variant": 1,
            "misconceptions": misconceptions,
            "tested_concept_id": "in-scope-concept",
            "permitted_concept_ids": ["in-scope-concept"],
        })
        self.assertEqual(out["mode"], "focused")
        # Pool fell back; off-topic misconceptions still get assigned
        assigned_ids = {s.get("misconception_id") for s in out["slots"]}
        self.assertEqual(len(assigned_ids), 3)


if __name__ == "__main__":
    unittest.main()
