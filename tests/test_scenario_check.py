import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import scenario_check

EXPECT = {"type": "collision", "actors": ["v1", "v2"], "time_s": 7.0, "contact": "side", "window_s": 1.5}


def ev(actor, other_actor, t, impulse=(0, 800, 0), other="vehicle.x", ego_yaw=0.0, rel=(0.0, 2.0)):
    return {"t_sim": t, "actor": actor, "other_actor": other_actor, "other": other,
            "impulse_vec": list(impulse), "ego_yaw": ego_yaw, "rel": list(rel)}


def good_result():
    return {"outcome": "collision", "events": [ev("v1", "v2", 7.2), ev("v2", "v1", 7.2)], "min_pair_dist_m": {"v1-v2": 1.0}}


class TestEvaluate(unittest.TestCase):
    def test_side_contact_in_window_passes(self):
        ok, detail = scenario_check.evaluate(EXPECT, good_result())
        self.assertTrue(ok, detail)

    def test_longitudinal_contact_fails_direction(self):
        r = {"events": [ev("v1", "v2", 7.2, impulse=(900, 0, 0), rel=(2.0, 0.0))]}
        ok, detail = scenario_check.evaluate(EXPECT, r)
        self.assertFalse(ok)
        self.assertTrue(any("direction" in x for x in detail["reasons"]), detail)

    def test_contact_outside_window_fails_time(self):
        r = {"events": [ev("v1", "v2", 10.0)]}
        ok, detail = scenario_check.evaluate(EXPECT, r)
        self.assertFalse(ok)
        self.assertTrue(any("time" in x for x in detail["reasons"]), detail)

    def test_scenery_hit_by_ego_fails(self):
        r = {"events": [ev("v1", "static", 7.9, other="static.prop.streetbarrier"), ev("v1", "v2", 7.2)]}
        ok, detail = scenario_check.evaluate(EXPECT, r)
        self.assertFalse(ok)
        self.assertTrue(any("scenery" in x for x in detail["reasons"]), detail)

    def test_vertical_static_road_contact_is_ignored(self):
        r = {"events": [ev("v1", "static.road", 6.5, other="static.road", impulse=(0.0, 0.0, 155.0)),
                        ev("v1", "v2", 7.2), ev("v2", "v1", 7.2)]}
        ok, detail = scenario_check.evaluate(EXPECT, r)
        self.assertTrue(ok, detail)

    def test_no_pair_contact_fails(self):
        ok, detail = scenario_check.evaluate(EXPECT, {"events": []})
        self.assertFalse(ok)
        self.assertTrue(any("no contact" in x for x in detail["reasons"]), detail)

    def test_contact_type_any_skips_direction(self):
        exp = dict(EXPECT, contact="any")
        r = {"events": [ev("v1", "v2", 7.2, impulse=(900, 0, 0), rel=(2.0, 0.0))]}
        ok, _ = scenario_check.evaluate(exp, r)
        self.assertTrue(ok)

    def test_none_expectation_fails_on_ego_vehicle_contact(self):
        exp = {"type": "none", "actors": ["v1"]}
        ok, _ = scenario_check.evaluate(exp, {"events": [ev("v1", "v2", 3.0)]})
        self.assertFalse(ok)


class TestSearch(unittest.TestCase):
    def _setup(self, tmp):
        plan = {"ego": "v1", "expect": dict(EXPECT),
                "agents": {"v1": {"mode": "drive"}, "v2": {"mode": "drive", "intercept": {"kind": "cut_in", "target": "v1", "lead_s": 1.4, "aim_offset_m": 0.0}}}}
        with open(os.path.join(tmp, "s_motion.json"), "w", encoding="utf-8") as f:
            json.dump(plan, f)

    def test_retries_change_controller_parameters_until_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._setup(tmp)
            seen = []

            def run_fn(load_map):
                with open(os.path.join(tmp, "s_motion.json"), encoding="utf-8") as f:
                    plan = json.load(f)
                seen.append(plan["agents"]["v2"]["intercept"]["lead_s"])
                return good_result() if len(seen) == 3 else {"events": []}

            out = scenario_check.search(tmp, "s", run_fn, log=lambda m: None)
            self.assertEqual(out["status"], "ready")
            self.assertEqual(out["iterations"], 3)
            self.assertEqual(seen, [1.4, 1.9, 0.9])

    def test_always_failing_is_best_effort_after_four_runs(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._setup(tmp)
            out = scenario_check.search(tmp, "s", lambda load_map: {"events": []}, log=lambda m: None)
            self.assertEqual(out["status"], "best_effort")
            self.assertEqual(out["iterations"], 4)
            self.assertTrue(all(h["detail"]["reasons"] for h in out["history"]))

    def test_event_actor_without_intercept_gets_a_spawn_offset(self):
        with tempfile.TemporaryDirectory() as tmp:
            plan = {"ego": "v1", "expect": dict(EXPECT, contact="any"), "agents": {"v1": {"mode": "drive"}, "v2": {"mode": "drive"}}}
            with open(os.path.join(tmp, "s_motion.json"), "w", encoding="utf-8") as f:
                json.dump(plan, f)
            seen = []

            def run_fn(load_map):
                with open(os.path.join(tmp, "s_motion.json"), encoding="utf-8") as f:
                    seen.append(json.load(f)["agents"]["v2"].get("offset_m"))
                return {"events": []}

            scenario_check.search(tmp, "s", run_fn, log=lambda m: None)
            self.assertEqual(seen, [None, 3.0, -3.0, 6.0])

    def test_only_first_run_loads_the_map(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._setup(tmp)
            flags = []
            scenario_check.search(tmp, "s", lambda load_map: (flags.append(load_map), {"events": []})[1], log=lambda m: None)
            self.assertEqual(flags, [True, False, False, False])


if __name__ == "__main__":
    unittest.main()


class TestReplayInTheWay(unittest.TestCase):
    def test_only_replayed_vehicles_hit_by_the_crash_vehicles_before_the_event(self):
        from agents.scenario_check import replay_in_the_way
        plan = {"agents": {"v1": {}, "v2": {}, "v5": {"replay": "r3"}, "v6": {"replay": "r4"}, "v7": {}}}
        expect = {"actors": ["v1", "v2"], "time_s": 8.0, "window_s": 1.5}
        events = [{"actor": "v2", "other_actor": "v5", "t_sim": 4.5},     # the cut-in car hits a replayed car: out
                  {"actor": "v6", "other_actor": "v7", "t_sim": 3.0},     # replayed cars among themselves: not our business
                  {"actor": "v1", "other_actor": "v7", "t_sim": 2.0},     # not a replayed vehicle
                  {"actor": "v1", "other_actor": "v6", "t_sim": 12.0}]    # after the event
        self.assertEqual(replay_in_the_way(plan, expect, {"events": events}), ["v5"])
