import ast
import copy
import json
import math
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents import scene_builder
from agents.road_frame import RoadFrame, lane_offset

FIX = ROOT / "tests" / "fixtures"


def fixture(name):
    with open(FIX / (name + ".json"), encoding="utf-8") as f:
        return json.load(f)


def arc(radius=250.0, deg=90.0, sign=1, n=45):
    a1 = math.radians(deg)
    return [[round(radius * math.sin(i * a1 / n), 3), round(sign * radius * (1 - math.cos(i * a1 / n)), 3)] for i in range(n + 1)]


def build(spec, scene="s"):
    tmp = tempfile.TemporaryDirectory()
    plan = scene_builder.build(spec, tmp.name, scene)
    with open(os.path.join(tmp.name, scene + "_scene_obj.txt"), encoding="utf-8") as f:
        agents = ast.literal_eval(f.readline())
    with open(os.path.join(tmp.name, scene + ".net.xml"), encoding="utf-8") as f:
        net = f.read()
    return tmp, plan, agents, net


def lane_shape_d(net, lane_id, frame):
    m = re.search(r'<lane id="%s"[^>]*shape="([^"]+)"' % lane_id, net)
    off = re.search(r'netOffset="([-\d.]+),([-\d.]+)"', net)
    ox, oy = float(off.group(1)), float(off.group(2))
    ds = []
    for p in m.group(1).split():
        x, y = map(float, p.split(","))
        ds.append(frame.to_sd(x - ox, y - oy)[1])
    return ds


class TestCurvedRoad(unittest.TestCase):
    def _lanes_follow_frame(self, sign):
        spec = fixture("v1")
        spec["road"]["centerline"] = arc(sign=sign)
        tmp, plan, agents, net = build(spec)
        self.addCleanup(tmp.cleanup)
        frame = RoadFrame(plan["road"]["centerline"])
        for k in (0, 1):
            ds = lane_shape_d(net, "eastbound_%d" % k, frame)
            self.assertTrue(all(abs(d - lane_offset(2, k)) < 0.3 for d in ds), (k, min(ds), max(ds)))
        return plan, agents, frame

    def test_left_bend_net_lanes_follow_the_centerline(self):
        self._lanes_follow_frame(+1)

    def test_right_bend_net_lanes_follow_the_centerline(self):
        self._lanes_follow_frame(-1)

    def test_actors_stand_on_their_lanes_on_a_bend(self):
        plan, agents, frame = self._lanes_follow_frame(+1)
        for aid, a in agents.items():
            x, y, _ = a["location"]
            s, d = frame.to_sd(x, -y)
            self.assertTrue(any(abs(d - lane_offset(2, k)) < 0.3 for k in (0, 1)), (aid, d))
            self.assertAlmostEqual(-a["rotation"][1], frame.heading(s), delta=1.5)


class TestRoadOptions(unittest.TestCase):
    def test_one_way_writes_only_the_main_carriageway(self):
        spec = fixture("v1")
        spec["road"]["one_way"] = True
        tmp, plan, agents, net = build(spec)
        self.addCleanup(tmp.cleanup)
        self.assertIn('id="eastbound"', net)
        self.assertNotIn('id="westbound"', net)
        self.assertTrue(plan["road"]["one_way"])

    def test_plan_carries_road_options(self):
        spec = fixture("v1")
        spec["road"]["edge_right"] = "barrier"
        tmp, plan, agents, net = build(spec)
        self.addCleanup(tmp.cleanup)
        self.assertEqual(plan["road"]["edge_right"], "barrier")
        self.assertEqual(plan["road"]["edge_left"], "shoulder")
        self.assertEqual(plan["road"]["shoulder_m"], 3.0)
        self.assertEqual(plan["road"]["lanes"], 2)
        self.assertEqual(plan["road"]["centerline"], [[0.0, 0.0], [300.0, 0.0]])


class TestTraffic(unittest.TestCase):
    def test_traffic_is_capped_spaced_and_outside_the_crash_corridor(self):
        spec = fixture("v1")
        spec["traffic"] = {"left_lane": 8, "ego_lane_ahead": 6, "right_lane": 0, "oncoming": 0, "rel_speed_kmh": -10.0}
        tmp, plan, agents, net = build(spec)
        self.addCleanup(tmp.cleanup)
        extra = {k: v for k, v in agents.items() if k not in ("v1", "v2", "v3")}
        self.assertLessEqual(len(extra), 12)
        self.assertGreaterEqual(len(extra), 8)
        frame = RoadFrame(plan["road"]["centerline"])
        v_e = spec["ego"]["speed_kmh"] / 3.6
        by_lane = {}
        for k, a in extra.items():
            s, d = frame.to_sd(a["location"][0], -a["location"][1])
            by_lane.setdefault(round(d, 1), []).append(s)
            ahead = s > 15.0
            if ahead:
                v_t = plan["agents"][k]["speed_kmh"] / 3.6
                self.assertGreaterEqual(s - 15.0 - (v_e - v_t) * 10.0, 24.0, (k, s))
        for d, ss in by_lane.items():
            ss.sort()
            self.assertTrue(all(b - a >= 9.0 for a, b in zip(ss, ss[1:])), (d, ss))

    def test_no_traffic_leaves_the_scene_unchanged(self):
        spec = fixture("v1")
        base = build(copy.deepcopy(spec))
        spec["traffic"] = {"left_lane": 0, "ego_lane_ahead": 0, "right_lane": 0, "oncoming": 0, "rel_speed_kmh": 0.0}
        again = build(spec)
        self.addCleanup(base[0].cleanup)
        self.addCleanup(again[0].cleanup)
        self.assertEqual(base[2], again[2])

    def test_right_lane_traffic_is_skipped_when_the_ego_is_in_the_right_lane(self):
        spec = fixture("v1")
        spec["traffic"] = {"left_lane": 0, "ego_lane_ahead": 0, "right_lane": 4, "oncoming": 0, "rel_speed_kmh": 0.0}
        tmp, plan, agents, net = build(spec)
        self.addCleanup(tmp.cleanup)
        self.assertEqual(len(agents), 3)
        self.assertTrue(any("right lane" in n for n in plan["notes"]))

    def test_heavy_ego_lane_queue_at_high_speed_never_enters_the_corridor(self):
        spec = fixture("v1")
        spec["ego"]["speed_kmh"] = 110.0
        spec["traffic"] = {"left_lane": 0, "ego_lane_ahead": 12, "right_lane": 0, "oncoming": 0, "rel_speed_kmh": -40.0}
        tmp, plan, agents, net = build(spec)
        self.addCleanup(tmp.cleanup)
        v_e = 110.0 / 3.6
        frame = RoadFrame(plan["road"]["centerline"])
        for k, a in agents.items():
            if k in ("v1", "v2", "v3"):
                continue
            s, _ = frame.to_sd(a["location"][0], -a["location"][1])
            v_t = plan["agents"][k]["speed_kmh"] / 3.6
            self.assertGreaterEqual(s - 15.0 - (v_e - v_t) * 10.0, 24.0, (k, s))


class TestBrakingLead(unittest.TestCase):
    def test_braking_lead_gets_a_closed_loop_brake_trigger(self):
        spec = fixture("v1")
        spec["actors"] = [{"id": "c1", "type": "car", "relation": "ahead", "lane": "same", "distance_m": 30.0, "speed_kmh": 40.0,
                           "behavior": "braking", "color": None}]
        spec["event"] = {"type": "collision", "actor_id": "c1", "actor2_id": None, "time_s": 6.0, "description": "x"}
        tmp, plan, agents, net = build(spec)
        self.addCleanup(tmp.cleanup)
        self.assertEqual(plan["agents"]["v2"]["brake_trigger"], {"who": "v1", "ttc_s": 2.0})
        self.assertIn("stop_at_s", plan["agents"]["v2"])


class TestCutIn(unittest.TestCase):
    def test_cut_in_uses_an_intercept_and_expects_a_side_contact(self):
        tmp, plan, agents, net = build(fixture("v1"))
        self.addCleanup(tmp.cleanup)
        icp = plan["agents"]["v2"]["intercept"]
        self.assertEqual(icp["kind"], "cut_in")
        self.assertEqual(icp["target"], "v1")
        self.assertEqual(icp["lead_s"], 1.4)
        self.assertEqual(icp["lane_src_d"], lane_offset(2, 1))
        self.assertEqual(icp["lane_dst_d"], lane_offset(2, 0))
        self.assertNotIn("polyline", plan["agents"]["v2"])
        self.assertEqual(plan["expect"]["contact"], "side")
        self.assertEqual(plan["expect"]["window_s"], 1.5)

    def test_cut_in_from_the_right_mirrors_the_left_case(self):
        spec = fixture("v1")
        spec["ego"]["lane"] = "left"
        spec["actors"][0]["lane"] = "right"
        tmp, plan, agents, net = build(spec)
        self.addCleanup(tmp.cleanup)
        icp = plan["agents"]["v2"]["intercept"]
        self.assertEqual(icp["lane_src_d"], lane_offset(2, 0))
        self.assertEqual(icp["lane_dst_d"], lane_offset(2, 1))

    def test_cut_in_on_a_one_lane_road_is_placed_ahead_and_expects_any_contact(self):
        spec = fixture("v1")
        spec["road"]["lanes_per_direction"] = 1
        tmp, plan, agents, net = build(spec)
        self.addCleanup(tmp.cleanup)
        self.assertNotIn("intercept", plan["agents"]["v2"])
        self.assertEqual(plan["expect"]["contact"], "any")
        self.assertTrue(any("no neighbouring lane" in n for n in plan["notes"]))


class TestCrossroadTrafficAndTiming(unittest.TestCase):
    def _crossroad_spec(self, speed_kmh=50.0, time_s=4.2):
        return {
            "road": {"type": "crossroad", "lanes_per_direction": 2, "traffic_signals": True, "one_way": False},
            "traffic": {"ego_lane_ahead": 2, "left_lane": 2, "right_lane": 0, "oncoming": 1, "rel_speed_kmh": 0.0},
            "ego": {"type": "car", "lane": "right", "speed_kmh": speed_kmh, "behavior": "cruise"},
            "actors": [{"id": "c1", "type": "suv", "relation": "cross_left", "lane": "same", "distance_m": 15.0,
                        "speed_kmh": 20.0, "behavior": "crossing", "color": "silver"}],
            "event": {"type": "collision", "actor_id": "c1", "actor2_id": None, "time_s": time_s, "description": "cross"},
        }

    def test_crossroad_traffic_queues_adjacent_lanes_and_clears_junction(self):
        spec = self._crossroad_spec()
        tmp, plan, agents, net = build(spec)
        self.addCleanup(tmp.cleanup)
        road_c_y = 100.0
        half_j = 2 * 3.5 + 4.0
        y_stop = road_c_y - half_j - 3.0
        extra = {k: v for k, v in agents.items() if k not in ("v1", "v2")}
        self.assertEqual(len(extra), 5)
        stopped_left = [k for k, a in extra.items() if plan["agents"][k]["mode"] == "static" and -a["location"][1] < road_c_y]
        stopped_onc = [k for k, a in extra.items() if plan["agents"][k]["mode"] == "static" and -a["location"][1] > road_c_y]
        ahead_north = [k for k, a in extra.items() if plan["agents"][k]["mode"] == "drive" and -a["location"][1] >= road_c_y + half_j + 20.0]
        self.assertEqual(len(stopped_left), 2)
        self.assertEqual(len(stopped_onc), 1)
        self.assertEqual(len(ahead_north), 2)
        for k in stopped_left:
            y_net = -agents[k]["location"][1]
            self.assertLessEqual(y_net, y_stop + 1e-6)
        for k, a in extra.items():
            y_net = -a["location"][1]
            self.assertFalse(road_c_y - half_j <= y_net <= road_c_y + half_j, (k, y_net))

    def test_crossroad_full_left_queue_packs_to_stop_line_blocking_sightline(self):
        spec = self._crossroad_spec(speed_kmh=50.0, time_s=4.55)
        spec["traffic"] = {"ego_lane_ahead": 0, "left_lane": 5, "right_lane": 0, "oncoming": 0, "rel_speed_kmh": 0.0}
        tmp, plan, agents, net = build(spec)
        self.addCleanup(tmp.cleanup)
        road_c_y = 100.0
        half_j = 2 * 3.5 + 4.0
        v1_y = -agents["v1"]["location"][1]
        stopped_left_ys = sorted(
            -a["location"][1] for k, a in agents.items()
            if k not in ("v1", "v2") and plan["agents"][k]["mode"] == "static" and -a["location"][1] < road_c_y
        )
        self.assertEqual(len(stopped_left_ys), 5)
        self.assertAlmostEqual(stopped_left_ys[-1], road_c_y - half_j - 3.0, delta=0.1)
        self.assertGreater(stopped_left_ys[0], v1_y + 2.5)
        for a, b in zip(stopped_left_ys, stopped_left_ys[1:]):
            self.assertAlmostEqual(b - a, 7.5, delta=0.1)

    def test_moving_ego_on_crossroad_with_clamped_y0_syncs_conflict_time(self):
        spec = self._crossroad_spec(speed_kmh=90.0, time_s=8.0)
        tmp, plan, agents, net = build(spec)
        self.addCleanup(tmp.cleanup)
        v1_x, v1_y_carla, _ = agents["v1"]["location"]
        v2_x, v2_y_carla, _ = agents["v2"]["location"]
        v1_y = -v1_y_carla
        v_e = 90.0 / 3.6
        v_a = 20.0 / 3.6
        conflict_y = -v2_y_carla
        t_ego = (conflict_y - v1_y) / v_e
        t_act = (v1_x - v2_x) / v_a
        self.assertAlmostEqual(t_ego, t_act, places=1)
        self.assertAlmostEqual(plan["expect"]["time_s"], t_ego, places=1)

    def test_straight_road_crossing_vehicle_drives_perpendicular_across_lanes(self):
        spec = {
            "road": {"type": "straight", "lanes_per_direction": 2, "traffic_signals": False, "one_way": False},
            "ego": {"type": "car", "lane": "right", "speed_kmh": 50.0, "behavior": "cruise"},
            "actors": [{"id": "c1", "type": "car", "relation": "cross_left", "lane": "same", "distance_m": 15.0,
                        "speed_kmh": 20.0, "behavior": "crossing", "color": "black"}],
            "event": {"type": "collision", "actor_id": "c1", "actor2_id": None, "time_s": 7.0, "description": "cross"},
        }
        tmp, plan, agents, net = build(spec)
        self.addCleanup(tmp.cleanup)
        v2_x, v2_y_carla, _ = agents["v2"]["location"]
        self.assertAlmostEqual(agents["v2"]["rotation"][1], 90.0, places=1)
        poly = plan["agents"]["v2"]["polyline"]
        self.assertGreater(-v2_y_carla, 0.0)
        self.assertLess(poly[-1][1], 0.0)
        self.assertAlmostEqual(poly[0][0], poly[-1][0], places=1)
        self.assertEqual(plan["road"]["shoulder_m"], 6.0)
        d_start = -v2_y_carla
        d_ego = lane_offset(2, 0)
        t_cross = abs(d_start - d_ego) / (20.0 / 3.6)
        self.assertAlmostEqual(plan["expect"]["time_s"], t_cross, places=1)
        self.assertAlmostEqual(v2_x, 15.0 + (50.0 / 3.6) * t_cross, delta=0.5)

    def test_straight_road_third_party_crossing_keeps_straight_road_and_places_all_cars(self):
        spec = {
            "road": {"type": "straight", "lanes_per_direction": 2, "traffic_signals": False, "one_way": False},
            "ego": {"type": "car", "lane": "left", "speed_kmh": 0.0, "behavior": "stopped"},
            "actors": [
                {"id": "c1", "type": "car", "relation": "cross_left", "lane": "same", "distance_m": 7.5,
                 "speed_kmh": 20.0, "behavior": "crossing", "color": "black"},
                {"id": "c2", "type": "van", "relation": "behind", "lane": "right", "distance_m": 20.0,
                 "speed_kmh": 50.0, "behavior": "driving", "color": "brown"},
                {"id": "b1", "type": "car", "relation": "ahead", "lane": "same", "distance_m": 12.0,
                 "speed_kmh": 0.0, "behavior": "stopped", "color": "white"},
            ],
            "event": {"type": "collision", "actor_id": "c1", "actor2_id": "c2", "time_s": 14.2, "description": "tp"},
        }
        tmp, plan, agents, net = build(spec)
        self.addCleanup(tmp.cleanup)
        self.assertEqual(plan["road"]["centerline"], [[0.0, 0.0], [300.0, 0.0]])
        self.assertEqual(plan["road"]["shoulder_m"], 6.0)
        self.assertEqual(plan["agents"]["v1"]["mode"], "static")
        self.assertAlmostEqual(agents["v1"]["location"][0], 65.0, places=1)
        self.assertAlmostEqual(-agents["v1"]["location"][1], lane_offset(2, 1), places=2)
        self.assertAlmostEqual(agents["v2"]["location"][0], 72.5, places=1)
        self.assertEqual(agents["v2"]["rotation"][1], 90.0)
        self.assertAlmostEqual(-agents["v3"]["location"][1], lane_offset(2, 0), places=2)
        self.assertGreater(agents["v4"]["location"][0], agents["v2"]["location"][0] + 7.5)
        self.assertEqual(plan["agents"]["v4"]["mode"], "static")
        self.assertEqual(plan["expect"]["actors"], ["v2", "v3"])

    def test_crossroad_stopped_cross_street_vehicle_parks_at_stop_line(self):
        spec = self._crossroad_spec()
        spec["actors"].extend([
            {"id": "b1", "type": "van", "relation": "cross_right", "lane": "same",
             "distance_m": 13.0, "speed_kmh": 0.0, "behavior": "stopped", "color": "black"},
            {"id": "b2", "type": "car", "relation": "cross_right", "lane": "same",
             "distance_m": 20.0, "speed_kmh": 0.0, "behavior": "stopped", "color": "white"},
        ])
        tmp, plan, agents, net = build(spec)
        self.addCleanup(tmp.cleanup)
        half_j = 2 * 3.5 + 4.0
        self.assertGreater(agents["v3"]["location"][0], 100.0 + half_j)
        self.assertGreaterEqual(abs(agents["v4"]["location"][0] - agents["v3"]["location"][0]), 8.5)
        self.assertEqual(plan["agents"]["v3"]["mode"], "static")
        self.assertEqual(plan["agents"]["v4"]["mode"], "static")

    def test_crossroad_moving_ego_places_background_driving_ahead_past_junction(self):
        spec = self._crossroad_spec(speed_kmh=50.0, time_s=4.5)
        spec["traffic"] = {}
        spec["actors"].append({"id": "b1", "type": "car", "relation": "ahead", "lane": "same",
                               "distance_m": 18.0, "speed_kmh": 50.0, "behavior": "driving", "color": "white"})
        tmp, plan, agents, net = build(spec)
        self.addCleanup(tmp.cleanup)
        half_j = 2 * 3.5 + 4.0
        y_net = -agents["v3"]["location"][1]
        self.assertGreaterEqual(y_net, 100.0 + half_j + 8.0 - 1e-4)


if __name__ == "__main__":
    unittest.main()
