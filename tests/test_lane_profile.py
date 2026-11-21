import ast
import copy
import os
import re
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents import lane_profile, local_pipeline, motion_runner, scene_builder, scene_spec
from agents.road_frame import RoadFrame, lane_offset


def sample(t, n, conf=0.8):
    probs = {k: (conf if k == n else (1.0 - conf) / 5.0) for k in lane_profile.COUNTS}
    return {"t": t, "probs": probs}


def series(*runs, step=0.5):
    """runs = (count, number of samples), ..."""
    out, t = [], 0.0
    for n, k in runs:
        for _ in range(k):
            out.append(sample(round(t, 2), n))
            t += step
    return out


class TestSegmentCounts(unittest.TestCase):
    def test_constant_count_is_one_segment(self):
        segs = lane_profile.segment_counts(series((3, 12)))
        self.assertEqual([s["n"] for s in segs], [3])

    def test_a_single_noisy_frame_does_not_split_the_road(self):
        s = series((5, 6), (3, 1), (5, 6))
        self.assertEqual([x["n"] for x in lane_profile.segment_counts(s)], [5])

    def test_five_lanes_then_three(self):
        segs = lane_profile.segment_counts(series((5, 8), (3, 10)))
        self.assertEqual([s["n"] for s in segs], [5, 3])
        self.assertAlmostEqual(lane_profile.boundary_times(segs, series((5, 8), (3, 10)))[0], 3.75)

    def test_short_segment_is_merged_into_a_neighbour(self):
        s = series((4, 6), (3, 2), (2, 8))        # 2 samples = 0.5 s < 1 s minimum
        segs = lane_profile.segment_counts(s, switch_cost=0.5)
        self.assertEqual([x["n"] for x in segs], [4, 2])

    def test_string_keys_from_a_json_cache(self):
        s = [{"t": x["t"], "probs": {str(k): v for k, v in x["probs"].items()}} for x in series((2, 4), (4, 4))]
        self.assertEqual([x["n"] for x in lane_profile.segment_counts(s)], [2, 4])

    def test_empty(self):
        self.assertEqual(lane_profile.segment_counts([]), [])


class TestBuildProfile(unittest.TestCase):
    def test_right_side_drop_keeps_the_left_lanes(self):
        segs = [{"n": 5, "t0": 0, "t1": 3.5, "k0": 0, "k1": 7}, {"n": 3, "t0": 4, "t1": 9, "k0": 8, "k1": 18}]
        prof = lane_profile.build_profile(segs, [3.75], ["right"], t_start=1.0, speed_ms=10.0)
        self.assertEqual(prof, [{"d0": None, "d1": 27.5, "n": 5, "m": 0}, {"d0": 27.5, "d1": None, "n": 3, "m": 0}])

    def test_left_side_drop_raises_the_missing_left_lanes(self):
        segs = [{"n": 5}, {"n": 3}]
        prof = lane_profile.build_profile(segs, [3.0], ["left"], 0.0, 10.0)
        self.assertEqual([(s["n"], s["m"]) for s in prof], [(5, 0), (3, 2)])

    def test_lane_added_on_the_left_shifts_earlier_sections(self):
        prof = lane_profile.build_profile([{"n": 2}, {"n": 3}], [3.0], ["left"], 0.0, 10.0)
        self.assertEqual([(s["n"], s["m"]) for s in prof], [(2, 1), (3, 0)])

    def test_stopped_ego_or_no_change_gives_no_profile(self):
        self.assertEqual(lane_profile.build_profile([{"n": 5}, {"n": 3}], [3.0], ["right"], 0.0, 0.0), [])
        self.assertEqual(lane_profile.build_profile([{"n": 3}], [], [], 0.0, 10.0), [])

    def test_sections_too_short_to_see_are_folded(self):
        prof = lane_profile.build_profile([{"n": 4}, {"n": 3}, {"n": 2}], [3.0, 3.5], ["right", "right"], 0.0, 10.0)
        self.assertEqual([s["n"] for s in prof], [4, 2])
        self.assertIsNone(prof[0]["d0"])
        self.assertIsNone(prof[-1]["d1"])


class TestSpec(unittest.TestCase):
    def test_up_to_six_lanes_and_a_profile_survive_normalisation(self):
        prof = [{"d0": None, "d1": 60.0, "n": 5, "m": 0}, {"d0": 60.0, "d1": None, "n": 3, "m": 0}]
        spec, err = scene_spec.normalize({"road": {"lanes_per_direction": 5, "lane_profile": prof}, "ego": {"lane_index": 2}})
        self.assertIsNone(err)
        self.assertEqual(spec["road"]["lanes_per_direction"], 5)
        self.assertEqual(spec["road"]["lane_profile"], prof)
        self.assertEqual(spec["ego"]["lane_index"], 2)

    def test_broken_profiles_are_dropped(self):
        for bad in ([{"d0": None, "d1": 60.0, "n": 5, "m": 0}, {"d0": 50.0, "d1": None, "n": 3, "m": 0}],     # gap
                    [{"d0": None, "d1": 60.0, "n": 9, "m": 0}, {"d0": 60.0, "d1": None, "n": 3, "m": 0}],     # 9 lanes
                    [{"d0": None, "d1": 60.0, "n": 3, "m": 1}, {"d0": 60.0, "d1": None, "n": 3, "m": 1}]):    # m not anchored
            spec, _ = scene_spec.normalize({"road": {"lane_profile": bad}})
            self.assertIsNone(spec["road"]["lane_profile"])

    def test_old_specs_still_normalise(self):
        spec, _ = scene_spec.normalize({"road": {"lanes_per_direction": 2}})
        self.assertIsNone(spec["road"]["lane_profile"])
        self.assertIsNone(spec["ego"]["lane_index"])


def base_spec(profile=None, lane_index=None, actors=None, event=None):
    spec, _ = scene_spec.normalize({
        "road": {"type": "straight", "lanes_per_direction": 3, "lane_profile": profile},
        "ego": {"speed_kmh": 72.0, "behavior": "cruise", "lane": "right", "lane_index": lane_index},
        "actors": actors or [], "event": event or {"type": "none"}})
    return spec


def build(spec, scene="lp"):
    tmp = tempfile.TemporaryDirectory()
    plan = scene_builder.build(spec, tmp.name, scene)
    with open(os.path.join(tmp.name, scene + "_scene_obj.txt"), encoding="utf-8") as f:
        agents = ast.literal_eval(f.readline())
    with open(os.path.join(tmp.name, scene + ".net.xml"), encoding="utf-8") as f:
        net = f.read()
    return tmp, plan, agents, net


FIVE_TO_THREE = [{"d0": None, "d1": 80.0, "n": 5, "m": 0}, {"d0": 80.0, "d1": None, "n": 3, "m": 0}]


class TestBuilderProfile(unittest.TestCase):
    def test_road_with_sections_knows_its_lanes(self):
        r = scene_builder.Road("straight", 3, None, False,
                               [{"s0": 0, "s1": 100, "n": 5, "m": 0}, {"s0": 100, "s1": 300, "n": 3, "m": 0}])
        self.assertEqual(r.L, 5)
        self.assertEqual(r.Lw, 3)
        self.assertEqual(r.lanes_at(50), [0, 1, 2, 3, 4])
        self.assertEqual(r.lanes_at(150), [2, 3, 4])
        self.assertEqual(r.clamp_lane(150, 0), 2)
        self.assertEqual(r.right_edge_d(150), -3 * 3.5)

    def test_net_is_one_edge_with_the_widest_lane_count(self):
        tmp, plan, agents, net = build(base_spec(FIVE_TO_THREE))
        self.assertEqual(len(re.findall(r'<lane id="eastbound_\d"', net)), 5)
        self.assertEqual(len(re.findall(r'<lane id="westbound_\d"', net)), 3)
        self.assertNotIn('function="internal"', net)                # no junction where the lanes end
        self.assertEqual(plan["road"]["lanes"], 5)
        self.assertEqual([s["n"] for s in plan["road"]["sections"]], [5, 3])
        self.assertAlmostEqual(plan["road"]["sections"][1]["s0"], scene_builder.S_START + 80.0)
    def test_ego_uses_the_measured_lane_and_its_route_ends_on_a_lane_that_exists(self):
        tmp, plan, agents, net = build(base_spec(FIVE_TO_THREE, lane_index=4))   # right-most of five: that lane ends
        fr = RoadFrame(plan["road"]["centerline"])
        x, y = agents["v1"]["location"][0], -agents["v1"]["location"][1]
        self.assertAlmostEqual(fr.to_sd(x, y)[1], lane_offset(5, 0), delta=0.05)
        ex, ey = plan["agents"]["v1"]["path"][-1]
        self.assertAlmostEqual(fr.to_sd(ex, ey)[1], lane_offset(5, 2), delta=0.05)
        self.assertTrue(any("lane profile" in n for n in plan["notes"]))

    def test_without_a_measured_lane_right_means_the_right_most_lane_at_the_start(self):
        tmp, plan, agents, net = build(base_spec(FIVE_TO_THREE))
        fr = RoadFrame(plan["road"]["centerline"])
        x, y = agents["v1"]["location"][0], -agents["v1"]["location"][1]
        self.assertAlmostEqual(fr.to_sd(x, y)[1], lane_offset(5, 0), delta=0.05)

    def test_traffic_ahead_in_a_lane_that_ended_moves_to_the_nearest_lane(self):
        actors = [{"id": "a1", "type": "car", "relation": "ahead", "lane": "same", "distance_m": 120.0, "speed_kmh": 0.0,
                   "behavior": "stopped"}]
        tmp, plan, agents, net = build(base_spec(FIVE_TO_THREE, lane_index=4, actors=actors))
        fr = RoadFrame(plan["road"]["centerline"])
        vid = plan["idmap"]["a1"]
        x, y = agents[vid]["location"][0], -agents[vid]["location"][1]
        self.assertAlmostEqual(fr.to_sd(x, y)[1], lane_offset(5, 2), delta=0.05)

    def test_profile_that_removes_the_merging_cars_lane_is_dropped(self):
        prof = [{"d0": None, "d1": 40.0, "n": 2, "m": 0}, {"d0": 40.0, "d1": None, "n": 1, "m": 1}]    # left lane ends
        actors = [{"id": "c1", "type": "car", "relation": "beside", "lane": "left", "distance_m": 5.0, "speed_kmh": 72.0,
                   "behavior": "cutting_in"}]
        spec = base_spec(prof, actors=actors, event={"type": "collision", "actor_id": "c1", "time_s": 6.0})
        spec["road"]["lanes_per_direction"] = 2
        tmp, plan, agents, net = build(spec)
        self.assertNotIn("sections", plan["road"])
        self.assertTrue(any("lane profile dropped" in n for n in plan["notes"]))
        self.assertIn("intercept", plan["agents"][plan["idmap"]["c1"]])

    def test_no_profile_keeps_the_old_single_edge(self):
        tmp, plan, agents, net = build(base_spec(None))
        self.assertIn('id="eastbound"', net)
        self.assertNotIn("sections", plan["road"])


def xodr_lanes(spec):
    """Build the scene, export the map and apply the lane sections; returns (plan, parsed lane sections, laneOffsets)."""
    import xml.etree.ElementTree as ET
    from agents import map_builder
    tmp, plan, agents, net = build(spec)
    x = map_builder.net_to_xodr(os.path.join(tmp.name, "lp.net.xml"), os.path.join(tmp.name, "lp.xodr"))
    n = map_builder.apply_lane_sections(x, plan["road"]["sections"], RoadFrame(plan["road"]["centerline"]).length)
    road = next(r for r in ET.parse(x).getroot().iter("road")
                if r.find("userData") is not None and r.find("userData").get("value") == "eastbound")
    secs = []
    for sec in road.findall("lanes/laneSection"):
        lanes = []
        for ln in sec.findall("right/lane"):
            w = ln.find("width")
            lk = ln.find("link")
            lanes.append({"id": int(ln.get("id")), "w": tuple(float(w.get(k)) for k in "abcd"),
                          "pred": None if lk.find("predecessor") is None else int(lk.find("predecessor").get("id")),
                          "succ": None if lk.find("successor") is None else int(lk.find("successor").get("id"))})
        secs.append({"s": float(sec.get("s")), "lanes": lanes})
    offs = [(float(o.get("s")),) + tuple(float(o.get(k)) for k in "abcd") for o in road.findall("lanes/laneOffset")]
    tmp.cleanup()
    return plan, n, secs, offs, float(road.get("length"))


def lane_centres(sec, off, ds):
    """Lateral centres of the lanes of a lane section ds metres into it (laneOffset and cubic widths)."""
    poly = lambda c, x: c[0] + c[1] * x + c[2] * x * x + c[3] * x ** 3
    edge = poly(off[1:], ds)
    out = []
    for ln in sorted(sec["lanes"], key=lambda l: -l["id"]):
        w = poly(ln["w"], ds)
        out.append(edge - 0.5 * w)
        edge -= w
    return out


class TestXodrLaneSections(unittest.TestCase):
    def test_right_drop_tapers_the_two_outer_lanes_and_links_them_into_the_lane_that_goes_on(self):
        plan, n, secs, offs, length = xodr_lanes(base_spec(FIVE_TO_THREE))
        self.assertEqual(n, 3)
        self.assertEqual([len(s["lanes"]) for s in secs], [5, 5, 3])
        taper, after = secs[1], secs[2]
        T = after["s"] - taper["s"]
        self.assertAlmostEqual(T, 30.0, delta=0.5)
        ends = [l for l in taper["lanes"] if l["id"] in (-4, -5)]
        for l in ends:
            self.assertAlmostEqual(sum(c * T ** k for k, c in enumerate(l["w"])), 0.0, places=3)   # width 0 at the end
            self.assertEqual(l["succ"], -3)                                                     # merges into the outer lane left
        self.assertEqual([l["succ"] for l in taper["lanes"] if l["id"] in (-1, -2, -3)], [-1, -2, -3])

    def test_lanes_that_go_on_keep_their_position_through_the_taper(self):
        for prof, keep in ((FIVE_TO_THREE, (0, 1, 2)),
                           ([{"d0": None, "d1": 80.0, "n": 5, "m": 0}, {"d0": 80.0, "d1": None, "n": 3, "m": 2}], (2, 3, 4))):
            plan, n, secs, offs, length = xodr_lanes(base_spec(prof))
            taper = secs[1]
            T = secs[2]["s"] - taper["s"]
            c0, c1 = lane_centres(taper, offs[1], 0.0), lane_centres(taper, offs[1], T)
            j_of = sorted(range(5))                                   # taper holds all five lanes (j = 0 .. 4, left to right)
            after = lane_centres(secs[2], offs[2], 0.0)
            for idx, j in enumerate(keep):
                self.assertAlmostEqual(c0[j], -(j + 0.5) * 3.5, places=2)
                self.assertAlmostEqual(c1[j], -(j + 0.5) * 3.5, places=2)
                self.assertAlmostEqual(after[idx], -(j + 0.5) * 3.5, places=2)

    def test_lane_sections_start_where_the_builder_put_the_change(self):
        plan, n, secs, offs, length = xodr_lanes(base_spec(FIVE_TO_THREE))
        frame_len = RoadFrame(plan["road"]["centerline"]).length
        self.assertAlmostEqual(secs[2]["s"], plan["road"]["sections"][1]["s0"] * length / frame_len, delta=0.01)

class TestBarriers(unittest.TestCase):
    def test_right_barrier_follows_the_section(self):
        road = {"lanes": 5, "shoulder_m": 3.0, "edge_right": "barrier",
                "sections": [{"s0": 0, "s1": 100, "n": 5, "m": 0}, {"s0": 100, "s1": 300, "n": 3, "m": 0}]}
        self.assertEqual(motion_runner.barrier_offsets(road, 50.0), [-(5 * 3.5 + 3.3)])
        self.assertEqual(motion_runner.barrier_offsets(road, 150.0), [-(3 * 3.5 + 3.3)])
        self.assertEqual(motion_runner.barrier_offsets(road), [-(5 * 3.5 + 3.3)])


NO_BRANCH = lambda *a: (0.0, 0.0)


def detector_rows(*runs, step=0.5, left=None):
    """Detector windows (agents/lane_lines.finish_timeline format) with confident counts; runs = (count, samples), ..."""
    out = []
    for x in series(*runs, step=step):
        n = max(x["probs"], key=x["probs"].get)
        out.append({"t": x["t"], "probs": x["probs"], "lanes_left": float(n - 1 if left is None else left), "lanes_right": 0.0})
    return out


class TestPipelineProfile(unittest.TestCase):
    def spec(self):
        s = base_spec(None, actors=[{"id": "c1", "type": "car", "relation": "ahead", "lane": "same", "distance_m": 30.0,
                                     "speed_kmh": 30.0, "behavior": "braking"}],
                      event={"type": "collision", "actor_id": "c1", "time_s": 9.0})
        return s

    def test_profile_is_written_into_the_spec(self):
        spec = self.spec()                     # T = 8 s: the simulation starts at video time 1 s, ego at 20 m/s
        samples = detector_rows((5, 8), (3, 18), left=3)   # change at 3.75 s -> 2.75 s after the start -> 55 m
        out = local_pipeline.add_lane_profile(spec, "v.mp4", 20.0, t_crash=9.0, samples=samples, log_fn=lambda m: None, branch_probe=NO_BRANCH)
        self.assertIs(out, samples)
        prof = spec["road"]["lane_profile"]
        self.assertEqual([(s["n"], s["m"]) for s in prof], [(5, 0), (3, 0)])   # lanes_right does not change: right side
        self.assertAlmostEqual(prof[0]["d1"], 55.0)
        self.assertEqual(spec["road"]["lanes_per_direction"], 3)
        self.assertEqual(spec["ego"]["lane_index"], 3)
        checked, _ = scene_spec.normalize(copy.deepcopy(spec))
        self.assertEqual(checked["road"]["lane_profile"], prof)

    def test_constant_count_sets_the_lane_number_only(self):
        spec = self.spec()
        local_pipeline.add_lane_profile(spec, "v.mp4", 20.0, t_crash=9.0, samples=detector_rows((5, 20)), log_fn=lambda m: None, branch_probe=NO_BRANCH)
        self.assertIsNone(spec["road"]["lane_profile"])
        self.assertEqual(spec["road"]["lanes_per_direction"], 5)

    def test_no_measurement_leaves_the_road_alone(self):
        spec = self.spec()
        local_pipeline.add_lane_profile(spec, "v.mp4", 20.0, t_crash=9.0, samples=[{"t": 1.0, "probs": None}],
                                        log_fn=lambda m: None, branch_probe=NO_BRANCH)
        self.assertIsNone(spec["road"]["lane_profile"])
        self.assertEqual(spec["road"]["lanes_per_direction"], 3)

    def test_crossroads_are_left_alone(self):
        spec = self.spec()
        spec["road"]["type"] = "crossroad"
        self.assertIsNone(local_pipeline.add_lane_profile(spec, "v.mp4", 20.0, samples=None, log_fn=lambda m: None, branch_probe=NO_BRANCH))
        self.assertIsNone(spec["road"]["lane_profile"])


class TestDetectorCounting(unittest.TestCase):
    def test_ego_lane_is_the_strongest_plausible_pair_and_other_lines_must_fit_the_lane_grid(self):
        from agents import lane_lines
        d_ref = 100.0                                         # lane 300 px = 3 x depth
        peaks = [(-200.0, 0.5), (100.0, 0.6), (400.0, 1.0), (450.0, 0.1), (700.0, 0.9), (820.0, 0.4)]
        lanes, n_left, lane, xl = lane_lines.count_lanes(peaks, 480.0, d_ref)
        self.assertEqual((xl, lane), (400.0, 300.0))          # the weak line at 450 px is not the ego lane's left line
        self.assertEqual((lanes, n_left), (3, 2))             # 820 px is 0.4 lane beyond: a guard rail, not a lane

    def test_no_plausible_ego_lane_is_a_reason(self):
        from agents import lane_lines
        self.assertIsInstance(lane_lines.count_lanes([(470.0, 1.0), (490.0, 1.0)], 480.0, 100.0), str)

    def test_one_lane_ratio_per_video_and_soft_probabilities(self):
        from agents import lane_lines
        rows = [{"t": 0.5 * i, "ratio": r, "lines_lanes": 1, "extent_left": 1.5 * 3.0, "extent_right": 0.5 * 3.0}
                for i, r in enumerate((3.0, 3.0, 2.9, 3.1, 9.0))]     # one wrong ego lane does not change the ratio
        out = lane_lines.finish_timeline(rows)
        self.assertEqual(out[4]["road_lanes"], 2.0)
        self.assertEqual(max(out[0]["probs"], key=out[0]["probs"].get), 2)
        self.assertAlmostEqual(sum(out[0]["probs"].values()), 1.0, places=3)

    def test_implausible_ratio_is_limited_to_a_real_camera_height(self):
        from agents import lane_lines
        rows = [{"t": float(i), "ratio": 4.5, "extent_left": 9.0, "extent_right": 0.0} for i in range(3)]
        out = lane_lines.finish_timeline(rows)
        self.assertAlmostEqual(out[0]["road_lanes"], 1.0 + 9.0 / lane_lines.RATIO_RANGE[1] - 0.5, places=2)

    def test_change_side_is_where_the_road_narrowed(self):
        from agents import lane_lines
        rows = [{"t": t, "lanes_left": 2.0 if t < 5 else 2.0, "lanes_right": 2.0 if t < 5 else 0.0} for t in range(10)]
        self.assertEqual(lane_lines.change_side(rows, 4.0, 5.0), "right")
        rows = [{"t": t, "lanes_left": 3.0 if t < 5 else 1.0, "lanes_right": 0.0} for t in range(10)]
        self.assertEqual(lane_lines.change_side(rows, 4.0, 5.0), "left")


LEFT_FLYOVER = [{"d0": None, "d1": 60.0, "n": 5, "m": 0},
                {"d0": 60.0, "d1": None, "n": 3, "m": 2, "branch": {"side": "left", "rise_m": 6.0}}]


class TestBranches(unittest.TestCase):
    def test_spec_keeps_a_branch_only_on_the_side_where_lanes_leave(self):
        spec, _ = scene_spec.normalize({"road": {"lane_profile": LEFT_FLYOVER}})
        self.assertEqual(spec["road"]["lane_profile"][1]["branch"], {"side": "left", "rise_m": 6.0})
        wrong = copy.deepcopy(LEFT_FLYOVER)
        wrong[1]["branch"]["side"] = "right"                 # the lanes leave on the left
        spec, _ = scene_spec.normalize({"road": {"lane_profile": wrong}})
        self.assertNotIn("branch", spec["road"]["lane_profile"][1])

    def test_tagging_follows_the_rise_and_split_cues(self):
        prof = lambda: [{"d0": None, "d1": 60.0, "n": 5, "m": 0}, {"d0": 60.0, "d1": None, "n": 3, "m": 2}]
        for probe, want in ((lambda *a: (0.3, 0.6), {"side": "left", "rise_m": 6.0}),
                            (lambda *a: (0.6, 0.3), {"side": "left", "rise_m": 0.0}),
                            (lambda *a: (0.3, 0.3), None)):
            p = prof()
            note = local_pipeline.tag_branches(p, 1.0, 15.0, "v.mp4", 20.0, probe)
            self.assertEqual(p[1].get("branch"), want)
            self.assertIn("video 5.0 s", note)                # 60 m at 15 m/s after the start at 1 s

    def test_a_clearly_rising_other_side_moves_the_leaving_lanes_there(self):
        prof = [{"d0": None, "d1": 30.0, "n": 3, "m": 0}, {"d0": 30.0, "d1": 90.0, "n": 5, "m": 0},
                {"d0": 90.0, "d1": None, "n": 3, "m": 0}]                      # measured: two lanes leave on the right
        probe = lambda video, t, side, dur: (0.35, 0.66) if side == "left" else (0.3, 0.3)
        note = local_pipeline.tag_branches(prof, 1.0, 15.0, "v.mp4", 20.0, probe)
        self.assertEqual(prof[2]["branch"], {"side": "left", "rise_m": 6.0})
        self.assertEqual([(s["n"], s["m"]) for s in prof], [(3, 0), (5, 0), (3, 2)])  # the right three lanes stay
        self.assertIn("on the left", note)
        spec, _ = scene_spec.normalize({"road": {"lane_profile": prof}})
        self.assertEqual(spec["road"]["lane_profile"][2]["branch"]["side"], "left")

    def test_fork_has_one_to_one_connections_and_a_climbing_branch(self):
        tmp, plan, agents, net = build(base_spec(LEFT_FLYOVER, lane_index=4))
        for eid, lanes in (("eastbound", 5), ("eastbound_after", 3), ("branch", 2), ("westbound", 3)):
            self.assertEqual(len(re.findall(r'<lane id="%s_\d"' % eid, net)), lanes, eid)
        cons = set(re.findall(r'<connection from="eastbound" to="(\w+)" fromLane="(\d)" toLane="(\d)"', net))
        self.assertEqual(cons, {("branch", "4", "1"), ("branch", "3", "0"), ("eastbound_after", "2", "2"),
                                ("eastbound_after", "1", "1"), ("eastbound_after", "0", "0")})
        z_of = lambda p: float(p.split(",")[2]) if p.count(",") == 2 else 0.0      # netconvert leaves out z = 0
        zs = [z_of(p) for p in re.search(r'<lane id="branch_0"[^>]*shape="([^"]+)"', net).group(1).split()]
        self.assertAlmostEqual(max(zs), 6.0, delta=0.05)
        wz = [z_of(p) for p in re.search(r'<lane id="westbound_0"[^>]*shape="([^"]+)"', net).group(1).split()]
        self.assertAlmostEqual(max(wz), 6.0, delta=0.05)    # the opposite carriageway climbs with the branch
        fr = RoadFrame(plan["road"]["centerline"])
        x, y = agents["v1"]["location"][0], -agents["v1"]["location"][1]
        self.assertAlmostEqual(fr.to_sd(x, y)[1], lane_offset(5, 0), delta=0.05)

    def test_right_branch_keeps_the_opposite_carriageway_and_ego_on_the_left_lanes(self):
        prof = [{"d0": None, "d1": 60.0, "n": 4, "m": 0}, {"d0": 60.0, "d1": None, "n": 2, "m": 0, "branch": {"side": "right", "rise_m": 0.0}}]
        tmp, plan, agents, net = build(base_spec(prof, lane_index=0))
        self.assertEqual(len(re.findall(r'<lane id="branch_\d"', net)), 2)
        cons = set(re.findall(r'<connection from="eastbound" to="(\w+)" fromLane="(\d)" toLane="(\d)"', net))
        self.assertIn(("branch", "0", "0"), cons)
        self.assertIn(("eastbound_after", "2", "0"), cons)


class TestVehicleEvidence(unittest.TestCase):
    def test_cars_count_only_on_the_ego_road_parallel_and_in_range(self):
        from agents import lane_lines
        car = lambda x, z=15.0, ar=1.4: {"X": x, "Z": z, "ar": ar}
        self.assertEqual(lane_lines.vehicle_lane(car(-4.0), True), ("left", 1))
        self.assertEqual(lane_lines.vehicle_lane(car(-7.6), True), ("left", 2))
        self.assertEqual(lane_lines.vehicle_lane(car(3.2), True), ("right", 1))
        self.assertIsNone(lane_lines.vehicle_lane(car(-4.0), False))          # oncoming carriageway
        self.assertIsNone(lane_lines.vehicle_lane(car(-4.0, ar=2.5), True))   # side view: crossing
        self.assertIsNone(lane_lines.vehicle_lane(car(-4.0, z=60.0), True))   # too far to place
        self.assertIsNone(lane_lines.vehicle_lane(car(0.8), True))            # in the ego lane

    def test_vehicles_raise_the_lanes_the_road_surface_missed(self):
        from agents import lane_lines
        rows = [{"t": 1.0, "lanes_left": 0.2, "lanes_right": 0.0, "road_lanes": 1.2, "probs": lane_lines.count_probs(1.2)},
                {"t": 5.0, "lanes_left": 0.1, "lanes_right": 0.0, "road_lanes": 1.1, "probs": lane_lines.count_probs(1.1)}]
        lane_lines.apply_vehicle_evidence(rows, [{"t": 1.2, "left": 2, "right": 0}])
        self.assertEqual(rows[0]["road_lanes"], 3.0)
        self.assertEqual(max(rows[0]["probs"], key=rows[0]["probs"].get), 3)
        self.assertEqual(rows[1]["road_lanes"], 1.1)                          # no car seen near 5 s


if __name__ == "__main__":
    unittest.main()
