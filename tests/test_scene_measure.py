"""Camera geometry, vehicle tracks in road coordinates, the bird's-eye lane map, and replaying measured vehicles."""
import ast
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents import bev_lanes, camera_geom, local_pipeline, motion_runner, scene_builder, scene_spec, vehicle_tracks
from agents.road_frame import RoadFrame, lane_offset

CAM = camera_geom.Camera(960, 540, 480.0, 480.0, 230.0, 1.3)


class TestCamera(unittest.TestCase):
    def test_ground_and_pixel_are_inverse(self):
        for X, Z in ((0.0, 10.0), (-3.5, 20.0), (5.0, 7.5)):
            u, v = CAM.pixel_from_ground(X, Z)
            X2, Z2 = CAM.ground_from_pixel(u, v)
            self.assertAlmostEqual(X, X2, places=6)
            self.assertAlmostEqual(Z, Z2, places=6)
        self.assertIsNone(CAM.ground_from_pixel(400, 229))                 # above the horizon

    def test_lateral_position_does_not_depend_on_the_focal_length(self):
        other = camera_geom.Camera(960, 540, 300.0, 480.0, 230.0, 1.3)
        u, v = 300.0, 300.0
        self.assertAlmostEqual(CAM.ground_from_pixel(u, v)[0], other.ground_from_pixel(u, v)[0], places=9)

    def test_height_from_the_lane_ratio_is_clamped(self):
        self.assertAlmostEqual(camera_geom.height_from_ratio(2.5), 1.4)
        self.assertEqual(camera_geom.height_from_ratio(10.0), camera_geom.H_RANGE[0])
        self.assertEqual(camera_geom.height_from_ratio(None), 1.3)

    def test_top_view_puts_a_ground_point_where_it_belongs(self):
        Hm, size = camera_geom.bev_homography(CAM, (-10.0, 10.0), (5.0, 25.0), 0.1)
        i, j = (2.0 - (-10.0)) / 0.1, (25.0 - 12.0) / 0.1                 # X = 2 m right, Z = 12 m ahead
        p = Hm @ np.array([i, j, 1.0])
        u, v = CAM.pixel_from_ground(2.0, 12.0)
        self.assertAlmostEqual(p[0] / p[2], u, places=6)
        self.assertAlmostEqual(p[1] / p[2], v, places=6)

    def test_travel_integrates_the_speed(self):
        s = camera_geom.travel([(0.0, 10.0), (2.0, 10.0), (4.0, 20.0)], 1.0)
        self.assertAlmostEqual(s(1.0), 0.0)
        self.assertAlmostEqual(s(2.0), 10.0)
        self.assertAlmostEqual(s(4.0), 10.0 + 30.0)
        self.assertAlmostEqual(s(5.0), 40.0 + 20.0)                         # constant speed after the series

    def test_road_curve_offset_on_a_left_bend(self):
        R = 200.0
        arc = [[R * math.sin(a), R * (1 - math.cos(a))] for a in np.linspace(0.0, 1.2, 241)]
        curve = camera_geom.RoadCurve(arc)
        for Z in (10.0, 20.0, 30.0):
            self.assertAlmostEqual(float(curve.offset(50.0, Z)), Z * Z / (2 * R), delta=0.05 * Z * Z / (2 * R) + 0.01)
        self.assertEqual(float(camera_geom.RoadCurve(None).offset(10.0, 20.0)), 0.0)


class TestVehicleTracks(unittest.TestCase):
    def test_box_bottom_is_the_ground_contact_and_the_centre_is_half_a_car_further(self):
        u, v = CAM.pixel_from_ground(-3.5, 15.0)
        X, Z = vehicle_tracks.ground_point(CAM, [u - 40, v - 50, u + 40, v], "car")
        self.assertAlmostEqual(Z, 15.0 + vehicle_tracks.HALF_LEN["car"], places=6)
        self.assertAlmostEqual(X, -3.5 * Z / 15.0, places=6)
        self.assertIsNone(vehicle_tracks.ground_point(CAM, [u - 40, v - 50, u + 40, 400], "car", y_bottom=400))   # cut by the hood

    def test_kinds(self):
        t = [0.0, 1.0, 2.0]
        self.assertEqual(vehicle_tracks.classify(t, [10, 25, 40], [3.5, 3.5, 3.6], None), "same")
        self.assertEqual(vehicle_tracks.classify(t, [60, 45, 30], [-7, -7, -7], None), "oncoming")
        self.assertEqual(vehicle_tracks.classify(t, [30, 30.5, 31], [6, 6, 6], None), "parked")
        self.assertEqual(vehicle_tracks.classify(t, [20, 20.5, 21], [8, 3, -2], None), "crossing")

    def test_crash_tracks_with_and_without_the_ego_car(self):
        def tr(i, z, d):
            ts = [round(4.0 + 0.1 * k, 2) for k in range(21)]
            return {"id": i, "t": ts, "s": [50.0 + z] * 21, "z": [z] * 21, "d": [d] * 21}
        tracks = [tr(1, 3.0, 2.0), tr(2, 20.0, 3.5), tr(3, 21.0, 6.0)]
        self.assertEqual(vehicle_tracks.crash_tracks(tracks, 5.0, ego_involved=True), [1])
        self.assertEqual(sorted(vehicle_tracks.crash_tracks(tracks, 5.0, ego_involved=False)), [1, 2] if False else [2, 3])


def synthetic_grid(lanes_left, lanes_right, w=3.5, s_max=60.0, cars_hide=()):
    """A lane map seen everywhere: lines between lanes, road over the lanes, grass beyond; cars_hide: (d0, d1) not seen."""
    g = bev_lanes.LaneGrid(s_max)
    d = np.array([g.d_of(j) for j in range(g.nd)])
    left_edge, right_edge = (lanes_left + 0.5) * w, -(lanes_right + 0.5) * w
    road = (d <= left_edge) & (d >= right_edge)
    edges = (np.abs(d - left_edge) < 0.25) | (np.abs(d - right_edge) < 0.25)          # solid edge lines
    inner = np.zeros(g.nd, bool)                                                    # dashed lines between lanes
    for k in range(-lanes_right, lanes_left):
        inner |= np.abs(d - (k + 0.5) * w) < 0.25
    inner &= ~edges
    seen = np.ones(g.nd, bool)
    for a, b in cars_hide:
        seen &= ~((d >= a) & (d <= b))
    dash_rows = (np.arange(g.ns) % 6) < 2                                           # 1 m of dash in every 3 m
    g.seen[:] = 10.0 * seen
    g.seen_r[:] = 10.0 * seen
    g.line[:] = 10.0 * ((edges[None, :] | (inner[None, :] & dash_rows[:, None])) & seen[None, :])
    g.road[:] = 10.0 * (road & seen)
    return g


class TestLaneMap(unittest.TestCase):
    def test_counts_lanes_from_lines_and_road(self):
        g = synthetic_grid(2, 1)
        nl, nr, det = bev_lanes.count_slice(g, 30.0, 3.5)
        self.assertEqual((nl, nr), (2, 1))

    def test_lane_width_from_the_ego_lane(self):
        g = synthetic_grid(2, 0, w=3.7)
        self.assertAlmostEqual(bev_lanes.lane_width(g, [10.0, 20.0, 30.0, 40.0, 50.0]), 3.7, delta=0.25)

    def test_cars_show_lanes_whose_road_they_hide(self):
        g = synthetic_grid(0, 0, cars_hide=[(2.0, 12.0)])                 # left of the ego lane: hidden by a queue
        self.assertEqual(bev_lanes.count_slice(g, 30.0, 3.5)[0], 0)
        self.assertEqual(bev_lanes.count_slice(g, 30.0, 3.5, vehicles_d=[3.6, 7.1])[0], 2)

    def test_cars_on_ground_seen_as_off_road_do_not_count(self):
        g = synthetic_grid(1, 0)
        self.assertTrue(bev_lanes.seen_off_road(g, 30.0, -12.0))          # beyond the road edge: seen, not road
        self.assertFalse(bev_lanes.seen_off_road(g, 30.0, 3.5))

    def test_a_lane_seen_is_carried_where_nothing_is_seen_but_not_onto_seen_grass(self):
        g = synthetic_grid(2, 0, s_max=120.0)
        rows = [{"t": float(s), "lanes_left": 2.0 if s < 40 else 0.0, "lanes_right": 0.0, "detail": {}} for s in range(0, 100, 5)]
        g.seen[int((50 - g.s0) / bev_lanes.S_RES):] = 0.0                 # nothing seen beyond 50 m
        g.seen_r[int((50 - g.s0) / bev_lanes.S_RES):] = 0.0
        bev_lanes.carry_lanes(g, rows, 3.5)
        by_s = {r["t"]: r["lanes_left"] for r in rows}
        self.assertEqual(by_s[60.0], 2.0)                                  # carried into the unseen stretch
        self.assertEqual(by_s[95.0], 0.0)                                  # but not further than CARRY_M
        g2 = synthetic_grid(0, 0, s_max=120.0)                             # grass seen left of the ego lane everywhere
        rows2 = [{"t": float(s), "lanes_left": 2.0 if s < 20 else 0.0, "lanes_right": 0.0, "detail": {}} for s in range(0, 60, 5)]
        bev_lanes.carry_lanes(g2, rows2, 3.5)
        self.assertEqual(rows2[-1]["lanes_left"], 0.0)


def spec_with(actors=(), event=None, behavior="cruise", speed=50.0):
    spec, _ = scene_spec.normalize({"road": {"type": "straight", "lanes_per_direction": 2},
                                    "ego": {"speed_kmh": speed, "behavior": behavior},
                                    "actors": list(actors), "event": event or {"type": "none"}})
    return spec


class TestPipelineScene(unittest.TestCase):
    def scene(self, tracks=(), speeds=None, source="overlay"):
        return {"tracks": list(tracks), "speeds": speeds or [[0.0, 20.0], [20.0, 10.0]], "speed_source": source,
                "t_v0": 1.0, "t_hi": 14.0, "lane_samples": []}

    def test_ego_speed_profile_and_mean_speed_until_the_event(self):
        spec = spec_with([{"id": "c1", "relation": "ahead", "behavior": "braking"}], {"type": "collision", "actor_id": "c1", "time_s": 9.0})
        local_pipeline.apply_ego_speed(spec, self.scene())
        self.assertEqual(spec["ego"]["speed_profile"][0][0], 0.0)
        # speed falls linearly from 20 m/s at 0 s to 10 m/s at 20 s: mean over video 1..9 s = 17.5 m/s
        self.assertAlmostEqual(spec["ego"]["speed_kmh"], 17.5 * 3.6, delta=0.2)
        stopped = spec_with(behavior="stopped", speed=0.0)
        local_pipeline.apply_ego_speed(stopped, self.scene())
        self.assertIsNone(stopped["ego"].get("speed_profile"))

    def test_replay_drops_guessed_background_and_the_crash_vehicle(self):
        actors = [{"id": "c1", "relation": "beside", "lane": "left", "behavior": "cutting_in"},
                  {"id": "b1", "relation": "ahead", "lane": "left", "behavior": "driving"}]
        spec = spec_with(actors, {"type": "collision", "actor_id": "c1", "time_s": 9.0})
        spec["traffic"]["left_lane"] = 3
        ts = [round(7.0 + 0.1 * k, 2) for k in range(41)]
        crash = {"id": 5, "type": "car", "color": "white", "kind": "same", "t": ts, "s": [100.0] * 41, "z": [2.0] * 41,
                 "d": [1.0] * 41, "v": [15.0] * 41, "first_t": 7.0, "last_t": 11.0}
        other = dict(crash, id=6, z=[25.0] * 41, d=[3.5] * 41, color="red")
        local_pipeline.apply_replay(spec, self.scene([crash, other]), 9.0)
        self.assertEqual([r["id"] for r in spec["replay"]], ["r6"])
        self.assertEqual([a["id"] for a in spec["actors"]], ["c1"])
        self.assertEqual(spec["traffic"]["left_lane"], 0)
        self.assertEqual(spec["replay"][0]["path"][0][0], 6.0)            # video 7 s = 6 s after the simulation starts
        checked, _ = scene_spec.normalize(spec)
        self.assertEqual(len(checked["replay"]), 1)

    def test_lane_profile_from_the_map_in_metres(self):
        from agents import lane_lines
        samples = []
        for s in range(0, 200, 2):
            nl = 4.0 if s < 100 else 2.0
            samples.append({"t": float(s), "lanes_left": nl, "lanes_right": 0.0, "road_lanes": 1 + nl,
                            "probs": lane_lines.count_probs(1 + nl)})
        scene = self.scene(speeds=[[0.0, 15.0], [20.0, 15.0]])
        scene["lane_samples"] = samples
        spec = spec_with()
        probe = lambda video, t, side, dur: (0.3, 0.65) if side == "left" else (0.2, 0.2)
        self.assertTrue(local_pipeline.lane_profile_from_map(spec, scene, "v.mp4", 20.0, probe, log_fn=lambda m: None))
        prof = spec["road"]["lane_profile"]
        self.assertEqual([(p["n"], p["m"]) for p in prof], [(5, 0), (3, 2)])
        self.assertAlmostEqual(prof[0]["d1"], 99.0, delta=2.0)
        self.assertEqual(prof[1]["branch"], {"side": "left", "rise_m": 6.0})
        self.assertEqual(spec["ego"]["lane_index"], 4)                    # the right-most of five


class TestReplayInBuilder(unittest.TestCase):
    def build(self, replay, profile=None):
        spec = spec_with(speed=54.0)
        spec["replay"] = replay
        if profile:
            spec["ego"]["speed_profile"] = profile
        tmp = tempfile.TemporaryDirectory()
        plan = scene_builder.build(spec, tmp.name, "rp")
        with open(os.path.join(tmp.name, "rp_scene_obj.txt"), encoding="utf-8") as f:
            agents = ast.literal_eval(f.readline())
        tmp.cleanup()
        return plan, agents

    def test_vehicle_first_seen_later_starts_where_its_speed_puts_it_and_follows_its_path(self):
        rep = [{"id": "r1", "type": "car", "color": "red", "kind": "same",
                "path": [[2.0, 60.0, 3.5], [3.0, 70.0, 3.5], [4.0, 80.0, 3.6]], "speed": [[2.0, 36.0], [4.0, 36.0]]}]
        plan, agents = self.build(rep)
        vid = plan["idmap"]["r1"]
        fr = RoadFrame(plan["road"]["centerline"])
        s, d = fr.to_sd(agents[vid]["location"][0], -agents[vid]["location"][1])
        self.assertAlmostEqual(s, scene_builder.S_START + 40.0, delta=0.05)   # 60 m at 2 s, 10 m/s -> 40 m at 0 s
        self.assertAlmostEqual(d, lane_offset(2, 0) + 3.5, delta=0.05)
        p = plan["agents"][vid]
        self.assertEqual(p["mode"], "drive")
        self.assertAlmostEqual(p["speed_kmh"], 36.0, delta=0.1)
        self.assertEqual(p["speed_profile"][0], [0.0, 36.0])
        end = fr.to_sd(*p["polyline"][-1])
        self.assertGreater(end[0], fr.length - 20.0)                     # keeps its lane to the end of the road
        self.assertEqual(p["color"], scene_spec.color_rgb("red"))

    def test_parked_vehicle_stands_and_a_vehicle_on_the_ego_car_is_skipped(self):
        rep = [{"id": "r1", "type": "truck", "color": None, "kind": "parked", "path": [[0.0, 30.0, -4.0], [3.0, 30.2, -4.0]], "speed": []},
               {"id": "r2", "type": "car", "color": None, "kind": "same", "path": [[0.0, 1.0, 0.0], [1.0, 15.0, 0.0]], "speed": []}]
        plan, agents = self.build(rep)
        self.assertEqual(plan["agents"][plan["idmap"]["r1"]]["mode"], "static")
        self.assertNotIn("r2", plan["idmap"])
        self.assertTrue(any("r2" in n for n in plan["notes"]))

    def test_ego_gets_the_measured_speed_profile(self):
        plan, _ = self.build([], profile=[[0.0, 60.0], [5.0, 40.0]])
        self.assertEqual(plan["agents"]["v1"]["speed_profile"], [[0.0, 60.0], [5.0, 40.0]])

    def test_runner_profile_speed(self):
        prof = [[0.0, 60.0], [5.0, 40.0]]
        self.assertEqual(motion_runner.profile_speed(prof, -1.0), 60.0)
        self.assertAlmostEqual(motion_runner.profile_speed(prof, 2.5), 50.0)
        self.assertEqual(motion_runner.profile_speed(prof, 9.0), 40.0)


if __name__ == "__main__":
    unittest.main()


class TestDrivableMotion(unittest.TestCase):
    def test_speed_smoothing_drops_a_misread_value_and_keeps_acceleration_comfortable(self):
        from agents import traj_smoothing as ts
        t = [0.5 * i for i in range(14)]
        kmh = [80.5, 70.8, 70.8, 70.8, 66, 66, 66, 40.2, 62.7, 62.7, 59.5, 59.5, 59.5, 56.3]
        v = ts.smooth_speed_series(t, [x / 3.6 for x in kmh])
        self.assertGreater(min(v) * 3.6, 55.0)                           # the 40 km/h misreading is gone
        acc = np.diff(v) / 0.5
        self.assertLessEqual(acc.max(), ts.COMFORT["max_accel"] + 1e-6)
        self.assertGreaterEqual(acc.min(), -ts.COMFORT["max_decel"] - 1e-6)

    def test_vehicles_off_the_scene_roads_are_not_replayed(self):
        road = scene_builder.Road("straight", 2)
        self.assertTrue(scene_builder._on_carriageway(road, 50.0, -5.0, "same"))
        self.assertFalse(scene_builder._on_carriageway(road, 50.0, -14.0, "same"))     # frontage road beyond the barrier
        self.assertFalse(scene_builder._on_carriageway(road, 50.0, 3.0, "same"))       # that is the opposite carriageway
        self.assertTrue(scene_builder._on_carriageway(road, 50.0, 3.0, "oncoming"))
        self.assertTrue(scene_builder._on_carriageway(road, 50.0, -9.5, "parked"))     # on the shoulder


class TestForkGeometry(unittest.TestCase):
    def test_branch_stays_level_until_it_has_pulled_away_and_has_no_inner_shoulder(self):
        import re
        from test_lane_profile import LEFT_FLYOVER, base_spec, build
        tmp, plan, agents, net = build(base_spec(LEFT_FLYOVER, lane_index=4))
        s_b = plan["road"]["sections"][1]["s0"]
        fr = RoadFrame(plan["road"]["centerline"])
        pts = re.search(r'<lane id="branch_0"[^>]*shape="([^"]+)"', net).group(1).split()
        for p in pts:
            v = [float(x) for x in p.split(",")]
            s, d = fr.to_sd(v[0], v[1])
            z = v[2] if len(v) > 2 else 0.0
            if s < s_b + scene_builder.RISE_START_M - 1.0:
                self.assertAlmostEqual(z, 0.0, places=2)
        self.assertEqual(plan["road"]["no_shoulder"], ["branch"])


class TestDashFocal(unittest.TestCase):
    def test_dash_pattern_shift_between_frames(self):
        z = np.arange(400) * camera_geom.DASH_RES                    # 20 m of road
        dash = lambda off: ((z + off) % 12.0 < 3.0).astype(float) * 40.0 + 2.0   # 3 m dashes every 12 m
        a = dash(0.0)
        b = np.concatenate([np.full(17, 2.0), a[:-17]])              # marks moved 17 rows toward the camera
        k, r = camera_geom.dash_shift(a, b, 80)
        self.assertEqual(k, 17)
        self.assertGreater(r, 0.95)


class TestLaneSnapping(unittest.TestCase):
    def test_replayed_positions_snap_to_lanes_that_exist(self):
        road = scene_builder.Road("straight", 3, None, False,
                                  [{"s0": 0, "s1": 100, "n": 3, "m": 0}, {"s0": 100, "s1": 300, "n": 2, "m": 1}])
        self.assertAlmostEqual(scene_builder.snap_lane(road, 50.0, -4.0), lane_offset(3, 1))      # between lanes -> nearest
        self.assertAlmostEqual(scene_builder.snap_lane(road, 150.0, -1.8), lane_offset(3, 1))     # left lane ended: next one
        self.assertAlmostEqual(scene_builder.snap_lane(road, 150.0, 4.0, oncoming=True), -lane_offset(3, 1))

    def test_branch_lanes_count_after_a_fork(self):
        from test_lane_profile import LEFT_FLYOVER
        secs = [{"s0": 0.0, "s1": 75.0, "n": 5, "m": 0}, {"s0": 75.0, "s1": 300.0, "n": 3, "m": 2, "branch": {"side": "left", "rise_m": 6.0}}]
        road = scene_builder.Road("straight", 3, None, False, secs)
        far = 75.0 + scene_builder.DIVERGE_LEN_M + 10.0
        c = scene_builder.lane_centres(road, far)
        self.assertEqual(len(c), 5)                                    # three ego-road lanes and the two branch lanes
        self.assertAlmostEqual(max(c), -0.5 * 3.5 + scene_builder.DIVERGE_D_M, places=6)


def drift_track(i, d0, slope, s0=90.0, n=30, ratio=None, kind="same"):
    s = [s0 + 1.0 * k for k in range(n)]
    return {"id": i, "kind": kind, "type": "car", "s": s, "d": [d0 + slope * (x - s0) for x in s], "elev_ratio": ratio,
            "t": [0.1 * k for k in range(n)]}


class TestForkFromTraffic(unittest.TestCase):
    def test_elevation_ratio_is_one_on_flat_ground_and_grows_on_a_rise(self):
        Z = 20.0
        u, v = CAM.pixel_from_ground(-3.0, Z)
        h = CAM.f * vehicle_tracks.CAR_HEIGHT_M / Z
        self.assertAlmostEqual(vehicle_tracks.elevation_ratio(CAM, [u - 30, v - h, u + 30, v], "car"), 1.0, places=6)
        v_up = CAM.vy + CAM.f * (CAM.H - 0.5) / Z                          # the same car 0.5 m higher up
        self.assertAlmostEqual(vehicle_tracks.elevation_ratio(CAM, [u - 30, v_up - h, u + 30, v_up], "car"),
                               CAM.H / (CAM.H - 0.5), places=6)
        self.assertIsNone(vehicle_tracks.elevation_ratio(CAM, [u - 30, v - h, u + 30, v], "truck"))

    def test_cars_pulling_away_on_one_side_make_a_fork(self):
        tracks = [drift_track(1, 8.0, 0.25, s0=100.0), drift_track(2, 15.0, 0.2, s0=110.0), drift_track(3, 4.0, 0.0),
                  drift_track(4, -4.0, -0.01)]
        fork = local_pipeline.fork_from_tracks(tracks, 3.5)
        self.assertEqual(fork["side"], "left")
        self.assertAlmostEqual(fork["s_split"], 105.0)
        self.assertEqual(sorted(fork["ids"]), [1, 2])
        self.assertIsNone(local_pipeline.fork_from_tracks(tracks[:1] + tracks[2:], 3.5))   # one car is not a fork

    def test_branch_climbs_when_its_cars_read_higher_than_the_ego_road(self):
        base = [drift_track(10 + k, 3.5, 0.0, ratio=1.0) for k in range(4)]
        high = [drift_track(1, 12.0, 0.2, ratio=1.5), drift_track(2, 13.0, 0.2, ratio=1.3)]
        self.assertTrue(local_pipeline.branch_climbs(base + high, "left", 9.0)[0])
        flat = [drift_track(1, 12.0, 0.2, ratio=1.05), drift_track(2, 13.0, 0.2, ratio=1.1)]
        self.assertFalse(local_pipeline.branch_climbs(base + flat, "left", 9.0)[0])

    def test_fork_moves_the_change_to_the_split_and_makes_a_branch(self):
        prof = [{"d0": None, "d1": 60.0, "n": 4, "m": 0}, {"d0": 60.0, "d1": None, "n": 3, "m": 1}]
        tracks = [drift_track(1, 12.0, 0.25, s0=100.0, ratio=1.5), drift_track(2, 14.0, 0.2, s0=110.0, ratio=1.4)] + \
                 [drift_track(10 + k, 3.5, 0.0, ratio=1.0) for k in range(4)]
        fork = local_pipeline.fork_from_tracks(tracks, 3.5)
        note = local_pipeline.apply_fork(prof, fork, tracks, 3.5, ego_j=3)
        self.assertAlmostEqual(prof[1]["d0"], 105.0)
        self.assertEqual(prof[0]["d1"], prof[1]["d0"])
        self.assertEqual(prof[1]["branch"], {"side": "left", "rise_m": 6.0})
        self.assertIn("climbs", note)


class TestReplayPathDensity(unittest.TestCase):
    def test_replayed_path_follows_a_bend_instead_of_cutting_it(self):
        import math as m
        R = 190.0
        arc = [[R * m.sin(a), -R * (1 - m.cos(a))] for a in np.linspace(0.0, 1.5, 301)]      # bends right
        spec = spec_with(speed=54.0)
        spec["road"]["centerline"] = arc
        spec["replay"] = [{"id": "r1", "type": "car", "color": None, "kind": "same",
                           "path": [[3.0, 120.0, 3.5], [4.0, 130.0, 3.5]], "speed": [[3.0, 36.0], [4.0, 36.0]]}]
        tmp = tempfile.TemporaryDirectory()
        plan = scene_builder.build(spec, tmp.name, "bend")
        tmp.cleanup()
        fr = RoadFrame(plan["road"]["centerline"])
        poly = plan["agents"][plan["idmap"]["r1"]]["polyline"]
        gaps = [m.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(poly, poly[1:])]
        self.assertLessEqual(max(gaps), 4.5)                             # no long chords
        ds = [fr.to_sd(*q)[1] for q in poly]
        self.assertLess(max(ds) - min(ds), 0.2)                          # stays in its lane through the bend


class TestSceneFixes(unittest.TestCase):
    def test_split_branch_road_extends_past_incline_and_exports_to_xodr(self):
        from agents import map_builder
        from test_lane_profile import LEFT_FLYOVER, base_spec, build
        spec = base_spec(LEFT_FLYOVER, lane_index=4)
        tmp, plan, agents, net = build(spec)
        self.addCleanup(tmp.cleanup)
        fr = RoadFrame(plan["road"]["centerline"])
        self.assertGreaterEqual(fr.length, 550.0)
        xodr = map_builder.net_to_xodr(os.path.join(tmp.name, "lp.net.xml"), os.path.join(tmp.name, "lp.xodr"))
        self.assertTrue(os.path.exists(xodr))

    def test_on_carriageway_includes_diverging_branch_lanes(self):
        secs = [{"s0": 0.0, "s1": 75.0, "n": 4, "m": 0},
                {"s0": 75.0, "s1": 550.0, "n": 3, "m": 1, "branch": {"side": "left", "rise_m": 6.0}}]
        road = scene_builder.Road("straight", 4, None, False, secs, min_length=550.0)
        far_s = 75.0 + scene_builder.DIVERGE_LEN_M + 20.0
        branch_lat = -0.5 * 3.5 + scene_builder.DIVERGE_D_M
        self.assertTrue(scene_builder._on_carriageway(road, far_s, branch_lat, "same"))

    def test_replay_defers_negative_start_snaps_parked_and_smooths_lane_change(self):
        spec = spec_with(speed=60.0)
        spec["road"]["lanes_per_direction"] = 2
        spec["road"]["centerline"] = [[0.0, 0.0], [500.0, 0.0]]
        # r1 starts behind road start at t=0 (s=5 at t=3 with v=10 m/s -> s0=-25, road start -10 < 3), enters later via spawn_at_s
        # r2 is parked inside the oncoming lane (lat=+1.75) -> snapped to lane centre
        # r3 switches lanes noisily -> ramped smoothly without instantaneous jumps > 1.0 m per 4 m step
        spec["replay"] = [
            {"id": "r1", "type": "car", "color": "blue", "kind": "same",
             "path": [[3.0, 5.0, 3.5], [4.0, 15.0, 3.5]], "speed": [[3.0, 36.0], [4.0, 36.0]]},
            {"id": "r2", "type": "car", "color": "black", "kind": "parked",
             "path": [[0.0, 40.0, 6.2], [2.0, 40.2, 6.3]], "speed": []},
            {"id": "r3", "type": "car", "color": "white", "kind": "same",
             "path": [[0.0, 60.0, 0.0], [1.0, 70.0, 0.0], [2.0, 80.0, 3.4], [3.0, 90.0, 3.5]],
             "speed": [[0.0, 36.0], [3.0, 36.0]]},
        ]
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        plan = scene_builder.build(spec, tmp.name, "fx")
        with open(os.path.join(tmp.name, "fx_scene_obj.txt"), encoding="utf-8") as f:
            agents = ast.literal_eval(f.readline())
        v_r1 = plan["idmap"]["r1"]
        self.assertGreater(plan["agents"][v_r1].get("spawn_at_s", 0.0), 0.0)
        v_r2 = plan["idmap"]["r2"]
        self.assertAlmostEqual(-agents[v_r2]["location"][1], -lane_offset(2, 1), delta=0.1)
        v_r3 = plan["idmap"]["r3"]
        fr = RoadFrame(plan["road"]["centerline"])
        ds = [fr.to_sd(*p)[1] for p in plan["agents"][v_r3]["polyline"]]
        max_step = max(abs(b - a) for a, b in zip(ds, ds[1:]))
        self.assertLess(max_step, 1.2)

    def test_apply_replay_marks_slow_overtake_and_reclassifies_false_oncoming(self):
        spec = spec_with([{"id": "c1", "relation": "beside", "lane": "left", "behavior": "cutting_in", "color": "white"}],
                         {"type": "collision", "actor_id": "c1", "time_s": 8.0}, speed=65.0)
        spec["road"]["lanes_per_direction"] = 3
        spec["ego"]["lane"] = "right"
        ts = [round(7.0 + 0.1 * k, 2) for k in range(15)]
        crash = {"id": 76, "type": "car", "color": "white", "kind": "same", "t": ts,
                 "s": [140.0 + 5.0 * (t - 7.0) for t in ts], "z": [10.0 - 12.0 * (t - 7.0) for t in ts],
                 "d": [3.5] * len(ts), "v": [5.0] * len(ts), "first_t": 7.0, "last_t": 8.4}
        false_onc = {"id": 99, "type": "car", "color": "gray", "kind": "oncoming", "t": ts,
                     "s": [200.0 - 5.0 * (t - 7.0) for t in ts], "z": [30.0] * len(ts),
                     "d": [5.0] * len(ts), "v": [-5.0] * len(ts), "first_t": 7.0, "last_t": 8.4}
        slower = {"id": 31, "type": "car", "color": "brown", "kind": "same", "t": [4.0 + 0.1 * k for k in range(20)],
                  "s": [100.0 + 6.0 * 0.1 * k for k in range(20)], "z": [25.0] * 20,
                  "d": [3.5] * 20, "v": [6.0] * 20, "first_t": 4.0, "last_t": 5.9}
        scene = {"tracks": [crash, false_onc, slower], "speeds": [[0.0, 20.0], [20.0, 10.0]],
                 "speed_source": "overlay", "t_v0": 1.0, "t_hi": 14.0, "lane_samples": []}
        local_pipeline.apply_replay(spec, scene, 8.0)
        self.assertTrue(spec["actors"][0].get("measured", {}).get("overtake"))
        by_id = {r["id"]: r for r in spec["replay"]}
        self.assertEqual(by_id["r99"]["kind"], "parked")
        checked, _ = scene_spec.normalize(spec)
        self.assertTrue(checked["actors"][0]["measured"]["overtake"])
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        plan = scene_builder.build(checked, tmp.name, "ov")
        self.assertIn("cruise_speed_ms", plan["agents"][plan["idmap"]["c1"]]["intercept"])
        self.assertIn("r31", plan["idmap"])

    def test_lane_profile_from_map_drops_short_leading_fov_blindspot_section(self):
        from agents import lane_lines
        samples = []
        for s in range(0, 200, 2):
            nl = 0.0 if s < 16 else (3.0 if s < 105 else 2.0)
            samples.append({"t": float(s), "lanes_left": nl, "lanes_right": 0.0, "road_lanes": 1 + nl,
                            "probs": lane_lines.count_probs(1 + nl)})
        scene = {"tracks": [], "speeds": [[0.0, 15.0], [20.0, 15.0]], "speed_source": "overlay",
                 "t_v0": 1.0, "t_hi": 14.0, "lane_samples": samples}
        spec = spec_with()
        probe = lambda video, t, side, dur: (0.3, 0.65) if side == "left" else (0.2, 0.2)
        self.assertTrue(local_pipeline.lane_profile_from_map(spec, scene, "v.mp4", 20.0, probe, log_fn=lambda m: None))
        prof = spec["road"]["lane_profile"]
        self.assertEqual([(p["n"], p["m"]) for p in prof], [(4, 0), (3, 1)])
        self.assertEqual(prof[1]["branch"], {"side": "left", "rise_m": 6.0})

    def test_parked_replay_is_not_back_extrapolated_and_late_queue_cars_slot_cleanly(self):
        spec = spec_with(speed=60.0)
        spec["road"]["lanes_per_direction"] = 2
        spec["road"]["centerline"] = [[0.0, 0.0], [500.0, 0.0]]
        spec["replay"] = [
            {"id": "rp", "type": "car", "color": "gray", "kind": "parked",
             "path": [[12.0, 220.0, 3.5], [13.0, 215.0, 3.5]], "speed": [[12.0, 18.0], [13.0, 18.0]]},
            {"id": "rq1", "type": "car", "color": "white", "kind": "same",
             "path": [[4.0, 100.0, 3.5], [5.0, 108.0, 3.5]], "speed": [[4.0, 28.8], [5.0, 28.8]]},
            {"id": "rq2", "type": "car", "color": "brown", "kind": "same",
             "path": [[4.0, 101.0, 3.5], [5.0, 109.0, 3.5]], "speed": [[4.0, 28.8], [5.0, 28.8]]},
        ]
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        plan = scene_builder.build(spec, tmp.name, "nq")
        with open(os.path.join(tmp.name, "nq_scene_obj.txt"), encoding="utf-8") as f:
            agents = ast.literal_eval(f.readline())
        self.assertIn("rp", plan["idmap"])
        self.assertIn("rq1", plan["idmap"])
        self.assertIn("rq2", plan["idmap"])
        self.assertAlmostEqual(agents[plan["idmap"]["rp"]]["location"][0], scene_builder.S_START + 220.0, delta=0.5)
        x1 = agents[plan["idmap"]["rq1"]]["location"][0]
        x2 = agents[plan["idmap"]["rq2"]]["location"][0]
        self.assertGreaterEqual(abs(x2 - x1), 7.0)

    def test_enforce_replay_headway_prevents_truck_car_and_parked_overlaps(self):
        ts = [0.0, 0.5, 1.0, 1.5, 2.0]
        tracks = [
            {"id": "lead_car", "type": "car", "back": False, "ts": ts,
             "ss": [50.0, 52.0, 54.0, 56.0, 58.0], "ds": [-3.5] * 5},
            {"id": "fast_truck", "type": "truck", "back": False, "ts": ts,
             "ss": [42.0, 48.0, 54.0, 60.0, 66.0], "ds": [-3.5] * 5},
            {"id": "car_vs_parked", "type": "car", "back": False, "ts": ts,
             "ss": [20.0, 26.0, 32.0, 38.0, 44.0], "ds": [-7.0] * 5},
        ]
        static_obs = [(35.0, -7.0, "car")]
        scene_builder.enforce_replay_headway(tracks, static_obstacles=static_obs, gap_margin=3.5)
        truck_need = scene_builder.VEHICLE_HALF_LEN["truck"] + scene_builder.VEHICLE_HALF_LEN["car"] + 3.5
        for ls, ts_ in zip(tracks[0]["ss"], tracks[1]["ss"]):
            self.assertGreaterEqual(ls - ts_, truck_need - 1e-6)
        parked_need = scene_builder.VEHICLE_HALF_LEN["car"] * 2 + 3.5
        for cs in tracks[2]["ss"]:
            self.assertLessEqual(cs, 35.0 - parked_need + 1e-6)

    def test_kinematic_track_emitted_and_corridors_protect_third_party(self):
        spec = {
            "road": {"type": "straight", "lanes_per_direction": 2, "traffic_signals": False, "one_way": False,
                     "centerline": [[0.0, 0.0], [500.0, 0.0]]},
            "ego": {"type": "car", "lane": "left", "speed_kmh": 0.0, "behavior": "stopped"},
            "actors": [
                {"id": "c1", "type": "car", "relation": "cross_left", "lane": "same", "distance_m": 7.0,
                 "speed_kmh": 20.0, "behavior": "crossing", "color": "black"},
                {"id": "c2", "type": "van", "relation": "behind", "lane": "right", "distance_m": 9.0,
                 "speed_kmh": 50.0, "behavior": "driving", "color": "brown"},
            ],
            "event": {"type": "collision", "actor_id": "c1", "actor2_id": "c2", "time_s": 14.2, "description": "tp"},
            "replay": [
                {"id": "blocker", "type": "car", "color": "white", "kind": "parked",
                 "path": [[0.0, 7.0, 0.0], [1.0, 7.0, 0.0]], "speed": []},
                {"id": "free_onc", "type": "car", "color": "gray", "kind": "oncoming",
                 "path": [[0.0, 90.0, 5.25], [2.0, 70.0, 5.25]], "speed": [[0.0, 36.0], [2.0, 36.0]]},
            ],
        }
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        plan = scene_builder.build(spec, tmp.name, "tpk")
        self.assertNotIn("blocker", plan["idmap"])
        self.assertIn("free_onc", plan["idmap"])
        kt = plan["agents"][plan["idmap"]["free_onc"]].get("kinematic_track")
        self.assertIsNotNone(kt)
        self.assertEqual(len(kt[0]), 5)


class TestMultiLaneCutInAndParts(unittest.TestCase):
    def test_ground_point_recovers_side_entering_car_below_hood_row_and_keeps_center_rejected(self):
        # Side-entering white car in far-left pixels (xc < 0.20*w) stays valid while center hood-clipped box stays None
        self.assertIsNotNone(vehicle_tracks.ground_point(CAM, [0, 250, 130, 400], "car", y_bottom=356))
        self.assertIsNone(vehicle_tracks.ground_point(CAM, [420, 250, 540, 400], "car", y_bottom=356))
        self.assertIsNone(vehicle_tracks.ground_point(CAM, [10, 240, 950, 520], "car", y_bottom=356))

    def test_converging_cutin_picks_far_left_cutter_and_stitches_predecessor(self):
        # Straight middle-lane SUV closer at d=3.7..4.0 should not beat the converging far-left cutter d=6.1..3.6
        t_suv = [7.5 + 0.1 * i for i in range(9)]
        suv_mid = {"id": 76, "type": "car", "color": "white", "kind": "same", "t": t_suv,
                   "s": [140 + 6.0 * 0.1 * i for i in range(9)], "z": [28.0 - 0.9 * i for i in range(9)],
                   "d": [3.7 + 0.04 * i for i in range(9)], "v": [6.0] * 9, "first_t": 7.5, "last_t": 8.3}
        t_prev = [7.8 + 0.1 * i for i in range(6)]
        cutter_prev = {"id": 79, "type": "car", "color": "white", "kind": "same", "t": t_prev,
                       "s": [100 + 6.0 * 0.1 * i for i in range(6)], "z": [16.9 - 0.8 * i for i in range(6)],
                       "d": [6.7 + 0.1 * i for i in range(6)], "v": [6.0] * 6, "first_t": 7.8, "last_t": 8.3}
        t_late = [8.9 + 0.1 * i for i in range(11)]
        cutter_late = {"id": 85, "type": "car", "color": "white", "kind": "same", "t": t_late,
                       "s": [110 + 10.5 * 0.1 * i for i in range(11)], "z": [7.2 - 0.1 * i for i in range(11)],
                       "d": [4.8, 5.1, 5.3, 5.4, 5.35, 5.2, 5.0, 4.85, 4.7, 4.6, 4.5],
                       "v": [11.5] * 11, "first_t": 8.9, "last_t": 9.9}
        res = vehicle_tracks.converging_cutin_track([suv_mid, cutter_prev, cutter_late], 8.6, side="left", color="white")
        self.assertIsNotNone(res)
        self.assertEqual(res["skip_ids"], [79, 85])
        self.assertEqual(res["lane_offset"], 2)

        spec = spec_with([{"id": "c1", "relation": "beside", "lane": "left", "behavior": "cutting_in", "color": "white"}],
                         {"type": "collision", "actor_id": "c1", "time_s": 8.6,
                          "contact_parts": {"ego": "front_left", "c1": "right_side"}})
        spec["road"]["lanes_per_direction"] = 3
        spec["ego"]["lane"] = "right"
        scene = {"tracks": [suv_mid, cutter_prev, cutter_late], "speeds": [[0.0, 15.0], [15.0, 15.0]],
                 "speed_source": "overlay", "t_v0": 0.6, "t_hi": 13.6, "lane_samples": []}
        local_pipeline.apply_replay(spec, scene, 8.6)
        self.assertEqual(spec["actors"][0]["measured"]["lane_offset"], 2)
        self.assertEqual([r["id"] for r in spec["replay"]], ["r76"])

        # Add a middle-lane blocker during the sweep window and an early middle-lane car passing earlier
        spec["replay"].extend([
            {"id": "mid_blk", "type": "car", "color": "gray", "kind": "same",
             "path": [[6.9, 118.0, 3.5], [8.8, 128.0, 3.5]], "speed": [[6.9, 19.0], [8.8, 19.0]]},
            {"id": "mid_early", "type": "car", "color": "blue", "kind": "same",
             "path": [[0.5, 82.0, 3.5], [4.0, 89.0, 3.5]], "speed": [[0.5, 7.2], [4.0, 7.2]]},
        ])
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        plan = scene_builder.build(spec, tmp.name, "ml")
        ic = plan["agents"][plan["idmap"]["c1"]]["intercept"]
        self.assertAlmostEqual(ic["lane_src_d"], lane_offset(3, 2))
        self.assertAlmostEqual(ic["lane_dst_d"], lane_offset(3, 0))
        self.assertEqual(ic["max_lateral_speed_ms"], 3.2)
        self.assertGreaterEqual(ic["lead_s"], 2.4)
        self.assertEqual(plan["expect"]["contact_parts"], {"v1": "front_left", "v2": "right_side"})
        self.assertNotIn("mid_blk", plan["idmap"])
        self.assertIn("mid_early", plan["idmap"])
        self.assertIn("r76", plan["idmap"])

    def test_converging_cutin_syncs_vehicle_type_when_impact_time_shifts(self):
        # VLM picked an SUV at coarse t_crash=8.6s, but the converging cutter hits at 10.1s and is a white sedan ("car")
        t_late = [8.9 + 0.1 * i for i in range(11)]
        cutter = {"id": 64, "type": "car", "color": "white", "kind": "same", "t": t_late,
                  "s": [110 + 10.5 * 0.1 * i for i in range(11)], "z": [7.2 - 0.1 * i for i in range(11)],
                  "d": [5.8, 5.6, 5.4, 5.2, 5.0, 4.7, 4.4, 4.1, 3.8, 3.5, 3.2],
                  "v": [11.5] * 11, "first_t": 8.9, "last_t": 9.9}
        spec = spec_with([{"id": "c1", "type": "suv", "relation": "beside", "lane": "left",
                           "behavior": "cutting_in", "color": "240,240,240"}],
                         {"type": "collision", "actor_id": "c1", "time_s": 8.6})
        spec["road"]["lanes_per_direction"] = 3
        scene = {"tracks": [cutter], "speeds": [[0.0, 15.0], [15.0, 15.0]],
                 "speed_source": "overlay", "t_v0": 0.6, "t_hi": 13.6, "lane_samples": []}
        local_pipeline.apply_replay(spec, scene, 8.6)
        self.assertEqual(spec["actors"][0]["type"], "car")
        self.assertGreater(spec["event"]["time_s"], 9.5)

        # When VLM's t_crash is already close (<0.8s) to t_impact, keep VLM's finer "suv" class
        spec_close = spec_with([{"id": "c1", "type": "suv", "relation": "beside", "lane": "left",
                                 "behavior": "cutting_in", "color": "240,240,240"}],
                               {"type": "collision", "actor_id": "c1", "time_s": 9.8})
        spec_close["road"]["lanes_per_direction"] = 3
        local_pipeline.apply_replay(spec_close, scene, 9.8)
        self.assertEqual(spec_close["actors"][0]["type"], "suv")

    def test_enforce_replay_headway_no_teleport_when_lead_clears(self):
        ts = [round(0.2 * k, 2) for k in range(31)]  # 0.0 .. 6.0 s
        # Lead moves slowly at 4 m/s for first 2.6 s then leaves into adjacent lane (-3.5 -> 0.0)
        lead = {"id": "lead", "type": "car", "back": False, "ts": ts,
                "ss": [40.0 + 4.0 * t for t in ts],
                "ds": [-3.5 if t <= 2.6 else 0.0 for t in ts]}
        # Follower measured cruising at 20 m/s behind lead
        fol = {"id": "fol", "type": "car", "back": False, "ts": ts,
               "ss": [25.0 + 20.0 * t for t in ts], "ds": [-3.5] * len(ts)}
        scene_builder.enforce_replay_headway([lead, fol], gap_margin=3.5)
        for t0, t1, s0, s1 in zip(ts, ts[1:], fol["ss"], fol["ss"][1:]):
            dt = t1 - t0
            self.assertLessEqual((s1 - s0) / dt, 30.5)
        spds = [(s1 - s0) / (t1 - t0) for t0, t1, s0, s1 in zip(ts, ts[1:], fol["ss"], fol["ss"][1:])]
        for v0, v1, t0, t1 in zip(spds, spds[1:], ts, ts[1:]):
            acc = (v1 - v0) / (t1 - t0)
            self.assertLessEqual(acc, 3.6)

    def test_kinematic_track_has_smooth_heading_and_no_sideways_slide_when_stopped(self):
        spec = spec_with([{"id": "c1", "relation": "ahead", "lane": "same", "behavior": "braking"}],
                         {"type": "collision", "actor_id": "c1", "time_s": 5.0})
        spec["road"]["lanes_per_direction"] = 2
        spec["road"]["centerline"] = [[0.0, 0.0], [400.0, 0.0]]
        # Include a brief noisy rocket track (dur 0.3s, 38 m/s) in scene -> must be filtered by apply_replay
        rocket = {"id": 66, "type": "car", "color": "brown", "kind": "same",
                  "t": [9.8, 9.9, 10.0, 10.1], "s": [200.0, 203.8, 207.6, 211.4],
                  "z": [40.0] * 4, "d": [7.0] * 4, "v": [38.0] * 4, "first_t": 9.8, "last_t": 10.1}
        slow = {"id": 20, "type": "car", "color": "blue", "kind": "same",
                "t": [0.0 + 0.1 * i for i in range(41)],
                "s": [40.0 + 1.0 * min(i * 0.1, 1.5) for i in range(41)],  # stops at t=1.5
                "z": [40.0] * 41, "d": [3.5 + 0.2 * math.sin(i) for i in range(41)],
                "v": [1.0 if i < 15 else 0.0 for i in range(41)], "first_t": 0.0, "last_t": 4.0}
        scene = {"tracks": [rocket, slow], "speeds": [[0.0, 15.0], [10.0, 15.0]],
                 "speed_source": "overlay", "t_v0": 0.0, "t_hi": 10.0, "lane_samples": []}
        local_pipeline.apply_replay(spec, scene, 5.0)
        self.assertEqual([r["id"] for r in spec["replay"]], ["r20"])
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        plan = scene_builder.build(spec, tmp.name, "smh")
        kt = plan["agents"][plan["idmap"]["r20"]]["kinematic_track"]
        for r0, r1 in zip(kt, kt[1:]):
            dh = abs(((r1[3] - r0[3] + 180.0) % 360.0) - 180.0)
            self.assertLessEqual(dh, 6.5)

    def test_to_road_tracks_filters_brief_noisy_rocket_and_single_lane_drift_stays_flat(self):
        # Raw track with 4 noisy observations moving > 25 m/s over 0.3 s must be dropped in to_road_tracks
        obs_fast = []
        for i in range(4):
            u, v = CAM.pixel_from_ground(-10.0, 40.0 + 3.8 * i)
            obs_fast.append({"t": 9.8 + 0.1 * i, "box": [u - 20, v - 20, u + 20, v], "cls": 2, "conf": 0.6, "color": "brown"})
        obs_norm = []
        for i in range(8):
            u, v = CAM.pixel_from_ground(-3.5, 25.0 - 0.5 * i)
            obs_norm.append({"t": 5.0 + 0.1 * i, "box": [u - 30, v - 30, u + 30, v], "cls": 2, "conf": 0.8, "color": "white"})
        tracks = vehicle_tracks.to_road_tracks({66: obs_fast, 55: obs_norm}, CAM, lambda t: 15.0 * t)
        self.assertEqual([t["id"] for t in tracks], [55])

        # A track with 2.3 m lateral drift (curve/depth noise, < 2.8 m) stays locked in one lane with no swerve
        spec = spec_with(speed=50.0)
        spec["road"]["lanes_per_direction"] = 3
        spec["road"]["centerline"] = [[0.0, 0.0], [400.0, 0.0]]
        spec["replay"] = [{"id": "r34", "type": "car", "color": "black", "kind": "same",
                           "path": [[1.0, 50.0, 3.8], [2.5, 62.0, 4.9], [4.0, 75.0, 6.1]],
                           "speed": [[1.0, 30.0], [4.0, 30.0]]}]
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        plan = scene_builder.build(spec, tmp.name, "drift")
        kt = plan["agents"][plan["idmap"]["r34"]]["kinematic_track"]
        ys = [row[2] for row in kt]
        self.assertLess(max(ys) - min(ys), 0.3)


class TestForkAndResilienceFixes(unittest.TestCase):
    def test_paint_channel_prevents_twinlitenet_ribbon_from_clipping_dashed_highway_lanes(self):
        # TwinLiteNet outputs a continuous ribbon on a dashed left divider (line=10 everywhere),
        # while the raw paint channel only fires on dashes (~33% of rows) -> must NOT clip left lanes.
        g = synthetic_grid(2, 0, s_max=60.0)
        d = np.array([g.d_of(j) for j in range(g.nd)])
        dash_rows = (np.arange(g.ns) % 6) < 2
        left_div = np.abs(d - 1.75) < 0.25
        right_edge = np.abs(d - (-1.75)) < 0.25
        g.line[:, left_div] = 10.0                                          # TwinLiteNet ribbon fills dashes solid
        g.seen_p[:] = 10.0
        g.paint[:, left_div] = 10.0 * dash_rows[:, None]                    # raw paint preserves dashes
        g.paint[:, right_edge] = 10.0                                       # right edge truly solid
        nl, nr, _ = bev_lanes.count_slice(g, 30.0, 3.5)
        self.assertEqual(nl, 2)
        self.assertEqual(nr, 0)

    def test_apply_fork_synthesizes_diverging_branch_even_from_constant_lane_profile(self):
        from agents import lane_lines
        # BEV lane map saw constant 2 lanes, but 2 tracked cars physically diverge to the left from s=70 m
        samples = [{"t": float(s), "lanes_left": 1.0, "lanes_right": 0.0, "road_lanes": 2.0, "lane_w": 3.5,
                    "probs": lane_lines.count_probs(2.0)} for s in range(0, 180, 3)]
        tracks = [drift_track(1, 5.5, 0.22, s0=68.0, ratio=1.5), drift_track(2, 8.5, 0.24, s0=74.0, ratio=1.4)] + \
                 [drift_track(10 + k, 3.5, 0.0, ratio=1.0) for k in range(4)]
        scene = {"tracks": tracks, "speeds": [[0.0, 15.0], [15.0, 15.0]], "speed_source": "overlay",
                 "t_v0": 0.0, "t_hi": 12.0, "lane_samples": samples}
        spec = spec_with([{"id": "c1", "relation": "beside", "lane": "left", "behavior": "cutting_in"}],
                         {"type": "collision", "actor_id": "c1", "time_s": 8.0})
        ok = local_pipeline.lane_profile_from_map(spec, scene, "v.mp4", 15.0, log_fn=lambda m: None)
        self.assertTrue(ok)
        prof = spec["road"]["lane_profile"]
        self.assertIsNotNone(prof)
        self.assertEqual(len(prof), 2)
        self.assertEqual(prof[1]["branch"], {"side": "left", "rise_m": 6.0})
        self.assertGreater(prof[0]["n"], prof[1]["n"])

    def test_yolo_no_download_guard_and_measure_scene_degrades_without_yolo(self):
        from unittest import mock
        from agents import lane_lines
        saved_m = vehicle_tracks._MODEL.pop("m", None)
        try:
            with mock.patch.dict(os.environ, {"AS_NO_DOWNLOAD": "1"}), \
                    mock.patch.object(vehicle_tracks, "WEIGHTS", os.path.join(tempfile.gettempdir(), "nonexistent_yolo26n.pt")):
                with self.assertRaises(FileNotFoundError):
                    vehicle_tracks._yolo()
        finally:
            if saved_m is not None:
                vehicle_tracks._MODEL["m"] = saved_m
        # measure_scene keeps overlay ego speeds and BEV samples even when YOLO detect_tracks raises
        spec = spec_with(speed=60.0)
        caps = [{"t": float(i), "speed_kmh": 60.0} for i in range(15)]
        lane_rows = [{"t": 0.5 * i, "vp": [480.0, 220.0], "ratio": 2.8} for i in range(20)]
        fake_samples = [{"t": float(s), "lanes_left": 1.0, "lanes_right": 0.0, "road_lanes": 2.0,
                         "probs": lane_lines.count_probs(2.0)} for s in range(0, 120, 5)]
        with mock.patch.object(lane_lines, "road_bottom", return_value=(360, 540)), \
                mock.patch.object(camera_geom, "focal_from_dashes", return_value=(None, 0)), \
                mock.patch.object(vehicle_tracks, "detect_tracks", side_effect=RuntimeError("offline")), \
                mock.patch.object(bev_lanes, "accumulate", return_value=synthetic_grid(1, 0, s_max=150.0)), \
                mock.patch.object(bev_lanes, "lane_samples", return_value=fake_samples):
            sc = local_pipeline.measure_scene(spec, "v.mp4", 15.0, caps, lane_rows=lane_rows, log_fn=lambda m: None)
        self.assertIsNotNone(sc)
        self.assertEqual(sc["speed_source"], "overlay")
        self.assertEqual(sc["tracks"], [])
        self.assertEqual(len(sc["lane_samples"]), len(fake_samples))
        self.assertIn("YOLO degraded", sc["note"])

    def test_branch_replay_track_is_not_flattened_and_empty_branch_gets_cruisers(self):
        from test_lane_profile import LEFT_FLYOVER
        spec = spec_with([{"id": "c1", "relation": "beside", "lane": "left", "behavior": "cutting_in"},
                          {"id": "b1", "relation": "ahead", "lane": "left", "behavior": "driving", "distance_m": 25.0, "speed_kmh": 35.0}],
                         {"type": "collision", "actor_id": "c1", "time_s": 8.0}, speed=54.0)
        spec["road"]["lane_profile"] = LEFT_FLYOVER
        spec["road"]["one_way"] = True
        spec["ego"]["lane_index"] = 4
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        plan = scene_builder.build(spec, tmp.name, "br_pop")
        self.assertIn("br1", plan["idmap"])
        self.assertIn("fork", plan["road"])
        self.assertEqual(plan["road"]["no_shoulder_sides"], {"branch": ["right"], "eastbound_after": ["left"]})
