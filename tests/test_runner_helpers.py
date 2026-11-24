import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import motion_runner


class TestTtc(unittest.TestCase):
    def test_non_positive_closing_speed_is_infinite(self):
        self.assertEqual(motion_runner.ttc(20.0, 0.0), math.inf)
        self.assertEqual(motion_runner.ttc(20.0, -3.0), math.inf)

    def test_time_to_collision(self):
        self.assertAlmostEqual(motion_runner.ttc(20.0, 10.0), 2.0)

    def test_negative_gap_counts_as_zero(self):
        self.assertEqual(motion_runner.ttc(-1.0, 5.0), 0.0)


class TestRelativePosition(unittest.TestCase):
    def test_other_ahead_and_to_the_right(self):
        fwd, right = motion_runner.rel_in_frame(10.0, 2.0, 0.0)      # CARLA: +y is right when the yaw is 0
        self.assertAlmostEqual(fwd, 10.0)
        self.assertAlmostEqual(right, 2.0)

    def test_rotated_frame(self):
        fwd, right = motion_runner.rel_in_frame(0.0, 5.0, 90.0)      # heading +y: the point is straight ahead
        self.assertAlmostEqual(fwd, 5.0)
        self.assertAlmostEqual(right, 0.0, places=6)


class TestBarrierOffsets(unittest.TestCase):
    def test_right_barrier_stands_outside_lanes_and_shoulder(self):
        road = {"lanes": 2, "shoulder_m": 3.0, "edge_right": "barrier", "edge_left": "shoulder", "one_way": False}
        self.assertEqual(motion_runner.barrier_offsets(road), [-(2 * 3.5 + 3.0 + 0.3)])

    def test_left_barrier_only_on_a_one_way_road(self):
        road = {"lanes": 2, "shoulder_m": 3.0, "edge_right": "open", "edge_left": "barrier", "one_way": False}
        self.assertEqual(motion_runner.barrier_offsets(road), [])
        road["one_way"] = True
        self.assertEqual(motion_runner.barrier_offsets(road), [3.3])

    def test_no_road_plan_no_barriers(self):
        self.assertEqual(motion_runner.barrier_offsets(None), [])


class TestContactCap(unittest.TestCase):
    def test_all_contacts_are_kept_up_to_the_cap(self):
        self.assertGreaterEqual(motion_runner.MAX_CONTACTS, 200)


class TestCrashFocusTransform(unittest.TestCase):
    def test_crash_focus_transform_aims_spectator_at_crash_point(self):
        cx, cy, cz, pitch, yaw = motion_runner.crash_focus_transform(72.5, 5.25, 0.0, back_m=20.0, side_m=4.0, height_m=14.0)
        self.assertAlmostEqual(cx, 52.5)
        self.assertAlmostEqual(cy, 9.25)
        self.assertAlmostEqual(cz, 14.0)
        aim_dx, aim_dy = 72.5 - cx, 5.25 - cy
        self.assertAlmostEqual(yaw, math.degrees(math.atan2(aim_dy, aim_dx)), places=1)
        self.assertLess(pitch, 0.0)


class TestBgraBytesToBgr(unittest.TestCase):
    def test_bgra_bytes_to_bgr_strips_alpha(self):
        raw = bytes([10, 20, 30, 255, 40, 50, 60, 128])
        bgr = motion_runner.bgra_bytes_to_bgr(raw, 2, 1)
        self.assertEqual(bgr.shape, (1, 2, 3))
        self.assertEqual(bgr[0, 0].tolist(), [10, 20, 30])
        self.assertEqual(bgr[0, 1].tolist(), [40, 50, 60])


class TestPredictPointAndYaw(unittest.TestCase):
    def test_follows_curved_route_tangent(self):
        class _Loc:
            def __init__(self, x, y): self.x, self.y = x, y
        class _Rot:
            yaw = 0.0
        class _Tf:
            def __init__(self, x, y): self.location, self.rotation = _Loc(x, y), _Rot()
        class _Wp:
            def __init__(self, x, y): self.transform = _Tf(x, y)
        class _Actor:
            def get_location(self): return _Loc(0.0, 0.0)
            def get_transform(self): return _Tf(0.0, 0.0)
        class _Fol:
            route = [_Wp(10.0, 0.0), _Wp(10.0, 10.0)]
        px, py, yaw = motion_runner._predict_point_and_yaw(_Actor(), _Fol(), 36.0, 1.5)
        self.assertAlmostEqual(px, 10.0)
        self.assertAlmostEqual(py, 5.0)
        self.assertAlmostEqual(yaw, 90.0, places=1)


class TestSmoothCameraPose(unittest.TestCase):
    def test_smooth_camera_pose_wraps_yaw_shortest_arc(self):
        cur = (0.0, 0.0, 7.5, -20.0, 175.0)
        tgt = (10.0, 0.0, 7.5, -20.0, -175.0)
        nxt = motion_runner.smooth_camera_pose(cur, tgt, alpha=0.5)
        self.assertAlmostEqual(nxt[0], 5.0)
        self.assertAlmostEqual(nxt[4], 180.0)


class TestRouteFollowerGeometryAndAvoidance(unittest.TestCase):
    def _setup_carla(self):
        import types
        class _Loc:
            def __init__(self, x=0.0, y=0.0, z=0.0): self.x, self.y, self.z = x, y, z
            def distance(self, o): return math.sqrt((self.x - o.x)**2 + (self.y - o.y)**2 + (self.z - o.z)**2)
        class _Rot:
            def __init__(self, yaw=0.0): self.yaw = yaw
        class _Tf:
            def __init__(self, x, y, z=0.0, yaw=0.0): self.location, self.rotation = _Loc(x, y, z), _Rot(yaw)
        class _Vec:
            def __init__(self, vx, vy): self.x, self.y = vx, vy
            def length(self): return math.hypot(self.x, self.y)
        class _Ctl:
            def __init__(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False):
                self.throttle, self.steer, self.brake, self.hand_brake = throttle, steer, brake, hand_brake
        class _Car:
            def __init__(self, x, y, z, yaw, vx, vy, type_id="vehicle.dodge.charger"):
                self._loc, self._tf, self._vel, self.type_id = _Loc(x, y, z), _Tf(x, y, z, yaw), _Vec(vx, vy), type_id
            def get_location(self): return self._loc
            def get_transform(self): return self._tf
            def get_velocity(self): return self._vel
            def get_physics_control(self): return types.SimpleNamespace(wheels=[types.SimpleNamespace(max_steer_angle=70.0)])
        return _Loc, _Car, _Ctl

    def test_elevated_car_advances_waypoints_and_ignores_other_levels_and_adjacent_bend_cars(self):
        import types
        from agents.road_frame import RoadFrame
        _Loc, _Car, _Ctl = self._setup_carla()
        saved = sys.modules.get("carla")
        sys.modules["carla"] = types.SimpleNamespace(Location=_Loc, VehicleControl=_Ctl)
        try:
            R = 187.0
            pts = [[R * math.sin(s / R), -R * (1.0 - math.cos(s / R))] for s in range(0, 300, 4)]
            fr = RoadFrame(pts)
            route = [motion_runner._RoutePoint(_Loc(x, -y, 0.0)) for s in range(0, 260, 4) for x, y in [fr.to_xy(float(s), -3.5)]]
            sx, sy = fr.to_xy(20.0, -3.5)
            h = -fr.heading(20.0)
            me = _Car(sx, -sy, 6.0, h, 15.0 * math.cos(math.radians(h)), 15.0 * math.sin(math.radians(h)))
            fol = motion_runner.RouteFollower(me, route, 54.0, avoid=True, road_frame=fr)
            # Same-s/d car on a lower bridge deck (z=0 vs z=6) must be ignored
            under = _Car(sx + 10.0 * math.cos(math.radians(h)), -sy + 10.0 * math.sin(math.radians(h)), 0.0, h, 0.0, 0.0)
            # Car 35 m ahead on an adjacent lane (d = -7.0 vs -3.5) on R=187 m curve must be ignored
            ax, ay = fr.to_xy(55.0, -7.0)
            adj = _Car(ax, -ay, 6.0, -fr.heading(55.0), 0.0, 0.0)
            ctl = fol.control([under, adj], 0.05, 1.0, 1.0)
            self.assertGreater(fol.i, 0)                      # waypoint index advanced despite z=6 vs z=0
            self.assertEqual(ctl.brake, 0.0)                  # did not phantom-brake
        finally:
            if saved is None:
                sys.modules.pop("carla", None)
            else:
                sys.modules["carla"] = saved


class TestKinematicAndCleanup(unittest.TestCase):
    def test_sample_kinematic_track_interpolates_pose_and_wraps_heading(self):
        track = [[0.0, 10.0, 0.0, 175.0, 10.0], [2.0, 30.0, 4.0, -175.0, 14.0]]
        self.assertIsNone(motion_runner.sample_kinematic_track(track, -0.1))
        self.assertIsNone(motion_runner.sample_kinematic_track(track, 2.1))
        x, y, h, s = motion_runner.sample_kinematic_track(track, 1.0)
        self.assertAlmostEqual(x, 20.0)
        self.assertAlmostEqual(y, 2.0)
        self.assertAlmostEqual(h, 180.0)
        self.assertAlmostEqual(s, 12.0)

    def test_cleanup_destroys_orphaned_world_actors(self):
        import json, tempfile, os, types
        destroyed = []
        class _Destroy:
            def __init__(self, aid): self.aid = aid
        class _Client:
            def apply_batch(self, cmds): destroyed.extend(c.aid for c in cmds)
        class _Actor:
            def __init__(self, aid, tid): self.id, self.type_id = aid, tid
        class _World:
            def get_actors(self):
                return [_Actor(10, "vehicle.dodge.charger"), _Actor(11, "vehicle.nissan.patrol"),
                        _Actor(12, "sensor.other.collision"), _Actor(13, "static.prop.streetbarrier"),
                        _Actor(99, "traffic.traffic_light")]
        saved = sys.modules.get("carla")
        sys.modules["carla"] = types.SimpleNamespace(command=types.SimpleNamespace(DestroyActor=_Destroy))
        try:
            with tempfile.TemporaryDirectory() as tmp:
                ids_fn = os.path.join(tmp, "ids.json")
                with open(ids_fn, "w") as f:
                    json.dump([10], f)
                motion_runner._cleanup(_Client(), ids_fn, _World())
            self.assertEqual(destroyed, [10, 11, 12, 13])
        finally:
            if saved is None:
                sys.modules.pop("carla", None)
            else:
                sys.modules["carla"] = saved

    def test_extend_route_along_frame_recovers_truncated_route(self):
        from agents.road_frame import RoadFrame
        class _Loc:
            def __init__(self, x=0.0, y=0.0, z=0.0): self.x, self.y, self.z = x, y, z
        pts = [[float(s), 0.0005 * s * s] for s in range(0, 560, 20)]
        fr = RoadFrame(pts)
        # Truncated route ending near s=160 at lane offset d=-15.75
        d_lane = -15.75
        route = [motion_runner._RoutePoint(_Loc(x, -y))
                 for s in range(20, 162, 2)
                 for x, y in [fr.to_xy(float(s), d_lane)]]
        tgt_r = fr.to_xy(530.0, d_lane)
        tgt_carla = (tgt_r[0], -tgt_r[1])
        out = motion_runner.extend_route_along_frame(route, tgt_carla, fr, motion_runner._RoutePoint, _Loc)
        self.assertGreater(len(out), 230)
        end_loc = out[-1].transform.location
        s_end, d_end = fr.to_sd(end_loc.x, -end_loc.y)
        self.assertGreater(s_end, 520.0)
        self.assertAlmostEqual(d_end, d_lane, delta=0.2)

    def test_sanitize_straight_route_rebuilds_when_opendrive_jumps_onto_shoulder(self):
        from agents.road_frame import RoadFrame
        class _Loc:
            def __init__(self, x=0.0, y=0.0, z=0.0): self.x, self.y, self.z = x, y, z
        pts = [[float(s), 0.0005 * s * s] for s in range(0, 560, 20)]
        fr = RoadFrame(pts)
        d_ego, d_shoulder = -15.75, -19.0
        bad_route = ([motion_runner._RoutePoint(_Loc(x, -y)) for s in range(20, 110, 2) for x, y in [fr.to_xy(float(s), d_ego)]] +
                     [motion_runner._RoutePoint(_Loc(x, -y)) for s in range(110, 530, 2) for x, y in [fr.to_xy(float(s), d_shoulder)]])
        s0_r = fr.to_xy(20.0, d_ego)
        tgt_r = fr.to_xy(530.0, d_ego)
        clean = motion_runner.sanitize_straight_route(bad_route, (s0_r[0], -s0_r[1]), (tgt_r[0], -tgt_r[1]),
                                                      fr, motion_runner._RoutePoint, _Loc)
        self.assertGreater(len(clean), 230)
        for wp in clean:
            _, dw = fr.to_sd(wp.transform.location.x, -wp.transform.location.y)
            self.assertAlmostEqual(dw, d_ego, delta=0.2)

    def test_fork_barriers_never_enter_branch_lanes_and_gore_starts_after_clearance(self):
        road = {
            "lanes": 4, "shoulder_m": 3.0, "edge_right": "barrier", "edge_left": "barrier", "one_way": True,
            "sections": [{"s0": 0.0, "s1": 80.0, "n": 4, "m": 0},
                         {"s0": 80.0, "s1": 550.0, "n": 2, "m": 2, "branch": {"side": "left", "rise_m": 6.0}}],
            "fork": {"s_b": 80.0, "side": "left", "rise_m": 6.0, "leave_j": [0, 1],
                     "diverge_len_m": 100.0, "diverge_d_m": 10.0, "rise_start_m": 55.0, "rise_len_m": 100.0},
        }
        # Right after split (s=85): left barrier MUST stay outside branch's outer left edge (d > 3.0), NEVER at -3.7
        b_right, b_left = motion_runner.barrier_offsets(road, 85.0)
        self.assertLess(b_right, -14.0)
        self.assertGreater(b_left, 3.0)
        self.assertIsNone(motion_runner.gore_barrier_specs(road, 90.0))      # < 3.2 m separation yet
        # Well past split (s=180): branch moved +10 m left -> outer left barrier at +13.3 m, gore divider appears cleanly
        b_right2, b_left2 = motion_runner.barrier_offsets(road, 180.0)
        self.assertAlmostEqual(b_left2, 13.3, places=2)
        gb = motion_runner.gore_barrier_specs(road, 180.0)
        self.assertIsNotNone(gb)
        self.assertAlmostEqual(gb[0], -7.0 + 0.7, places=2)

    def test_sanitize_straight_route_preserves_diverging_branch_path(self):
        from agents.road_frame import RoadFrame
        class _Loc:
            def __init__(self, x=0.0, y=0.0, z=0.0): self.x, self.y, self.z = x, y, z
        fr = RoadFrame([[0.0, 0.0], [300.0, 0.0]])
        road_plan = {"fork": {"s_b": 80.0, "side": "left", "rise_m": 6.0, "leave_j": [0, 1],
                              "diverge_len_m": 100.0, "diverge_d_m": 10.0, "rise_start_m": 55.0, "rise_len_m": 100.0}}
        # Vehicle in lane j=0 (d=-1.75) follows diverging branch up to d=+8.25
        route = []
        for s in range(20, 260, 4):
            div_d, _ = motion_runner._fork_disp(road_plan["fork"], float(s))
            x, y = fr.to_xy(float(s), -1.75 + div_d)
            route.append(motion_runner._RoutePoint(_Loc(x, -y)))
        out = motion_runner.sanitize_straight_route(route, (20.0, 1.75), (260.0, -8.25), fr,
                                                    motion_runner._RoutePoint, _Loc, road_plan=road_plan)
        _, d_end = fr.to_sd(out[-1].transform.location.x, -out[-1].transform.location.y)
        self.assertGreater(d_end, 6.0)

    def test_weather_for_maps_environment_presets(self):
        day_clear = motion_runner.weather_for({"time_of_day": "day", "weather": "clear"})
        day_cloudy = motion_runner.weather_for({"time_of_day": "day", "weather": "cloudy"})
        night = motion_runner.weather_for({"time_of_day": "night", "weather": "clear"})
        self.assertGreater(day_clear["sun_altitude_angle"], 30.0)
        self.assertGreater(day_cloudy["cloudiness"], 70.0)
        self.assertLess(night["sun_altitude_angle"], 0.0)


if __name__ == "__main__":
    unittest.main()
