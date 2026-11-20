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


if __name__ == "__main__":
    unittest.main()
