import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents import road_curve

V1 = ROOT / "data" / "user_videos" / "v1" / "v1.mp4"


def series(rate, secs=21.0, dt=0.5, speed=14.0):
    n = int(secs / dt)
    return [i * dt for i in range(n)], [rate] * n, [speed] * n


def end_heading_deg(points):
    (x0, y0), (x1, y1) = points[-2], points[-1]
    return math.degrees(math.atan2(y1 - y0, x1 - x0))


class TestCenterline(unittest.TestCase):
    def test_left_arc(self):
        t, w, v = series(0.02)
        c = road_curve.centerline_from_yaw(t, w, v)
        self.assertGreater(c["points"][-1][1], 5.0)
        self.assertAlmostEqual(end_heading_deg(c["points"]), 24.0, delta=3.0)
        self.assertAlmostEqual(c["radius_min_m"], 700.0, delta=35.0)

    def test_right_bend_goes_to_negative_y(self):
        t, w, v = series(-0.02)
        c = road_curve.centerline_from_yaw(t, w, v)
        self.assertLess(c["points"][-1][1], -5.0)

    def test_straight(self):
        t, w, v = series(0.0)
        c = road_curve.centerline_from_yaw(t, w, v)
        self.assertLess(max(abs(p[1]) for p in c["points"]), 0.5)

    def test_clamps_radius(self):
        t, w, v = series(0.2)
        c = road_curve.centerline_from_yaw(t, w, v, min_radius_m=150.0)
        self.assertGreaterEqual(c["radius_min_m"], 150.0 - 1e-6)

    def test_stopped_ego_is_straight_with_low_confidence(self):
        t, w, v = series(0.05, speed=0.0)
        c = road_curve.centerline_from_yaw(t, w, v)
        self.assertTrue(all(math.isfinite(p[0]) and math.isfinite(p[1]) for p in c["points"]))
        self.assertLess(max(abs(p[1]) for p in c["points"]), 0.5)
        self.assertLess(c["confidence"], 0.3)

    def test_unknown_speeds_use_the_default(self):
        t, w, _ = series(0.02)
        c = road_curve.centerline_from_yaw(t, w, [None] * len(t), default_speed_ms=14.0)
        self.assertAlmostEqual(c["radius_min_m"], 700.0, delta=35.0)

    def test_starts_at_origin_heading_east(self):
        t, w, v = series(0.02)
        c = road_curve.centerline_from_yaw(t, w, v)
        self.assertEqual(tuple(c["points"][0]), (0.0, 0.0))
        self.assertGreater(c["points"][1][0], c["points"][0][0])


class TestReconcile(unittest.TestCase):
    def _left(self):
        t, w, v = series(0.02)
        return road_curve.centerline_from_yaw(t, w, v)

    def test_probe_disagreement_forces_straight(self):
        c = road_curve.reconcile_with_probe(self._left(), "right")
        self.assertLess(max(abs(p[1]) for p in c["points"]), 0.5)
        self.assertIn("fallback", c)

    def test_probe_agreement_keeps_the_arc(self):
        c = road_curve.reconcile_with_probe(self._left(), "left")
        self.assertGreater(c["points"][-1][1], 5.0)
        self.assertNotIn("fallback", c)

    def test_no_probe_keeps_the_arc(self):
        c = road_curve.reconcile_with_probe(self._left(), None)
        self.assertGreater(c["points"][-1][1], 5.0)

    def test_probe_straight_does_not_contradict_a_gentle_bend(self):
        c = road_curve.reconcile_with_probe(self._left(), "straight")
        self.assertGreater(c["points"][-1][1], 5.0)

    def test_low_confidence_falls_back_to_straight(self):
        c = self._left()
        c["confidence"] = 0.1
        c = road_curve.reconcile_with_probe(c, "left")
        self.assertLess(max(abs(p[1]) for p in c["points"]), 0.5)
        self.assertIn("fallback", c)


@unittest.skipUnless(V1.exists(), "v1.mp4 is missing")
class TestVideo(unittest.TestCase):
    def test_v1_sanity(self):
        y = road_curve.estimate_ego_yaw(str(V1), 0.0, 20.0)
        dts = [b - a for a, b in zip(y["t"], y["t"][1:])]
        heading = abs(sum(w * dt for w, dt in zip(y["yaw_rate"][1:], dts)))
        self.assertTrue(5.0 < math.degrees(heading) < 30.0, math.degrees(heading))
        self.assertTrue(0.0 <= y["confidence"] <= 1.0)


if __name__ == "__main__":
    unittest.main()
