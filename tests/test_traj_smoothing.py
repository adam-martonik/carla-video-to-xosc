import math
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import traj_smoothing as ts


class TestHampel(unittest.TestCase):
    def test_single_spike_is_replaced_and_input_untouched(self):
        raw = np.linspace(0.0, 10.0, 41)
        raw[20] += 25.0
        copy = raw.copy()
        out = ts.hampel(raw)
        self.assertLess(abs(out[20] - np.linspace(0.0, 10.0, 41)[20]), 1.0)
        np.testing.assert_array_equal(raw, copy)

    def test_short_input_is_returned_unchanged(self):
        np.testing.assert_array_equal(ts.hampel([1.0, 50.0, 2.0]), [1.0, 50.0, 2.0])


class TestRts(unittest.TestCase):
    def test_recovers_constant_velocity_from_noisy_positions(self):
        rng = np.random.default_rng(1)
        t = np.arange(0.0, 10.0, 0.1)
        z = 12.0 * t + rng.normal(0.0, 0.4, len(t))
        pos, vel, _ = ts.rts_smoother_1d(t, z, meas_std=0.4, process_std=0.3)
        self.assertLess(np.std(pos - 12.0 * t), 0.4)
        self.assertAlmostEqual(float(np.mean(vel[10:-10])), 12.0, delta=0.5)

    def test_degenerate_lengths(self):
        self.assertEqual(len(ts.rts_smoother_1d([], [])[0]), 0)
        p, v, a = ts.rts_smoother_1d([0.0], [3.0])
        self.assertEqual((p[0], v[0], a[0]), (3.0, 0.0, 0.0))


class TestSpeedLimits(unittest.TestCase):
    def test_acceleration_and_speed_are_bounded(self):
        t = np.arange(0.0, 6.0, 0.1)
        v = ts.limit_speed_profile(t, np.where(t < 1.0, 0.0, 40.0), max_speed=30.0, max_accel=3.0, max_decel=8.0, max_jerk=6.0)
        self.assertLessEqual(v.max(), 30.0 + 1e-9)
        self.assertLessEqual((np.diff(v) / np.diff(t)).max(), 3.0 + 1e-6)

    def test_braking_is_bounded(self):
        t = np.arange(0.0, 4.0, 0.1)
        v = ts.limit_speed_profile(t, np.where(t < 1.0, 20.0, 0.0), 40.0, 4.0, 6.0, 20.0)
        self.assertGreaterEqual((np.diff(v) / np.diff(t)).min(), -6.0 - 1e-6)
        self.assertGreaterEqual(v.min(), 0.0)


class TestSmoothTrajectory(unittest.TestCase):
    def test_straight_noisy_drive_stays_close_and_is_smooth(self):
        rng = np.random.default_rng(2)
        t = np.arange(0.0, 8.0, 0.05)
        x = 15.0 * t + rng.normal(0, 0.1, len(t))
        y = np.full_like(t, 3.0) + rng.normal(0, 0.1, len(t))
        r = ts.smooth_trajectory(t, x, y, "car")
        self.assertLess(r["max_deviation_m"], 1.0)
        self.assertAlmostEqual(float(np.median(r["v"])), 15.0, delta=0.6)
        self.assertLess(float(np.max(np.abs(r["h"]))), 0.1)             # heading east
        self.assertTrue(np.all(np.diff(r["times"]) > 0))

    def test_output_is_uniformly_sampled(self):
        t = np.arange(0.0, 3.0, 0.05)
        r = ts.smooth_trajectory(t, 5.0 * t, np.zeros_like(t), "car", dt=0.1)
        np.testing.assert_allclose(np.diff(r["times"])[:-1], 0.1, atol=1e-9)

    def test_stationary_actor_keeps_heading_and_zero_speed(self):
        t = np.arange(0.0, 3.0, 0.05)
        r = ts.smooth_trajectory(t, np.full_like(t, 4.0), np.full_like(t, -2.0), "car", initial_heading=1.2)
        self.assertTrue(np.allclose(r["v"], 0.0))
        self.assertTrue(np.allclose(r["h"], 1.2))
        self.assertTrue(np.allclose(r["x"], 4.0))

    def test_pedestrian_speed_is_capped(self):
        t = np.arange(0.0, 4.0, 0.05)
        r = ts.smooth_trajectory(t, 12.0 * t, np.zeros_like(t), "pedestrian")      # recorded as 12 m/s: impossible on foot
        self.assertLessEqual(float(r["v"].max()), ts.LIMITS["pedestrian"]["max_speed"] + 0.3)

    def test_curve_is_followed_and_speed_capped_in_the_corner(self):
        t = np.arange(0.0, 10.0, 0.05)
        ang = 0.25 * t                                                           # 20 m/s on a 80 m radius circle
        x, y = 80.0 * np.sin(ang), 80.0 * (1.0 - np.cos(ang))
        r = ts.smooth_trajectory(t, x, y, "truck")                               # a truck: lat accel limit 3.5 -> ~16.7 m/s
        self.assertLessEqual(float(r["v"].max()), math.sqrt(3.5 * 80.0) + 1.0)
        self.assertGreater(float(r["h"][-1]), 0.5)                               # the heading really turned

    def test_too_short_input_is_rejected(self):
        with self.assertRaises(ValueError):
            ts.smooth_trajectory([0.0], [0.0], [0.0])


if __name__ == "__main__":
    unittest.main()
