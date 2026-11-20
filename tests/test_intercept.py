import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.intercept import CutInConfig, CutInController

DT = 0.05
EGO_D = -5.25
SRC_D = -1.75


def simulate(ego_factor, ds0, t_event=7.0, set_touched=True, aim=0.0, t_end=12.0):
    """Tiny kinematic world: the ego car drives straight at a constant speed, the controlled car follows the controller exactly."""
    cfg = CutInConfig(lane_src_d=SRC_D, lane_dst_d=EGO_D, t_event_s=t_event, aim_offset_m=aim)
    ctl = CutInController(cfg)
    v_ego = 17.4 * ego_factor
    ego_s, me_s, me_d, me_v = 15.0, 15.0 + ds0, SRC_D, 17.4
    touched, t_touch, d_at_touch, log = False, None, None, []
    t = 0.0
    while t < t_end:
        d_t, v_t = ctl.update(t, (ego_s, EGO_D, v_ego), (me_s, me_d, me_v), touched)
        me_d, me_v = d_t, v_t
        ego_s += v_ego * DT
        me_s += me_v * DT
        t += DT
        overlap = abs(me_d - EGO_D) < 2.0 and abs(me_s - ego_s) < 4.5
        if overlap and not touched and set_touched:
            touched, t_touch, d_at_touch = True, t, me_d
        log.append((t, me_d, me_s - ego_s))
    return t_touch, d_at_touch, me_d, log


class TestCutIn(unittest.TestCase):
    def test_contact_happens_in_the_window_for_speed_and_offset_variation(self):
        for factor in (0.9, 1.0, 1.1):
            for ds0 in (-3.0, 0.0, 3.0):
                t_touch, _, _, _ = simulate(factor, ds0)
                self.assertIsNotNone(t_touch, (factor, ds0))
                self.assertTrue(5.6 <= t_touch <= 8.6, (factor, ds0, t_touch))

    def test_after_contact_the_lateral_target_is_frozen(self):
        t_touch, d_at_touch, d_end, log = simulate(1.0, 0.0)
        later = [d for (t, d, _) in log if t > t_touch + 0.5]
        self.assertTrue(later)
        self.assertTrue(all(abs(d - d_at_touch) < 1e-9 for d in later))

    def test_without_contact_feedback_the_car_ends_in_the_ego_lane(self):
        _, _, d_end, _ = simulate(1.0, 0.0, set_touched=False)
        self.assertAlmostEqual(d_end, EGO_D, delta=0.3)

    def test_it_stays_beside_the_ego_car_before_the_manoeuvre(self):
        _, _, _, log = simulate(1.1, 3.0)
        before = [(g) for (t, d, g) in log if 1.0 < t < 5.0]
        self.assertTrue(all(abs(g - 3.0) < 1.0 for g in before), (min(before), max(before)))
        self.assertTrue(all(abs(d - SRC_D) < 1e-9 for (t, d, g) in log if t < 5.5))

    def test_a_near_miss_aim_offset_keeps_a_lateral_gap(self):
        _, _, d_end, _ = simulate(1.0, 0.0, set_touched=False, aim=2.6)
        self.assertAlmostEqual(abs(d_end - EGO_D), 2.6, delta=0.15)

    def test_lateral_speed_is_limited(self):
        _, _, _, log = simulate(1.0, 0.0)
        for (t0, d0, _), (t1, d1, _) in zip(log, log[1:]):
            self.assertLessEqual(abs(d1 - d0) / (t1 - t0), 1.5 + 1e-6)


if __name__ == "__main__":
    unittest.main()
