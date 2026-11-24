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

    def test_overtake_cut_in_starts_ahead_at_lower_speed_and_contacts_in_window(self):
        cfg = CutInConfig(lane_src_d=SRC_D, lane_dst_d=EGO_D, t_event_s=8.0, cruise_speed_ms=7.0)
        ctl = CutInController(cfg)
        v_ego, v_me = 17.4, 7.0
        ego_s = 15.0
        me_s = ego_s + (v_ego - v_me) * 8.0 + 1.0
        me_d = SRC_D
        touched, t_touch = False, None
        t = 0.0
        while t < 12.0:
            d_t, v_t = ctl.update(t, (ego_s, EGO_D, v_ego), (me_s, me_d, v_me), touched)
            me_d, v_me = d_t, v_t
            ego_s += v_ego * DT
            me_s += v_me * DT
            t += DT
            if not touched and abs(me_d - EGO_D) < 2.0 and abs(me_s - ego_s) < 4.5:
                touched, t_touch = True, t
        self.assertIsNotNone(t_touch)
        self.assertTrue(6.5 <= t_touch <= 9.5, t_touch)

    def test_multi_lane_overtake_cut_in_crosses_middle_lane_and_contacts(self):
        ego_d, mid_d, far_left_d = -12.25, -8.75, -5.25
        v_ego, v_me = 15.0, 8.0
        t_ev = 9.5
        cfg = CutInConfig(lane_src_d=far_left_d, lane_dst_d=ego_d, t_event_s=t_ev,
                          lead_s=2.5, max_lateral_speed_ms=3.2, cruise_speed_ms=v_me)
        ctl = CutInController(cfg)
        ego_s = 15.0
        me_s = ego_s + (v_ego - v_me) * t_ev + 1.0
        me_d = far_left_d
        touched, t_touch, ds_seen = False, None, []
        t = 0.0
        while t < 13.0:
            d_t, v_t = ctl.update(t, (ego_s, ego_d, v_ego), (me_s, me_d, v_me), touched)
            me_d, v_me = d_t, v_t
            ego_s += v_ego * DT
            me_s += v_me * DT
            t += DT
            ds_seen.append(me_d)
            if not touched and abs(me_d - ego_d) < 2.0 and abs(me_s - ego_s) < 4.5:
                touched, t_touch = True, t
        self.assertIsNotNone(t_touch)
        self.assertTrue(8.0 <= t_touch <= 11.0, t_touch)
        self.assertTrue(any(abs(d - mid_d) < 0.25 for d in ds_seen))

    def test_side_strike_multi_lane_cutin_hits_ego_from_side_with_decelerating_ego(self):
        # Video 1 geometry: ego decelerates 19.6 -> 14.3 m/s while far-left white sedan cuts across 2 lanes into ego's side
        ego_d, far_left_d = -15.75, -8.75
        v_c = 6.5
        t_ev = 9.5
        cfg = CutInConfig(lane_src_d=far_left_d, lane_dst_d=ego_d, t_event_s=t_ev, lead_s=2.5,
                          max_lateral_speed_ms=3.2, cruise_speed_ms=v_c, long_offset_m=-1.2, side_strike=True)
        ctl = CutInController(cfg)
        ego_s = 15.0
        # Average ego speed over 0..9.5 s is ~16.95 m/s -> s_ev ~ 161 m
        me_s = ego_s + 161.0 - 1.2 - v_c * t_ev
        me_d, me_v = far_left_d, v_c
        touched, t_touch, gap_at_touch = False, None, None
        t = 0.0
        while t < 13.0:
            v_ego = max(14.3, 19.6 - 0.56 * t)
            d_t, v_t = ctl.update(t, (ego_s, ego_d, v_ego), (me_s, me_d, me_v), touched)
            me_d, me_v = d_t, v_t
            ego_s += v_ego * DT
            me_s += me_v * DT
            t += DT
            if not touched and abs(me_d - ego_d) < 2.1 and abs(me_s - ego_s) < 4.5:
                touched, t_touch, gap_at_touch = True, t, me_s - ego_s
        self.assertIsNotNone(t_touch)
        self.assertTrue(8.0 <= t_touch <= 11.0, t_touch)
        # Cutter must strike ego from the side (not pull metres ahead of ego where ego hits cutter's rear!)
        self.assertLessEqual(gap_at_touch, 0.3, gap_at_touch)
        self.assertGreaterEqual(gap_at_touch, -2.8, gap_at_touch)


if __name__ == "__main__":
    unittest.main()
