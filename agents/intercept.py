"""
Closed-loop controller for a car that cuts into the ego car's lane (no CARLA import, so it can be tested with plain numbers).

Lateral positions d are offsets across the road (+ = left), s is the distance along it. The controller returns a lateral target and
a speed target; the motion runner steers and accelerates the car toward them.

  before t_event - lead_s : stay in the source lane, beside the ego car (speed follows the ego speed and the initial gap)
  then                    : move toward the ego car's lateral position (+ aim_offset) at max_lateral_speed, from live positions,
                            so the contact does not depend on a pre-computed path
  after contact           : stop steering (the lateral target freezes where the car is) and brake
"""
from dataclasses import dataclass
import math


@dataclass
class CutInConfig:
    lane_src_d: float
    lane_dst_d: float
    t_event_s: float
    lead_s: float = 1.4
    aim_offset_m: float = 0.0
    max_lateral_speed_ms: float = 1.5
    width_m: float = 2.0
    cruise_speed_ms: float = None
    long_offset_m: float = 1.0
    side_strike: bool = False


class CutInController:
    def __init__(self, cfg):
        self.cfg = cfg
        self._d = cfg.lane_src_d
        self._t_prev = None
        self._gap0 = None
        self._frozen = None

    def update(self, t, ego, me, touched):
        """ego, me = (s, d, speed_ms). Returns (d_target, v_target_ms)."""
        cfg = self.cfg
        ego_s, ego_d, ego_v = ego
        me_s, me_d, me_v = me
        dt = 0.0 if self._t_prev is None else max(t - self._t_prev, 0.0)
        self._t_prev = t
        if touched:
            if self._frozen is None:
                self._frozen = me_d
            return self._frozen, 0.0
        rem = cfg.t_event_s - t
        cutting = t >= cfg.t_event_s - cfg.lead_s
        multi = abs(cfg.lane_dst_d - cfg.lane_src_d) > 4.0
        if cfg.cruise_speed_ms is not None:
            if cfg.side_strike:
                gap_s = me_s - ego_s
                if not cutting:
                    # Hold near measured slow cruise speed before cutting so early ego deceleration never pulls
                    # the cutter metres ahead of ego.
                    t_goal = max(rem, 0.35)
                    v_req = (ego_s + ego_v * t_goal + cfg.long_offset_m - me_s) / t_goal
                    v_hi = max(cfg.cruise_speed_ms * 1.05, 4.0) if rem > cfg.lead_s + 0.5 else max(ego_v, 2.0 * cfg.cruise_speed_ms)
                    v_lo = max(2.5, 0.5 * cfg.cruise_speed_ms)
                    v_target = max(v_lo, min(v_hi, v_req))
                else:
                    # Double-integrator rendezvous onto (gap_s = long_offset_m, v = ego_v) so the cutter accelerates
                    # smoothly to match ego's speed as it arrives beside ego's flank rather than being passed or cutting ahead.
                    dist = gap_s - cfg.long_offset_m
                    if dist > 0.0:
                        v_rend = ego_v - math.sqrt(2.0 * 2.8 * dist)
                        v_lo = max(3.5, 0.55 * cfg.cruise_speed_ms)
                        v_target = max(v_lo, min(ego_v, v_rend))
                    else:
                        v_match = ego_v + 1.4 * (-dist)
                        v_hi = max(ego_v + 4.5, 2.0 * cfg.cruise_speed_ms)
                        v_target = max(cfg.cruise_speed_ms, min(v_hi, v_match))
            elif rem > 0.3:
                target_s_ev = ego_s + ego_v * rem + cfg.long_offset_m
                v_req = (target_s_ev - me_s) / rem
                cut_floor = 6.5 if multi else 5.5
                v_lo = max(cfg.cruise_speed_ms, cut_floor) if cutting else max(3.0, 0.65 * cfg.cruise_speed_ms)
                v_target = max(v_lo, min(max(ego_v, 2.0 * cfg.cruise_speed_ms), v_req))
            else:
                v_target = max(cfg.cruise_speed_ms, (6.5 if multi else 5.5) if cutting else min(ego_v, me_v))
        else:
            if self._gap0 is None:
                self._gap0 = cfg.long_offset_m if (cfg.side_strike or abs(cfg.long_offset_m - 1.0) > 0.1) else (me_s - ego_s)
            v_target = max(ego_v + 0.5 * (ego_s + self._gap0 - me_s), 0.0)
        if cutting:
            lo, hi = sorted((cfg.lane_src_d, cfg.lane_dst_d))
            goal = max(lo, min(hi, ego_d + cfg.aim_offset_m))
            if cfg.side_strike and abs(cfg.aim_offset_m) < 0.5:
                # Hold just outside ego's lane boundary (2.65 m) until cutter is beside/slightly behind ego's centre
                # so its front corner strikes ego's side rather than entering ahead of ego's bumper.
                sign = 1.0 if cfg.lane_src_d > cfg.lane_dst_d else -1.0
                d_guard = max(lo, min(hi, ego_d + sign * 2.55))
                gap_s = me_s - ego_s
                ready_strike = (-2.4 <= gap_s <= 0.45) or (t >= cfg.t_event_s + 1.0)
                if not ready_strike:
                    goal = d_guard
            lat_spd = max(cfg.max_lateral_speed_ms, 2.2) if cfg.cruise_speed_ms is not None else cfg.max_lateral_speed_ms
            step = lat_spd * dt
            self._d += max(-step, min(step, goal - self._d))
        return self._d, v_target
