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
        if cfg.cruise_speed_ms is not None:
            rem = cfg.t_event_s - t
            if rem > 0.3:
                target_s_ev = ego_s + ego_v * rem + 1.0
                v_req = (target_s_ev - me_s) / rem
                v_target = max(max(1.5, 0.5 * cfg.cruise_speed_ms), min(max(ego_v, 2.0 * cfg.cruise_speed_ms), v_req))
            else:
                v_target = max(cfg.cruise_speed_ms, min(ego_v, me_v))
        else:
            if self._gap0 is None:
                self._gap0 = me_s - ego_s
            v_target = max(ego_v + 0.5 * (ego_s + self._gap0 - me_s), 0.0)
        if t >= cfg.t_event_s - cfg.lead_s:
            lo, hi = sorted((cfg.lane_src_d, cfg.lane_dst_d))
            goal = max(lo, min(hi, ego_d + cfg.aim_offset_m))
            step = cfg.max_lateral_speed_ms * dt
            self._d += max(-step, min(step, goal - self._d))
        return self._d, v_target
