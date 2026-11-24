"""
Vehicle control primitives, standard autonomous-driving training action spaces, and interactive
keyboard control for CARLA scenarios (pure simulation/training tool — no bundled AI models).

Standard action spaces supported for training external models:
  1. `continuous` (default, standard for PPO / SAC / TD3 / Behavior Cloning):
     2D float vector `[steer, accel]` in `[-1, 1] x [-1, 1]`:
       - `steer`: -1.0 = full left, 0.0 = straight, +1.0 = full right
       - `accel`: > 0 maps to throttle (0..1), < 0 maps to brake (0..1)
     (Combining throttle and brake into one signed axis prevents pressing gas and brake at the
     same time. 3D `[steer, throttle, brake]` and dict controls are also accepted.)
  2. `discrete` (9 standard driving primitives, suited for DQN / Categorical RL):
     0=coast, 1=accelerate, 2=gentle_brake, 3=hard_brake, 4=steer_left, 5=steer_right,
     6=brake_left, 7=brake_right, 8=hard_accelerate.
  3. `target_speed` (high-level planning control, popular in CARLA Leaderboard / TransFuser):
     2D float vector `[steer, target_speed_ms]` where steer is in `[-1, 1]` and `target_speed_ms`
     is in `[0, max_speed_ms]` (default 36 m/s ~ 130 km/h); an internal longitudinal controller
     converts target speed into smooth throttle and brake pedals.
"""
import importlib
import importlib.util
import math
import os

import numpy as np

ACTION_SPACES = ("continuous", "discrete", "target_speed")

# 9 standard discrete driving primitives: (name, steer, signed_accel)
DISCRETE_ACTIONS = (
    ("coast",           0.00,  0.00),
    ("accelerate",      0.00,  0.65),
    ("gentle_brake",    0.00, -0.45),
    ("hard_brake",      0.00, -1.00),
    ("steer_left",     -0.35,  0.15),
    ("steer_right",     0.35,  0.15),
    ("brake_left",     -0.55, -0.80),
    ("brake_right",     0.55, -0.80),
    ("hard_accelerate", 0.00,  1.00),
)

STATE_DIM = 12


def _clamp(v, lo, hi):
    return max(lo, min(hi, float(v)))


def _wrap_rad(a):
    return (float(a) + math.pi) % (2.0 * math.pi) - math.pi


class ControlAdapter:
    """Translates external model or keyboard actions in `continuous`, `discrete`, or `target_speed` space
    into smooth, physically valid CARLA vehicle controls `(throttle, steer, brake, hand_brake, reverse)`."""

    def __init__(self, action_space="continuous", dt=0.05, max_speed_ms=36.0,
                 steer_rate=3.0, throttle_rate=3.0, brake_rate=8.0, smooth=True):
        if action_space not in ACTION_SPACES:
            raise ValueError("unknown action_space %r (available: %s)" % (action_space, ", ".join(ACTION_SPACES)))
        self.action_space = action_space
        self.dt = float(dt)
        self.max_speed_ms = float(max_speed_ms)
        self.steer_rate = float(steer_rate)
        self.throttle_rate = float(throttle_rate)
        self.brake_rate = float(brake_rate)
        self.smooth = bool(smooth)
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.prev_brake = 0.0
        self.prev_hand_brake = False
        self.prev_reverse = False

    def reset(self):
        self.prev_steer = 0.0
        self.prev_throttle = 0.0
        self.prev_brake = 0.0
        self.prev_hand_brake = False
        self.prev_reverse = False

    def decode_action(self, action, speed_ms=0.0):
        """Convert raw action into target `(throttle, steer, brake, hand_brake, reverse)` before rate-limiting."""
        if isinstance(action, dict):
            steer = _clamp(action.get("steer", 0.0), -1.0, 1.0)
            if "accel" in action and "throttle" not in action and "brake" not in action:
                acc = _clamp(action["accel"], -1.0, 1.0)
                thr = max(0.0, acc)
                brk = max(0.0, -acc)
            else:
                thr = _clamp(action.get("throttle", 0.0), 0.0, 1.0)
                brk = _clamp(action.get("brake", 0.0), 0.0, 1.0)
            hb = bool(action.get("hand_brake", False))
            rev = bool(action.get("reverse", False))
            return thr, steer, brk, hb, rev
        if hasattr(action, "throttle") and hasattr(action, "steer") and hasattr(action, "brake"):
            return (
                _clamp(action.throttle, 0.0, 1.0),
                _clamp(action.steer, -1.0, 1.0),
                _clamp(action.brake, 0.0, 1.0),
                bool(getattr(action, "hand_brake", False)),
                bool(getattr(action, "reverse", False)),
            )

        if self.action_space == "discrete":
            idx = int(np.asarray(action).reshape(-1)[0])
            idx = max(0, min(len(DISCRETE_ACTIONS) - 1, idx))
            _, steer, acc = DISCRETE_ACTIONS[idx]
            return max(0.0, acc), steer, max(0.0, -acc), False, False

        arr = np.asarray(action, dtype=np.float32).reshape(-1)
        if self.action_space == "target_speed":
            if arr.size < 2:
                raise ValueError("target_speed action needs [steer, target_speed_ms], got shape %s" % (arr.shape,))
            steer = _clamp(arr[0], -1.0, 1.0)
            v_tgt = _clamp(arr[1], 0.0, self.max_speed_ms)
            err = v_tgt - float(speed_ms)
            a_des = _clamp((2.0 if err < 0.0 else 1.2) * err, -6.5, 3.2)
            thr = _clamp(0.0145 * float(speed_ms) + 0.30 * max(a_des, 0.0), 0.0, 1.0) if a_des > -0.15 else 0.0
            brk = _clamp(-a_des / 6.5, 0.0, 1.0) if a_des < -0.25 else 0.0
            if v_tgt <= 0.05 and float(speed_ms) < 0.5:
                thr, brk = 0.0, 1.0
            return thr, steer, brk, False, False

        # Continuous: accept [steer, accel] (2D standard) or [steer, throttle, brake] (3D)
        if arr.size == 2:
            steer = _clamp(arr[0], -1.0, 1.0)
            acc = _clamp(arr[1], -1.0, 1.0)
            return max(0.0, acc), steer, max(0.0, -acc), False, False
        if arr.size >= 3:
            steer = _clamp(arr[0], -1.0, 1.0)
            thr = _clamp(arr[1], 0.0, 1.0)
            brk = _clamp(arr[2], 0.0, 1.0)
            return thr, steer, brk, False, False
        raise ValueError("continuous action needs 2 ([steer, accel]) or 3 ([steer, throttle, brake]) values, got %d" % arr.size)

    def apply(self, action, speed_ms=0.0):
        """Decode and optionally rate-limit `action`, returning `(throttle, steer, brake, hand_brake, reverse)`."""
        thr, steer, brk, hb, rev = self.decode_action(action, speed_ms=speed_ms)
        if self.smooth:
            dt = self.dt
            steer = self.prev_steer + _clamp(steer - self.prev_steer, -self.steer_rate * dt, self.steer_rate * dt)
            thr = self.prev_throttle + _clamp(thr - self.prev_throttle, -self.throttle_rate * dt, self.throttle_rate * dt)
            brk = self.prev_brake + _clamp(brk - self.prev_brake, -self.brake_rate * dt, self.brake_rate * dt)
        if brk > 0.15 and thr > 0.0:
            thr = 0.0
        self.prev_throttle = _clamp(thr, 0.0, 1.0)
        self.prev_steer = _clamp(steer, -1.0, 1.0)
        self.prev_brake = _clamp(brk, 0.0, 1.0)
        self.prev_hand_brake = bool(hb)
        self.prev_reverse = bool(rev)
        return self.prev_throttle, self.prev_steer, self.prev_brake, self.prev_hand_brake, self.prev_reverse

    def to_carla_control(self, action, carla_mod, speed_ms=0.0):
        """Convert `action` directly into a `carla.VehicleControl` instance."""
        thr, steer, brk, hb, rev = self.apply(action, speed_ms=speed_ms)
        return carla_mod.VehicleControl(
            throttle=float(thr),
            steer=float(steer),
            brake=float(brk),
            hand_brake=bool(hb),
            reverse=bool(rev),
        )


# ----------------------------------------------------------------------------------------- Nominal target trajectory
class NominalTargetTrack:
    """Analytical reference trajectory `(s(t), d0, v(t))` of the subject vehicle had it continued blindly
    along its planned lane and speed. Used by `InterceptFollower` when the subject vehicle is under external/manual
    control so the cutting-in aggressor performs a realistic maneuver into the original lane instead of
    homing onto an evading vehicle."""

    def __init__(self, s0, d0, speed_kmh, speed_profile=None):
        self.s0 = float(s0)
        self.d0 = float(d0)
        self.base_v = max(0.0, float(speed_kmh) / 3.6)
        self.profile = list(speed_profile) if speed_profile else None

    def speed_at(self, t):
        t = max(0.0, float(t))
        if not self.profile:
            return self.base_v
        if t <= self.profile[0][0]:
            return float(self.profile[0][1]) / 3.6
        for (t0, v0), (t1, v1) in zip(self.profile, self.profile[1:]):
            if t <= t1:
                f = (t - t0) / max(t1 - t0, 1e-6)
                return float(v0 + (v1 - v0) * f) / 3.6
        return float(self.profile[-1][1]) / 3.6

    def pose_at(self, t):
        """Return `(s_nom, d_nom, v_nom_ms)` at simulation time `t`."""
        t = max(0.0, float(t))
        if not self.profile:
            return self.s0 + self.base_v * t, self.d0, self.base_v
        dt = 0.05
        n = int(t / dt)
        s = self.s0
        for k in range(n):
            s += self.speed_at((k + 0.5) * dt) * dt
        rem = t - n * dt
        if rem > 1e-9:
            s += self.speed_at(n * dt + 0.5 * rem) * rem
        return s, self.d0, self.speed_at(t)


# ----------------------------------------------------------------------------------------- Ego state extraction
def route_geometry_metrics(loc_xy, yaw_deg, route_xy):
    """Compute `(lane_offset_m, heading_err_rad, progress_frac, dist_to_end_m, lookahead_steer_rad)`
    of a vehicle at CARLA `(x, y, yaw_deg)` relative to a reference polyline `route_xy = [(x, y), ...]`."""
    if not route_xy or len(route_xy) < 2:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    vx, vy = float(loc_xy[0]), float(loc_xy[1])
    pts = np.asarray(route_xy, dtype=np.float64)
    segs = np.diff(pts, axis=0)
    seg_lens = np.hypot(segs[:, 0], segs[:, 1])
    total_len = max(float(np.sum(seg_lens)), 1e-3)
    cum = np.concatenate([[0.0], np.cumsum(seg_lens)])

    d2 = (pts[:, 0] - vx) ** 2 + (pts[:, 1] - vy) ** 2
    idx = int(np.argmin(d2))
    j0 = max(0, min(idx, len(pts) - 2))
    dx = pts[j0 + 1, 0] - pts[j0, 0]
    dy = pts[j0 + 1, 1] - pts[j0, 1]
    seg_l = max(math.hypot(dx, dy), 1e-6)
    tx, ty = dx / seg_l, dy / seg_l
    # CARLA frame (+X forward, +Y right): signed lateral offset (+ = right of route, - = left of route)
    rx, ry = vx - pts[j0, 0], vy - pts[j0, 1]
    proj = _clamp(rx * tx + ry * ty, 0.0, seg_l)
    lat_offset = -rx * ty + ry * tx

    route_yaw = math.atan2(dy, dx)
    veh_yaw = math.radians(float(yaw_deg))
    heading_err = _wrap_rad(route_yaw - veh_yaw)

    s_cur = float(cum[j0] + proj)
    progress = _clamp(s_cur / total_len, 0.0, 1.0)
    dist_end = max(0.0, total_len - s_cur)

    s_look = min(total_len, s_cur + 10.0)
    k = int(np.searchsorted(cum, s_look))
    k = max(0, min(k, len(pts) - 1))
    lx, ly = float(pts[k, 0]), float(pts[k, 1])
    alpha = _wrap_rad(math.atan2(ly - vy, lx - vx) - veh_yaw)
    return float(lat_offset), float(heading_err), float(progress), float(dist_end), float(alpha)


def extract_ego_state(actor, target_speed_ms=14.0, route_xy=None, prev_control=(0.0, 0.0, 0.0), prev_yaw_deg=None, dt=0.05):
    """Extract a 12-float kinematic + navigation state vector and readable telemetry dict."""
    tf = actor.get_transform()
    loc = tf.location
    yaw_deg = float(tf.rotation.yaw)
    yaw_rad = math.radians(yaw_deg)
    vel = actor.get_velocity()
    speed_ms = math.sqrt(vel.x * vel.x + vel.y * vel.y + vel.z * vel.z)
    fx, fy = math.cos(yaw_rad), math.sin(yaw_rad)
    rx, ry = -fy, fx
    try:
        acc = actor.get_acceleration()
        acc_long = acc.x * fx + acc.y * fy
        acc_lat = acc.x * rx + acc.y * ry
    except Exception:
        acc_long = acc_lat = 0.0
    if prev_yaw_deg is not None and dt > 1e-6:
        dyaw = ((yaw_deg - float(prev_yaw_deg) + 180.0) % 360.0) - 180.0
        yaw_rate = math.radians(dyaw) / float(dt)
    else:
        yaw_rate = 0.0

    lat_off, head_err, prog, dist_end, look_alpha = route_geometry_metrics((loc.x, loc.y), yaw_deg, route_xy)
    p_steer, p_thr, p_brk = prev_control
    vec = np.array([
        speed_ms,
        float(target_speed_ms),
        _clamp(acc_long, -15.0, 15.0),
        _clamp(acc_lat, -15.0, 15.0),
        _clamp(yaw_rate, -5.0, 5.0),
        float(p_steer),
        float(p_thr),
        float(p_brk),
        _clamp(lat_off, -20.0, 20.0),
        float(head_err),
        float(prog),
        min(dist_end, 500.0),
    ], dtype=np.float32)
    meta = {
        "x": round(float(loc.x), 3),
        "y": round(float(loc.y), 3),
        "z": round(float(loc.z), 3),
        "yaw_deg": round(yaw_deg, 2),
        "speed_ms": round(speed_ms, 3),
        "target_speed_ms": round(float(target_speed_ms), 3),
        "accel_long": round(float(acc_long), 3),
        "accel_lat": round(float(acc_lat), 3),
        "yaw_rate_rads": round(float(yaw_rate), 3),
        "lane_offset_m": round(lat_off, 3),
        "heading_err_rad": round(head_err, 4),
        "route_progress": round(prog, 4),
        "dist_to_end_m": round(dist_end, 2),
        "lookahead_alpha_rad": round(look_alpha, 4),
    }
    return vec, meta


# ----------------------------------------------------------------------------------------- Controller Interface & Keyboard
class BaseController:
    """Interface for plugging any external user controller/model into `motion_runner`."""

    def reset(self, info=None):
        pass

    def act(self, obs, info=None):
        raise NotImplementedError

    def close(self):
        pass


class KeyboardController(BaseController):
    """Interactive live keyboard driving controller via Pygame (WASD / Arrow keys, Space=handbrake, Q=reverse, ESC=quit).
    Renders a live HUD window showing active smart-car cameras, LiDAR BEV radar, speed, and control bars."""

    def __init__(self, window_title="AutoScenario - Smart Car Control (WASD/Arrows, Space=Brake, ESC=Quit)"):
        self.window_title = window_title
        self._pygame = None
        self._screen = None
        self._steer = 0.0
        self._reverse = False
        self.quit_requested = False

    def _ensure_window(self, width=960, height=540):
        if self._pygame is None:
            import pygame
            pygame.init()
            self._pygame = pygame
        if self._screen is None:
            self._screen = self._pygame.display.set_mode((int(width), int(height)))
            self._pygame.display.set_caption(self.window_title)

    def update_hud(self, bgr_frame):
        """Blit a BGR NumPy mosaic/camera image onto the interactive Pygame HUD window."""
        if bgr_frame is None:
            return
        h, w = bgr_frame.shape[:2]
        self._ensure_window(w, h)
        if self._screen.get_size() != (w, h):
            self._screen = self._pygame.display.set_mode((w, h))
        rgb = bgr_frame[:, :, ::-1]
        surf = self._pygame.surfarray.make_surface(np.transpose(rgb, (1, 0, 2)))
        self._screen.blit(surf, (0, 0))
        self._pygame.display.flip()

    def act(self, obs, info=None):
        self._ensure_window()
        pg = self._pygame
        for ev in pg.event.get():
            if ev.type == pg.QUIT:
                self.quit_requested = True
            elif ev.type == pg.KEYDOWN:
                if ev.key == pg.K_ESCAPE:
                    self.quit_requested = True
                elif ev.key == pg.K_q:
                    self._reverse = not self._reverse

        keys = pg.key.get_pressed()
        throttle = 0.85 if (keys[pg.K_w] or keys[pg.K_UP]) else 0.0
        brake = 1.0 if (keys[pg.K_s] or keys[pg.K_DOWN]) else 0.0
        hand_brake = bool(keys[pg.K_SPACE])

        steer_dir = 0.0
        if keys[pg.K_a] or keys[pg.K_LEFT]:
            steer_dir -= 1.0
        if keys[pg.K_d] or keys[pg.K_RIGHT]:
            steer_dir += 1.0
        if steer_dir != 0.0:
            self._steer = _clamp(self._steer + 0.14 * steer_dir, -0.85, 0.85)
        else:
            self._steer *= 0.65
            if abs(self._steer) < 0.02:
                self._steer = 0.0

        return {
            "throttle": throttle,
            "steer": self._steer,
            "brake": brake,
            "hand_brake": hand_brake,
            "reverse": self._reverse,
        }

    def close(self):
        if self._pygame is not None:
            try:
                self._pygame.display.quit()
            except Exception:
                pass
            self._screen = None


class _CallableControllerWrapper(BaseController):
    def __init__(self, fn):
        self._fn = fn

    def act(self, obs, info=None):
        try:
            return self._fn(obs, info)
        except TypeError:
            return self._fn(obs)


def load_controller(spec):
    """Load an external user controller or interactive keyboard controller.
    Accepted formats:
      - 'keyboard' / 'manual'        -> KeyboardController (interactive WASD / Arrows)
      - 'package.module:ClassOrFunc' -> Imported external Python class (instantiated) or callable
      - 'path/to/file.py:ClassOrFunc'-> Loaded from a Python file path
      - Already-instantiated BaseController or callable
    """
    if spec is None:
        raise ValueError("an external controller is required in external control mode (pass 'module:ClassOrFunc', 'path/to/file.py:act', or a Python callable)")
    if isinstance(spec, BaseController):
        return spec
    if callable(spec):
        return _CallableControllerWrapper(spec)
    raw = str(spec).strip()
    if raw.lower() in ("keyboard", "manual"):
        return KeyboardController()
    if ":" not in raw:
        raise ValueError("unknown controller spec %r (expected 'keyboard', 'module:ClassOrFunc', or 'path/to/file.py:func')" % (raw,))
    mod_part, attr_name = raw.rsplit(":", 1)
    if mod_part.endswith(".py") or os.path.exists(mod_part):
        spec_obj = importlib.util.spec_from_file_location("_user_controller", mod_part)
        if spec_obj is None or spec_obj.loader is None:
            raise ImportError("could not load controller file %r" % (mod_part,))
        mod = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(mod)
    else:
        mod = importlib.import_module(mod_part)
    sym = getattr(mod, attr_name)
    if isinstance(sym, type):
        obj = sym()
        return obj if isinstance(obj, BaseController) else _CallableControllerWrapper(obj)
    return _CallableControllerWrapper(sym)
