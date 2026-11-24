"""
Gymnasium-compatible reinforcement learning & evaluation environment for CARLA corner-case scenarios.

Allows training any ML/RL model (PyTorch PPO/SAC/DQN, Stable-Baselines3, RLlib, CleanRL, Imitation
Learning) to control the smart car using multi-camera vision + 360-degree LiDAR + vehicle telemetry
and learn to avoid the reconstructed crash.

Example:
    from agents.carla_env import CarlaScenarioEnv

    env = CarlaScenarioEnv(
        "results/1",                 # single scenario folder OR variants_manifest.json
        sensors="front_lidar",       # or "tesla_lidar", "surround_lidar", "tesla", "lidar"
        action_space="continuous",   # or "discrete", "target_speed"
    )
    obs, info = env.reset()
    for _ in range(300):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            print("Avoided crash:", info["avoided_crash"])
            obs, info = env.reset()
    env.close()

Works natively with `gymnasium` (when installed, so Stable-Baselines3 accepts `CarlaScenarioEnv`
directly) and cleanly falls back to lightweight built-in Space classes when `gymnasium` is absent.
"""
from dataclasses import dataclass
import glob
import json
import math
import os
import time

import numpy as np

from agents import ego_control, sensor_suite
from agents.intercept import CutInConfig, CutInController
from agents.road_frame import RoadFrame

try:
    import gymnasium as _gym
    from gymnasium import spaces as _gym_spaces
    HAS_GYMNASIUM = True
except ImportError:
    _gym = None
    _gym_spaces = None
    HAS_GYMNASIUM = False


# ----------------------------------------------------------------------------------------- Lightweight Space fallbacks
class _SimpleBox:
    """Minimal Gymnasium-compatible Box space fallback when `gymnasium` is not installed."""

    def __init__(self, low, high, shape=None, dtype=np.float32):
        self.dtype = np.dtype(dtype)
        if shape is None:
            low_a = np.asarray(low, dtype=self.dtype)
            high_a = np.asarray(high, dtype=self.dtype)
            self.shape = low_a.shape
            self.low = low_a
            self.high = high_a
        else:
            self.shape = tuple(shape)
            self.low = np.broadcast_to(np.asarray(low, dtype=self.dtype), self.shape).copy()
            self.high = np.broadcast_to(np.asarray(high, dtype=self.dtype), self.shape).copy()

    def sample(self, rng=None):
        r = rng or np.random.default_rng()
        if np.issubdtype(self.dtype, np.integer):
            return r.integers(self.low, self.high + 1, size=self.shape, dtype=self.dtype)
        lo = np.where(np.isfinite(self.low), self.low, -1.0)
        hi = np.where(np.isfinite(self.high), self.high, 1.0)
        return r.uniform(lo, hi, size=self.shape).astype(self.dtype)

    def contains(self, x):
        arr = np.asarray(x)
        return arr.shape == self.shape and bool(np.all(arr >= self.low - 1e-6) and np.all(arr <= self.high + 1e-6))


class _SimpleDiscrete:
    """Minimal Gymnasium-compatible Discrete space fallback when `gymnasium` is not installed."""

    def __init__(self, n):
        self.n = int(n)
        self.shape = ()
        self.dtype = np.dtype(np.int64)

    def sample(self, rng=None):
        r = rng or np.random.default_rng()
        return int(r.integers(0, self.n))

    def contains(self, x):
        try:
            v = int(x)
        except (TypeError, ValueError):
            return False
        return 0 <= v < self.n


class _SimpleDict:
    """Minimal Gymnasium-compatible Dict space fallback when `gymnasium` is not installed."""

    def __init__(self, spaces_dict):
        self.spaces = dict(spaces_dict)

    def sample(self, rng=None):
        return {k: sp.sample(rng) if not HAS_GYMNASIUM else sp.sample() for k, sp in self.spaces.items()}

    def __getitem__(self, key):
        return self.spaces[key]

    def keys(self):
        return self.spaces.keys()


class _EnvBase:
    metadata = {"render_modes": ["rgb_array"]}

    def reset(self, seed=None, options=None):
        raise NotImplementedError

    def step(self, action):
        raise NotImplementedError

    def render(self):
        return None

    def close(self):
        pass


_BASE_ENV_CLS = _gym.Env if HAS_GYMNASIUM else _EnvBase


def _make_box(low, high, shape=None, dtype=np.float32):
    if HAS_GYMNASIUM:
        return _gym_spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
    return _SimpleBox(low=low, high=high, shape=shape, dtype=dtype)


def _make_discrete(n):
    if HAS_GYMNASIUM:
        return _gym_spaces.Discrete(int(n))
    return _SimpleDiscrete(int(n))


def _make_dict(mapping):
    if HAS_GYMNASIUM:
        return _gym_spaces.Dict(mapping)
    return _SimpleDict(mapping)


# ----------------------------------------------------------------------------------------- Reward shaping
@dataclass
class RewardConfig:
    """Configurable reward weights for learning crash avoidance while staying on-road and making progress."""
    collision_vehicle_penalty: float = -100.0
    collision_impulse_scale: float = -0.002       # extra penalty scaled by impact impulse (clamped to -25)
    collision_scenery_penalty: float = -40.0
    survival_bonus: float = 30.0                  # granted upon clean non-collision completion
    progress_weight: float = 0.35                 # per metre progressed along the reference route
    speed_match_weight: float = 0.04              # small bonus for cruising near target speed when road is clear
    near_miss_margin_m: float = 4.5               # safety bubble radius (centre-to-centre minus car length proxy)
    near_miss_weight: float = -0.60               # smooth penalty inside the safety bubble
    off_road_limit_m: float = 5.5                 # lateral offset beyond which off-road penalty applies
    off_road_terminal_m: float = 11.0             # far off-road terminates episode
    off_road_step_penalty: float = -1.5
    off_road_terminal_penalty: float = -35.0
    standstill_speed_ms: float = 0.8              # below this speed with no nearby threat counts as stalling
    standstill_grace_s: float = 1.5               # seconds before standstill penalty starts
    standstill_penalty: float = -0.15
    action_smooth_weight: float = -0.04           # tiny penalty on harsh steering/pedal jerk
    threat_clear_dist_m: float = 18.0             # distance above which the road ahead is considered clear


def compute_step_reward(cfg, *, collided_vehicle=False, collided_scenery=False, collision_impulse=0.0,
                        delta_progress_m=0.0, speed_ms=0.0, target_speed_ms=14.0,
                        min_obstacle_dist_m=99.0, abs_lane_offset_m=0.0,
                        standstill_time_s=0.0, action_delta_norm=0.0,
                        survived_episode=False, off_road_limit_m=None):
    """Pure, deterministic crash-avoidance reward computation returning `(total_reward, terms_dict, off_road_terminal)`."""
    terms = {}
    off_lim = float(off_road_limit_m if off_road_limit_m is not None else cfg.off_road_limit_m)
    off_term_lim = max(cfg.off_road_terminal_m, off_lim + 4.5)

    # 1. Forward progress along route (capped per tick to avoid teleport spikes)
    prog = max(-1.0, min(4.0, float(delta_progress_m)))
    terms["progress"] = cfg.progress_weight * prog

    # 2. Speed tracking when road is clear (never punishes braking when an obstacle is nearby!)
    threat_near = float(min_obstacle_dist_m) < cfg.threat_clear_dist_m
    if not threat_near and target_speed_ms > 1.0:
        speed_err = abs(float(speed_ms) - float(target_speed_ms)) / max(float(target_speed_ms), 1.0)
        terms["speed_match"] = cfg.speed_match_weight * max(0.0, 1.0 - speed_err)
    else:
        terms["speed_match"] = 0.0

    # 3. Standstill exploit prevention (only when no threat is close)
    if (not threat_near) and float(speed_ms) < cfg.standstill_speed_ms and float(standstill_time_s) > cfg.standstill_grace_s:
        terms["standstill"] = cfg.standstill_penalty
    else:
        terms["standstill"] = 0.0

    # 4. Near-miss / proximity safety shaping
    d_obs = max(0.0, float(min_obstacle_dist_m))
    if d_obs < cfg.near_miss_margin_m:
        frac = (cfg.near_miss_margin_m - d_obs) / max(cfg.near_miss_margin_m, 1e-3)
        terms["near_miss"] = cfg.near_miss_weight * (frac * frac)
    else:
        terms["near_miss"] = 0.0

    # 5. Off-road / carriageway keeping (allows temporary evasive swerves up to `off_lim`)
    off_road_terminal = False
    lat = abs(float(abs_lane_offset_m))
    if lat > off_term_lim:
        terms["off_road"] = cfg.off_road_terminal_penalty
        off_road_terminal = True
    elif lat > off_lim:
        terms["off_road"] = cfg.off_road_step_penalty * min(3.0, (lat - off_lim) / 2.0 + 1.0)
    else:
        terms["off_road"] = 0.0

    # 6. Action smoothness
    terms["smoothness"] = cfg.action_smooth_weight * float(action_delta_norm)

    # 7. Collisions & clean episode completion bonus
    if collided_vehicle:
        imp_pen = max(-25.0, cfg.collision_impulse_scale * float(collision_impulse))
        terms["collision"] = cfg.collision_vehicle_penalty + imp_pen
    elif collided_scenery:
        terms["collision"] = cfg.collision_scenery_penalty
    else:
        terms["collision"] = 0.0

    if survived_episode and not collided_vehicle and not collided_scenery and not off_road_terminal:
        terms["survival"] = cfg.survival_bonus
    else:
        terms["survival"] = 0.0

    total = float(sum(terms.values()))
    return total, {k: round(float(v), 4) for k, v in terms.items()}, off_road_terminal


def flatten_obs_vector(obs):
    """Concatenate compact numeric perception features (`state` + optional `lidar_sectors`) into a 1D float32
    vector for lightweight MLP RL policies (PPO/SAC/DQN/BC)."""
    parts = [np.asarray(obs["state"], dtype=np.float32).reshape(-1)]
    if "lidar_sectors" in obs and obs["lidar_sectors"] is not None:
        parts.append(np.asarray(obs["lidar_sectors"], dtype=np.float32).reshape(-1))
    return np.concatenate(parts, axis=0).astype(np.float32)


# ----------------------------------------------------------------------------------------- Scenario list discovery
def discover_scenarios(source, scene_id=None, use_variants=False):
    """Resolve a scenario folder, a `variants_manifest.json` path, or a list of folders into
    `[(out_dir, scene_id), ...]`."""
    def _find_scene_id(folder, hint=None):
        if hint and os.path.exists(os.path.join(folder, f"{hint}_motion.json")):
            return hint
        default_sid = "video_interpreter_0000_split"
        if os.path.exists(os.path.join(folder, f"{default_sid}_motion.json")):
            return default_sid
        cands = sorted(glob.glob(os.path.join(folder, "*_motion.json")))
        cands = [c for c in cands if not c.endswith("_motion_base.json")]
        if cands:
            return os.path.basename(cands[0])[: -len("_motion.json")]
        raise FileNotFoundError("no *_motion.json found in scenario directory %r" % (folder,))

    if isinstance(source, (list, tuple)):
        out = []
        for item in source:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                out.append((str(item[0]), str(item[1])))
            else:
                out.extend(discover_scenarios(item, scene_id=scene_id, use_variants=use_variants))
        return out

    path = os.path.abspath(str(source))
    if os.path.isfile(path) and path.endswith(".json"):
        with open(path, encoding="utf-8") as f:
            man = json.load(f)
        base_dir = os.path.dirname(path)
        entries = man.get("variants") or []
        out = []
        for e in entries:
            vdir = e["dir"] if os.path.isabs(e.get("dir", "")) else os.path.join(base_dir, e["rel_dir"])
            if not os.path.isdir(vdir) and e.get("rel_dir"):
                vdir = os.path.join(base_dir, e["rel_dir"])
            sid = e.get("scene_id") or _find_scene_id(vdir, scene_id)
            out.append((vdir, sid))
        if not out:
            raise ValueError("variants manifest %r contains no variants" % (path,))
        return out

    if os.path.isdir(path):
        man_fn = os.path.join(path, "variants_manifest.json")
        if use_variants and os.path.exists(man_fn):
            return discover_scenarios(man_fn, scene_id=scene_id, use_variants=True)
        return [(path, _find_scene_id(path, scene_id))]

    raise FileNotFoundError("scenario source not found: %r" % (source,))


# ----------------------------------------------------------------------------------------- CarlaScenarioEnv
class CarlaScenarioEnv(_BASE_ENV_CLS):
    """Gymnasium-compatible CARLA scenario environment for training crash-avoidance policies."""

    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        scenarios,
        scene_id=None,
        use_variants=False,
        sensors="front_lidar",
        sensor_res=(160, 90),
        lidar_bev_size=64,
        lidar_sectors=36,
        action_space="continuous",
        control_actor=None,
        adversarial_cutter=False,
        reward_config=None,
        host="localhost",
        port=2000,
        dt=0.05,
        real_time=False,
        spectator_mode="first_person",
        scenario_order="round_robin",
        terminate_on_collision=True,
        max_steps=None,
        save_sensors_dir=None,
        record_video_dir=None,
        seed=0,
    ):
        super().__init__()
        self.scenario_list = discover_scenarios(scenarios, scene_id=scene_id, use_variants=use_variants)
        self.sensors_spec = sensors
        self.sensor_configs = sensor_suite.resolve_sensor_spec(sensors)
        self.sensor_res = sensor_suite.parse_resolution(sensor_res, default=(160, 90))
        self.lidar_bev_size = int(lidar_bev_size)
        self.lidar_sectors = int(lidar_sectors)
        self.action_space_kind = action_space
        self.control_actor_override = control_actor
        self.adversarial_cutter = bool(adversarial_cutter)
        self.reward_cfg = reward_config or RewardConfig()
        self.host = host
        self.port = int(port)
        self.dt = float(dt)
        self.real_time = bool(real_time)
        self.spectator_mode = spectator_mode
        self.scenario_order = scenario_order
        self.terminate_on_collision = bool(terminate_on_collision)
        self.max_steps_override = int(max_steps) if max_steps is not None else None
        self.save_sensors_dir = save_sensors_dir
        self.record_video_dir = record_video_dir
        self._rng = np.random.default_rng(seed)
        self._ep_counter = 0

        # Build Gym/Gymnasium-compatible action_space and observation_space
        if action_space == "discrete":
            self.action_space = _make_discrete(len(ego_control.DISCRETE_ACTIONS))
        elif action_space == "target_speed":
            self.action_space = _make_box(
                low=np.array([-1.0, 0.0], dtype=np.float32),
                high=np.array([1.0, 36.0], dtype=np.float32),
                dtype=np.float32,
            )
        elif action_space == "continuous":
            self.action_space = _make_box(
                low=np.array([-1.0, -1.0], dtype=np.float32),
                high=np.array([1.0, 1.0], dtype=np.float32),
                dtype=np.float32,
            )
        else:
            raise ValueError("unknown action_space %r" % (action_space,))

        obs_spaces = {
            "state": _make_box(low=-np.inf, high=np.inf, shape=(ego_control.STATE_DIM,), dtype=np.float32),
        }
        cam_w, cam_h = self.sensor_res
        for cfg in self.sensor_configs:
            if cfg["kind"] == "camera":
                obs_spaces["cam_" + cfg["name"]] = _make_box(low=0, high=255, shape=(cam_h, cam_w, 3), dtype=np.uint8)
        if any(cfg["kind"] in ("lidar", "semantic_lidar") for cfg in self.sensor_configs):
            obs_spaces["lidar_bev"] = _make_box(
                low=0.0, high=1.0, shape=(3, self.lidar_bev_size, self.lidar_bev_size), dtype=np.float32
            )
            obs_spaces["lidar_sectors"] = _make_box(
                low=0.0, high=1.0, shape=(self.lidar_sectors,), dtype=np.float32
            )
        self.observation_space = _make_dict(obs_spaces)

        # Runtime CARLA state (lazily connected on first reset)
        self._client = None
        self._world = None
        self._tm = None
        self._loaded_map_sig = None
        self._spawned_ids = []
        self._actors = {}
        self._sensors = []
        self._rig = None
        self._followers = {}
        self._plan = None
        self._agents_meta = None
        self._control_id = None
        self._control_adapter = ego_control.ControlAdapter(action_space=action_space, dt=self.dt)
        self._route_xy = []
        self._road_frame = None
        self._pending_spawns = {}
        self._started = {}
        self._wp_idx = {}
        self._walker_speed = {}
        self._kin_z = {}
        self._kin_spd = {}
        self._collisions = []
        self._id_of = {}
        self._t0_abs = 0.0
        self._tick = 0
        self._max_steps = 300
        self._target_speed_ms = 14.0
        self._prev_progress_m = 0.0
        self._route_total_len = 1.0
        self._prev_yaw_deg = None
        self._prev_action_vec = (0.0, 0.0, 0.0)
        self._standstill_s = 0.0
        self._min_obs_dist_ep = math.inf
        self._collided_veh_ep = False
        self._collided_scenery_ep = False
        self._off_road_limit_m = self.reward_cfg.off_road_limit_m
        self._last_obs = None
        self._last_raw_sensor_obs = None
        self._last_telemetry = None
        self._cur_scenario = None
        self._last_tick_wall = None

    def _ensure_client(self):
        if self._client is not None:
            return
        import carla
        self._client = carla.Client(self.host, self.port)
        self._client.set_timeout(120.0)
        try:
            self._world = self._client.get_world()
        except RuntimeError as e:
            raise RuntimeError("CARLA server not reachable on %s:%s" % (self.host, self.port)) from e
        self._tm = self._client.get_trafficmanager()

    def _cleanup_episode(self):
        if self._rig is not None:
            try:
                self._rig.close()
            except Exception:
                pass
            self._rig = None
        for s in self._sensors:
            try:
                s.stop()
            except Exception:
                pass
        self._sensors = []
        if self._client is not None and self._world is not None:
            import carla
            ids = list(self._spawned_ids)
            seen = set(ids)
            try:
                for a in self._world.get_actors():
                    tid = getattr(a, "type_id", "")
                    if tid.startswith(("vehicle.", "walker.", "sensor.", "static.prop.streetbarrier")) and a.id not in seen:
                        ids.append(a.id)
                        seen.add(a.id)
            except Exception:
                pass
            if ids:
                try:
                    self._client.apply_batch([carla.command.DestroyActor(i) for i in ids])
                except Exception:
                    pass
        self._spawned_ids = []
        self._actors = {}
        self._followers = {}
        self._pending_spawns = {}
        self._started = {}
        self._wp_idx = {}
        self._walker_speed = {}
        self._kin_z = {}
        self._kin_spd = {}
        self._collisions = []
        self._id_of = {}

    def _select_scenario(self, options=None):
        if isinstance(options, dict) and "scenario_index" in options:
            idx = int(options["scenario_index"]) % len(self.scenario_list)
        elif self.scenario_order == "random":
            idx = int(self._rng.integers(0, len(self.scenario_list)))
        else:
            idx = self._ep_counter % len(self.scenario_list)
        return self.scenario_list[idx]

    def _pack_obs(self, state_vec, raw_sensor_obs):
        obs = {"state": state_vec.astype(np.float32)}
        cams_rgb = (raw_sensor_obs or {}).get("cameras") or {}
        if cams_rgb:
            obs["cameras"] = cams_rgb
            for nm, arr in cams_rgb.items():
                obs["cam_" + nm] = arr
        if raw_sensor_obs:
            for k in ("lidar", "lidar_bev", "lidar_sectors", "semantic_lidar", "radar_front"):
                if k in raw_sensor_obs:
                    obs[k] = raw_sensor_obs[k]
        return obs

    def _nearest_obstacle(self):
        """Return `(min_dist_m, (fwd, right))` to the closest other live vehicle/walker relative to controlled vehicle."""
        ctrl_a = self._actors.get(self._control_id)
        if ctrl_a is None or not getattr(ctrl_a, "is_alive", False):
            return math.inf, None
        from agents import motion_runner
        loc = ctrl_a.get_location()
        yaw = ctrl_a.get_transform().rotation.yaw
        best_d = math.inf
        best_rel = None
        for aid, a in self._actors.items():
            if aid == self._control_id or not getattr(a, "is_alive", False):
                continue
            ol = a.get_location()
            if abs(getattr(ol, "z", 0.0) - getattr(loc, "z", 0.0)) > 3.0:
                continue
            d = math.hypot(ol.x - loc.x, ol.y - loc.y)
            if d < best_d:
                best_d = d
                best_rel = motion_runner.rel_in_frame(ol.x - loc.x, ol.y - loc.y, yaw)
        return best_d, best_rel

    def _sync_spectator(self, lead_s=0.0):
        if self.spectator_mode == "none" or self._world is None:
            return
        ctrl_a = self._actors.get(self._control_id)
        if ctrl_a is None or not getattr(ctrl_a, "is_alive", False):
            return
        import carla
        from agents import motion_runner
        try:
            tf = ctrl_a.get_transform()
            if self.spectator_mode == "first_person":
                vtype = (self._agents_meta.get(self._control_id) or {}).get("type", "car")
                bbox = getattr(getattr(ctrl_a, "bounding_box", None), "extent", None)
                mount = motion_runner.dashcam_mount_transform(vtype, bbox)
                vel = ctrl_a.get_velocity()
                wx, wy, wz, wp, wyaw, wroll = motion_runner.predict_dashcam_pose(
                    (tf.location.x, tf.location.y, tf.location.z),
                    (tf.rotation.pitch, tf.rotation.yaw, tf.rotation.roll),
                    mount,
                    vel_xyz=(vel.x, vel.y, vel.z),
                    lead_s=lead_s,
                )
                self._world.get_spectator().set_transform(
                    carla.Transform(carla.Location(wx, wy, wz), carla.Rotation(pitch=wp, yaw=wyaw, roll=wroll))
                )
            elif self.spectator_mode == "chase":
                cx, cy, cz, cpitch, cyaw = motion_runner.crash_focus_transform(
                    tf.location.x, tf.location.y, tf.rotation.yaw, back_m=12.0, side_m=0.0, height_m=5.5
                )
                self._world.get_spectator().set_transform(
                    carla.Transform(carla.Location(cx, cy, cz), carla.Rotation(pitch=cpitch, yaw=cyaw))
                )
        except Exception:
            pass

    def reset(self, seed=None, options=None):
        import carla
        from agents import local_pipeline, map_builder, motion_runner

        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._ensure_client()
        self._cleanup_episode()

        out_dir, scene_id = self._select_scenario(options)
        self._cur_scenario = (out_dir, scene_id)
        self._ep_counter += 1

        obj_fn = os.path.join(out_dir, f"{scene_id}_scene_obj.txt")
        plan_fn = os.path.join(out_dir, f"{scene_id}_motion.json")
        net_fn = os.path.join(out_dir, f"{scene_id}.net.xml")
        agents_meta, statics = motion_runner.read_objects(obj_fn)
        with open(plan_fn, encoding="utf-8") as f:
            plan = json.load(f)
        self._agents_meta = agents_meta
        self._plan = plan

        # Reuse already-loaded OpenDRIVE map whenever the road geometry signature matches!
        sig = local_pipeline._road_map_signature(out_dir, scene_id)
        if sig != self._loaded_map_sig:
            self._world = map_builder.build_and_load_map(self._client, net_fn, out_dir, plan.get("road"))
            self._loaded_map_sig = sig
        world = self._world

        if isinstance(plan.get("environment"), dict):
            try:
                world.set_weather(carla.WeatherParameters(**motion_runner.weather_for(plan["environment"])))
            except Exception:
                pass

        try:
            world.apply_settings(carla.WorldSettings(
                synchronous_mode=True, fixed_delta_seconds=self.dt,
                substepping=True, max_substep_delta_time=0.01, max_substeps=10,
            ))
        except TypeError:
            world.apply_settings(carla.WorldSettings(synchronous_mode=True, fixed_delta_seconds=self.dt))
        self._tm.set_synchronous_mode(True)
        self._tm.set_random_device_seed(int(self._rng.integers(0, 100000)))

        bp_lib = world.get_blueprint_library()
        pa = plan.get("agents") or {}
        road_pl = plan.get("road") or {}
        self._road_frame = RoadFrame(road_pl["centerline"]) if road_pl.get("centerline") else RoadFrame.straight()
        is_kin = lambda aid_: bool((pa.get(aid_) or {}).get("kinematic_track"))

        # Decide which vehicle is controlled BEFORE spawning so we can optionally use Tesla Model 3 blueprint
        provisional_actors = {
            aid: type("_Stub", (), {"type_id": "walker.pedestrian" if a.get("type") == "pedestrian" else "vehicle.car", "is_alive": True})()
            for aid, a in agents_meta.items()
            if (pa.get(aid) or {}).get("mode") != "skip" and float((pa.get(aid) or {}).get("spawn_at_s") or 0.0) <= 0.0
        }
        if self.control_actor_override and self.control_actor_override in provisional_actors:
            self._control_id = self.control_actor_override
        else:
            cands = motion_runner.resolve_dashcam_actors(plan, provisional_actors, agents_meta, mode="primary")
            self._control_id = cands[0] if cands else plan.get("ego", "v1")

        use_tesla_bp = sensor_suite.wants_tesla_blueprint(self.sensors_spec)

        def _spawn_one(aid, a):
            x, y, z = a["location"]
            yaw = a["rotation"][1]
            if a["type"] == "pedestrian":
                bp = bp_lib.filter("walker.pedestrian.*")[0]
            else:
                bp = None
                if aid == self._control_id and use_tesla_bp and a["type"] == "car":
                    tbps = bp_lib.filter("vehicle.tesla.model3")
                    if tbps:
                        bp = tbps[0]
                if bp is None:
                    bp = bp_lib.find(motion_runner.VEHICLE_BLUEPRINTS[a["type"]])
                color = pa.get(aid, {}).get("color")
                if color and bp.has_attribute("color"):
                    bp.set_attribute("color", color)
            pspec = pa.get(aid, {})
            off = float(pspec.get("offset_m", 0.0))
            if pspec.get("polyline") and off:
                sx, sy = motion_runner.polyline_shift(pspec)
                x, y = x + sx, y - sy
                off = 0.0
            if off:
                wp0 = world.get_map().get_waypoint(carla.Location(x, y, 0.0), project_to_road=True)
                cand = wp0.next(off) if off > 0 else wp0.previous(-off)
                if cand:
                    x, y = cand[0].transform.location.x, cand[0].transform.location.y
            heights = (z, 1.0, 0.5)
            if a["type"] != "pedestrian":
                wp_z = world.get_map().get_waypoint(carla.Location(x, y, 0.0), project_to_road=True)
                if wp_z is not None and math.hypot(wp_z.transform.location.x - x, wp_z.transform.location.y - y) < 4.0:
                    gz = wp_z.transform.location.z
                    heights = (gz + 0.25, gz + 0.5, gz + 1.0, z)
            for zz in heights:
                actor = world.try_spawn_actor(bp, carla.Transform(carla.Location(x, y, zz), carla.Rotation(yaw=yaw)))
                if actor is not None:
                    return actor
            return None

        self._spawn_one_fn = _spawn_one

        for aid, a in agents_meta.items():
            pspec = pa.get(aid, {})
            if pspec.get("mode") == "skip":
                continue
            spawn_at_s = float(pspec.get("spawn_at_s") or 0.0)
            if spawn_at_s > 0.0 and aid != self._control_id:
                self._pending_spawns[aid] = spawn_at_s
                continue
            actor = _spawn_one(aid, a)
            if actor is None:
                continue
            if is_kin(aid) and aid != self._control_id:
                try:
                    actor.set_simulate_physics(False)
                except Exception:
                    pass
                self._kin_z[aid] = actor.get_location().z
                self._kin_spd[aid] = float(pspec["kinematic_track"][0][4])
            self._actors[aid] = actor
            self._spawned_ids.append(actor.id)

        for _, o in statics.items():
            bps = bp_lib.filter("static.prop." + o["type"])
            if bps:
                x, y, z = o["location"]
                p = world.try_spawn_actor(bps[0], carla.Transform(carla.Location(x, y, z), carla.Rotation(yaw=o["rotation"][1])))
                if p is not None:
                    self._spawned_ids.append(p.id)
        motion_runner.spawn_barriers(world, bp_lib, road_pl, self._spawned_ids)

        def _record_contact(ev, aid):
            t = world.get_snapshot().timestamp.elapsed_seconds
            self._collisions.append({
                "t": round(t, 2),
                "t_sim": round(t - self._t0_abs, 2),
                "actor": aid,
                "other": ev.other_actor.type_id,
                "other_id": ev.other_actor.id,
                "impulse": round(ev.normal_impulse.length(), 1),
            })

        self._record_contact_fn = _record_contact
        for aid, actor in self._actors.items():
            motion_runner._apply_vehicle_lights(actor, plan.get("environment"))
            if actor.type_id.startswith("vehicle") and not (is_kin(aid) and aid != self._control_id):
                actor.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
                s = world.spawn_actor(bp_lib.find("sensor.other.collision"), carla.Transform(), attach_to=actor)
                s.listen(lambda ev, aid=aid: _record_contact(ev, aid))
                self._sensors.append(s)
                self._spawned_ids.append(s.id)
        self._id_of = {a.id: k for k, a in self._actors.items()}

        for _ in range(6):
            world.tick()

        if self._control_id not in self._actors:
            self._control_id = next(iter(self._actors))

        # Reference route and nominal track for the controlled vehicle
        ctrl_actor = self._actors[self._control_id]
        ctrl_spec = pa.get(self._control_id) or {}
        self._target_speed_ms = max(float(ctrl_spec.get("speed_kmh", 50.0)) / 3.6, 4.0)
        ctrl_fol_ref = self._build_follower(self._control_id, ctrl_actor, ctrl_spec, nominal_targets={})
        self._route_xy = [(wp.transform.location.x, wp.transform.location.y) for wp in ctrl_fol_ref.route]
        if len(self._route_xy) >= 2:
            pts = np.asarray(self._route_xy, dtype=np.float64)
            self._route_total_len = max(float(np.sum(np.hypot(np.diff(pts[:, 0]), np.diff(pts[:, 1])))), 1.0)
        else:
            self._route_total_len = 1.0

        # Build nominal reference trajectory of controlled car so cut-in aggressors don't home unfairly
        nominal_targets = {}
        if not self.adversarial_cutter:
            cloc = ctrl_actor.get_location()
            s0, d0 = self._road_frame.to_sd(cloc.x, -cloc.y)
            nominal_targets[self._control_id] = ego_control.NominalTargetTrack(
                s0=s0, d0=d0,
                speed_kmh=float(ctrl_spec.get("speed_kmh", 50.0)),
                speed_profile=ctrl_spec.get("speed_profile"),
            )
        self._nominal_targets = nominal_targets

        # Build scripted followers for all NON-controlled vehicles
        for aid, spec in pa.items():
            if aid == self._control_id:
                continue
            if aid in self._actors and spec.get("mode") == "drive" and not is_kin(aid):
                fol = self._build_follower(aid, self._actors[aid], spec, nominal_targets=nominal_targets)
                fol.stop_at = spec.get("stop_at_s")
                if spec.get("speed_profile") and not spec.get("intercept"):
                    fol.profile = spec["speed_profile"]
                self._followers[aid] = fol

        # Mount smart-car sensors on the controlled vehicle
        if self.sensor_configs:
            vtype_c = (agents_meta.get(self._control_id) or {}).get("type", "car")
            vid_path = (
                os.path.join(self.record_video_dir, f"ep{self._ep_counter:04d}_{scene_id}_sensors.mp4")
                if self.record_video_dir else None
            )
            save_path = (
                os.path.join(self.save_sensors_dir, f"ep{self._ep_counter:04d}_{scene_id}")
                if self.save_sensors_dir else None
            )
            self._rig = sensor_suite.SmartCarSensorRig(
                world, bp_lib, ctrl_actor, self.sensors_spec, vtype=vtype_c, dt=self.dt,
                image_size=self.sensor_res, video_path=vid_path, save_dir=save_path,
                lidar_bev_size=self.lidar_bev_size, lidar_sectors=self.lidar_sectors,
                spawned_ids=self._spawned_ids,
            )
            for _ in range(5):
                world.tick()
                self._rig.flush_queues(timeout=0.25)

        self._t0_abs = world.get_snapshot().timestamp.elapsed_seconds
        self._collisions.clear()

        # Launch already_moving vehicles (including controlled vehicle so it starts at its road speed!)
        if ctrl_spec.get("already_moving", False) and not motion_runner._has_delayed_start(ctrl_spec):
            ctrl_actor.apply_control(carla.VehicleControl(brake=0.0, hand_brake=False))
            vx, vy = motion_runner._launch_velocity_vec(ctrl_actor, ctrl_fol_ref, ctrl_spec, now=0.0)
            ctrl_actor.set_target_velocity(carla.Vector3D(vx, vy, 0.0))
        else:
            ctrl_actor.apply_control(carla.VehicleControl(brake=0.0, hand_brake=False))

        for aid, fol in self._followers.items():
            if pa[aid].get("already_moving", False) and not motion_runner._has_delayed_start(pa[aid]):
                v = self._actors[aid]
                v.apply_control(carla.VehicleControl(brake=0.0, hand_brake=False))
                vx, vy = motion_runner._launch_velocity_vec(v, fol, pa[aid], now=0.0)
                v.set_target_velocity(carla.Vector3D(vx, vy, 0.0))
                self._started[aid] = True

        # Reset episode trackers
        self._control_adapter.reset()
        self._tick = 0
        dur_s = float(plan.get("duration_s", 12.0))
        self._max_steps = self.max_steps_override or max(40, int(round((dur_s + 1.5) / self.dt)))
        lanes_n = int(road_pl.get("lanes", 2) or 2)
        self._off_road_limit_m = max(self.reward_cfg.off_road_limit_m, lanes_n * 1.75 + 2.0)
        self._prev_yaw_deg = float(ctrl_actor.get_transform().rotation.yaw)
        self._prev_action_vec = (0.0, 0.0, 0.0)
        self._standstill_s = 0.0
        self._min_obs_dist_ep = math.inf
        self._collided_veh_ep = False
        self._collided_scenery_ep = False
        self._last_tick_wall = None

        self._sync_spectator(lead_s=0.0)
        world.tick()
        raw_obs = self._rig.read_tick(timeout=0.5, record_frame=False) if self._rig is not None else {}
        st_vec, st_meta = ego_control.extract_ego_state(
            ctrl_actor, target_speed_ms=self._target_speed_ms, route_xy=self._route_xy,
            prev_control=(0.0, 0.0, 0.0), prev_yaw_deg=self._prev_yaw_deg, dt=self.dt,
        )
        self._prev_progress_m = st_meta["route_progress"] * self._route_total_len
        self._prev_yaw_deg = st_meta["yaw_deg"]
        self._last_raw_sensor_obs = raw_obs
        self._last_obs = self._pack_obs(st_vec, raw_obs)

        min_d, near_rel = self._nearest_obstacle()
        info = {
            "scenario_dir": out_dir,
            "scene_id": scene_id,
            "control_actor": self._control_id,
            "t_sim": 0.0,
            "ego": st_meta,
            "min_obstacle_dist_m": round(min_d, 3) if math.isfinite(min_d) else None,
            "nearest_obstacle_rel": near_rel,
            "collided": False,
            "avoided_crash": False,
        }
        return self._last_obs, info

    def _build_follower(self, aid, vehicle, spec, nominal_targets):
        import carla
        from agents import motion_runner
        world = self._world
        road_pl = self._plan.get("road") or {}
        ego_id = self._plan.get("ego", "v1")
        speed = float(spec.get("speed_kmh", 40.0))
        ic = spec.get("intercept")
        target_id = ic.get("target", ego_id) if ic else None
        target = self._actors.get(target_id) if target_id else None
        if ic and target is not None:
            vl = vehicle.get_location()
            s0 = self._road_frame.to_sd(vl.x, -vl.y)[0]
            route = []
            s_ = s0
            while s_ < self._road_frame.length - 2.0:
                x, y = self._road_frame.to_xy(s_, ic["lane_src_d"])
                route.append(motion_runner._RoutePoint(carla.Location(x, -y, 0.0)))
                s_ += 2.0
            cfg = CutInConfig(
                lane_src_d=float(ic["lane_src_d"]), lane_dst_d=float(ic["lane_dst_d"]),
                t_event_s=float(ic["t_event_s"]), lead_s=float(ic.get("lead_s", 1.4)),
                aim_offset_m=float(ic.get("aim_offset_m", 0.0)),
                max_lateral_speed_ms=float(ic.get("max_lateral_speed_ms", 1.5)),
                cruise_speed_ms=float(ic["cruise_speed_ms"]) if ic.get("cruise_speed_ms") is not None else None,
                long_offset_m=float(ic.get("long_offset_m", 1.0)),
                side_strike=bool(ic.get("side_strike", False)),
            )
            return motion_runner.InterceptFollower(
                vehicle, route, speed, target, self._road_frame, CutInController(cfg),
                nominal_target=nominal_targets.get(target_id),
            )
        if spec.get("polyline"):
            sx, sy = motion_runner.polyline_shift(spec)
            route = [motion_runner._RoutePoint(carla.Location(p_[0] + sx, -(p_[1] + sy), 0.0)) for p_ in spec["polyline"]]
        elif spec.get("path"):
            tgt_carla = [motion_runner._to_carla(pt) for pt in spec["path"]]
            v_loc = vehicle.get_location()
            route = motion_runner.build_route(
                world.get_map(), v_loc, tgt_carla, max_len=max(250.0, self._road_frame.length + 50.0)
            )
            if road_pl.get("centerline") and tgt_carla:
                route = motion_runner.sanitize_straight_route(
                    route, (v_loc.x, v_loc.y), tgt_carla[-1], self._road_frame,
                    motion_runner._RoutePoint, carla.Location, road_plan=road_pl,
                )
        else:
            v_loc = vehicle.get_location()
            route = [motion_runner._RoutePoint(v_loc)]
        return motion_runner.RouteFollower(
            vehicle, route, speed, bool(spec.get("avoid", False)),
            road_frame=(self._road_frame if road_pl.get("centerline") else None),
        )

    def step(self, action):
        import carla
        from agents import motion_runner

        if self._world is None or self._control_id not in self._actors:
            raise RuntimeError("call env.reset() before env.step()")

        world = self._world
        pa = self._plan.get("agents") or {}
        now = self._tick * self.dt
        ego_id = self._plan.get("ego", "v1")
        is_kin = lambda aid_: bool((pa.get(aid_) or {}).get("kinematic_track")) and aid_ != self._control_id
        n_coll_before = len(self._collisions)

        # 1. Pending delayed spawns
        for aid in list(self._pending_spawns):
            if now < self._pending_spawns[aid]:
                continue
            self._pending_spawns.pop(aid)
            actor = self._spawn_one_fn(aid, self._agents_meta[aid])
            if actor is None:
                continue
            self._actors[aid] = actor
            motion_runner._apply_vehicle_lights(actor, self._plan.get("environment"))
            self._spawned_ids.append(actor.id)
            self._id_of[actor.id] = aid
            spec_d = pa[aid]
            if is_kin(aid):
                try:
                    actor.set_simulate_physics(False)
                except Exception:
                    pass
                self._kin_z[aid] = actor.get_location().z
                self._kin_spd[aid] = float(spec_d["kinematic_track"][0][4])
            elif actor.type_id.startswith("vehicle"):
                s = world.spawn_actor(world.get_blueprint_library().find("sensor.other.collision"), carla.Transform(), attach_to=actor)
                s.listen(lambda ev, aid=aid: self._record_contact_fn(ev, aid))
                self._sensors.append(s)
                self._spawned_ids.append(s.id)
                actor.apply_control(carla.VehicleControl(brake=0.0, hand_brake=False))
                if spec_d.get("mode") == "drive":
                    fol = self._build_follower(aid, actor, spec_d, self._nominal_targets)
                    fol.stop_at = spec_d.get("stop_at_s")
                    if spec_d.get("speed_profile") and not spec_d.get("intercept"):
                        fol.profile = spec_d["speed_profile"]
                    self._followers[aid] = fol
                    if spec_d.get("already_moving", False):
                        vx, vy = motion_runner._launch_velocity_vec(actor, fol, spec_d, now=now)
                        actor.set_target_velocity(carla.Vector3D(vx, vy, 0.0))
            self._started[aid] = True

        # 2. Step kinematic replayed vehicles
        for aid in [k for k in list(self._actors) if is_kin(k)]:
            actor = self._actors[aid]
            if not actor.is_alive:
                continue
            kt = pa[aid]["kinematic_track"]
            pose = motion_runner.sample_kinematic_track(kt, now)
            if pose is None:
                continue
            kx, ky, kh_deg, kspd = pose
            wp = world.get_map().get_waypoint(carla.Location(kx, -ky, self._kin_z.get(aid, 0.0)), project_to_road=True)
            z_road = (wp.transform.location.z + 0.05) if wp is not None else self._kin_z.get(aid, 0.2)
            self._kin_z[aid], self._kin_spd[aid] = z_road, kspd
            actor.set_transform(carla.Transform(carla.Location(kx, -ky, z_road), carla.Rotation(yaw=-kh_deg)))

        # 3. Trigger delayed starts for other actors
        def _ref_ttc(trig):
            ref = self._actors.get(trig.get("who", ego_id))
            if ref is None:
                return 1e9
            px, py = motion_runner._to_carla(trig["point"])
            loc = ref.get_location()
            spd = ref.get_velocity().length()
            if spd < 1.0:
                spd = max(float(pa.get(trig.get("who", ego_id), {}).get("speed_kmh", 30)) / 3.6, 1.0)
            return math.hypot(loc.x - px, loc.y - py) / spd

        for aid, spec in pa.items():
            if aid == self._control_id or aid not in self._actors or spec.get("mode") == "static" or aid in self._started:
                continue
            trig = spec.get("start") or {"type": "time", "t": 0.0}
            go = (now >= float(trig.get("t", 0))) if trig["type"] == "time" else (_ref_ttc(trig) <= float(trig.get("seconds", 2)))
            if go:
                self._started[aid] = True
                if spec.get("mode") == "drive":
                    self._actors[aid].apply_control(carla.VehicleControl(brake=0.0, hand_brake=False))
                    if spec.get("already_moving", False) and aid in self._followers:
                        vx, vy = motion_runner._launch_velocity_vec(self._actors[aid], self._followers[aid], spec, now=now)
                        self._actors[aid].set_target_velocity(carla.Vector3D(vx, vy, 0.0))

        # 4. Brake triggers & post-crash braking for scripted followers
        now_abs = world.get_snapshot().timestamp.elapsed_seconds
        for aid, fol in self._followers.items():
            bt = pa[aid].get("brake_trigger")
            ref = self._actors.get(bt.get("who", ego_id)) if bt else None
            if ref is not None and aid in self._started and fol.v.is_alive and ref.is_alive and (fol.stop_at is None or fol.stop_at > now):
                rl, ll, tf = ref.get_location(), fol.v.get_location(), ref.get_transform().get_forward_vector()
                ahead = (ll.x - rl.x) * tf.x + (ll.y - rl.y) * tf.y
                if ahead > 0.0:
                    vr, vl = ref.get_velocity(), fol.v.get_velocity()
                    closing = (vr.x - vl.x) * tf.x + (vr.y - vl.y) * tf.y
                    if motion_runner.ttc(ahead - motion_runner.CAR_LENGTH, closing) <= float(bt.get("ttc_s", 2.0)):
                        fol.stop_at = now
        for c in self._collisions:
            f_ = self._followers.get(c["actor"])
            if f_ is not None and f_.crash_time is None and c["other"].startswith(("vehicle", "walker")):
                f_.crash_time = c["t"]

        # 5. Apply controls: agent action -> controlled car, scripted followers -> NPCs
        ctrl_actor = self._actors[self._control_id]
        cur_spd = ctrl_actor.get_velocity().length() if ctrl_actor.is_alive else 0.0
        vctl = self._control_adapter.to_carla_control(action, carla, speed_ms=cur_spd)
        if ctrl_actor.is_alive:
            ctrl_actor.apply_control(vctl)

        for aid, fol in self._followers.items():
            if aid in self._started and fol.v.is_alive:
                fol.v.apply_control(fol.control([a for k, a in self._actors.items() if k != aid and a.is_alive], self.dt, now_abs, now))

        for aid, spec in pa.items():
            if aid in self._actors and spec.get("mode") == "walk" and aid in self._started:
                w = self._actors[aid]
                i = self._wp_idx.get(aid, 0)
                if i < len(spec["path"]):
                    tx, ty = motion_runner._to_carla(spec["path"][i])
                    loc = w.get_location()
                    dx, dy = tx - loc.x, ty - loc.y
                    d = math.hypot(dx, dy)
                    if d < 0.4:
                        self._wp_idx[aid] = i + 1
                    else:
                        cur = min(float(spec.get("speed_ms", 1.4)), self._walker_speed.get(aid, 0.0) + 1.5 * self.dt)
                        self._walker_speed[aid] = cur
                        w.apply_control(carla.WalkerControl(direction=carla.Vector3D(dx / d, dy / d, 0.0), speed=cur))
                else:
                    w.apply_control(carla.WalkerControl(speed=0.0))

        # 6. Tick CARLA synchronously
        self._sync_spectator(lead_s=self.dt)
        world.tick()
        if self.real_time:
            wnow = time.perf_counter()
            if self._last_tick_wall is not None:
                wait = self.dt - (wnow - self._last_tick_wall)
                if wait > 0:
                    time.sleep(wait)
            self._last_tick_wall = time.perf_counter()
        self._tick += 1
        t_sim = round(self._tick * self.dt, 3)

        # 7. Evaluate new collisions involving the controlled car
        new_colls = self._collisions[n_coll_before:]
        step_coll_veh = False
        step_coll_scenery = False
        step_impulse = 0.0
        for c in new_colls:
            other_aid = self._id_of.get(c["other_id"])
            if c["actor"] == self._control_id or other_aid == self._control_id:
                if str(c["other"]).startswith(("vehicle", "walker")):
                    step_coll_veh = True
                    step_impulse = max(step_impulse, float(c.get("impulse", 0.0)))
                elif str(c["other"]) != "static.road":
                    step_coll_scenery = True
                    step_impulse = max(step_impulse, float(c.get("impulse", 0.0)))
        if step_coll_veh:
            self._collided_veh_ep = True
        if step_coll_scenery:
            self._collided_scenery_ep = True

        # 8. Extract ego state, obstacle distances, and sensor observations
        cur_ctrl = (self._control_adapter.prev_steer, self._control_adapter.prev_throttle, self._control_adapter.prev_brake)
        act_delta = abs(cur_ctrl[0] - self._prev_action_vec[0]) + 0.5 * (
            abs(cur_ctrl[1] - self._prev_action_vec[1]) + abs(cur_ctrl[2] - self._prev_action_vec[2])
        )
        self._prev_action_vec = cur_ctrl

        st_vec, st_meta = ego_control.extract_ego_state(
            ctrl_actor, target_speed_ms=self._target_speed_ms, route_xy=self._route_xy,
            prev_control=cur_ctrl, prev_yaw_deg=self._prev_yaw_deg, dt=self.dt,
        )
        self._prev_yaw_deg = st_meta["yaw_deg"]
        cur_prog_m = st_meta["route_progress"] * self._route_total_len
        delta_prog = cur_prog_m - self._prev_progress_m
        self._prev_progress_m = cur_prog_m

        min_d, near_rel = self._nearest_obstacle()
        if min_d < self._min_obs_dist_ep:
            self._min_obs_dist_ep = min_d
        if st_meta["speed_ms"] < self.reward_cfg.standstill_speed_ms and min_d >= self.reward_cfg.threat_clear_dist_m:
            self._standstill_s += self.dt
        else:
            self._standstill_s = 0.0

        truncated = self._tick >= self._max_steps
        collided_any = self._collided_veh_ep or self._collided_scenery_ep
        survived = truncated and not collided_any

        reward, r_terms, off_road_term = compute_step_reward(
            self.reward_cfg,
            collided_vehicle=step_coll_veh,
            collided_scenery=step_coll_scenery,
            collision_impulse=step_impulse,
            delta_progress_m=delta_prog,
            speed_ms=st_meta["speed_ms"],
            target_speed_ms=self._target_speed_ms,
            min_obstacle_dist_m=min_d,
            abs_lane_offset_m=st_meta["lane_offset_m"],
            standstill_time_s=self._standstill_s,
            action_delta_norm=act_delta,
            survived_episode=survived,
            off_road_limit_m=self._off_road_limit_m,
        )
        terminated = bool(off_road_term or (self.terminate_on_collision and (step_coll_veh or step_coll_scenery)))
        if terminated:
            survived = False

        telemetry = {
            "t_sim": t_sim,
            "control_mode": self.action_space_kind,
            "speed_ms": st_meta["speed_ms"],
            "steer": cur_ctrl[0],
            "throttle": cur_ctrl[1],
            "brake": cur_ctrl[2],
            "lane_offset_m": st_meta["lane_offset_m"],
            "heading_err_rad": st_meta["heading_err_rad"],
            "min_obstacle_dist_m": round(min_d, 3) if math.isfinite(min_d) else None,
            "collided": collided_any,
            "reward": round(reward, 4),
            "ego_state": [round(float(x), 4) for x in st_vec],
        }
        self._last_telemetry = telemetry

        raw_obs = self._rig.read_tick(timeout=0.5, telemetry=telemetry, record_frame=True) if self._rig is not None else {}
        self._last_raw_sensor_obs = raw_obs
        self._last_obs = self._pack_obs(st_vec, raw_obs)

        info = {
            "t_sim": t_sim,
            "step": self._tick,
            "control_actor": self._control_id,
            "ego": st_meta,
            "control": {"steer": cur_ctrl[0], "throttle": cur_ctrl[1], "brake": cur_ctrl[2]},
            "reward_terms": r_terms,
            "collided": collided_any,
            "collided_vehicle": self._collided_veh_ep,
            "collided_scenery": self._collided_scenery_ep,
            "off_road_terminal": off_road_term,
            "avoided_crash": bool(survived and not off_road_term),
            "min_obstacle_dist_m": round(min_d, 3) if math.isfinite(min_d) else None,
            "episode_min_dist_m": round(self._min_obs_dist_ep, 3) if math.isfinite(self._min_obs_dist_ep) else None,
            "nearest_obstacle_rel": near_rel,
        }
        return self._last_obs, reward, terminated, truncated, info

    def render(self):
        """Return the latest BGR multi-camera + LiDAR BEV mosaic frame (if sensors are enabled)."""
        if self._rig is not None and self._last_raw_sensor_obs is not None:
            return self._rig.render_mosaic(self._last_raw_sensor_obs, telemetry=self._last_telemetry)
        return None

    def close(self):
        self._cleanup_episode()
        if self._world is not None:
            try:
                import carla
                self._world.apply_settings(carla.WorldSettings(synchronous_mode=False, fixed_delta_seconds=0.0))
                if self._tm is not None:
                    self._tm.set_synchronous_mode(False)
            except Exception:
                pass
        self._world = None
        self._client = None
        self._tm = None
        self._loaded_map_sig = None
