"""Tests for smart-car multi-camera + 360-deg LiDAR suite, controllable ego interface
(continuous / discrete / target_speed action spaces, external controller loading, nominal cut-in
tracks), and Gymnasium-compatible CarlaScenarioEnv."""
import json
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents import (
    carla_env,
    ego_control,
    intercept,
    local_pipeline,
    map_builder,
    motion_runner,
    scene_builder,
    sensor_suite,
)
from agents.road_frame import RoadFrame

FIX = ROOT / "tests" / "fixtures"


# ----------------------------------------------------------------------------------------- Mocked CARLA Harness
def _build_mock_carla_world(obstacle_ahead_x=None, dt=0.05):
    """Lightweight deterministic mocked CARLA world with physics, RGB cameras, 360-deg LiDAR,
    collision detection, and OpenDRIVE map load counting."""
    class _Vec3:
        def __init__(self, x=0.0, y=0.0, z=0.0):
            self.x, self.y, self.z = float(x), float(y), float(z)
        def length(self):
            return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    class _Loc(_Vec3):
        def distance(self, other):
            return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)

    class _Rot:
        def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
            self.pitch, self.yaw, self.roll = float(pitch), float(yaw), float(roll)

    class _Tf:
        def __init__(self, loc=None, rot=None):
            self.location = loc or _Loc()
            self.rotation = rot or _Rot()
        def get_forward_vector(self):
            r = math.radians(self.rotation.yaw)
            return _Vec3(math.cos(r), math.sin(r), 0.0)
        def get_right_vector(self):
            r = math.radians(self.rotation.yaw)
            return _Vec3(-math.sin(r), math.cos(r), 0.0)

    class _VCtl:
        def __init__(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False, reverse=False):
            self.throttle = float(throttle)
            self.steer = float(steer)
            self.brake = float(brake)
            self.hand_brake = bool(hand_brake)
            self.reverse = bool(reverse)

    class _WCtl:
        def __init__(self, direction=None, speed=0.0):
            self.direction, self.speed = direction, float(speed)

    class _Settings:
        def __init__(self, synchronous_mode=False, fixed_delta_seconds=0.0, **kw):
            self.synchronous_mode = synchronous_mode
            self.fixed_delta_seconds = fixed_delta_seconds

    class _Weather:
        def __init__(self, **kw):
            self.params = dict(kw)

    class _Bp:
        def __init__(self, tid):
            self.id = tid
            self.attrs = {}
        def has_attribute(self, k):
            return True
        def set_attribute(self, k, v):
            self.attrs[k] = v

    class _BpLib:
        def __init__(self):
            self.created = []
        def find(self, tid):
            b = _Bp(tid)
            self.created.append(b)
            return b
        def filter(self, pat):
            if pat == "vehicle.tesla.model3":
                b = _Bp("vehicle.tesla.model3")
                self.created.append(b)
                return [b]
            if pat.startswith("walker."):
                return [_Bp("walker.pedestrian.0001")]
            return [_Bp(pat)]

    class _Actor:
        def __init__(self, aid, tid, tf, parent=None, bp_attrs=None, attachment_type=None):
            self.id = aid
            self.type_id = tid
            self._tf = tf
            self.parent = parent
            self.bp_attrs = dict(bp_attrs or {})
            self.attachment_type = attachment_type
            self.is_alive = True
            self._vel = _Vec3()
            self._pending_vel = None
            self._acc = _Vec3()
            self._ctrl = _VCtl()
            self._cb = None
            self.bounding_box = NS(extent=_Vec3(2.35, 0.95, 0.74))
        def get_transform(self):
            return _Tf(_Loc(self._tf.location.x, self._tf.location.y, self._tf.location.z),
                       _Rot(self._tf.rotation.pitch, self._tf.rotation.yaw, self._tf.rotation.roll))
        def get_location(self):
            return _Loc(self._tf.location.x, self._tf.location.y, self._tf.location.z)
        def get_velocity(self):
            return _Vec3(self._vel.x, self._vel.y, self._vel.z)
        def get_acceleration(self):
            return _Vec3(self._acc.x, self._acc.y, self._acc.z)
        def set_transform(self, tf):
            self._tf = tf
        def apply_control(self, c):
            self._ctrl = c
        def set_autopilot(self, v):
            pass
        def set_simulate_physics(self, v):
            pass
        def set_target_velocity(self, v):
            self._pending_vel = _Vec3(v.x, v.y, v.z)
        def set_target_angular_velocity(self, v):
            pass
        def get_physics_control(self):
            return NS(wheels=[NS(max_steer_angle=70.0)])
        def listen(self, cb):
            self._cb = cb
        def stop(self):
            self._cb = None
        def destroy(self):
            self.is_alive = False

    class _Wp:
        def __init__(self, x, y):
            self.transform = _Tf(_Loc(x, y, 0.0), _Rot(0.0, 0.0, 0.0))
            self.is_junction = False
        def next(self, step):
            return [_Wp(self.transform.location.x + step, self.transform.location.y)]
        def previous(self, step):
            return [_Wp(self.transform.location.x - step, self.transform.location.y)]

    class _Map:
        def get_waypoint(self, loc, project_to_road=True, lane_type=None):
            return _Wp(loc.x, loc.y)

    spectator = _Actor(9999, "spectator", _Tf())
    bp_lib = _BpLib()
    stats = {"map_loads": 0, "spawned_sensors": [], "spawned_vehicles": []}

    class _World:
        def __init__(self):
            self._t = 0.0
            self._next_id = 1
            self._actors = []
            self._dt = float(dt)
            self._weather = None
        def get_map(self):
            return _Map()
        def get_blueprint_library(self):
            return bp_lib
        def get_spectator(self):
            return spectator
        def get_actors(self):
            return [a for a in self._actors if a.is_alive]
        def get_snapshot(self):
            return NS(timestamp=NS(elapsed_seconds=self._t))
        def apply_settings(self, s):
            if getattr(s, "fixed_delta_seconds", 0.0):
                self._dt = float(s.fixed_delta_seconds)
        def set_weather(self, w):
            self._weather = w
        def try_spawn_actor(self, bp, tf, attach_to=None, attachment_type=None):
            a = _Actor(self._next_id, bp.id, tf, parent=attach_to, bp_attrs=bp.attrs, attachment_type=attachment_type)
            self._next_id += 1
            self._actors.append(a)
            if bp.id.startswith("sensor."):
                stats["spawned_sensors"].append(a)
            elif bp.id.startswith("vehicle."):
                stats["spawned_vehicles"].append(a)
            return a
        def spawn_actor(self, bp, tf, attach_to=None, attachment_type=None):
            return self.try_spawn_actor(bp, tf, attach_to=attach_to, attachment_type=attachment_type)
        def tick(self):
            self._t += self._dt
            vehs = [a for a in self._actors if a.is_alive and a.type_id.startswith("vehicle.")]
            for v in vehs:
                if v._pending_vel is not None:
                    v._vel = v._pending_vel
                    v._pending_vel = None
                spd = v._vel.length()
                yaw_rad = math.radians(v._tf.rotation.yaw)
                if v._ctrl.hand_brake:
                    v._vel = _Vec3(0.0, 0.0, 0.0)
                    v._acc = _Vec3(0.0, 0.0, 0.0)
                    continue
                a_long = 3.5 * v._ctrl.throttle - 7.5 * v._ctrl.brake
                spd_new = max(0.0, min(35.0, spd + a_long * self._dt))
                yaw_new = yaw_rad + (spd_new / 2.8) * math.tan(0.55 * v._ctrl.steer) * self._dt
                v._tf.rotation.yaw = math.degrees(yaw_new)
                vx, vy = spd_new * math.cos(yaw_new), spd_new * math.sin(yaw_new)
                v._tf.location.x += 0.5 * (v._vel.x + vx) * self._dt
                v._tf.location.y += 0.5 * (v._vel.y + vy) * self._dt
                v._acc = _Vec3((vx - v._vel.x) / self._dt, (vy - v._vel.y) / self._dt, 0.0)
                v._vel = _Vec3(vx, vy, 0.0)

            for s in self._actors:
                if not s.is_alive or s._cb is None:
                    continue
                if s.type_id == "sensor.other.collision" and s.parent is not None:
                    p = s.parent
                    for o in vehs:
                        if o is p:
                            continue
                        if p.get_location().distance(o.get_location()) < 3.8:
                            ev = NS(
                                other_actor=o,
                                normal_impulse=_Vec3(500.0, 0.0, 0.0),
                                transform=p.get_transform(),
                            )
                            s._cb(ev)
                elif s.type_id == "sensor.camera.rgb":
                    w = int(s.bp_attrs.get("image_size_x", 160))
                    h = int(s.bp_attrs.get("image_size_y", 90))
                    raw = np.full((h, w, 4), 120, dtype=np.uint8).tobytes()
                    s._cb(NS(width=w, height=h, raw_data=raw))
                elif s.type_id == "sensor.lidar.ray_cast" and s.parent is not None:
                    p = s.parent
                    pl = p.get_location()
                    pyaw = math.radians(p.get_transform().rotation.yaw)
                    c, sn = math.cos(pyaw), math.sin(pyaw)
                    pts = [[10.0, 0.0, -1.75, 0.2]]
                    for o in vehs:
                        if o is p:
                            continue
                        dx = o.get_location().x - pl.x
                        dy = o.get_location().y - pl.y
                        pts.append([dx * c + dy * sn, -dx * sn + dy * c, -0.4, 0.9])
                    if obstacle_ahead_x is not None:
                        dx = float(obstacle_ahead_x) - pl.x
                        dy = 0.0 - pl.y
                        pts.append([dx * c + dy * sn, -dx * sn + dy * c, -0.3, 0.95])
                    raw = np.asarray(pts, dtype=np.float32).tobytes()
                    s._cb(NS(raw_data=raw))

    world = _World()
    tm = NS(set_synchronous_mode=lambda v: None, set_random_device_seed=lambda s: None)

    def _apply_batch(cmds):
        id_set = set(cmds)
        for a in world._actors:
            if a.id in id_set:
                a.is_alive = False

    client = NS(
        set_timeout=lambda t: None,
        get_world=lambda: world,
        get_trafficmanager=lambda: tm,
        apply_batch=_apply_batch,
    )
    fake_carla = types.SimpleNamespace(
        Client=lambda h, p: client,
        WorldSettings=_Settings,
        WeatherParameters=_Weather,
        Location=_Loc,
        Rotation=_Rot,
        Transform=_Tf,
        Vector3D=_Vec3,
        VehicleControl=_VCtl,
        WalkerControl=_WCtl,
        LaneType=NS(Driving=1),
        AttachmentType=NS(Rigid="RIGID", SpringArm="SPRING"),
        command=NS(DestroyActor=lambda i: i),
    )

    def fake_build_and_load_map(c, net_fn, out_dir, road_plan=None):
        stats["map_loads"] += 1
        return world

    return fake_carla, world, stats, fake_build_and_load_map


def _write_minimal_scenario(folder, scene_id="sc", speed_kmh=54.0, lead_x=None, dur=2.0):
    objs = {"v1": {"type": "car", "location": [10.0, 0.0, 0.0], "rotation": [0.0, 0.0, 0.0]}}
    agents_plan = {
        "v1": {"mode": "drive", "speed_kmh": speed_kmh, "already_moving": True, "path": [[220.0, 0.0]]},
    }
    expect = {"type": "none", "actors": ["v1"], "time_s": dur}
    if lead_x is not None:
        objs["v2"] = {"type": "car", "location": [float(lead_x), 0.0, 0.0], "rotation": [0.0, 0.0, 0.0]}
        agents_plan["v2"] = {"mode": "static"}
        expect = {"type": "collision", "actors": ["v1", "v2"], "time_s": 1.5, "window_s": 1.5}
    with open(os.path.join(folder, f"{scene_id}_scene_obj.txt"), "w", encoding="utf-8") as f:
        f.write(repr(objs) + "\n{}\n")
    plan = {
        "scene_id": scene_id,
        "ego": "v1",
        "duration_s": dur,
        "expect": expect,
        "road": {"lanes": 2, "one_way": False, "centerline": [[0.0, 0.0], [250.0, 0.0]]},
        "agents": agents_plan,
    }
    with open(os.path.join(folder, f"{scene_id}_motion.json"), "w", encoding="utf-8") as f:
        json.dump(plan, f)
    Path(folder, f"{scene_id}.nod.xml").write_text("<nodes/>", encoding="utf-8")
    Path(folder, f"{scene_id}.edg.xml").write_text("<edges/>", encoding="utf-8")
    Path(folder, f"{scene_id}.net.xml").write_text("<net/>", encoding="utf-8")


# ----------------------------------------------------------------------------------------- 1. Sensor Suite Tests
class TestSmartCarSensorSuite(unittest.TestCase):
    def test_presets_and_custom_sensor_lists(self):
        self.assertEqual(sensor_suite.resolve_sensor_spec(None), [])
        self.assertEqual(sensor_suite.resolve_sensor_spec("none"), [])

        tesla = sensor_suite.resolve_sensor_spec("tesla")
        self.assertEqual(len(tesla), 8)
        self.assertTrue(all(c["kind"] == "camera" for c in tesla))
        yaws = {c["name"]: c["yaw"] for c in tesla}
        self.assertEqual(yaws["front_main"], 0.0)
        self.assertLess(yaws["left_pillar"], -45.0)
        self.assertGreater(yaws["right_pillar"], 45.0)
        self.assertLess(yaws["left_repeater"], -120.0)
        self.assertGreater(yaws["right_repeater"], 120.0)
        self.assertEqual(yaws["rear"], 180.0)

        tl = sensor_suite.resolve_sensor_spec("tesla_lidar")
        self.assertEqual(len(tl), 9)
        self.assertEqual([c["kind"] for c in tl].count("lidar"), 1)
        self.assertTrue(sensor_suite.wants_tesla_blueprint("tesla_lidar"))
        self.assertFalse(sensor_suite.wants_tesla_blueprint("front_lidar"))

        custom = sensor_suite.resolve_sensor_spec("front_wide,lidar,rear,lidar")
        self.assertEqual([c["name"] for c in custom], ["front_wide", "lidar", "rear"])

        with self.assertRaises(ValueError):
            sensor_suite.resolve_sensor_spec("not_a_real_sensor")

    def test_mount_geometry_scales_across_vehicle_classes(self):
        lidar_cfg = sensor_suite.SENSOR_CATALOG["lidar"]
        left_p_cfg = sensor_suite.SENSOR_CATALOG["left_pillar"]
        rear_cfg = sensor_suite.SENSOR_CATALOG["rear"]

        z_car = sensor_suite.sensor_mount_transform(lidar_cfg, "car")[2]
        z_suv = sensor_suite.sensor_mount_transform(lidar_cfg, "suv")[2]
        z_van = sensor_suite.sensor_mount_transform(lidar_cfg, "van")[2]
        z_trk = sensor_suite.sensor_mount_transform(lidar_cfg, "truck")[2]
        self.assertLess(z_car, z_suv)
        self.assertLess(z_suv, z_van)
        self.assertLess(z_van, z_trk)
        self.assertGreater(z_car, sensor_suite.VEHICLE_DIMENSIONS["car"][2] + 0.15)
        self.assertLess(sensor_suite.sensor_mount_transform(left_p_cfg, "car")[1], -0.95)
        self.assertLess(sensor_suite.sensor_mount_transform(rear_cfg, "car")[0], -2.35)

    def test_raw_decoders_bev_and_360_polar_range_sectors(self):
        raw_px = bytes([10, 20, 30, 255, 40, 50, 60, 255])
        rgb = sensor_suite.bgra_to_rgb(raw_px, 2, 1)
        self.assertEqual(rgb[0, 0].tolist(), [30, 20, 10])
        self.assertEqual(rgb[0, 1].tolist(), [60, 50, 40])

        pts = np.array([
            [12.0, 0.0, -0.4, 0.8],    # straight ahead (+X), 12 m -> sector 0
            [0.0, 24.0, -0.2, 0.9],    # right (+Y), 24 m -> sector 9
            [-18.0, 0.0, 0.0, 0.7],    # rear (-X), 18 m -> sector 18
            [0.0, -30.0, 0.2, 0.6],    # left (-Y), 30 m -> sector 27
            [5.0, 0.0, -1.8, 0.3],     # ground plane -> ignored
            [0.8, 0.1, 0.0, 1.0],      # ego roof self-hit -> ignored
        ], dtype=np.float32)
        dec = sensor_suite.lidar_bytes_to_points(pts.tobytes())
        self.assertEqual(dec.shape, (6, 4))

        bev = sensor_suite.lidar_to_bev(dec, range_m=40.0, size=64)
        self.assertEqual(bev.shape, (3, 64, 64))
        self.assertGreater(float(bev[1].max()), 0.0)
        self.assertLessEqual(float(bev.max()), 1.0)

        sec = sensor_suite.lidar_to_range_sectors(dec, n_sectors=36, max_range=60.0)
        self.assertEqual(sec.shape, (36,))
        self.assertAlmostEqual(float(sec[0]) * 60.0, 12.0, delta=0.2)
        self.assertAlmostEqual(float(sec[9]) * 60.0, 24.0, delta=0.2)
        self.assertAlmostEqual(float(sec[18]) * 60.0, 18.0, delta=0.2)
        self.assertAlmostEqual(float(sec[27]) * 60.0, 30.0, delta=0.2)
        self.assertAlmostEqual(float(sec[4]), 1.0, places=4)

        cams = {nm: np.full((90, 160, 3), 80, dtype=np.uint8) for nm in ("front_wide", "rear")}
        mosaic = sensor_suite.compose_sensor_mosaic(
            cams, lidar_points=dec,
            telemetry={"t_sim": 1.5, "speed_ms": 14.0, "steer": -0.2, "throttle": 0.0, "brake": 0.8, "collided": False},
            tile_size=(160, 90),
        )
        self.assertEqual(mosaic.ndim, 3)
        self.assertEqual(mosaic.shape[2], 3)
        self.assertGreater(mosaic.shape[0], 90)


# ----------------------------------------------------------------------------------------- 2. Vehicle Control & Action Spaces
class TestEgoControlInterface(unittest.TestCase):
    def test_control_adapter_continuous_discrete_and_target_speed(self):
        ca = ego_control.ControlAdapter("continuous", smooth=False)
        thr, st, brk, hb, rev = ca.apply([0.4, 0.7])
        self.assertAlmostEqual(thr, 0.7, places=4)
        self.assertAlmostEqual(st, 0.4, places=4)
        self.assertEqual(brk, 0.0)
        thr2, st2, brk2, _, _ = ca.apply([-0.6, -0.9])
        self.assertEqual(thr2, 0.0)
        self.assertAlmostEqual(st2, -0.6, places=4)
        self.assertAlmostEqual(brk2, 0.9, places=4)

        cd = ego_control.ControlAdapter("discrete", smooth=False)
        for idx in range(len(ego_control.DISCRETE_ACTIONS)):
            t_, s_, b_, _, _ = cd.apply(idx)
            self.assertTrue(0.0 <= t_ <= 1.0 and -1.0 <= s_ <= 1.0 and 0.0 <= b_ <= 1.0)
        self.assertEqual(cd.apply(3)[2], 1.0)
        self.assertLess(cd.apply(6)[1], 0.0)

        ct = ego_control.ControlAdapter("target_speed", smooth=False)
        t_acc, _, b_acc, _, _ = ct.apply([0.0, 20.0], speed_ms=5.0)
        self.assertGreater(t_acc, 0.5)
        self.assertEqual(b_acc, 0.0)
        t_brk, _, b_brk, _, _ = ct.apply([0.0, 0.0], speed_ms=15.0)
        self.assertEqual(t_brk, 0.0)
        self.assertGreater(b_brk, 0.7)

    def test_nominal_target_track_stops_cutter_from_chasing_evading_ego(self):
        fr = RoadFrame([[0.0, 0.0], [300.0, 0.0]])
        nom = ego_control.NominalTargetTrack(s0=20.0, d0=-5.25, speed_kmh=54.0)
        s5, d5, v5 = nom.pose_at(5.0)
        self.assertAlmostEqual(s5, 95.0, places=2)
        self.assertAlmostEqual(d5, -5.25, places=2)
        self.assertAlmostEqual(v5, 15.0, places=2)

        cfg = intercept.CutInConfig(lane_src_d=-1.75, lane_dst_d=-5.25, t_event_s=5.0, lead_s=1.4)
        ctl = intercept.CutInController(cfg)

        class _Loc:
            def __init__(self, x, y, z=0.0): self.x, self.y, self.z = x, y, z
        class _Vec:
            def __init__(self, x=0.0, y=0.0): self.x, self.y = x, y
            def length(self): return math.hypot(self.x, self.y)
        class _Veh:
            def __init__(self, x, y, vx): self._l, self._v = _Loc(x, y), _Vec(vx, 0.0)
            def get_location(self): return self._l
            def get_velocity(self): return self._v
            def get_physics_control(self): return NS(wheels=[NS(max_steer_angle=70.0)])

        saved = sys.modules.get("carla")
        sys.modules["carla"] = types.SimpleNamespace(Location=_Loc)
        try:
            ego_evading = _Veh(25.0, 8.5, 0.0)
            cutter = _Veh(88.0, 1.75, 15.0)
            route = [motion_runner._RoutePoint(_Loc(float(s), 1.75)) for s in range(0, 200, 4)]
            fol = motion_runner.InterceptFollower(cutter, route, 54.0, ego_evading, fr, ctl, nominal_target=nom)
            for k in range(30):
                t_s = 3.6 + k * 0.05
                pt = fol.pursuit_point(_Loc(20.0 + 15.0 * t_s, 1.75), 15.0, 10.0, t_s)
            _, d_aim = fr.to_sd(pt.x, -pt.y)
            self.assertGreater(fol.speed, 12.0)
            self.assertGreater(d_aim, -6.0)
        finally:
            if saved is None:
                sys.modules.pop("carla", None)
            else:
                sys.modules["carla"] = saved

    def test_load_controller_external_file_and_rejects_missing_spec(self):
        with self.assertRaises(ValueError):
            ego_control.load_controller(None)
        with self.assertRaises(ValueError):
            ego_control.load_controller("invalid_without_colon")
        self.assertIsInstance(ego_control.load_controller("keyboard"), ego_control.KeyboardController)
        with tempfile.TemporaryDirectory() as tmp:
            fn = os.path.join(tmp, "user_ctrl.py")
            Path(fn).write_text("def act(obs): return [0.1, -0.5]\n", encoding="utf-8")
            ctrl = ego_control.load_controller(fn + ":act")
            self.assertEqual(ctrl.act({"state": np.zeros(12)}), [0.1, -0.5])


# ----------------------------------------------------------------------------------------- 3. Rewards, Runner & RL Env
class TestCarlaEnvAndRunnerIntegration(unittest.TestCase):
    def test_reward_shaping_penalizes_crashes_and_standstill_rewards_clean_avoidance(self):
        cfg = carla_env.RewardConfig()
        r_crash, terms_c, _ = carla_env.compute_step_reward(cfg, collided_vehicle=True, collision_impulse=1000.0)
        self.assertLess(r_crash, -100.0)
        self.assertLess(terms_c["collision"], -100.0)

        r_surv, terms_s, off_t = carla_env.compute_step_reward(
            cfg, delta_progress_m=0.8, speed_ms=14.0, target_speed_ms=14.0, survived_episode=True
        )
        self.assertFalse(off_t)
        self.assertGreater(r_surv, 30.0)
        self.assertEqual(terms_s["survival"], cfg.survival_bonus)

        _, terms_brk_threat, _ = carla_env.compute_step_reward(
            cfg, speed_ms=0.2, min_obstacle_dist_m=5.0, standstill_time_s=3.0
        )
        self.assertEqual(terms_brk_threat["standstill"], 0.0)

        _, terms_stall, _ = carla_env.compute_step_reward(
            cfg, speed_ms=0.2, min_obstacle_dist_m=50.0, standstill_time_s=3.0
        )
        self.assertLess(terms_stall["standstill"], 0.0)

        r_off, _, off_term = carla_env.compute_step_reward(cfg, abs_lane_offset_m=15.0)
        self.assertTrue(off_term)
        self.assertLess(r_off, -30.0)

    def test_run_motion_auto_playback_with_tesla_lidar_records_dataset_and_video(self):
        """Default --control auto plays the scenario normally while capturing 8 Tesla cameras + 20 Hz 360-deg LiDAR."""
        fake_carla, world, stats, _ = _build_mock_carla_world(dt=motion_runner.DT)
        saved_carla = sys.modules.get("carla")
        sys.modules["carla"] = fake_carla
        try:
            with tempfile.TemporaryDirectory() as tmp:
                _write_minimal_scenario(tmp, "sc", speed_kmh=54.0, lead_x=None, dur=1.2)
                res = motion_runner.run_motion(
                    tmp, "sc", load_map=False, hold_sec=0.2, real_time=False,
                    sensors="tesla_lidar", sensor_res=(160, 90), save_sensors=True, sensor_video=True,
                    control_mode="auto",
                )
                self.assertEqual(res["outcome"], "no_collision")
                self.assertEqual(len(res["sensors_active"]), 9)
                self.assertIn("vehicle.tesla.model3", [v.type_id for v in stats["spawned_vehicles"]])
                rgb_s = [s for s in stats["spawned_sensors"] if s.type_id == "sensor.camera.rgb"]
                lid_s = [s for s in stats["spawned_sensors"] if s.type_id == "sensor.lidar.ray_cast"]
                self.assertEqual(len(rgb_s), 8)
                self.assertEqual(len(lid_s), 1)
                self.assertEqual(lid_s[0].bp_attrs.get("rotation_frequency"), "20.0")
                self.assertTrue(all(s.attachment_type == "RIGID" for s in rgb_s + lid_s))
                self.assertTrue(os.path.exists(res["sensor_video_path"]))
                self.assertGreater(os.path.getsize(res["sensor_video_path"]), 1000)
                data_dir = res["sensor_data_dir"]
                self.assertTrue(os.path.exists(os.path.join(data_dir, "manifest.json")))
                frames_fn = os.path.join(data_dir, "frames.jsonl")
                lines = [json.loads(l) for l in open(frames_fn, encoding="utf-8") if l.strip()]
                self.assertGreater(len(lines), 10)
                self.assertIn("lidar", lines[0]["files"])
                self.assertIn("front_wide", lines[0]["files"])
                self.assertEqual(len(lines[0]["lidar_sectors"]), 36)
                self.assertTrue(os.path.exists(os.path.join(data_dir, lines[0]["files"]["lidar"])))
                self.assertTrue(os.path.exists(os.path.join(data_dir, lines[0]["files"]["front_wide"])))
        finally:
            if saved_carla is None:
                sys.modules.pop("carla", None)
            else:
                sys.modules["carla"] = saved_carla

    def test_run_motion_external_user_controller_avoids_crash(self):
        """Passing a user callable in --control external mode controls the vehicle directly without any built-in model."""
        fake_carla, world, stats, _ = _build_mock_carla_world(dt=motion_runner.DT)
        saved_carla = sys.modules.get("carla")
        sys.modules["carla"] = fake_carla
        try:
            with tempfile.TemporaryDirectory() as tmp:
                _write_minimal_scenario(tmp, "sc", speed_kmh=54.0, lead_x=38.0, dur=2.0)

                def user_braker(obs):
                    return np.array([0.0, -1.0], dtype=np.float32)

                res = motion_runner.run_motion(
                    tmp, "sc", load_map=False, hold_sec=0.1, real_time=False,
                    sensors="front_lidar", sensor_res=(128, 72),
                    control_mode="external", controller=user_braker,
                )
                self.assertEqual(res["control_mode"], "external")
                self.assertEqual(res["control_actor"], "v1")
                self.assertTrue(res["avoided_crash"], res)
                self.assertEqual(res["outcome"], "no_collision")
                self.assertEqual(res["sensors_active"], ["front_main", "front_wide", "lidar"])
        finally:
            if saved_carla is None:
                sys.modules.pop("carla", None)
            else:
                sys.modules["carla"] = saved_carla

    def test_carla_env_reset_step_action_spaces_and_map_reuse(self):
        """End-to-end mocked CARLA test of CarlaScenarioEnv across a 2-variant manifest:
        verifies observations, all 3 action spaces, render(), and OpenDRIVE map reuse."""
        fake_carla, world, stats, fake_map_fn = _build_mock_carla_world(dt=0.05)
        saved_carla = sys.modules.get("carla")
        saved_map_fn = map_builder.build_and_load_map
        sys.modules["carla"] = fake_carla
        map_builder.build_and_load_map = fake_map_fn
        try:
            with tempfile.TemporaryDirectory() as tmp:
                v1_dir = os.path.join(tmp, "variants", "v_a")
                v2_dir = os.path.join(tmp, "variants", "v_b")
                os.makedirs(v1_dir)
                os.makedirs(v2_dir)
                _write_minimal_scenario(v1_dir, "sc", speed_kmh=40.0, lead_x=45.0, dur=0.5)
                _write_minimal_scenario(v2_dir, "sc", speed_kmh=40.0, lead_x=50.0, dur=0.5)
                man = {
                    "scene_id": "sc",
                    "variants": [
                        {"id": "v_a", "rel_dir": os.path.join("variants", "v_a"), "scene_id": "sc"},
                        {"id": "v_b", "rel_dir": os.path.join("variants", "v_b"), "scene_id": "sc"},
                    ],
                }
                man_fn = os.path.join(tmp, "variants_manifest.json")
                with open(man_fn, "w", encoding="utf-8") as f:
                    json.dump(man, f)

                env = carla_env.CarlaScenarioEnv(
                    man_fn, sensors="front_lidar", sensor_res=(96, 64),
                    action_space="continuous", max_steps=10,
                )
                obs, info = env.reset(seed=7)
                self.assertEqual(obs["state"].shape, (ego_control.STATE_DIM,))
                self.assertEqual(obs["lidar_sectors"].shape, (36,))
                self.assertEqual(obs["lidar_bev"].shape, (3, 64, 64))
                self.assertIn("front_wide", obs["cameras"])
                self.assertEqual(carla_env.flatten_obs_vector(obs).shape, (48,))
                self.assertEqual(stats["map_loads"], 1)

                for _ in range(10):
                    obs, r, term, trunc, info = env.step(np.array([0.0, -0.5], dtype=np.float32))
                    if term or trunc:
                        break
                self.assertTrue(trunc or term)
                self.assertIsNotNone(env.render())

                # Reset to variant v_b sharing same road signature -> map MUST NOT reload!
                env.reset()
                self.assertEqual(stats["map_loads"], 1)
                env.close()

                # Discrete & target_speed action space smoke tests
                for sp_kind, sample_act in (("discrete", 3), ("target_speed", np.array([0.0, 5.0], dtype=np.float32))):
                    env_s = carla_env.CarlaScenarioEnv(v1_dir, sensors="lidar", action_space=sp_kind, max_steps=4)
                    o0, _ = env_s.reset()
                    self.assertEqual(o0["state"].shape, (ego_control.STATE_DIM,))
                    o1, r1, _, _, inf1 = env_s.step(sample_act)
                    self.assertTrue(math.isfinite(r1))
                    env_s.close()
        finally:
            map_builder.build_and_load_map = saved_map_fn
            if saved_carla is None:
                sys.modules.pop("carla", None)
            else:
                sys.modules["carla"] = saved_carla

    def test_local_pipeline_phase2_external_mode_skips_crash_retry_loop(self):
        """In --control external mode, local_pipeline.phase2 must NOT invoke scenario_check.search to force a crash."""
        fake_carla, world, stats, fake_map_fn = _build_mock_carla_world(dt=motion_runner.DT)
        saved_carla = sys.modules.get("carla")
        saved_map_fn = map_builder.build_and_load_map
        sys.modules["carla"] = fake_carla
        map_builder.build_and_load_map = fake_map_fn
        try:
            with tempfile.TemporaryDirectory() as tmp:
                _write_minimal_scenario(tmp, local_pipeline.SCENE_ID, speed_kmh=54.0, lead_x=38.0, dur=1.5)
                ready = local_pipeline.phase2(
                    tmp, max_iter=4, sensors="lidar", control_mode="external",
                    controller=lambda obs: [0.0, -1.0],
                    client=fake_carla.Client("localhost", 2000), skip_map_load=True,
                )
                self.assertEqual(ready["iterations"], 1)
                self.assertEqual(ready["status"], "avoided")
                self.assertTrue(ready["avoided_crash"])
                self.assertEqual(ready["sensors_active"], ["lidar"])
                rep = local_pipeline.report(tmp, None, ready)
                self.assertIn("avoided_crash=True", rep)
                self.assertIn("smart-car sensors", rep)
        finally:
            map_builder.build_and_load_map = saved_map_fn
            if saved_carla is None:
                sys.modules.pop("carla", None)
            else:
                sys.modules["carla"] = saved_carla

    def test_third_party_crash_equips_sensors_and_control_on_collider_not_bystander(self):
        """In a third-party crash (crash1: v1=parked bystander, v2=crossing aggressor, v3=struck van),
        enabling --sensors and --control automatically targets v3 (never the parked bystander v1)."""
        fake_carla, world, stats, _ = _build_mock_carla_world(dt=motion_runner.DT)
        saved_carla = sys.modules.get("carla")
        sys.modules["carla"] = fake_carla
        try:
            with open(FIX / "crash1.json", encoding="utf-8") as f:
                spec = json.load(f)
            with tempfile.TemporaryDirectory() as tmp:
                scene_builder.build(spec, tmp, "tp")
                res = motion_runner.run_motion(
                    tmp, "tp", load_map=False, hold_sec=0.1, real_time=False,
                    sensors="front_lidar", sensor_res=(128, 72),
                    control_mode="external", controller=lambda obs: [0.0, -1.0],
                )
                self.assertEqual(res["control_actor"], "v3")
                self.assertEqual(res["sensor_actor"], "v3")
                self.assertEqual(res["sensors_active"], ["front_main", "front_wide", "lidar"])
                self.assertNotEqual(res["control_actor"], "v1")
        finally:
            if saved_carla is None:
                sys.modules.pop("carla", None)
            else:
                sys.modules["carla"] = saved_carla


if __name__ == "__main__":
    unittest.main()
