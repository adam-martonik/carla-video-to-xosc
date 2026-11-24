"""
Executes a motion plan (agents/motion_planner.py) in CARLA:

  net.xml -> OpenDRIVE map -> spawn actors from *_scene_obj.txt -> drive / walk them -> collision result

  python -m agents.motion_runner <scene_id> [--out auto_result] [--no-map] [--hold 5]

Files used/written in --out:  <scene_id>.net.xml, <scene_id>_scene_obj.txt, <scene_id>_motion.json (plan),
<scene_id>_motion_result.json (outcome), <scene_id>_motion_ids.json (actors to clean up on the next run).
"""
import argparse
import ast
import json
import math
import os
import time

from agents.intercept import CutInConfig, CutInController
from agents.road_frame import RoadFrame
from agents.xosc_export import TRAJ_SUFFIX, TrajectoryRecorder

VEHICLE_BLUEPRINTS = {
    "bike": "vehicle.diamondback.century",
    "motorcycle": "vehicle.harley-davidson.low_rider",
    "car": "vehicle.dodge.charger",
    "jeep": "vehicle.nissan.patrol",
    "suv": "vehicle.nissan.patrol",
    "van": "vehicle.sprinter.mercedes",
    "truck": "vehicle.carlacola.actors",
}
# (forward_x_m, lateral_y_m, height_z_m, pitch_deg) windshield dashcam mounts per vehicle class
DASHCAM_MOUNTS = {
    "car": (1.20, 0.0, 1.35, -3.0),
    "suv": (1.35, 0.0, 1.65, -3.5),
    "jeep": (1.35, 0.0, 1.65, -3.5),
    "van": (1.75, 0.0, 2.00, -4.0),
    "truck": (2.65, 0.0, 2.55, -5.0),
    "motorcycle": (0.45, 0.0, 1.25, -2.0),
    "bike": (0.35, 0.0, 1.25, -2.0),
}
DT = 0.05
MAX_CONTACTS = 200      # contacts kept in the result (all of them are counted in contacts_total)
LANE_WIDTH = 3.5
CAR_LENGTH = 4.8


def ttc(gap_m, closing_speed_ms):
    """Time to collision; infinite when the gap does not close."""
    if closing_speed_ms <= 0.0:
        return math.inf
    return max(gap_m, 0.0) / closing_speed_ms


def rel_in_frame(dx, dy, yaw_deg):
    """(forward, right) of a point (dx, dy) seen from an actor with the given CARLA yaw (+y is right at yaw 0)."""
    yaw = math.radians(yaw_deg)
    return dx * math.cos(yaw) + dy * math.sin(yaw), -dx * math.sin(yaw) + dy * math.cos(yaw)


def crash_focus_transform(cx, cy, approach_yaw_deg, back_m=22.0, side_m=6.0, height_m=15.0):
    """Compute spectator (x, y, z, pitch, yaw) behind/above (cx, cy) aiming directly at the crash site."""
    rad = math.radians(approach_yaw_deg)
    fx, fy = math.cos(rad), math.sin(rad)
    rx, ry = -fy, fx
    cam_x = cx - back_m * fx + side_m * rx
    cam_y = cy - back_m * fy + side_m * ry
    yaw = math.degrees(math.atan2(cy - cam_y, cx - cam_x))
    pitch = math.degrees(math.atan2(0.8 - height_m, max(math.hypot(cx - cam_x, cy - cam_y), 1.0)))
    return round(cam_x, 3), round(cam_y, 3), round(height_m, 3), round(pitch, 2), round(yaw, 2)


def smooth_camera_pose(cur_pose, target_pose, alpha=0.12):
    """Exponentially interpolate (x, y, z, pitch, yaw) with wrapped shortest-arc yaw."""
    a = max(0.0, min(1.0, float(alpha)))
    cx, cy, cz, cpitch, cyaw = cur_pose
    tx, ty, tz, tpitch, tyaw = target_pose
    dyaw = ((tyaw - cyaw + 180.0) % 360.0) - 180.0
    dpitch = ((tpitch - cpitch + 180.0) % 360.0) - 180.0
    return (
        round(cx + a * (tx - cx), 3),
        round(cy + a * (ty - cy), 3),
        round(cz + a * (tz - cz), 3),
        round(cpitch + a * dpitch, 2),
        round(cyaw + a * dyaw, 2),
    )


def bgra_bytes_to_bgr(raw_bytes, width, height):
    """Convert raw CARLA BGRA sensor bytes into a contiguous HxWx3 uint8 BGR frame."""
    import numpy as np
    arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((height, width, 4))
    return arr[:, :, :3].copy()


def dashcam_mount_transform(vtype, bbox_extent=None):
    """Return (x, y, z, pitch_deg) for a clear forward windshield dashcam mount for `vtype`.
    Calibrated per vehicle class so vans, SUVs and trucks never clip inside the cab/dashboard."""
    key = str(vtype or "car").strip().lower()
    if key in DASHCAM_MOUNTS:
        return DASHCAM_MOUNTS[key]
    if bbox_extent is not None:
        ex = float(getattr(bbox_extent, "x", 0.0) or 0.0)
        ez = float(getattr(bbox_extent, "z", 0.0) or 0.0)
        if ex > 0.5 and ez > 0.4:
            return (round(max(0.4, 0.55 * ex), 2), 0.0, round(max(1.15, 2.0 * ez - 0.12), 2), -3.5)
    return DASHCAM_MOUNTS["car"]


def _is_vehicle_actor(aid, actor, agents_meta=None):
    if actor is None or not getattr(actor, "is_alive", True):
        return False
    tid = str(getattr(actor, "type_id", "") or "")
    if tid:
        return tid.startswith("vehicle.")
    meta = (agents_meta or {}).get(aid) or {}
    return meta.get("type", "car") != "pedestrian"


def resolve_dashcam_actors(plan, actors, agents_meta=None, mode="colliders"):
    """Choose the vehicle actor ID(s) that carry the dashcam, guaranteeing dashcams sit on the
    colliding/interacting vehicles rather than on a third-party bystander camera car.

    Ordering: [primary, secondary] where primary is ego if ego is part of the colliding pair,
    or the straight-driving/victim colliding vehicle in a third-party crash. Walkers are excluded.
    mode: 'colliders' / 'all_colliders' / 'all' (up to 2 colliding vehicles), 'primary' (1 vehicle),
          or an explicit actor id (e.g. 'v2').
    """
    plan = plan or {}
    pa = plan.get("agents") or {}
    ego = plan.get("ego", "v1")
    idmap = plan.get("idmap") or {}
    rev_id = {vid: sid for sid, vid in idmap.items()}

    if mode and mode not in ("colliders", "all_colliders", "all", "primary", "auto"):
        if mode in actors and _is_vehicle_actor(mode, actors[mode], agents_meta):
            return [mode]

    # Prefer explicit dashcam_actors (set by variants even when aggressor is absent), else expect.actors
    raw_cands = []
    for src in (plan.get("dashcam_actors"), (plan.get("expect") or {}).get("actors")):
        if isinstance(src, (list, tuple)) and src:
            raw_cands = [str(x) for x in src]
            break

    cands = []
    for aid in raw_cands:
        if aid in actors and _is_vehicle_actor(aid, actors[aid], agents_meta) and aid not in cands:
            cands.append(aid)

    if not cands:
        if ego in actors and _is_vehicle_actor(ego, actors[ego], agents_meta):
            cands = [ego]
        else:
            for aid in sorted(actors):
                if _is_vehicle_actor(aid, actors[aid], agents_meta):
                    cands = [aid]
                    break

    def _priority(aid):
        if aid == ego and aid in raw_cands:
            return (0, 0, aid)
        spec = pa.get(aid) or {}
        is_aggressor = bool(spec.get("intercept")) or (bool(spec.get("polyline")) and rev_id.get(aid) == "c1")
        if not is_aggressor and rev_id.get(aid) == "c1" and any(rev_id.get(x) == "c2" for x in cands):
            is_aggressor = True
        spd = float(spec.get("speed_kmh", 0.0) or 0.0)
        is_static = 1 if spec.get("mode") == "static" else 0
        return (1 if is_aggressor else 0, is_static, -spd, aid)

    if len(cands) > 1:
        cands.sort(key=_priority)

    if mode == "primary":
        return cands[:1]
    return cands[:2]


def _has_delayed_start(spec):
    """True when an actor's motion plan specifies a non-zero start delay or TTC trigger."""
    trig = (spec or {}).get("start")
    if not isinstance(trig, dict):
        return False
    ttype = trig.get("type", "time")
    if ttype == "ttc":
        return True
    return float(trig.get("t", 0.0) or 0.0) > 1e-6


def _launch_velocity_vec(actor, follower, spec, now=0.0):
    """(vx, vy) initial velocity vector along follower.route for an already_moving vehicle."""
    tl = follower.route[min(follower.i + 3, len(follower.route) - 1)].transform.location
    vl = actor.get_location()
    dd = max(math.hypot(tl.x - vl.x, tl.y - vl.y), 1e-3)
    prof = spec.get("speed_profile")
    sp = (profile_speed(prof, now) if prof and not spec.get("intercept") else float(spec.get("speed_kmh", 40))) / 3.6
    return (tl.x - vl.x) / dd * sp, (tl.y - vl.y) / dd * sp


def _predict_point_and_yaw(actor, follower, speed_kmh, t_s):
    loc = actor.get_location()
    yaw0 = actor.get_transform().rotation.yaw
    if follower is None or not follower.route or speed_kmh <= 0.1 or t_s <= 0.0:
        return loc.x, loc.y, yaw0
    rem = (float(speed_kmh) / 3.6) * float(t_s)
    px, py, yaw = loc.x, loc.y, yaw0
    for wp in follower.route:
        wx, wy = wp.transform.location.x, wp.transform.location.y
        seg = math.hypot(wx - px, wy - py)
        if seg > 1e-6:
            yaw = math.degrees(math.atan2(wy - py, wx - px))
        if seg >= rem and seg > 1e-6:
            f = rem / seg
            return px + f * (wx - px), py + f * (wy - py), yaw
        rem -= seg
        px, py = wx, wy
    return px, py, yaw


def _predict_point_at(actor, follower, speed_kmh, t_s):
    x, y, _ = _predict_point_and_yaw(actor, follower, speed_kmh, t_s)
    return x, y


def profile_speed(profile, t):
    """Speed in km/h of a speed profile [[t, km/h], ...] at time t (linear in between, held before and after)."""
    if t <= profile[0][0]:
        return float(profile[0][1])
    for (t0, v0), (t1, v1) in zip(profile, profile[1:]):
        if t <= t1:
            return float(v0 + (v1 - v0) * (t - t0) / max(t1 - t0, 1e-6))
    return float(profile[-1][1])


def _smooth_step(x):
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _fork_disp(fork, s):
    """(lateral_shift_m >= 0, elevation_z_m) of a diverging branch at distance s along the road frame."""
    if not fork or s is None or s <= float(fork.get("s_b", 1e9)):
        return 0.0, 0.0
    s_b = float(fork["s_b"])
    div_d = float(fork.get("diverge_d_m", 10.0)) * _smooth_step((s - s_b) / max(float(fork.get("diverge_len_m", 100.0)), 1.0))
    rise_m = float(fork.get("rise_m", 0.0))
    z = rise_m * _smooth_step((s - s_b - float(fork.get("rise_start_m", 55.0))) / max(float(fork.get("rise_len_m", 100.0)), 1.0))
    return div_d, z


def barrier_offsets(road_plan, s=None):
    """Lateral offsets d (+ = left) of the outer barrier lines: just outside the shoulder, on the ego carriageway's outer edges.
    When the lane count changes along the road (road_plan["sections"]), the offsets follow the section at distance s, and on a
    diverging fork (road_plan["fork"]) the branch-side outer barrier follows the outer edge of the branch instead of cutting
    across its driving lanes."""
    if not road_plan:
        return []
    lanes, shoulder = int(road_plan.get("lanes", 2)), float(road_plan.get("shoulder_m", 3.0))
    left_m = 0
    secs = road_plan.get("sections")
    if secs and s is not None:
        sc = next((x for x in secs if s < float(x["s1"])), secs[-1])
        lanes, left_m = int(sc["m"]) + int(sc["n"]), int(sc["m"])
    fork = road_plan.get("fork")
    div_d, _ = _fork_disp(fork, s) if (fork and s is not None) else (0.0, 0.0)
    out = []
    if road_plan.get("edge_right") == "barrier":
        if fork and s is not None and s >= float(fork["s_b"]) and fork.get("side") == "right" and fork.get("leave_j"):
            right_edge = (max(fork["leave_j"]) + 1) * LANE_WIDTH + div_d
            out.append(-(right_edge + shoulder + 0.3))
        else:
            out.append(-(lanes * LANE_WIDTH + shoulder + 0.3))
    if road_plan.get("edge_left") == "barrier" and road_plan.get("one_way"):
        if fork and s is not None and s >= float(fork["s_b"]) and fork.get("side") == "left" and fork.get("leave_j"):
            left_edge = -min(fork["leave_j"]) * LANE_WIDTH + div_d
            out.append(left_edge + shoulder + 0.3)
        else:
            out.append(-left_m * LANE_WIDTH + shoulder + 0.3)    # with two carriageways the left side is the median: no barrier
    return out


def gore_barrier_specs(road_plan, s):
    """Optional gore divider barrier offset (d, z_hint) between the ego road and a diverging branch once lateral separation
    exceeds safe clearance (>= 3.2 m), or None before clearance is reached."""
    fork = (road_plan or {}).get("fork")
    if not fork or s is None or s < float(fork.get("s_b", 1e9)):
        return None
    div_d, _ = _fork_disp(fork, s)
    if div_d < 3.2:
        return None
    secs = road_plan.get("sections") or []
    sc = next((x for x in secs if s < float(x["s1"])), secs[-1]) if secs else {"m": 0, "n": int(road_plan.get("lanes", 2))}
    if fork.get("side") == "left":
        return (-int(sc["m"]) * LANE_WIDTH + 0.7, 0.2)
    return (-(int(sc["m"]) + int(sc["n"])) * LANE_WIDTH - 0.7, 0.2)


def weather_for(env):
    """Map spec['environment'] ({'time_of_day', 'weather'}) to CARLA WeatherParameters keyword dict."""
    env = env if isinstance(env, dict) else {}
    tod = str(env.get("time_of_day", "day")).lower()
    w = str(env.get("weather", "clear")).lower()
    sun_alt = {"day": 45.0, "dusk": 8.0, "night": -15.0}.get(tod, 45.0)
    presets = {
        "clear":  {"cloudiness": 15.0, "precipitation": 0.0, "precipitation_deposits": 0.0, "fog_density": 0.0, "wetness": 0.0},
        "cloudy": {"cloudiness": 85.0, "precipitation": 0.0, "precipitation_deposits": 0.0, "fog_density": 5.0, "wetness": 10.0},
        "rain":   {"cloudiness": 90.0, "precipitation": 60.0, "precipitation_deposits": 40.0, "fog_density": 10.0, "wetness": 70.0},
        "fog":    {"cloudiness": 70.0, "precipitation": 0.0, "precipitation_deposits": 0.0, "fog_density": 45.0, "wetness": 15.0},
    }
    out = dict(presets.get(w, presets["clear"]))
    out["sun_altitude_angle"] = sun_alt
    out["sun_azimuth_angle"] = 180.0
    return out


def spawn_barriers(world, bp_lib, road_plan, spawned_ids, step=4.0):
    """Barrier props along the road edges that the plan marks as 'barrier', plus a gore divider on diverging forks once the
    two roads separate safely. Uses CARLA waypoint / branch elevation so barriers sit cleanly on climbing ramps."""
    import carla
    offsets = barrier_offsets(road_plan)
    fork = (road_plan or {}).get("fork")
    bps = bp_lib.filter("static.prop.streetbarrier")
    if (not offsets and not fork) or not bps or not (road_plan or {}).get("centerline"):
        return 0, 0
    frame = RoadFrame(road_plan["centerline"])
    cmap = world.get_map() if hasattr(world, "get_map") else None
    placed = failed = 0

    def _place(s_, d, z_hint=0.2):
        nonlocal placed, failed
        x, y = frame.to_xy(s_, d)
        z_use = z_hint
        if cmap is not None:
            try:
                wp = cmap.get_waypoint(carla.Location(x, -y, z_hint), project_to_road=True)
                if wp is not None and math.hypot(wp.transform.location.x - x, wp.transform.location.y - (-y)) < 6.0:
                    z_use = wp.transform.location.z + 0.15
            except Exception:
                pass
        p = world.try_spawn_actor(bps[0], carla.Transform(carla.Location(x, -y, z_use), carla.Rotation(yaw=-frame.heading(s_))))
        if p is None:
            failed += 1
        else:
            placed += 1
            spawned_ids.append(p.id)

    for i in range(len(offsets)):
        is_left_line = (i == 1) or (i == 0 and road_plan.get("edge_right") != "barrier" and road_plan.get("edge_left") == "barrier")
        s_ = 0.0
        while s_ <= frame.length:
            here = barrier_offsets(road_plan, s_)
            d = here[i] if len(here) == len(offsets) else offsets[i]
            z_hint = 0.2
            if fork and s_ >= float(fork["s_b"]):
                if (is_left_line and fork.get("side") == "left") or ((not is_left_line) and fork.get("side") == "right"):
                    _, z_br = _fork_disp(fork, s_)
                    z_hint = z_br + 0.2
            _place(s_, d, z_hint)
            s_ += step
    if fork:
        s_ = float(fork["s_b"])
        while s_ <= frame.length:
            gb = gore_barrier_specs(road_plan, s_)
            if gb is not None:
                _place(s_, gb[0], gb[1])
            s_ += step
    return placed, failed


def read_objects(obj_fn):
    """*_scene_obj.txt holds two printed dict literals: agents and static objects (CARLA frame)."""
    dicts = []
    for line in open(obj_fn, encoding="utf-8"):
        line = line.strip()
        if line.startswith("{"):
            dicts.append(ast.literal_eval(line))
    if not dicts:
        raise ValueError("no object dictionaries found in " + obj_fn)
    return dicts[0], (dicts[1] if len(dicts) > 1 else {})


class _RoutePoint:
    """Minimal stand-in for a map waypoint: the route follower only needs transform.location."""

    def __init__(self, loc):
        import types
        self.transform = types.SimpleNamespace(location=loc)


def polyline_shift(spec):
    """Shift (dx, dy) in the SUMO frame that the timing search applies to a polyline route (along its first segment)."""
    pts = spec.get("polyline")
    off = float(spec.get("offset_m", 0.0))
    if not pts or not off or len(pts) < 2:
        return 0.0, 0.0
    dx, dy = pts[1][0] - pts[0][0], pts[1][1] - pts[0][1]
    d = math.hypot(dx, dy) or 1.0
    return dx / d * off, dy / d * off


def _to_carla(p):
    return (float(p[0]), -float(p[1]))


def _cleanup(client, ids_fn, world=None):
    import carla
    ids = []
    if os.path.exists(ids_fn):
        ids.extend(json.load(open(ids_fn)))
    seen = set(ids)
    if world is not None:
        try:
            for a in world.get_actors():
                tid = getattr(a, "type_id", "")
                if tid.startswith(("vehicle.", "walker.", "sensor.", "static.prop.streetbarrier")) and a.id not in seen:
                    ids.append(a.id)
                    seen.add(a.id)
        except Exception:
            pass
    if ids:
        client.apply_batch([carla.command.DestroyActor(i) for i in ids])
        print(f"[motion] removed {len(ids)} actors of the previous run")
        time.sleep(1)


def sample_kinematic_track(track, t):
    """Interpolate (x, y, heading_deg, speed_ms) along [[t, x, y, heading_deg, speed_ms], ...] at time t, or None outside."""
    if not track or t < track[0][0] - 1e-6 or t > track[-1][0] + 1e-6:
        return None
    if t <= track[0][0]:
        _, x, y, h, s = track[0]
        return float(x), float(y), float(h), float(s)
    if t >= track[-1][0]:
        _, x, y, h, s = track[-1]
        return float(x), float(y), float(h), float(s)
    lo, hi = 0, len(track) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if track[mid][0] <= t:
            lo = mid
        else:
            hi = mid
    t0, x0, y0, h0, s0 = track[lo]
    t1, x1, y1, h1, s1 = track[hi]
    f = (t - t0) / max(t1 - t0, 1e-6)
    dh = ((h1 - h0 + 180.0) % 360.0) - 180.0
    return (x0 + f * (x1 - x0), y0 + f * (y1 - y0), h0 + f * dh, s0 + f * (s1 - s0))


def _dist2d(a, b):
    return math.hypot(a.x - b.x, a.y - b.y)


def extend_route_along_frame(route, target_xy_carla, road_frame, route_point_cls=_RoutePoint, loc_cls=None, step=2.0):
    """Extend a CARLA waypoint route that stops early at a road-section change along `road_frame` at its lane offset."""
    if not route or road_frame is None:
        return route
    if loc_cls is None:
        loc_cls = type(route[-1].transform.location)
    last_loc = route[-1].transform.location
    tx, ty = float(target_xy_carla[0]), float(target_xy_carla[1])
    if math.hypot(tx - last_loc.x, ty - last_loc.y) <= 15.0:
        return route
    s_last, d_last = road_frame.to_sd(last_loc.x, -last_loc.y)
    s_tgt, _ = road_frame.to_sd(tx, -ty)
    if s_tgt <= s_last + 10.0:
        return route
    s_ = s_last + step
    s_cap = min(s_tgt, road_frame.length - 2.0)
    while s_ <= s_cap:
        rx, ry = road_frame.to_xy(s_, d_last)
        route.append(route_point_cls(loc_cls(rx, -ry, 0.0)))
        s_ += step
    return route


def sanitize_straight_route(route, start_xy_carla, target_xy_carla, road_frame, route_point_cls=_RoutePoint, loc_cls=None, step=2.0, road_plan=None):
    """Ensure a straight-road route stays in its starting lane (rebuilding along `road_frame` if CARLA's OpenDRIVE
    waypoint graph jumps laterally > 2 m onto a shoulder or neighbouring lane) and reaches the target. Waypoints following
    a diverging fork branch in `road_plan['fork']` are preserved instead of being flattened back onto `d0`."""
    if road_frame is None:
        return route
    if loc_cls is None and route:
        loc_cls = type(route[-1].transform.location)
    s0, d0 = road_frame.to_sd(float(start_xy_carla[0]), -float(start_xy_carla[1]))
    s_tgt, _ = road_frame.to_sd(float(target_xy_carla[0]), -float(target_xy_carla[1]))
    fork = (road_plan or {}).get("fork")
    jumped = not route
    if route:
        for wp in route:
            loc = wp.transform.location
            sw, dw = road_frame.to_sd(loc.x, -loc.y)
            if abs(dw - d0) > 2.0:
                on_branch = False
                if fork and sw >= float(fork.get("s_b", 1e9)) and fork.get("leave_j"):
                    div_d, _ = _fork_disp(fork, sw)
                    sign = 1.0 if fork.get("side") == "left" else -1.0
                    br_ds = [-(j + 0.5) * LANE_WIDTH + sign * div_d for j in fork["leave_j"]]
                    on_branch = any(abs(dw - bd) <= 2.0 for bd in br_ds)
                if not on_branch:
                    jumped = True
                    break
    if jumped and loc_cls is not None and s_tgt > s0 + 5.0:
        out = []
        s_ = s0
        s_cap = min(s_tgt, road_frame.length - 2.0)
        while s_ <= s_cap:
            rx, ry = road_frame.to_xy(s_, d0)
            out.append(route_point_cls(loc_cls(rx, -ry, 0.0)))
            s_ += step
        return out
    return extend_route_along_frame(route, target_xy_carla, road_frame, route_point_cls, loc_cls, step)


def build_route(cmap, start_loc, targets, step=2.0, look=15, max_len=800.0):
    """Waypoint route from start_loc through the target points. At every fork the branch is chosen by following
    each candidate `look` steps ahead and keeping the one that ends closest to the current target."""
    import carla
    import math as _m
    wp = cmap.get_waypoint(start_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
    route, length = [wp], 0.0
    if wp.is_junction:
        # Inside a junction several connecting lanes overlap. Pick the one that passes through the start position
        # and leads toward the first target, then continue from its exit.
        tgt0 = carla.Location(targets[0][0], targets[0][1], 0.0)
        best, best_d = None, 1e9
        for entry, _exit in wp.get_junction().get_waypoints(carla.LaneType.Driving):
            poly, w = [entry], entry
            while w.is_junction:
                n = w.next(step)
                if not n:
                    break
                w = n[0]
                poly.append(w)
            near = min(range(len(poly)), key=lambda k: poly[k].transform.location.distance(start_loc))
            if poly[near].transform.location.distance(start_loc) > 2.5:
                continue
            tail = w
            for _ in range(look):
                n = tail.next(step)
                if not n:
                    break
                tail = n[0]
            d = tail.transform.location.distance(tgt0)
            if d < best_d:
                best, best_d = poly[near:], d
        if best:
            route = best
            wp = route[-1]
    for tx, ty in targets:
        tgt = carla.Location(tx, ty, 0.0)
        while length < max_len:
            nxt = wp.next(step)
            if not nxt:
                break
            if len(nxt) > 1:
                best, best_d = None, 1e9
                for cand in nxt:
                    w = cand
                    for _ in range(look):
                        n = w.next(step)
                        if not n:
                            break
                        w = n[0]
                    d = w.transform.location.distance(tgt)
                    if d < best_d:
                        best, best_d = cand, d
                wp = best
            else:
                wp = nxt[0]
            route.append(wp)
            length += step
            if wp.transform.location.distance(tgt) < step * 1.5:
                break
    return route


class RouteFollower:
    """Route follower with road-vehicle limits: comfortable acceleration and braking, speed reduction in curves
    (lateral acceleration limit), pedal and steering rate limits, brake-for-obstacle headway, and braking after a crash."""

    ACC = 3.0            # m/s^2 comfortable acceleration of a passenger car
    DEC = 3.5            # m/s^2 comfortable braking
    DEC_EMERGENCY = 6.0  # m/s^2 emergency braking (measured: brake pedal 1.0 gives about 6.3 m/s^2 in CARLA)
    A_LAT = 3.0          # m/s^2 comfortable lateral acceleration in curves
    HEADWAY = 1.2        # s distance kept to an obstacle ahead when avoid=True

    def __init__(self, vehicle, route, speed_kmh, avoid=False, step=2.0, road_frame=None):
        self.v, self.route, self.speed, self.avoid, self.step = vehicle, route, speed_kmh / 3.6, avoid, step
        self.road_frame = road_frame
        self.i = 0
        self.thr = self.brk = self.str = 0.0
        self.crash_time = None
        try:
            self.max_steer = math.radians(vehicle.get_physics_control().wheels[0].max_steer_angle)
        except Exception:
            self.max_steer = math.radians(70.0)
        self.wheelbase = 2.8

    def curve_speed(self):
        """Speed that keeps lateral acceleration below A_LAT over the next ~30 m of the route."""
        end = min(self.i + 15, len(self.route) - 2)
        worst = 0.0
        for k in range(self.i, end):
            a = self.route[k].transform.location
            b = self.route[k + 1].transform.location
            c = self.route[k + 2].transform.location
            h1 = math.atan2(b.y - a.y, b.x - a.x)
            h2 = math.atan2(c.y - b.y, c.x - b.x)
            dh = abs((h2 - h1 + math.pi) % (2 * math.pi) - math.pi)
            ds = 0.5 * (math.hypot(b.x - a.x, b.y - a.y) + math.hypot(c.x - b.x, c.y - b.y))      # real point spacing
            worst = max(worst, dh / max(ds, 0.5))
        return max(3.0, math.sqrt(self.A_LAT / worst)) if worst > 1e-3 else 1e9

    def pursuit_point(self, loc, spd, look, t_sim):
        """Point on the route that the steering aims at."""
        j = self.i
        while j < len(self.route) - 1 and _dist2d(self.route[j].transform.location, loc) < look:
            j += 1
        return self.route[j].transform.location

    stop_at = None          # simulation time after which the vehicle brakes to a stop (braking lead vehicle)
    profile = None          # [[t_sim, km/h], ...]: speed measured in the video (ego overlay speed, replayed vehicles)

    def control(self, others, dt, now_abs, t_sim=0.0):
        import carla
        loc, yaw = self.v.get_location(), math.radians(self.v.get_transform().rotation.yaw)
        # after a crash the driver brakes hard (reaction 0.4 s)
        if self.crash_time is not None and now_abs - self.crash_time > 0.4:
            self.thr, self.brk = 0.0, 1.0
            return carla.VehicleControl(throttle=0.0, steer=0.0, brake=1.0)
        while self.i < len(self.route) - 1 and _dist2d(self.route[self.i + 1].transform.location, loc) < _dist2d(self.route[self.i].transform.location, loc):
            self.i += 1
        spd = self.v.get_velocity().length()
        look = max(5.0, 0.8 * spd)
        tl = self.pursuit_point(loc, spd, look, t_sim)
        alpha = math.atan2(tl.y - loc.y, tl.x - loc.x) - yaw
        alpha = (alpha + math.pi) % (2 * math.pi) - math.pi
        steer_t = math.atan2(2 * self.wheelbase * math.sin(alpha), max(look, 1.0)) / max(self.max_steer, 0.1)
        steer_t = max(-1.0, min(1.0, steer_t))
        self.str += max(-2.0 * dt, min(2.0 * dt, steer_t - self.str))     # steering rate limit (2 / s)

        # target speed: planned speed (or the measured speed profile at this time), limited by curvature, stop at the route end
        if self.profile:
            self.speed = profile_speed(self.profile, t_sim) / 3.6
        remaining = _dist2d(self.route[-1].transform.location, loc)
        target = min(self.speed, self.curve_speed())
        if (remaining < spd * spd / (2 * self.DEC) + 2.0 and self.i >= len(self.route) - 3) or getattr(self, "ended", False):
            target = 0.0
            if self.i >= len(self.route) - 2 and remaining < 8.0:
                self.ended = True                       # stay stopped once the end of the route is reached
        dec = self.DEC
        if self.stop_at is not None and t_sim >= self.stop_at:
            target = 0.0
            dec = self.DEC_EMERGENCY * 0.8
        if self.avoid:
            # nearest obstacle in the own lane corridor ahead; keep a 2 m gap with a comfortable braking curve
            fx, fy = math.cos(yaw), math.sin(yaw)
            s_me = d_me = dir_sign = None
            if self.road_frame is not None:
                s_me, d_me = self.road_frame.to_sd(loc.x, -loc.y)
                h_fr = math.radians(-self.road_frame.heading(s_me))
                dir_sign = 1.0 if (math.cos(h_fr) * fx + math.sin(h_fr) * fy) >= 0.0 else -1.0
            best = None
            for o in others:
                ol = o.get_location()
                if abs(getattr(ol, "z", 0.0) - getattr(loc, "z", 0.0)) > 2.5:
                    continue
                if hasattr(o, "get_transform") and not getattr(o, "type_id", "").startswith("walker"):
                    o_yaw = math.radians(o.get_transform().rotation.yaw)
                    if math.cos(o_yaw) * fx + math.sin(o_yaw) * fy < -0.3:
                        continue
                if self.road_frame is not None:
                    s_o, d_o = self.road_frame.to_sd(ol.x, -ol.y)
                    ahead = (s_o - s_me) * dir_sign
                    lateral = d_o - d_me
                else:
                    rx, ry = ol.x - loc.x, ol.y - loc.y
                    ahead = rx * fx + ry * fy
                    lateral = -rx * fy + ry * fx
                if 0.0 < ahead < 60.0 and abs(lateral) < 1.8:
                    half = 1.0 if o.type_id.startswith("walker") else 2.4
                    gap = ahead - half - 2.4
                    if best is None or gap < best[0]:
                        ov = o.get_velocity()
                        best = (gap, max(0.0, ov.x * fx + ov.y * fy))
            if best is not None:
                gap, v_lead = best
                gap -= 0.35 * spd                      # brakes need about 0.35 s to build up pressure
                v_safe = v_lead + math.sqrt(2 * 2.5 * max(gap - 3.5, 0.0))
                target = min(target, v_safe)
                a_req = (spd * spd - v_lead * v_lead) / (2 * max(gap - 2.5, 0.1)) if spd > v_lead else 0.0
                if a_req > dec:                        # comfortable braking is not enough: brake as hard as needed
                    dec = min(self.DEC_EMERGENCY, a_req * 1.15)
        a_des = max(-dec, min(self.ACC, (2.0 if target < spd else 1.2) * (target - spd)))
        thr_t = max(0.0, min(1.0, 0.0145 * spd + 0.3 * max(a_des, 0.0))) if a_des > -0.2 else 0.0
        brk_t = max(0.0, min(1.0, -a_des / 6.5)) if a_des < -0.3 else 0.0       # pedal 1.0 ~ 6.5 m/s^2
        self.thr += max(-1.5 * dt, min(1.0 * dt, thr_t - self.thr))        # throttle slew
        self.brk += max(-4.0 * dt, min(4.0 * dt, brk_t - self.brk))        # brake slew
        return carla.VehicleControl(throttle=self.thr, steer=self.str, brake=self.brk, hand_brake=False)


class InterceptFollower(RouteFollower):
    """Cut-in car: steered from live positions by agents/intercept.CutInController instead of following a pre-computed path.
    The route (along the source lane) is only used for the end-of-road stop; the contact feedback is the crash time."""

    def __init__(self, vehicle, route, speed_kmh, ego, frame, controller):
        super().__init__(vehicle, route, speed_kmh, False)
        self.ego, self.frame, self.ctl = ego, frame, controller

    def curve_speed(self):
        return 1e9

    def pursuit_point(self, loc, spd, look, t_sim):
        import carla
        el = self.ego.get_location()
        s_e, d_e = self.frame.to_sd(el.x, -el.y)
        s_m, d_m = self.frame.to_sd(loc.x, -loc.y)
        d_t, v_t = self.ctl.update(t_sim, (s_e, d_e, self.ego.get_velocity().length()), (s_m, d_m, spd), self.crash_time is not None)
        self.speed = v_t
        if getattr(self.ctl.cfg, "side_strike", False) and abs(d_t - d_m) > 0.25:
            look_eff = max(4.5, 0.45 * spd)
        elif abs(d_t - d_m) > 0.25 and spd < 10.0:
            look_eff = max(3.5, 0.55 * spd)
        else:
            look_eff = look
        x, y = self.frame.to_xy(s_m + look_eff, d_t)
        return carla.Location(x, -y, 0.0)


def run_motion(out_dir, scene_id, load_map=True, hold_sec=5.0, host="localhost", port=2000, seed=0, record_video=None,
               real_time=True, camera="chase", dashcam=False, dashcam_mode="colliders"):
    """real_time: every simulation step takes at least DT of wall time, so the scene plays at the speed of the video in the CARLA
    window (synchronous mode would otherwise run it as fast as the server can step).
    camera: 'chase' (smooth chase/crash-focus), 'dashcam' (colliding vehicle windshield mount), 'dashcam_all' (side-by-side of
    colliding vehicles), or 'both' (primary colliding vehicle dashcam + chase).
    dashcam: when True, ensures dashcam footage is recorded from the colliding vehicle(s) (never a third-party bystander)."""
    import carla
    from agents.map_builder import build_and_load_map

    pace = {"last": None}

    def paced_tick(world):
        world.tick()
        if real_time:
            now = time.perf_counter()
            if pace["last"] is not None:
                wait = DT - (now - pace["last"])
                if wait > 0:
                    time.sleep(wait)
            pace["last"] = time.perf_counter()

    if record_video is None:
        env_rec = os.getenv("AS_RECORD_VIDEO")
        record_video = bool(env_rec and env_rec != "0")
    if dashcam and not record_video:
        record_video = True
        if camera == "chase":
            camera = "dashcam"
    if record_video is True:
        record_path = os.path.join(out_dir, f"{scene_id}_carla.mp4")
    elif isinstance(record_video, str) and record_video:
        record_path = record_video
    else:
        record_path = None
    want_dashcam = bool(dashcam) or (bool(record_path) and camera in ("dashcam", "both", "dashcam_all"))

    obj_fn = os.path.join(out_dir, f"{scene_id}_scene_obj.txt")
    plan_fn = os.path.join(out_dir, f"{scene_id}_motion.json")
    net_fn = os.path.join(out_dir, f"{scene_id}.net.xml")
    ids_fn = os.path.join(out_dir, f"{scene_id}_motion_ids.json")
    agents, statics = read_objects(obj_fn)
    plan = json.load(open(plan_fn))

    client = carla.Client(host, port)
    client.set_timeout(120.0)
    try:
        client.get_world()
    except RuntimeError as e:
        raise RuntimeError("CARLA server not reachable on %s:%s (start the editor and press Play)" % (host, port)) from e

    world = client.get_world()
    if load_map:
        world = build_and_load_map(client, net_fn, out_dir, plan.get("road"))
    else:
        _cleanup(client, ids_fn, world)
    if isinstance(plan.get("environment"), dict):
        try:
            world.set_weather(carla.WeatherParameters(**weather_for(plan["environment"])))
        except Exception:
            pass
    bp_lib = world.get_blueprint_library()
    tm = client.get_trafficmanager()

    try:
        world.apply_settings(carla.WorldSettings(synchronous_mode=True, fixed_delta_seconds=DT, substepping=True,
                                                 max_substep_delta_time=0.01, max_substeps=10))
    except TypeError:
        world.apply_settings(carla.WorldSettings(synchronous_mode=True, fixed_delta_seconds=DT))
    tm.set_synchronous_mode(True)
    tm.set_random_device_seed(seed)

    actors, sensors, spawned_ids = {}, [], []
    collisions, overflow, t0_abs = [], [0], [None]

    def record_contact(ev, aid):
        """Collision callback: one record per contact with the geometry the checker needs (all contacts, not just the first)."""
        t = world.get_snapshot().timestamp.elapsed_seconds
        rec = {"t": round(t, 2), "t_sim": round(t - t0_abs[0], 2) if t0_abs[0] is not None else 0.0, "actor": aid,
               "other": ev.other_actor.type_id, "other_id": ev.other_actor.id, "impulse": round(ev.normal_impulse.length(), 1),
               "impulse_vec": [round(ev.normal_impulse.x, 1), round(ev.normal_impulse.y, 1), round(ev.normal_impulse.z, 1)],
               "ego_yaw": round(ev.transform.rotation.yaw, 1)}
        try:
            me, ot = ev.transform.location, ev.other_actor.get_location()
            rec["rel"] = [round(v, 2) for v in rel_in_frame(ot.x - me.x, ot.y - me.y, ev.transform.rotation.yaw)]
            ot_yaw = ev.other_actor.get_transform().rotation.yaw
            rec["other_yaw"] = round(ot_yaw, 1)
            rec["other_rel"] = [round(v, 2) for v in rel_in_frame(me.x - ot.x, me.y - ot.y, ot_yaw)]
        except Exception:
            rec.setdefault("rel", None)
            rec.setdefault("other_yaw", None)
            rec.setdefault("other_rel", None)
        if len(collisions) < MAX_CONTACTS:
            collisions.append(rec)
        else:
            overflow[0] += 1

    result = {"scene_id": scene_id, "outcome": "no_collision", "events": [], "plan_note": plan.get("note")}
    cam_sensor, cam_q, cam_writer, last_frame = None, None, None, [None]
    dash_targets = []
    dash_streams = []          # list of {"aid", "sensor", "q", "last", "writer", "path"}

    def _capture_tick():
        if cam_writer is None and not dash_streams:
            return
        import queue
        import numpy as np

        def _drain(q, slot):
            if q is None:
                return
            frame = None
            while not q.empty():
                frame = q.get_nowait()
            if frame is None:
                try:
                    frame = q.get(timeout=0.5)
                except queue.Empty:
                    frame = None
            if frame is not None:
                slot[0] = bgra_bytes_to_bgr(frame.raw_data, frame.width, frame.height)

        _drain(cam_q, last_frame)
        for ds in dash_streams:
            _drain(ds["q"], ds["last"])
            if ds["writer"] is not None and ds["last"][0] is not None:
                ds["writer"].write(ds["last"][0])
        if cam_writer is None:
            return
        primary_dash = dash_streams[0]["last"][0] if dash_streams else None
        if camera == "both":
            if last_frame[0] is not None and primary_dash is not None:
                cam_writer.write(np.hstack([primary_dash, last_frame[0]]))
        elif camera == "dashcam_all":
            frames = [ds["last"][0] for ds in dash_streams if ds["last"][0] is not None]
            if len(frames) == len(dash_streams) and frames:
                cam_writer.write(np.hstack(frames) if len(frames) > 1 else frames[0])
        elif camera == "dashcam":
            if primary_dash is not None:
                cam_writer.write(primary_dash)
        else:
            if last_frame[0] is not None:
                cam_writer.write(last_frame[0])

    try:
        def _spawn_one(aid, a):
            x, y, z = a["location"]
            yaw = a["rotation"][1]
            if a["type"] == "pedestrian":
                bp = bp_lib.filter("walker.pedestrian.*")[0]
            else:
                bp = bp_lib.find(VEHICLE_BLUEPRINTS[a["type"]])
                color = plan["agents"].get(aid, {}).get("color")
                if color and bp.has_attribute("color"):
                    bp.set_attribute("color", color)
            pspec = plan["agents"].get(aid, {})
            off = float(pspec.get("offset_m", 0.0))
            if pspec.get("polyline") and off:
                sx, sy = polyline_shift(pspec)
                x, y = x + sx, y - sy                         # obj coordinates are CARLA frame (y negated)
                off = 0.0
            if off:                                          # timing search: move the spawn point along the lane
                wp0 = world.get_map().get_waypoint(carla.Location(x, y, 0.0), project_to_road=True)
                cand = wp0.next(off) if off > 0 else wp0.previous(-off)
                if cand:
                    x, y = cand[0].transform.location.x, cand[0].transform.location.y
            heights = (z, 1.0, 0.5)
            if a["type"] != "pedestrian":
                # just above the road surface under the spawn point, so that the car does not drop onto it (and is not put
                # under a road that climbs)
                wp_z = world.get_map().get_waypoint(carla.Location(x, y, 0.0), project_to_road=True)
                if wp_z is not None and math.hypot(wp_z.transform.location.x - x, wp_z.transform.location.y - y) < 4.0:
                    gz = wp_z.transform.location.z
                    heights = (gz + 0.25, gz + 0.5, gz + 1.0, z)
            for zz in heights:                              # big vehicles may not fit at the first height
                actor = world.try_spawn_actor(bp, carla.Transform(carla.Location(x, y, zz), carla.Rotation(yaw=yaw)))
                if actor is not None:
                    return actor
            return None

        pending_spawns = {}
        kin_z, kin_spd = {}, {}
        is_kinematic = lambda aid_: bool((plan["agents"].get(aid_) or {}).get("kinematic_track"))
        # ---- spawn
        for aid, a in agents.items():
            pspec = plan["agents"].get(aid, {})
            if pspec.get("mode") == "skip":      # a replayed vehicle the checker took out
                continue
            spawn_at_s = float(pspec.get("spawn_at_s") or 0.0)
            if spawn_at_s > 0.0:
                pending_spawns[aid] = spawn_at_s
                continue
            actor = _spawn_one(aid, a)
            if actor is None:
                print(f"[motion] spawn failed for {aid}")
                continue
            if is_kinematic(aid):
                try:
                    actor.set_simulate_physics(False)
                except Exception:
                    pass
                kin_z[aid] = actor.get_location().z
                kin_spd[aid] = float(pspec["kinematic_track"][0][4])
            actors[aid] = actor
            spawned_ids.append(actor.id)
        for oid, o in statics.items():
            bps = bp_lib.filter("static.prop." + o["type"])
            if bps:
                x, y, z = o["location"]
                p = world.try_spawn_actor(bps[0], carla.Transform(carla.Location(x, y, z), carla.Rotation(yaw=o["rotation"][1])))
                if p is not None:
                    spawned_ids.append(p.id)
        n_bar, n_bar_failed = spawn_barriers(world, bp_lib, plan.get("road"), spawned_ids)
        if n_bar or n_bar_failed:
            print(f"[motion] barrier props: {n_bar} placed, {n_bar_failed} could not be spawned")
        for aid, actor in actors.items():
            if actor.type_id.startswith("vehicle") and not is_kinematic(aid):
                actor.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
                s = world.spawn_actor(bp_lib.find("sensor.other.collision"), carla.Transform(), attach_to=actor)
                s.listen(lambda ev, aid=aid: record_contact(ev, aid))
                sensors.append(s)
                spawned_ids.append(s.id)
        json.dump(spawned_ids, open(ids_fn, "w"))
        id_of = {a.id: k for k, a in actors.items()}

        for _ in range(10):          # in synchronous mode new actors report (0, 0) until the world has ticked
            world.tick()
        t0_abs[0] = world.get_snapshot().timestamp.elapsed_seconds      # scenario time zero for the contact records

        # ---- behaviour
        pa = plan["agents"]
        ego = plan.get("ego") if plan.get("ego") in actors else next(iter(actors))
        followers = {}
        road_pl = plan.get("road") or {}
        road_frame = RoadFrame(road_pl["centerline"]) if road_pl.get("centerline") else RoadFrame.straight()

        def make_follower(aid, v, spec):
            speed = float(spec.get("speed_kmh", 40))
            ic = spec.get("intercept")
            target = actors.get(ic.get("target", ego)) if ic else None
            if ic and target is not None:
                vl = v.get_location()
                s0 = road_frame.to_sd(vl.x, -vl.y)[0]
                route = []
                s_ = s0
                while s_ < road_frame.length - 2.0:
                    x, y = road_frame.to_xy(s_, ic["lane_src_d"])
                    route.append(_RoutePoint(carla.Location(x, -y, 0.0)))
                    s_ += 2.0
                cfg = CutInConfig(lane_src_d=float(ic["lane_src_d"]), lane_dst_d=float(ic["lane_dst_d"]), t_event_s=float(ic["t_event_s"]),
                                  lead_s=float(ic.get("lead_s", 1.4)), aim_offset_m=float(ic.get("aim_offset_m", 0.0)),
                                  max_lateral_speed_ms=float(ic.get("max_lateral_speed_ms", 1.5)),
                                  cruise_speed_ms=float(ic["cruise_speed_ms"]) if ic.get("cruise_speed_ms") is not None else None,
                                  long_offset_m=float(ic.get("long_offset_m", 1.0)),
                                  side_strike=bool(ic.get("side_strike", False)))
                return InterceptFollower(v, route, speed, target, road_frame, CutInController(cfg))
            if spec.get("polyline"):
                sx, sy = polyline_shift(spec)
                route = [_RoutePoint(carla.Location(p_[0] + sx, -(p_[1] + sy), 0.0)) for p_ in spec["polyline"]]
            else:
                tgt_carla = [_to_carla(pt) for pt in spec["path"]]
                v_loc = v.get_location()
                route = build_route(world.get_map(), v_loc, tgt_carla,
                                    max_len=max(250.0, road_frame.length + 50.0))
                if road_pl.get("centerline") and tgt_carla:
                    route = sanitize_straight_route(route, (v_loc.x, v_loc.y), tgt_carla[-1], road_frame, _RoutePoint, carla.Location,
                                                    road_plan=road_pl)
            return RouteFollower(v, route, speed, bool(spec.get("avoid", False)),
                                 road_frame=(road_frame if road_pl.get("centerline") else None))

        for aid, spec in pa.items():
            if aid in actors and spec["mode"] == "drive" and not is_kinematic(aid):
                followers[aid] = make_follower(aid, actors[aid], spec)
                followers[aid].stop_at = spec.get("stop_at_s")
                if spec.get("speed_profile") and not spec.get("intercept"):
                    followers[aid].profile = spec["speed_profile"]
                print(f"[motion] {aid}: route of {len(followers[aid].route)} waypoints" + (" (closed-loop cut-in)" if spec.get("intercept") else ""))

        # focus spectator camera smoothly along the approach and glide onto the impact site
        ego_yaw = 0.0
        cur_cam_pose = None
        locked_cam_target = None
        exp = plan.get("expect") or {}
        exp_actors = exp.get("actors") or []

        def _chase_target_pose():
            if locked_cam_target is not None:
                # after the contact: follow the crash pair (the cars keep moving while they brake), looking the way the ego
                # car looked at the moment of contact so that the framing stays steady
                ha, hb, yaw0 = locked_cam_target
                if ha in actors and hb in actors and actors[ha].is_alive and actors[hb].is_alive:
                    la, lb = actors[ha].get_location(), actors[hb].get_location()
                    return crash_focus_transform(0.5 * (la.x + lb.x), 0.5 * (la.y + lb.y), yaw0, back_m=16.0, side_m=0.0, height_m=7.5)
                return cur_cam_pose
            if ego in actors and actors[ego].is_alive:
                el = actors[ego].get_location()
                ey = actors[ego].get_transform().rotation.yaw
                if len(exp_actors) >= 2 and ego not in exp_actors and exp_actors[0] in actors and exp_actors[1] in actors:
                    la, lb = actors[exp_actors[0]].get_location(), actors[exp_actors[1]].get_location()
                    return crash_focus_transform(0.5 * (la.x + lb.x), 0.5 * (la.y + lb.y), ey,
                                                 back_m=16.0, side_m=0.0, height_m=7.5)
                spd_ego = float(pa.get(ego, {}).get("speed_kmh", 0.0))
                if spd_ego >= 5.0:
                    # Look slightly ahead of moving ego so crossing/cut-in vehicles stay centered
                    rad = math.radians(ey)
                    return crash_focus_transform(el.x + 8.0 * math.cos(rad), el.y + 8.0 * math.sin(rad), ey,
                                                 back_m=16.0, side_m=0.0, height_m=7.5)
                if len(exp_actors) >= 2 and exp_actors[0] in actors and exp_actors[1] in actors:
                    la, lb = actors[exp_actors[0]].get_location(), actors[exp_actors[1]].get_location()
                    return crash_focus_transform(0.5 * (la.x + lb.x), 0.5 * (la.y + lb.y), ey,
                                                 back_m=16.0, side_m=0.0, height_m=7.5)
                if len(exp_actors) == 1 and exp_actors[0] != ego and exp_actors[0] in actors and actors[exp_actors[0]].is_alive:
                    tl_a = actors[exp_actors[0]].get_location()
                    ty_a = actors[exp_actors[0]].get_transform().rotation.yaw
                    rad = math.radians(ty_a)
                    return crash_focus_transform(tl_a.x + 8.0 * math.cos(rad), tl_a.y + 8.0 * math.sin(rad), ty_a,
                                                 back_m=16.0, side_m=0.0, height_m=7.5)
                rad = math.radians(ey)
                return crash_focus_transform(el.x + 12.0 * math.cos(rad), el.y + 12.0 * math.sin(rad), ey,
                                             back_m=16.0, side_m=0.0, height_m=7.5)
            return None

        def _apply_cam_pose(pose):
            tf = carla.Transform(carla.Location(pose[0], pose[1], pose[2]), carla.Rotation(pitch=pose[3], yaw=pose[4]))
            world.get_spectator().set_transform(tf)
            if cam_sensor is not None and cam_sensor.is_alive:
                cam_sensor.set_transform(tf)

        # On crossroad scenarios, align traffic signals so the ego approach has a green light rather than a random red
        if not road_pl.get("centerline") and ego in actors and actors[ego].is_alive:
            try:
                el = actors[ego].get_location()
                ey_rad = math.radians(actors[ego].get_transform().rotation.yaw)
                efx, efy = math.cos(ey_rad), math.sin(ey_rad)
                tls = list(world.get_actors().filter("traffic.traffic_light*"))
                ego_tl = actors[ego].get_traffic_light() if hasattr(actors[ego], "get_traffic_light") else None
                for tl in tls:
                    tl.freeze(True)
                    tl_loc = tl.get_location()
                    ahead = (tl_loc.x - el.x) * efx + (tl_loc.y - el.y) * efy
                    same_dir = (tl == ego_tl) or (ahead > 0.0 and abs(math.cos(math.radians(tl.get_transform().rotation.yaw)) * efx +
                                                                      math.sin(math.radians(tl.get_transform().rotation.yaw)) * efy) < 0.5)
                    tl.set_state(carla.TrafficLightState.Green if same_dir else carla.TrafficLightState.Red)
            except Exception:
                pass

        try:
            ego_yaw = actors[ego].get_transform().rotation.yaw
            cur_cam_pose = _chase_target_pose()
            if cur_cam_pose is not None:
                spec_tf = carla.Transform(
                    carla.Location(cur_cam_pose[0], cur_cam_pose[1], cur_cam_pose[2]),
                    carla.Rotation(pitch=cur_cam_pose[3], yaw=cur_cam_pose[4]),
                )
                world.get_spectator().set_transform(spec_tf)
                if record_path or want_dashcam:
                    import queue, cv2
                    if want_dashcam:
                        dash_targets = resolve_dashcam_actors(plan, actors, agents, mode=dashcam_mode)
                    split_main = (camera == "both") or (camera == "dashcam_all" and len(dash_targets) >= 2)
                    cam_w, cam_h = (640, 360) if split_main else (1280, 720)
                    cam_bp = bp_lib.find("sensor.camera.rgb")
                    cam_bp.set_attribute("image_size_x", str(cam_w))
                    cam_bp.set_attribute("image_size_y", str(cam_h))
                    cam_bp.set_attribute("fov", "90")
                    if record_path and camera in ("chase", "both"):
                        cam_sensor = world.try_spawn_actor(cam_bp, spec_tf)
                        if cam_sensor is not None:
                            cam_q = queue.Queue()
                            cam_sensor.listen(cam_q.put)
                            sensors.append(cam_sensor)
                            spawned_ids.append(cam_sensor.id)
                    if want_dashcam:
                        for daid in dash_targets:
                            dactor = actors.get(daid)
                            if dactor is None or not dactor.is_alive:
                                continue
                            dash_bp = bp_lib.find("sensor.camera.rgb")
                            dash_bp.set_attribute("image_size_x", str(cam_w))
                            dash_bp.set_attribute("image_size_y", str(cam_h))
                            dash_bp.set_attribute("fov", "100")
                            vtype_d = (agents.get(daid) or {}).get("type", "car")
                            bbox_ext = getattr(getattr(dactor, "bounding_box", None), "extent", None)
                            mx, my, mz, mpitch = dashcam_mount_transform(vtype_d, bbox_ext)
                            dash_tf = carla.Transform(carla.Location(x=mx, y=my, z=mz), carla.Rotation(pitch=mpitch))
                            dsensor = world.try_spawn_actor(dash_bp, dash_tf, attach_to=dactor)
                            if dsensor is not None:
                                dq = queue.Queue()
                                dsensor.listen(dq.put)
                                sensors.append(dsensor)
                                spawned_ids.append(dsensor.id)
                                dpath = os.path.join(out_dir, f"{scene_id}_dashcam_{daid}.mp4")
                                dwriter = cv2.VideoWriter(dpath, cv2.VideoWriter_fourcc(*"mp4v"), round(1.0 / DT, 1), (cam_w, cam_h))
                                dash_streams.append({"aid": daid, "sensor": dsensor, "q": dq, "last": [None],
                                                     "writer": dwriter, "path": dpath})
                    if cam_sensor is not None or dash_streams:
                        json.dump(spawned_ids, open(ids_fn, "w"))
                        if record_path:
                            if camera == "both" and cam_sensor is not None and dash_streams:
                                out_w = cam_w * 2
                            elif camera == "dashcam_all" and len(dash_streams) >= 2:
                                out_w = cam_w * len(dash_streams)
                            else:
                                out_w = cam_w
                            cam_writer = cv2.VideoWriter(record_path, cv2.VideoWriter_fourcc(*"mp4v"), round(1.0 / DT, 1), (out_w, cam_h))
                            print(f"[motion] recording CARLA {camera} video -> {record_path}")
                        if dash_streams:
                            print("[motion] dashcam mounted on colliding vehicle(s): " + ", ".join(
                                "%s (%s)" % (ds["aid"], os.path.basename(ds["path"])) for ds in dash_streams))
        except Exception:
            pass

        def ref_ttc(trig):
            ref = actors.get(trig.get("who", ego))
            if ref is None:
                return 1e9
            px, py = _to_carla(trig["point"])
            loc = ref.get_location()
            spd = ref.get_velocity().length()
            if spd < 1.0:
                spd = max(float(pa.get(trig.get("who", ego), {}).get("speed_kmh", 30)) / 3.6, 1.0)
            return math.hypot(loc.x - px, loc.y - py) / spd

        started, wp_idx, start_time, trace = {}, {}, {}, []
        walker_speed, peak = {}, {aid: {"long": 0.0, "lat": 0.0} for aid in followers}
        min_pair = {}                                    # closest approach (centre to centre) of every pair of actors
        # vehicles that are already driving when the scene begins start with their planned speed along the route
        # (unless they carry a delayed start trigger); all others pull away from rest with the controller's accel limit
        for aid, fol in followers.items():
            if pa[aid].get("already_moving", False) and not _has_delayed_start(pa[aid]):
                v = actors[aid]
                v.apply_control(carla.VehicleControl(brake=0.0, hand_brake=False))
                vx, vy = _launch_velocity_vec(v, fol, pa[aid], now=0.0)
                v.set_target_velocity(carla.Vector3D(vx, vy, 0.0))
                started[aid], start_time[aid] = True, 0.0
        t_end, first_hit = float(plan.get("duration_s", 15)), None
        tick, printed = 0, 0
        recorder = TrajectoryRecorder()                 # raw poses of every actor, every tick -> <scene>_trajectories.json
        while tick * DT < t_end + 3.0:
            now = tick * DT
            for aid in list(pending_spawns):
                if now < pending_spawns[aid]:
                    continue
                pending_spawns.pop(aid)
                actor = _spawn_one(aid, agents[aid])
                if actor is None:
                    continue
                actors[aid] = actor
                spawned_ids.append(actor.id)
                id_of[actor.id] = aid
                spec_d = pa[aid]
                if is_kinematic(aid):
                    try:
                        actor.set_simulate_physics(False)
                    except Exception:
                        pass
                    kin_z[aid] = actor.get_location().z
                    kin_spd[aid] = float(spec_d["kinematic_track"][0][4])
                    json.dump(spawned_ids, open(ids_fn, "w"))
                else:
                    if actor.type_id.startswith("vehicle"):
                        s = world.spawn_actor(bp_lib.find("sensor.other.collision"), carla.Transform(), attach_to=actor)
                        s.listen(lambda ev, aid=aid: record_contact(ev, aid))
                        sensors.append(s)
                        spawned_ids.append(s.id)
                        actor.apply_control(carla.VehicleControl(brake=0.0, hand_brake=False))
                    json.dump(spawned_ids, open(ids_fn, "w"))
                    if spec_d["mode"] == "drive":
                        fol = make_follower(aid, actor, spec_d)
                        fol.stop_at = spec_d.get("stop_at_s")
                        if spec_d.get("speed_profile") and not spec_d.get("intercept"):
                            fol.profile = spec_d["speed_profile"]
                        followers[aid] = fol
                        peak[aid] = {"long": 0.0, "lat": 0.0}
                        if spec_d.get("already_moving", False):
                            tl = fol.route[min(fol.i + 3, len(fol.route) - 1)].transform.location
                            vl_x, vl_y = agents[aid]["location"][0], agents[aid]["location"][1]
                            dd = max(math.hypot(tl.x - vl_x, tl.y - vl_y), 1e-3)
                            prof = spec_d.get("speed_profile")
                            sp = (profile_speed(prof, now) if prof and not spec_d.get("intercept") else float(spec_d.get("speed_kmh", 40))) / 3.6
                            actor.set_target_velocity(carla.Vector3D((tl.x - vl_x) / dd * sp, (tl.y - vl_y) / dd * sp, 0.0))
                started[aid], start_time[aid] = True, round(now, 2)
            for aid in [k for k in list(actors) if is_kinematic(k)]:
                actor = actors[aid]
                if not actor.is_alive:
                    continue
                kt = pa[aid]["kinematic_track"]
                pose = sample_kinematic_track(kt, now)
                if pose is None:
                    if now > kt[-1][0] + 0.05:
                        try:
                            actor.destroy()
                        except Exception:
                            pass
                        actors.pop(aid, None)
                    continue
                kx, ky, kh_deg, kspd = pose
                wp = world.get_map().get_waypoint(carla.Location(kx, -ky, kin_z.get(aid, 0.0)), project_to_road=True)
                z_road = (wp.transform.location.z + 0.05) if wp is not None else kin_z.get(aid, 0.2)
                pitch = wp.transform.rotation.pitch if wp is not None else 0.0
                if wp is not None:
                    w_rad = math.radians(wp.transform.rotation.yaw)
                    c_rad = math.radians(-kh_deg)
                    if math.cos(w_rad) * math.cos(c_rad) + math.sin(w_rad) * math.sin(c_rad) < 0.0:
                        pitch = -pitch
                kin_z[aid], kin_spd[aid] = z_road, kspd
                actor.set_transform(carla.Transform(carla.Location(kx, -ky, z_road), carla.Rotation(pitch=pitch, yaw=-kh_deg, roll=0.0)))
            for aid, spec in pa.items():
                if aid not in actors or spec["mode"] == "static" or aid in started:
                    continue
                trig = spec.get("start") or {"type": "time", "t": 0.0}
                go = (now >= float(trig.get("t", 0))) if trig["type"] == "time" else (ref_ttc(trig) <= float(trig.get("seconds", 2)))
                if go:
                    started[aid] = True
                    start_time[aid] = round(now, 2)
                    if spec["mode"] == "drive":
                        actors[aid].apply_control(carla.VehicleControl(brake=0.0, hand_brake=False))
                        if spec.get("already_moving", False) and aid in followers:
                            vx, vy = _launch_velocity_vec(actors[aid], followers[aid], spec, now=now)
                            actors[aid].set_target_velocity(carla.Vector3D(vx, vy, 0.0))
            now_abs = world.get_snapshot().timestamp.elapsed_seconds
            for c in collisions[printed:]:
                if c["actor"] == ego and not c["other"].startswith(("vehicle", "walker")):
                    print(f"[motion] the ego car touched {c['other']} at {c['t_sim']} s")
            printed = len(collisions)
            for aid, fol in followers.items():            # a lead car that brakes when the ego car is about ttc_s away
                bt = pa[aid].get("brake_trigger")
                ref = actors.get(bt.get("who", ego)) if bt else None
                if ref is not None and aid in started and fol.v.is_alive and ref.is_alive and (fol.stop_at is None or fol.stop_at > now):
                    rl, ll, tf = ref.get_location(), fol.v.get_location(), ref.get_transform().get_forward_vector()
                    ahead = (ll.x - rl.x) * tf.x + (ll.y - rl.y) * tf.y
                    if ahead > 0.0:
                        vr, vl = ref.get_velocity(), fol.v.get_velocity()
                        closing = (vr.x - vl.x) * tf.x + (vr.y - vl.y) * tf.y
                        if ttc(ahead - CAR_LENGTH, closing) <= float(bt.get("ttc_s", 2.0)):
                            fol.stop_at = now
            for c in collisions:                          # a vehicle that hit another vehicle or a pedestrian starts braking
                f_ = followers.get(c["actor"])
                if f_ is not None and f_.crash_time is None and c["other"].startswith(("vehicle", "walker")):
                    f_.crash_time = c["t"]
            for aid, fol in followers.items():
                if aid in started and fol.v.is_alive:
                    fol.v.apply_control(fol.control([a for k, a in actors.items() if k != aid and a.is_alive], DT, now_abs, now))
            for aid, spec in pa.items():
                if aid in actors and spec["mode"] == "walk" and aid in started:
                    w = actors[aid]
                    i = wp_idx.get(aid, 0)
                    if i < len(spec["path"]):
                        tx, ty = _to_carla(spec["path"][i])
                        loc = w.get_location()
                        dx, dy = tx - loc.x, ty - loc.y
                        d = math.hypot(dx, dy)
                        if d < 0.4:
                            wp_idx[aid] = i + 1
                        else:
                            cur = walker_speed.get(aid, 0.0)           # pedestrians accelerate at about 1.5 m/s^2
                            cur = min(float(spec.get("speed_ms", 1.4)), cur + 1.5 * DT)
                            walker_speed[aid] = cur
                            w.apply_control(carla.WalkerControl(direction=carla.Vector3D(dx / d, dy / d, 0.0), speed=cur))
                    else:
                        w.apply_control(carla.WalkerControl(speed=0.0))
            paced_tick(world)
            _capture_tick()
            tick += 1
            recorder.record(tick * DT, actors, speed_overrides=kin_spd)
            alive = {k: a.get_location() for k, a in actors.items() if a.is_alive}
            ks = sorted(alive)
            for ia in range(len(ks)):
                for ib in range(ia + 1, len(ks)):
                    d_ = alive[ks[ia]].distance(alive[ks[ib]])
                    key = ks[ia] + "-" + ks[ib]
                    if d_ < min_pair.get(key, 1e9):
                        min_pair[key] = d_
            if tick % 10 == 0:                              # trace every 0.5 s: position and speed, None once destroyed
                trace.append({"t": round(tick * DT, 2), **{k: ([round(a.get_location().x, 1), round(a.get_location().y, 1),
                              round(kin_spd[k] if is_kinematic(k) else a.get_velocity().length(), 1)] if a.is_alive else None)
                              for k, a in actors.items()}})
            hits = [c for c in collisions if not c["other"].startswith("static")
                    or c["other"].startswith("static.prop")]
            exp = plan.get("expect") or {}
            exp_pair = set(exp.get("actors", [])) if exp.get("type") in ("collision", "near_miss") and len(exp.get("actors", [])) == 2 else None
            if exp_pair is not None:
                # the run ends 1.5 s after the expected event; other collisions are recorded but do not end it
                hits = [c for c in collisions if {c["actor"], id_of.get(c["other_id"])} == exp_pair]
            if hits and first_hit is None:
                first_hit = now
                try:
                    c0 = hits[0]
                    ha, hb = c0["actor"], id_of.get(c0["other_id"])
                    if ha in actors and hb in actors and actors[ha].is_alive and actors[hb].is_alive:
                        cur_yaw = actors[ego].get_transform().rotation.yaw if ego in actors and actors[ego].is_alive else ego_yaw
                        locked_cam_target = (ha, hb, cur_yaw)
                except Exception:
                    pass
            try:
                tgt_pose = _chase_target_pose()
                if tgt_pose is not None:
                    cur_cam_pose = tgt_pose if cur_cam_pose is None else smooth_camera_pose(cur_cam_pose, tgt_pose, alpha=0.14)
                    _apply_cam_pose(cur_cam_pose)
            except Exception:
                pass
            if not hits and now > 0.5:      # skip the start-up jump caused by the initial velocity
                for aid, fol in followers.items():
                    if aid in started and fol.v.is_alive:
                        acc, tf = fol.v.get_acceleration(), fol.v.get_transform()
                        fw, rt = tf.get_forward_vector(), tf.get_right_vector()
                        peak[aid]["long"] = max(peak[aid]["long"], abs(acc.x * fw.x + acc.y * fw.y))
                        peak[aid]["lat"] = max(peak[aid]["lat"], abs(acc.x * rt.x + acc.y * rt.y))
            if first_hit is not None and now > first_hit + 1.5:
                break

        for c in collisions:
            c["other_actor"] = id_of.get(c["other_id"], c["other"])
        result["peak_accel_before_first_collision_ms2"] = {k: {a: round(b, 1) for a, b in v_.items()} for k, v_ in peak.items()}
        result["min_pair_dist_m"] = {k: round(v, 2) for k, v in sorted(min_pair.items(), key=lambda kv: kv[1])[:12]}
        result.update({"trace": trace, "events": collisions[:MAX_CONTACTS], "contacts_total": len(collisions) + overflow[0], "started_at_s": start_time, "sim_time_s": round(tick * DT, 2),
                       "actors_spawned": sorted(actors), "ego": ego})
        pair = [c for c in collisions if c["other"].startswith(("vehicle", "walker"))]
        if pair:
            result["outcome"] = "collision"
            result["first_collision"] = pair[0]
        elif collisions:
            result["outcome"] = "contact_with_scenery"
        # let the final scene stay visible while finishing any camera glide
        for _ in range(int(hold_sec / DT)):
            try:
                tgt_pose = _chase_target_pose()
                if tgt_pose is not None and cur_cam_pose is not None:
                    cur_cam_pose = smooth_camera_pose(cur_cam_pose, tgt_pose, alpha=0.12)
                    _apply_cam_pose(cur_cam_pose)
            except Exception:
                pass
            paced_tick(world)
            _capture_tick()
        result["actors_alive_at_end"] = sorted(k for k, a in actors.items() if a.is_alive)
        for aid, a in actors.items():
            if a.is_alive and a.type_id.startswith("vehicle"):
                try:
                    a.set_autopilot(False)
                    a.apply_control(carla.VehicleControl(brake=1.0))
                except RuntimeError:
                    pass
    finally:
        try:
            if spawned_ids:
                json.dump(spawned_ids, open(ids_fn, "w"))
        except Exception:
            pass
        if cam_writer is not None:
            try:
                cam_writer.release()
                if record_path and os.path.exists(record_path):
                    result["video_path"] = record_path
            except Exception:
                pass
        if dash_streams:
            dash_vids = {}
            for ds in dash_streams:
                try:
                    if ds["writer"] is not None:
                        ds["writer"].release()
                    if ds["path"] and os.path.exists(ds["path"]):
                        dash_vids[ds["aid"]] = ds["path"]
                except Exception:
                    pass
            result["dashcam_actors"] = [ds["aid"] for ds in dash_streams]
            if dash_vids:
                result["dashcam_videos"] = dash_vids
        for s in sensors:
            try:
                s.stop()
            except Exception:
                pass
        try:
            world.apply_settings(carla.WorldSettings(synchronous_mode=False, fixed_delta_seconds=0.0))
            tm.set_synchronous_mode(False)
        except Exception as e:
            print("[motion] could not restore asynchronous mode:", e)
    res_fn = os.path.join(out_dir, f"{scene_id}_motion_result.json")
    json.dump(result, open(res_fn, "w"), indent=1)
    try:                                                # a failed export must never lose the run result
        meta = {aid: {"type": a["type"], "blueprint": actors[aid].type_id if aid in actors else None,
                      "color": plan["agents"].get(aid, {}).get("color")} for aid, a in agents.items()}
        recorder.save(os.path.join(out_dir, scene_id + TRAJ_SUFFIX), meta, DT)
        print(f"[motion] trajectories -> {scene_id}{TRAJ_SUFFIX}  (python -m agents.xosc_export {out_dir} {scene_id})")
    except Exception as e:
        print("[motion] could not save trajectories:", e)
    print(f"[motion] outcome: {result['outcome']}  ({res_fn})")
    return result


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("scene_id")
    ap.add_argument("--out", default="auto_result")
    ap.add_argument("--no-map", action="store_true", help="keep the currently loaded CARLA map")
    ap.add_argument("--hold", type=float, default=5.0)
    ap.add_argument("--record", action="store_true", help="save CARLA camera video")
    ap.add_argument("--camera", choices=("chase", "dashcam", "both", "dashcam_all"), default="chase", help="camera view for --record")
    ap.add_argument("--dashcam", action="store_true", help="record dashcam footage mounted on the colliding vehicle(s)")
    ap.add_argument("--dashcam-mode", default="colliders", help="which vehicle(s) carry the dashcam: colliders (default), primary, or actor id")
    args = ap.parse_args()
    run_motion(args.out, args.scene_id, load_map=not args.no_map, hold_sec=args.hold, record_video=args.record,
               camera=args.camera, dashcam=args.dashcam, dashcam_mode=args.dashcam_mode)
