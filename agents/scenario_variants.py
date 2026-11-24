"""
Scenario variation generator from a single analysed video / scene spec:
  - Timing variations: shift collision timing earlier/later, create near-misses or clean non-collisions
    by shifting the aggressor's arrival time, or remove the aggressor completely.
  - Road design variations: transfer the exact same corner-case interaction onto different road designs
    (straight arterial, highway, left/right curves, S-curve, lane drop, diverging fork, 1/2/3-lane crossroads)
    without losing geometric quality or CARLA OpenDRIVE validity.
  - Combined matrix: generate road designs and timing variations simultaneously (Cartesian product or paired),
    grouped by road design so CARLA reuses the loaded OpenDRIVE map across timing variants on the same road.

Standalone usage (reuses any existing *_spec.json without re-running VLM analysis):
  python -m agents.scenario_variants --spec results/1/video_interpreter_0000_split_spec.json --out out_var ^
      --road-variants 3 --timing-variants 4 --dashcam [--sim]
"""
import argparse
import copy
import json
import math
import os
import re

from agents import scene_builder, scene_spec

SCENE_ID = "video_interpreter_0000_split"
A_LAT_COMFORT = 3.0       # matches RouteFollower.A_LAT so curved variants never cause unintended slowdown


# ----------------------------------------------------------------------------------------- corner case inspection
def corner_case_info(spec):
    """Extract the semantic corner-case structure from a normalized scene spec."""
    ev = spec.get("event") or {}
    actors = spec.get("actors") or []
    ego = spec.get("ego") or {}
    road = spec.get("road") or {}
    ev_type = ev.get("type", "none")
    ev1 = next((a for a in actors if a.get("id") == ev.get("actor_id")), None) if ev_type != "none" else None
    ev2 = next((a for a in actors if a.get("id") == ev.get("actor2_id")), None) if ev1 is not None else None
    third_party = ev1 is not None and ev2 is not None and ev2 is not ev1

    kind = "none"
    if ev1 is not None:
        if third_party:
            if ev1.get("relation") in ("cross_left", "cross_right") and ev2.get("relation") in ("behind", "beside"):
                kind = "third_party_crossing"
            else:
                kind = "third_party_junction"
        elif ev1.get("type") == "pedestrian":
            kind = "pedestrian"
        elif ev1.get("relation") == "beside" or ev1.get("behavior") == "cutting_in":
            kind = "cut_in"
        elif ev1.get("relation") in ("cross_left", "cross_right"):
            kind = "crossing"
        elif ev1.get("relation") == "ahead":
            kind = "ahead"
        elif ev1.get("relation") == "oncoming":
            kind = "oncoming"
        else:
            kind = "other"

    if third_party:
        dashcam_pair = []
        if ev2.get("type") != "pedestrian":
            dashcam_pair.append(ev2["id"])
        if ev1.get("type") != "pedestrian":
            dashcam_pair.append(ev1["id"])
        subject_id = ev2["id"]
        aggressor_id = ev1["id"]
        subj_speed_kmh = float(ev2.get("speed_kmh") or 50.0)
    elif ev1 is not None:
        dashcam_pair = ["ego"]
        if ev1.get("type") != "pedestrian":
            dashcam_pair.append(ev1["id"])
        subject_id = "ego"
        aggressor_id = ev1["id"]
        subj_speed_kmh = float(ego.get("speed_kmh") or 50.0)
    else:
        dashcam_pair = list(spec.get("dashcam_actors") or ["ego"])
        subject_id = dashcam_pair[0] if dashcam_pair else "ego"
        aggressor_id = None
        subj_speed_kmh = float(ego.get("speed_kmh") or 50.0)

    t_meas = ((ev1.get("measured") or {}).get("t_event_s")) if (ev1 is not None and not third_party) else None
    if t_meas is not None:
        t_sim = min(max(float(t_meas), 3.0), 12.0)
    else:
        t_sim = min(max(float(ev.get("time_s", 5.0)), 3.0), 8.0) if ev1 is not None else 6.0

    min_lanes = 2 if kind in ("cut_in", "third_party_crossing") else 1
    cut_side = (ev1.get("lane") if ev1.get("lane") in ("left", "right") else "left") if kind == "cut_in" else None

    return {
        "kind": kind,
        "third_party": third_party,
        "aggressor_id": aggressor_id,
        "subject_id": subject_id,
        "ev1": ev1,
        "ev2": ev2,
        "dashcam_pair": dashcam_pair,
        "min_lanes": min_lanes,
        "requires_two_way": kind == "oncoming",
        "has_crossing": kind in ("crossing", "third_party_crossing", "third_party_junction"),
        "cut_side": cut_side,
        "subject_speed_ms": max(subj_speed_kmh / 3.6, 1.0),
        "event_time_s": t_sim,
        "road_type": road.get("type", "straight"),
        "lanes": int(road.get("lanes_per_direction", 2) or 2),
    }


# ----------------------------------------------------------------------------------------- smooth road geometries
def _integrate_curvature(k_fn, length_m, step_m=4.0):
    """Integrate a smooth curvature profile k(s) [rad/m] from (0, 0) heading east (+x) to `length_m`."""
    sub_step = 0.5
    n_sub = max(int(math.ceil(length_m / sub_step)), 2)
    ds = length_m / n_sub
    x = y = h = 0.0
    xs, ys, ss = [0.0], [0.0], [0.0]
    for i in range(n_sub):
        s_mid = (i + 0.5) * ds
        k = float(k_fn(s_mid))
        h_mid = h + 0.5 * k * ds
        x += ds * math.cos(h_mid)
        y += ds * math.sin(h_mid)
        h += k * ds
        xs.append(x)
        ys.append(y)
        ss.append((i + 1) * ds)
    pts = []
    n_out = max(int(round(length_m / step_m)), 2)
    idx = 0
    for j in range(n_out + 1):
        s_q = length_m * j / n_out
        while idx + 1 < len(ss) - 1 and ss[idx + 1] < s_q:
            idx += 1
        s0, s1 = ss[idx], ss[idx + 1]
        f = (s_q - s0) / max(s1 - s0, 1e-9)
        px = xs[idx] + f * (xs[idx + 1] - xs[idx])
        py = ys[idx] + f * (ys[idx + 1] - ys[idx])
        pts.append([round(px, 3), round(py, 3)])
    return pts


def _comfort_radius(v_ms, base_radius_m=190.0):
    """Radius (m) keeping lateral acceleration v^2/R <= ~2.2 m/s^2 (< RouteFollower.A_LAT = 3.0) so cars stay at speed."""
    return max(float(base_radius_m), round(1.35 * v_ms * v_ms / A_LAT_COMFORT, 1))


def make_curve_centerline(length_m, v_ms, sign=+1, base_radius_m=190.0, max_deg=65.0, lead_in_m=14.0):
    """Clothoid-eased smooth curve (+1 = left bend, -1 = right bend) with C2 curvature ramp-in/out."""
    R = _comfort_radius(v_ms, base_radius_m)
    k_max = (1.0 if sign >= 0 else -1.0) / R
    arc_len = min(max(length_m - lead_in_m - 25.0, 80.0), R * math.radians(max_deg))
    ramp = min(24.0, 0.25 * arc_len)
    s_start = lead_in_m
    s_end = s_start + arc_len

    def k_fn(s):
        if s <= s_start or s >= s_end:
            return 0.0
        if s < s_start + ramp:
            u = (s - s_start) / ramp
            return k_max * (math.sin(0.5 * math.pi * u) ** 2)
        if s > s_end - ramp:
            u = (s_end - s) / ramp
            return k_max * (math.sin(0.5 * math.pi * u) ** 2)
        return k_max

    return _integrate_curvature(k_fn, length_m, step_m=4.0), R


def make_s_curve_centerline(length_m, v_ms, sign=+1, base_radius_m=210.0, lead_in_m=14.0):
    """Smooth S-bend (left then right, or vice versa) with strictly monotonic +x and comfortable lateral accel."""
    R = _comfort_radius(v_ms, base_radius_m)
    k_max = (1.0 if sign >= 0 else -1.0) / R
    s_span = min(max(length_m - lead_in_m - 40.0, 140.0), 280.0)
    s_end = lead_in_m + s_span

    def k_fn(s):
        if s <= lead_in_m or s >= s_end:
            return 0.0
        u = (s - lead_in_m) / s_span
        return k_max * math.sin(2.0 * math.pi * u)

    return _integrate_curvature(k_fn, length_m, step_m=4.0), R


# ----------------------------------------------------------------------------------------- road design presets
ROAD_PRESETS = {
    "original": {
        "label": "Original road from video",
        "kind": "original",
    },
    "straight_2lane_twoway": {
        "label": "Straight 2-lane two-way arterial with shoulders",
        "road_type": "straight", "lanes": 2, "one_way": False,
        "edge_right": "shoulder", "edge_left": "shoulder", "geometry": "straight",
    },
    "straight_3lane_highway": {
        "label": "Straight 3-lane one-way highway with barriers",
        "road_type": "straight", "lanes": 3, "one_way": True,
        "edge_right": "barrier", "edge_left": "barrier", "geometry": "straight",
    },
    "wide_4lane_highway": {
        "label": "Straight 4-lane one-way motorway with barriers",
        "road_type": "straight", "lanes": 4, "one_way": True,
        "edge_right": "barrier", "edge_left": "barrier", "geometry": "straight",
    },
    "curve_left": {
        "label": "Sweeping left-bending curved road",
        "road_type": "straight", "lanes": 2, "one_way": False,
        "edge_right": "shoulder", "edge_left": "shoulder", "geometry": "curve_left",
    },
    "curve_right": {
        "label": "Sweeping right-bending curved road",
        "road_type": "straight", "lanes": 2, "one_way": False,
        "edge_right": "shoulder", "edge_left": "shoulder", "geometry": "curve_right",
    },
    "s_curve": {
        "label": "Smooth S-bend highway/arterial",
        "road_type": "straight", "lanes": 3, "one_way": True,
        "edge_right": "barrier", "edge_left": "barrier", "geometry": "s_curve",
    },
    "lane_drop": {
        "label": "Multi-lane road with tapered 3-to-2 lane drop",
        "road_type": "straight", "lanes": 3, "one_way": True,
        "edge_right": "barrier", "edge_left": "barrier", "geometry": "lane_drop",
    },
    "highway_fork": {
        "label": "Diverging motorway fork with elevated exit ramp",
        "road_type": "straight", "lanes": 4, "one_way": True,
        "edge_right": "barrier", "edge_left": "barrier", "geometry": "highway_fork",
    },
    "crossroad_2lane": {
        "label": "Signalized 4-way urban intersection (2 lanes/dir)",
        "road_type": "crossroad", "lanes": 2, "one_way": False,
        "traffic_signals": True, "edge_right": "open", "edge_left": "open", "geometry": "crossroad",
    },
    "crossroad_3lane": {
        "label": "Wide signalized boulevard intersection (3 lanes/dir)",
        "road_type": "crossroad", "lanes": 3, "one_way": False,
        "traffic_signals": True, "edge_right": "open", "edge_left": "open", "geometry": "crossroad",
    },
    "crossroad_1lane": {
        "label": "Compact unsignalized rural crossroad (1 lane/dir)",
        "road_type": "crossroad", "lanes": 1, "one_way": False,
        "traffic_signals": False, "edge_right": "open", "edge_left": "open", "geometry": "crossroad",
    },
}


def _original_road_preset_match(spec):
    """Identify which built-in road preset closely matches the original spec, so auto-selection avoids duplicates."""
    road = spec.get("road") or {}
    rtype = road.get("type", "straight")
    lanes = int(road.get("lanes_per_direction", 2) or 2)
    if rtype == "crossroad":
        if lanes <= 1:
            return "crossroad_1lane"
        return "crossroad_3lane" if lanes >= 3 else "crossroad_2lane"
    prof = road.get("lane_profile") or []
    if any(isinstance(sc, dict) and sc.get("branch") for sc in prof):
        return "highway_fork"
    if len(prof) >= 2:
        return "lane_drop"
    cl = road.get("centerline")
    if isinstance(cl, list) and len(cl) > 2:
        y_end = float(cl[-1][1]) - float(cl[0][1])
        if y_end > 4.0:
            return "curve_left"
        if y_end < -4.0:
            return "curve_right"
    if road.get("one_way"):
        return "wide_4lane_highway" if lanes >= 4 else "straight_3lane_highway"
    return "straight_2lane_twoway"


def default_road_presets(spec, count):
    """Choose `count` diverse, corner-case-compatible road design presets starting with 'original'."""
    count = max(1, int(count))
    info = corner_case_info(spec)
    dup = _original_road_preset_match(spec)
    kind = info["kind"]
    if kind == "cut_in":
        pool = [
            "straight_2lane_twoway", "curve_left", "curve_right", "lane_drop",
            "s_curve", "straight_3lane_highway", "highway_fork", "wide_4lane_highway",
        ]
    elif kind in ("crossing", "third_party_crossing", "third_party_junction"):
        pool = [
            "crossroad_2lane", "curve_left", "crossroad_3lane", "straight_2lane_twoway",
            "curve_right", "crossroad_1lane", "s_curve", "straight_3lane_highway",
        ]
    elif kind == "oncoming":
        pool = [
            "curve_left", "curve_right", "straight_2lane_twoway", "s_curve",
            "crossroad_2lane", "crossroad_1lane", "crossroad_3lane",
        ]
    else:   # ahead, pedestrian, none
        pool = [
            "curve_left", "crossroad_2lane", "straight_3lane_highway", "curve_right",
            "lane_drop", "s_curve", "crossroad_3lane", "straight_2lane_twoway", "highway_fork",
        ]

    # Filter out single-lane crossroad when the corner case requires >= 2 lanes, and skip duplicate of original
    ordered = ["original"]
    for p in pool:
        if p == dup:
            continue
        if p == "crossroad_1lane" and info["min_lanes"] >= 2:
            continue
        if p not in ordered:
            ordered.append(p)
    if len(ordered) < count:
        for p in ROAD_PRESETS:
            if p not in ordered and not (p == "crossroad_1lane" and info["min_lanes"] >= 2):
                ordered.append(p)
    return ordered[:count]


def _seed_traffic_from_replay(spec, lanes, one_way, ego_lane):
    """When transferring a straight road with measured replay onto a crossroad, populate modest traffic bins so
    surrounding vehicles remain present instead of turning the intersection empty."""
    tr = dict(spec.get("traffic") or {"ego_lane_ahead": 0, "left_lane": 0, "right_lane": 0, "oncoming": 0, "rel_speed_kmh": 0.0})
    existing_total = sum(int(tr.get(k, 0) or 0) for k in ("ego_lane_ahead", "left_lane", "right_lane", "oncoming"))
    if existing_total > 0:
        return tr
    rep = spec.get("replay") or []
    if not rep:
        return tr
    left_c = right_c = ahead_c = onc_c = 0
    for r in rep:
        kind = r.get("kind", "same")
        path = r.get("path") or []
        if not path:
            continue
        d0 = float(path[0][2])
        if kind == "oncoming" and not one_way:
            onc_c += 1
        elif kind == "same":
            if abs(d0) < 1.8:
                ahead_c += 1
            elif d0 >= 1.8:
                left_c += 1
            else:
                right_c += 1
    tr["ego_lane_ahead"] = min(ahead_c, 2)
    tr["left_lane"] = min(left_c, 3) if (lanes >= 2 and ego_lane == "right") else 0
    tr["right_lane"] = min(right_c, 3) if (lanes >= 2 and ego_lane == "left") else 0
    tr["oncoming"] = min(onc_c, 2) if not one_way else 0
    return tr


def _filter_replay_for_road(replay, lanes, one_way, ego_lane_k):
    """Keep replayed tracks that fit cleanly inside the new road's carriageway without crossing onto opposing lanes."""
    if not replay:
        return []
    W = scene_builder.W
    d_ego = -(lanes - 0.5 - ego_lane_k) * W
    kept = []
    for r in replay:
        path = r.get("path") or []
        if len(path) < 2:
            continue
        kind = r.get("kind", "same")
        if kind == "oncoming" and one_way:
            continue
        d_med = sorted(float(p[2]) for p in path)[len(path) // 2]
        lat = d_ego + d_med
        if kind == "same":
            if -lanes * W - 1.5 <= lat <= 1.5:
                kept.append(copy.deepcopy(r))
        elif kind == "oncoming":
            if -0.5 <= lat <= lanes * W + 1.5:
                kept.append(copy.deepcopy(r))
        else:   # parked
            lo = -lanes * W - 4.0
            hi = 4.0 if one_way else lanes * W + 2.0
            if lo <= lat <= hi:
                kept.append(copy.deepcopy(r))
    return kept


def _reconcile_traffic_lanes(spec):
    """Zero impossible traffic bins for the new road's lane count, one-way flag, and ego lane."""
    road = spec.get("road") or {}
    traffic = spec.get("traffic")
    if not isinstance(traffic, dict):
        return
    if road.get("type") == "crossroad":
        road["one_way"] = False
    if road.get("one_way"):
        traffic["oncoming"] = 0
    lanes = int(road.get("lanes_per_direction", 2) or 2)
    ego_lane = (spec.get("ego") or {}).get("lane", "right")
    if lanes <= 1:
        traffic["left_lane"] = 0
        traffic["right_lane"] = 0
    elif ego_lane == "right":
        traffic["right_lane"] = 0
    elif ego_lane == "left":
        traffic["left_lane"] = 0


def apply_road_design(base_spec, preset_name):
    """Transform `base_spec` onto a different road design (`preset_name`) while preserving the exact same
    corner-case interaction, contact geometry, and CARLA buildability. Returns a new normalized spec."""
    if preset_name not in ROAD_PRESETS:
        raise ValueError("unknown road preset %r (available: %s)" % (preset_name, ", ".join(ROAD_PRESETS)))
    spec = copy.deepcopy(base_spec)
    info = corner_case_info(spec)
    spec["dashcam_actors"] = list(info["dashcam_pair"])
    if preset_name == "original":
        return spec

    cfg = ROAD_PRESETS[preset_name]
    road = spec["road"]
    ego = spec["ego"]
    ev = spec["event"]
    orig_rtype = road.get("type", "straight")
    new_rtype = cfg["road_type"]
    lanes = max(int(cfg["lanes"]), info["min_lanes"])
    one_way = bool(cfg.get("one_way", False))

    # Corner-case physical compatibility guards
    if info["requires_two_way"] or new_rtype == "crossroad":
        one_way = False
    edge_right = cfg.get("edge_right", "shoulder")
    edge_left = cfg.get("edge_left", "shoulder")
    if info["has_crossing"] and new_rtype == "straight":
        # Crossing vehicles need clear shoulders/open edges, never concrete walls blocking entry
        edge_right = "shoulder"
        edge_left = "shoulder"
        if preset_name in ("curve_left", "curve_right", "s_curve"):
            one_way = False

    # Keep lane count wide enough on curves/straight for multi-lane cut-ins when possible
    if info["kind"] == "cut_in" and new_rtype == "straight":
        ev1 = next((a for a in spec["actors"] if a.get("id") == info["aggressor_id"]), None)
        meas = (ev1 or {}).get("measured") if isinstance((ev1 or {}).get("measured"), dict) else None
        if meas and meas.get("lane_offset"):
            max_off = max(1, lanes - 1)
            meas["lane_offset"] = min(int(meas["lane_offset"]), max_off)
            if meas["lane_offset"] <= 1:
                meas.pop("lane_offset", None)

    # Set ego lane cleanly for the new road
    if info["kind"] == "cut_in":
        ego["lane"] = "right" if info["cut_side"] != "right" else "left"
    elif info["kind"] == "third_party_crossing":
        ev2 = next((a for a in spec["actors"] if a.get("id") == info["subject_id"]), None)
        if ev2 is not None:
            ego["lane"] = "left" if ev2.get("lane") == "right" else "right"
    elif lanes == 1:
        ego["lane"] = "right"
    ego["lane_index"] = None

    # Road geometry construction
    v_ref = info["subject_speed_ms"]
    T_est = info["event_time_s"]
    s_ego_est = 65.0 if info["kind"] == "third_party_crossing" else scene_builder.S_START
    geom = cfg["geometry"]
    road_len = max(380.0, round(s_ego_est + v_ref * (T_est + 9.0) + 180.0, 1))
    d_ev_est = max(45.0, v_ref * T_est)

    road["type"] = new_rtype
    road["lanes_per_direction"] = lanes
    road["one_way"] = one_way
    road["edge_right"] = edge_right
    road["edge_left"] = edge_left
    road["traffic_signals"] = bool(cfg.get("traffic_signals", False)) if new_rtype == "crossroad" else False
    road["centerline"] = None
    road["lane_profile"] = None
    road["curve_note"] = None

    if geom == "curve_left":
        cl, R = make_curve_centerline(road_len, v_ref, sign=+1)
        road["centerline"] = cl
        road["curve_note"] = "synthetic road variant: left curve (R=%.0f m, L=%.0f m)" % (R, road_len)
    elif geom == "curve_right":
        cl, R = make_curve_centerline(road_len, v_ref, sign=-1)
        road["centerline"] = cl
        road["curve_note"] = "synthetic road variant: right curve (R=%.0f m, L=%.0f m)" % (R, road_len)
    elif geom == "s_curve":
        cl, R = make_s_curve_centerline(road_len, v_ref, sign=+1)
        road["centerline"] = cl
        road["curve_note"] = "synthetic road variant: S-curve (R_min=%.0f m, L=%.0f m)" % (R, road_len)
    elif geom == "lane_drop":
        # 3 -> 2 lane drop cleanly placed after the event so ego and event actor always have valid lanes
        lanes = max(3, lanes)
        road["lanes_per_direction"] = lanes
        d_drop = round(max(d_ev_est + 30.0, 85.0), 1)
        # Drop the outer lane on the cut-in side (realistic forced merge ahead!) or left side by default
        drop_left = info["cut_side"] != "right"
        if drop_left:
            road["lane_profile"] = [
                {"d0": None, "d1": d_drop, "n": lanes, "m": 0},
                {"d0": d_drop, "d1": None, "n": lanes - 1, "m": 1},
            ]
            ego["lane_index"] = lanes - 1 if ego["lane"] == "right" else 1
        else:
            road["lane_profile"] = [
                {"d0": None, "d1": d_drop, "n": lanes, "m": 0},
                {"d0": d_drop, "d1": None, "n": lanes - 1, "m": 0},
            ]
            ego["lane_index"] = 0 if ego["lane"] == "left" else lanes - 2
        road["curve_note"] = "synthetic road variant: %d->%d lane drop at %.0f m" % (lanes, lanes - 1, d_drop)
    elif geom == "highway_fork":
        # 4 -> 3 (or 3 -> 2) diverging motorway fork with elevated exit ramp on the non-conflicting side
        lanes = max(4, lanes) if info["kind"] == "cut_in" else max(3, lanes)
        road["lanes_per_direction"] = lanes
        branch_left = info["cut_side"] != "right"
        # Place split before the event while keeping >= 2 continuing lanes on the ego side
        d_split = round(max(45.0, min(d_ev_est * 0.55, 110.0)), 1)
        if branch_left:
            road["lane_profile"] = [
                {"d0": None, "d1": d_split, "n": lanes, "m": 0},
                {"d0": d_split, "d1": None, "n": lanes - 1, "m": 1, "branch": {"side": "left", "rise_m": 6.0}},
            ]
            ego["lane_index"] = lanes - 1 if ego["lane"] == "right" else 1
        else:
            road["lane_profile"] = [
                {"d0": None, "d1": d_split, "n": lanes, "m": 0},
                {"d0": d_split, "d1": None, "n": lanes - 1, "m": 0, "branch": {"side": "right", "rise_m": 6.0}},
            ]
            ego["lane_index"] = 0 if ego["lane"] == "left" else lanes - 2
        road["curve_note"] = "synthetic road variant: %d->%d diverging highway fork at %.0f m" % (lanes, lanes - 1, d_split)
    elif geom == "straight":
        road["curve_note"] = "synthetic road variant: %s" % cfg["label"]

    # Adapt actors / traffic / replay cleanly across road topology changes
    if new_rtype == "crossroad":
        if orig_rtype != "crossroad":
            spec["traffic"] = _seed_traffic_from_replay(spec, lanes, one_way=False, ego_lane=ego["lane"])
            spec["replay"] = []
        # Ensure ego approach fits inside the 100 m crossroad arm cleanly
        if ego.get("behavior") == "cruise" and ego.get("speed_kmh", 0.0) > 65.0:
            ego["speed_kmh"] = 60.0
            ego["speed_profile"] = None
        if info["kind"] == "cut_in":
            # Clamp T so cut-in completes within the 180 m crossroad corridor
            v_use = max(float(ego.get("speed_kmh", 50.0)) / 3.6, 5.0)
            max_T = min(6.0, 115.0 / v_use)
            ev["time_s"] = round(min(float(ev.get("time_s", 5.0)), max_T), 2)
            ev1 = next((a for a in spec["actors"] if a.get("id") == info["aggressor_id"]), None)
            if ev1 and isinstance(ev1.get("measured"), dict) and ev1["measured"].get("t_event_s") is not None:
                ev1["measured"]["t_event_s"] = round(min(float(ev1["measured"]["t_event_s"]), max_T), 2)
    else:
        # Moving from crossroad -> straight/curved: drop crossroad-only stopped cross-street queues
        if orig_rtype == "crossroad":
            keep_ids = {info["aggressor_id"], info["subject_id"]}
            spec["actors"] = [
                a for a in spec.get("actors") or []
                if a.get("id") in keep_ids or not (a.get("relation") in ("cross_left", "cross_right") and a.get("behavior") == "stopped")
            ]
        # Filter measured replay tracks so none sit on removed or opposing lanes
        ego_k = 0 if ego["lane"] == "right" or lanes == 1 else lanes - 1
        spec["replay"] = _filter_replay_for_road(spec.get("replay") or [], lanes, one_way, ego_k)
        if spec.get("replay") and isinstance(spec.get("traffic"), dict):
            for k in ("ego_lane_ahead", "left_lane", "right_lane", "oncoming"):
                spec["traffic"][k] = 0

    _reconcile_traffic_lanes(spec)
    notes = list(spec.get("notes") or [])
    notes.append("road design variant: %s (%s)" % (preset_name, cfg["label"]))
    spec["notes"] = notes
    return spec


# ----------------------------------------------------------------------------------------- timing variation presets
TIMING_PRESETS = {
    "original": {
        "label": "Original collision timing from footage",
        "mode": "original",
    },
    "earlier_collision": {
        "label": "Same collision ~1.5 s earlier",
        "mode": "shift_collision", "time_shift_s": -1.5,
    },
    "later_collision": {
        "label": "Same collision ~1.5 s later",
        "mode": "shift_collision", "time_shift_s": 1.5,
    },
    "near_miss_early": {
        "label": "Aggressor arrives ~0.9 s early (close near-miss, no collision)",
        "mode": "offset", "outcome": "near_miss", "aggressor_dt_s": -0.9, "cutin_long_m": 6.8,
    },
    "near_miss_late": {
        "label": "Aggressor arrives ~0.9 s late (close near-miss, no collision)",
        "mode": "offset", "outcome": "near_miss", "aggressor_dt_s": 0.9, "cutin_long_m": -7.2,
    },
    "miss_early": {
        "label": "Aggressor arrives ~2.2 s early (clean miss, no collision)",
        "mode": "offset", "outcome": "miss", "aggressor_dt_s": -2.2, "cutin_long_m": 14.0,
    },
    "miss_late": {
        "label": "Aggressor arrives ~2.2 s late (clean miss, no collision)",
        "mode": "offset", "outcome": "miss", "aggressor_dt_s": 2.2, "cutin_long_m": -14.0,
    },
    "no_aggressor": {
        "label": "Aggressor vehicle does not appear at all (benign baseline)",
        "mode": "no_aggressor", "outcome": "none",
    },
}

DEFAULT_TIMING_ORDER = (
    "original",
    "miss_late",
    "no_aggressor",
    "near_miss_early",
    "earlier_collision",
    "miss_early",
    "later_collision",
    "near_miss_late",
)

CUSTOM_DT_RE = re.compile(r"^(dt|shift):([+-]?\d+(?:\.\d+)?)$", re.I)


def _resolve_timing_cfg(preset_name):
    if preset_name in TIMING_PRESETS:
        return dict(TIMING_PRESETS[preset_name])
    m = CUSTOM_DT_RE.match(str(preset_name).strip())
    if m:
        kind, val = m.group(1).lower(), float(m.group(2))
        if kind == "shift":
            return {"label": "Collision shifted by %+.1f s" % val, "mode": "shift_collision", "time_shift_s": val}
        abs_v = abs(val)
        if abs_v < 0.45:
            outcome = "collision"
        elif abs_v <= 1.25:
            outcome = "near_miss"
        else:
            outcome = "miss"
        sign = -1.0 if val > 0 else 1.0      # positive aggressor_dt (late) -> behind (-long), negative (early) -> ahead (+long)
        return {
            "label": "Aggressor offset %+.1f s (%s)" % (val, outcome),
            "mode": "offset", "outcome": outcome, "aggressor_dt_s": val,
            "cutin_long_m": round(sign * max(5.5, min(20.0, abs_v * 6.5)), 1),
        }
    raise ValueError("unknown timing preset %r (available: %s, or dt:<sec> / shift:<sec>)" % (
        preset_name, ", ".join(TIMING_PRESETS)))


def default_timing_presets(spec, count):
    """Choose `count` well-balanced timing variations (collision, non-collision timing shifts, and aggressor-absent)."""
    count = max(1, int(count))
    out = list(DEFAULT_TIMING_ORDER[:count])
    extra = ("dt:+1.5", "dt:-1.5", "shift:+2.5", "shift:-2.5", "dt:+3.0", "dt:-3.0")
    for ex in extra:
        if len(out) >= count:
            break
        out.append(ex)
    return out[:count]


def apply_timing_variant(base_spec, preset_name):
    """Apply a timing variation to `base_spec` (which may already have been transformed onto another road design).
    Supports shifted-time collisions, near-misses, clean timing misses where cars don't collide, and removing the
    aggressor completely while keeping dashcam_actors pointed at the surviving colliding/subject vehicle(s)."""
    cfg = _resolve_timing_cfg(preset_name)
    spec = copy.deepcopy(base_spec)
    info = corner_case_info(spec)
    ev = spec["event"]
    ego = spec["ego"]
    road = spec["road"]
    mode = cfg["mode"]

    spec["dashcam_actors"] = list(info["dashcam_pair"])

    if mode == "original" or info["aggressor_id"] is None:
        return spec

    ev1 = next((a for a in spec["actors"] if a.get("id") == info["aggressor_id"]), None)
    ev2 = next((a for a in spec["actors"] if a.get("id") == info["subject_id"]), None) if info["third_party"] else None

    if mode == "no_aggressor":
        # Remove the aggressor completely while preserving road, surrounding traffic, and subject vehicle!
        spec["actors"] = [a for a in spec["actors"] if a.get("id") != info["aggressor_id"]]
        if info["third_party"] and ev2 is not None:
            # Keep the surviving colliding vehicle (e.g. straight-driving van c2) driving smoothly & carrying dashcam!
            spec["dashcam_actors"] = [ev2["id"]]
            if road.get("type") == "straight" and ev2.get("relation") == "behind":
                spec["ego_s_start"] = 65.0
                ev2["distance_m"] = min(max(float(ev2.get("distance_m", 25.0)), 18.0), 45.0)
            elif road.get("type") == "crossroad" and ev2.get("relation") == "behind":
                ev2["distance_m"] = min(max(float(ev2.get("distance_m", 25.0)), 18.0), 45.0)
        else:
            spec["dashcam_actors"] = ["ego"]
        ev["type"] = "none"
        ev["actor_id"] = None
        ev["actor2_id"] = None
        ev.pop("contact_parts", None)
        ev.pop("aggressor_time_offset_s", None)
        ev.pop("aggressor_long_offset_m", None)
        ev.pop("time_shift_s", None)
        ev["description"] = "aggressor absent (timing variant: %s)" % preset_name
        notes = list(spec.get("notes") or [])
        notes.append("timing variant: %s (%s)" % (preset_name, cfg["label"]))
        spec["notes"] = notes
        return spec

    if mode == "shift_collision":
        shift_s = float(cfg["time_shift_s"])
        t_cur = info["event_time_s"]
        if road.get("type") == "crossroad":
            v_ref = info["subject_speed_ms"]
            max_arm_t = min(7.5, 86.0 / max(v_ref, 1.0))
            t_new = round(min(max(t_cur + shift_s, 2.8), max(max_arm_t, 3.0)), 2)
        else:
            t_new = round(min(max(t_cur + shift_s, 3.0), 11.5), 2)
        ev["time_s"] = t_new
        ev["time_shift_s"] = round(t_new - t_cur, 2)
        if ev1 is not None and isinstance(ev1.get("measured"), dict) and ev1["measured"].get("t_event_s") is not None:
            ev1["measured"]["t_event_s"] = t_new
        notes = list(spec.get("notes") or [])
        notes.append("timing variant: %s (event time %.2f s -> %.2f s)" % (preset_name, t_cur, t_new))
        spec["notes"] = notes
        return spec

    # mode == "offset": shift aggressor arrival relative to subject -> near_miss or clean miss (no collision!)
    outcome = cfg["outcome"]
    dt_s = float(cfg["aggressor_dt_s"])
    ev["type"] = outcome
    if outcome != "collision":
        ev.pop("contact_parts", None)

    if info["kind"] == "cut_in":
        long_m = float(cfg["cutin_long_m"])
        ev["aggressor_long_offset_m"] = round(long_m, 2)
        ev["aggressor_time_offset_s"] = round(dt_s, 2)
        # If cut-in happens ahead of ego, ensure cutter speed matches ego so ego doesn't rear-end a slow crawler
        if ev1 is not None and long_m > 3.0:
            v_ego_kmh = float(ego.get("speed_kmh", 50.0))
            ev1["speed_kmh"] = max(float(ev1.get("speed_kmh", 20.0)), v_ego_kmh)
            if isinstance(ev1.get("measured"), dict):
                ev1["measured"] = dict(ev1["measured"])
                ev1["measured"]["overtake"] = False
                ev1["measured"].pop("side_strike", None)
        elif ev1 is not None and long_m < -3.0:
            if isinstance(ev1.get("measured"), dict):
                ev1["measured"] = dict(ev1["measured"])
                ev1["measured"].pop("side_strike", None)
                ev1["measured"]["overtake"] = False
    else:
        ev["aggressor_time_offset_s"] = round(dt_s, 2)

    ev["description"] = "%s (%s)" % (ev.get("description") or info["kind"], cfg["label"])
    notes = list(spec.get("notes") or [])
    notes.append("timing variant: %s (%s)" % (preset_name, cfg["label"]))
    spec["notes"] = notes
    return spec


# ----------------------------------------------------------------------------------------- parsing & matrix generation
def _safe_slug(text):
    return re.sub(r"[^a-zA-Z0-9_-]+", "_", str(text).strip()).strip("_") or "var"


def parse_variant_arg(arg, kind, spec):
    """Parse CLI/API variant specification: None -> ['original'], integer/digits -> default N presets,
    'all' -> all compatible presets, or comma-separated preset names."""
    if arg is None or str(arg).strip() == "":
        return ["original"]
    if isinstance(arg, (list, tuple)):
        items = [str(x).strip() for x in arg if str(x).strip()]
        return items or ["original"]
    raw = str(arg).strip()
    if raw.lower() == "all":
        if kind == "road":
            return default_road_presets(spec, len(ROAD_PRESETS))
        return list(DEFAULT_TIMING_ORDER)
    if raw.isdigit():
        n = int(raw)
        if n <= 0:
            return ["original"]
        return default_road_presets(spec, n) if kind == "road" else default_timing_presets(spec, n)
    items = [x.strip() for x in raw.split(",") if x.strip()]
    return items or ["original"]


def generate_variants(base_spec, road_presets=None, timing_presets=None, combine="product"):
    """Generate a list of variant specs from `base_spec`, supporting road-only, timing-only, or combined
    road + timing generation.

    combine:
      - 'product' (default): full Cartesian product road_presets x timing_presets, grouped by road_preset
        so CARLA can reuse the same loaded OpenDRIVE map across timing variants on each road.
      - 'zip': 1-to-1 pairing (cycling the shorter list if lengths differ).

    Returns [{"id", "road", "timing", "expected_outcome", "spec"}, ...].
    """
    clean_base, err = scene_spec.normalize(base_spec)
    if clean_base is None:
        raise ValueError("invalid base spec: %s" % err)
    roads = parse_variant_arg(road_presets, "road", clean_base)
    timings = parse_variant_arg(timing_presets, "timing", clean_base)

    pairs = []
    if combine == "zip":
        n = max(len(roads), len(timings))
        for i in range(n):
            pairs.append((roads[i % len(roads)], timings[i % len(timings)]))
        # Stable sort by road preset to maximize CARLA map reuse while keeping deterministic order
        road_rank = {r: idx for idx, r in enumerate(roads)}
        pairs.sort(key=lambda rt: road_rank.get(rt[0], 999))
    else:
        for r in roads:
            for t in timings:
                pairs.append((r, t))

    out = []
    seen_ids = set()
    for r_name, t_name in pairs:
        road_spec = apply_road_design(clean_base, r_name)
        var_spec = apply_timing_variant(road_spec, t_name)
        vid = "road_%s__timing_%s" % (_safe_slug(r_name), _safe_slug(t_name))
        if vid in seen_ids:
            k = 2
            while "%s_%d" % (vid, k) in seen_ids:
                k += 1
            vid = "%s_%d" % (vid, k)
        seen_ids.add(vid)
        var_spec["variant"] = {
            "id": vid,
            "road": r_name,
            "timing": t_name,
            "expected_outcome": var_spec["event"]["type"],
            "dashcam_actors": list(var_spec.get("dashcam_actors") or []),
        }
        norm_var, verr = scene_spec.normalize(var_spec)
        if norm_var is None:
            raise RuntimeError("generated invalid variant %s: %s" % (vid, verr))
        out.append({
            "id": vid,
            "road": r_name,
            "timing": t_name,
            "expected_outcome": norm_var["event"]["type"],
            "spec": norm_var,
        })
    return out


def format_variants_report(manifest):
    """Readable summary table of all built (and optionally simulated) scenario variants."""
    lines = [
        "Scenario variants: %d total (roads=%s, timings=%s, combine=%s)" % (
            len(manifest.get("variants") or []),
            ",".join(manifest.get("roads") or []),
            ",".join(manifest.get("timings") or []),
            manifest.get("combine", "product"),
        )
    ]
    for item in manifest.get("variants") or []:
        parts = [
            "  - %s: road=%s (%dL%s), timing=%s, expect=%s" % (
                item["id"], item["road_type"], item["lanes_per_direction"],
                ", curved" if item.get("curved") else (", fork/drop" if item.get("varying") else ""),
                item["timing"], item["expected_outcome"],
            )
        ]
        if item.get("dashcam_actors"):
            parts.append("dashcam=%s" % ",".join(item["dashcam_actors"]))
        if item.get("sim_status"):
            parts.append("sim=%s" % item["sim_status"])
        if item.get("xosc"):
            parts.append("xosc=%s" % item["xosc"])
        if item.get("dashcam_videos"):
            parts.append("videos=%s" % ",".join(os.path.basename(p) for p in item["dashcam_videos"].values()))
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def build_variants(base_spec, out_dir, road_presets=None, timing_presets=None, combine="product",
                   scene_id=SCENE_ID, log_fn=print):
    """Build every requested road/timing variant into `<out_dir>/variants/<variant_id>/` and write
    `variants_manifest.json` and `variants_report.txt`. Returns the manifest dict."""
    os.makedirs(out_dir, exist_ok=True)
    variants = generate_variants(base_spec, road_presets=road_presets, timing_presets=timing_presets, combine=combine)
    var_root = os.path.join(out_dir, "variants")
    os.makedirs(var_root, exist_ok=True)
    entries = []
    for item in variants:
        vdir = os.path.join(var_root, item["id"])
        os.makedirs(vdir, exist_ok=True)
        plan = scene_builder.build(item["spec"], vdir, scene_id)
        road_sp = item["spec"]["road"]
        entry = {
            "id": item["id"],
            "dir": vdir,
            "rel_dir": os.path.join("variants", item["id"]),
            "road": item["road"],
            "timing": item["timing"],
            "road_type": road_sp["type"],
            "lanes_per_direction": road_sp["lanes_per_direction"],
            "one_way": bool(road_sp.get("one_way")),
            "curved": bool(road_sp.get("centerline") and len(road_sp["centerline"]) > 2),
            "varying": bool(road_sp.get("lane_profile")),
            "expected_outcome": item["expected_outcome"],
            "expect": plan.get("expect"),
            "dashcam_actors": list(plan.get("dashcam_actors") or (plan.get("expect") or {}).get("actors") or ["v1"]),
            "scene_id": scene_id,
        }
        entries.append(entry)
        if log_fn:
            log_fn("[variants] built %s (road=%s, timing=%s, expect=%s, dashcam=%s)" % (
                item["id"], item["road"], item["timing"], item["expected_outcome"], entry["dashcam_actors"]))

    manifest = {
        "scene_id": scene_id,
        "combine": combine,
        "roads": list(dict.fromkeys(e["road"] for e in entries)),
        "timings": list(dict.fromkeys(e["timing"] for e in entries)),
        "variants": entries,
    }
    with open(os.path.join(out_dir, "variants_manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=1)
    rep = format_variants_report(manifest)
    with open(os.path.join(out_dir, "variants_report.txt"), "w", encoding="utf-8") as f:
        f.write(rep)
    return manifest


def main(argv=None):
    ap = argparse.ArgumentParser(description="Generate timing and road-design scenario variants from a scene spec")
    ap.add_argument("--spec", required=True, help="path to base *_spec.json")
    ap.add_argument("--out", required=True, help="output directory for generated variants")
    ap.add_argument("--timing-variants", default=None,
                    help="timing variants: count (e.g. 4), 'all', or comma-separated presets (%s)" % ",".join(TIMING_PRESETS))
    ap.add_argument("--road-variants", default=None,
                    help="road variants: count (e.g. 4), 'all', or comma-separated presets (%s)" % ",".join(ROAD_PRESETS))
    ap.add_argument("--combine", choices=("product", "zip"), default="product",
                    help="how to combine road and timing variants: full Cartesian 'product' (default) or 1-to-1 'zip'")
    ap.add_argument("--sim", action="store_true", help="also simulate every variant in CARLA and export .xosc")
    ap.add_argument("--max-iter", type=int, default=8)
    ap.add_argument("--record", action="store_true", help="record CARLA video for each simulated variant")
    ap.add_argument("--camera", choices=("chase", "dashcam", "both", "dashcam_all"), default="chase")
    ap.add_argument("--dashcam", action="store_true", help="record dashcam footage mounted on the colliding vehicle(s)")
    ap.add_argument("--dashcam-mode", default="colliders", help="colliders (default), primary, or actor id")
    args = ap.parse_args(argv)

    with open(args.spec, encoding="utf-8") as f:
        base_spec = json.load(f)
    manifest = build_variants(
        base_spec, args.out,
        road_presets=args.road_variants,
        timing_presets=args.timing_variants,
        combine=args.combine,
    )
    if args.sim:
        from agents import local_pipeline
        manifest = local_pipeline.phase2_variants(
            args.out, manifest=manifest, max_iter=args.max_iter,
            record_video=args.record, camera=args.camera,
            dashcam=args.dashcam, dashcam_mode=args.dashcam_mode,
        )
    print(format_variants_report(manifest))


if __name__ == "__main__":
    main()
