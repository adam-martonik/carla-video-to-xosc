"""Road layout estimation, ASAM OpenDRIVE (.xodr) generation, and Frenet-to-World projection."""

from __future__ import annotations

import math
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .config import PipelineConfig

try:
    import carla
except ImportError:
    carla = None


def estimate_road_geometry(resampled: Dict[str, Any], cfg: PipelineConfig) -> Dict[str, Any]:
    """
    Estimate road corridor parameters from smoothed ego & actor Frenet trajectories:
      - total road length covering all actors with margin
      - mean constant curvature kappa (1/m) from ego yaw_rate / speed (clamped for stability)
      - number of left and right driving lanes from observed lateral spread
    """
    road_cfg = cfg.road
    all_s = list(resampled["ego"]["s"])
    all_d = list(resampled["ego"]["d"])
    for act in resampled.get("actors", []):
        all_s.extend(act["s"])
        all_d.extend(act["d"])

    min_s = float(min(all_s)) if all_s else 0.0
    max_s = float(max(all_s)) if all_s else 200.0
    road_start_s = min(0.0, min_s - 25.0)
    road_length = max(250.0, (max_s - road_start_s) + 80.0)

    # Estimate mean curvature from ego yaw_rate / speed where speed > 3 m/s
    ego_v = np.asarray(resampled["ego"]["v"], dtype=float)
    ego_yr = np.asarray(resampled["ego"].get("yaw_rate", np.zeros_like(ego_v)), dtype=float)
    moving = ego_v > 3.0
    if np.any(moving):
        kappas = ego_yr[moving] / np.maximum(ego_v[moving], 3.0)
        kappa = float(np.clip(np.median(kappas), -road_cfg.max_curvature_inv_m, road_cfg.max_curvature_inv_m))
        # Treat tiny noise curvatures as straight road
        if abs(kappa) < 5e-4:
            kappa = 0.0
    else:
        kappa = 0.0

    max_right_d = max(0.0, max(all_d)) if all_d else 0.0
    max_left_d = max(0.0, -min(all_d)) if all_d else 0.0
    n_right = max(road_cfg.default_lanes_right, int(math.ceil((max_right_d + 1.5) / road_cfg.lane_width_m)))
    n_left = max(road_cfg.default_lanes_left, int(math.ceil((max_left_d + 1.5) / road_cfg.lane_width_m)))
    n_right = min(n_right, 4)
    n_left = min(n_left, 4)

    return {
        "road_start_s": road_start_s,
        "length": road_length,
        "curvature": kappa,
        "lane_width": road_cfg.lane_width_m,
        "shoulder_width": road_cfg.shoulder_width_m,
        "n_right_lanes": n_right,
        "n_left_lanes": n_left,
    }


def write_opendrive_xodr(road_geom: Dict[str, Any], output_xodr: Path) -> Path:
    """
    Generate a clean, self-contained ASAM OpenDRIVE 1.4 (.xodr) road network
    representing the reconstructed multi-lane road corridor (straight or constant-curvature arc).
    """
    root = ET.Element("OpenDRIVE")
    ET.SubElement(
        root,
        "header",
        attrib={
            "revMajor": "1",
            "revMinor": "4",
            "name": "DashcamReconstructedRoad",
            "version": "1.00",
            "date": "2026-11-19T12:00:00",
            "north": f"{road_geom['length']:.2f}",
            "south": "-50.00",
            "east": f"{road_geom['length']:.2f}",
            "west": "-50.00",
        },
    )

    road = ET.SubElement(
        root,
        "road",
        attrib={
            "name": "ReconstructedCorridor",
            "length": f"{road_geom['length']:.3f}",
            "id": "1",
            "junction": "-1",
        },
    )
    ET.SubElement(road, "link")
    rtype = ET.SubElement(road, "type", attrib={"s": "0.000", "type": "motorway"})
    ET.SubElement(rtype, "speed", attrib={"max": "36.11", "unit": "m/s"})

    plan = ET.SubElement(road, "planView")
    geom = ET.SubElement(
        plan,
        "geometry",
        attrib={
            "s": "0.000",
            "x": "0.000",
            "y": "0.000",
            "hdg": "0.000000",
            "length": f"{road_geom['length']:.3f}",
        },
    )
    if abs(road_geom["curvature"]) < 1e-6:
        ET.SubElement(geom, "line")
    else:
        ET.SubElement(geom, "arc", attrib={"curvature": f"{road_geom['curvature']:.6f}"})

    elev = ET.SubElement(road, "elevationProfile")
    ET.SubElement(elev, "elevation", attrib={"s": "0.000", "a": "0.0", "b": "0.0", "c": "0.0", "d": "0.0"})

    lanes = ET.SubElement(road, "lanes")
    lsec = ET.SubElement(lanes, "laneSection", attrib={"s": "0.000"})

    lw = road_geom["lane_width"]
    sw = road_geom["shoulder_width"]

    # Left lanes (id > 0)
    left_el = ET.SubElement(lsec, "left")
    for lid in range(road_geom["n_left_lanes"] + 1, 0, -1):
        is_shoulder = lid == road_geom["n_left_lanes"] + 1
        lane_el = ET.SubElement(
            left_el,
            "lane",
            attrib={"id": str(lid), "type": "shoulder" if is_shoulder else "driving", "level": "false"},
        )
        ET.SubElement(lane_el, "width", attrib={"sOffset": "0.0", "a": f"{sw if is_shoulder else lw:.3f}", "b": "0.0", "c": "0.0", "d": "0.0"})
        ET.SubElement(lane_el, "roadMark", attrib={"sOffset": "0.0", "type": "solid" if is_shoulder else "broken", "weight": "standard", "color": "white", "width": "0.12"})

    # Center lane (id = 0)
    center_el = ET.SubElement(lsec, "center")
    c_lane = ET.SubElement(center_el, "lane", attrib={"id": "0", "type": "none", "level": "false"})
    ET.SubElement(c_lane, "roadMark", attrib={"sOffset": "0.0", "type": "broken", "weight": "standard", "color": "yellow", "width": "0.15"})

    # Right lanes (id < 0)
    right_el = ET.SubElement(lsec, "right")
    for idx in range(1, road_geom["n_right_lanes"] + 2):
        lid = -idx
        is_shoulder = idx == road_geom["n_right_lanes"] + 1
        lane_el = ET.SubElement(
            right_el,
            "lane",
            attrib={"id": str(lid), "type": "shoulder" if is_shoulder else "driving", "level": "false"},
        )
        ET.SubElement(lane_el, "width", attrib={"sOffset": "0.0", "a": f"{sw if is_shoulder else lw:.3f}", "b": "0.0", "c": "0.0", "d": "0.0"})
        ET.SubElement(lane_el, "roadMark", attrib={"sOffset": "0.0", "type": "solid" if is_shoulder else "broken", "weight": "standard", "color": "white", "width": "0.12"})

    xml_bytes = ET.tostring(root, encoding="utf-8")
    pretty_xml = minidom.parseString(xml_bytes).toprettyxml(indent="  ")
    output_xodr.parent.mkdir(parents=True, exist_ok=True)
    output_xodr.write_text(pretty_xml, encoding="utf-8")
    return output_xodr


def _frenet_to_world_analytic(s_val: float, d_val: float, road_geom: Dict[str, Any]) -> Tuple[float, float, float]:
    """
    Map Frenet (s, d) -> right-handed OpenDRIVE world (x, y, road_hdg) along a straight line
    or constant-curvature circular arc. In Frenet frame d > 0 is to the right of road heading,
    which in right-handed (x forward, y left) corresponds to normal vector (sin(hdg), -cos(hdg)).
    """
    s_shifted = float(s_val) - road_geom["road_start_s"]
    kappa = road_geom["curvature"]
    if abs(kappa) < 1e-6:
        x_ref = s_shifted
        y_ref = 0.0
        hdg = 0.0
    else:
        r = 1.0 / kappa
        theta = kappa * s_shifted
        x_ref = r * math.sin(theta)
        y_ref = r * (1.0 - math.cos(theta))
        hdg = theta

    # Right-hand normal vector for d > 0 (right of travel direction)
    x = x_ref + d_val * math.sin(hdg)
    y = y_ref - d_val * math.cos(hdg)
    return x, y, hdg


def _compute_smooth_headings(
    world_poses: list[Dict[str, float]],
    times: list[float] | None = None,
    is_oncoming: bool = False,
    max_yaw_rate_rad_s: float = 0.50,
) -> None:
    """
    Compute rate-limited tangent headings from world poses while avoiding jitter or 180-deg
    flips when a vehicle is stationary or crawling slowly.
    """
    n = len(world_poses)
    if n < 2:
        return
    last_valid_h = world_poses[0]["h"] + (math.pi if is_oncoming else 0.0)
    for i in range(n - 1):
        dx = world_poses[i + 1]["x"] - world_poses[i]["x"]
        dy = world_poses[i + 1]["y"] - world_poses[i]["y"]
        dt = (times[i + 1] - times[i]) if (times is not None and i + 1 < len(times)) else 0.25
        dt = max(float(dt), 0.05)
        if math.hypot(dx, dy) > 0.35:
            raw_h = math.atan2(dy, dx)
            dh = (raw_h - last_valid_h + math.pi) % (2.0 * math.pi) - math.pi
            max_dh = max_yaw_rate_rad_s * dt
            last_valid_h = last_valid_h + float(np.clip(dh, -max_dh, max_dh))
        world_poses[i]["h"] = last_valid_h
    world_poses[-1]["h"] = last_valid_h


def project_frenet_offline(
    resampled: Dict[str, Any],
    cfg: PipelineConfig,
    output_xodr: Path | None = None,
    town_label: str = "Town04_Opt",
) -> Dict[str, Any]:
    """
    Project Frenet trajectories to world coordinates using the reconstructed OpenDRIVE
    road geometry (no live CARLA server required). Optionally writes the .xodr file.
    """
    projected = dict(resampled)
    road_geom = estimate_road_geometry(resampled, cfg)
    projected["road_geometry"] = road_geom

    if output_xodr is not None and cfg.road.generate_xodr:
        write_opendrive_xodr(road_geom, output_xodr)
        projected["xodr_path"] = str(output_xodr)

    all_entities = [projected["ego"]] + projected["actors"]
    for ent in all_entities:
        z_surf = 0.0
        poses = []
        for s_val, d_val in zip(ent["s"], ent["d"]):
            x, y, hdg = _frenet_to_world_analytic(s_val, d_val, road_geom)
            poses.append({"x": x, "y": y, "z": z_surf, "h": hdg})
        _compute_smooth_headings(poses, times=ent.get("times"), is_oncoming=bool(ent.get("is_oncoming", False)))
        ent["world_poses"] = poses

    projected["town"] = town_label
    projected["projection_mode"] = "offline_xodr"
    return projected


def project_frenet_to_carla_map(
    resampled: Dict[str, Any],
    cfg: PipelineConfig,
    host: str = "127.0.0.1",
    port: int = 2000,
    town: str = "Town04_Opt",
    output_xodr: Path | None = None,
) -> Dict[str, Any]:
    """
    Connect to CARLA server, load `town`, find a multi-lane road corridor, and project all
    (s, d) Frenet trajectories into CARLA WorldPosition (x, y, z, h).
    Falls back cleanly to offline OpenDRIVE projection if CARLA server is unreachable.
    """
    if carla is None:
        return project_frenet_offline(resampled, cfg, output_xodr=output_xodr, town_label=town)

    try:
        client = carla.Client(host, port)
        client.set_timeout(10.0)
        world = client.get_world()
    except Exception as exc:
        print(f"[Stage D] CARLA server unreachable ({exc}); using offline OpenDRIVE projection.")
        return project_frenet_offline(resampled, cfg, output_xodr=output_xodr, town_label=town)

    current_map_name = world.get_map().name.split("/")[-1]
    if current_map_name != town:
        print(f"[Stage D] Loading CARLA map '{town}' (current: '{current_map_name}')...")
        world = client.load_world(town)

    carla_map = world.get_map()
    required_length = max(
        max(resampled["ego"]["s"]),
        max((max(a["s"]) for a in resampled["actors"]), default=100.0),
    ) + 60.0

    best_start_wp = None
    best_chain_len = 0.0

    for sp in carla_map.get_spawn_points():
        wp = carla_map.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            continue
        left_wp = wp.get_left_lane()
        right_wp = wp.get_right_lane()
        has_both_sides = (
            left_wp is not None
            and left_wp.lane_type == carla.LaneType.Driving
            and right_wp is not None
            and right_wp.lane_type == carla.LaneType.Driving
        )
        if not has_both_sides:
            continue
        cur = wp
        dist = 0.0
        step = 5.0
        while dist < required_length:
            nxt = cur.next(step)
            if not nxt:
                break
            dyaw = abs((nxt[0].transform.rotation.yaw - cur.transform.rotation.yaw + 180.0) % 360.0 - 180.0)
            if dyaw > 25.0:
                break
            cur = nxt[0]
            dist += step
        if dist > best_chain_len:
            best_chain_len = dist
            best_start_wp = wp
            if dist >= required_length:
                break

    if best_start_wp is None:
        return project_frenet_offline(resampled, cfg, output_xodr=output_xodr, town_label=town)

    # Pre-sample centerline waypoints at 1.0 m resolution for fast, smooth projection
    centerline = [best_start_wp]
    cur = best_start_wp
    for _ in range(int(best_chain_len) + 5):
        nxt = cur.next(1.0)
        if not nxt:
            break
        cur = nxt[0]
        centerline.append(cur)

    def s_d_to_carla_pose(s_val: float, d_val: float, z_offset: float = 0.50):
        idx = int(np.clip(round(max(0.0, float(s_val))), 0, len(centerline) - 1))
        tf = centerline[idx].transform
        right_vec = tf.get_right_vector()
        d_clamped = float(np.clip(d_val, -cfg.road.lane_width_m * 2.0, cfg.road.lane_width_m * 2.0))
        target_loc = carla.Location(
            x=tf.location.x + right_vec.x * d_clamped,
            y=tf.location.y + right_vec.y * d_clamped,
            z=tf.location.z + 0.5,
        )
        road_wp = carla_map.get_waypoint(target_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        z_surf = road_wp.transform.location.z if road_wp is not None else tf.location.z
        return target_loc.x, target_loc.y, z_surf + z_offset, math.radians(tf.rotation.yaw)

    projected = dict(resampled)
    road_geom = estimate_road_geometry(resampled, cfg)
    projected["road_geometry"] = road_geom
    if output_xodr is not None and cfg.road.generate_xodr:
        write_opendrive_xodr(road_geom, output_xodr)
        projected["xodr_path"] = str(output_xodr)

    for ent in [projected["ego"]] + projected["actors"]:
        z_off = 0.85 if ent.get("entity_kind") == "pedestrian" else 0.50
        poses = []
        for s_val, d_val in zip(ent["s"], ent["d"]):
            x, y, z, h_rad = s_d_to_carla_pose(s_val, d_val, z_offset=z_off)
            poses.append({"x": x, "y": y, "z": z, "h": h_rad})
        _compute_smooth_headings(poses, times=ent.get("times"), is_oncoming=bool(ent.get("is_oncoming", False)))
        ent["world_poses"] = poses

    projected["town"] = town
    projected["projection_mode"] = "carla_map"
    return projected
