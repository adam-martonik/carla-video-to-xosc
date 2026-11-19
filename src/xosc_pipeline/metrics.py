"""Quantitative scenario realism & fidelity metrics (kinematics, OBB collisions, TTC, ADE/FDE)."""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from shapely.geometry import Polygon


def _make_obb(x: float, y: float, h: float, length: float, width: float) -> Polygon:
    """Create a 2D oriented bounding box (OBB) polygon centered at (x, y) with heading h (rad)."""
    hl = 0.5 * length
    hw = 0.5 * width
    c = math.cos(h)
    s = math.sin(h)
    corners = [
        (x + hl * c - hw * s, y + hl * s + hw * c),
        (x + hl * c + hw * s, y + hl * s - hw * c),
        (x - hl * c + hw * s, y - hl * s - hw * c),
        (x - hl * c - hw * s, y - hl * s + hw * c),
    ]
    return Polygon(corners)


def compute_kinematic_metrics(times: np.ndarray, xs: np.ndarray, ys: np.ndarray, hs: np.ndarray) -> Dict[str, float]:
    """Compute speed, acceleration, jerk, yaw-rate, and lateral acceleration metrics."""
    t = np.asarray(times, dtype=float)
    x = np.asarray(xs, dtype=float)
    y = np.asarray(ys, dtype=float)
    h = np.asarray(hs, dtype=float)
    if len(t) < 2:
        return {
            "n_verts": len(t),
            "v_min": 0.0,
            "v_max": 0.0,
            "max_abs_acc": 0.0,
            "max_abs_jerk": 0.0,
            "rms_jerk": 0.0,
            "max_yaw_rate": 0.0,
            "max_lat_acc": 0.0,
        }

    dt = np.maximum(np.diff(t), 1e-4)
    ds = np.hypot(np.diff(x), np.diff(y))
    v = ds / dt
    a = np.diff(v) / dt[1:] if len(v) > 1 else np.array([0.0])
    j = np.diff(a) / dt[2:] if len(a) > 1 else np.array([0.0])
    dh = np.abs((np.diff(h) + np.pi) % (2.0 * np.pi) - np.pi)
    yaw_rate = dh / dt
    lat_acc = v * yaw_rate

    return {
        "n_verts": int(len(t)),
        "v_min": round(float(np.min(v)), 3),
        "v_max": round(float(np.max(v)), 3),
        "max_abs_acc": round(float(np.max(np.abs(a))), 3),
        "max_abs_jerk": round(float(np.max(np.abs(j))), 3),
        "rms_jerk": round(float(np.sqrt(np.mean(j ** 2))), 3),
        "max_yaw_rate": round(float(np.max(yaw_rate)), 4),
        "max_lat_acc": round(float(np.max(np.abs(lat_acc))), 3),
    }


def evaluate_xosc_file(xosc_path: Path, xsd_path: Path | None = None) -> Dict[str, Any]:
    """Evaluate an OpenSCENARIO 1.0 (.xosc) file for schema validity, kinematics, OBB collisions, and TTC."""
    xsd_valid = None
    if xsd_path is not None and xsd_path.exists():
        import xmlschema

        xsd = xmlschema.XMLSchema(str(xsd_path))
        xsd_valid = bool(xsd.is_valid(str(xosc_path)))

    tree = ET.parse(xosc_path)
    root = tree.getroot()
    params = root.findall(".//ParameterDeclarations/ParameterDeclaration")

    dims: Dict[str, Tuple[float, float, str]] = {}
    perf_max_acc: Dict[str, float] = {}
    for so in root.findall(".//Entities/ScenarioObject"):
        name = so.attrib["name"]
        veh = so.find("Vehicle")
        ped = so.find("Pedestrian")
        if veh is not None:
            d_el = veh.find("BoundingBox/Dimensions")
            p_el = veh.find("Performance")
            dims[name] = (float(d_el.attrib["length"]), float(d_el.attrib["width"]), veh.attrib["vehicleCategory"])
            perf_max_acc[name] = float(p_el.attrib["maxAcceleration"])
        elif ped is not None:
            d_el = ped.find("BoundingBox/Dimensions")
            dims[name] = (float(d_el.attrib["length"]), float(d_el.attrib["width"]), ped.attrib["pedestrianCategory"])

    trajs: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    traj_metrics: Dict[str, Dict[str, float]] = {}

    for mg in root.findall(".//ManeuverGroup"):
        eref = mg.find(".//Actors/EntityRef")
        if eref is None:
            continue
        ename = eref.attrib["entityRef"]
        verts = mg.findall(".//Polyline/Vertex")
        if not verts:
            continue
        ts, xs, ys, hs = [], [], [], []
        for v in verts:
            ts.append(float(v.attrib["time"]))
            wp = v.find("Position/WorldPosition")
            xs.append(float(wp.attrib["x"]))
            ys.append(float(wp.attrib["y"]))
            hs.append(float(wp.attrib["h"]))
        t_arr, x_arr, y_arr, h_arr = np.array(ts), np.array(xs), np.array(ys), np.array(hs)
        trajs[ename] = (t_arr, x_arr, y_arr, h_arr)
        traj_metrics[ename] = compute_kinematic_metrics(t_arr, x_arr, y_arr, h_arr)

    # 2D OBB overlap & minimum TTC on common time grid
    names = list(trajs.keys())
    obb_collisions = 0
    min_center_dist = 999.0
    min_ttc = 999.0

    if len(names) >= 2:
        t_min = max(float(trajs[n][0][0]) for n in names)
        t_max = min(float(trajs[n][0][-1]) for n in names)
        if t_max > t_min:
            grid = np.arange(t_min, t_max + 1e-6, 0.25)
            interp_states = {}
            for n in names:
                t_a, x_a, y_a, h_a = trajs[n]
                xi = np.interp(grid, t_a, x_a)
                yi = np.interp(grid, t_a, y_a)
                hi = np.interp(grid, t_a, np.unwrap(h_a))
                vx = np.gradient(xi, grid)
                vy = np.gradient(yi, grid)
                interp_states[n] = (xi, yi, hi, vx, vy)

            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    n1, n2 = names[i], names[j]
                    l1, w1, _ = dims.get(n1, (4.5, 1.9, "car"))
                    l2, w2, _ = dims.get(n2, (4.5, 1.9, "car"))
                    x1, y1, h1, vx1, vy1 = interp_states[n1]
                    x2, y2, h2, vx2, vy2 = interp_states[n2]
                    for k in range(len(grid)):
                        dx = x2[k] - x1[k]
                        dy = y2[k] - y1[k]
                        dist = math.hypot(dx, dy)
                        if dist < min_center_dist:
                            min_center_dist = dist
                        if dist < 0.5 * (l1 + l2) + 2.0:
                            poly1 = _make_obb(x1[k], y1[k], h1[k], l1, w1)
                            poly2 = _make_obb(x2[k], y2[k], h2[k], l2, w2)
                            if poly1.intersects(poly2):
                                obb_collisions += 1
                        # Longitudinal closing rate for TTC (Hayward, 1972)
                        if dist > 0.5:
                            closing = -((dx * (vx2[k] - vx1[k]) + dy * (vy2[k] - vy1[k])) / dist)
                            if closing > 0.5:
                                gap = max(0.0, dist - 0.5 * (l1 + l2))
                                ttc = gap / closing
                                if ttc < min_ttc:
                                    min_ttc = ttc

    worst_acc = max((m["max_abs_acc"] for m in traj_metrics.values()), default=0.0)
    worst_jerk = max((m["max_abs_jerk"] for m in traj_metrics.values()), default=0.0)
    worst_yaw = max((m["max_yaw_rate"] for m in traj_metrics.values()), default=0.0)
    mean_rms_jerk = float(np.mean([m["rms_jerk"] for m in traj_metrics.values()])) if traj_metrics else 0.0

    return {
        "file": xosc_path.name,
        "xsd_valid": xsd_valid,
        "param_count": len(params),
        "entity_count": len(dims),
        "categories": {k: v[2] for k, v in dims.items()},
        "max_perf_acc_xml": max(perf_max_acc.values(), default=0.0),
        "worst_max_abs_acc": round(worst_acc, 3),
        "worst_max_abs_jerk": round(worst_jerk, 3),
        "mean_rms_jerk": round(mean_rms_jerk, 3),
        "worst_max_yaw_rate": round(worst_yaw, 4),
        "min_center_dist_m": round(min_center_dist, 3) if min_center_dist < 999 else None,
        "obb_collision_steps": obb_collisions,
        "min_ttc_s": round(min_ttc, 3) if min_ttc < 999 else None,
        "trajectories": traj_metrics,
    }


def compute_ade_fde(
    pred_xy: np.ndarray,
    ref_xy: np.ndarray,
) -> Tuple[float, float]:
    """Compute Average Displacement Error (ADE) and Final Displacement Error (FDE) in meters."""
    p = np.asarray(pred_xy, dtype=float)
    r = np.asarray(ref_xy, dtype=float)
    n = min(len(p), len(r))
    if n == 0:
        return 0.0, 0.0
    errs = np.hypot(p[:n, 0] - r[:n, 0], p[:n, 1] - r[:n, 1])
    return float(np.mean(errs)), float(errs[-1])
