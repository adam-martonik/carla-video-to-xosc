#!/usr/bin/env python
"""Quantitative audit of baseline video_to_xosc.py outputs and existing .xosc scenarios."""
import json
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import xmlschema

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import video_to_xosc as baseline


def analyze_xosc(xosc_path: Path, xsd: xmlschema.XMLSchema) -> dict:
    valid = xsd.is_valid(str(xosc_path))
    tree = ET.parse(xosc_path)
    root = tree.getroot()
    params = root.findall(".//ParameterDeclarations/ParameterDeclaration")
    entities = root.findall(".//Entities/ScenarioObject")

    entity_info = []
    for so in entities:
        name = so.attrib["name"]
        veh = so.find("Vehicle")
        ped = so.find("Pedestrian")
        if veh is not None:
            perf = veh.find("Performance")
            dim = veh.find("BoundingBox/Dimensions")
            entity_info.append({
                "name": name,
                "kind": "vehicle",
                "blueprint": veh.attrib["name"],
                "category": veh.attrib["vehicleCategory"],
                "length": float(dim.attrib["length"]),
                "width": float(dim.attrib["width"]),
                "height": float(dim.attrib["height"]),
                "max_acc": float(perf.attrib["maxAcceleration"]),
                "max_dec": float(perf.attrib["maxDeceleration"]),
            })
        elif ped is not None:
            dim = ped.find("BoundingBox/Dimensions")
            entity_info.append({
                "name": name,
                "kind": "pedestrian",
                "blueprint": ped.attrib["model"],
                "category": ped.attrib["pedestrianCategory"],
                "length": float(dim.attrib["length"]),
                "width": float(dim.attrib["width"]),
                "height": float(dim.attrib["height"]),
            })

    trajs = {}
    traj_metrics = {}
    for mg in root.findall(".//ManeuverGroup"):
        eref = mg.find(".//Actors/EntityRef")
        if eref is None:
            continue
        ename = eref.attrib["entityRef"]
        verts = mg.findall(".//Polyline/Vertex")
        if not verts:
            continue
        pts = []
        for v in verts:
            t = float(v.attrib["time"])
            wp = v.find("Position/WorldPosition")
            pts.append((t, float(wp.attrib["x"]), float(wp.attrib["y"]), float(wp.attrib["z"]), float(wp.attrib["h"])))
        trajs[ename] = pts
        ts = np.array([p[0] for p in pts])
        xs = np.array([p[1] for p in pts])
        ys = np.array([p[2] for p in pts])
        hs = np.array([p[4] for p in pts])
        dt = np.diff(ts)
        ds = np.hypot(np.diff(xs), np.diff(ys))
        vs = ds / np.maximum(dt, 1e-6)
        accs = np.diff(vs) / np.maximum(dt[1:], 1e-6) if len(vs) > 1 else np.array([0.0])
        jerks = np.diff(accs) / np.maximum(dt[2:], 1e-6) if len(accs) > 1 else np.array([0.0])
        dh = np.abs((np.diff(hs) + np.pi) % (2 * np.pi) - np.pi)
        yaw_rates = dh / np.maximum(dt, 1e-6)
        traj_metrics[ename] = {
            "n_verts": len(pts),
            "t_start": float(ts[0]),
            "t_end": float(ts[-1]),
            "v_min": float(vs.min()),
            "v_max": float(vs.max()),
            "max_abs_acc": float(np.max(np.abs(accs))),
            "max_abs_jerk": float(np.max(np.abs(jerks))),
            "rms_jerk": float(np.sqrt(np.mean(jerks ** 2))),
            "max_yaw_rate": float(np.max(yaw_rates)),
        }

    names = list(trajs.keys())
    min_dist = 999.0
    overlaps = 0
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            p1 = {round(p[0], 1): (p[1], p[2]) for p in trajs[names[i]]}
            p2 = {round(p[0], 1): (p[1], p[2]) for p in trajs[names[j]]}
            for t in set(p1.keys()) & set(p2.keys()):
                d = math.hypot(p1[t][0] - p2[t][0], p1[t][1] - p2[t][1])
                if d < min_dist:
                    min_dist = d
                if d < 3.5:
                    overlaps += 1

    return {
        "file": xosc_path.name,
        "xsd_valid": valid,
        "param_count": len(params),
        "entity_count": len(entities),
        "entities": entity_info,
        "trajectories": traj_metrics,
        "min_pairwise_dist_m": round(min_dist, 3) if min_dist < 999 else None,
        "overlap_frames_lt_3_5m": overlaps,
    }


def analyze_frenet_extraction(video_path: Path) -> dict:
    ann_out = ROOT / "reports" / "evaluation" / f"baseline_{video_path.stem}_ann.mp4"
    extracted = baseline.estimate_ego_motion_and_tracks(video_path, ann_out)
    resampled = baseline.smooth_and_resample_trajectories(extracted, sample_dt=1.0)
    if ann_out.exists():
        ann_out.unlink()

    def traj_stats(t_arr, s_arr, d_arr):
        t = np.asarray(t_arr, dtype=float)
        s = np.asarray(s_arr, dtype=float)
        d = np.asarray(d_arr, dtype=float)
        dt = np.diff(t)
        ds = np.diff(s)
        dd = np.diff(d)
        v = np.hypot(ds, dd) / np.maximum(dt, 1e-6)
        a = np.diff(v) / np.maximum(dt[1:], 1e-6) if len(v) > 1 else np.array([0.0])
        j = np.diff(a) / np.maximum(dt[2:], 1e-6) if len(a) > 1 else np.array([0.0])
        # heading from ds, dd
        h = np.arctan2(dd, np.maximum(ds, 1e-6))
        dh = np.abs((np.diff(h) + np.pi) % (2 * np.pi) - np.pi) if len(h) > 1 else np.array([0.0])
        yr = dh / np.maximum(dt[1:], 1e-6) if len(h) > 1 else np.array([0.0])
        return {
            "n_points": len(t),
            "v_min": round(float(v.min()), 2),
            "v_max": round(float(v.max()), 2),
            "max_abs_acc": round(float(np.max(np.abs(a))), 2),
            "max_abs_jerk": round(float(np.max(np.abs(j))), 2),
            "rms_jerk": round(float(np.sqrt(np.mean(j ** 2))), 2),
            "max_yaw_rate": round(float(np.max(yr)), 3),
            "d_min": round(float(d.min()), 2),
            "d_max": round(float(d.max()), 2),
        }

    # Check raw ego speed jitter before smoothing
    ego_raw = extracted["ego"]
    raw_v = np.array([r["speed"] for r in ego_raw])
    raw_t = np.array([r["time"] for r in ego_raw])
    raw_a = np.diff(raw_v) / np.maximum(np.diff(raw_t), 1e-6)

    # Check pairwise overlaps in Frenet space
    all_ents = [resampled["ego"]] + resampled["actors"]
    min_dist = 999.0
    overlaps = 0
    for i in range(len(all_ents)):
        for j in range(i + 1, len(all_ents)):
            e1, e2 = all_ents[i], all_ents[j]
            n = min(len(e1["s"]), len(e2["s"]))
            for k in range(n):
                dist = math.hypot(e1["s"][k] - e2["s"][k], e1["d"][k] - e2["d"][k])
                if dist < min_dist:
                    min_dist = dist
                if dist < 3.5:
                    overlaps += 1

    return {
        "video": video_path.name,
        "fps": round(extracted["fps"], 2),
        "duration_s": round(extracted["duration"], 2),
        "is_night": extracted["is_night"],
        "camera_cut_time": extracted["camera_cut_time"],
        "collision_event": extracted["collision_event"] is not None,
        "raw_tracks_selected": len(extracted["tracks"]),
        "raw_ego_max_abs_acc": round(float(np.max(np.abs(raw_a))), 2),
        "raw_ego_std_speed": round(float(np.std(raw_v)), 2),
        "ego_resampled": traj_stats(resampled["ego"]["times"], resampled["ego"]["s"], resampled["ego"]["d"]),
        "actors_resampled": {
            a["role_name"]: {
                "class_id": a.get("class_id"),
                "entity_kind": a.get("entity_kind"),
                "blueprint": a.get("blueprint"),
                **traj_stats(a["times"], a["s"], a["d"]),
            }
            for a in resampled["actors"]
        },
        "min_frenet_dist_m": round(min_dist, 2) if min_dist < 999 else None,
        "frenet_overlap_steps_lt_3_5m": overlaps,
    }


def main():
    xsd = xmlschema.XMLSchema(str(ROOT / "scenario_runner/srunner/openscenario/OpenSCENARIO.xsd"))
    xosc_results = [analyze_xosc(p, xsd) for p in sorted((ROOT / "demo").glob("*.xosc"))]

    videos = [
        ROOT / "demo" / "dashcam_input.mp4",
        ROOT / "demo" / "brake.mp4",
        ROOT / "demo" / "jelen.mp4",
        ROOT / "demo" / "stiahnute_video.mp4",
    ]
    video_results = [analyze_frenet_extraction(v) for v in videos if v.exists()]

    report = {"existing_xosc": xosc_results, "baseline_video_runs": video_results}
    out_path = ROOT / "reports" / "evaluation" / "baseline_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
