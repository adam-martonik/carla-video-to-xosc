"""
Export a finished CARLA run as a standard OpenSCENARIO file.

  motion_runner (live run)  --TrajectoryRecorder-->  <scene>_trajectories.json   (raw CARLA poses, every tick)
  python -m agents.xosc_export <out_dir> <scene_id>
        -> smooth each path (agents/traj_smoothing.py) -> <scene>.xosc  (agents/xosc_writer.py) next to <scene>.xodr

Frame: CARLA is left handed. The map is the SUMO net exported to OpenDRIVE (agents/map_builder.py) with
SUMO (x, y) = CARLA (x, -y) and SUMO heading = -CARLA yaw, so recorded poses are converted with carla_to_world().
"""
import argparse
import json
import math
import os

import numpy as np

from agents import traj_smoothing, xosc_writer

TRAJ_SUFFIX = "_trajectories.json"
PAIR_PREFIXES = ("vehicle", "walker")


def carla_to_world(x, y, yaw_deg):
    """CARLA (x, y, yaw in degrees) -> OpenDRIVE/SUMO frame (x, y, heading in radians)."""
    return x, -y, -math.radians(yaw_deg)


class TrajectoryRecorder:
    """Collects one pose per actor per tick. Duck-typed on the CARLA actor API so it can be tested without CARLA."""

    def __init__(self):
        self.samples = {}

    def record(self, t, actors, speed_overrides=None):
        overrides = speed_overrides or {}
        for aid, actor in actors.items():
            if not actor.is_alive:
                continue
            tf, vel = actor.get_transform(), actor.get_velocity()
            spd = float(overrides[aid]) if aid in overrides else math.sqrt(vel.x ** 2 + vel.y ** 2 + vel.z ** 2)
            self.samples.setdefault(aid, []).append([
                round(float(t), 3), round(tf.location.x, 3), round(tf.location.y, 3), round(tf.location.z, 3),
                round(tf.rotation.yaw, 2), round(spd, 3)])

    def to_dict(self, meta=None, dt=None):
        """meta: {actor_id: {"type": ..., "blueprint": ..., "color": ...}}"""
        meta = meta or {}
        return {"frame": "carla", "dt": dt,
                "actors": {aid: {**meta.get(aid, {}), "samples": s} for aid, s in self.samples.items()}}

    def save(self, path, meta=None, dt=None):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(meta, dt), f)
        return path


def crop_time(result, plan, hold_s=0.3):
    """
    Simulation time after which the recording is dropped: the first contact between two vehicles/walkers (the expected
    pair when the plan names one) plus hold_s. After an impact the motion is not drivable behaviour, so it is not exported.
    Returns None when nothing collided.
    """
    expect = (plan or {}).get("expect") or {}
    pair = set(expect.get("actors", [])) if len(expect.get("actors", [])) == 2 else None
    times = []
    for c in (result or {}).get("events", []):
        if not str(c.get("other", "")).startswith(PAIR_PREFIXES):
            continue
        if pair is not None and {c.get("actor"), c.get("other_actor")} != pair:
            continue
        times.append(float(c.get("t_sim", c.get("t", 0.0))))
    return (min(times) + hold_s) if times else None


def build_scenario(recording, plan, xodr_name=None, result=None, dt=0.1, hold_s=0.3, description=None):
    """Turn a recording into the dict that xosc_writer.write_openscenario_xosc expects. Also returns a per-actor report."""
    ego = (plan or {}).get("ego")
    t_cut = crop_time(result, plan, hold_s)
    entities, report = [], {}
    t_end = 0.0
    for aid, rec in sorted(recording["actors"].items()):
        s = np.asarray(rec["samples"], dtype=float)
        if t_cut is not None:
            s = s[s[:, 0] <= t_cut + 1e-9]
        if len(s) < 2:
            report[aid] = "skipped (fewer than 2 samples)"
            continue
        wx, wy, wh = zip(*(carla_to_world(r[1], r[2], r[4]) for r in s))
        wz = s[:, 3]
        vtype = rec.get("type", "car")
        sm = traj_smoothing.smooth_trajectory(s[:, 0], wx, wy, vtype, dt=dt, initial_heading=wh[0])
        z_out = np.interp(sm["times"], s[:, 0], wz)
        lim = traj_smoothing.limits_for(vtype)
        entities.append({
            "name": aid, "ego": aid == ego, "kind": "pedestrian" if vtype == "pedestrian" else "vehicle", "vtype": vtype,
            "blueprint": rec.get("blueprint"), "color": rec.get("color"),
            "times": sm["times"].tolist(), "x": sm["x"].tolist(), "y": sm["y"].tolist(), "z": z_out.tolist(),
            "h": sm["h"].tolist(), "v": sm["v"].tolist(),
            "limits": {"max_speed": lim["max_speed"], "max_accel": lim["max_accel"], "max_decel": lim["max_decel"]}})
        report[aid] = {"samples": int(len(s)), "max_deviation_m": round(sm["max_deviation_m"], 3)}
        t_end = max(t_end, float(sm["times"][-1]))
    if not entities:
        raise ValueError("the recording has no actor with at least 2 samples")
    if ego is None or not any(e["ego"] for e in entities):
        entities[0]["ego"] = True       # ScenarioRunner needs an actor called "hero"
    sc_out = {"scene_id": (plan or {}).get("scene_id"), "duration": t_end, "xodr": xodr_name, "entities": entities,
              "description": description or ("Recorded CARLA run; cropped at first impact" if t_cut is not None else "Recorded CARLA run")}
    if isinstance((plan or {}).get("environment"), dict):
        sc_out["environment"] = dict(plan["environment"])
    return sc_out, report


def export_scene(out_dir, scene_id, dt=0.1, hold_s=0.3, xsd_path=None):
    """Read the files of a finished run in out_dir and write <scene_id>.xosc there. Returns (xosc_path, report)."""
    def path(suffix):
        return os.path.join(out_dir, scene_id + suffix)

    traj_fn = path(TRAJ_SUFFIX)
    if not os.path.exists(traj_fn):
        raise FileNotFoundError("%s not found: run agents.motion_runner first (it records the trajectories)" % traj_fn)
    def load(fn, default):
        if not os.path.exists(fn):
            return default
        with open(fn, encoding="utf-8") as f:
            return json.load(f)

    recording = load(traj_fn, None)
    plan = load(path("_motion.json"), {})
    result = load(path("_motion_result.json"), None)
    xodr = scene_id + ".xodr" if os.path.exists(path(".xodr")) else None
    scenario, report = build_scenario(recording, plan, xodr, result, dt=dt, hold_s=hold_s)
    scenario["scene_id"] = scene_id
    out = xosc_writer.write_openscenario_xosc(scenario, path(".xosc"), xsd_path=xsd_path)
    return out, report


def main(argv=None):
    ap = argparse.ArgumentParser(description="Export a recorded CARLA run as OpenSCENARIO 1.0")
    ap.add_argument("out_dir")
    ap.add_argument("scene_id")
    ap.add_argument("--dt", type=float, default=0.1, help="trajectory sample step in seconds")
    ap.add_argument("--hold", type=float, default=0.3, help="seconds kept after the first impact")
    ap.add_argument("--xsd", default=None, help="validate against this OpenSCENARIO 1.0 schema")
    args = ap.parse_args(argv)
    out, report = export_scene(args.out_dir, args.scene_id, args.dt, args.hold, args.xsd)
    for aid, r in report.items():
        print("[xosc] %s: %s" % (aid, r))
    print("[xosc] wrote", out)


if __name__ == "__main__":
    main()
