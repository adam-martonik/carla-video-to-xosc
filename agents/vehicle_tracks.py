"""
Vehicle tracks over the whole simulated stretch of a dashcam video, in road coordinates.

  1. detect + track   YOLO26n (models/yolo/yolo26n.pt, AGPL-3.0, downloaded on first use from the official ultralytics
                      release) with ByteTrack, ~10 frames per second, frames at the analysis width of agents/lane_lines
  2. road position    the bottom centre of a box is where the vehicle touches the road; the flat-road camera
                      (agents/camera_geom) turns it into metres right (X) and ahead (Z) of the camera; the vehicle's centre is
                      half a vehicle length further along the line of sight. With the ego car's travel s_e(t):
                          s = s_e(t) + Z_centre   (along the road from the ego start)      d = -X   (+ = left of the ego lane)
  3. smoothing        agents/traj_smoothing (outlier filter, Kalman smoother, kinematic limits) per track
  4. kind             same direction / oncoming / parked / crossing from the smoothed motion over the ground

The crash vehicles are replayed by the scene builder's closed-loop controllers; every other track can be replayed along its
measured path and speed (agents/scene_builder).
"""
import math
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEIGHTS = os.path.join(ROOT, "models", "yolo", "yolo26n.pt")
WEIGHTS_URL = "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt"
COCO = {1: "bike", 2: "car", 3: "motorcycle", 5: "truck", 7: "truck"}           # bicycle, car, motorcycle, bus, truck
HALF_LEN = {"car": 2.3, "truck": 4.5, "motorcycle": 1.0, "bike": 0.9}
TRACK_FPS = 10.0
MIN_OBS = 4
SIDE_VIEW_AR = 1.9            # width / height of a box above which the vehicle is seen from the side (crossing)

_MODEL = {}


def _download_weights(url=WEIGHTS_URL, dest=WEIGHTS, timeout=60):
    """Download YOLO weights atomically to `dest` (creating parent directories)."""
    import urllib.request
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    tmp = dest + ".tmp"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "carla-video-to-xosc/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(tmp, "wb") as out:
            while True:
                chunk = resp.read(1 << 16)
                if not chunk:
                    break
                out.write(chunk)
        os.replace(tmp, dest)
    except Exception:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass
        raise
    return dest


def _yolo():
    if "m" not in _MODEL:
        from ultralytics import YOLO
        if not os.path.exists(WEIGHTS):
            if os.environ.get("AS_NO_DOWNLOAD") == "1":
                raise FileNotFoundError("YOLO weights missing (AS_NO_DOWNLOAD=1): %s" % WEIGHTS)
            print("[tracks] downloading %s -> %s" % (WEIGHTS_URL, WEIGHTS), flush=True)
            _download_weights(WEIGHTS_URL, WEIGHTS)
        _MODEL["m"] = YOLO(WEIGHTS)
    return _MODEL["m"]


def detect_tracks(path, t0, t1, width, fps_track=TRACK_FPS, conf=0.25):
    """Raw tracks {id: [{"t", "box": [x1, y1, x2, y2], "cls", "conf"}]} between video times t0 and t1, with boxes in a picture
    `width` px wide (the analysis width). Frames are read in order (no seeking) so that the tracker sees a steady frame rate."""
    from agents import lane_lines
    crop = lane_lines._video_crop(path)
    model = _yolo()
    cap = cv2.VideoCapture(path)
    tracks = {}
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        stride = max(int(round(fps / fps_track)), 1)
        i0 = int(max(t0, 0.0) * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, i0)
        i = i0
        reset = True
        while i <= t1 * fps:
            ok, fr = cap.read()
            if not ok:
                break
            if (i - i0) % stride == 0:
                if crop is not None:
                    fr = fr[crop[0]:crop[1], crop[2]:crop[3]]
                h, w = fr.shape[:2]
                fr = cv2.resize(fr, (width, int(round(h * width / w))))
                res = model.track(fr, persist=not reset, tracker="bytetrack.yaml", classes=list(COCO), conf=conf,
                                  agnostic_nms=True, verbose=False, imgsz=640)
                reset = False
                b = res[0].boxes
                if b is not None and b.id is not None:
                    for box, tid, c, p in zip(b.xyxy.numpy(), b.id.numpy().astype(int), b.cls.numpy().astype(int), b.conf.numpy()):
                        tracks.setdefault(int(tid), []).append({"t": round(i / fps, 3), "box": [round(float(v), 1) for v in box],
                                                                "cls": int(c), "conf": round(float(p), 3),
                                                                "color": box_color(fr, box)})
            i += 1
    finally:
        cap.release()
    return tracks


def box_color(frame, box):
    """Colour name of the vehicle in the box (median of its upper-middle part, nearest of scene_spec.COLORS), or None."""
    from agents.scene_spec import COLORS
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    w, h = x2 - x1, y2 - y1
    if w < 12 or h < 12:
        return None
    patch = frame[max(y1 + int(0.2 * h), 0):max(y1 + int(0.6 * h), 1), max(x1 + int(0.25 * w), 0):max(x1 + int(0.75 * w), 1)]
    if patch.size == 0:
        return None
    b, g, r = np.median(patch.reshape(-1, 3), axis=0)
    names = {k: v for k, v in COLORS.items() if k not in ("grey", "dark", "tan")}
    return min(names, key=lambda k: (names[k][0] - r) ** 2 + (names[k][1] - g) ** 2 + (names[k][2] - b) ** 2)


def ground_point(cam, box, vtype, y_bottom=None):
    """(X right, Z ahead) of the vehicle's centre on the road from its box, or None. The box's bottom centre is where it touches
    the road (a box cut off by the hood has no usable bottom); seen from behind or in front, the centre is half a vehicle length
    further along the line of sight, seen from the side (crossing) it is about a metre further."""
    x1, y1, x2, y2 = box
    if (x2 - x1) > 0.75 * cam.w and y2 >= 0.90 * cam.h:
        return None
    if y_bottom is not None and y2 >= y_bottom - 2:
        xc = 0.5 * (x1 + x2)
        in_center = 0.20 * cam.w <= xc <= 0.80 * cam.w
        if in_center or y2 >= 0.92 * cam.h or (x2 - x1) >= 0.45 * cam.w:
            return None
    g = cam.ground_from_pixel(0.5 * (x1 + x2), y2)
    if g is None:
        return None
    X, Z = g
    side = (x2 - x1) / max(y2 - y1, 1.0) >= SIDE_VIEW_AR
    extra = 0.9 if side else HALF_LEN.get(vtype, 2.3)
    k = (Z + extra) / Z
    return X * k, Z + extra


CAR_HEIGHT_M = 1.5          # typical passenger car (sedan 1.45, SUV 1.75): trucks vary too much to use


def elevation_ratio(cam, box, vtype, y_bottom=None, z_range=(8.0, 45.0)):
    """Distance from where the wheels touch the road (flat-ground assumption) divided by distance from the car's size in the
    picture (a car is about CAR_HEIGHT_M tall), or None. On the ego car's ground level the two agree (about 0.8 .. 1.2 with the
    spread of car heights); a car on a road that climbs above that level shows its wheels higher in the picture, so the
    ground distance grows like H / (H - rise) while its size does not: the ratio rises clearly above 1."""
    if vtype != "car":
        return None
    x1, y1, x2, y2 = box
    if (y_bottom is not None and y2 >= y_bottom - 2) or y2 - y1 < 8:
        return None
    g = cam.ground_from_pixel(0.5 * (x1 + x2), y2)
    if g is None:
        return None
    z_size = cam.f * CAR_HEIGHT_M / (y2 - y1)
    if not z_range[0] <= z_size <= z_range[1]:
        return None
    return g[1] / z_size


def classify(t, s, d, v_ego):
    """'same', 'oncoming', 'parked' or 'crossing' from a smoothed track (t, s along the road, d across it) and the ego speed
    at those times: the speed over the ground is ds/dt (s already includes the ego car's travel)."""
    if len(t) < 2:
        return "same"
    dur = max(t[-1] - t[0], 1e-3)
    vs = (s[-1] - s[0]) / dur
    vd = (d[-1] - d[0]) / dur
    if abs(vd) > 2.0 and abs(vd) > 0.6 * abs(vs):
        return "crossing"
    if abs(vs) < 1.5:
        return "parked"
    return "same" if vs > 0 else "oncoming"


def to_road_tracks(raw, cam, s_ego, y_bottom=None, min_obs=MIN_OBS, curve=None, vx_at=None):
    """Raw tracks -> [{"id", "type", "color", "kind", "t", "s", "z", "d", "v", "first_t", "last_t", "n_obs"}] in road
    coordinates (lists sampled every 0.1 s after smoothing; z = distance ahead of the ego car). s_ego(t): the ego car's travel since the start of the simulated stretch.
    The colour is the most common one of the five largest boxes. curve: agents/camera_geom.RoadCurve of a bending road; vx_at(t): the
    vanishing point's column at t (the camera's heading against the road changes on a bend)."""
    import copy
    cam = copy.copy(cam)
    from agents import traj_smoothing
    out = []
    for tid, obs in sorted(raw.items()):
        types = [COCO.get(o["cls"], "car") for o in obs]
        vtype = max(set(types), key=types.count)
        pts, ratios = [], []
        for o in obs:
            if vx_at is not None:
                cam.vx = vx_at(o["t"])
            r = elevation_ratio(cam, o["box"], vtype, y_bottom)
            if r is not None:
                ratios.append(r)
            g = ground_point(cam, o["box"], vtype, y_bottom)
            if g is not None and g[1] <= 75.0:
                se = s_ego(o["t"])
                off = float(curve.offset(se, g[1])) if curve is not None else 0.0
                pts.append((o["t"], se + g[1], -g[0] - off))
        if len(pts) < min_obs:
            continue
        ts, ss, ds = (np.array(v, dtype=float) for v in zip(*pts))
        try:
            sm = traj_smoothing.smooth_trajectory(ts, ss, ds, vtype=vtype if vtype in ("car", "truck", "motorcycle", "bike") else "car",
                                                  dt=0.1, meas_std=0.8, process_std=1.5)
            t_s, s_s, d_s, v_s = sm["times"], sm["x"], sm["y"], sm["v"]
        except Exception:
            continue
        cols = [o.get("color") for o in sorted(obs, key=lambda o_: -(o_["box"][2] - o_["box"][0]))[:5] if o.get("color")]
        color = max(set(cols), key=cols.count) if cols else None
        kind = classify(t_s, s_s, d_s, None)
        dur = float(t_s[-1] - t_s[0]) if len(t_s) >= 2 else 0.0
        v_med = float(np.median(np.abs(v_s))) if len(v_s) else 0.0
        if kind != "parked" and dur < 0.6 and v_med > 25.0:
            continue
        sign = -1.0 if kind == "oncoming" else 1.0
        out.append({"id": int(tid), "type": vtype, "color": color, "kind": kind,
                    "t": [round(float(x), 2) for x in t_s], "s": [round(float(x), 2) for x in s_s],
                    "z": [round(float(x - s_ego(tt)), 2) for x, tt in zip(s_s, t_s)],
                    "d": [round(float(x), 2) for x in d_s], "v": [round(float(x) * sign, 2) for x in v_s],
                    "first_t": round(float(t_s[0]), 2), "last_t": round(float(t_s[-1]), 2), "n_obs": len(pts),
                    "elev_ratio": round(float(np.median(ratios)), 3) if len(ratios) >= 3 else None})
    return out


def position_at(track, t):
    """(s, d) of a road track at time t (inside its time range), or None."""
    if not track["t"] or t < track["t"][0] - 1e-6 or t > track["t"][-1] + 1e-6:
        return None
    return float(np.interp(t, track["t"], track["s"])), float(np.interp(t, track["t"], track["d"]))


def crash_tracks(tracks, t_crash, ego_involved=True, window=(-2.0, 0.5), max_ego_m=15.0, max_pair_m=6.0, color=None):
    """Ids of the tracks that are the crash vehicles, so that they are not replayed twice: with the ego car involved, the track
    that comes closest to the ego car in the seconds before t_crash (from its distance ahead "z" and across "d"; a car that
    gets very close is cut off by the hood, so its last position can still be metres away), a track of the crash vehicle's
    colour first; in a crash between two other vehicles, the pair closest to each other at t_crash."""
    if ego_involved:
        ts = [t_crash + window[0] + 0.1 * i for i in range(int(round((window[1] - window[0]) / 0.1)) + 1)]
        cands = []
        for tr in tracks:
            dmin = min((math.hypot(z, d) for z, d in ego_relative(tr, ts)), default=None)
            if dmin is not None and dmin <= max_ego_m:
                cands.append((0 if color and tr.get("color") == color else 1, dmin, tr["id"]))
        return [min(cands)[2]] if cands else []
    best = None
    for i, a in enumerate(tracks):
        for b in tracks[i + 1:]:
            pa, pb = position_at(a, t_crash), position_at(b, t_crash)
            if pa is None or pb is None:
                continue
            dist = math.hypot(pa[0] - pb[0], pa[1] - pb[1])
            if dist <= max_pair_m and (best is None or dist < best[0]):
                best = (dist, [a["id"], b["id"]])
    return best[1] if best else []


def converging_cutin_track(tracks, t_crash, side="left", window=(-2.5, 2.5), color=None, lane_w=3.5):
    """Find the track actively cutting into the ego lane around `t_crash` (starting on `side` >= 0.4 lanes away and
    moving laterally toward d = 0), plus an optional occluded predecessor track slightly earlier on the same path."""
    sign = 1.0 if side != "right" else -1.0
    lo_t, hi_t = t_crash + window[0], t_crash + window[1]
    cands = []
    for tr in tracks:
        ts, ds, zs = tr.get("t") or [], tr.get("d") or [], tr.get("z") or []
        if len(ts) < 2 or not ds or not zs:
            continue
        if tr.get("last_t", ts[-1]) < lo_t or tr.get("first_t", ts[0]) > hi_t:
            continue
        i_pk = max(range(len(ds)), key=lambda i: ds[i] * sign)
        d_pk = ds[i_pk]
        if d_pk * sign < 0.4 * lane_w:
            continue
        d_inward = (d_pk - ds[-1]) * sign
        dur = max(ts[-1] - ts[i_pk], 0.1)
        min_z = min(zs)
        if d_inward >= 0.7 and d_inward / dur >= 0.8 and min_z <= 18.0 and abs(ds[-1]) <= 6.5:
            score = abs(ds[-1]) + 0.3 * min_z - 0.5 * d_inward
            cands.append((0 if (color and tr.get("color") == color) else 1, score, tr["id"], tr, d_pk))
    if not cands:
        return None
    _, _, _, tr, d_pk = min(cands)
    prev_cands = []
    for p in tracks:
        if p["id"] == tr["id"] or not p.get("d") or not p.get("z") or not p.get("t"):
            continue
        dt_gap = tr["t"][0] - p["t"][-1]
        if 0.05 <= dt_gap <= 1.8 and p["d"][0] * sign >= 0.4 * lane_w and p["d"][-1] * sign >= d_pk * sign - 0.5:
            d_gap = min(abs(p["d"][-1] - tr["d"][0]), abs(p["d"][-1] - d_pk))
            if d_gap <= 3.0 and abs(p["z"][-1] - tr["z"][0]) <= 12.0:
                col_pen = 0 if (tr.get("color") and p.get("color") == tr.get("color")) else 1
                prev_cands.append((col_pen, d_gap, p["id"], p))
    prev = min(prev_cands)[-1] if prev_cands else None
    skip_ids = ([prev["id"]] if prev else []) + [tr["id"]]
    d_start = max(abs(d_pk), abs(prev["d"][0]) if prev else 0.0)
    lane_offset = max(1, int(round(d_start / max(lane_w, 1.0))))
    v_tr = [abs(v) for v in (tr.get("v") or [])]
    v_prev = [abs(v) for v in (prev.get("v") or [])] if prev else []
    if len(v_prev) >= 4 and v_tr:
        v_meas = min(float(np.median(v_prev)), float(np.median(v_tr)))
    else:
        v_vals = v_tr + v_prev
        v_meas = float(np.median(v_vals)) if v_vals else 0.0
    t_impact = round(float(tr["t"][-1]) + 0.15, 2)
    # A cutter that disappears off the side of the forward dashcam FOV (|d_last| >= 1.8 m, right beside ego z <= 12 m)
    # strikes ego from the side (v2 front-corner into ego's side), rather than crossing in front of ego's bumper.
    side_strike = bool(abs(float(tr["d"][-1])) >= 1.8 and float(tr["z"][-1]) <= 12.0)
    return {"track": tr, "prev_track": prev, "skip_ids": skip_ids, "d_start": round(float(d_start), 2),
            "lane_offset": lane_offset, "v_meas": round(v_meas, 2), "t_impact": t_impact,
            "side_strike": side_strike}


def ego_relative(track, ts):
    """[(Z ahead of the ego car, d across)] of a track at the times it covers (needs track["z"] sampled like track["t"])."""
    out = []
    for t in ts:
        if track["t"] and track["t"][0] <= t <= track["t"][-1] and track.get("z"):
            out.append((float(np.interp(t, track["t"], track["z"])), float(np.interp(t, track["t"], track["d"]))))
    return out
