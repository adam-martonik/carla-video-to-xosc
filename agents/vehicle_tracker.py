"""Grounded 2D-to-3D multi-frame vehicle detection and tracking using VLM 2D bounding boxes and camera geometry."""
import math
import re

from agents import llm_backend, video_events

DET_PROMPT = (
    "Locate up to 6 closest, clearly visible vehicles (car, suv, van, truck, pickup, bus, motorcycle) in this dashcam image. "
    "Ignore tiny distant dots on the horizon. "
    'Output ONLY a JSON list: [{"bbox_2d": [x1, y1, x2, y2], "label": "<color> <type>"}]'
)

OBJ_RE = re.compile(
    r'\{\s*"bbox_2d"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]'
    r'\s*,\s*"label"\s*:\s*"([^"]+)"\s*\}'
)

CROP_TYPE_Q = (
    "What type of vehicle is shown in this cropped dashcam image?\n"
    "A) car (sedan or hatchback)\nB) SUV or crossover\nC) pickup truck\nD) van or minivan\n"
    "E) truck or bus\nF) motorcycle\nG) bicycle"
)
CROP_TYPE_KEYS = {"A": "car", "B": "suv", "C": "truck", "D": "van", "E": "truck", "F": "motorcycle", "G": "bike"}
CROP_COLOR_Q = (
    "What colour is the vehicle shown in this cropped dashcam image?\n"
    "A) white\nB) black\nC) silver\nD) gray\nE) red\nF) blue\nG) brown\nH) green"
)
CROP_COLOR_KEYS = {"A": "white", "B": "black", "C": "silver", "D": "gray", "E": "red", "F": "blue", "G": "brown", "H": "green"}

_COLORS = ("white", "black", "silver", "gray", "grey", "red", "blue", "green", "brown", "tan", "beige", "gold", "maroon", "yellow", "orange")
_TYPES = (
    ("motorcycle", "motorcycle"), ("motorbike", "motorcycle"),
    ("bicycle", "bike"), ("bike", "bike"),
    ("minivan", "van"), ("van", "van"),
    ("pickup", "truck"), ("truck", "truck"), ("semi", "truck"), ("lorry", "truck"), ("bus", "truck"),
    ("suv", "suv"), ("crossover", "suv"), ("jeep", "suv"),
    ("sedan", "car"), ("hatchback", "car"), ("coupe", "car"), ("taxi", "car"), ("police", "car"), ("car", "car"),
)


def parse_vehicle_label(label):
    """Map free-text label like 'brown van' or 'white police car' -> (color, canonical_type)."""
    low = " " + str(label or "").lower() + " "
    color = None
    for c in _COLORS:
        if re.search(r'\b' + re.escape(c) + r'\b', low):
            color = "gray" if c == "grey" else c
            break
    vtype = "car"
    for tok, canon in _TYPES:
        if re.search(r'\b' + re.escape(tok) + r'\b', low):
            vtype = canon
            break
    return color, vtype


def _iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / max(ua, 1e-6)


def _h_gap(a, b):
    if a[2] < b[0]:
        return b[0] - a[2]
    if b[2] < a[0]:
        return a[0] - b[2]
    return 0.0


def _v_overlap(a, b):
    return max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def project_box_3d(box, w, h, vtype):
    """Project 2D bounding box [x1, y1, x2, y2] in cropped active image -> (X_right_m, Z_ahead_m)."""
    x1, y1, x2, y2 = box
    bh = max(y2 - y1, 12.0)
    f = 0.85 * w
    yh = 0.45 * h
    hv = 1.85 if vtype in ("suv", "van") else (2.5 if vtype == "truck" else 1.55)
    zh = f * hv / bh
    zg = f * 1.35 / max(y2 - yh, 0.04 * h)
    z = max(3.0, min(70.0, 0.55 * zh + 0.45 * zg))
    xc = 0.5 * (x1 + x2)
    x = ((xc - 0.5 * w) / f) * z
    return round(x, 2), round(z, 2)


def _detect_with_image(video, t, width=640):
    import cv2
    import numpy as np

    raws = video_events.grab_frames(video, [t], width=width)
    if not raws:
        return [], (width, 360), None
    arr = np.frombuffer(raws[0], np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_COLOR) if arr.size else None
    if im is None:
        return [], (width, 360), None
    g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    y0, y1 = video_events.active_row_bounds(g)
    im = im[y0:y1, :]
    h, w = im.shape[:2]
    jpg = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
    txt = llm_backend.chat(DET_PROMPT, images=[jpg], json_mode=False, max_tokens=180, temperature=0.0)
    dets = []
    for m in OBJ_RE.finditer(txt):
        x1, ya, x2, yb = map(float, m.groups()[:4])
        if max(x2, yb) > max(w, h) * 1.15:
            x1, x2 = x1 * w / 1000.0, x2 * w / 1000.0
            ya, yb = ya * h / 1000.0, yb * h / 1000.0
        x1, x2 = max(0.0, min(float(w), x1)), max(0.0, min(float(w), x2))
        ya, yb = max(0.0, min(float(h), ya)), max(0.0, min(float(h), yb))
        bw, bh = x2 - x1, yb - ya
        if bw < 22.0 or bh < 18.0 or bw > 0.95 * w or bh > 0.95 * h:
            continue
        box = [round(x1, 1), round(ya, 1), round(x2, 1), round(yb, 1)]
        if any(_iou(box, d["box"]) > 0.55 for d in dets):
            continue
        color, vtype = parse_vehicle_label(m.group(5))
        x_m, z_m = project_box_3d(box, w, h, vtype)
        dets.append({
            "t": float(t),
            "box": box,
            "label": m.group(5).strip(),
            "type": vtype,
            "color": color,
            "X": x_m,
            "Z": z_m,
            "ar": round(bw / bh, 2),
        })
    return dets, (w, h), im


def detect_frame_vehicles(video, t, width=640):
    """Detect and project vehicles in one frame at time t. Returns (detections, (w, h))."""
    dets, size, _ = _detect_with_image(video, t, width=width)
    return dets, size


def classify_crop(im, box):
    """Zoomed VLM classification on a vehicle crop to refine (type, color)."""
    if im is None or box is None:
        return None, None
    import cv2

    h, w = im.shape[:2]
    bw, bh = box[2] - box[0], box[3] - box[1]
    pad_x, pad_y = 0.15 * bw, 0.15 * bh
    x1 = max(0, int(math.floor(box[0] - pad_x)))
    y1 = max(0, int(math.floor(box[1] - pad_y)))
    x2 = min(w, int(math.ceil(box[2] + pad_x)))
    y2 = min(h, int(math.ceil(box[3] + pad_y)))
    if x2 - x1 < 16 or y2 - y1 < 16:
        return None, None
    jpg = cv2.imencode(".jpg", im[y1:y2, x1:x2], [cv2.IMWRITE_JPEG_QUALITY, 90])[1].tobytes()
    vtype, color = None, None
    try:
        pt = llm_backend.choice_probabilities(CROP_TYPE_Q, [jpg], "ABCDEFG")
        vtype = CROP_TYPE_KEYS[max(pt, key=pt.get)]
    except Exception:
        pass
    try:
        pc = llm_backend.choice_probabilities(CROP_COLOR_Q, [jpg], "ABCDEFGH")
        color = CROP_COLOR_KEYS[max(pc, key=pc.get)]
    except Exception:
        pass
    return vtype, color


def _mode_vote(items, weights=None):
    counts = {}
    for idx, val in enumerate(items):
        if val:
            w = weights[idx] if weights else 1.0
            counts[val] = counts.get(val, 0.0) + w
    return max(counts, key=counts.get) if counts else None


def _median(vals):
    s = sorted(vals)
    return s[len(s) // 2] if s else 0.0


def build_tracks(frame_dets, frame_w):
    """Link per-frame detections [(t, dets), ...] into persistent multi-frame vehicle tracks."""
    tracks = []
    for t, dets in sorted(frame_dets, key=lambda kv: kv[0]):
        used_tracks = set()
        for d in dets:
            best_i, best_cost = None, 1e9
            for ti, tr in enumerate(tracks):
                if ti in used_tracks:
                    continue
                prev = tr[-1]
                if t - prev["t"] > 1.6 or t <= prev["t"]:
                    continue
                ov = _iou(prev["box"], d["box"])
                close_3d = (
                    abs(d["Z"] - prev["Z"]) <= 8.5
                    and abs(d["X"] - prev["X"]) <= 4.0
                    and abs(d["box"][3] - prev["box"][3]) <= 0.16 * frame_w
                    and ((d["X"] * prev["X"] >= -1.5) or ov > 0.05)
                )
                if ov > 0.40 or close_3d:
                    cost = (1.0 - ov) * 4.0 + abs(d["Z"] - prev["Z"]) * 1.2 + abs(d["X"] - prev["X"])
                    if cost < best_cost:
                        best_cost, best_i = cost, ti
            if best_i is not None:
                tracks[best_i].append(d)
                used_tracks.add(best_i)
            else:
                tracks.append([d])

    out = []
    for tid, obs in enumerate(tracks, start=1):
        xs = [o["X"] for o in obs]
        zs = [o["Z"] for o in obs]
        ars = [o["ar"] for o in obs]
        d_x = obs[-1]["X"] - obs[0]["X"]
        d_z = obs[-1]["Z"] - obs[0]["Z"]
        max_ar = max(ars)
        min_z = min(zs)
        max_bw = max(o["box"][2] - o["box"][0] for o in obs)
        weights = [min(o["ar"], 3.0) for o in obs]
        c_dir = None
        if len(obs) >= 2 and min_z <= 15.0 and max_ar >= 1.8:
            if d_x >= 2.0:
                c_dir = "left"
            elif d_x <= -2.0:
                c_dir = "right"
        out.append({
            "id": tid,
            "type": _mode_vote([o["type"] for o in obs], weights) or "car",
            "color": _mode_vote([o["color"] for o in obs], weights),
            "n_frames": len(obs),
            "first_t": obs[0]["t"],
            "last_t": obs[-1]["t"],
            "min_Z": round(min_z, 2),
            "med_Z": round(_median(zs), 2),
            "med_X": round(_median(xs), 2),
            "first_X": obs[0]["X"],
            "last_X": obs[-1]["X"],
            "dX": round(d_x, 2),
            "dZ": round(d_z, 2),
            "max_ar": round(max_ar, 2),
            "max_box_w": round(max_bw, 1),
            "crossing_dir": c_dir,
            "stationary": bool(len(obs) >= 2 and abs(d_x) < 1.0 and abs(d_z) < 2.5),
            "obs": obs,
        })
    return out


def analyze_tracks(tracks, frame_w, t_crash, roads=None, ego_motion=None, edges=None,
                   yellow_left=0.0, ego_speed_kmh=50.0, crop_classifier=None):
    """Fuse grounded vehicle tracks with road probes and ego motion into structured scene evidence."""
    tc = float(t_crash)
    motion = ego_motion or {}
    camera_steady = bool(motion.get("camera_steady", False))
    ego_stopped = bool(motion.get("ego_stopped", False))
    lanes = int((roads or {}).get("lanes", 2))
    road_type = "crossroad" if (roads or {}).get("intersection", 0.0) >= 0.55 else "straight"
    one_way = bool((edges or {}).get("one_way", False))

    def _active_near(tr, lo_dt, hi_dt):
        return any(tc + lo_dt <= o["t"] <= tc + hi_dt for o in tr["obs"])

    cross_cands = [tr for tr in tracks if tr["crossing_dir"] is not None and _active_near(tr, -2.2, 0.8)]
    if cross_cands:
        c1_tr = min(cross_cands, key=lambda tr: (tr["min_Z"], -tr["n_frames"]))
    else:
        fg_cands = [tr for tr in tracks if (not ego_stopped or not tr["stationary"])
                    and tr["min_Z"] <= 13.0 and _active_near(tr, -1.2, 0.8)]
        c1_tr = min(fg_cands, key=lambda tr: (tr["min_Z"], -tr["n_frames"])) if fg_cands else None

    c2_tr = None
    ego_involved = True
    second_vehicle = None
    if camera_steady and c1_tr is not None:
        best_c2 = None
        for tr in tracks:
            if tr is c1_tr or tr["stationary"] or tr["min_Z"] > 15.0:
                continue
            for o2 in tr["obs"]:
                if o2["t"] < tc - 0.3:
                    continue
                for o1 in c1_tr["obs"]:
                    if abs(o1["t"] - o2["t"]) > 0.1:
                        continue
                    gap = _h_gap(o1["box"], o2["box"])
                    vov = _v_overlap(o1["box"], o2["box"])
                    z_diff = abs(o1["Z"] - o2["Z"])
                    if gap <= 0.08 * frame_w and vov > 0.0 and z_diff <= 5.0:
                        score = (gap, z_diff, o2["Z"])
                        if best_c2 is None or score < best_c2[0]:
                            best_c2 = (score, tr, o2)
        if best_c2 is not None:
            _, c2_tr, o2_hit = best_c2
            ego_involved = False
            c2_type, c2_color = c2_tr["type"], c2_tr["color"]
            if crop_classifier is not None:
                rt, rc = crop_classifier(c2_tr, o2_hit)
                c2_type = rt or c2_type
                c2_color = rc or c2_color
            c2_x = o2_hit["X"]
            if c2_x >= 2.0:
                c2_app, c2_spd = "right_lane_straight", "fast"
            elif c2_x <= -2.0:
                c2_app, c2_spd = "left_lane_straight", "fast"
            else:
                c2_app = "crossing_from_right" if c1_tr["crossing_dir"] == "left" else "crossing_from_left"
                c2_spd = "slow"
            second_vehicle = {
                "type": c2_type,
                "color": c2_color,
                "approach": c2_app,
                "speed": c2_spd,
                "distance_m": round(max(o2_hit["Z"], 5.0), 1),
            }

    first_vehicle = None
    if c1_tr is not None:
        app = None
        if c1_tr["crossing_dir"] == "left":
            app = "crossing_from_left"
        elif c1_tr["crossing_dir"] == "right":
            app = "crossing_from_right"
        elif c1_tr["first_X"] <= -2.0:
            app = "merging_from_left" if road_type == "straight" else "crossing_from_left"
        elif c1_tr["first_X"] >= 2.0:
            app = "merging_from_right" if road_type == "straight" else "crossing_from_right"
        first_vehicle = {
            "type": c1_tr["type"],
            "color": c1_tr["color"],
            "approach": app,
            "speed": "slow",
            "distance_m": round(max(c1_tr["min_Z"], 6.0), 1),
        }

    # Infer ego_lane ("left" vs "right")
    if lanes <= 1:
        ego_lane = "right"
    elif not ego_involved and second_vehicle and second_vehicle["approach"] == "right_lane_straight":
        ego_lane = "left"
    elif not ego_involved and second_vehicle and second_vehicle["approach"] == "left_lane_straight":
        ego_lane = "right"
    elif not one_way and yellow_left >= 0.14:
        ego_lane = "left"
    elif (edges or {}).get("edge_right") == "barrier":
        ego_lane = "right"
    else:
        parallel = [tr for tr in tracks if tr is not c1_tr and tr is not c2_tr
                    and tr["first_t"] <= tc - 0.5 and tr["max_ar"] < 1.8
                    and tr["crossing_dir"] is None and 5.0 <= tr["med_Z"] <= 55.0]
        n_left = sum(1 for tr in parallel if -5.5 <= tr["med_X"] < -1.8)
        n_right = sum(1 for tr in parallel if 1.8 < tr["med_X"] <= 5.5)
        ego_lane = "left" if n_right > n_left else "right"

    # Grounded non-crash background actors sorted by track strength & proximity
    cands = []
    for tr in tracks:
        if tr is c1_tr or tr is c2_tr:
            continue
        if not camera_steady and tr["first_t"] > tc + 0.1:
            continue
        if not (5.0 <= tr["med_Z"] <= 55.0):
            continue
        if tr["n_frames"] < 2 and not (tr["first_t"] <= tc - 0.5 and tr["max_box_w"] >= 35.0):
            continue
        x, z, ar = tr["med_X"], tr["med_Z"], tr["max_ar"]
        actor = None
        if road_type == "crossroad" and x >= 4.5 and ar >= 1.8:
            actor = {"relation": "cross_right", "lane": "same", "distance_m": round(max(z, 12.0), 1),
                     "behavior": "stopped", "speed_kmh": 0.0}
        elif road_type == "crossroad" and x <= -4.5 and ar >= 1.8:
            actor = {"relation": "cross_left", "lane": "same", "distance_m": round(max(z, 12.0), 1),
                     "behavior": "stopped", "speed_kmh": 0.0}
        elif -1.8 <= x <= 1.8:
            stopped = ego_stopped and tr["stationary"]
            actor = {"relation": "ahead", "lane": "same", "distance_m": round(max(z, 8.0), 1),
                     "behavior": "stopped" if stopped else "driving",
                     "speed_kmh": 0.0 if stopped else max(float(ego_speed_kmh or 50.0), 35.0)}
        elif -5.5 <= x < -1.8 and ego_lane == "right" and lanes >= 2:
            stopped = (road_type == "crossroad") or ego_stopped
            actor = {"relation": "ahead", "lane": "left", "distance_m": round(max(z, 8.0), 1),
                     "behavior": "stopped" if stopped else "driving",
                     "speed_kmh": 0.0 if stopped else max(float(ego_speed_kmh or 50.0) - 15.0, 20.0)}
        elif 1.8 < x <= 5.5 and ego_lane == "left" and lanes >= 2:
            stopped = (road_type == "crossroad") or ego_stopped
            actor = {"relation": "ahead", "lane": "right", "distance_m": round(max(z, 8.0), 1),
                     "behavior": "stopped" if stopped else "driving",
                     "speed_kmh": 0.0 if stopped else max(float(ego_speed_kmh or 50.0) - 15.0, 20.0)}
        elif not one_way and ((ego_lane == "right" and x < -5.5) or (ego_lane == "left" and x < -1.8)):
            if tr["n_frames"] >= 2 or (road_type == "crossroad" and abs(x) <= 8.0):
                stopped = (road_type == "crossroad")
                actor = {"relation": "oncoming", "lane": "same", "distance_m": round(max(z, 20.0), 1),
                         "behavior": "stopped" if stopped else "driving",
                         "speed_kmh": 0.0 if stopped else 40.0}
        if actor is not None:
            actor["type"] = tr["type"]
            actor["color"] = tr["color"]
            cands.append(((-tr["n_frames"], actor["distance_m"]), actor))

    cands.sort(key=lambda kv: kv[0])
    bg = []
    by_group = {}
    for _, a in cands:
        key = (a["relation"], a["lane"])
        prev_ds = by_group.get(key, [])
        if len(prev_ds) >= 3 or any(abs(a["distance_m"] - d0) < 6.0 for d0 in prev_ds):
            continue
        prev_ds.append(a["distance_m"])
        by_group[key] = prev_ds
        bg.append(a)
        if len(bg) >= 6:
            break

    return {
        "ego_involved": ego_involved,
        "ego_lane": ego_lane,
        "first_vehicle": first_vehicle,
        "second_vehicle": second_vehicle,
        "background_actors": bg,
        "tracks": [{k: v for k, v in tr.items() if k != "obs"} for tr in tracks],
    }


def analyze_video(video, t_crash, duration, roads=None, ego_motion=None, edges=None,
                  yellow_left=0.0, ego_speed_kmh=50.0):
    """Run grounded multi-frame 3D vehicle detection and fusion over the crash approach window."""
    if t_crash is None:
        return None
    tc = float(t_crash)
    dur = float(duration)
    times = sorted({round(min(max(tc - 2.2 + 0.7 * i, 0.3), max(dur - 0.1, 0.3)), 2) for i in range(5)})
    frame_dets, images = [], {}
    frame_w = 640
    for t in times:
        dets, (w, _), im = _detect_with_image(video, t, width=640)
        frame_w = w or frame_w
        frame_dets.append((t, dets))
        if im is not None:
            images[t] = im
    tracks = build_tracks(frame_dets, frame_w)

    def _crop_clf(tr, obs):
        return classify_crop(images.get(obs["t"]), obs["box"])

    return analyze_tracks(
        tracks, frame_w, tc, roads=roads, ego_motion=ego_motion, edges=edges,
        yellow_left=yellow_left, ego_speed_kmh=ego_speed_kmh, crop_classifier=_crop_clf,
    )
