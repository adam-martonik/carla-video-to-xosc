"""
Scene specification: the small, fixed JSON that a (local, small) vision model fills in from a video.
Everything after this step is deterministic code (agents/scene_builder.py), so a weak model cannot break the
pipeline: values are validated, clamped and defaulted here.

{
 "road":  {"type": "straight|crossroad", "lanes_per_direction": 1-6, "traffic_signals": bool,
           "lane_profile": null or [{"d0": m|null, "d1": m|null, "n": 1-6, "m": lanes missing on the left}]},
                                          # lane_profile: lanes along the road (agents/lane_profile.py); d from the ego start
 "ego":   {"type": "car|motorcycle|truck", "lane": "right|left", "lane_index": null or 0-5 (from the left, see lane_profile),
           "speed_kmh": 0-130,
           "behavior": "cruise|slow_to_stop|stopped|turn_left|turn_right"},
 "actors": [{"id": "a1", "type": "car|suv|truck|van|motorcycle|bike|pedestrian",
             "relation": "ahead|behind|oncoming|cross_left|cross_right",
             "lane": "same|left|right",          # lane relative to the ego lane (ahead/behind), ignored for cross/oncoming
             "distance_m": 3-120, "speed_kmh": 0-130,
             "behavior": "stopped|driving|braking|cutting_in|turning_left|turning_right|crossing"}],
 "event": {"type": "collision|near_miss|none", "actor_id": "a1", "time_s": 0-30, "description": "..."}
}
"""
import copy

ROAD_TYPES = ("straight", "crossroad")
EGO_TYPES = ("car", "motorcycle", "truck")
ACTOR_TYPES = ("car", "suv", "truck", "van", "motorcycle", "bike", "pedestrian")
RELATIONS = ("ahead", "behind", "beside", "oncoming", "cross_left", "cross_right")
BEHAVIORS = ("stopped", "driving", "braking", "cutting_in", "turning_left", "turning_right", "crossing")
EGO_BEHAVIORS = ("cruise", "slow_to_stop", "stopped", "turn_left", "turn_right")
EVENTS = ("collision", "near_miss", "none")

EDGES = ("barrier", "shoulder", "open")
TRAFFIC_LANES = ("ego_lane_ahead", "left_lane", "right_lane", "oncoming")
MAX_TRAFFIC = 12
MAX_LANES = 6

DEFAULT = {
    "road": {"type": "straight", "lanes_per_direction": 2, "traffic_signals": False, "one_way": False,
             "edge_right": "shoulder", "edge_left": "shoulder", "centerline": None, "curve_note": None,
             "lane_profile": None},
    "traffic": {"ego_lane_ahead": 0, "left_lane": 0, "right_lane": 0, "oncoming": 0, "rel_speed_kmh": 0.0},
    "ego": {"type": "car", "lane": "right", "lane_index": None, "speed_kmh": 50.0, "behavior": "cruise"},
    "actors": [],
    "event": {"type": "none", "actor_id": None, "actor2_id": None, "time_s": 5.0, "description": "fallback scene"},
}

PROMPT_SCHEMA = """{
 "road": {"type": "straight" or "crossroad", "lanes_per_direction": 1 to 6, "traffic_signals": true or false},
 "ego": {"type": "car" or "motorcycle" or "truck", "lane": "right" or "left", "speed_kmh": number,
         "behavior": "cruise" or "slow_to_stop" or "stopped" or "turn_left" or "turn_right"},
 "actors": [ {"id": "a1", "type": "car" or "suv" or "truck" or "van" or "motorcycle" or "bike" or "pedestrian",
              "relation": "ahead" or "behind" or "beside" or "oncoming" or "cross_left" or "cross_right",
              "lane": "same" or "left" or "right", "color": "white" or "black" or "silver" or "gray" or "red" or "blue" or ...,
              "distance_m": number, "speed_kmh": number,
              "behavior": "stopped" or "driving" or "braking" or "cutting_in" or "turning_left" or "turning_right" or "crossing"} ],
 "event": {"type": "collision" or "near_miss" or "none", "actor_id": "id of the actor involved or null",
           "time_s": seconds from the start of the video, "description": "one sentence"}
}"""


COLORS = {"white": (235, 235, 235), "black": (25, 25, 25), "silver": (175, 175, 180), "gray": (125, 125, 130), "grey": (125, 125, 130),
          "red": (180, 30, 30), "blue": (40, 60, 160), "green": (40, 120, 60), "yellow": (225, 190, 40), "orange": (220, 110, 30),
          "brown": (110, 70, 40), "beige": (200, 180, 140), "tan": (200, 180, 140), "gold": (200, 170, 70),
          "maroon": (100, 30, 30), "dark": (40, 40, 45)}


def color_rgb(v):
    """'silver' or 'dark gray' or '120,120,125' -> 'r,g,b' string for the CARLA colour attribute, or None."""
    if v is None:
        return None
    t = str(v).lower()
    for name, rgb in COLORS.items():
        if name in t:
            return "%d,%d,%d" % rgb
    import re as _re
    m = _re.findall(r"\d+", t)
    if len(m) == 3 and all(0 <= int(x) <= 255 for x in m):
        return ",".join(m)
    return None


def _num(v, lo, hi, default):
    try:
        x = float(v)
    except (TypeError, ValueError):
        return default
    return max(lo, min(hi, x))


SYNONYMS = {"motorbike": "motorcycle", "scooter": "motorcycle", "moped": "motorcycle", "bicycle": "bike", "cyclist": "bike",
            "pickup": "truck", "pickup_truck": "truck", "lorry": "truck", "bus": "truck", "semi": "truck", "minivan": "van",
            "suv": "suv", "sedan": "car", "vehicle": "car", "pedestrian": "pedestrian", "person": "pedestrian",
            "walker": "pedestrian", "t-junction": "crossroad", "t_junction": "crossroad", "intersection": "crossroad",
            "junction": "crossroad", "t-intersection": "crossroad", "highway": "straight", "road": "straight",
            "left": "cross_left", "right": "cross_right", "front": "ahead", "opposite": "oncoming", "oncoming_traffic": "oncoming"}


def _pick(v, allowed, default):
    v = str(v).strip().lower().replace(" ", "_") if v is not None else ""
    v = SYNONYMS.get(v, v) if SYNONYMS.get(v, v) in allowed else v
    if v in allowed:
        return v
    for a in allowed:                     # tolerate near matches such as "t-junction", "pickup"
        if a in v or v in a and v:
            return a
    return default


def _bool(v):
    if isinstance(v, str):
        return v.strip().lower() in ("true", "yes", "1", "y")
    return bool(v)


def _centerline(v):
    """List of >= 2 finite [x, y] points, else None."""
    import math
    if not isinstance(v, (list, tuple)) or len(v) < 2:
        return None
    out = []
    for p in v:
        try:
            x, y = float(p[0]), float(p[1])
        except (TypeError, ValueError, IndexError):
            return None
        if not (math.isfinite(x) and math.isfinite(y)):
            return None
        out.append([x, y])
    return out


def _lane_profile(v):
    """Ordered, contiguous sections [{"d0", "d1", "n", "m"}] (first d0 and last d1 open), else None. A section may carry
    "branch": {"side": "left"|"right", "rise_m": metres}: the lanes that leave the ego road where it starts go on as a road of
    their own (a fork or flyover) instead of ending."""
    if not isinstance(v, (list, tuple)) or len(v) < 2:
        return None
    out = []
    for i, sc in enumerate(v):
        if not isinstance(sc, dict):
            return None
        try:
            n = int(round(float(sc.get("n"))))
            m = int(round(float(sc.get("m", 0) or 0)))
            d0 = None if i == 0 else float(sc.get("d0"))
            d1 = None if i == len(v) - 1 else float(sc.get("d1"))
        except (TypeError, ValueError):
            return None
        if not 1 <= n <= MAX_LANES or not 0 <= m < MAX_LANES or n + m > MAX_LANES:
            return None
        if out and (d0 is None or abs(d0 - out[-1]["d1"]) > 1e-6):
            return None
        if d0 is not None and d1 is not None and d1 <= d0:
            return None
        out.append({"d0": d0, "d1": d1, "n": n, "m": m})
        br = sc.get("branch")
        if i > 0 and isinstance(br, dict) and br.get("side") in ("left", "right") and not any("branch" in x for x in out):
            a = out[-2]
            leaves = (m > a["m"]) if br["side"] == "left" else (m + n < a["m"] + a["n"])
            if leaves:                       # only the first split is kept; it must be on the side where lanes leave
                out[-1]["branch"] = {"side": br["side"], "rise_m": _num(br.get("rise_m"), -20.0, 20.0, 0.0)}
    if min(s["m"] for s in out) != 0:
        return None
    return out


def _series(v, lo, hi, width=2):
    """[[t, value, ...], ...] with finite numbers, t increasing and the values clamped to lo..hi; None when unusable."""
    import math
    if not isinstance(v, (list, tuple)) or len(v) < 2:
        return None
    out = []
    for p in v:
        try:
            row = [float(x) for x in p[:width]]
        except (TypeError, ValueError, IndexError):
            return None
        if len(row) < width or not all(math.isfinite(x) for x in row) or (out and row[0] <= out[-1][0]):
            return None
        out.append([row[0]] + [max(lo, min(hi, x)) for x in row[1:]] if width == 2 else row)
    return out


REPLAY_KINDS = ("same", "oncoming", "parked", "crossing")
MAX_REPLAY = 30


def _replay(v):
    """Measured vehicles to replay: [{"id", "type", "color", "kind", "path": [[t, s, d], ...], "speed": [[t, km/h], ...]}]."""
    if not isinstance(v, (list, tuple)):
        return []
    out = []
    for r in v:
        if not isinstance(r, dict) or len(out) >= MAX_REPLAY:
            continue
        path = _series(r.get("path"), -1e6, 1e6, width=3)
        if not path:
            continue
        out.append({"id": str(r.get("id") or "r%d" % len(out)), "type": _pick(r.get("type"), ACTOR_TYPES, "car"),
                    "color": color_rgb(r.get("color")), "kind": r.get("kind") if r.get("kind") in REPLAY_KINDS else "same",
                    "path": path, "speed": _series(r.get("speed"), 0.0, 200.0) or []})
    return out


def _traffic(obj):
    t = dict(DEFAULT["traffic"])
    obj = obj if isinstance(obj, dict) else {}
    for k in TRAFFIC_LANES:
        t[k] = int(round(_num(obj.get(k), 0, MAX_TRAFFIC, 0)))
    while sum(t[k] for k in TRAFFIC_LANES) > MAX_TRAFFIC:       # trim the largest lane until the total fits
        big = max(TRAFFIC_LANES, key=lambda k: t[k])
        t[big] -= 1
    t["rel_speed_kmh"] = float(_num(obj.get("rel_speed_kmh"), -60.0, 60.0, 0.0))
    return t


def normalize(obj):
    """Return (clean_spec, error). Never raises; only returns an error when the object is unusable."""
    if not isinstance(obj, dict):
        return None, "reply is not a JSON object"
    spec = copy.deepcopy(DEFAULT)
    road = obj.get("road") or {}
    spec["road"]["type"] = _pick(road.get("type"), ROAD_TYPES,
                                 "crossroad" if "junction" in str(road.get("type", "")).lower() or
                                 "intersection" in str(road.get("type", "")).lower() else "straight")
    spec["road"]["lanes_per_direction"] = int(round(_num(road.get("lanes_per_direction"), 1, MAX_LANES, 2)))
    spec["road"]["traffic_signals"] = bool(road.get("traffic_signals", False))
    spec["road"]["one_way"] = _bool(road.get("one_way", False))
    for side in ("edge_right", "edge_left"):
        e = str(road.get(side, "")).strip().lower()
        spec["road"][side] = e if e in EDGES else "shoulder"
    spec["road"]["centerline"] = _centerline(road.get("centerline"))
    note = road.get("curve_note")
    spec["road"]["curve_note"] = str(note)[:200] if note else None
    spec["road"]["lane_profile"] = _lane_profile(road.get("lane_profile"))
    spec["traffic"] = _traffic(obj.get("traffic"))
    ego = obj.get("ego") or {}
    spec["ego"]["type"] = _pick(ego.get("type"), EGO_TYPES, "car")
    spec["ego"]["lane"] = "left" if str(ego.get("lane", "")).lower().startswith("l") else "right"
    li = ego.get("lane_index")
    spec["ego"]["lane_index"] = int(_num(li, 0, MAX_LANES - 1, 0)) if li is not None and str(li).strip() != "" else None
    spec["ego"]["speed_kmh"] = _num(ego.get("speed_kmh"), 0.0, 130.0, 50.0)
    spec["ego"]["behavior"] = _pick(ego.get("behavior"), EGO_BEHAVIORS, "cruise")
    spec["ego"]["color"] = color_rgb(ego.get("color"))
    spec["ego"]["speed_profile"] = _series(ego.get("speed_profile"), 0.0, 130.0)
    actors = []
    for i, a in enumerate(obj.get("actors") or []):
        if not isinstance(a, dict) or len(actors) >= 6:
            continue
        rel = _pick(a.get("relation"), RELATIONS, "ahead")
        actors.append({
            "id": str(a.get("id") or "a%d" % (i + 1)),
            "type": _pick(a.get("type"), ACTOR_TYPES, "car"),
            "relation": rel,
            "lane": _pick(a.get("lane"), ("same", "left", "right"), "same"),
            "distance_m": _num(a.get("distance_m"), 3.0, 120.0, 25.0),
            "speed_kmh": _num(a.get("speed_kmh"), 0.0, 130.0, 30.0),
            "behavior": _pick(a.get("behavior"), BEHAVIORS, "driving"),
            "color": color_rgb(a.get("color")),
        })
    spec["actors"] = actors
    spec["replay"] = _replay(obj.get("replay"))
    ev = obj.get("event") or {}
    spec["event"]["type"] = _pick(ev.get("type"), EVENTS, "none")
    ids = {a["id"] for a in actors}
    aid = ev.get("actor_id")
    if isinstance(aid, list):
        aid = aid[0] if aid else None
    spec["event"]["actor_id"] = str(aid) if aid is not None and str(aid) in ids else (actors[0]["id"] if actors and spec["event"]["type"] != "none" else None)
    a2 = ev.get("actor2_id")
    if isinstance(a2, list):
        a2 = a2[0] if a2 else None
    spec["event"]["actor2_id"] = str(a2) if a2 is not None and str(a2) in ids and str(a2) != str(spec["event"]["actor_id"]) else None
    spec["event"]["time_s"] = _num(ev.get("time_s"), 1.0, 30.0, 5.0)
    spec["event"]["description"] = str(ev.get("description", ""))[:200]
    if spec["event"]["type"] != "none" and spec["event"]["actor_id"] is None:
        spec["event"]["type"] = "none"      # an event needs an actor
    return spec, None


def check(obj):
    spec, err = normalize(obj)
    return spec, err
