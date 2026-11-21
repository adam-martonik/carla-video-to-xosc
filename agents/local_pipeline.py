"""
Unattended pipeline: dashcam video -> CARLA scenario that is checked and adjusted until it is ready.
No API key and no cost: video analysis uses OpenCV, the language steps use a local vision model (llama.cpp).

  python -m agents.local_pipeline <video> --out <folder>             # phase 1 (no CARLA) and phase 2 (CARLA)
  python -m agents.local_pipeline <video> --out <folder> --no-sim    # phase 1 only (CARLA can stay closed)
  python -m agents.local_pipeline --sim-only --out <folder>          # phase 2 only for an existing scenario
  python -m agents.local_pipeline <video> --out <folder> --spec spec.json   # skip the model, use a given spec

Phase 1 steps
  1. ego motion (stopped phases, video cuts) from optical flow. No jolt/shake detection.
  2. the model describes the WHOLE video at --fps frames per second (default 6): the road, the vehicles with type, colour,
     lane and distance, and a speed read from the video overlay when one is printed.
  3. the crash moment is the strongest model probability for 'a vehicle cuts in / vehicles touch' over sliding 4-frame
     windows (log-probabilities of YES/NO, not the model's hard answers); about 16 frames around it are analysed together: who is involved, their type and colour, and how they approach
     (merging from the left/right lane, crossing, oncoming, ahead, from behind)
  4. one text step turns the timeline into the scene description (validated, retried, safe fallback)
  5. deterministic builder: road network, object layout, motion plan, expectation
Folder result: video_events.json, frame_descriptions.json, crash_window.json, <scene>_spec.json, .net.xml,
_scene_obj.txt, _motion.json, _motion_result.json, _ready.json, report.txt
"""
import argparse
import json
import os
import re
import sys
import time

SCENE_ID = "video_interpreter_0000_split"


def log(msg):
    print("[pipeline] " + msg, flush=True)


def _save(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------------------- timeline captions
CAPTION_PROMPT = (
    "One frame from a dashcam video. The camera is mounted in the ego vehicle (the car that carries the camera). "
    "Answer in ONE short sentence: the road (number of lanes in the ego vehicle's direction, which lane the ego vehicle "
    "is in, and mention an intersection only if traffic lights or a cross-street are clearly visible) and every vehicle you "
    "see with its type (car, SUV, pickup, van, truck, motorcycle, bus), colour, position (ahead in my lane, in the left lane, "
    "in the right lane, oncoming, crossing) and distance (very close under 3 m, near, far). "
    "If a speed is printed in the video overlay, finish with SPEED=<number><MPH or KMH>. Only state what you can see.")

LABEL_RE = re.compile(r"^\W*(CONTACT|CLOSE|NONE)\b[\s:\-\.]*", re.I)
SPEED_RE = re.compile(r"(?:the\s+speed\s+is\s+|speed\s*[:=]\s*)?(\d{1,3})\s*(MPH|KMH|KM/H)\b|SPEED\s*=\s*(\d{1,3})\s*(MPH|KMH|KM/H)?", re.I)


def parse_caption(t, text):
    m = LABEL_RE.match(text)
    label = m.group(1).upper() if m else "NONE"
    body = text[m.end():] if m else text
    sp = SPEED_RE.search(text)
    speed = None
    if sp:
        num = sp.group(1) or sp.group(3)
        unit = (sp.group(2) or sp.group(4) or "MPH").upper()
        speed = float(num) * (1.609 if unit == "MPH" else 1.0)
    return {"t": t, "label": label, "text": SPEED_RE.sub("", body).strip(), "speed_kmh": speed}


def caption_timeline(video, duration, fps, width=640, log_fn=log):
    """One labelled description per sampled frame over the whole video."""
    from agents import llm_backend, video_events
    n = max(int(duration * fps), 4)
    times = [round(duration * (i + 0.5) / n, 2) for i in range(n)]
    caps = []
    t0 = time.time()
    for i, t in enumerate(times):
        fr = video_events.grab_frames(video, [t], width=width)
        if not fr:
            continue
        try:
            c = llm_backend.chat(CAPTION_PROMPT, images=fr, json_mode=False, max_tokens=90, temperature=0.1).strip()
        except Exception as e:
            c = "NONE: (description failed: %s)" % e
        caps.append(parse_caption(t, " ".join(c.split())))
        if (i + 1) % 20 == 0 or i + 1 == len(times):
            log_fn("described %d/%d frames (%.0f s)" % (i + 1, len(times), time.time() - t0))
    return caps


# ----------------------------------------------------------------------------------------- crash moment
SCORE_PREFIX = ("These 4 consecutive frames (0.75 seconds in total) are from a dashcam video. The camera is in the ego vehicle "
                "(the car carrying the camera). ")
SCORE_QUESTIONS = (
    SCORE_PREFIX + "Does a vehicle from a neighbouring lane move INTO the ego vehicle's lane or toward the ego vehicle across these "
                   "frames, getting so close that it touches the ego vehicle or almost does? Answer YES or NO.",
    SCORE_PREFIX + "Do two vehicles touch each other (a collision) in these frames? Answer YES or NO.")


def window_scores(video, t, duration, width=448):
    """[cut-in probability, contact probability] for the 4-frame window around t, from the model's YES/NO log-probabilities."""
    from agents import llm_backend, video_events
    fr = video_events.grab_frames(video, [min(max(t - 0.375 + 0.25 * i, 0.0), duration - 0.1) for i in range(4)], width=width)
    try:
        return [llm_backend.yes_probability(q, fr) for q in SCORE_QUESTIONS]
    except Exception:
        return [0.0, 0.0]


def crash_scores(video, duration, step=0.5, width=448, log_fn=log):
    """Probability of a cut-in/contact for a sliding 4-frame window, from the model's YES/NO log-probabilities."""
    out = []
    t = 0.5
    t0 = time.time()
    while t < duration - 0.4:
        ps = window_scores(video, t, duration, width)
        out.append({"t": round(t, 2), "cutin": round(ps[0], 3), "contact": round(ps[1], 3), "score": round(sum(ps) / len(ps), 3)})
        t += step
    log_fn("crash scores for %d windows (%.0f s)" % (len(out), time.time() - t0))
    return out


def pick_crash_time(scores, duration):
    """Strongest smoothed window. Returns (time, how it was found)."""
    if not scores:
        return float(0.7 * duration), "no scores: assumed at 70% of the video"
    sm = []
    for i in range(len(scores)):
        w = [scores[j]["score"] for j in range(max(i - 1, 0), min(i + 2, len(scores)))]
        sm.append(sum(w) / len(w))
    i = max(range(len(sm)), key=lambda k: sm[k])
    return float(scores[i]["t"]), "strongest cut-in/contact score %.2f" % sm[i]


CTX_NOTE = ("The images are consecutive frames from a dashcam video. The camera is mounted in the ego vehicle (the car that carries "
            "the camera). Something dangerous happens in this moment: a vehicle comes very close to the ego vehicle or hits it. ")

APPROACH_Q = (CTX_NOTE + "Look at the vehicle that comes closest to the ego vehicle or hits it. Where does it come from?\n"
              "A) it drives in the same direction in the lane to the LEFT of the ego vehicle and moves into the ego vehicle's lane\n"
              "B) it drives in the same direction in the lane to the RIGHT of the ego vehicle and moves into the ego vehicle's lane\n"
              "C) it is in front of the ego vehicle in the same lane\n"
              "D) it crosses the road in front of the ego vehicle from the left (at an intersection)\n"
              "E) it crosses the road in front of the ego vehicle from the right (at an intersection)\n"
              "F) it comes toward the ego vehicle on the opposite side of the road\n"
              "G) it comes from behind")
APPROACH_KEYS = {"A": "merging_from_left", "B": "merging_from_right", "C": "ahead", "D": "crossing_from_left", "E": "crossing_from_right",
                 "F": "oncoming", "G": "from_behind"}
TYPE_Q = (CTX_NOTE + "What type of vehicle is the one that comes closest to the ego vehicle or hits it?\n"
          "A) car (sedan or hatchback)\nB) SUV or crossover\nC) pickup truck\nD) van or minivan\nE) truck or bus\nF) motorcycle\nG) bicycle\nH) pedestrian")
TYPE_KEYS = {"A": "car", "B": "suv", "C": "truck", "D": "van", "E": "truck", "F": "motorcycle", "G": "bike", "H": "pedestrian"}
COLOR_Q = (CTX_NOTE + "What colour is the vehicle that comes closest to the ego vehicle or hits it?\n"
           "A) white\nB) black\nC) silver\nD) gray\nE) red\nF) blue\nG) green\nH) another colour")
COLOR_KEYS = {"A": "white", "B": "black", "C": "silver", "D": "gray", "E": "red", "F": "blue", "G": "green"}
LANES_Q = ("A frame from a dashcam video (the camera is in the ego vehicle). How many lanes does the road have in the ego vehicle's "
           "own driving direction (not counting the opposite direction)?\nA) 1\nB) 2\nC) 3\nD) 4\nE) 5\nF) 6 or more")
CROSS_Q = ("A frame from a dashcam video (the camera is in the ego vehicle). Is there an intersection where another road crosses "
           "the ego vehicle's road within about 30 meters ahead, with traffic lights or a crossing street? Answer YES or NO.")


def road_scores(video, duration, n=10, t0=None, t1=None, t_crash=None, log_fn=log):
    """Number of lanes and intersection yes/no from several frames (averaged probabilities)."""
    from agents import llm_backend, video_events
    hi = float(duration) if t1 is None else float(t1)
    if t0 is None and t_crash is not None:
        cand_lo = max(0.0, float(t_crash) - 4.5)
        lo = cand_lo if hi - cand_lo >= 2.0 else 0.0
    else:
        lo = 0.0 if t0 is None else float(t0)
    times = [round(lo + (hi - lo) * (i + 0.5) / n, 2) for i in range(n)]
    lanes = {k: 0.0 for k in "ABCDEF"}
    cross = 0.0
    used = 0
    for t in times:
        fr = video_events.grab_frames(video, [t], width=512)
        if not fr:
            continue
        try:
            pl = llm_backend.choice_probabilities(LANES_Q, fr, "ABCDEF")
            cr = llm_backend.yes_probability(CROSS_Q, fr)
            for k in lanes:
                lanes[k] += pl[k]
            cross += cr
            used += 1
        except Exception:
            pass
    if used:
        lanes = {k: v / used for k, v in lanes.items()}
        cross /= used
    n_lanes = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}[max(lanes, key=lanes.get)]
    return {"lanes": n_lanes, "lane_probs": {k: round(v, 2) for k, v in lanes.items()}, "intersection": round(cross, 2)}


FRAME_PREFIX = "A frame from a dashcam video. The camera is in the ego vehicle (the car that carries the camera). "
POS_Q = (FRAME_PREFIX + "Which vehicle is closest to the ego vehicle, and where is it?\n"
         "A) directly ahead in the ego vehicle's own lane, driving the same way\n"
         "B) on the LEFT side (in the left lane or entering/crossing from the left)\n"
         "C) on the RIGHT side (in the right lane or entering/crossing from the right)\n"
         "D) on the opposite side of the road, driving head-on toward the ego vehicle\n"
         "E) behind the ego vehicle")
POS_KEYS = {"A": "ahead", "B": "left", "C": "right", "D": "oncoming", "E": "from_behind"}
TYPE_Q1 = (FRAME_PREFIX + "What type of vehicle is the closest vehicle to the ego vehicle?\n"
           "A) car (sedan or hatchback)\nB) SUV or crossover\nC) pickup truck\nD) van or minivan\n"
           "E) truck or bus\nF) motorcycle\nG) bicycle\nH) pedestrian")
COLOR_Q1 = (FRAME_PREFIX + "What colour is the closest vehicle to the ego vehicle?\n"
            "A) white\nB) black\nC) silver\nD) gray\nE) red\nF) blue\nG) green\nH) another colour")
MANEUVER_Q = (FRAME_PREFIX + "Look at the closest vehicle ahead of the ego vehicle. Which view of that vehicle is visible, and how is it moving?\n"
              "A) Its full SIDE profile (side doors and both front and rear wheels visible), turned sideways and crossing ACROSS the road from one side to the other\n"
              "B) Its REAR end (taillights and rear bumper visible), facing forward in the same direction as the road and driving along or changing lanes\n"
              "C) Its FRONT end (headlights and grille visible), facing head-on toward the ego vehicle")
MANEUVER_KEYS = {"A": "crossing", "B": "merging", "C": "oncoming"}


def _weighted_avg(dicts, weights):
    tot = sum(weights) or 1.0
    keys = dicts[0].keys()
    return {k: sum(w * d[k] for w, d in zip(weights, dicts)) / tot for k in keys}


def _avg(dicts):
    return _weighted_avg(dicts, [1.0] * len(dicts))


def analyse_crash_window(video, t_crash, duration, roads=None, fps=3.0, span=1.2, width=640, log_fn=log):
    """Single-frame multiple-choice questions over the pre-impact approach window, weighted toward contact onset."""
    from agents import llm_backend, video_events
    t0 = max(t_crash - span, 0.0)
    t1 = min(max(t_crash - 0.15, t0 + 0.4), duration - 0.05)
    t_peak = max(t_crash - 0.35, t0)
    n = max(int((t1 - t0) * fps) + 1, 3)
    times = [round(t0 + (t1 - t0) * i / (n - 1), 2) for i in range(n)]
    pp, pt, pc, pm, ws = [], [], [], [], []
    for t in times:
        fr = video_events.grab_frames(video, [t], width=width)
        if not fr:
            continue
        try:
            pp.append(llm_backend.choice_probabilities(POS_Q, fr, "ABCDE"))
            pt.append(llm_backend.choice_probabilities(TYPE_Q1, fr, "ABCDEFGH"))
            pc.append(llm_backend.choice_probabilities(COLOR_Q1, fr, "ABCDEFGH"))
            pm.append(llm_backend.choice_probabilities(MANEUVER_Q, fr, "ABC"))
            ws.append(1.0 / (1.0 + ((t - t_peak) / 0.45) ** 2))
        except Exception as e:
            return None, "constrained questions failed: %s" % e, times
    if not pp:
        return None, "no frames could be read", times
    pm_list = list(pm)
    pm = _weighted_avg(pm_list, ws)
    max_side = max((m.get("A", 0.0) for m in pm_list), default=0.0)
    sideways_cross = bool(pm.get("A", 0.0) >= 0.30 or max_side >= 0.50)
    ws_attr = [w * (1.0 + 4.0 * m.get("A", 0.0)) for w, m in zip(ws, pm_list)] if sideways_cross else ws
    pp, pt, pc = _weighted_avg(pp, ws), _weighted_avg(pt, ws_attr), _weighted_avg(pc, ws_attr)
    l2r, r2l = video_events.lateral_entry_flow(video, t_crash)
    pp_eff = dict(pp)
    pp_eff["E"] = 0.5 * pp.get("E", 0.0)
    p_lat = pp_eff["B"] + pp_eff["C"]
    if p_lat >= max(pp_eff["A"], pp_eff["D"], pp_eff["E"]):
        if l2r > 1.8 * r2l and l2r > 25.0:
            pos = "left"
        elif r2l > 1.8 * l2r and r2l > 25.0:
            pos = "right"
        else:
            pos = "left" if pp_eff["B"] >= pp_eff["C"] else "right"
    else:
        pos = POS_KEYS[max(("A", "D", "E"), key=lambda k: pp_eff[k])]
    crossing = bool((roads and roads.get("intersection", 0) >= 0.55) or sideways_cross)
    approach = {"ahead": "ahead", "oncoming": "oncoming", "from_behind": "from_behind",
                "left": "crossing_from_left" if crossing else "merging_from_left",
                "right": "crossing_from_right" if crossing else "merging_from_right"}[pos]
    typ = TYPE_KEYS[max(pt, key=pt.get)]
    col = COLOR_KEYS.get(max(pc, key=pc.get))
    word = {"merging_from_left": "changes from the left lane into the lane of", "merging_from_right": "changes from the right lane into the lane of"}
    crash = {"collision": True, "ego_involved": True,
             "first_vehicle": {"type": typ, "color": col, "approach": approach, "speed": "slow"}, "second_vehicle": None,
             "probabilities": {"position": {POS_KEYS[k]: round(v, 2) for k, v in pp.items()},
                               "type": {TYPE_KEYS[k]: round(v, 2) for k, v in pt.items()},
                               "color": {COLOR_KEYS.get(k, "other"): round(v, 2) for k, v in pc.items()},
                               "maneuver": {MANEUVER_KEYS[k]: round(v, 2) for k, v in pm.items()},
                               "entry_flow": {"left_to_right": round(l2r, 1), "right_to_left": round(r2l, 1)}},
             "description": "a %s %s %s the ego vehicle (%s)" % (col or "", typ, word.get(approach, "meets"), approach)}
    return crash, None, times


# approach -> (relation, lane, behaviour)
APPROACH = {"merging_from_left": ("beside", "left", "cutting_in"), "merging_from_right": ("beside", "right", "cutting_in"),
            "crossing_from_left": ("cross_left", "same", "crossing"), "crossing_from_right": ("cross_right", "same", "crossing"),
            "right_lane_straight": ("behind", "right", "driving"), "left_lane_straight": ("behind", "left", "driving"),
            "oncoming": ("oncoming", "same", "driving"), "ahead": ("ahead", "same", "braking"), "from_behind": ("behind", "same", "driving")}


def _approach(v, road_type):
    a = str(v.get("approach", "")).lower().replace(" ", "_").replace("-", "_")
    if a in APPROACH:
        return APPROACH[a]
    if a in ("from_left", "left"):                 # ambiguous wording: merging on a plain road, crossing at an intersection
        return APPROACH["merging_from_left"] if road_type == "straight" else APPROACH["crossing_from_left"]
    if a in ("from_right", "right"):
        return APPROACH["merging_from_right"] if road_type == "straight" else APPROACH["crossing_from_right"]
    return APPROACH["merging_from_left"] if road_type == "straight" else APPROACH["crossing_from_left"]


def apply_crash(spec, crash, t_crash, ego_speed=None, tracked=None):
    """Put the analysed crash into the scene description."""
    from agents.scene_spec import _pick, ACTOR_TYPES, color_rgb

    def mk(i, v):
        v = v if isinstance(v, dict) else {}
        rel, lane, beh = _approach(v, spec["road"]["type"])
        speed = {"stopped": 0.0, "slow": 20.0, "fast": 50.0}.get(str(v.get("speed", "slow")).lower(), 25.0)
        if speed == 0.0 and beh in ("crossing", "cutting_in"):
            beh, speed = "driving", 20.0
        if speed == 0.0:
            beh = "stopped"
        typ = {"pickup": "truck"}.get(str(v.get("type", "")).lower(), v.get("type"))
        return {"id": "c%d" % i, "type": _pick(typ, ACTOR_TYPES, "car"), "relation": rel, "lane": lane,
                "distance_m": float(v.get("distance_m") or 15.0),
                "speed_kmh": speed, "behavior": beh, "color": color_rgb(v.get("color"))}

    second_v = crash.get("second_vehicle")
    ego_inv = bool(crash.get("ego_involved", True)) and not isinstance(second_v, dict)   # two vehicles named = third-party crash
    collision = bool(crash.get("collision", True))
    first = mk(1, crash.get("first_vehicle"))
    actors = [first]
    event = {"type": "collision" if collision else "near_miss", "actor_id": "c1", "actor2_id": None, "time_s": float(t_crash),
             "description": str(crash.get("description", ""))[:160]}
    if tracked and tracked.get("ego_involved") is False and isinstance(tracked.get("second_vehicle"), dict):
        v1 = dict(crash.get("first_vehicle") or {})
        v1.update({k: v for k, v in (tracked.get("first_vehicle") or {}).items() if v is not None})
        first = mk(1, v1)
        second = mk(2, tracked["second_vehicle"])
        actors = [first, second]
        ego_inv = False
        event["actor2_id"] = "c2"
    elif not ego_inv:
        second = mk(2, second_v if isinstance(second_v, dict) else {"type": "car", "approach": "crossing_from_right", "speed": "slow"})
        actors.append(second)
        event["actor2_id"] = "c2"
    elif first["relation"] == "ahead":
        first["distance_m"] = float((crash.get("first_vehicle") or {}).get("distance_m") or 25.0)
    if tracked and tracked.get("ego_lane") in ("left", "right") and spec["road"]["lanes_per_direction"] >= 2:
        spec["ego"]["lane"] = tracked["ego_lane"]
    if first["relation"] == "beside" and ego_inv:       # the ego lane is the one the vehicle merges into; the road needs two lanes
        spec["ego"]["lane"] = "right" if first["lane"] == "left" else "left"
        spec["road"]["lanes_per_direction"] = max(2, spec["road"]["lanes_per_direction"])
    try:
        lanes = int(float(crash.get("lanes_in_ego_direction")))
        if 1 <= lanes <= 3:
            spec["road"]["lanes_per_direction"] = max(lanes, spec["road"]["lanes_per_direction"] if first["relation"] == "beside" else lanes)
    except (TypeError, ValueError):
        pass
    if ego_speed and spec["ego"]["behavior"] != "stopped":
        spec["ego"]["speed_kmh"] = float(ego_speed)
    if spec["ego"]["behavior"] == "stopped":
        spec["ego"]["speed_kmh"] = 0.0
    if first["behavior"] == "cutting_in" and spec["ego"]["speed_kmh"] > 20.0:
        first["speed_kmh"] = round(float(spec["ego"]["speed_kmh"]), 1)
    elif first["behavior"] == "braking" and spec["ego"]["speed_kmh"] > 20.0:
        first["speed_kmh"] = round(max(float(spec["ego"]["speed_kmh"]) * 0.6, 20.0), 1)
    reconcile_traffic(spec)
    # keep either the grounded tracked vehicles or the model-described actors, without colliding on top of crash actors
    bg_tracked = (tracked or {}).get("background_actors") or []
    if bg_tracked:
        keep = [{"id": "b%d" % idx_b, "type": _pick(ba.get("type"), ACTOR_TYPES, "car"),
                 "relation": ba["relation"], "lane": ba.get("lane", "same"),
                 "distance_m": float(ba.get("distance_m", 20.0)), "speed_kmh": float(ba.get("speed_kmh", 0.0)),
                 "behavior": ba.get("behavior", "stopped"), "color": color_rgb(ba.get("color"))}
                for idx_b, ba in enumerate(bg_tracked, start=1)]
        if isinstance(spec.get("traffic"), dict):
            for k_tr in ("ego_lane_ahead", "left_lane", "right_lane", "oncoming"):
                spec["traffic"][k_tr] = 0
    else:
        keep = [a for a in spec["actors"] if a["relation"] in ("ahead", "behind", "oncoming") and not a["id"].startswith("c")][:4]
        traffic = spec.get("traffic") or {}
        if sum(int(traffic.get(k, 0) or 0) for k in ("ego_lane_ahead", "left_lane", "right_lane", "oncoming")) > 0:
            keep = []                                    # the traffic measured per lane covers the other vehicles
    for a in keep:
        if bg_tracked:
            if not any(a["relation"] == x["relation"] and a["lane"] == x["lane"] and abs(a["distance_m"] - x["distance_m"]) < 6.0 for x in actors):
                actors.append(a)
        elif all(not (a["relation"] == x["relation"] and a["lane"] == x["lane"] and a["relation"] != "beside") for x in actors if x["relation"] != "beside"):
            actors.append(a)
        elif a["relation"] == "ahead" and a["lane"] != first.get("lane"):
            actors.append(a)
    spec["actors"] = actors[:8] if bg_tracked else actors[:7]
    spec["event"] = event
    return reconcile_traffic(spec)


# ----------------------------------------------------------------------------------------- scene description
def condensed(caps, t_focus=None, limit=70):
    """Caption lines for the text step: all lines within 2 s of the crash moment plus an even spread of the rest."""
    keep = {i for i, c in enumerate(caps) if t_focus is not None and abs(c["t"] - t_focus) <= 2.0}
    rest = [i for i in range(len(caps)) if i not in keep]
    room = max(limit - len(keep), 12)
    step = max(len(rest) // room, 1)
    keep |= set(rest[::step][:room])
    lines, last = [], None
    for i in sorted(keep):
        c = caps[i]
        if c["text"] != last:
            lines.append("t=%.1fs: %s" % (c["t"], c["text"]))
        last = c["text"]
    return lines


def build_prompt2(caps, duration, t_focus=None):
    from agents.scene_spec import PROMPT_SCHEMA
    return (
        "A dashcam video of %.0f seconds was described frame by frame (the camera is in the ego vehicle):\n%s\n"
        "Using only these descriptions, produce ONE JSON object with exactly this structure and nothing else:\n%s\n"
        "Rules: use only the allowed words. 'actors' are the other road users that matter, at most 5, each listed once. "
        "relation beside = a vehicle driving next to the ego vehicle in the neighbouring lane; ahead/behind = in front of/behind it; "
        "cross_left/cross_right = a vehicle crossing in front of the ego vehicle at an intersection. "
        "Use road type crossroad only if a crossing street or traffic lights are described. "
        "'ego.lane' = the lane of the ego vehicle (right = right-most lane). 'event' can be left as none." %
        (duration, "\n".join(condensed(caps, t_focus=t_focus)), PROMPT_SCHEMA))


NEGATIONS = ("no visible traffic light", "no traffic light", "without traffic light", "no intersection", "not an intersection",
             "no visible intersection", "no crossing street")


def _positive_mentions(low, words):
    n = 0
    for c in low:
        for neg in NEGATIONS:
            c = c.replace(neg, " ")
        if any(w in c for w in words):
            n += 1
    return n


def ego_speed_near(caps, t, window=4.0):
    if len(caps) >= 10:
        pos_count = sum(1 for c in caps if (c.get("speed_kmh") or 0) > 0)
        if pos_count < 0.4 * len(caps):
            return None
    near = [c["speed_kmh"] for c in caps if c.get("speed_kmh") is not None and abs(c["t"] - t) <= window]
    if near:
        near.sort()
        return near[len(near) // 2]
    allv = sorted(c["speed_kmh"] for c in caps if c.get("speed_kmh") is not None)
    return allv[len(allv) // 2] if allv else None


def speed_series(caps, ts, window=1.5):
    """Ego speed in m/s at the given times from the overlay speeds in the captions (None where no overlay speed exists)."""
    out = []
    for t in ts:
        v = ego_speed_near(caps, t, window)
        out.append(None if v is None else v / 3.6)
    return out


def add_curve(spec, video, duration, t_crash, caps, log_fn=log):
    """Road centreline from the ego yaw rate over the stretch the scene shows (it starts T seconds before the crash)."""
    from agents import road_curve, road_probes
    tc = float(t_crash) if t_crash is not None else 6.0
    T = min(max(tc, 3.0), 8.0)
    t0, t1 = max(tc - T, 0.0), min(tc + 3.0, duration)
    est = road_curve.estimate_ego_yaw(video, t0, t1)
    if not est["t"]:
        spec["road"]["curve_note"] = "no yaw estimate: straight road"
        return spec
    sp_est = speed_series(caps, est["t"])
    v_est = max([v for v in sp_est if v is not None] or [float(spec["ego"]["speed_kmh"]) / 3.6, 14.0])
    c = road_curve.centerline_from_yaw(est["t"], est["yaw_rate"], sp_est,
                                       length_m=max(550.0, v_est * (T + 8.0) + 250.0))
    c["confidence"] = round(c["confidence"] * est["confidence"], 3)
    c = road_curve.reconcile_with_probe(c, road_probes.curve_direction(video, duration, t0=t0, t1=t1))
    if c.get("fallback"):
        spec["road"]["centerline"], spec["road"]["curve_note"] = None, c["fallback"]
    elif abs(c["points"][-1][1]) < 2.0:
        spec["road"]["centerline"], spec["road"]["curve_note"] = None, "straight road (sideways deviation %.1f m over %.0f m)" % (
            abs(c["points"][-1][1]), c["points"][-1][0])
    else:
        spec["road"]["centerline"] = c["points"]
        spec["road"]["curve_note"] = "curve from the ego yaw rate: confidence %.2f, smallest radius %.0f m, bends %s" % (
            c["confidence"], c["radius_min_m"] or 0.0, "left" if c["points"][-1][1] > 0 else "right")
    log_fn(spec["road"]["curve_note"])
    return spec


def sim_window(spec):
    """(video time at which the simulation starts, simulated event time T): the scene builder starts the ego car T seconds before
    the event, with T = event time clamped to 3..8 s (6 s and the video start without an event)."""
    ev = spec.get("event") or {}
    if ev.get("type") in ("collision", "near_miss") and ev.get("actor_id"):
        t_ev = float(ev.get("time_s", 5.0))
        T = min(max(t_ev, 3.0), 8.0)
        return max(t_ev - T, 0.0), T
    return 0.0, 6.0


def vehicle_lane_evidence(video, t0, t1, samples, step=1.0):
    """[{"t", "left", "right"}]: lanes beside the ego lane that vehicles stand in, one frame every `step` seconds (vehicles
    from the local model's boxes, agents/vehicle_tracker; only cars on the ego carriageway count, agents/lane_lines). A queue of
    cars hides its lanes' road surface, so the cars are the evidence that those lanes are there. [] without the model."""
    import cv2
    from agents import lane_lines, vehicle_tracker
    out = []
    try:
        y_b, _ = lane_lines.road_bottom(video)
        t = float(t0) + 0.5 * step
        while t <= float(t1):
            dets, (w, h), im = vehicle_tracker._detect_with_image(video, t, width=640)
            near = min(samples, key=lambda r: abs(r["t"] - t), default=None)
            vp_y = (near or {}).get("vp", [None, None])[1] if near and near.get("vp") else None
            frame = cv2.resize(im, (lane_lines.WIDTH, int(round(h * lane_lines.WIDTH / float(w))))) if im is not None else None
            ev = {"t": round(t, 2), "left": 0, "right": 0}
            for d in dets:
                ok = frame is not None and lane_lines.on_ego_road(frame, d["box"], y_b, src_w=w,
                                                                  vp_y=None if vp_y is None else vp_y)
                vl = lane_lines.vehicle_lane(d, ok)
                if vl:
                    ev[vl[0]] = max(ev[vl[0]], vl[1])
            out.append(ev)
            t += step
    except Exception as e:
        log("vehicle lane evidence failed: %s" % e)
    return out


def _move_leaving_lanes(prof, k, side):
    """Re-attribute the lanes that leave at section k to `side`: section k and every later one keep their lane counts, and their
    m (lanes missing on the left) is set so that the lanes that stay are on the other side."""
    a, b = prof[k - 1], prof[k]
    gone = a["n"] - b["n"]
    if gone <= 0:
        return
    new_m = a["m"] + gone if side == "left" else a["m"]
    delta = new_m - b["m"]
    for sc in prof[k:]:
        sc["m"] += delta
    lo = min(sc["m"] for sc in prof)
    for sc in prof:
        sc["m"] -= lo


def tag_branches(prof, t_v0, v, video, duration, probe=None, time_of=None):
    """Mark the first place where lanes leave the ego road as a branch when the frames after it show lanes going on as a road of
    their own: rising (a flyover or ramp) or, with a stronger cue, flat. Both sides are asked; a clearly stronger cue on the
    other side than the one the road width points to wins (a paved shoulder can make the wrong side look narrower).
    time_of(d) -> the video time at which the ego car passed distance d (default: constant speed v from t_v0).
    Returns a note or None."""
    from agents import road_probes
    probe = probe or road_probes.branch_evidence
    for k in range(1, len(prof)):
        a, b = prof[k - 1], prof[k]
        if b["n"] >= a["n"] or (time_of is None and v < 1.0):
            continue
        side = "left" if b["m"] > a["m"] else "right"
        t_b = time_of(b["d0"]) if time_of else t_v0 + b["d0"] / v
        ev = {sd: probe(video, t_b, sd, duration) for sd in ("left", "right")}
        other = "right" if side == "left" else "left"
        if max(ev[other]) >= max(road_probes.RISE_P, road_probes.SPLIT_P) and max(ev[other]) > max(ev[side]) + 0.2:
            side = other
            _move_leaving_lanes(prof, k, side)
        p_split, p_rise = ev[side]
        if p_rise >= road_probes.RISE_P:
            b["branch"] = {"side": side, "rise_m": 6.0}
        elif p_split >= road_probes.SPLIT_P:
            b["branch"] = {"side": side, "rise_m": 0.0}
        what = ("a branch that climbs" if p_rise >= road_probes.RISE_P else "a flat branch") if b.get("branch") else "lanes that end"
        return "lanes leaving on the %s at %.0f m (video %.1f s): %s (split/rise left %.2f/%.2f, right %.2f/%.2f)" % (
            side, b["d0"], t_b, what, ev["left"][0], ev["left"][1], ev["right"][0], ev["right"][1])
    return None


# ----------------------------------------------------------------------------------------- measured scene: tracks + lane map
def overlay_speeds(caps, times):
    """[(t, m/s or None)]: the ego speed printed in the video overlay near each time (from the frame descriptions)."""
    out = []
    for t in times:
        v = ego_speed_near(caps, t, 0.75) if caps else None
        out.append((t, None if v is None else v / 3.6))
    return out


def measure_scene(spec, video, duration, caps, lane_rows=None, log_fn=log):
    """Camera, ego travel, vehicle tracks and the bird's-eye lane map of the simulated stretch (agents/camera_geom,
    vehicle_tracks, bev_lanes). Returns a JSON-able dict {"camera", "speeds", "speed_source", "tracks", "lane_samples", "t_v0",
    "t_hi", "note"} or None when the tracker or the lane model is not available."""
    from agents import bev_lanes, camera_geom, lane_lines, vehicle_tracks
    t_v0, T = sim_window(spec)
    t_lo, t_hi = max(t_v0 - 0.5, 0.0), min(max(duration - 0.1, 0.0), t_v0 + T + 5.0)
    try:
        if lane_rows is None:
            lane_rows = lane_lines.lane_timeline(video, t_lo, t_hi)
        y_b, h = lane_lines.road_bottom(video)
        cam, cnote = camera_geom.calibrate(lane_rows, (h, lane_lines.WIDTH))
        times = [round(t_lo + 0.5 * i, 2) for i in range(int((t_hi - t_lo) / 0.5) + 2)]
        ov = overlay_speeds(caps, times)
        known = [v for _, v in ov if v is not None]
        stopped = spec["ego"]["behavior"] == "stopped"
        if known and len(known) >= 0.5 * len(ov) and not stopped:
            speeds = [(t, v) for t, v in ov if v is not None]
            source = "overlay"
        else:
            v0 = 0.0 if stopped else float(spec["ego"]["speed_kmh"]) / 3.6
            speeds, source = [(t_lo, v0), (t_hi, v0)], "scene speed (no overlay)"
        s_ego = camera_geom.travel(speeds, t_v0)
        curve = camera_geom.RoadCurve(spec["road"].get("centerline"))
        vx_at = camera_geom.vx_series(lane_rows, cam.vx)
        if source == "overlay":                    # distances ahead (and so other cars' speeds) depend on the focal length
            import numpy as np
            sp_t, sp_v = [t for t, _ in speeds], [v for _, v in speeds]
            f_est, n_dash = camera_geom.focal_from_dashes(video, cam, times, lambda t: float(np.interp(t, sp_t, sp_v)), y_b, vx_at)
            if f_est:
                cnote += "; focal %.0f px measured from %d dashed-line matches against the overlay speed (default %.0f)" % (
                    f_est, n_dash, cam.f)
                cam.f = f_est
            else:
                cnote += "; focal length could not be measured (%d dashed-line matches), default kept" % n_dash
        raw = vehicle_tracks.detect_tracks(video, t_lo, t_hi, lane_lines.WIDTH)
        tracks = vehicle_tracks.to_road_tracks(raw, cam, s_ego, y_b, curve=curve, vx_at=vx_at)
        obs = [(o["t"], o["box"]) for v_ in raw.values() for o in v_]
        grid = bev_lanes.accumulate(video, cam, s_ego, t_lo, t_hi, y_b, lambda t: [b for tt, b in obs if abs(tt - t) < 0.06],
                                    curve=curve, vx_at=vx_at)
        samples = bev_lanes.lane_samples(grid, 0.0, s_ego(t_hi), tracks) if grid is not None else []    # beyond: only seen from afar
    except Exception as e:
        log_fn("measured scene not available: %s" % e)
        return None
    counts = ", ".join("%d %s" % (sum(1 for t in tracks if t["kind"] == k), k) for k in ("same", "oncoming", "parked", "crossing"))
    note = "%s; ego speed from the %s; %d vehicle tracks (%s)" % (cnote, source, len(tracks), counts)
    return {"camera": cam.to_dict(), "speeds": [[round(t, 2), round(v, 2)] for t, v in speeds], "speed_source": source,
            "tracks": tracks, "lane_samples": samples, "t_v0": t_v0, "t_hi": t_hi, "note": note}


def _travel_fns(scene):
    """(s_ego(t), time_of(d)) of a measured scene: the ego car's travel since the start of the simulated stretch and back."""
    import numpy as np
    from agents import camera_geom
    s_ego = camera_geom.travel([tuple(x) for x in scene["speeds"]], scene["t_v0"])
    ts = np.arange(scene["t_v0"], scene["t_hi"] + 5.0, 0.05)
    ss = np.array([s_ego(t) for t in ts])
    if ss[-1] > ss[0] + 1.0:
        return s_ego, (lambda d: float(np.interp(d, ss, ts)))
    return s_ego, (lambda d: scene["t_v0"])


MIN_MAP_TRAVEL_M = 30.0      # the lane map needs the ego car to drive along the road; a stopped car keeps the earlier count


FORK_SLOPE = 0.15            # m sideways per m along the road: a car on lanes that leave as a branch pulls away at least this fast
FORK_MIN_SPAN_M = 8.0        # ... over at least this much road
FORK_MIN_TRACKS = 2
CLIMB_RATIO = 1.2            # ground / size distance of cars on the branch above which it climbs (flat: about 0.8 .. 1.2)
CLIMB_REL = 1.15             # ... and at least this much above the cars on the ego road (the same ratio on flat ground)


def diverging_tracks(tracks, lane_w):
    """Same-direction tracks that pull away sideways from the ego road (outward slope of d over s), with their side."""
    import numpy as np
    out = []
    for t in tracks:
        if t.get("kind") != "same" or len(t["s"]) < 6:
            continue
        s, d = np.array(t["s"]), np.array(t["d"])
        if s[-1] - s[0] < FORK_MIN_SPAN_M or abs(d.mean()) < 0.5 * lane_w:
            continue
        side = 1.0 if d.mean() > 0 else -1.0
        if np.polyfit(s, d, 1)[0] * side >= FORK_SLOPE:
            out.append((t, "left" if side > 0 else "right"))
    return out


def fork_from_tracks(tracks, lane_w):
    """A fork seen in the traffic itself: FORK_MIN_TRACKS or more cars on one side pulling away from the ego road. The split is
    where those cars were first seen (median). Returns {"side", "s_split", "ids"} or None."""
    import numpy as np
    div = diverging_tracks(tracks, lane_w)
    for side in ("left", "right"):
        ts = [t for t, sd in div if sd == side]
        if len(ts) >= FORK_MIN_TRACKS:
            return {"side": side, "s_split": float(np.median([t["s"][0] for t in ts])), "ids": [t["id"] for t in ts]}
    return None


def branch_climbs(tracks, side, beyond_m, ids=()):
    """Whether the lanes that leave on `side` climb: the cars beyond the lanes that stay (|d| > beyond_m on that side) or pulling
    away (ids) measure further by where their wheels are than by their size (vehicle_tracks.elevation_ratio), median at least
    CLIMB_RATIO, with at least two such cars. (climbs, median ratio, number of cars)."""
    import numpy as np
    sign = 1.0 if side == "left" else -1.0
    cars = [t for t in tracks if t.get("elev_ratio") and t.get("kind") == "same"]
    rs = [t["elev_ratio"] for t in cars if float(np.median(t["d"])) * sign > beyond_m]
    base = [t["elev_ratio"] for t in cars if float(np.median(t["d"])) * sign <= beyond_m]   # cars on the ego road
    if len(rs) < 2:
        return False, None, len(rs)
    med = float(np.median(rs))
    ref = float(np.median(base)) if len(base) >= 3 else 1.0        # the ratio on flat ground in this video (car mix, calibration)
    return med >= CLIMB_RATIO and med >= CLIMB_REL * ref, med, len(rs)


def apply_fork(prof, fork, tracks, lane_w, ego_j=None):
    """Put a fork seen in the traffic into the lane profile: the place where lanes leave on the fork's side moves to the measured
    split, those lanes become a branch, climbing when the cars on it show it. Returns a note, or None when the profile has no
    lanes leaving to attach the fork to. ego_j: the ego lane counted from the left of the widest extent (lane_profile "m" frame),
    so that the lanes that stay beside the ego car are told apart from the ones that leave."""
    drops = [k for k in range(1, len(prof)) if prof[k]["n"] < prof[k - 1]["n"]]
    if not drops:
        return None
    k = min(drops, key=lambda i: abs(prof[i]["d0"] - fork["s_split"]))
    a, b = prof[k - 1], prof[k]
    side = fork["side"]
    if (side == "left") != (b["m"] > a["m"]):
        _move_leaving_lanes(prof, k, side)
    lo = (a["d0"] if a["d0"] is not None else 0.0) + 15.0
    hi = (b["d1"] - 15.0) if b["d1"] is not None else fork["s_split"]
    new_s = min(max(fork["s_split"], lo), max(hi, lo))
    a["d1"] = b["d0"] = round(new_s, 2)
    if ego_j is None:
        ego_j = b["m"] + b["n"] - 1                                  # unknown: assume the right-most lane
    # lanes that stay beside the ego car on the fork's side: past them (in the video's lane widths) the cars are on the branch
    stay = (ego_j - b["m"]) if side == "left" else (b["m"] + b["n"] - 1 - ego_j)
    beyond = (max(stay, 0) + 0.5) * lane_w
    climbs, med, n = branch_climbs(tracks, side, beyond, fork["ids"])
    b["branch"] = {"side": side, "rise_m": 6.0 if climbs else 0.0}
    return "fork seen in the traffic: %d cars pull away on the %s from %.0f m; %s (ground/size distance %s over %d cars)" % (
        len(fork["ids"]), side, new_s, "the branch climbs" if climbs else "level branch", "-" if med is None else "%.2f" % med, n)


def lane_profile_from_map(spec, scene, video, duration, branch_probe=None, log_fn=log):
    """Lane profile from the bird's-eye lane map (a count every few metres along the road): smoothed into sections directly in
    metres, the side of every change from where lanes left, branches tagged like in add_lane_profile. Returns True when the map
    gave a usable count."""
    from agents import lane_profile, lane_lines
    road = spec["road"]
    s_ego, time_of = _travel_fns(scene)
    if s_ego(scene["t_hi"]) < MIN_MAP_TRAVEL_M:
        spec.setdefault("notes", []).append("lane map not used: the ego car covers only %.0f m (a stopped car sees one stretch)" % (
            s_ego(scene["t_hi"])))
        return False
    measured = [x for x in scene["lane_samples"] if x.get("probs")]
    segs = lane_profile.segment_counts(measured, switch_cost=4.0, min_dur_s=10.0)          # "t" is metres here
    if not segs:
        return False
    t_v0, T = sim_window(spec)
    need = 2 if any(a["relation"] == "beside" for a in spec["actors"]) else 1
    bounds = lane_profile.boundary_times(segs, measured)
    sides = [lane_lines.change_side(measured, a["t1"], b["t0"], span=15.0) for a, b in zip(segs, segs[1:])]
    prof = lane_profile.build_profile(segs, bounds, sides, 0.0, 1.0)
    d_ev = s_ego(t_v0 + T)
    left0 = sorted(x["lanes_left"] for x in measured if x["t"] <= 30.0)       # the very start is barely seen yet
    s_last = max(x["t"] for x in measured)
    if len(prof) > 1 and prof[0]["d1"] is not None and prof[0]["d1"] < 25.0:
        prof[1]["d0"] = None                                         # a start stretch barely seen in near-field FOV: not used
        prof.pop(0)
        lo_m = min(sc["m"] for sc in prof)
        for sc in prof:
            sc["m"] -= lo_m
    if len(prof) > 1 and prof[-1]["d0"] is not None and s_last - prof[-1]["d0"] < 25.0:
        prof[-2]["d1"] = None                                        # a last change the map barely saw: not used
        prof.pop()
    only = prof[0]["n"] if len(prof) == 1 else None
    if len(prof) < 2:
        prof = []
    if not prof:
        n = only or next((sg["n"] for sg in segs if sg["t1"] >= d_ev), segs[-1]["n"])
        road["lanes_per_direction"] = max(n, need)
        road["lane_profile"] = None
        if left0:
            spec["ego"]["lane_index"] = int(round(left0[len(left0) // 2]))
        spec.setdefault("notes", []).append("lane map: %d lanes along the simulated stretch" % road["lanes_per_direction"])
        return True
    sec_ev = lane_profile.section_at(prof, d_ev)
    if sec_ev["n"] < need:
        sec_ev["n"] = need
    lane_w = next((x["lane_w"] for x in measured if x.get("lane_w")), 3.5)
    sec0_ = lane_profile.section_at(prof, 0.0) if prof else None
    ego_j = (sec0_["m"] + min(int(round(left0[len(left0) // 2])), sec0_["n"] - 1)) if (left0 and sec0_) else None
    fork = fork_from_tracks(scene.get("tracks") or [], lane_w)
    bnote = apply_fork(prof, fork, scene["tracks"], lane_w, ego_j) if fork and len(prof) > 1 else None   # traffic first
    if bnote is None:                                                            # else ask the vision model
        bnote = tag_branches(prof, t_v0, 0.0, video, duration, branch_probe, time_of=time_of)
    if bnote:
        spec.setdefault("notes", []).append(bnote)
        log_fn(bnote)
    road["lane_profile"] = prof
    road["lanes_per_direction"] = sec_ev["n"]
    sec0 = lane_profile.section_at(prof, 0.0)
    j = min(int(round(left0[len(left0) // 2])), sec0["n"] - 1) if left0 else None
    spec["ego"]["lane_index"] = None if j is None else sec0["m"] + j
    txt = ", ".join("%d lanes%s" % (sc["n"], "" if sc["d0"] is None else " from %.0f m" % sc["d0"]) for sc in prof)
    spec.setdefault("notes", []).append("lane map: %s (changes on the %s); ego lane %s from the left at the start" % (
        txt, "/".join(sides) or "-", "?" if j is None else j + 1))
    log_fn("lane map: " + txt)
    return True


def apply_ego_speed(spec, scene):
    """With the ego speed from the overlay: spec.ego.speed_profile [[sim time, km/h], ...] for the simulation, and speed_kmh =
    the mean speed until the event, so that the scene builder's timing puts the ego car where the video had it."""
    if scene.get("speed_source") != "overlay" or spec["ego"]["behavior"] == "stopped":
        return
    import numpy as np
    t_v0, T = sim_window(spec)
    s_ego, _ = _travel_fns(scene)
    from agents import traj_smoothing
    ts = [t for t, _ in scene["speeds"]]
    vs = [v for _, v in scene["speeds"]]
    grid = np.arange(t_v0, max(scene["t_hi"], t_v0 + T) + 0.01, 0.5)
    v_grid = traj_smoothing.smooth_speed_series(grid, np.interp(grid, ts, vs))      # misread digits, steps -> drivable
    spec["ego"]["speed_profile"] = [[round(t - t_v0, 2), round(float(v) * 3.6, 1)] for t, v in zip(grid, v_grid)]
    if T > 0.5:
        spec["ego"]["speed_kmh"] = round(s_ego(t_v0 + T) / T * 3.6, 1)


def apply_replay(spec, scene, t_crash, log_fn=log):
    """Every measured vehicle track that is not one of the crash vehicles is replayed along its path (spec.replay, in simulation
    time and metres from the ego start); the guessed background vehicles and per-lane traffic counts are dropped."""
    import numpy as np
    from agents import vehicle_tracks
    tracks = scene["tracks"]
    if not tracks:
        return
    ev = spec.get("event") or {}
    third = bool(ev.get("actor2_id"))
    from agents.scene_spec import COLORS
    c1 = next((a for a in spec["actors"] if a["id"] == ev.get("actor_id")), None)
    rgb = (c1 or {}).get("color")
    color = next((k for k, v in COLORS.items() if rgb and "%d,%d,%d" % v == rgb), None)     # crash car's colour name
    skip = set(vehicle_tracks.crash_tracks(tracks, float(t_crash), ego_involved=not third, color=color)) if t_crash is not None else set()
    t_v0, T = sim_window(spec)
    if not third and skip and c1 is not None and (c1.get("relation") == "beside" or c1.get("behavior") == "cutting_in"):
        tr_crash = next((tr for tr in tracks if tr["id"] in skip), None)
        if tr_crash and tr_crash.get("kind") == "same" and tr_crash.get("v"):
            v_meas = float(np.median([abs(v) for v in tr_crash["v"]]))
            sp_prof = spec["ego"].get("speed_profile")
            v_ego_ev = (float(np.interp(T, [p[0] for p in sp_prof], [p[1] for p in sp_prof])) / 3.6) if sp_prof else (float(spec["ego"]["speed_kmh"]) / 3.6)
            if v_ego_ev > 5.0 and v_meas < 0.75 * v_ego_ev:
                v_kmh = round(max(v_meas * 3.6, 12.0), 1)
                c1["speed_kmh"] = v_kmh
                c1["measured"] = {"speed_kmh": v_kmh, "overtake": True}
    lane_w = next((x["lane_w"] for x in scene.get("lane_samples") or [] if x.get("lane_w")), None)
    k_d = 3.5 / lane_w if lane_w else 1.0          # positions across the road in simulation lanes (3.5 m)
    prof = spec["road"].get("lane_profile")
    if prof:
        L_max = max(int(sc["m"]) + int(sc["n"]) for sc in prof)
        idx = spec["ego"].get("lane_index")
        k_ego = (L_max - 1 - int(idx)) if idx is not None else (0 if spec["ego"]["lane"] == "right" else L_max - 1)
    else:
        L_max = int(spec["road"].get("lanes_per_direction", 2))
        k_ego = 0 if spec["ego"]["lane"] == "right" or L_max == 1 else L_max - 1
    d_ego = -(L_max - 0.5 - k_ego) * 3.5
    rep = []
    for tr in tracks:
        if tr["id"] in skip or tr["last_t"] < t_v0:
            continue
        from agents import traj_smoothing
        kind = tr["kind"]
        if kind == "oncoming":
            lat_med = d_ego + float(np.median(tr["d"])) * k_d
            v_med = float(np.median([abs(v) for v in tr["v"]])) if tr.get("v") else 0.0
            if spec["road"].get("one_way") or (lat_med < -1.0 and v_med < 10.0):
                kind = "parked"
        d_s = [float(x) for x in traj_smoothing.smooth_offsets(tr["d"], window=9)]   # detection jitter across the road
        pts = list(zip(tr["t"], tr["s"], d_s))[::5]
        if len(tr["t"]) > 1 and tr["t"][-1] - pts[-1][0] >= 0.2:
            pts.append((tr["t"][-1], tr["s"][-1], d_s[-1]))
        # speed from the positions themselves (distance over time between samples), so that speed and path agree, then made
        # drivable (no steps, comfortable acceleration)
        spd = [abs(b[1] - a[1]) / max(b[0] - a[0], 1e-3) for a, b in zip(pts, pts[1:])]
        spd = spd + spd[-1:] if spd else [0.0]
        if len(spd) >= 3:
            spd = [float(x) for x in traj_smoothing.smooth_speed_series([p[0] for p in pts], spd,
                                                                          vtype=tr["type"] if tr["type"] != "bike" else "car")]
        rep.append({"id": "r%d" % tr["id"], "type": tr["type"], "color": tr["color"], "kind": kind,
                    "path": [[round(t - t_v0, 2), round(s_, 2), round(d_ * k_d, 2)] for t, s_, d_ in pts],
                    "speed": [[round(p[0] - t_v0, 2), round(v * 3.6, 1)] for p, v in zip(pts, spd)]})
    spec["replay"] = rep
    spec["actors"] = [a for a in spec["actors"] if not a["id"].startswith("b")]
    if isinstance(spec.get("traffic"), dict):
        for k in ("ego_lane_ahead", "left_lane", "right_lane", "oncoming"):
            spec["traffic"][k] = 0
    msg = "replaying %d measured vehicles (crash vehicles excluded: %s)" % (len(rep), sorted(skip) or "none matched")
    spec.setdefault("notes", []).append(msg)
    log_fn(msg)


def add_lane_profile(spec, video, duration, t_crash=None, samples=None, step=0.5, log_fn=log, branch_probe=None):
    """Lane count along the simulated stretch of road (straight roads), from the lane-line and road-surface detector
    (agents/lane_lines.py): the count is measured every `step` seconds, smoothed into sections, the side of every change is
    where the road got narrower or wider, and the times become distances driven by the simulated ego car.
    Sets spec.road.lane_profile (or None), spec.road.lanes_per_direction (count at the event) and spec.ego.lane_index.
    Returns the samples so that the caller can store them."""
    from agents import lane_profile, lane_lines
    road = spec["road"]
    if road["type"] != "straight":
        return samples
    t_v0, T = sim_window(spec)
    t_lo, t_hi = max(t_v0 - 0.5, 0.0), min(max(duration - 0.1, 0.0), t_v0 + T + 5.0)
    if samples is None:
        samples = lane_lines.lane_timeline(video, t_lo, t_hi, step=step)
        lane_lines.apply_vehicle_evidence(samples, vehicle_lane_evidence(video, t_lo, t_hi, samples))
    measured = [s for s in samples if s.get("probs")]
    segs = lane_profile.segment_counts(measured)
    if not segs:
        road["lane_profile"] = None
        spec.setdefault("notes", []).append("lane count: no usable lane measurement (%d windows)" % len(samples))
        return samples
    t_ev = t_v0 + T if t_crash is None else float(t_crash)
    at_ev = next((sg for sg in segs if sg["t1"] >= t_ev), segs[-1])
    need = 2 if any(a["relation"] == "beside" for a in spec["actors"]) else 1
    v = 0.0 if spec["ego"]["behavior"] == "stopped" else float(spec["ego"]["speed_kmh"]) / 3.6
    bounds = lane_profile.boundary_times(segs, measured)
    sides = [lane_lines.change_side(measured, a["t1"], b["t0"]) for a, b in zip(segs, segs[1:])]
    prof = lane_profile.build_profile(segs, bounds, sides, t_v0, v)
    if not prof:
        road["lanes_per_direction"] = max(at_ev["n"], need)
        road["lane_profile"] = None
        spec.setdefault("notes", []).append("lane count from %d windows: %d (no change on the simulated stretch)" % (
            len(measured), road["lanes_per_direction"]))
        return samples
    sec_ev = lane_profile.section_at(prof, v * T)
    if sec_ev["n"] < need:                                    # the merging car needs a lane beside the ego car
        sec_ev["n"] = need
    bnote = tag_branches(prof, t_v0, v, video, duration, branch_probe)
    if bnote:
        spec.setdefault("notes", []).append(bnote)
        log_fn(bnote)
    road["lane_profile"] = prof
    road["lanes_per_direction"] = sec_ev["n"]
    sec0 = lane_profile.section_at(prof, 0.0)
    left = sorted(s["lanes_left"] for s in measured if s.get("lanes_left") is not None and t_v0 <= s["t"] <= t_v0 + 1.5)
    j = min(int(round(left[len(left) // 2])), sec0["n"] - 1) if left else None
    spec["ego"]["lane_index"] = None if j is None else sec0["m"] + j
    txt = ", ".join("%d lanes%s" % (sc["n"], "" if sc["d0"] is None else " from %.0f m" % sc["d0"]) for sc in prof)
    spec.setdefault("notes", []).append("lane profile from %d windows: %s (changes on the %s); ego lane %s from the left at the start" % (
        len(measured), txt, "/".join(sides) or "-", "?" if j is None else j + 1))
    log_fn("lane profile: " + txt)
    return samples


def refine(spec, caps, events, duration, roads=None, t_crash=None, ego_motion=None):
    """Deterministic fusion: road from the frame scores, ego motion from the pre-event motion analysis."""
    notes = []
    low = [c["text"].lower() for c in caps]
    if roads:
        spec["road"]["lanes_per_direction"] = roads["lanes"]
        is_cross = roads["intersection"] >= 0.55 or (
            roads["intersection"] >= 0.42 and _positive_mentions(low, ("intersection", "traffic light", "crossroad")) >= 3)
        spec["road"]["type"] = "crossroad" if is_cross else "straight"
        spec["road"]["traffic_signals"] = bool(is_cross)
        notes.append("road from frame scores: %d lane(s) per direction (%s), intersection probability %.2f -> %s" % (
            roads["lanes"], roads["lane_probs"], roads["intersection"], spec["road"]["type"]))
    win_hi = float(t_crash) if t_crash is not None else float(duration)
    win_lo = max(0.0, win_hi - 4.5)
    win_dur = max(win_hi - win_lo, 1.0)
    stopped = sum(max(0.0, min(float(e.get("until", e["t"])), win_hi) - max(float(e["t"]), win_lo))
                  for e in events if e["type"] == "stopped")
    brake_late = (t_crash is None) and any(
        e["type"] == "brake_to_stop" and e["t"] > 0.7 * duration for e in events)
    highway = _positive_mentions(low, ("highway", "freeway")) >= max(2, len(low) // 3)
    ego = spec["ego"]
    if ego_motion and ego_motion.get("ego_stopped"):
        ego["behavior"], ego["speed_kmh"] = "stopped", 0.0
        notes.append("ego stopped from background-stable flow (pre_edge=%.2f)" % ego_motion.get("pre_edge", 0.0))
    elif stopped / win_dur > 0.6 and not (ego_motion and not ego_motion.get("ego_stopped")):
        ego["behavior"], ego["speed_kmh"] = "stopped", 0.0
    else:
        if ego["speed_kmh"] < 10:
            ego["speed_kmh"] = 90.0 if highway else 50.0
        ego["behavior"] = "slow_to_stop" if brake_late else ("cruise" if ego["behavior"] in ("stopped", "slow_to_stop") else ego["behavior"])
    notes.append("ego behaviour from motion analysis: %.0f%% stopped, late braking %s -> %s" % (100 * stopped / win_dur, brake_late, ego["behavior"]))
    spec.setdefault("notes", []).extend(notes)
    return spec


# ----------------------------------------------------------------------------------------- phases
def _cached(out, name, reuse):
    fn = os.path.join(out, name)
    if reuse and os.path.exists(fn):
        return _load(fn)
    return None


def reconcile_traffic(spec):
    """Zero out impossible traffic bins once road one-way status and final ego lane are known."""
    road = spec.get("road") or {}
    if road.get("type") == "crossroad":
        road["one_way"] = False
    traffic = spec.get("traffic")
    if not isinstance(traffic, dict):
        return spec
    if road.get("one_way"):
        traffic["oncoming"] = 0
    if (spec.get("ego") or {}).get("lane") == "right":
        traffic["right_lane"] = 0
    if (spec.get("event") or {}).get("type") in ("collision", "near_miss") and traffic.get("rel_speed_kmh", 0.0) == 0.0:
        cut_actor = next((a for a in spec.get("actors") or []
                          if a.get("behavior") == "cutting_in" or a.get("relation") == "beside"), None)
        if cut_actor is not None:
            src_key = "left_lane" if cut_actor.get("lane") == "left" else "right_lane"
            if int(traffic.get(src_key, 0) or 0) >= 2:
                traffic["rel_speed_kmh"] = -15.0
    return spec


def phase1(video, out, spec_path=None, fps=6.0, assume_crash=True, reuse=False):
    from agents import video_events, scene_spec, scene_builder, llm_backend, road_probes, vehicle_tracker
    os.makedirs(out, exist_ok=True)
    t0 = time.time()
    t_stages = {}
    degraded = None
    if spec_path:
        spec, err = scene_spec.normalize(_load(spec_path))
        log("using the given spec")
    else:
        log("analysing the ego motion of the whole video")
        ts = time.time()
        ev = video_events.analyze(video)
        _save(ev, os.path.join(out, "video_events.json"))
        dur = ev.get("effective_duration_s") or ev["duration_s"]
        t_stages["video_events"] = time.time() - ts
        if not llm_backend.ensure_server():
            degraded = "the local model could not be started (is A:\\llm complete?): fallback scene"
            log(degraded)
            spec, _ = scene_spec.normalize({})
        else:
            ts = time.time()
            caps = _cached(out, "frame_descriptions.json", reuse) or caption_timeline(video, dur, fps)
            _save(caps, os.path.join(out, "frame_descriptions.json"))
            t_stages["caption_timeline"] = time.time() - ts
            t_crash = how = None
            if assume_crash:
                ts = time.time()
                scores = _cached(out, "crash_scores.json", reuse) or crash_scores(video, dur)
                _save(scores, os.path.join(out, "crash_scores.json"))
                cands = road_probes.pick_crash_candidates(scores)
                t_crash = None
                if cands:                      # coarse scan -> strongest peaks -> fine search around each
                    score_cache = {round(float(s["t"]), 2): float(s["score"]) for s in scores}

                    def cached_win(t):
                        k = round(float(t), 2)
                        if k not in score_cache:
                            score_cache[k] = sum(window_scores(video, t, dur)) / 2.0
                        return score_cache[k]
                    t_crash, sc = road_probes.refine_crash_time(cands, cached_win)
                    how = "fine search around the candidates %s, score %.2f" % (cands, sc)
                if t_crash is None:
                    t_crash, how = pick_crash_time(scores, dur)
                t_stages["crash_search"] = time.time() - ts
                log("crash moment %.1f s (%s)" % (t_crash, how))
            log("turning the frame descriptions into a scene description")
            ts = time.time()
            spec, err = llm_backend.ask_json(build_prompt2(caps, dur, t_crash), check=scene_spec.check, retries=3, log=log, max_tokens=900)
            t_stages["scene_llm"] = time.time() - ts
            if spec is None:
                degraded = "model reply unusable (%s): default scene" % err
                log(degraded)
                spec, _ = scene_spec.normalize({})
            t_active = min(dur, max(float(t_crash) + 2.0, 4.0)) if t_crash is not None else dur
            ts = time.time()
            roads = _cached(out, "road_scores.json", reuse) or road_scores(video, dur, t1=t_active, t_crash=t_crash)
            _save(roads, os.path.join(out, "road_scores.json"))
            ego_motion = video_events.ego_motion_metrics(video, t_crash, dur) if t_crash is not None else None
            spec = refine(spec, caps, ev["events"], dur, roads, t_crash=t_crash, ego_motion=ego_motion)
            tb = road_probes.traffic_bins(video, dur, t_crash, t1=t_active)
            if tb:
                spec["traffic"] = tb
                spec.setdefault("notes", []).append("traffic measured from the frames: %s" % {k: v for k, v in tb.items() if k != "rel_speed_kmh"})
            edges = road_probes.road_edges(video, dur, t1=t_active)
            if edges:
                spec["road"].update(edges)
                spec.setdefault("notes", []).append("road edges from the frames: right %s, left %s, %s" % (
                    edges["edge_right"], edges["edge_left"], "one-way" if edges["one_way"] else "two-way"))
            yellow_left = road_probes.yellow_centerline_left(video, dur, t_crash=t_crash)
            t_stages["road_probes"] = time.time() - ts
            tracked = None
            if assume_crash and t_crash is not None:
                log("tracking 3D vehicle positions around the crash moment")
                ts_tr = time.time()
                ego_spd_est = 0.0 if spec["ego"]["behavior"] == "stopped" else (ego_speed_near(caps, t_crash) or spec["ego"]["speed_kmh"])
                tracked = vehicle_tracker.analyze_video(
                    video, t_crash, dur, roads=roads, ego_motion=ego_motion, edges=edges,
                    yellow_left=yellow_left, ego_speed_kmh=ego_spd_est)
                t_stages["vehicle_tracker"] = time.time() - ts_tr
                _save(tracked, os.path.join(out, "tracked_vehicles.json"))
            if assume_crash:
                log("analysing the frames around the crash moment")
                ts = time.time()
                crash, cerr, ctimes = analyse_crash_window(video, t_crash, dur, roads=roads)
                t_stages["crash_window"] = time.time() - ts
                if crash and tracked:
                    if tracked.get("ego_involved") is False and tracked.get("second_vehicle"):
                        crash = dict(crash)
                        crash["ego_involved"] = False
                        fv = dict(crash.get("first_vehicle") or {})
                        fv.update({k: v for k, v in (tracked.get("first_vehicle") or {}).items() if v is not None})
                        crash["first_vehicle"] = fv
                        crash["second_vehicle"] = dict(tracked["second_vehicle"])
                        crash["description"] = "third-party collision between %s and %s" % (
                            fv.get("approach", "crossing"), crash["second_vehicle"].get("approach", "driving"))
                    elif tracked.get("first_vehicle"):
                        tfv = tracked["first_vehicle"]
                        if tfv.get("distance_m"):
                            crash["first_vehicle"] = dict(crash["first_vehicle"], distance_m=tfv["distance_m"])
                        if tfv.get("approach") in ("crossing_from_left", "crossing_from_right") and \
                                crash["first_vehicle"].get("approach") in ("merging_from_left", "merging_from_right"):
                            crash["first_vehicle"]["approach"] = tfv["approach"]
                _save({"t_crash": t_crash, "found_by": how, "frame_times": ctimes, "answer": crash, "error": cerr},
                      os.path.join(out, "crash_window.json"))
                if crash:
                    sp = None if spec["ego"]["behavior"] == "stopped" else ego_speed_near(caps, t_crash)
                    spec = apply_crash(spec, crash, t_crash, ego_speed=sp, tracked=tracked)
                    spec.setdefault("notes", []).append("crash at %.1f s found by %s" % (t_crash, how))
                else:
                    spec.setdefault("notes", []).append("crash analysis failed: %s" % cerr)
            reconcile_traffic(spec)
            if spec["road"]["type"] == "straight":
                ts = time.time()
                spec = add_curve(spec, video, dur, t_crash, caps)
                t_stages["curve"] = time.time() - ts
                ts = time.time()
                scene = _cached(out, "scene_measure.json", reuse) or measure_scene(spec, video, dur, caps, log_fn=log)
                from_map = False
                if scene:
                    _save(scene, os.path.join(out, "scene_measure.json"))
                    log(scene["note"])
                    spec.setdefault("notes", []).append(scene["note"])
                    apply_ego_speed(spec, scene)
                    from_map = lane_profile_from_map(spec, scene, video, dur)
                    apply_replay(spec, scene, t_crash)
                t_stages["measured_scene"] = time.time() - ts
                if not from_map:                       # no tracker / lane model: lane count from the per-window detector
                    ts = time.time()
                    lane_samples = add_lane_profile(spec, video, dur, t_crash=t_crash,
                                                    samples=_cached(out, "lane_timeline.json", reuse))
                    if lane_samples:
                        _save(lane_samples, os.path.join(out, "lane_timeline.json"))
                    t_stages["lane_profile"] = time.time() - ts
    spec["degraded"] = degraded
    if not spec_path and os.environ.get("LLM_KEEP") != "1":
        llm_backend.stop_server()            # free the memory for CARLA
    log("building the scene from the description")
    ts = time.time()
    plan = scene_builder.build(spec, out, SCENE_ID)
    t_stages["build"] = time.time() - ts
    _save(spec, os.path.join(out, SCENE_ID + "_spec.json"))
    if t_stages:
        log("stage timings (s): " + ", ".join("%s=%.1f" % (k, v) for k, v in t_stages.items()))
    log("phase 1 done in %.1f s: road=%s lanes=%d, ego %s %.0f km/h, %d actors, event=%s" % (
        time.time() - t0, spec["road"]["type"], spec["road"]["lanes_per_direction"], spec["ego"]["type"],
        spec["ego"]["speed_kmh"], len(spec["actors"]), spec["event"]["type"]))
    return spec, plan


def _load_map_with_fallback(client, out):
    """Build and load the OpenDRIVE map of the scene. A curved road that CARLA rejects is rebuilt straight (same scene, straight
    centre line) and the reason is returned; a straight road that fails raises."""
    from agents import map_builder, scene_builder
    plan_fn = os.path.join(out, SCENE_ID + "_motion.json")
    net_fn = os.path.join(out, SCENE_ID + ".net.xml")
    with open(plan_fn, encoding="utf-8") as f:
        road = (json.load(f).get("road") or {})
    try:
        map_builder.build_and_load_map(client, net_fn, out, road)
        return None
    except Exception as e:
        curved = len(road.get("centerline") or []) > 2
        if not curved:
            raise
        reason = "CARLA rejected the curved map (%s): rebuilt with a straight road" % str(e)[:120]
        log(reason)
    spec_fn = os.path.join(out, SCENE_ID + "_spec.json")
    with open(spec_fn, encoding="utf-8") as f:
        spec = json.load(f)
    spec["road"]["centerline"] = None
    spec["road"]["curve_note"] = reason
    spec.setdefault("notes", []).append(reason)
    scene_builder.build(spec, out, SCENE_ID)
    with open(plan_fn, encoding="utf-8") as f:
        road = (json.load(f).get("road") or {})
    map_builder.build_and_load_map(client, net_fn, out, road)
    return reason


def phase2(out, max_iter=8, record_video=False):
    from agents import motion_runner, scenario_check
    import carla
    ok = False
    for attempt in range(6):                       # the server can be busy right after a map change: wait up to ~2 minutes
        try:
            c = carla.Client("localhost", 2000)
            c.set_timeout(20)
            c.get_world()
            ok = True
            break
        except Exception:
            time.sleep(5)
    if not ok:
        log("CARLA is not reachable: start the editor, press Play, then run again with --sim-only")
        return None

    fallback = _load_map_with_fallback(c, out)

    def run_fn(load_map):                          # the map is already loaded (once, above); every run starts with a clean cleanup
        return motion_runner.run_motion(out, SCENE_ID, load_map=False, hold_sec=2.0, record_video=record_video)
    ready = scenario_check.search(out, SCENE_ID, run_fn, max_iter=max_iter, log=log)
    if fallback:
        ready["map_fallback"] = fallback
    return ready


def report(out, spec, ready):
    lines = []
    if spec:
        lines.append("Scene: %s road, %d lanes per direction; ego %s at %.0f km/h (%s, %s lane)" % (
            spec["road"]["type"], spec["road"]["lanes_per_direction"], spec["ego"]["type"], spec["ego"]["speed_kmh"],
            spec["ego"]["behavior"], spec["ego"]["lane"]))
        for a in spec["actors"]:
            lines.append("  actor %s: %s %s, %s %s lane %s, %s, %.0f m, %.0f km/h" % (
                a["id"], a.get("color") or "", a["type"], a["relation"], a["lane"], "", a["behavior"], a["distance_m"], a["speed_kmh"]))
        ev = spec["event"]
        lines.append("Event: %s at %.1fs (%s) %s" % (ev["type"], ev["time_s"], ", ".join(x for x in (ev.get("actor_id"), ev.get("actor2_id")) if x) or "no actor", ev["description"]))
        for n in spec.get("notes", []):
            lines.append("  note: " + n)
        if spec.get("degraded"):
            lines.append("WARNING: " + spec["degraded"])
    if ready:
        lines.append("Simulation: %s after %d run(s); expectation %s" % (ready["status"], ready["iterations"], ready["expect"]))
        hist = ready.get("history") or []
        last = hist[-1]["detail"] if hist else {}
        if last.get("direction"):
            lines.append("  contact: %s at %.1f s" % (last["direction"], last.get("contact_t_sim") or 0.0))
        for r in last.get("reasons", []):
            lines.append("  failed rule: " + r)
        if ready.get("map_fallback"):
            lines.append("  note: " + ready["map_fallback"])
    txt = "\n".join(lines)
    with open(os.path.join(out, "report.txt"), "w", encoding="utf-8") as f:
        f.write(txt)
    return txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("video", nargs="?")
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-sim", action="store_true")
    ap.add_argument("--sim-only", action="store_true")
    ap.add_argument("--spec")
    ap.add_argument("--fps", type=float, default=6.0, help="frames per second described over the whole video (default 6)")
    ap.add_argument("--no-assume-crash", action="store_true", help="do not look for a crash moment")
    ap.add_argument("--max-iter", type=int, default=8)
    ap.add_argument("--reuse", action="store_true", help="reuse frame descriptions and scores already in the output folder")
    ap.add_argument("--record", action="store_true", help="record CARLA spectator video for the simulation")
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    spec = ready = None
    if not a.sim_only:
        if not a.video and not a.spec:
            ap.error("a video (or --spec) is required")
        spec, _ = phase1(a.video, a.out, a.spec, a.fps, not a.no_assume_crash, a.reuse)
    else:
        spec = _load(os.path.join(a.out, SCENE_ID + "_spec.json"))
    if not a.no_sim:
        ready = phase2(a.out, a.max_iter, record_video=a.record)
    print(report(a.out, spec, ready))


if __name__ == "__main__":
    main()
