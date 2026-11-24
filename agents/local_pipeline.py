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
    "If a speed is printed in the video overlay, START with SPEED=<number><MPH or KMH>. "
    "Then answer in ONE short sentence: the road (number of lanes in the ego vehicle's direction, which lane the ego vehicle "
    "is in, and mention an intersection only if traffic lights or a cross-street are clearly visible) and every vehicle you "
    "see with its type (car, SUV, pickup, van, truck, motorcycle, bus), colour, position (ahead in my lane, in the left lane, "
    "in the right lane, oncoming, crossing) and distance (very close under 3 m, near, far). Only state what you can see.")

LABEL_RE = re.compile(r"^\W*(CONTACT|CLOSE|NONE)\b[\s:\-\.]*", re.I)
SPEED_RE = re.compile(r"(?:the\s+speed\s+is\s+|speed\s*[:=]\s*)?(\d{1,3})\s*(MPH|KMH|KM/H)\b|SPEED\s*=\s*(\d{1,3})\s*(MPH|KMH|KM/H)?", re.I)
UNITLESS_SPEED_RE = re.compile(r"(?:the\s+speed\s+is\s+|speed\s*[:=]\s*)(\d{1,3})(?:\b|$)", re.I)


def parse_caption(t, text):
    m = LABEL_RE.match(text)
    label = m.group(1).upper() if m else "NONE"
    body = text[m.end():] if m else text
    sp = SPEED_RE.search(text)
    speed = None
    unit = None
    if sp:
        num = sp.group(1) or sp.group(3)
        unit = (sp.group(2) or sp.group(4) or "MPH").upper()
        speed = float(num) * (1.609 if unit == "MPH" else 1.0)
    out = {"t": t, "label": label, "text": SPEED_RE.sub("", body).strip(" .,:;-"), "speed_kmh": speed}
    if unit is not None:
        out["speed_unit"] = unit
    return out


def backfill_caption_speeds(caps):
    """Recover truncated unitless overlay speeds (e.g. 'The speed is 39') when other captions establish the video's unit."""
    units = [c.get("speed_unit") for c in caps if c.get("speed_unit")]
    if not units:
        # Cached captions from older runs may not have speed_unit: infer MPH if any speed matches n * 1.609
        mph_votes = sum(1 for c in caps if c.get("speed_kmh") and abs(c["speed_kmh"] / 1.609 - round(c["speed_kmh"] / 1.609)) < 0.02)
        kmh_votes = sum(1 for c in caps if c.get("speed_kmh") and abs(c["speed_kmh"] - round(c["speed_kmh"])) < 0.02)
        if mph_votes + kmh_votes == 0:
            return caps
        dom_unit = "MPH" if mph_votes >= kmh_votes else "KMH"
    else:
        dom_unit = max(set(units), key=units.count)
    scale = 1.609 if dom_unit == "MPH" else 1.0
    for c in caps:
        if c.get("speed_kmh") is not None:
            continue
        m = UNITLESS_SPEED_RE.search(c.get("text") or "")
        if m:
            val = float(m.group(1))
            if 5.0 <= val <= 140.0:
                c["speed_kmh"] = val * scale
    return caps


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
            c = llm_backend.chat(CAPTION_PROMPT, images=fr, json_mode=False, max_tokens=120, temperature=0.1).strip()
        except Exception as e:
            c = "NONE: (description failed: %s)" % e
        caps.append(parse_caption(t, " ".join(c.split())))
        if (i + 1) % 20 == 0 or i + 1 == len(times):
            log_fn("described %d/%d frames (%.0f s)" % (i + 1, len(times), time.time() - t0))
    return backfill_caption_speeds(caps)


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
PART_KEYS = {"A": "front", "B": "front_left", "C": "front_right", "D": "left_side", "E": "right_side", "F": "rear"}
EGO_PART_Q = (FRAME_PREFIX + "Which part of the EGO vehicle (the camera car) is closest to impact or gets hit?\n"
              "A) front bumper / hood center\nB) front-left corner / fender\nC) front-right corner / fender\n"
              "D) left side\nE) right side\nF) rear")
OTHER_PART_Q = (FRAME_PREFIX + "Which part of the OTHER vehicle involved in the crash gets hit or makes contact?\n"
                "A) front bumper / grille\nB) front-left corner\nC) front-right corner\n"
                "D) left side\nE) right side\nF) rear bumper / back")


def infer_contact_parts(first_approach, ego_involved=True, second_approach=None):
    """Expected impacted zones of each vehicle inferred from their approach geometry in the dashcam video."""
    a1 = str(first_approach or "").lower().replace(" ", "_").replace("-", "_")
    if not ego_involved:
        if "left" in a1:
            return {"c1": "right_side", "c2": "front"}
        if "right" in a1:
            return {"c1": "left_side", "c2": "front"}
        return {"c1": "side", "c2": "front"}
    mapping = {
        "merging_from_left": {"ego": "front_left", "c1": "right_side"},
        "merging_from_right": {"ego": "front_right", "c1": "left_side"},
        "crossing_from_left": {"ego": "front", "c1": "right_side"},
        "crossing_from_right": {"ego": "front", "c1": "left_side"},
        "ahead": {"ego": "front", "c1": "rear"},
        "from_behind": {"ego": "rear", "c1": "front"},
        "oncoming": {"ego": "front", "c1": "front"},
    }
    return dict(mapping.get(a1, {"ego": "front", "c1": "side"}))


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
    parts = infer_contact_parts(approach, ego_involved=True)
    probs = {"position": {POS_KEYS[k]: round(v, 2) for k, v in pp.items()},
             "type": {TYPE_KEYS[k]: round(v, 2) for k, v in pt.items()},
             "color": {COLOR_KEYS.get(k, "other"): round(v, 2) for k, v in pc.items()},
             "maneuver": {MANEUVER_KEYS[k]: round(v, 2) for k, v in pm.items()},
             "entry_flow": {"left_to_right": round(l2r, 1), "right_to_left": round(r2l, 1)}}
    try:
        from agents.contact_analysis import zones_compatible
        pe_list, po_list = [], []
        for t in times[-2:]:
            fr = video_events.grab_frames(video, [t], width=width)
            if not fr:
                continue
            pe_list.append(llm_backend.choice_probabilities(EGO_PART_Q, fr, "ABCDEF"))
            po_list.append(llm_backend.choice_probabilities(OTHER_PART_Q, fr, "ABCDEF"))
        if pe_list and po_list:
            pe, po = _avg(pe_list), _avg(po_list)
            probs["ego_part"] = {PART_KEYS[k]: round(v, 2) for k, v in pe.items()}
            probs["other_part"] = {PART_KEYS[k]: round(v, 2) for k, v in po.items()}
            for who, pdict in (("ego", pe), ("c1", po)):
                top_k = max(pdict, key=pdict.get)
                if pdict[top_k] >= 0.55 and zones_compatible(PART_KEYS[top_k], parts[who]):
                    parts[who] = PART_KEYS[top_k]
    except Exception:
        pass
    word = {"merging_from_left": "changes from the left lane into the lane of", "merging_from_right": "changes from the right lane into the lane of"}
    crash = {"collision": True, "ego_involved": True,
             "first_vehicle": {"type": typ, "color": col, "approach": approach, "speed": "slow"}, "second_vehicle": None,
             "contact_parts": parts,
             "probabilities": probs,
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
            covered = set()
            for ba in keep:
                rel_b, lane_b = ba["relation"], ba["lane"]
                if rel_b == "oncoming":
                    covered.add("oncoming")
                elif rel_b == "ahead":
                    if lane_b == "same":
                        covered.add("ego_lane_ahead")
                    elif lane_b == "left":
                        covered.add("left_lane")
                    elif lane_b == "right":
                        covered.add("right_lane")
            for k_tr in covered:
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
    if collision:
        if not ego_inv:
            v1_app = (crash.get("first_vehicle") or {}).get("approach")
            if tracked and (tracked.get("first_vehicle") or {}).get("approach"):
                v1_app = tracked["first_vehicle"]["approach"]
            v2_app = second["behavior"] if isinstance(second, dict) else None
            parts = infer_contact_parts(v1_app, ego_involved=False, second_approach=v2_app)
        else:
            cp_raw = crash.get("contact_parts")
            parts = dict(cp_raw) if isinstance(cp_raw, dict) else infer_contact_parts((crash.get("first_vehicle") or {}).get("approach"))
        if parts:
            event["contact_parts"] = parts
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


def _clamp_post_branch(prof, k0="d0", k1="d1"):
    """After a highway fork (`branch` on left or right), subsequent sections of the continuing road cannot regrow lanes back
    into the diverging branch's side. Clamp any regrowth and merge resulting identical adjacent sections."""
    bi = next((i for i in range(1, len(prof)) if isinstance(prof[i], dict) and prof[i].get("branch")), None)
    if bi is None:
        return prof
    b = prof[bi]
    side = b["branch"].get("side")
    for sc in prof[bi + 1:]:
        if side == "left":
            if sc["m"] < b["m"]:
                cut = b["m"] - sc["m"]
                sc["m"] = b["m"]
                sc["n"] = max(1, sc["n"] - cut)
        elif side == "right":
            right_lim = b["m"] + b["n"]
            if sc["m"] >= right_lim:
                sc["m"] = max(0, right_lim - 1)
                sc["n"] = 1
            elif sc["m"] + sc["n"] > right_lim:
                sc["n"] = max(1, right_lim - sc["m"])
    i = bi
    while i + 1 < len(prof):
        cur, nxt = prof[i], prof[i + 1]
        if not nxt.get("branch") and cur["n"] == nxt["n"] and cur["m"] == nxt["m"]:
            cur[k1] = nxt[k1]
            prof.pop(i + 1)
        else:
            i += 1
    return prof


def tag_branches(prof, t_v0, v, video, duration, probe=None, time_of=None, fork_side_probe=None, blocked_sides=()):
    """Mark the first place where lanes leave the ego road as a branch when the frames around/after it show lanes going on as
    a road of their own: rising (a flyover or ramp) or, with a stronger cue, flat. Both sides are asked; a comparative
    left-vs-right fork probe, physical road-edge barriers (`blocked_sides`), or a clearly stronger cue on the other side can
    move the leaving lanes to the true branch side.
    time_of(d) -> the video time at which the ego car passed distance d (default: constant speed v from t_v0).
    Returns a note or None."""
    from agents import road_probes
    custom_probe = probe is not None
    probe = probe or road_probes.branch_evidence
    if fork_side_probe is None and not custom_probe:
        fork_side_probe = road_probes.fork_side
    for k in range(1, len(prof)):
        a, b = prof[k - 1], prof[k]
        if b["n"] >= a["n"] or (time_of is None and v < 1.0):
            continue
        side = "left" if b["m"] > a["m"] else "right"
        t_b = time_of(b["d0"]) if time_of else t_v0 + b["d0"] / v
        ev = {sd: probe(video, t_b, sd, duration) for sd in ("left", "right")}
        other = "right" if side == "left" else "left"
        fs = fork_side_probe(video, t_b, duration) if fork_side_probe is not None else None
        move_other = False
        if fs and fs.get(other, 0.0) >= 0.45 and fs.get(other, 0.0) > fs.get(side, 0.0) + 0.15:
            move_other = True
        elif side in blocked_sides and other not in blocked_sides and max(ev[other]) >= road_probes.RISE_P:
            move_other = True
        elif max(ev[other]) >= max(road_probes.RISE_P, road_probes.SPLIT_P) and max(ev[other]) > max(ev[side]) + 0.2:
            move_other = True
        if move_other:
            side = other
            _move_leaving_lanes(prof, k, side)
        p_split, p_rise = ev[side]
        if p_rise >= road_probes.RISE_P:
            b["branch"] = {"side": side, "rise_m": 6.0}
        elif p_split >= road_probes.SPLIT_P or (fs and fs.get(side, 0.0) >= 0.55 and fs.get(side, 0.0) > fs.get("none", 0.0)):
            b["branch"] = {"side": side, "rise_m": 0.0}
        what = ("a branch that climbs" if (b.get("branch") and b["branch"]["rise_m"] > 0) else "a flat branch") if b.get("branch") else "lanes that end"
        fs_txt = (", fork-side L=%.2f/R=%.2f/N=%.2f" % (fs["left"], fs["right"], fs["none"])) if fs else ""
        return "lanes leaving on the %s at %.0f m (video %.1f s): %s (split/rise left %.2f/%.2f, right %.2f/%.2f%s)" % (
            side, b["d0"], t_b, what, ev["left"][0], ev["left"][1], ev["right"][0], ev["right"][1], fs_txt)
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
    vehicle_tracks, bev_lanes). Each subsystem degrades independently so a missing/offline YOLO still keeps camera
    calibration, overlay ego speed profile, and TwinLiteNet BEV lane mapping. Returns a JSON-able dict or None."""
    from agents import bev_lanes, camera_geom, lane_lines, vehicle_tracks
    t_v0, T = sim_window(spec)
    t_lo, t_hi = max(t_v0 - 0.5, 0.0), min(max(duration - 0.1, 0.0), t_v0 + T + 5.0)
    try:
        if lane_rows is None:
            lane_rows = lane_lines.lane_timeline(video, t_lo, t_hi)
        y_b, h = lane_lines.road_bottom(video)
        if y_b is None or h is None:
            raise RuntimeError("video could not be read for road_bottom")
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
    except Exception as e:
        log_fn("measured scene not available: %s" % e)
        return None
    tracks, obs, yolo_err = [], [], None
    try:
        raw = vehicle_tracks.detect_tracks(video, t_lo, t_hi, lane_lines.WIDTH)
        tracks = vehicle_tracks.to_road_tracks(raw, cam, s_ego, y_b, curve=curve, vx_at=vx_at)
        obs = [(o["t"], o["box"]) for v_ in raw.values() for o in v_]
    except Exception as e:
        yolo_err = str(e)
        log_fn("vehicle tracking unavailable (continuing with BEV lane map and ego speed): %s" % yolo_err)
    samples, bev_err = [], None
    try:
        boxes_fn = (lambda t: [b for tt, b in obs if abs(tt - t) < 0.06]) if obs else None
        grid = bev_lanes.accumulate(video, cam, s_ego, t_lo, t_hi, y_b, boxes_fn, curve=curve, vx_at=vx_at)
        road_sp = spec.get("road") or {}
        ego_lane = (spec.get("ego") or {}).get("lane")
        blocked_ego = tuple(
            sd for sd, key in (("right", "edge_right"), ("left", "edge_left"))
            if road_sp.get(key) == "barrier" and ego_lane == sd
        )
        fork_tr = fork_from_tracks(tracks, 3.5) if tracks else None
        excl = (fork_tr["ids"], fork_tr["s_split"]) if fork_tr else None
        samples = bev_lanes.lane_samples(
            grid, 0.0, s_ego(t_hi), tracks, blocked_sides=blocked_ego, exclude_after=excl
        ) if grid is not None else []    # beyond: only seen from afar
    except Exception as e:
        bev_err = str(e)
        log_fn("BEV lane map unavailable: %s" % bev_err)
    if not tracks and not samples and source != "overlay":
        return None
    counts = ", ".join("%d %s" % (sum(1 for t in tracks if t["kind"] == k), k) for k in ("same", "oncoming", "parked", "crossing"))
    note = "%s; ego speed from the %s; %d vehicle tracks (%s)" % (cnote, source, len(tracks), counts)
    if yolo_err:
        note += " [YOLO degraded: %s]" % yolo_err[:120]
    if bev_err:
        note += " [BEV degraded: %s]" % bev_err[:120]
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
    """Same-direction tracks that pull away sideways beyond the continuing lanes onto a diverging branch, with their side
    and the distance s_div where divergence starts. Checks both the whole track and contiguous sub-windows (handling cars
    seen before a split that diverge and then run parallel on the branch); ignores ordinary interior lane changes
    (final |d| < 2.5 * lane_w)."""
    import numpy as np
    out = []
    min_end_d = 2.5 * max(float(lane_w or 3.5), 2.5)
    for t in tracks:
        if t.get("kind") != "same" or len(t["s"]) < 6:
            continue
        s, d = np.array(t["s"], float), np.array(t["d"], float)
        if s[-1] - s[0] < FORK_MIN_SPAN_M or abs(d.mean()) < 0.5 * lane_w or abs(d[-1]) < min_end_d:
            continue
        side = 1.0 if d[-1] > 0 else -1.0
        if np.polyfit(s, d, 1)[0] * side >= FORK_SLOPE:
            out.append((t, "left" if side > 0 else "right", float(s[0])))
            continue
        # Track may drive straight before or after the fork transition: look for any sustained outward-diverging segment
        # (>= FORK_MIN_SPAN_M, >= 6 pts, slope >= FORK_SLOPE, net outward shift >= 3.0 m) where the car stays outward.
        best_s = None
        for i in range(0, len(s) - 5):
            for j in range(i + 5, len(s)):
                if s[j] - s[i] < FORK_MIN_SPAN_M:
                    continue
                if (d[j] - d[i]) * side >= 3.0 and np.polyfit(s[i:j + 1], d[i:j + 1], 1)[0] * side >= FORK_SLOPE:
                    # Report where the car reaches the outer edge of the diverging transition (midpoint of ramp)
                    best_s = float(0.5 * (s[i] + s[j]))
                    break
            if best_s is not None:
                break
        if best_s is not None:
            out.append((t, "left" if side > 0 else "right", best_s))
    return out


def fork_from_tracks(tracks, lane_w):
    """A fork seen in the traffic itself: FORK_MIN_TRACKS or more cars on one side pulling away from the ego road. The split is
    where those cars began diverging (median). Returns {"side", "s_split", "ids"} or None."""
    import numpy as np
    div = diverging_tracks(tracks, lane_w)
    for side in ("left", "right"):
        ts = [(t, s_div) for t, sd, s_div in div if sd == side]
        if len(ts) >= FORK_MIN_TRACKS:
            return {"side": side, "s_split": float(np.median([s_div for _, s_div in ts])), "ids": [t["id"] for t, _ in ts]}
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


def apply_fork(prof, fork, tracks, lane_w, ego_j=None, base_lanes=None, branch_probe=None, video=None, duration=None, time_of=None):
    """Put a fork seen in the traffic into the lane profile: the place where lanes leave on the fork's side moves to the measured
    split, those lanes become a branch, climbing when the cars on it (or the VLM rise probe) show it. When `prof` has no
    existing lane-count drop (e.g. constant width or empty), synthesize a 2-section fork directly from the diverging tracks so
    confirmed diverging traffic always produces a diverging road. Returns a note, or None when `fork` is unusable."""
    import numpy as np
    from agents import lane_profile, road_probes
    side = fork["side"]
    drops = [k for k in range(1, len(prof)) if prof[k]["n"] < prof[k - 1]["n"]]
    if drops:
        k = min(drops, key=lambda i: abs(prof[i]["d0"] - fork["s_split"]))
        a, b = prof[k - 1], prof[k]
        if (side == "left") != (b["m"] > a["m"]):
            _move_leaving_lanes(prof, k, side)
        # Collapse a short (<= 35 m) same-side staircase step immediately before the split (common when the
        # outer lane leaves the camera FOV shortly before the gore) so all departing lanes branch together.
        while k >= 2:
            prev, cur, nxt = prof[k - 2], prof[k - 1], prof[k]
            prev_drop_side = (cur["m"] > prev["m"]) if side == "left" else (cur["m"] + cur["n"] < prev["m"] + prev["n"])
            if cur["n"] < prev["n"] and prev_drop_side and cur.get("d0") is not None and nxt.get("d0") is not None and (nxt["d0"] - cur["d0"]) <= 35.0:
                prev["d1"] = nxt["d0"]
                prof.pop(k - 1)
                k -= 1
                a, b = prof[k - 1], prof[k]
            else:
                break
        lo = (a["d0"] if a["d0"] is not None else 0.0) + 15.0
        hi = (b["d1"] - 15.0) if b["d1"] is not None else fork["s_split"]
        new_s = min(max(fork["s_split"], lo), max(hi, lo))
        a["d1"] = b["d0"] = round(new_s, 2)
        if ego_j is None:
            ego_j = b["m"] + b["n"] - 1                                  # unknown: assume the right-most lane
    else:
        if base_lanes is None:
            return None
        stay_n = max(1, min(int(base_lanes), lane_profile.MAX_LANES - 1))
        div_tracks = [t for t in tracks if t.get("id") in set(fork.get("ids") or []) and t.get("d")]
        d_spread = (max(abs(float(np.median(t["d"]))) for t in div_tracks) - min(abs(float(np.median(t["d"]))) for t in div_tracks)) if len(div_tracks) >= 2 else 0.0
        n_leave = max(1, min(2 if len(div_tracks) >= 2 and d_spread >= 0.7 * max(lane_w, 2.5) else 1, lane_profile.MAX_LANES - stay_n))
        if len(div_tracks) >= 3 and stay_n + 2 <= lane_profile.MAX_LANES:
            n_leave = max(n_leave, 2)
        new_s = round(max(float(fork["s_split"]), 25.0), 2)
        if side == "left":
            prof[:] = [
                {"d0": None, "d1": new_s, "n": stay_n + n_leave, "m": 0},
                {"d0": new_s, "d1": None, "n": stay_n, "m": n_leave},
            ]
        else:
            prof[:] = [
                {"d0": None, "d1": new_s, "n": stay_n + n_leave, "m": 0},
                {"d0": new_s, "d1": None, "n": stay_n, "m": 0},
            ]
        b = prof[1]
        if ego_j is None:
            ego_j = b["m"] + b["n"] - 1
    # lanes that stay beside the ego car on the fork's side: past them (in the video's lane widths) the cars are on the branch
    stay = (ego_j - b["m"]) if side == "left" else (b["m"] + b["n"] - 1 - ego_j)
    beyond = (max(stay, 0) + 0.5) * lane_w
    climbs, med, n = branch_climbs(tracks, side, beyond, fork["ids"])
    if not climbs and n < 2 and video and duration and time_of is not None:
        probe = branch_probe or road_probes.branch_evidence
        _, p_rise = probe(video, time_of(new_s), side, duration)
        if p_rise >= road_probes.RISE_P:
            climbs = True
    b["branch"] = {"side": side, "rise_m": 6.0 if climbs else 0.0}
    return "fork seen in the traffic: %d cars pull away on the %s from %.0f m; %s (ground/size distance %s over %d cars)" % (
        len(fork["ids"]), side, new_s, "the branch climbs" if climbs else "level branch", "-" if med is None else "%.2f" % med, n)


def _ego_lane_in_section(sec0, left0, right0, spec):
    """Pick ego's lane offset j within sec0 (0 = leftmost in sec0 .. sec0['n']-1 = rightmost), anchoring to an outer edge
    when zero lanes are measured on that side so ego never drifts inward leaving a phantom lane beside a barrier."""
    if not left0 and not right0:
        return None
    l_med = left0[len(left0) // 2] if left0 else None
    r_med = right0[len(right0) // 2] if right0 else None
    ego_lane = (spec.get("ego") or {}).get("lane")
    road_sp = spec.get("road") or {}
    if (r_med is not None and r_med < 0.5) or (ego_lane == "right" and road_sp.get("edge_right") == "barrier" and (r_med is None or r_med < 1.0)):
        if l_med is None or l_med >= 0.5 or sec0["n"] == 1:
            return sec0["n"] - 1
    if (l_med is not None and l_med < 0.5) and (r_med is not None and r_med >= 0.5):
        return 0
    if ego_lane == "left" and road_sp.get("edge_left") == "barrier" and (l_med is None or l_med < 1.0):
        return 0
    if l_med is not None:
        return min(int(round(l_med)), sec0["n"] - 1)
    return max(0, sec0["n"] - 1 - int(round(r_med)))


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
    segs = lane_profile.segment_counts(measured, switch_cost=4.0, min_dur_s=20.0)          # "t" is metres here
    if not segs:
        return False
    t_v0, T = sim_window(spec)
    need = 2 if any(a["relation"] == "beside" for a in spec["actors"]) else 1
    bounds = lane_profile.boundary_times(segs, measured)
    blocked = tuple(sd for sd, key in (("left", "edge_left"), ("right", "edge_right")) if road.get(key) == "barrier")
    sides = [lane_lines.change_side(measured, a["t1"], b["t0"], span=15.0,
                                    seg_before=(a["t0"], a["t1"]), seg_after=(b["t0"], b["t1"]), blocked_sides=blocked)
             for a, b in zip(segs, segs[1:])]
    prof = lane_profile.build_profile(segs, bounds, sides, 0.0, 1.0)
    d_ev = s_ego(t_v0 + T)
    left0 = sorted(x["lanes_left"] for x in measured if x["t"] <= 30.0)       # the very start is barely seen yet
    right0 = sorted(x["lanes_right"] for x in measured if x["t"] <= 30.0 and x.get("lanes_right") is not None)
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
    lane_w = next((x["lane_w"] for x in measured if x.get("lane_w")), 3.5)
    fork = fork_from_tracks(scene.get("tracks") or [], lane_w)
    if not prof and fork:
        n_base = max(only or next((sg["n"] for sg in segs if sg["t1"] >= d_ev), segs[-1]["n"]), need)
        bnote = apply_fork(prof, fork, scene["tracks"], lane_w, base_lanes=n_base,
                           branch_probe=branch_probe, video=video, duration=duration, time_of=time_of)
        if bnote and len(prof) >= 2:
            spec.setdefault("notes", []).append(bnote)
            log_fn(bnote)
            sec_ev = lane_profile.section_at(prof, d_ev)
            road["lane_profile"] = prof
            road["lanes_per_direction"] = max(sec_ev["n"], need)
            sec0 = lane_profile.section_at(prof, 0.0)
            if fork["side"] == "left":
                spec["ego"]["lane_index"] = sec0["m"] + sec0["n"] - 1 if spec["ego"]["lane"] == "right" else sec0["m"] + max(sec0["n"] - sec_ev["n"], 0)
            else:
                spec["ego"]["lane_index"] = sec0["m"] + (sec_ev["n"] - 1 if spec["ego"]["lane"] == "right" else 0)
            txt = ", ".join("%d lanes%s" % (sc["n"], "" if sc["d0"] is None else " from %.0f m" % sc["d0"]) for sc in prof)
            spec.setdefault("notes", []).append("lane map (with traffic fork): %s" % txt)
            log_fn("lane map (with traffic fork): " + txt)
            return True
    if not prof:
        n = only or next((sg["n"] for sg in segs if sg["t1"] >= d_ev), segs[-1]["n"])
        road["lanes_per_direction"] = max(n, need)
        road["lane_profile"] = None
        j_flat = _ego_lane_in_section({"n": road["lanes_per_direction"], "m": 0}, left0, right0, spec)
        if j_flat is not None:
            spec["ego"]["lane_index"] = j_flat
        spec.setdefault("notes", []).append("lane map: %d lanes along the simulated stretch" % road["lanes_per_direction"])
        return True
    def _ensure_min_lanes(sec, req):
        if sec["n"] < req:
            grow = req - sec["n"]
            if sec["m"] > 0:
                shift = min(grow, sec["m"])
                sec["m"] -= shift
            sec["n"] = req

    sec_ev = lane_profile.section_at(prof, d_ev)
    _ensure_min_lanes(sec_ev, need)
    sec0_ = lane_profile.section_at(prof, 0.0) if prof else None
    j0 = _ego_lane_in_section(sec0_, left0, right0, spec) if sec0_ else None
    ego_j = (sec0_["m"] + j0) if (j0 is not None and sec0_) else None
    bnote = apply_fork(prof, fork, scene["tracks"], lane_w, ego_j,
                       base_lanes=max(sec_ev["n"], need), branch_probe=branch_probe,
                       video=video, duration=duration, time_of=time_of) if fork and len(prof) > 1 else None
    if bnote is None:                                                            # else ask the vision model
        bnote = tag_branches(prof, t_v0, 0.0, video, duration, branch_probe, time_of=time_of, blocked_sides=blocked)
    _clamp_post_branch(prof)
    sec_ev = lane_profile.section_at(prof, d_ev)
    _ensure_min_lanes(sec_ev, need)
    if bnote:
        spec.setdefault("notes", []).append(bnote)
        log_fn(bnote)
    road["lane_profile"] = prof
    road["lanes_per_direction"] = sec_ev["n"]
    sec0 = lane_profile.section_at(prof, 0.0)
    j = _ego_lane_in_section(sec0, left0, right0, spec)
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
    lane_w = next((x["lane_w"] for x in scene.get("lane_samples") or [] if x.get("lane_w")), None)
    k_d = 3.5 / lane_w if lane_w else 1.0          # positions across the road in simulation lanes (3.5 m)
    t_v0, T = sim_window(spec)
    conv = None
    if not third and t_crash is not None and c1 is not None and (c1.get("relation") == "beside" or c1.get("behavior") == "cutting_in"):
        conv = vehicle_tracks.converging_cutin_track(tracks, float(t_crash), side=c1.get("lane", "left"),
                                                     color=color, lane_w=lane_w or 3.5)
    if conv is not None:
        skip = set(conv["skip_ids"])
        tr_crash = conv["track"]
        lane_off = max(1, int(round(conv["d_start"] * k_d / 3.5)))
        v_meas = conv["v_meas"]
        t_sim_ev = round(conv["t_impact"] - t_v0, 2)
        c_msg = "cut-in track matched: ids %s, starts %.1f m (%d lane(s)) from ego, impact at video %.1f s (sim %.1f s)" % (
            conv["skip_ids"], conv["d_start"], lane_off, conv["t_impact"], t_sim_ev)
        spec.setdefault("notes", []).append(c_msg)
        log_fn(c_msg)
    else:
        skip = set(vehicle_tracks.crash_tracks(tracks, float(t_crash), ego_involved=not third, color=color)) if t_crash is not None else set()
        tr_crash = next((tr for tr in tracks if tr["id"] in skip), None)
        lane_off = max(1, int(round(abs(tr_crash["d"][0]) * k_d / 3.5))) if (tr_crash and tr_crash.get("d")) else 1
        v_meas = float(np.median([abs(v) for v in tr_crash["v"]])) if (tr_crash and tr_crash.get("v")) else None
        t_sim_ev = None
    if not third and skip and c1 is not None and (c1.get("relation") == "beside" or c1.get("behavior") == "cutting_in"):
        sp_prof = spec["ego"].get("speed_profile")
        t_eval = t_sim_ev if (t_sim_ev is not None and 2.5 <= t_sim_ev <= 12.0) else T
        v_ego_ev = (float(np.interp(t_eval, [p[0] for p in sp_prof], [p[1] for p in sp_prof])) / 3.6) if sp_prof else (float(spec["ego"]["speed_kmh"]) / 3.6)
        ratio = 0.90 if conv is not None else 0.75
        overtake = bool(tr_crash and tr_crash.get("kind") == "same" and v_meas is not None and v_ego_ev > 5.0 and v_meas < ratio * v_ego_ev)
        side_strike = bool(conv and conv.get("side_strike"))
        if overtake or lane_off >= 2 or side_strike or (t_sim_ev is not None and 2.5 <= t_sim_ev <= 12.0):
            v_kmh = round(max((v_meas or 0.0) * 3.6, 12.0), 1) if overtake else round(float(c1["speed_kmh"]), 1)
            if overtake:
                c1["speed_kmh"] = v_kmh
            meas = {"speed_kmh": v_kmh, "overtake": overtake}
            if lane_off >= 2:
                meas["lane_offset"] = lane_off
            if t_sim_ev is not None and 2.5 <= t_sim_ev <= 12.0:
                meas["t_event_s"] = t_sim_ev
                if conv is not None and abs(float(conv["t_impact"]) - float(ev.get("time_s", 0.0))) >= 0.4:
                    ev["time_s"] = round(float(conv["t_impact"]), 2)
                if conv is not None and abs(float(conv["t_impact"]) - float(t_crash)) >= 0.8 and tr_crash:
                    if tr_crash.get("type") in ("car", "truck", "motorcycle", "bike"):
                        c1["type"] = tr_crash["type"]
                    if not c1.get("color") and tr_crash.get("color") in COLORS:
                        c1["color"] = "%d,%d,%d" % COLORS[tr_crash["color"]]
            if side_strike:
                meas["side_strike"] = True
                if ev.get("type") == "collision":
                    ev["contact_parts"] = (
                        {"ego": "left_side", "c1": "front_right"}
                        if c1.get("lane") != "right"
                        else {"ego": "right_side", "c1": "front_left"}
                    )
            c1["measured"] = meas
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
        dur_tr = float(tr["last_t"]) - float(tr["first_t"])
        v_tr_med = float(np.median([abs(v) for v in tr["v"]])) if tr.get("v") else 0.0
        if tr["kind"] != "parked" and dur_tr < 0.6 and v_tr_med > 25.0:
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
    keep_b = []
    if spec["ego"].get("behavior") == "stopped":
        has_rep_ahead = any(
            r["path"] and abs(float(np.median([p[2] for p in r["path"]]))) < 1.8
            and min(float(p[1]) for p in r["path"]) > 14.0
            for r in rep
        )
        if not has_rep_ahead:
            keep_b = [a for a in spec["actors"]
                      if a["id"].startswith("b") and a.get("relation") == "ahead"
                      and a.get("lane") == "same" and a.get("behavior") == "stopped"][:1]
    spec["actors"] = [a for a in spec["actors"] if not a["id"].startswith("b")] + keep_b
    if isinstance(spec.get("traffic"), dict):
        for k in ("ego_lane_ahead", "left_lane", "right_lane", "oncoming"):
            spec["traffic"][k] = 0
    msg = "replaying %d measured vehicles (crash vehicles excluded: %s)" % (len(rep), sorted(skip) or "none matched")
    spec.setdefault("notes", []).append(msg)
    log_fn(msg)


def add_lane_profile(spec, video, duration, t_crash=None, samples=None, step=0.5, log_fn=log, branch_probe=None, caps=None):
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
    blocked = tuple(sd for sd, key in (("left", "edge_left"), ("right", "edge_right")) if road.get(key) == "barrier")
    sides = [lane_lines.change_side(measured, a["t1"], b["t0"],
                                    seg_before=(a["t0"], a["t1"]), seg_after=(b["t0"], b["t1"]), blocked_sides=blocked)
             for a, b in zip(segs, segs[1:])]
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
    bnote = tag_branches(prof, t_v0, v, video, duration, branch_probe, blocked_sides=blocked)
    _clamp_post_branch(prof)
    sec_ev = lane_profile.section_at(prof, v * T)
    if sec_ev["n"] < need:
        sec_ev["n"] = need
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


def _is_genuine_oncoming(tr, spec, lane_w=3.5):
    """True when a measured 'oncoming' track is actually fast traffic on the opposite carriageway (left of the ego road)
    rather than a slow/jittery queue car inside the ego carriageway, a right frontage road, or a post-crash spin artifact."""
    if tr.get("kind") != "oncoming":
        return False
    if not tr.get("d") and not tr.get("v"):
        return True
    import numpy as np
    ev = spec.get("event") or {}
    ego = spec.get("ego") or {}
    if ev.get("type") == "collision" and not ev.get("actor2_id") and ego.get("behavior") != "stopped":
        t_ev = ev.get("time_s")
        if t_ev is not None and tr.get("first_t") is not None and float(tr["first_t"]) > float(t_ev) + 2.0:
            return False
    if tr.get("v"):
        v_med = float(np.median([abs(v) for v in tr["v"]]))
        if v_med < 5.0:
            return False
    if tr.get("d"):
        d_med = float(np.median(tr["d"]))
        road = spec.get("road") or {}
        prof = road.get("lane_profile")
        if prof:
            from agents import lane_profile
            s_med = float(np.median(tr["s"])) if tr.get("s") else 0.0
            sc = lane_profile.section_at(prof, s_med)
            idx = ego.get("lane_index")
            if idx is None:
                ego_j = sc["m"] + sc["n"] - 1 if ego.get("lane", "right") == "right" else sc["m"]
            else:
                ego_j = max(sc["m"], min(int(idx), sc["m"] + sc["n"] - 1))
            left_lanes = max(ego_j - sc["m"], 0)
        else:
            L = int(road.get("lanes_per_direction", 2) or 2)
            left_lanes = (L - 1) if (ego.get("lane", "right") == "right" and L > 1) else 0
        d_left_edge = (left_lanes + 0.5) * float(lane_w or 3.5)
        if d_med < d_left_edge - 1.0:
            return False
    return True


def reconcile_one_way(spec, yellow_left=0.0, tracks=None, lane_w=3.5, yellow_px=0.0):
    """Reconcile straight-road one-way status bidirectionally:
    - If marked one-way despite yellow centerline evidence (VLM or HSV paint) or genuine oncoming traffic/tracks, flip to two-way.
    - If marked two-way with no yellow centerline, zero genuine oncoming traffic/tracks, and strong highway cues (barrier edge,
      diverging highway branch, or >=3 same-direction tracks), promote to one-way."""
    road = spec.get("road") or {}
    if road.get("type") != "straight":
        return spec
    tr_list = tracks or []
    n_same = sum(1 for t in tr_list if t.get("kind") == "same")
    n_onc = sum(1 for t in tr_list if _is_genuine_oncoming(t, spec, lane_w=lane_w))
    has_yellow = float(yellow_left or 0.0) >= 0.15 or float(yellow_px or 0.0) >= 0.10
    traffic = spec.get("traffic") or {}
    has_traffic_onc = int(traffic.get("oncoming", 0) or 0) > 0
    has_actor_onc = any(a.get("relation") == "oncoming" for a in spec.get("actors") or [])
    if road.get("one_way"):
        if has_yellow or n_onc > 0 or has_actor_onc:
            road["one_way"] = False
            onc_ids = {"r%d" % t["id"] for t in tr_list if "id" in t and _is_genuine_oncoming(t, spec, lane_w=lane_w)}
            for r in spec.get("replay") or []:
                if r.get("id") in onc_ids and r.get("kind") == "parked":
                    r["kind"] = "oncoming"
            reasons = []
            if has_yellow:
                reasons.append("yellow centerline %.2f/%.2f" % (float(yellow_left or 0.0), float(yellow_px or 0.0)))
            if n_onc > 0:
                reasons.append("%d oncoming track(s)" % n_onc)
            if has_actor_onc:
                reasons.append("oncoming actor")
            spec.setdefault("notes", []).append("road set to two-way (%s)" % ", ".join(reasons))
        return spec
    if has_yellow or has_traffic_onc or has_actor_onc or n_onc > 0:
        return spec
    has_branch = any(isinstance(sc, dict) and sc.get("branch") for sc in (road.get("lane_profile") or []))
    has_barrier = road.get("edge_right") == "barrier" or road.get("edge_left") == "barrier"
    if has_branch or has_barrier or n_same >= 3:
        road["one_way"] = True
        for r in spec.get("replay") or []:
            if r.get("kind") == "oncoming":
                r["kind"] = "parked"
        reasons = []
        if has_branch:
            reasons.append("diverging highway branch")
        if has_barrier:
            reasons.append("barrier edge")
        if n_same >= 3:
            reasons.append("%d same-direction tracks, 0 oncoming" % n_same)
        spec.setdefault("notes", []).append("road set to one-way (%s; yellow_left=%.2f)" % (", ".join(reasons), float(yellow_left or 0.0)))
    return spec


def measure_crossroad_approach_traffic(spec, video, duration, t_crash, log_fn=log):
    """Count distinct queued/same-direction vehicles in adjacent lanes during the pre-crash approach to a crossroad
    using calibrated YOLO tracks (letterbox-cropped), raising uncovered spec['traffic'] bins so stopped queues blocking
    sightlines are faithfully populated."""
    if (spec.get("road") or {}).get("type") != "crossroad" or t_crash is None:
        return spec
    try:
        import numpy as np
        from agents import camera_geom, lane_lines, vehicle_tracks
        tc = float(t_crash)
        t_hi = max(0.5, min(float(duration) - 0.1, tc - 0.4))
        t_lo = max(0.0, tc - 4.5)
        if t_hi <= t_lo + 0.3:
            return spec
        y_b, h = lane_lines.road_bottom(video)
        if y_b is None or h is None:
            return spec
        lane_rows = lane_lines.lane_timeline(video, t_lo, t_hi)
        cam, _ = camera_geom.calibrate(lane_rows, (h, lane_lines.WIDTH))
        v0 = 0.0 if spec["ego"].get("behavior") == "stopped" else float(spec["ego"].get("speed_kmh", 50.0)) / 3.6
        s_ego = camera_geom.travel([(t_lo, v0), (t_hi, v0)], t_lo)
        vx_at = camera_geom.vx_series(lane_rows, cam.vx)
        raw = vehicle_tracks.detect_tracks(video, t_lo, t_hi, lane_lines.WIDTH)
        tracks = vehicle_tracks.to_road_tracks(raw, cam, s_ego, y_b, vx_at=vx_at)
        if not tracks:
            return spec
        L = int((spec.get("road") or {}).get("lanes_per_direction", 2) or 2)
        ego_lane = (spec.get("ego") or {}).get("lane", "right")
        left_lanes = (L - 1) if (ego_lane == "right" and L > 1) else 0
        right_lanes = (L - 1) if (ego_lane == "left" and L > 1) else 0
        lane_w = 3.5
        s_bins = {"left_lane": [], "right_lane": [], "ego_lane_ahead": []}
        for tr in tracks:
            if tr.get("kind") not in ("same", "parked") or len(tr.get("t") or []) < 3:
                continue
            ds, ss, zs = tr.get("d") or [], tr.get("s") or [], tr.get("z") or []
            if not ds or not ss or not zs:
                continue
            if min(zs) > 45.0 or max(zs) < 3.5:
                continue
            d_med = float(np.median(ds))
            s_med = float(np.median(ss))
            if left_lanes >= 1 and 1.6 <= d_med <= (left_lanes + 0.5) * lane_w + 1.5:
                s_bins["left_lane"].append(s_med)
            elif right_lanes >= 1 and -((right_lanes + 0.5) * lane_w + 1.5) <= d_med <= -1.6:
                s_bins["right_lane"].append(s_med)
            elif abs(d_med) < 1.6:
                s_bins["ego_lane_ahead"].append(s_med)

        def _distinct(vals, min_gap=4.5):
            if not vals:
                return 0
            vals = sorted(vals)
            kept = [vals[0]]
            for v in vals[1:]:
                if v - kept[-1] >= min_gap:
                    kept.append(v)
            return len(kept)

        covered_b = set()
        for a in spec.get("actors") or []:
            if not str(a.get("id", "")).startswith("b"):
                continue
            if a.get("relation") == "ahead":
                covered_b.add({"same": "ego_lane_ahead", "left": "left_lane", "right": "right_lane"}.get(a.get("lane")))
        traffic = spec.setdefault("traffic", {"ego_lane_ahead": 0, "left_lane": 0, "right_lane": 0, "oncoming": 0, "rel_speed_kmh": 0.0})
        counts = {k: min(_distinct(vs), 6) for k, vs in s_bins.items()}
        raised = {}
        for k, cnt in counts.items():
            if cnt > 0 and k not in covered_b and cnt > int(traffic.get(k, 0) or 0):
                traffic[k] = cnt
                raised[k] = cnt
        if raised:
            msg = "crossroad approach traffic from YOLO (%d tracks): raised %s" % (len(tracks), raised)
            spec.setdefault("notes", []).append(msg)
            log_fn(msg)
    except Exception as e:
        log_fn("crossroad approach traffic unavailable: %s" % e)
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
            backfill_caption_speeds(caps)
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
            t_pre = min(dur, max(float(t_crash) - 0.3, 2.0)) if (t_crash is not None and float(t_crash) >= 2.5) else t_active
            tb = road_probes.traffic_bins(video, dur, t_crash, t1=t_pre)
            if tb:
                spec["traffic"] = tb
                spec.setdefault("notes", []).append("traffic measured from the frames: %s" % {k: v for k, v in tb.items() if k != "rel_speed_kmh"})
            edges = road_probes.road_edges(video, dur, t1=t_active)
            if edges:
                spec["road"].update(edges)
                spec.setdefault("notes", []).append("road edges from the frames: right %s, left %s, %s" % (
                    edges["edge_right"], edges["edge_left"], "one-way" if edges["one_way"] else "two-way"))
            yellow_left = road_probes.yellow_centerline_left(video, dur, t_crash=t_crash)
            yellow_px = road_probes.yellow_line_left_score(video, dur, t_crash=t_crash)
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
            env_est = road_probes.estimate_environment(video, dur)
            if env_est:
                spec["environment"] = env_est
                spec.setdefault("notes", []).append("environment from frames: %s, %s" % (env_est["time_of_day"], env_est["weather"]))
            scene = None
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
                    lane_w_sc = next((x["lane_w"] for x in (scene.get("lane_samples") or []) if x.get("lane_w")), 3.5)
                    reconcile_one_way(spec, yellow_left=yellow_left, tracks=scene.get("tracks"), lane_w=lane_w_sc, yellow_px=yellow_px)
                    if scene.get("tracks"):
                        apply_replay(spec, scene, t_crash)
                t_stages["measured_scene"] = time.time() - ts
                if not from_map:                       # no tracker / lane model: lane count from the per-window detector
                    ts = time.time()
                    lane_samples = add_lane_profile(spec, video, dur, t_crash=t_crash,
                                                    samples=_cached(out, "lane_timeline.json", reuse), caps=caps)
                    if lane_samples:
                        _save(lane_samples, os.path.join(out, "lane_timeline.json"))
                    t_stages["lane_profile"] = time.time() - ts
                lane_w_sc = next((x["lane_w"] for x in ((scene or {}).get("lane_samples") or []) if x.get("lane_w")), 3.5)
                reconcile_one_way(spec, yellow_left=yellow_left, tracks=(scene or {}).get("tracks"), lane_w=lane_w_sc, yellow_px=yellow_px)
                reconcile_traffic(spec)
            elif spec["road"]["type"] == "crossroad":
                measure_crossroad_approach_traffic(spec, video, dur, t_crash, log_fn=log)
                reconcile_traffic(spec)
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


def _road_map_signature(out_dir, scene_id=SCENE_ID):
    """Stable hash of the deterministic SUMO road geometry inputs (.nod/.edg/.con) + OpenDRIVE road plan so
    timing variants on the exact same road reuse the already-loaded CARLA OpenDRIVE world (ignoring netconvert's
    non-deterministic timestamps and input file paths inside .net.xml)."""
    import hashlib
    h = hashlib.sha256()
    for suffix in (".nod.xml", ".edg.xml", ".con.xml"):
        fn = os.path.join(out_dir, scene_id + suffix)
        if os.path.exists(fn):
            h.update(suffix.encode("utf-8"))
            with open(fn, "rb") as f:
                h.update(f.read())
    plan_fn = os.path.join(out_dir, scene_id + "_motion.json")
    if os.path.exists(plan_fn):
        with open(plan_fn, encoding="utf-8") as f:
            road = json.load(f).get("road") or {}
        h.update(json.dumps(road, sort_keys=True).encode("utf-8"))
    return h.hexdigest()


def _connect_carla():
    import carla
    for _ in range(6):                       # the server can be busy right after a map change: wait up to ~2 minutes
        try:
            c = carla.Client("localhost", 2000)
            c.set_timeout(20)
            c.get_world()
            return c
        except Exception:
            time.sleep(5)
    return None


def phase2(out, max_iter=8, record_video=False, camera="chase", dashcam=False, dashcam_mode="colliders",
           ego_on_collider=False, client=None, skip_map_load=False):
    from agents import motion_runner, scenario_check, xosc_export
    c = client or _connect_carla()
    if c is None:
        log("CARLA is not reachable: start the editor, press Play, then run again with --sim-only")
        return None

    fallback = None if skip_map_load else _load_map_with_fallback(c, out)

    def run_fn(load_map):                          # the map is already loaded (once, above); every run starts with a clean cleanup
        return motion_runner.run_motion(
            out, SCENE_ID, load_map=False, hold_sec=2.0,
            record_video=record_video, camera=camera, dashcam=dashcam, dashcam_mode=dashcam_mode,
        )
    ready = scenario_check.search(out, SCENE_ID, run_fn, max_iter=max_iter, log=log)
    if fallback:
        ready["map_fallback"] = fallback
    try:
        xsd = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tests", "fixtures", "OpenSCENARIO.xsd")
        try:
            import xmlschema   # noqa: F401
            xsd_use = xsd if os.path.exists(xsd) else None
        except ImportError:
            xsd_use = None
        ego_override = None
        if ego_on_collider:
            da = ready.get("dashcam_actors") or (ready.get("expect") or {}).get("actors") or []
            if da:
                ego_override = da[0]
        xosc_path, _ = xosc_export.export_scene(out, SCENE_ID, xsd_path=xsd_use, ego_actor=ego_override)
        ready["xosc"] = os.path.basename(str(xosc_path))
        log("exported OpenSCENARIO -> %s%s" % (xosc_path, " (XSD validated)" if xsd_use else ""))
    except Exception as e:
        ready["xosc_error"] = str(e)
        log("OpenSCENARIO export failed: %s" % e)
    _save(ready, os.path.join(out, SCENE_ID + "_ready.json"))
    return ready


def phase2_variants(out, manifest=None, max_iter=8, record_video=False, camera="chase",
                    dashcam=False, dashcam_mode="colliders", ego_on_collider=False):
    """Simulate every generated variant in `out/variants/` in CARLA, reusing the loaded OpenDRIVE map across
    timing variants that share the exact same road geometry."""
    import shutil
    from agents import scenario_variants

    man_fn = os.path.join(out, "variants_manifest.json")
    if manifest is None:
        manifest = _load(man_fn)
    c = _connect_carla()
    if c is None:
        log("CARLA is not reachable: start the editor, press Play, then run again with --sim-only")
        return manifest

    last_sig = None
    last_xodr = None
    variants = manifest.get("variants") or []
    for idx, entry in enumerate(variants, start=1):
        vdir = entry["dir"] if os.path.isabs(entry.get("dir", "")) else os.path.join(out, entry["rel_dir"])
        sig = _road_map_signature(vdir, entry.get("scene_id", SCENE_ID))
        reuse_map = bool(last_sig is not None and sig == last_sig and last_xodr and os.path.exists(last_xodr))
        if reuse_map:
            dst_xodr = os.path.join(vdir, entry.get("scene_id", SCENE_ID) + ".xodr")
            if os.path.abspath(last_xodr) != os.path.abspath(dst_xodr):
                shutil.copyfile(last_xodr, dst_xodr)
        log("simulating variant %d/%d: %s%s" % (idx, len(variants), entry["id"], " (reusing loaded map)" if reuse_map else ""))
        ready = phase2(
            vdir, max_iter=max_iter, record_video=record_video, camera=camera,
            dashcam=dashcam, dashcam_mode=dashcam_mode, ego_on_collider=ego_on_collider,
            client=c, skip_map_load=reuse_map,
        )
        if not reuse_map:
            last_sig = _road_map_signature(vdir, entry.get("scene_id", SCENE_ID))
            cand_xodr = os.path.join(vdir, entry.get("scene_id", SCENE_ID) + ".xodr")
            last_xodr = cand_xodr if os.path.exists(cand_xodr) else None
        spec_v = _load(os.path.join(vdir, entry.get("scene_id", SCENE_ID) + "_spec.json"))
        report(vdir, spec_v, ready)
        if ready:
            entry["sim_status"] = ready.get("status")
            entry["sim_iterations"] = ready.get("iterations")
            if ready.get("xosc"):
                entry["xosc"] = ready["xosc"]
            if ready.get("dashcam_actors"):
                entry["dashcam_actors"] = ready["dashcam_actors"]
            if ready.get("dashcam_videos"):
                entry["dashcam_videos"] = ready["dashcam_videos"]
            if ready.get("video_path"):
                entry["video_path"] = ready["video_path"]
    _save(manifest, man_fn)
    rep = scenario_variants.format_variants_report(manifest)
    with open(os.path.join(out, "variants_report.txt"), "w", encoding="utf-8") as f:
        f.write(rep)
    return manifest


def report(out, spec, ready):
    lines = []
    if spec:
        if isinstance(spec.get("variant"), dict):
            v = spec["variant"]
            env_v = v.get("environment") or spec.get("environment") or {}
            lines.append("Variant: %s (road=%s, timing=%s, env=%s/%s, expected=%s)" % (
                v.get("id", "-"), v.get("road", "-"), v.get("timing", "-"),
                env_v.get("time_of_day", "day"), env_v.get("weather", "clear"), v.get("expected_outcome", "-")))
        lines.append("Scene: %s road, %d lanes per direction; ego %s at %.0f km/h (%s, %s lane)" % (
            spec["road"]["type"], spec["road"]["lanes_per_direction"], spec["ego"]["type"], spec["ego"]["speed_kmh"],
            spec["ego"]["behavior"], spec["ego"]["lane"]))
        for a in spec["actors"]:
            lines.append("  actor %s: %s %s, %s %s lane %s, %s, %.0f m, %.0f km/h" % (
                a["id"], a.get("color") or "", a["type"], a["relation"], a["lane"], "", a["behavior"], a["distance_m"], a["speed_kmh"]))
        ev = spec["event"]
        lines.append("Event: %s at %.1fs (%s) %s" % (ev["type"], ev["time_s"], ", ".join(x for x in (ev.get("actor_id"), ev.get("actor2_id")) if x) or "no actor", ev["description"]))
        if ev.get("contact_parts"):
            lines.append("  expected contact parts: " + ", ".join("%s=%s" % (k, v) for k, v in ev["contact_parts"].items()))
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
        if last.get("contact_parts"):
            lines.append("  simulated contact parts: " + ", ".join("%s=%s" % (k, v) for k, v in last["contact_parts"].items()) + " at %.1f s" % (last.get("contact_t_sim") or 0.0))
        for r in last.get("reasons", []):
            lines.append("  failed rule: " + r)
        if ready.get("dashcam_actors"):
            lines.append("  dashcam on colliding vehicle(s): " + ", ".join(ready["dashcam_actors"]))
        if ready.get("dashcam_videos"):
            lines.append("  dashcam videos: " + ", ".join("%s=%s" % (k, os.path.basename(v)) for k, v in ready["dashcam_videos"].items()))
        if ready.get("map_fallback"):
            lines.append("  note: " + ready["map_fallback"])
        if ready.get("xosc"):
            lines.append("  exported: " + ready["xosc"])
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
    ap.add_argument("--record", action="store_true", help="record CARLA camera video for the simulation")
    ap.add_argument("--camera", choices=("chase", "dashcam", "both", "dashcam_all"), default="chase", help="camera view when --record is set")
    ap.add_argument("--dashcam", action="store_true", help="record dashcam footage mounted on the colliding vehicle(s) (never a bystander)")
    ap.add_argument("--dashcam-mode", default="colliders", help="dashcam target: colliders (default, all colliding vehicles), primary, or actor id")
    ap.add_argument("--ego-on-collider", action="store_true", help="export the primary colliding vehicle as OpenSCENARIO 'hero'")
    ap.add_argument("--timing-variants", default=None,
                    help="generate timing variants: count (e.g. 4), 'all', or comma-separated presets (original,earlier_collision,later_collision,fast_collision,slow_collision,near_miss_early,near_miss_late,miss_early,miss_late,no_aggressor)")
    ap.add_argument("--crashing-only", "--crash-only", dest="crashing_only", action="store_true",
                    help="when changing timings, produce ONLY crashing scenarios (no misses, near-misses, or aggressor removal)")
    ap.add_argument("--road-variants", default=None,
                    help="generate road-design variants: count (e.g. 4), 'all', or comma-separated presets (original,straight_2lane_twoway,straight_3lane_highway,wide_4lane_highway,curve_left,curve_right,s_curve,lane_drop,highway_fork,crossroad_1lane,crossroad_2lane,crossroad_3lane)")
    ap.add_argument("--vary-weather", action="store_true",
                    help="when changing maps/roads, automatically vary weather and time-of-day (with vehicle headlights at night/rain/fog)")
    ap.add_argument("--weather-variants", default=None,
                    help="explicit weather/time presets: count (e.g. 4), 'all', or comma-separated presets (original,day_rain,dusk_cloudy,night_clear,day_fog,night_rain,dawn_fog,day_storm,... or <time>:<weather>)")
    ap.add_argument("--variant-combine", choices=("product", "zip"), default="product",
                    help="combine road, timing, and explicit weather variants as Cartesian 'product' (default) or 1-to-1 'zip'")
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    want_variants = bool(a.timing_variants or a.road_variants or a.vary_weather or a.weather_variants)
    man_fn = os.path.join(a.out, "variants_manifest.json")
    spec = ready = manifest = None
    if not a.sim_only:
        if not a.video and not a.spec:
            ap.error("a video (or --spec) is required")
        spec, _ = phase1(a.video, a.out, a.spec, a.fps, not a.no_assume_crash, a.reuse)
        if want_variants:
            from agents import scenario_variants
            manifest = scenario_variants.build_variants(
                spec, a.out, road_presets=a.road_variants, timing_presets=a.timing_variants,
                combine=a.variant_combine, crashing_only=a.crashing_only,
                weather_presets=a.weather_variants, vary_weather=a.vary_weather,
                scene_id=SCENE_ID, log_fn=log,
            )
    else:
        spec_fn = os.path.join(a.out, SCENE_ID + "_spec.json")
        if os.path.exists(spec_fn):
            spec = _load(spec_fn)
        if want_variants and spec is not None:
            from agents import scenario_variants
            manifest = scenario_variants.build_variants(
                spec, a.out, road_presets=a.road_variants, timing_presets=a.timing_variants,
                combine=a.variant_combine, crashing_only=a.crashing_only,
                weather_presets=a.weather_variants, vary_weather=a.vary_weather,
                scene_id=SCENE_ID, log_fn=log,
            )
        elif os.path.exists(man_fn):
            manifest = _load(man_fn)
    if manifest is not None:
        from agents import scenario_variants
        if not a.no_sim:
            manifest = phase2_variants(
                a.out, manifest=manifest, max_iter=a.max_iter,
                record_video=a.record, camera=a.camera,
                dashcam=a.dashcam, dashcam_mode=a.dashcam_mode,
                ego_on_collider=a.ego_on_collider,
            )
        print(scenario_variants.format_variants_report(manifest))
        return
    if not a.no_sim:
        ready = phase2(
            a.out, a.max_iter, record_video=a.record, camera=a.camera,
            dashcam=a.dashcam, dashcam_mode=a.dashcam_mode, ego_on_collider=a.ego_on_collider,
        )
    print(report(a.out, spec, ready))


if __name__ == "__main__":
    main()
