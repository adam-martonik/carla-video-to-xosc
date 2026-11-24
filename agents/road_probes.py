"""
Frame probes that read the road and the traffic from the video with the local model's multiple-choice probabilities, and the
coarse-to-fine crash-time search. Every probe returns None when the model is not available, so the caller keeps its defaults.
"""
from agents import llm_backend, video_events

BIN_LETTERS = "ABCD"
BIN_COUNTS = (0, 2, 4, 8)
BIN_OPTIONS = "\nA) none\nB) 1 or 2\nC) 3 to 5\nD) 6 or more"
CTX = "A frame from a dashcam video. The camera is in the ego vehicle (the car that carries the camera, its hood is visible). "
LANE_QUESTIONS = {
    "ego_lane_ahead": CTX + "How many vehicles are in the ego vehicle's own lane, ahead of it?",
    "left_lane": CTX + "How many vehicles are in the lane or lanes to the LEFT of the ego vehicle, driving the same way as it "
                       "(beside it or ahead; choose A if there is no lane to the left or no vehicles there)?",
    "right_lane": CTX + "How many vehicles are in a lane to the RIGHT of the ego vehicle, driving the same way as it "
                        "(choose A if the ego vehicle is already in the right-most lane or there are none)?",
    "oncoming": CTX + "How many oncoming vehicles are driving toward the camera in opposite-direction lanes "
                      "(choose A if all visible vehicles drive the same way as the ego vehicle)?",
}
RIGHT_EDGE_Q = (CTX + "Look at the RIGHT edge of the road, beside the right-most lane of the ego vehicle's driving direction. "
                "What forms the outer boundary there?\n"
                "A) a concrete barrier, jersey wall, retaining wall or metal guardrail (choose A whenever a wall or barrier runs along the right edge, even if a white edge line or narrow shoulder strip lies in front of it)\n"
                "B) a curb, sidewalk or wide paved shoulder with NO concrete wall, barrier or guardrail\n"
                "C) open ground, grass, dirt or sand with NO wall, barrier or guardrail")
LEFT_EDGE_Q = (CTX + "Look at the LEFT edge of the road, beside the left-most lane of the ego vehicle's driving direction. "
               "What is there?\n"
               "A) a concrete barrier, jersey wall or metal guardrail (even if a narrow shoulder line is in front of it)\n"
               "B) lanes with oncoming traffic driving toward the camera or a painted yellow median\n"
               "C) a curb, sidewalk or paved shoulder (with no barrier or wall)\n"
               "D) open ground, grass, dirt or sand (with no barrier or wall)")
ONCOMING_Q = (CTX + "Is there a separate set of lanes next to the ego vehicle's road with oncoming vehicles driving the opposite way "
              "(toward the camera, not the same way as the ego vehicle)? Answer YES or NO.")
CURVE_Q = (CTX + "Looking at the road far ahead, does the road bend?\nA) it bends to the left\nB) it bends to the right\nC) it is straight")
CURVE_KEYS = {"A": "left", "B": "right", "C": "straight"}
YELLOW_LEFT_Q = (
    "A dashcam frame from the ego vehicle. Look immediately to the LEFT edge of the ego vehicle's own lane. "
    "Is a yellow center line (dividing opposite traffic) directly next to the left side of the ego vehicle's lane? Answer YES or NO."
)
MIN_CHOICE_PROB = 0.34


def _times(duration, n, t0=None, t1=None, avoid=None, margin=1.5):
    t0 = 0.0 if t0 is None else t0
    t1 = duration if t1 is None else t1
    times = [round(t0 + (t1 - t0) * (i + 0.5) / n, 2) for i in range(n)]
    if avoid is not None:
        kept = [t for t in times if abs(t - avoid) >= margin]
        times = kept or times
    return times


def _avg_choice(video, times, question, letters):
    acc = {k: 0.0 for k in letters}
    used = 0
    for t in times:
        fr = video_events.grab_frames(video, [t], width=512)
        if not fr:
            continue
        p = llm_backend.choice_probabilities(question, fr, letters)
        for k in letters:
            acc[k] += p[k]
        used += 1
    if not used:
        raise RuntimeError("no frames could be read")
    return {k: v / used for k, v in acc.items()}


def traffic_bins(video, duration, t_crash, n=10, t0=None, t1=None):
    """Vehicles per lane group from the probabilities of four count bins, away from the crash moment.
    {"ego_lane_ahead", "left_lane", "right_lane", "oncoming": int, "rel_speed_kmh": 0.0} or None."""
    try:
        if t1 is None and t_crash is not None:
            t1 = min(duration, max(float(t_crash) + 2.0, 4.0))
        times = _times(duration, n, t0=t0, t1=t1, avoid=t_crash)
        acc = {lane: {k: 0.0 for k in BIN_LETTERS} for lane in LANE_QUESTIONS}
        used = 0
        for t in times:
            fr = video_events.grab_frames(video, [t], width=512)
            if not fr:
                continue
            for lane, q in LANE_QUESTIONS.items():
                p = llm_backend.choice_probabilities(q + BIN_OPTIONS, fr, BIN_LETTERS)
                for k in BIN_LETTERS:
                    acc[lane][k] += p[k]
            used += 1
        if not used:
            return None
        out = {}
        for lane in LANE_QUESTIONS:
            p = {k: v / used for k, v in acc[lane].items()}
            out[lane] = 0 if p["A"] > 0.5 else int(round(sum(BIN_COUNTS[i] * p[l] for i, l in enumerate(BIN_LETTERS))))
        out["rel_speed_kmh"] = 0.0
        return out
    except Exception:
        return None


def road_edges(video, duration, n=6, t0=None, t1=None):
    """{"edge_right": barrier|shoulder|open, "edge_left": ..., "one_way": bool} or None."""
    try:
        times = _times(duration, n, t0, t1)
        r = {k: 0.0 for k in "ABC"}
        l = {k: 0.0 for k in "ABCD"}
        yes = 0.0
        used = 0
        for t in times:
            fr = video_events.grab_frames(video, [t], width=640)
            if not fr:
                continue
            pr = llm_backend.choice_probabilities(RIGHT_EDGE_Q, fr, "ABC")
            for k in r:
                r[k] += pr[k]
            pl = llm_backend.choice_probabilities(LEFT_EDGE_Q, fr, "ABCD")
            for k in l:
                l[k] += pl[k]
            yes += llm_backend.yes_probability(ONCOMING_Q, fr)
            used += 1
        if not used:
            return None
        r = {k: v / used for k, v in r.items()}
        l = {k: v / used for k, v in l.items()}

        def pick(p, names):
            k = max(p, key=p.get)
            return names.get(k, "shoulder") if p[k] >= MIN_CHOICE_PROB else "shoulder"
        yes_avg = yes / used
        return {"edge_right": pick(r, {"A": "barrier", "B": "shoulder", "C": "open"}),
                "edge_left": pick(l, {"A": "barrier", "B": "shoulder", "C": "shoulder", "D": "open"}),
                "one_way": yes_avg < 0.25 or (yes_avg < 0.45 and l["B"] < 0.30)}
    except Exception:
        return None


def curve_direction(video, duration, n=6, t0=None, t1=None):
    """'left', 'right' or 'straight' for the road ahead (averaged over frames), or None."""
    try:
        p = _avg_choice(video, _times(duration, n, t0, t1), CURVE_Q, "ABC")
        return CURVE_KEYS[max(p, key=p.get)]
    except Exception:
        return None


def yellow_centerline_left(video, duration, t_crash=None, n=3):
    """Probability that a yellow center line runs directly to the left of the ego vehicle's lane."""
    try:
        if t_crash is not None:
            tc = float(t_crash)
            times = sorted({round(min(max(tc - 4.0 + 1.2 * i, 0.5), max(duration - 0.1, 0.5)), 2) for i in range(n)})
        else:
            times = _times(duration, n, t0=0.5, t1=max(duration - 0.5, 1.0))
        acc, used = 0.0, 0
        for t in times:
            fr = video_events.grab_frames(video, [t], width=640)
            if not fr:
                continue
            acc += llm_backend.yes_probability(YELLOW_LEFT_Q, fr)
            used += 1
        return round(acc / used, 3) if used else 0.0
    except Exception:
        return 0.0


def yellow_line_left_score(video, duration, t_crash=None, n=8):
    """Vision-only yellow-centerline score (0..1): pixel-weighted fraction of lane-marking pixels in the lower-left road half
    that have yellow HSV paint across pre-crash frames. Weighting by marking-pixel area naturally lets unobstructed frames
    dominate stopped/occluded frames where a queue hides the centerline."""
    try:
        import cv2
        import numpy as np
        from agents import lane_lines
        dur = max(float(duration), 0.5)
        t1 = min(dur - 0.2, float(t_crash) - 0.5) if (t_crash is not None and float(t_crash) > 1.5) else max(dur - 0.2, 0.5)
        t0 = min(0.2, max(t1 - 0.4, 0.0))
        times = _times(dur, n, t0=t0, t1=max(t1, t0 + 0.2))
        raws = video_events.grab_frames(video, times, width=640)
        if not raws:
            return 0.0
        kernel = np.ones((5, 5), np.uint8)
        tot_yel, tot_ll = 0, 0
        for raw in raws:
            arr = np.frombuffer(raw, np.uint8)
            im = cv2.imdecode(arr, cv2.IMREAD_COLOR) if arr.size else None
            if im is None or im.size == 0:
                continue
            h, w = im.shape[:2]
            y0, y1 = int(0.45 * h), int(0.92 * h)
            x0, x1 = int(0.05 * w), int(0.50 * w)
            if y1 - y0 < 16 or x1 - x0 < 16:
                continue
            mm = lane_lines.model_masks(im)
            ll = mm[0] if mm is not None else lane_lines.marking_mask(im, y0, y1)
            ll_roi = cv2.dilate((ll[y0:y1, x0:x1] > 0).astype(np.uint8), kernel) > 0
            denom = int(ll_roi.sum())
            if denom < 30:
                continue
            hsv = cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
            yellow = (hsv[..., 0] >= 12) & (hsv[..., 0] <= 38) & (hsv[..., 1] >= 70) & (hsv[..., 2] >= 90)
            tot_yel += int((yellow & ll_roi).sum())
            tot_ll += denom
        if tot_ll < 30 or tot_yel < 100:
            return 0.0
        return round(float(tot_yel) / float(tot_ll), 3)
    except Exception:
        return 0.0


def pick_crash_candidates(scores, k=3, min_gap_s=1.0):
    """The k strongest smoothed windows that are at least min_gap_s apart (ties: strongest raw score, then earliest)."""
    if not scores:
        return []
    sm = []
    for i in range(len(scores)):
        w = [scores[j]["score"] for j in range(max(i - 1, 0), min(i + 2, len(scores)))]
        sm.append(sum(w) / len(w))
    order = sorted(range(len(scores)), key=lambda i: (-sm[i], -scores[i]["score"], scores[i]["t"]))
    picked = []
    for i in order:
        t = float(scores[i]["t"])
        if all(abs(t - u) >= min_gap_s for u in picked):
            picked.append(t)
        if len(picked) >= k:
            break
    return picked


def refine_crash_time(candidates, score_fn, span=0.6, step=0.15):
    """Evaluate score_fn(t) around every candidate on a fine grid. Returns (best time, its score); ties go to the earlier
    candidate and to the grid point nearest to it. (None, 0.0) without candidates."""
    n = int(round(span / step))
    offsets = sorted((o * step for o in range(-n, n + 1)), key=lambda o: (abs(o), o))
    best_t, best = None, float("-inf")
    for c in candidates:
        for o in offsets:
            t = max(c + o, 0.0)
            sc = score_fn(t)
            if sc > best:
                best_t, best = t, sc
    return (best_t, best) if best_t is not None else (None, 0.0)



# ----------------------------------------------------------------------------------------- lanes that leave as a branch
BRANCH_RISE_Q = (CTX + "Is there a road or lanes on the %s that are HIGHER than the ego vehicle's road, rising up on a ramp, "
                 "embankment or bridge? Answer YES or NO.")
BRANCH_SPLIT_Q = (CTX + "Look at the %s side of the ego vehicle's road. Do some lanes on the %s separate from the ego vehicle's road "
                  "and continue as a different road (a fork, an exit ramp, a separate carriageway or a flyover), divided from it by a "
                  "barrier, a gap or a different height? Answer YES or NO.")
FORK_SIDE_Q = (CTX + "Look ahead along the highway or road. Does the road split or fork so that some lanes separate away as a "
               "different carriageway, main highway, exit ramp or flyover?\n"
               "A) yes, lanes separate away on the LEFT side of the ego vehicle's road\n"
               "B) yes, lanes separate away on the RIGHT side of the ego vehicle's road\n"
               "C) no, the road does not split into separate roads")
RISE_P = 0.5           # mean YES probability above which the lanes that leave climb (measured: 0.52-0.68 vs 0.08-0.40 without)
SPLIT_P = 0.5          # ... above which they go on as a flat road of their own (a weaker cue: the plain question overlaps more)


def branch_evidence(video, t, side, duration, n=4, span=4.0):
    """(P(lanes on `side` go on as a separate road), P(that road rises)) from frames around and after video time t
    (approaching the gore through passing it), when lanes leave the ego road on that side. (0, 0) when unavailable."""
    try:
        t0 = max(float(t) - 1.0, 0.0)
        t1 = min(float(t) + span, max(float(duration) - 0.1, 0.0))
        if t1 <= t0:
            t1 = max(float(duration) - 0.1, 0.0)
        times = [t0 + (t1 - t0) * i / max(n - 1, 1) for i in range(n)]
        word = side.upper()
        ps, pr, used = 0.0, 0.0, 0
        for tt in times:
            fr = video_events.grab_frames(video, [tt], width=640)
            if not fr:
                continue
            ps += llm_backend.yes_probability(BRANCH_SPLIT_Q % (word, word), fr)
            pr += llm_backend.yes_probability(BRANCH_RISE_Q % word, fr)
            used += 1
        return (ps / used, pr / used) if used else (0.0, 0.0)
    except Exception:
        return 0.0, 0.0


def fork_side(video, t, duration, n=4, span=4.5):
    """Comparative left-vs-right fork probe around video time t: {'left', 'right', 'none'} probabilities, or None."""
    try:
        t0 = max(float(t) - 1.5, 0.0)
        t1 = min(float(t) + span - 1.5, max(float(duration) - 0.1, 0.0))
        if t1 <= t0:
            t1 = max(float(duration) - 0.1, 0.0)
        times = [t0 + (t1 - t0) * i / max(n - 1, 1) for i in range(n)]
        p = _avg_choice(video, times, FORK_SIDE_Q, "ABC")
        return {"left": round(p["A"], 3), "right": round(p["B"], 3), "none": round(p["C"], 3)}
    except Exception:
        return None


def estimate_environment(video, duration, n=4):
    """Fast OpenCV estimate of {'time_of_day': day|dusk|night, 'weather': clear|cloudy|fog} from sky/road statistics."""
    default = {"time_of_day": "day", "weather": "clear"}
    try:
        import cv2
        import numpy as np
        times = _times(max(float(duration), 0.5), n, t0=0.2, t1=max(float(duration) - 0.2, 0.4))
        raws = video_events.grab_frames(video, times, width=320)
        if not raws:
            return default
        med_vs, sky_vs, sky_ss, sky_brs, road_stds = [], [], [], [], []
        for raw in raws:
            arr = np.frombuffer(raw, np.uint8)
            im = cv2.imdecode(arr, cv2.IMREAD_COLOR) if arr.size else None
            if im is None or im.size == 0:
                continue
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
            rs = np.where(g.mean(axis=1) > 15)[0]
            y0, y1 = (int(rs[0]), int(rs[-1]) + 1) if rs.size > 20 else (0, g.shape[0])
            act = im[y0:y1, :]
            ah = act.shape[0]
            if ah < 20:
                continue
            hsv = cv2.cvtColor(act, cv2.COLOR_BGR2HSV)
            med_vs.append(float(np.median(hsv[..., 2])))
            sky_bgr = act[:max(int(0.35 * ah), 4), :]
            sky_hsv = hsv[:max(int(0.35 * ah), 4), :]
            sky_vs.append(float(np.median(sky_hsv[..., 2])))
            sky_ss.append(float(np.median(sky_hsv[..., 1])))
            sky_brs.append(float(np.mean(sky_bgr[..., 0].astype(np.float32) - sky_bgr[..., 2].astype(np.float32))))
            road_g = cv2.cvtColor(act[int(0.50 * ah):int(0.85 * ah), :], cv2.COLOR_BGR2GRAY)
            if road_g.size:
                road_stds.append(float(np.std(road_g)))
        if not med_vs:
            return default
        med_v = float(np.median(med_vs))
        sky_v = float(np.median(sky_vs))
        sky_s = float(np.median(sky_ss))
        sky_br = float(np.median(sky_brs))
        road_std = float(np.median(road_stds)) if road_stds else 40.0
        if med_v < 58.0 and sky_v < 75.0:
            tod = "night"
        elif med_v < 95.0 and sky_v < 125.0:
            tod = "dusk"
        else:
            tod = "day"
        if tod == "night":
            weather = "clear"
        elif sky_br > 12.0 and sky_s > 35.0:
            weather = "clear"
        elif road_std < 18.0 and sky_s < 25.0 and sky_v > 140.0:
            weather = "fog"
        else:
            weather = "cloudy"
        return {"time_of_day": tod, "weather": weather}
    except Exception:
        return default
