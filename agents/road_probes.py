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
                "What is there?\nA) a concrete barrier, jersey wall or guardrail (even if a narrow shoulder line is in front of it)\n"
                "B) a curb, sidewalk or paved shoulder (with no barrier or wall)\n"
                "C) open ground, grass, dirt or sand (with no barrier or wall)")
LEFT_EDGE_Q = (CTX + "Look at the LEFT edge of the road, beside the left-most lane of the ego vehicle's driving direction. "
               "What is there?\nA) a concrete barrier, jersey wall or guardrail (even if a narrow shoulder line is in front of it)\n"
               "B) lanes with oncoming traffic or a painted median\n"
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
            fr = video_events.grab_frames(video, [t], width=512)
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
        return {"edge_right": pick(r, {"A": "barrier", "B": "shoulder", "C": "open"}),
                "edge_left": pick(l, {"A": "barrier", "B": "shoulder", "C": "shoulder", "D": "open"}),
                "one_way": (yes / used) < 0.25}
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
RISE_P = 0.5           # mean YES probability above which the lanes that leave climb (measured: 0.52-0.68 vs 0.08-0.40 without)
SPLIT_P = 0.5          # ... above which they go on as a flat road of their own (a weaker cue: the plain question overlaps more)


def branch_evidence(video, t, side, duration, n=4, span=4.0):
    """(P(lanes on `side` go on as a separate road), P(that road rises)) from frames in the `span` seconds after video time t,
    when lanes have left the ego road on that side. (0, 0) when the model is not available."""
    try:
        times = [min(t + 0.5 + span * i / max(n - 1, 1), max(duration - 0.1, 0.0)) for i in range(n)]
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
