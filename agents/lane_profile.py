"""
Lane count along the road, from per-frame lane-count probabilities (no CARLA, no model: plain numbers, so it can be tested).

  samples  [{"t": s, "probs": {1: p, 2: p, ... 6: p}}]           one per probed frame
     -> segment_counts()   piecewise-constant lane count over time (Viterbi with a switch penalty + minimum duration)
     -> build_profile()    sections along the road in metres from the ego start, with the side on which lanes end or begin

A section is {"d0": metres or None, "d1": metres or None, "n": lanes, "m": lanes missing on the LEFT}.
d is the distance along the road from the ego car's start position in the simulation (None = open end).
"m" counts lanes relative to the left-most lane that exists anywhere on the profile, so lane j counted from the left
(0 = left-most ever) is at the same lateral position in every section: a right-side drop keeps m, a left-side drop raises it.
"""
import math

MAX_LANES = 6
COUNTS = tuple(range(1, MAX_LANES + 1))


def _p(probs, n):
    v = probs.get(n, probs.get(str(n), 0.0))
    return float(v or 0.0)


def segment_counts(samples, switch_cost=4.0, min_dur_s=1.0):
    """Most likely piecewise-constant lane count. Returns [{"t0", "t1", "n", "k0", "k1"}] (k = sample indices, inclusive).

    Each change of the count costs `switch_cost` (in nats of the model's log-probabilities), so a single noisy frame does not
    split the road; segments shorter than `min_dur_s` are then merged into the neighbour that explains them better."""
    samples = sorted((s for s in samples if s.get("probs")), key=lambda s: float(s["t"]))
    if not samples:
        return []
    cost = [{n: -math.log(_p(s["probs"], n) + 1e-6) for n in COUNTS} for s in samples]
    acc = [dict(cost[0])]
    back = [{}]
    for i in range(1, len(samples)):
        row, brow = {}, {}
        for n in COUNTS:
            prev_best = min(COUNTS, key=lambda m: acc[-1][m] + (0.0 if m == n else switch_cost))
            row[n] = acc[-1][prev_best] + (0.0 if prev_best == n else switch_cost) + cost[i][n]
            brow[n] = prev_best
        acc.append(row)
        back.append(brow)
    n = min(COUNTS, key=lambda m: acc[-1][m])
    path = [n]
    for i in range(len(samples) - 1, 0, -1):
        n = back[i][n]
        path.append(n)
    path.reverse()

    segs = []
    for i, n in enumerate(path):
        if segs and segs[-1]["n"] == n:
            segs[-1]["k1"] = i
        else:
            segs.append({"n": n, "k0": i, "k1": i})

    def dur(sg):
        return float(samples[sg["k1"]]["t"]) - float(samples[sg["k0"]]["t"])

    def fit(n, sg):
        return sum(cost[k][n] for k in range(sg["k0"], sg["k1"] + 1))

    while len(segs) > 1:
        short = [i for i, sg in enumerate(segs) if dur(sg) < min_dur_s]
        if not short:
            break
        i = min(short, key=lambda j: dur(segs[j]))
        nb = [j for j in (i - 1, i + 1) if 0 <= j < len(segs)]
        j = min(nb, key=lambda j: fit(segs[j]["n"], segs[i]))
        lo, hi = min(i, j), max(i, j)
        merged = {"n": segs[j]["n"], "k0": segs[lo]["k0"], "k1": segs[hi]["k1"]}
        segs[lo:hi + 1] = [merged]
        k = 0                                        # neighbours with the same count join
        while k < len(segs) - 1:
            if segs[k]["n"] == segs[k + 1]["n"]:
                segs[k:k + 2] = [{"n": segs[k]["n"], "k0": segs[k]["k0"], "k1": segs[k + 1]["k1"]}]
            else:
                k += 1
    for sg in segs:
        sg["t0"] = float(samples[sg["k0"]]["t"])
        sg["t1"] = float(samples[sg["k1"]]["t"])
    return segs


def boundary_times(segs, samples):
    """Time of each change of the count: halfway between the last sample of one segment and the first of the next."""
    ts = sorted(float(s["t"]) for s in samples if s.get("probs"))
    return [0.5 * (ts[a["k1"]] + ts[b["k0"]]) for a, b in zip(segs, segs[1:])]


def left_missing(counts, sides):
    """Lanes missing on the left per section (see module doc). sides[i] = 'left' or 'right' for the change between section i
    and i + 1 (where lanes end or begin)."""
    m = [0]
    for i in range(len(counts) - 1):
        delta = counts[i] - counts[i + 1]                  # > 0: lanes end, < 0: lanes begin
        m.append(m[-1] + (delta if sides[i] == "left" else 0))
    lo = min(m)
    return [x - lo for x in m]


def build_profile(segs, t_bounds, sides, t_start, speed_ms, min_section_m=15.0):
    """Sections along the road from the time segments. The simulated ego car drives at a constant `speed_ms` from the video time
    `t_start` on, so the change it sees at video time t lies d = speed * (t - t_start) metres down the road. Sections shorter than
    `min_section_m` (the ego car would not see them) are folded into their neighbour. Returns [] when the count never changes."""
    if len(segs) < 2 or speed_ms < 1.0:
        return []
    counts = [sg["n"] for sg in segs]
    ms = left_missing(counts, sides)
    secs = []
    for i, n in enumerate(counts):
        d0 = None if i == 0 else round(speed_ms * (t_bounds[i - 1] - t_start), 2)
        d1 = None if i == len(counts) - 1 else round(speed_ms * (t_bounds[i] - t_start), 2)
        secs.append({"d0": d0, "d1": d1, "n": n, "m": ms[i]})
    changed = True
    while changed and len(secs) > 1:
        changed = False
        for i, sc in enumerate(secs):
            # the simulation starts at d = 0, so a first section that ends within min_section_m is just as short
            start = sc["d0"] if sc["d0"] is not None else (0.0 if sc["d1"] is not None and i == 0 else None)
            if start is not None and sc["d1"] is not None and sc["d1"] - start < min_section_m:
                j = i - 1 if i > 0 else i + 1
                keep = secs[j]
                lo, hi = min(i, j), max(i, j)
                secs[lo:hi + 1] = [{"d0": secs[lo]["d0"], "d1": secs[hi]["d1"], "n": keep["n"], "m": keep["m"]}]
                changed = True
                break
    k = 0
    while k < len(secs) - 1:                               # equal neighbours join
        if secs[k]["n"] == secs[k + 1]["n"] and secs[k]["m"] == secs[k + 1]["m"]:
            secs[k:k + 2] = [{"d0": secs[k]["d0"], "d1": secs[k + 1]["d1"], "n": secs[k]["n"], "m": secs[k]["m"]}]
        else:
            k += 1
    if len(secs) < 2:
        return []
    lo = min(s["m"] for s in secs)
    for s in secs:
        s["m"] -= lo
    return secs


def section_at(secs, d):
    """The section that holds distance d (sections are ordered and contiguous)."""
    for s in secs:
        if (s["d1"] is None or d < s["d1"]):
            return s
    return secs[-1]
