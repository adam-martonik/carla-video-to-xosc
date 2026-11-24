"""
Bird's-eye lane map: lane lines and road surface from many frames collected on one grid in road coordinates.

  grid      s along the road from the ego car's start (m), d across it (m, + = left of the ego lane centre)
  per frame TwinLiteNet's lane-line and drivable-area masks (agents/lane_lines) are warped into a top view with the flat-road
            camera (agents/camera_geom) and added to the grid at s = s_ego(t) + Z. Pixels under a detected vehicle
            (agents/vehicle_tracks boxes) are left out: a car hides the road, it does not mean there is none.
  per slice the lines are the peaks of P(line | seen) across the road; the lane width is the spacing of neighbouring lines
            (one value for the video); the road reaches as far as P(road | seen) stays high (a strip narrower than
            SHOULDER_LANES beyond the outermost line is a shoulder). The lanes beside the ego lane are the most that the lines
            (on the lane grid), the road surface or the vehicles driving the same way show.

The result is a lane count every SLICE_M metres in the same sample format as agents/lane_lines ("t" is the distance s),
ready for agents/lane_profile.segment_counts().
"""
import math

import cv2
import numpy as np

S_RES = 0.5                 # m per grid cell along the road
D_RES = 0.2                 # m per grid cell across the road
D_RANGE = (-20.0, 25.0)     # across the road (m, + left)
Z_RANGE = (4.0, 55.0)       # ahead of the camera that is used (m): nearer is under the hood, further is too coarse
ROAD_Z_MAX = 55.0           # road / not-road evidence only from this near (tried 30 m: video 1 got worse, so the full range)
SLICE_M = 2.5               # distance between lane counts
SLICE_HALF_M = 5.0          # half length of the stretch one count pools
MIN_SEEN = 25               # grid hits a cross-road cell needs in a slice to count as seen
LINE_P = 0.12               # line probability of a peak
ROAD_P = 0.45               # road probability of a cell that is road
SHOULDER_LANES = 1.2         # surface beyond the outermost line counts as a lane only from this many lane widths on
FRAME_STEP_S = 0.2


class LaneGrid:
    def __init__(self, s_max, s_min=-5.0):
        self.s0 = float(s_min)
        self.ns = int(math.ceil((s_max - s_min) / S_RES)) + 1
        self.nd = int(round((D_RANGE[1] - D_RANGE[0]) / D_RES)) + 1
        self.seen = np.zeros((self.ns, self.nd), np.float32)
        self.line = np.zeros((self.ns, self.nd), np.float32)
        self.road = np.zeros((self.ns, self.nd), np.float32)
        self.seen_r = np.zeros((self.ns, self.nd), np.float32)       # seen near enough to say road / not road
        self.paint = np.zeros((self.ns, self.nd), np.float32)        # raw painted marking hits (keeps dashes distinct from solid)
        self.seen_p = np.zeros((self.ns, self.nd), np.float32)

    def d_of(self, j):
        return D_RANGE[0] + j * D_RES

    def j_of(self, d):
        return int(round((d - D_RANGE[0]) / D_RES))

    def add(self, s_off, X, Z, line, road, valid, curve=None, paint=None):
        """Add one top view: X, Z (m) per top-view pixel, masks line / road / valid of the same shape, at ego travel s_off;
        curve: agents/camera_geom.RoadCurve, so that a bending road's lanes stay straight in the grid."""
        d = -X - (curve.offset(s_off, Z) if curve is not None else 0.0)
        si = np.round((s_off + Z - self.s0) / S_RES).astype(int)
        dj = np.round((d - D_RANGE[0]) / D_RES).astype(int)
        ok = valid & (si >= 0) & (si < self.ns) & (dj >= 0) & (dj < self.nd)
        si, dj = si[ok], dj[ok]
        np.add.at(self.seen, (si, dj), 1.0)
        np.add.at(self.line, (si, dj), line[ok].astype(np.float32))
        near = Z[ok] <= ROAD_Z_MAX
        np.add.at(self.seen_r, (si[near], dj[near]), 1.0)
        np.add.at(self.road, (si[near], dj[near]), road[ok][near].astype(np.float32))
        if paint is not None:
            np.add.at(self.seen_p, (si, dj), 1.0)
            np.add.at(self.paint, (si, dj), paint[ok].astype(np.float32))

    def slice(self, s, half=SLICE_HALF_M, road_seen=False):
        """(seen, P(line), P(road)) across the road for the stretch s +- half; P(road) is over the cells seen near enough
        (ROAD_Z_MAX), whose counts are returned as a fourth value with road_seen=True."""
        a = max(int((s - half - self.s0) / S_RES), 0)
        b = min(int((s + half - self.s0) / S_RES) + 1, self.ns)
        seen = self.seen[a:b].sum(0)
        seen_r = self.seen_r[a:b].sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            pl = np.where(seen > 0, self.line[a:b].sum(0) / seen, 0.0)
            pr = np.where(seen_r > 0, self.road[a:b].sum(0) / seen_r, 0.0)
        return (seen, pl, pr, seen_r) if road_seen else (seen, pl, pr)


def top_view_coords(cam, res=D_RES):
    """X, Z (m) of every pixel of the top view used for the grid, and the warp homography and size."""
    from agents import camera_geom
    xr = (-D_RANGE[1], -D_RANGE[0])                      # X = -d
    Hm, size = camera_geom.bev_homography(cam, xr, Z_RANGE, res)
    ii, jj = np.meshgrid(np.arange(size[0]), np.arange(size[1]))
    X = xr[0] + ii * res
    Z = Z_RANGE[1] - jj * res
    return X, Z, Hm, size


def _paint_mask(bgr):
    """Classical white/yellow road-paint mask that preserves dashes (TwinLiteNet fills dashes into solid ribbons)."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    k = max(int(0.035 * bgr.shape[1]) | 1, 3)
    th = np.maximum(
        cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))),
        cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(k // 2 | 1, 3)))),
    ).astype(np.float32)
    white = (th > 10.0) & (th > 2.5 * cv2.blur(th, (61, 61)))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    yellow = (hsv[..., 0] >= 12) & (hsv[..., 0] <= 38) & (hsv[..., 1] >= 70) & (hsv[..., 2] >= 100)
    return ((white | yellow) * 255).astype(np.uint8)


def accumulate(path, cam, s_ego, t0, t1, y_bottom, boxes_at=None, step=FRAME_STEP_S, curve=None, vx_at=None):
    """Fill a LaneGrid from the frames between t0 and t1 (every `step` s). boxes_at(t) -> [[x1, y1, x2, y2], ...] vehicle boxes
    at the analysis width (left out of the map); vx_at(t) -> the vanishing point's column at t (the camera's heading against
    the road changes on a bend; one value for the whole video would slant the lines). None without the lane model."""
    import copy
    from agents import lane_lines
    if lane_lines._model() is None:
        return None
    X, Z, Hm, size = top_view_coords(cam)
    grid = LaneGrid(s_max=s_ego(t1) + Z_RANGE[1] + 5.0, s_min=min(s_ego(t0), 0.0) - 5.0)
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        t = float(t0)
        while t <= float(t1) + 1e-6:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
            ok, fr = cap.read()
            if not ok:
                break
            h, w = fr.shape[:2]
            fr = cv2.resize(fr, (cam.w, int(round(h * cam.w / w))))
            mm = lane_lines.model_masks(fr)
            ll, da = mm
            pm = _paint_mask(fr)
            valid = np.zeros(fr.shape[:2], np.uint8)
            valid[int(cam.vy + 3):int(y_bottom)] = 1
            for b in (boxes_at(t) if boxes_at else []):
                x1, y1, x2, y2 = b
                pad = 0.1 * (x2 - x1)
                valid[max(int(y1), 0):int(y2) + 1, max(int(x1 - pad), 0):int(x2 + pad) + 1] = 0
            Ht = Hm
            if vx_at is not None:
                c_t = copy.copy(cam)
                c_t.vx = vx_at(t)
                Ht = top_view_coords(c_t)[2]
            warp = lambda m: cv2.warpPerspective(m, Ht, size, flags=cv2.INTER_NEAREST | cv2.WARP_INVERSE_MAP, borderValue=0)
            v_t = warp(valid) > 0
            grid.add(s_ego(t), X, Z, warp(ll) > 0, warp(np.maximum(da, ll)) > 0, v_t, curve, paint=warp(pm) > 0)
            t += step
    finally:
        cap.release()
    return grid


def line_peaks(pl, seen, min_sep_m=2.0):
    """Cross-road positions (m) of the lane lines in one slice: peaks of the smoothed line probability among seen cells."""
    p = np.where(seen >= MIN_SEEN, pl, 0.0)
    p = cv2.GaussianBlur(p.reshape(1, -1).astype(np.float32), (0, 0), sigmaX=1.5).ravel()
    order = np.argsort(-p)
    keep = []
    sep = int(round(min_sep_m / D_RES))
    for j in order:
        if p[j] < LINE_P:
            break
        if all(abs(j - k) >= sep for k in keep):
            keep.append(int(j))
    return sorted(D_RANGE[0] + j * D_RES for j in keep)


def lane_width(grid, slices, default=3.5):
    """The video's lane width: the width of the ego lane (its two lines around d = 0, 2.8 .. 4.1 m apart), median over all
    slices; failing that the median spacing of neighbouring lines, clamped to realistic road lane widths (2.8 .. 4.0 m)."""
    ego, gaps = [], []
    for s in slices:
        seen, pl, _ = grid.slice(s)
        pk = line_peaks(pl, seen)
        gaps += [b - a for a, b in zip(pk, pk[1:]) if 2.6 <= b - a <= 4.3]
        left = [d for d in pk if d > 0]
        right = [d for d in pk if d < 0]
        if left and right and 2.8 <= min(left) - max(right) <= 4.1:
            ego.append(min(left) - max(right))
    if len(ego) >= 5:
        return min(max(float(np.median(ego)), 2.8), 4.0)
    return min(max(float(np.median(gaps)), 2.8), 4.0) if len(gaps) >= 5 else default


def road_reach(pr, seen, side, max_gap_m=1.5):
    """How far (m) the road surface reaches from the ego lane centre to one side ('left' = + d): cells with P(road) above
    ROAD_P, gaps of unseen or non-road cells up to max_gap_m bridged."""
    j0 = int(round(-D_RANGE[0] / D_RES))
    step = 1 if side == "left" else -1
    last, gap, j = 0.0, 0, j0
    while 0 <= j < len(pr):
        if seen[j] >= MIN_SEEN and pr[j] >= ROAD_P:
            last, gap = abs(D_RANGE[0] + j * D_RES), 0
        else:
            gap += 1
            if gap * D_RES > max_gap_m:
                break
        j += step
    return last


def solid(grid, s, d, half=SLICE_HALF_M, tol_m=0.4):
    """Share of the grid rows of the stretch s +- half on which a line was seen at d (+- tol_m): a solid line is seen on most of
    them, a dashed one on about a third. Uses the raw painted-marking channel (`grid.paint`) when populated so TwinLiteNet's
    continuous lane-ribbon output does not turn dashed lane dividers into solid edges."""
    a = max(int((s - half - grid.s0) / S_RES), 0)
    b = min(int((s + half - grid.s0) / S_RES) + 1, grid.ns)
    j0, j1 = max(grid.j_of(d - tol_m), 0), min(grid.j_of(d + tol_m) + 1, grid.nd)
    if j0 >= j1:
        return 0.0
    if hasattr(grid, "seen_p") and grid.seen_p[a:b, j0:j1].sum() >= 8:
        seen = grid.seen_p[a:b, j0:j1].sum(1)
        hit = grid.paint[a:b, j0:j1].sum(1)
        thresh = 0.12
    else:
        seen = grid.seen[a:b, j0:j1].sum(1)
        hit = grid.line[a:b, j0:j1].sum(1)
        thresh = 0.15
    rows = seen >= 2
    if rows.sum() < 4:
        return 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(((hit[rows] / seen[rows]) > thresh).mean())


SOLID = 0.7
LATTICE_TOL_BEV = 0.2          # a line counts as the edge of a further lane within this share of a lane width of the grid


def count_slice(grid, s, w, vehicles_d=()):
    """Lanes beside the ego lane in the slice at s: (left, right, details) from lines on the lane grid, the road surface and
    the vehicles driving the same way (vehicles_d: their d in this slice)."""
    seen, pl, pr, seen_r = grid.slice(s, road_seen=True)
    if seen.max() < MIN_SEEN:
        return None
    pk = line_peaks(pl, seen)
    left_l = [d for d in pk if d > 0.25 * w]
    right_l = [d for d in pk if d < -0.25 * w]

    def lattice(lines, sign):
        if not lines:
            return 0, None
        first = min(lines, key=abs)                   # the ego lane's line on this side
        beyond = [d for d in vehicles_d if d * sign > abs(first)]
        more_lines = [d for d in lines if abs(d) > abs(first) + 0.5 * w
                      and abs(((abs(d) - abs(first)) / w) - round((abs(d) - abs(first)) / w)) <= LATTICE_TOL_BEV]
        if solid(grid, s, first) >= SOLID and not beyond and not more_lines:
            return 0, first                           # a solid edge line with neither traffic nor further lane lines beyond
        n = 0
        for d in lines:
            k = (abs(d) - abs(first)) / w
            if abs(k - round(k)) <= LATTICE_TOL_BEV:
                n = max(n, int(round(k)))
        return n, max(lines, key=abs)

    nl_lines, out_l = lattice(left_l, 1)
    nr_lines, out_r = lattice(right_l, -1)
    reach_l, reach_r = road_reach(pr, seen_r, "left"), road_reach(pr, seen_r, "right")
    # beyond the outermost line: a strip narrower than SHOULDER_LANES is a shoulder; so is a modest strip beyond a solid edge
    # line that no car drives beyond (but a wide multi-lane expanse >= 2.2 lanes of drivable pavement is real highway lanes, e.g.
    # separated by a solid exit-lane divider before a fork)
    for side in ("left", "right"):
        out = out_l if side == "left" else out_r
        reach = reach_l if side == "left" else reach_r
        if out is None or reach <= abs(out):
            continue
        beyond = [d for d in vehicles_d if (d > abs(out) if side == "left" else d < -abs(out))]
        is_solid_edge = solid(grid, s, out) >= SOLID and not beyond and reach < abs(out) + 2.2 * w
        if reach < abs(out) + SHOULDER_LANES * w or is_solid_edge:
            if side == "left":
                reach_l = abs(out)
            else:
                reach_r = abs(out)
    # whole lanes of road beyond the ego lane's own line on each side (half a lane from the centre when that line is not seen)
    first_l = min(left_l, key=abs) if left_l else 0.5 * w
    first_r = abs(max(right_l)) if right_l else 0.5 * w
    nl_road = max(int(math.floor((reach_l - abs(first_l)) / w + 0.25)), 0)
    nr_road = max(int(math.floor((reach_r - first_r) / w + 0.25)), 0)
    nl_veh = max([int(round(d / w)) for d in vehicles_d if d > 0.5 * w] or [0])
    nr_veh = max([int(round(-d / w)) for d in vehicles_d if d < -0.5 * w] or [0])
    nl, nr = max(nl_lines, nl_road, nl_veh), max(nr_lines, nr_road, nr_veh)
    return nl, nr, {"lines": [round(d, 1) for d in pk], "reach": [round(reach_l, 1), round(reach_r, 1)],
                    "from": {"lines": [nl_lines, nr_lines], "road": [nl_road, nr_road], "vehicles": [nl_veh, nr_veh]}}


def lane_samples(grid, s_from, s_to, tracks=(), road_ok=None):
    """Lane counts every SLICE_M m between s_from and s_to in the sample format of agents/lane_lines ("t" = distance s):
    [{"t", "probs", "lanes_left", "lanes_right", "road_lanes", "detail"}]. tracks: road tracks of agents/vehicle_tracks; only
    those driving the same way count as lanes."""
    from agents import lane_lines
    slices = [s_from + SLICE_M * i for i in range(int((s_to - s_from) / SLICE_M) + 1)]
    w = lane_width(grid, slices)
    out = []
    for s in slices:
        vd = []
        for tr in tracks:
            if tr.get("kind") != "same":
                continue
            for ss, dd, zz in zip(tr["s"], tr["d"], tr.get("z") or [0.0] * len(tr["s"])):
                if abs(ss - s) <= SLICE_HALF_M and zz <= Z_RANGE[1] and not seen_off_road(grid, ss, dd):
                    vd.append(dd)
        r = count_slice(grid, s, w, vd)
        if r is None:
            out.append({"t": round(s, 2), "probs": None})
            continue
        nl, nr, det = r
        out.append({"t": round(s, 2), "lanes_left": float(nl), "lanes_right": float(nr), "lane_w": round(w, 2), "detail": det})
    carry_lanes(grid, out, w)
    for r in out:
        if "lanes_left" in r:
            det = r["detail"]
            r["road_lanes"] = 1.0 + r["lanes_left"] + r["lanes_right"]
            r["probs"] = lane_lines.count_probs(r["road_lanes"], 1 + det["from"]["lines"][0] + det["from"]["lines"][1])
    return out


CARRY_M = 40.0              # a lane seen is assumed to go on this far where nothing is seen
ROLL_SLICES = 5             # slices of the rolling median that is carried (one noisy slice is not)


def carry_lanes(grid, rows, w):
    """Not seeing a lane is no proof that it ended. The counts are first replaced by their rolling median over ROLL_SLICES
    slices (so that one noisy slice counts for nothing); a lane in that median is then carried along the road, forward and back,
    for up to CARRY_M metres, unless the map clearly saw non-road where that lane would be. Changes rows in place."""
    idx = [i for i, r in enumerate(rows) if "lanes_left" in r]
    for key in ("lanes_left", "lanes_right"):
        vals = [rows[i][key] for i in idx]
        h = ROLL_SLICES // 2
        med = [float(np.median(vals[max(k - h, 0):k + h + 1])) for k in range(len(vals))]
        for i, m in zip(idx, med):
            rows[i][key] = m
    for side, key, sign in (("left", "lanes_left", 1.0), ("right", "lanes_right", -1.0)):
        base = {i: rows[i][key] for i in idx}            # carry from what was measured, never from what was carried
        best = dict(base)
        for order in (idx, idx[::-1]):
            carry, carry_s = 0.0, None
            for i in order:
                r = rows[i]
                if base[i] >= carry or (carry_s is not None and abs(r["t"] - carry_s) > CARRY_M):
                    carry, carry_s = base[i], r["t"]
                n = carry
                while n > base[i] and seen_off_road(grid, r["t"], sign * n * w):
                    n -= 1.0
                best[i] = max(best[i], n)
        for i in idx:
            if best[i] > base[i]:
                rows[i][key] = best[i]
                rows[i]["detail"].setdefault("carried", []).append(side)


def seen_off_road(grid, s, d, half_m=1.0):
    """Whether the map clearly saw the ground at (s, d) and it was not road (e.g. a frontage road beyond a barrier); a car
    standing in a queue hides its own lane, which then counts as not seen, not as off the road."""
    a = max(int((s - SLICE_HALF_M - grid.s0) / S_RES), 0)
    b = min(int((s + SLICE_HALF_M - grid.s0) / S_RES) + 1, grid.ns)
    j0, j1 = max(grid.j_of(d - half_m), 0), min(grid.j_of(d + half_m) + 1, grid.nd)
    if j0 >= j1:
        return True
    seen = grid.seen_r[a:b, j0:j1].sum()
    if seen < 3 * MIN_SEEN:
        return False
    return grid.road[a:b, j0:j1].sum() / seen < 0.2


def on_road(grid, s, d, w=3.5):
    """Whether a point (s, d) lies on the road surface the ego car is on (within half a lane of its reach)."""
    seen, pl, pr, seen_r = grid.slice(s, road_seen=True)
    if seen_r.max() < MIN_SEEN:
        return True
    reach = road_reach(pr, seen_r, "left" if d >= 0 else "right")
    return abs(d) <= reach + 0.5 * w
