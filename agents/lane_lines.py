"""
Lane count from the road surface and the painted lane lines.

The lane lines and the drivable area come from TwinLiteNet (models/twinlitenet, MIT licence, trained on BDD100K; loaded with
weights_only=True); without it the lines come from the classical filter in marking_mask() and no road width is measured.
The ego lane (from the lines) gives the lane width; the connected road surface around the ego car, divided by that width,
gives the number of lanes (lane_timeline). The geometry of the lines:

Every lane line of a straight, flat road is a ray from the vanishing point (VP). A ray is identified by the x where it crosses
one fixed row near the bottom of the road area ("x_ref"); lines that are equally spaced on the ground are equally spaced in
x_ref too. So:

  1. markings     : thin bright (white) or yellow structures on the road, from a horizontal top-hat filter and a colour test
  2. segments     : probabilistic Hough line segments on the marking mask (near-horizontal ones are dropped)
  3. VP           : the robust meeting point of the segments, pooled over a short window of frames
  4. x_ref        : every segment that points at the VP is extended to the reference row; lengths are pooled into a histogram
                    over a window of frames, so the gaps of dashed lines fill in
  5. count        : peaks of the histogram are the lane lines; the ego lane is between the two peaks around the image centre,
                    its width is the lane width in pixels, and lanes = (right-most peak - left-most peak) / lane width

Lines hidden behind traffic do not matter as long as the outermost lines (usually the solid edge lines) are seen in the window.
"""
import math
import os

import cv2
import numpy as np

WIDTH = 960                    # analysis width in pixels
TOPHAT_K = 0.035               # top-hat kernel width as a fraction of the width (wider than any lane marking near the car)
MIN_SEG_FRAC = 0.025           # minimum Hough segment length as a fraction of the width
MIN_ANGLE_DEG = 12.0           # segments flatter than this are not lane lines (stop lines, shadows, overpasses)
VP_TOL_FRAC = 0.03             # a segment points at the VP when the VP lies this close to its line (fraction of the width)
BIN_PX = 3.0


def _read(cap, fps, t, crop=None):
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(t, 0.0) * fps))
    ok, fr = cap.read()
    if not ok:
        return None
    if crop is not None:
        fr = fr[crop[0]:crop[1], crop[2]:crop[3]]
    h, w = fr.shape[:2]
    return cv2.resize(fr, (WIDTH, int(round(h * WIDTH / w))))


def road_bottom(path, n=12):
    """Row (at WIDTH) where the road area ends (the hood begins), and the frame height.
    The painted lines that point at the vanishing point stop at the hood edge, so the bottom is the typical lowest point of
    those lines (hood reflections and overlays do not point at the vanishing point). When TwinLiteNet is available, also check
    where its drivable-area mask ends in the centre of the frame (lane lines can leave the sides above the hood)."""
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1) / fps
        lows, da_lows, h = [], [], None
        has_model = _model() is not None
        for i in range(n):
            f = _read(cap, fps, dur * (i + 0.5) / n)
            if f is None:
                continue
            h, w = f.shape[:2]
            if has_model:
                mm = model_masks(f)
                if mm is not None:
                    da = mm[1]
                    center_da = da[:, w // 4 : 3 * w // 4] > 0
                    rows = np.where(center_da.mean(axis=1) > 0.15)[0]
                    if len(rows):
                        da_lows.append(int(rows[-1]))
            segs = segments(marking_mask(f, int(0.3 * h), int(0.95 * h)))
            vp = vanishing_point(segs, w, h, 0.65 * h)
            if vp is None:
                continue
            ys = [max(sg[1], sg[3]) for sg in segs
                  if x_at_ref(sg, vp, h) is not None and math.hypot(sg[2] - sg[0], sg[3] - sg[1]) >= 0.03 * w]
            if len(ys) >= 3:
                lows.append(float(np.percentile(ys, 85)))
        if h is None:
            return None, None
        hough_y = int(np.median(lows) - 0.01 * h) if lows else int(0.62 * h)
        da_y = (int(np.median(da_lows)) - 2) if da_lows else None
        return (max(v for v in (hough_y, da_y) if v is not None), h)
    finally:
        cap.release()


MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "twinlitenet")
_MODEL = {}


def _model():
    """TwinLiteNet (models/twinlitenet: MIT licence, trained on BDD100K) or None when its files or torch are missing.
    The weights are loaded with weights_only=True: the file can only hold tensors."""
    if "net" not in _MODEL:
        _MODEL["net"] = None
        try:
            import sys
            import torch
            if MODEL_DIR not in sys.path:
                sys.path.insert(0, MODEL_DIR)
            from TwinLite import TwinLiteNet
            sd = torch.load(os.path.join(MODEL_DIR, "best.pth"), map_location="cpu", weights_only=True)
            net = TwinLiteNet()
            net.load_state_dict({k[7:] if k.startswith("module.") else k: v for k, v in sd.items()})
            net.eval()
            _MODEL["net"] = net
        except Exception as e:                                   # no model: the classical filter below is used
            _MODEL["error"] = str(e)
    return _MODEL["net"]


def model_masks(bgr):
    """(lane-line mask, drivable-area mask) at the frame size from TwinLiteNet, or None without the model."""
    net = _model()
    if net is None:
        return None
    import torch
    key = (id(bgr), bgr.shape)
    if _MODEL.get("key") == key:
        return _MODEL["masks"]
    h, w = bgr.shape[:2]
    img = cv2.resize(bgr, (640, 360))
    x = torch.from_numpy(np.ascontiguousarray(img[:, :, ::-1].transpose(2, 0, 1))).float().unsqueeze(0) / 255.0
    with torch.no_grad():
        da, ll = net(x)
    da = cv2.resize(da.argmax(1)[0].numpy().astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
    ll = cv2.resize(ll.argmax(1)[0].numpy().astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)
    _MODEL["key"], _MODEL["masks"] = key, (ll, da)
    return ll, da


def marking_mask(bgr, y0, y1):
    """Binary mask of lane-marking pixels between rows y0 and y1: TwinLiteNet's lane-line mask when the model is there,
    else the classical filter."""
    mm = model_masks(bgr)
    if mm is not None:
        full = np.zeros(bgr.shape[:2], np.uint8)
        full[y0:y1] = mm[0][y0:y1]
        return full
    roi = bgr[y0:y1]
    g = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    k = max(int(TOPHAT_K * bgr.shape[1]) | 1, 3)
    # lines far to the side lie almost flat in the picture: a vertical kernel finds those, a horizontal one the steep ones
    th = np.maximum(cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1))),
                    cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(k // 2 | 1, 3)))))
    # a marking is brighter than its own surroundings: compare with the local mean response (trees, cars and sky are busy
    # everywhere around them, so one global threshold set by them misses faint lines on light concrete)
    th = th.astype(np.float32)
    white = (th > 10.0) & (th > 3.0 * cv2.blur(th, (61, 61)))
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    yellow = (hsv[..., 0] >= 12) & (hsv[..., 0] <= 38) & (hsv[..., 1] >= 70) & (hsv[..., 2] >= 100)
    m = ((white | yellow) * 255).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    full = np.zeros(bgr.shape[:2], np.uint8)
    full[y0:y1] = m
    return full


def segments(mask):
    """[(x1, y1, x2, y2)] Hough segments that are steep enough to be lane lines."""
    w = mask.shape[1]
    ls = cv2.HoughLinesP(mask, 1, np.pi / 180, threshold=25, minLineLength=int(MIN_SEG_FRAC * w), maxLineGap=int(0.012 * w))
    out = []
    for l in (ls if ls is not None else []):
        x1, y1, x2, y2 = map(float, l[0])
        ang = abs(math.degrees(math.atan2(y2 - y1, x2 - x1)))
        ang = min(ang, 180.0 - ang)
        if ang >= MIN_ANGLE_DEG:
            out.append((x1, y1, x2, y2))
    return out


def _line(seg):
    x1, y1, x2, y2 = seg
    a, b = y2 - y1, x1 - x2
    n = math.hypot(a, b) or 1.0
    return a / n, b / n, (a * x1 + b * y1) / n          # a x + b y = c, unit normal


def vanishing_point(segs, w, h, y_bottom):
    """Robust meeting point of the segments: the grid cell crossed by the most (length-weighted) segment lines."""
    if len(segs) < 2:
        return None
    lines = [(_line(s), math.hypot(s[2] - s[0], s[3] - s[1])) for s in segs]
    xs = np.linspace(-0.25 * w, 1.25 * w, 121)
    ys = np.linspace(0.05 * h, y_bottom - 0.05 * h, 61)
    gx, gy = np.meshgrid(xs, ys)
    acc = np.zeros_like(gx)
    tol = VP_TOL_FRAC * w
    for (a, b, c), L in lines:
        d = np.abs(a * gx + b * gy - c)
        acc += L * np.exp(-0.5 * (d / tol) ** 2)
    i = np.unravel_index(int(np.argmax(acc)), acc.shape)
    return float(gx[i]), float(gy[i])


def x_at_ref(seg, vp, y_ref):
    """x where the ray from the VP through the segment's middle crosses the reference row, or None when the segment does not
    point at the VP."""
    vx, vy = vp
    (a, b, c) = _line(seg)
    if abs(a * vx + b * vy - c) > 2.0 * VP_TOL_FRAC * WIDTH:
        return None
    mx, my = 0.5 * (seg[0] + seg[2]), 0.5 * (seg[1] + seg[3])
    if my <= vy + 2.0:
        return None
    return vx + (mx - vx) * (y_ref - vy) / (my - vy)


def _peaks(hist, centers, min_sep_px, rel=0.12):
    sm = cv2.GaussianBlur(hist.reshape(1, -1).astype(np.float32), (0, 0), sigmaX=2.0).ravel()
    if sm.max() <= 0:
        return []
    order = np.argsort(-sm)
    keep = []
    for i in order:
        if sm[i] < rel * sm.max():
            break
        if all(abs(centers[i] - centers[j]) >= min_sep_px for j in keep):
            keep.append(i)
    return sorted((float(centers[i]), float(sm[i] / sm.max())) for i in keep)


LANE_PX_PER_DEPTH = (1.4, 5.5)  # ego lane width in px per px of (y_ref - vp_y): lane 3.5 m, camera 2.5 .. 0.65 m above the road
LATTICE_TOL = 0.25               # a line counts when it lies within this fraction of a lane width of a whole lane step


def count_lanes(peaks, x_cam, d_ref):
    """Lane count from the line positions at the reference row (peaks = [(x, strength)], d_ref = y_ref - vp_y).
    The ego lane is the strongest pair of lines around the camera with a plausible width and no strong line between them;
    the other lines count only when they lie a whole number of ego lane widths away, so that a curb, a guard rail or a car
    edge does not add a lane. Returns (lanes, ego lanes from the left, lane width px, left line x) or a reason string."""
    best = None
    for xl, sl in peaks:
        for xr, sr in peaks:
            if not xl < x_cam <= xr:
                continue
            lane = xr - xl
            if not LANE_PX_PER_DEPTH[0] * d_ref <= lane <= LANE_PX_PER_DEPTH[1] * d_ref:
                continue
            between = max([sc for x, sc in peaks if xl < x < xr] or [0.0])
            score = sl + sr - 1.5 * between
            if best is None or score > best[0]:
                best = (score, xl, xr)
    if best is None:
        return "no plausible ego lane (lines %s)" % [round(x) for x, _ in peaks]
    _, xl, xr = best
    lane = xr - xl
    n_left = n_right = 0
    for x, _ in peaks:
        if x < xl:
            k = (xl - x) / lane
            if abs(k - round(k)) <= LATTICE_TOL:
                n_left = max(n_left, int(round(k)))
        elif x > xr:
            k = (x - xr) / lane
            if abs(k - round(k)) <= LATTICE_TOL:
                n_right = max(n_right, int(round(k)))
    return 1 + n_left + n_right, n_left, lane, xl


def road_region(f, y_bottom):
    """Boolean mask of the road surface the ego car is on: TwinLiteNet's drivable area plus its lane lines, small gaps closed,
    the connected part that holds the road just above the hood in the middle of the picture. Being connected, it flows around
    cars, so a car in a neighbouring lane does not cut the road off at that car. None without the model or the road."""
    mm = model_masks(f)
    if mm is None:
        return None
    h, w = mm[1].shape
    m = cv2.morphologyEx(np.maximum(mm[1], mm[0]), cv2.MORPH_CLOSE, np.ones((3, max(int(0.03 * w), 3)), np.uint8))
    _, lab = cv2.connectedComponents((m > 0).astype(np.uint8))
    col = lab[:int(y_bottom), int(0.5 * w)]
    ys = np.where(col > 0)[0]
    if not ys.size or ys[-1] < y_bottom - 0.15 * h:
        return None
    return lab == col[ys[-1]]


SHOULDER_LANES = 0.8           # road surface beyond the outermost painted line narrower than this many lanes is a shoulder


def road_extent(frames, vp, x_ego_ref, d_ref, y_bottom, lines=None, lane_px=None):
    """Left and right extent of the road region from the ego lane centre, in px per px of depth below the VP (medians over rows
    and frames; rows where the region runs into the side of the picture are left out on that side). Lane widths in the same unit
    are the video's lane ratio, so extent / ratio - 0.5 = lanes beside the ego lane. (left, right), each None without data.
    lines: x of the painted lines at the reference row; surface that reaches less than SHOULDER_LANES lanes beyond the outermost
    line on a side ends at that line (a paved shoulder, not a lane)."""
    vx, vy = vp
    lo_line = min(lines) if lines else None
    hi_line = max(lines) if lines else None
    left, right = [], []
    for f in frames:
        reg = road_region(f, y_bottom)
        if reg is None:
            continue
        w = reg.shape[1]
        for y in range(int(vy + 0.12 * d_ref), int(y_bottom), 3):
            xs = np.where(reg[y])[0]
            dy = y - vy
            if not xs.size or dy < 4.0:
                continue
            xe = vx + (x_ego_ref - vx) * dy / d_ref
            xl, xr = float(xs[0]), float(xs[-1])
            if lane_px:
                lp = lane_px * dy / d_ref
                if lo_line is not None:
                    ll = vx + (lo_line - vx) * dy / d_ref
                    if ll - SHOULDER_LANES * lp < xl < ll:
                        xl = ll
                if hi_line is not None:
                    lr = vx + (hi_line - vx) * dy / d_ref
                    if lr < xr < lr + SHOULDER_LANES * lp:
                        xr = lr
            if xs[0] > 2:
                left.append((xe - xl) / dy)
            if xs[-1] < w - 3:
                right.append((xr - xe) / dy)
    med = lambda v: float(np.median(v)) if len(v) >= 5 else None
    return med(left), med(right)


def analyse_window(frames, y_bottom):
    """Lane lines pooled over a list of BGR frames (same video, consecutive). Returns a dict ("lanes" None when unsure)."""
    if not frames:
        return None
    h, w = frames[0].shape[:2]
    allsegs = []
    for f in frames:
        allsegs += segments(marking_mask(f, int(0.3 * h), y_bottom))
    vp = vanishing_point(allsegs, w, h, y_bottom)
    if vp is None or vp[1] >= y_bottom - 0.08 * h:
        return {"lanes": None, "reason": "no vanishing point", "n_segments": len(allsegs)}
    allsegs = [sg for sg in allsegs if 0.5 * (sg[1] + sg[3]) > vp[1] + 0.02 * h]
    y_ref = vp[1] + 0.5 * (y_bottom - vp[1])
    d_ref = y_ref - vp[1]
    x_cam = 0.5 * w
    lo, hi = x_cam - 30.0 * d_ref, x_cam + 30.0 * d_ref
    centers = np.arange(lo, hi, BIN_PX) + 0.5 * BIN_PX
    hist = np.zeros_like(centers)
    used = 0
    for sg in allsegs:
        x = x_at_ref(sg, vp, y_ref)
        if x is None or not lo <= x < hi:
            continue
        hist[int((x - lo) / BIN_PX)] += math.hypot(sg[2] - sg[0], sg[3] - sg[1])
        used += 1
    peaks = _peaks(hist, centers, min_sep_px=0.5 * LANE_PX_PER_DEPTH[0] * d_ref)
    out = {"vp": [round(vp[0], 1), round(vp[1], 1)], "y_ref": round(y_ref, 1), "n_segments": len(allsegs), "n_used": used,
           "peaks": [[round(x, 1), round(sc, 2)] for x, sc in peaks], "lanes": None, "ego_from_left": None, "lane_px": None}
    res = count_lanes(peaks, x_cam, d_ref)
    if isinstance(res, str):
        out["reason"] = res
        return out
    out["lines_lanes"], out["ego_from_left"], out["lane_px"] = res[0], res[1], round(res[2], 1)
    out["lanes"] = res[0]
    out["ratio"] = round(res[2] / d_ref, 3)
    el, er = road_extent(frames, vp, res[3] + 0.5 * res[2], d_ref, y_bottom, lines=[x for x, _ in peaks], lane_px=res[2])
    out["extent_left"], out["extent_right"] = el, er
    return out


RATIO_RANGE = (2.2, 3.7)       # lane width / camera height: the plausible range of the video's lane ratio
SIGMA_LANES = 0.6               # spread of the lane-count probabilities around the measured road width (in lanes)


def finish_timeline(rows):
    """Second pass over the windows: one lane-width ratio for the whole video (the camera height does not change), the road
    width in lanes from the road extents, and lane-count probabilities {1..6: p} for agents/lane_profile.segment_counts().
    Windows without a measurement get no "probs"."""
    ratios = [r["ratio"] for r in rows if r.get("ratio")]
    rho = float(np.median(ratios)) if len(ratios) >= 3 else None
    if rho is not None:                    # a dashcam sits 1.0 .. 1.6 m above a road with 3.5 .. 3.7 m lanes
        rho = min(max(rho, RATIO_RANGE[0]), RATIO_RANGE[1])
    for r in rows:
        r["probs"] = None
        el, er = r.pop("extent_left", None), r.pop("extent_right", None)
        if rho is None or el is None or er is None:
            continue
        nl, nr = max(el / rho - 0.5, 0.0), max(er / rho - 0.5, 0.0)
        width = 1.0 + nl + nr
        r.update({"lanes_left": round(nl, 2), "lanes_right": round(nr, 2), "road_lanes": round(width, 2)})
        r["probs"] = count_probs(width, r.get("lines_lanes") or 1)
    return rows


def count_probs(width, floor=1):
    """Lane-count probabilities {1..6: p} around a measured road width in lanes; counts below `floor` (lines or vehicles seen)
    are unlikely."""
    p = {n: math.exp(-0.5 * ((n - width) / SIGMA_LANES) ** 2) * (1.0 if n >= floor else 0.3) for n in range(1, 7)}
    tot = sum(p.values())
    return {n: round(v / tot, 4) for n, v in p.items()}


LANE_W_M = 3.5


def vehicle_lane(det, on_road):
    """Lanes beside the ego lane that a detected vehicle stands in ("left", k) / ("right", k), or None. det: a detection of
    agents/vehicle_tracker with X (m, + right) and Z (m ahead); on_road: whether it stands on the ego carriageway (not an oncoming
    carriageway or a car park). Side views (a car crossing) and far or very near cars do not count."""
    x, z = float(det["X"]), float(det["Z"])
    if not on_road or not 4.0 <= z <= 40.0 or float(det.get("ar", 1.0)) >= 1.9 or abs(x) < 0.5 * LANE_W_M:
        return None
    return ("left" if x < 0 else "right", max(int(round(abs(x) / LANE_W_M)), 1))


def on_ego_road(frame, box, y_bottom, src_w=640, tol_lanes=0.7, ratio=3.0, vp_y=None):
    """Whether a vehicle box (x1, y1, x2, y2 in a picture src_w wide) stands on or right beside the ego car's connected road
    surface: its bottom centre lies within the surface's extent on that row, widened by tol_lanes lane widths (a queue of
    cars hides the surface of its own lane). Without the model: True."""
    reg = road_region(frame, y_bottom)
    if reg is None:
        return _model() is None
    h, w = reg.shape
    k = w / float(src_w)
    xb, yb = 0.5 * (box[0] + box[2]) * k, min(box[3] * k, y_bottom - 1)
    y = int(yb)
    rows = [yy for yy in range(max(y - 6, 0), min(y + 7, h)) if reg[yy].any()]
    if not rows:
        return False
    xs = np.concatenate([np.where(reg[yy])[0] for yy in rows])
    vy = vp_y if vp_y is not None else 0.42 * h
    tol = tol_lanes * ratio * max(yb - vy, 1.0)
    return bool(xs.min() - tol <= xb <= xs.max() + tol)


def apply_vehicle_evidence(rows, evidence, span=0.75):
    """Raise the lanes beside the ego lane to at least what the vehicles seen within `span` seconds show, and recompute the
    road width and the count probabilities. evidence: [{"t", "left": lanes, "right": lanes}]."""
    for r in rows:
        if r.get("road_lanes") is None:
            continue
        near = [e for e in evidence if abs(e["t"] - r["t"]) <= span]
        if not near:
            continue
        vl, vr = max(e["left"] for e in near), max(e["right"] for e in near)
        nl, nr = max(r["lanes_left"], vl), max(r["lanes_right"], vr)
        if (nl, nr) == (r["lanes_left"], r["lanes_right"]):
            continue
        width = 1.0 + nl + nr
        r.update({"lanes_left": round(nl, 2), "lanes_right": round(nr, 2), "road_lanes": round(width, 2),
                  "vehicle_lanes": [vl, vr]})
        r["probs"] = count_probs(width, max(r.get("lines_lanes") or 1, 1 + vl + vr))
    return rows


def lane_timeline(path, t0, t1, step=0.5, window=1.0, fps_window=8.0):
    """[{"t", "probs", "road_lanes", "lanes_left", "lanes_right", ...}] every `step` seconds in [t0, t1]; each window pools the
    frames of `window` seconds centred on t. [] when the video cannot be read."""
    y_b, h = road_bottom(path)
    if y_b is None:
        return []
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        dur = (cap.get(cv2.CAP_PROP_FRAME_COUNT) or 1) / fps
        out = []
        t = float(t0)
        n = max(int(window * fps_window), 1)
        while t <= float(t1) + 1e-6:
            ts = [min(max(t - 0.5 * window + window * (i + 0.5) / n, 0.0), dur - 0.05) for i in range(n)]
            frames = [f for f in (_read(cap, fps, x) for x in ts) if f is not None]
            r = analyse_window(frames, y_b) or {"lanes": None}
            r.pop("peaks", None)
            r["t"] = round(t, 2)
            out.append(r)
            t += step
        return finish_timeline(out)
    finally:
        cap.release()


def change_side(rows, t_before, t_after, span=2.0):
    """'left' or 'right': the side whose number of lanes beside the ego lane changes more between the windows before and after
    a change of the count (medians over `span` seconds on each side)."""
    def med(key, lo, hi):
        v = [r[key] for r in rows if r.get(key) is not None and lo <= r["t"] <= hi]
        return float(np.median(v)) if v else None
    l0, l1 = med("lanes_left", t_before - span, t_before), med("lanes_left", t_after, t_after + span)
    r0, r1 = med("lanes_right", t_before - span, t_before), med("lanes_right", t_after, t_after + span)
    if None in (l0, l1, r0, r1):
        return "right"
    return "left" if abs(l1 - l0) > abs(r1 - r0) else "right"


def draw_debug(path, t, out_jpg, window=1.0, fps_window=8.0):
    """Debug picture: marking mask, segments, VP and the pooled lane lines at time t."""
    y_b, h = road_bottom(path)
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = max(int(window * fps_window), 1)
    frames = [_read(cap, fps, t - 0.5 * window + window * (i + 0.5) / n) for i in range(n)]
    cap.release()
    frames = [f for f in frames if f is not None]
    r = analyse_window(frames, y_b)
    img = frames[len(frames) // 2].copy()
    m = marking_mask(img, int(0.3 * h), y_b)
    img[m > 0] = (0, 0, 255)
    for s in segments(m):
        cv2.line(img, (int(s[0]), int(s[1])), (int(s[2]), int(s[3])), (0, 255, 0), 2)
    cv2.line(img, (0, y_b), (img.shape[1], y_b), (255, 0, 255), 1)
    if r and r.get("vp"):
        vx, vy = r["vp"]
        cv2.circle(img, (int(vx), int(vy)), 6, (255, 255, 0), -1)
        for x, _ in r.get("peaks", []):
            xb = vx + (x - vx) * (y_b - vy) / max(r["y_ref"] - vy, 1.0)
            cv2.line(img, (int(vx), int(vy)), (int(xb), int(y_b)), (0, 255, 255), 1)
    cv2.putText(img, "t=%.1f lanes=%s ego=%s" % (t, r.get("lanes") if r else None, r.get("ego_from_left") if r else None),
                (10, 30), 0, 0.9, (0, 255, 255), 2)
    cv2.imwrite(out_jpg, img)
    return r
