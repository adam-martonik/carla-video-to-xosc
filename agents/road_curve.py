"""
Road curvature from the dashcam video, with no model and no download.

The ego car follows its lane, so its path is the road centreline. The yaw rate comes from the horizontal optical flow of the
far field (centre columns of a band above the hood), where the flow of straight driving is near zero and a turn of the camera
shifts the whole picture sideways. With the ego speed the yaw rate integrates to a centreline.

Frame of the result: net frame (x east, y north), starting at (0, 0) heading east, a LEFT bend goes to +y.
"""
import math

import cv2
import numpy as np

# calibration constants (tune here)
FOV_DEG = 70.0                 # assumed horizontal field of view of the dashcam
BAND = (0.12, 0.50)            # rows of the frame used (fractions of the height): above the hood, below the overlay
CENTER_COLS = 120              # columns around the picture centre (at WIDTH) used for the median flow
WIDTH = 320
ANALYSIS_FPS = 15.0
MOVING_FLOW_PX = 0.3           # overall median flow above this means the ego car is moving
MAX_YAW_RATE = 0.5             # rad/s; larger values are noise
SMOOTH_S = 1.0
MIN_SPEED_MS = 2.0             # below this the yaw rate says nothing about the road
LOW_CONFIDENCE = 0.3


def _content_crop(cap, n):
    probe = []
    for k in range(5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k * max(n - 1, 1) / 5))
        ok, fr = cap.read()
        if ok:
            probe.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY).astype(np.float32))
    if not probe:
        return None
    m = np.mean(probe, axis=0)
    rows = np.nonzero(m.mean(axis=1) > 12)[0]
    cols = np.nonzero(m.mean(axis=0) > 12)[0]
    if len(rows) and len(cols):
        return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1
    return None


def estimate_ego_yaw(video, t0, t1, fov_deg=FOV_DEG):
    """{"t": [s], "yaw_rate": [rad/s, + = left], "speed": [None...], "confidence": 0..1} over [t0, t1] of the video."""
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    crop = _content_crop(cap, n)
    stride = max(int(round(fps / ANALYSIS_FPS)), 1)
    dt = stride / fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(t0, 0.0) * fps))
    first = int(max(t0, 0.0) * fps)
    last = int(t1 * fps)
    f_px = (WIDTH / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    prev, ts, yaw, moving = None, [], [], []
    idx = first
    while idx <= last:
        if not cap.grab():
            break
        if (idx - first) % stride == 0:
            ok, fr = cap.retrieve()
            if not ok:
                break
            if crop is not None:
                fr = fr[crop[0]:crop[1], crop[2]:crop[3]]
            h = fr.shape[0]
            fr = fr[int(h * BAND[0]):int(h * BAND[1])]
            fr = cv2.resize(fr, (WIDTH, max(int(fr.shape[0] * WIDTH / fr.shape[1]), 8)))
            g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            if prev is not None:
                flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                cx = g.shape[1] // 2
                fx = float(np.median(flow[:, cx - CENTER_COLS // 2:cx + CENTER_COLS // 2, 0]))
                mag = float(np.median(np.hypot(flow[..., 0], flow[..., 1])))
                # the camera turning left shifts the picture to the right
                yaw.append(fx / (f_px * dt))
                moving.append(mag > MOVING_FLOW_PX)
                ts.append(idx / fps)
            prev = g
        idx += 1
    cap.release()
    if not ts:
        return {"t": [], "yaw_rate": [], "speed": [], "confidence": 0.0}
    ok_frac = float(np.mean([m and abs(w) < MAX_YAW_RATE for m, w in zip(moving, yaw)]))
    yaw = [w if abs(w) < MAX_YAW_RATE else 0.0 for w in yaw]
    return {"t": ts, "yaw_rate": yaw, "speed": [None] * len(ts), "confidence": round(ok_frac, 3)}


def _smooth(x, k):
    if k <= 1 or len(x) < 2:
        return list(x)
    out = []
    for i in range(len(x)):
        lo, hi = max(0, i - k // 2), min(len(x), i + k // 2 + 1)
        out.append(sum(x[lo:hi]) / (hi - lo))
    return out


def centerline_from_yaw(t, yaw_rate, speed, min_radius_m=150.0, length_m=300.0, default_speed_ms=14.0):
    """Integrate yaw rate and speed into {"points": [[x, y]...] every 2 m, "confidence", "radius_min_m"}."""
    n = len(t)
    if n == 0:
        return {"points": [[0.0, 0.0], [float(length_m), 0.0]], "confidence": 0.0, "radius_min_m": None}
    dts = [t[i + 1] - t[i] for i in range(n - 1)] or [0.5]
    med_dt = sorted(dts)[len(dts) // 2] or 0.5
    w = _smooth([float(x) for x in yaw_rate], max(int(round(SMOOTH_S / med_dt)), 1))
    v = [default_speed_ms if s is None else float(s) for s in speed]
    kmax = 1.0 / float(min_radius_m)
    x = y = h = 0.0
    xs, ys, cum = [0.0], [0.0], [0.0]
    kabs = 0.0
    n_moving = 0
    for i in range(n):
        dt = dts[i] if i < n - 1 else med_dt
        if v[i] >= MIN_SPEED_MS:
            n_moving += 1
            k = max(-kmax, min(kmax, w[i] / v[i]))
            kabs = max(kabs, abs(k))
            ds = v[i] * dt
            h += k * ds
            x += ds * math.cos(h)
            y += ds * math.sin(h)
            xs.append(x)
            ys.append(y)
            cum.append(cum[-1] + ds)
    total = cum[-1]
    if total < length_m:                      # continue straight with the last heading
        xs.append(x + (length_m - total) * math.cos(h))
        ys.append(y + (length_m - total) * math.sin(h))
        cum.append(float(length_m))
    grid = np.arange(0.0, length_m + 1e-9, 2.0)
    px = np.interp(grid, cum, xs)
    py = np.interp(grid, cum, ys)
    pts = [[round(float(a), 3), round(float(b), 3)] for a, b in zip(px, py)]
    return {"points": pts, "confidence": round(n_moving / n, 3), "radius_min_m": (1.0 / kabs) if kabs > 1e-9 else None}


def _bend(points):
    y_end = points[-1][1]
    return "left" if y_end > 2.0 else "right" if y_end < -2.0 else "straight"


def _straight_like(curve, reason):
    length = 0.0
    for a, b in zip(curve["points"], curve["points"][1:]):
        length += math.hypot(b[0] - a[0], b[1] - a[1])
    out = dict(curve)
    out["points"] = [[0.0, 0.0], [round(length, 3), 0.0]]
    out["radius_min_m"] = None
    out["fallback"] = reason
    return out


def reconcile_with_probe(curve, probe_direction):
    """Keep the measured bend unless the evidence is weak or the model's left/right answer contradicts it."""
    if curve.get("confidence", 0.0) < LOW_CONFIDENCE:
        return _straight_like(curve, "weak yaw estimate (confidence %.2f): straight road" % curve.get("confidence", 0.0))
    bend = _bend(curve["points"])
    if bend in ("left", "right") and probe_direction in ("left", "right") and probe_direction != bend:
        return _straight_like(curve, "yaw says %s, the model says %s: straight road" % (bend, probe_direction))
    return curve
