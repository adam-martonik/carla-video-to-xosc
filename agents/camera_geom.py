"""
Camera geometry and ego motion shared by the vehicle tracks (agents/vehicle_tracks.py) and the bird's-eye lane map
(agents/bev_lanes.py). Flat-road pinhole camera at the analysis width of agents/lane_lines (WIDTH px):

  a ground point X metres to the right and Z metres ahead of the camera is seen at
      u = vx + f * X / Z,      v = vy + f * H / Z
  (vx, vy) = vanishing point of the road (horizon row vy), f = focal length in px, H = camera height in m.

Calibration (calibrate()):
  vx, vy  from the lane detector's vanishing points (median over the video)
  H       from the lane-width ratio the lane detector measures: lane px / (y - vy) = lane width / H
  f       where the overlay prints the ego speed: from dashed lane lines (focal_from_dashes(): a dash pattern in a top view
          made with a focal guess moves between frames by the ego travel scaled by f_guess / f); otherwise a typical dashcam
          field of view. Lateral positions X = (u - vx) H / (v - vy) do not depend on f, so lanes and the lane a car is in are
          measured either way; f scales the distances ahead (Z), and so the speeds of other cars relative to the ego car.

The ego speed comes from the speed the video overlay prints (agents/local_pipeline); measuring it from the road motion was
tried and is not reliable on these videos (see calibrate()).
"""
import math

import cv2
import numpy as np

LANE_W_M = 3.5
DEFAULT_HFOV_DEG = 100.0        # horizontal field of view of a typical dashcam (the focal length cannot be measured here)
H_RANGE = (0.9, 2.2)            # plausible camera heights in m (car .. van / truck cab)
BEV_X = (-15.0, 15.0)           # lateral range of a top view (m, + right)
BEV_Z = (4.0, 40.0)             # range ahead (m)
BEV_RES = 0.1                   # m per top-view pixel


class Camera:
    def __init__(self, w, h, f, vx, vy, height_m):
        self.w, self.h, self.f, self.vx, self.vy, self.H = int(w), int(h), float(f), float(vx), float(vy), float(height_m)

    def to_dict(self):
        return {"w": self.w, "h": self.h, "f": round(self.f, 2), "vx": round(self.vx, 2), "vy": round(self.vy, 2),
                "H": round(self.H, 3)}

    @classmethod
    def from_dict(cls, d):
        return cls(d["w"], d["h"], d["f"], d["vx"], d["vy"], d["H"])

    def ground_from_pixel(self, u, v, min_dv=2.0):
        """(X right, Z ahead) in m of the ground point seen at pixel (u, v), or None above (or too near) the horizon."""
        dv = float(v) - self.vy
        if dv < min_dv:
            return None
        z = self.f * self.H / dv
        return (float(u) - self.vx) * z / self.f, z

    def pixel_from_ground(self, x, z):
        """Pixel (u, v) where the ground point (X right, Z ahead) is seen, or None behind the camera."""
        if z <= 0.1:
            return None
        return self.vx + self.f * x / z, self.vy + self.f * self.H / z

    def lane_px(self, v, lane_w=LANE_W_M):
        """Width in px of a lane on row v."""
        return lane_w * max(float(v) - self.vy, 0.0) / self.H


def default_focal(w, hfov_deg=DEFAULT_HFOV_DEG):
    return 0.5 * w / math.tan(math.radians(0.5 * hfov_deg))


def height_from_ratio(ratio):
    """Camera height from the lane-width ratio (lane px per px below the horizon), clamped to a plausible range."""
    if not ratio or ratio <= 0:
        return 1.3
    return min(max(LANE_W_M / float(ratio), H_RANGE[0]), H_RANGE[1])


# ----------------------------------------------------------------------------------------- top view of the road
def bev_homography(cam, x_range=BEV_X, z_range=BEV_Z, res=BEV_RES):
    """(3x3 homography from top-view pixel to image pixel, (width, height)) for a top view of the road: column i is
    X = x0 + i res (right), row j is Z = z_far - j res (far at the top)."""
    x0, x1 = x_range
    z0, z1 = z_range
    size = (int(round((x1 - x0) / res)), int(round((z1 - z0) / res)))
    A = np.array([[res, 0.0, x0], [0.0, -res, z1], [0.0, 0.0, 1.0]])                  # top-view pixel -> (X, Z, 1)
    M = np.array([[cam.f, cam.vx, 0.0], [0.0, cam.vy, cam.f * cam.H], [0.0, 1.0, 0.0]])  # (X, Z, 1) -> image (homogeneous)
    return M @ A, size


def to_bev(img, cam, x_range=BEV_X, z_range=BEV_Z, res=BEV_RES, interp=cv2.INTER_LINEAR):
    Hm, size = bev_homography(cam, x_range, z_range, res)
    return cv2.warpPerspective(img, Hm, size, flags=interp | cv2.WARP_INVERSE_MAP, borderValue=0)


def calibrate(lane_rows, frame_hw, hfov_deg=None):
    """Camera for a video from the lane detector's windows (vp, ratio). The focal length is a typical dashcam field of view:
    measuring it from the ego car's motion against the overlay speed was tried (top-view phase correlation, dash corners) and
    is not reliable on these low-resolution videos. Returns (Camera, note)."""
    h, w = frame_hw
    vps = [r["vp"] for r in lane_rows if r.get("vp")]
    ratios = [r["ratio"] for r in lane_rows if r.get("ratio")]
    vx = float(np.median([p[0] for p in vps])) if vps else 0.5 * w
    vy = float(np.median([p[1] for p in vps])) if vps else 0.42 * h
    H = height_from_ratio(float(np.median(ratios)) if ratios else None)
    hfov = hfov_deg or DEFAULT_HFOV_DEG
    f = default_focal(w, hfov)
    cam = Camera(w, h, f, vx, vy, H)
    return cam, "camera: horizon row %.0f (from %d windows), height %.2f m (lane ratio), focal %.0f px (%.0f deg field of view)" % (
        vy, len(vps), H, f, hfov)


def travel(speeds, t0):
    """Distance driven since video time t0 as a function f(t) (trapezoid over the speed series; constant speed outside it)."""
    if not speeds:
        return lambda t: 0.0
    ts = np.array([t for t, _ in speeds], dtype=float)
    vs = np.array([v for _, v in speeds], dtype=float)
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (vs[1:] + vs[:-1]) * np.diff(ts))])

    def s_at(t):
        t = float(t)
        if t <= ts[0]:
            return (t - ts[0]) * vs[0]
        if t >= ts[-1]:
            return cum[-1] + (t - ts[-1]) * vs[-1]
        i = int(np.searchsorted(ts, t, side="right")) - 1              # speed linear between samples: exact integral
        dt = t - ts[i]
        a = (vs[i + 1] - vs[i]) / max(ts[i + 1] - ts[i], 1e-9)
        return float(cum[i] + vs[i] * dt + 0.5 * a * dt * dt)
    base = s_at(t0)
    return lambda t: s_at(t) - base


class RoadCurve:
    """The road's bend along the ego car's path (the scene's centre line, which starts where the simulated stretch starts).
    A point Z metres ahead along the camera's line of sight, with the car at travel s, lies off the curved road's centre line
    by offset(s, Z) = integral_0^Z (psi(s + u) - psi(s)) du  (psi = road heading; + when the road bends left), so its offset
    across the road is d = -X - offset(s, Z). Straight road (no centre line): offset 0."""

    def __init__(self, centerline=None, step=1.0):
        self.ok = bool(centerline) and len(centerline) >= 3
        if not self.ok:
            return
        from agents.road_frame import RoadFrame
        fr = RoadFrame(centerline)
        self.s = np.arange(0.0, fr.length + step, step)
        psi = np.unwrap(np.radians([fr.heading(x) for x in self.s]))
        self.psi = psi
        self.I = np.concatenate([[0.0], np.cumsum(0.5 * (psi[1:] + psi[:-1]) * np.diff(self.s))])

    def offset(self, s, Z):
        if not self.ok:
            return np.zeros_like(np.asarray(Z, dtype=float))
        s = np.asarray(s, dtype=float)
        Z = np.asarray(Z, dtype=float)
        return np.interp(s + Z, self.s, self.I) - np.interp(s, self.s, self.I) - np.interp(s, self.s, self.psi) * Z


def vx_series(lane_rows, default):
    """vx(t): the vanishing point's column over time from the lane detector's windows (median of the windows within 1 s, so a
    single bad window does not tilt a frame), the video's median where none is near."""
    pts = sorted((r["t"], r["vp"][0]) for r in lane_rows if r.get("vp"))
    if not pts:
        return lambda t: default

    def vx(t):
        near = [x for tt, x in pts if abs(tt - t) <= 1.0]
        return float(np.median(near)) if near else default
    return vx


# ----------------------------------------------------------------------------------------- focal length from dashed lines
DASH_RES = 0.05               # m per top-view row
DASH_X = (-12.0, 12.0)
DASH_Z = (6.0, 26.0)
DASH_DT = 0.1                 # s between the two frames compared
DASH_MIN_CORR = 0.95          # only clear matches of the dash pattern


def dash_shift(profile_a, profile_b, max_shift):
    """(shift in rows, correlation) of a 1-D dash pattern along the road between two frames: forward motion moves road marks
    toward the camera, so b is a shifted by k >= 1 rows. (0, -1) when there is no clear match."""
    a = (profile_a - profile_a.mean()) / (profile_a.std() + 1e-6)
    b = (profile_b - profile_b.mean()) / (profile_b.std() + 1e-6)
    best, best_k = -1e9, 0
    for k in range(1, max_shift):
        n = len(a) - k
        if n < 20:
            break
        r = float(np.dot(a[:n], b[k:k + n]) / n)
        if r > best:
            best, best_k = r, k
    return best_k, best


def focal_from_dashes(path, cam, times, speed_fn, y_bottom, vx_at=None):
    """Focal length from dashed lane lines: in a top view made with cam's focal guess f0, a dashed line's pattern moves between
    two frames by the ego car's travel scaled by f0 / f, so f = f0 * V * dt / shift. Columns of the top view that hold a dashed
    line (bright, varying strongly along the road; a solid line does not vary) are matched between frames DASH_DT apart wherever
    the overlay gives the speed V. Returns (f or None, number of clear matches)."""
    import copy
    from agents import lane_lines
    crop = lane_lines._video_crop(path)
    ests = []
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = max(int(round(DASH_DT * fps)), 1)
        for t in times:
            V = speed_fn(t)
            if V is None or V < 4.0:
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(t, 0.0) * fps))
            ok, a = cap.read()
            b = None
            for _ in range(n):
                ok, b = cap.read()
            if a is None or b is None:
                continue
            c = copy.copy(cam)
            if vx_at is not None:
                c.vx = vx_at(t)
            views = []
            for img in (a, b):
                if crop is not None:
                    img = img[crop[0]:crop[1], crop[2]:crop[3]]
                img = cv2.resize(img, (cam.w, int(round(img.shape[0] * cam.w / img.shape[1]))))
                g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                th = cv2.morphologyEx(g, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_RECT, (31, 1))).astype(np.float32)
                th[int(y_bottom):] = 0
                views.append(to_bev(th, c, DASH_X, DASH_Z, DASH_RES))
            A, B = views
            strength = A.mean(0)
            for i in range(3, A.shape[1] - 3):
                if strength[i] != strength[i - 3:i + 4].max() or strength[i] <= 4.0:
                    continue
                pa, pb = A[:, i - 2:i + 3].mean(1), B[:, i - 2:i + 3].mean(1)
                if pa.std() < 0.35 * pa.mean():
                    continue                                     # a solid line: no dash pattern to follow
                k, r = dash_shift(pa, pb, int(4.0 / DASH_RES))
                if r >= DASH_MIN_CORR and k > 0:
                    ests.append(cam.f * V * (n / fps) / (k * DASH_RES))
    finally:
        cap.release()
    if len(ests) < 20:
        return None, len(ests)
    f = float(np.median(ests))
    return (f if 0.4 * cam.w <= f <= 1.6 * cam.w else None), len(ests)
