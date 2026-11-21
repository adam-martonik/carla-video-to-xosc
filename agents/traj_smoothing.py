"""
Trajectory smoothing and kinematic limiting for recorded actor paths (numpy only, no CARLA).

Ported from carla-video-to-xosc (kinematics.py). That code works in a Frenet frame (s along the road, d across it);
AutoScenario has no such frame for its recorded runs, so everything here works on world x/y instead:

  1. Hampel (rolling-MAD) filter removes single-sample spikes.
  2. A Rauch-Tung-Striebel Kalman smoother (constant-acceleration model) smooths x(t) and y(t).
  3. Heading comes from the smoothed velocity, rate limited and held while the vehicle is (almost) standing.
  4. The speed along the smoothed path is limited: top speed, acceleration, braking, jerk, and a cornering cap
     v <= sqrt(a_lat / curvature). The path SHAPE is kept, only the timing along it changes.
"""
import math

import numpy as np

# Dynamic limits per class. max_accel / max_decel in m/s^2, max_jerk in m/s^3, max_speed in m/s.
# max_lat_accel is the comfortable-to-limit cornering acceleration used for the curve speed cap.
LIMITS = {
    "car":        {"max_speed": 45.0, "max_accel": 4.5, "max_decel": 9.5, "max_jerk": 8.0, "max_lat_accel": 6.0},
    "suv":        {"max_speed": 42.0, "max_accel": 4.0, "max_decel": 9.0, "max_jerk": 8.0, "max_lat_accel": 5.5},
    "van":        {"max_speed": 38.0, "max_accel": 3.0, "max_decel": 8.0, "max_jerk": 7.0, "max_lat_accel": 4.5},
    "truck":      {"max_speed": 30.0, "max_accel": 2.0, "max_decel": 7.0, "max_jerk": 6.0, "max_lat_accel": 3.5},
    "motorcycle": {"max_speed": 50.0, "max_accel": 6.0, "max_decel": 9.0, "max_jerk": 12.0, "max_lat_accel": 6.5},
    "bike":       {"max_speed": 15.0, "max_accel": 2.0, "max_decel": 4.5, "max_jerk": 6.0, "max_lat_accel": 3.0},
    "pedestrian": {"max_speed": 3.5, "max_accel": 2.0, "max_decel": 4.0, "max_jerk": 10.0, "max_lat_accel": 4.0},
}


def limits_for(vtype):
    return LIMITS.get(vtype, LIMITS["car"])


def hampel(values, window=11, thresh=3.0):
    """Replace single-sample spikes by the local median (rolling median absolute deviation). Input is not modified."""
    arr = np.asarray(values, dtype=float).copy()
    n = len(arr)
    if n < 5:
        return arr
    half = max(2, window // 2)
    out = arr.copy()
    for i in range(n):
        seg = arr[max(0, i - half):min(n, i + half + 1)]
        med = float(np.median(seg))
        sigma = 1.4826 * float(np.median(np.abs(seg - med)))
        if sigma > 1e-4 and abs(arr[i] - med) > thresh * sigma:
            out[i] = med
    return out


def rts_smoother_1d(times, observations, meas_std=0.3, process_std=0.5):
    """
    Fixed-interval Rauch-Tung-Striebel smoother with a constant-acceleration model x = [p, v, a].
    Handles non-uniform time steps. Returns (position, velocity, acceleration) arrays.
    """
    t = np.asarray(times, dtype=float)
    z = np.asarray(observations, dtype=float)
    n = len(t)
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    if n == 1:
        return z.copy(), np.zeros(1), np.zeros(1)

    r = np.full(n, float(meas_std) ** 2) if np.isscalar(meas_std) else np.maximum(np.asarray(meas_std, dtype=float), 1e-3) ** 2
    q = float(process_std) ** 2
    H = np.array([1.0, 0.0, 0.0])
    eye = np.eye(3)

    x_f, x_p = np.zeros((n, 3)), np.zeros((n, 3))
    P_f, P_p = np.zeros((n, 3, 3)), np.zeros((n, 3, 3))
    Fs = [None]
    x_f[0] = [z[0], (z[1] - z[0]) / max(t[1] - t[0], 1e-3), 0.0]
    P_f[0] = np.diag([r[0], 4.0, 2.0])

    for k in range(1, n):
        dt = max(float(t[k] - t[k - 1]), 1e-3)
        F = np.array([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]])
        Q = q * np.array([[dt ** 5 / 20.0, dt ** 4 / 8.0, dt ** 3 / 6.0],
                          [dt ** 4 / 8.0, dt ** 3 / 3.0, dt ** 2 / 2.0],
                          [dt ** 3 / 6.0, dt ** 2 / 2.0, dt]])
        Fs.append(F)
        xp = F @ x_f[k - 1]
        Pp = F @ P_f[k - 1] @ F.T + Q
        x_p[k], P_p[k] = xp, Pp
        K = (Pp @ H) / max(float(H @ Pp @ H) + r[k], 1e-9)
        x_f[k] = xp + K * (z[k] - xp[0])
        IKH = eye - np.outer(K, H)
        P_f[k] = IKH @ Pp @ IKH.T + r[k] * np.outer(K, K)      # Joseph form

    x_s = x_f.copy()
    for k in range(n - 2, -1, -1):
        try:
            G = P_f[k] @ Fs[k + 1].T @ np.linalg.inv(P_p[k + 1] + 1e-9 * eye)
        except np.linalg.LinAlgError:
            G = np.zeros((3, 3))
        x_s[k] = x_f[k] + G @ (x_s[k + 1] - x_p[k + 1])
    return x_s[:, 0], x_s[:, 1], x_s[:, 2]


def limit_speed_profile(times, v_target, max_speed, max_accel, max_decel, max_jerk, v_min=0.0):
    """
    Forward pass that makes a speed profile feasible: bounded speed, acceleration, braking and jerk
    (0.85 margin on the jerk bound for the discretisation). Returns the feasible speed array.
    """
    t = np.asarray(times, dtype=float)
    v_t = np.clip(np.asarray(v_target, dtype=float), v_min, max_speed)
    n = len(t)
    if n < 2:
        return v_t.copy()
    dt = np.maximum(np.diff(t), 1e-3)
    v = np.zeros(n)
    v[0] = v_t[0]
    j_max = 0.85 * max_jerk
    a_prev = float(np.clip((v_t[1] - v_t[0]) / dt[0], -max_decel, max_accel))
    for k in range(1, n):
        a_des = float(np.clip((v_t[k] - v[k - 1]) / dt[k - 1], -max_decel, max_accel))
        a_cur = float(np.clip(a_des, a_prev - j_max * dt[k - 1], a_prev + j_max * dt[k - 1]))
        v[k] = float(np.clip(v[k - 1] + a_cur * dt[k - 1], v_min, max_speed))
        a_prev = (v[k] - v[k - 1]) / dt[k - 1]
    return v


def _wrap(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def smooth_heading(times, vx, vy, h0=0.0, v_min=0.5, max_yaw_rate=1.2):
    """Heading from the velocity direction. Held while speed < v_min, rate limited to max_yaw_rate (rad/s)."""
    n = len(times)
    h = np.zeros(n)
    cur = float(h0)
    for k in range(n):
        if math.hypot(vx[k], vy[k]) >= v_min:
            raw = math.atan2(vy[k], vx[k])
            dt = max(float(times[k] - times[k - 1]), 1e-3) if k > 0 else 0.1
            step = max_yaw_rate * dt
            cur += float(np.clip(_wrap(raw - cur), -step, step))
        h[k] = cur
    return h


def _arc_lengths(x, y):
    return np.concatenate([[0.0], np.cumsum(np.hypot(np.diff(x), np.diff(y)))])


def smooth_trajectory(times, x, y, vtype="car", dt=0.1, meas_std=0.15, process_std=1.0, initial_heading=None):
    """
    Smooth one recorded path and make its timing kinematically feasible.

    times, x, y: raw samples (any spacing). Returns a dict with uniformly sampled
    times, x, y, h (heading, rad, world frame), v (speed, m/s) and max_deviation_m
    (largest distance between the raw path and the returned one, so the caller can see how much the limits changed).
    """
    t_raw = np.asarray(times, dtype=float)
    x_raw = np.asarray(x, dtype=float)
    y_raw = np.asarray(y, dtype=float)
    if len(t_raw) < 2:
        raise ValueError("a trajectory needs at least 2 samples")
    lim = limits_for(vtype)

    t = np.arange(t_raw[0], t_raw[-1] + 1e-9, dt)
    if t[-1] < t_raw[-1] - 1e-6:
        t = np.append(t, t_raw[-1])
    xs = hampel(np.interp(t, t_raw, x_raw))
    ys = hampel(np.interp(t, t_raw, y_raw))
    xs, vx, _ = rts_smoother_1d(t, xs, meas_std, process_std)
    ys, vy, _ = rts_smoother_1d(t, ys, meas_std, process_std)

    speed = np.hypot(vx, vy)
    s_path = _arc_lengths(xs, ys)
    if s_path[-1] < 0.05:                                   # not moving at all
        h0 = float(initial_heading) if initial_heading is not None else 0.0
        n = len(t)
        return {"times": t, "x": np.full(n, xs[0]), "y": np.full(n, ys[0]), "h": np.full(n, h0),
                "v": np.zeros(n), "max_deviation_m": 0.0}

    h = smooth_heading(t, vx, vy, h0=(math.atan2(vy[np.argmax(speed)], vx[np.argmax(speed)])
                                       if initial_heading is None else initial_heading))
    yaw_rate = np.abs(np.gradient(np.unwrap(h), t))
    kappa = yaw_rate / np.maximum(speed, 0.5)
    v_curve = np.sqrt(lim["max_lat_accel"] / np.maximum(kappa, 1e-6))
    v_lim = limit_speed_profile(t, np.minimum(speed, v_curve), lim["max_speed"], lim["max_accel"],
                                lim["max_decel"], lim["max_jerk"])

    # integrate the feasible speed along the smoothed path (trapezoid), then read the path back at that arc length
    s_new = np.concatenate([[0.0], np.cumsum(0.5 * (v_lim[1:] + v_lim[:-1]) * np.diff(t))])
    s_new = np.maximum(s_new, 0.0)
    s_ref, x_ref, y_ref = s_path, xs, ys
    if s_new[-1] > s_path[-1]:
        k = len(s_path) - 2
        while k > 0 and s_path[-1] - s_path[k] < 0.5:
            k -= 1
        dx, dy = float(xs[-1] - xs[k]), float(ys[-1] - ys[k])
        norm = math.hypot(dx, dy)
        if norm > 1e-6:
            s_ext = float(s_new[-1]) + 1.0
            extra = s_ext - float(s_path[-1])
            s_ref = np.append(s_path, s_ext)
            x_ref = np.append(xs, float(xs[-1]) + extra * dx / norm)
            y_ref = np.append(ys, float(ys[-1]) + extra * dy / norm)
        else:
            s_new = np.clip(s_new, 0.0, s_path[-1])
    x_out = np.interp(s_new, s_ref, x_ref)
    y_out = np.interp(s_new, s_ref, y_ref)
    v_out = np.hypot(np.gradient(x_out, t), np.gradient(y_out, t))
    dev = float(np.max(np.hypot(np.interp(t, t_raw, x_raw) - x_out, np.interp(t, t_raw, y_raw) - y_out)))
    return {"times": t, "x": x_out, "y": y_out, "h": h, "v": v_out, "max_deviation_m": dev}


COMFORT = {"max_accel": 2.5, "max_decel": 4.0, "max_jerk": 3.0}      # m/s^2, m/s^2, m/s^3: everyday driving


def smooth_speed_series(times, speeds, vtype="car", window_s=1.5):
    """A measured speed series (m/s) made fit for a driver to follow: single misreadings (e.g. an overlay digit read wrong)
    replaced by the local median, a centred moving average over window_s, then comfortable acceleration, braking and jerk
    limits. Returns an array of speeds (m/s) at the same times."""
    t = np.asarray(times, dtype=float)
    v = np.asarray(speeds, dtype=float)
    if len(v) < 3:
        return v.copy()
    v = hampel(v, window=7, thresh=2.5)
    dt = float(np.median(np.diff(t))) if len(t) > 1 else 0.5
    k = max(int(round(window_s / max(dt, 1e-3))), 1)
    if k > 1:
        pad = np.concatenate([np.full(k // 2, v[0]), v, np.full(k - 1 - k // 2, v[-1])])
        v = np.convolve(pad, np.ones(k) / k, mode="valid")
    lim = limits_for(vtype)
    return limit_speed_profile(t, v, lim["max_speed"], COMFORT["max_accel"], COMFORT["max_decel"], COMFORT["max_jerk"])


def smooth_offsets(values, window=5):
    """Lateral offsets with spikes removed and a short moving average (a car keeps its lane line; detection jitter does not)."""
    v = hampel(np.asarray(values, dtype=float), window=7, thresh=2.5)
    if len(v) < 3:
        return v
    k = min(window, len(v))
    pad = np.concatenate([np.full(k // 2, v[0]), v, np.full(k - 1 - k // 2, v[-1])])
    return np.convolve(pad, np.ones(k) / k, mode="valid")
