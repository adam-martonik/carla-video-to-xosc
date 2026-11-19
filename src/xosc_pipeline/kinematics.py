"""Trajectory smoothing (RTS Kalman), kinematic feasibility (bicycle model), and deconfliction."""

from __future__ import annotations

import math
from typing import Any, Dict, List, Tuple

import numpy as np

from .config import CARLA_COLOR_PALETTE, ClassSpec, KinematicsConfig, PipelineConfig


def reject_outliers_mad(values: np.ndarray, window: int = 11, thresh: float = 3.0) -> np.ndarray:
    """
    Hampel / rolling Median Absolute Deviation (MAD) filter that replaces single-frame
    depth or lateral spikes with the local median without mutating the input array.
    """
    arr = np.asarray(values, dtype=float).copy()
    n = len(arr)
    if n < 5:
        return arr
    half = max(2, window // 2)
    out = arr.copy()
    for i in range(n):
        lo = max(0, i - half)
        hi = min(n, i + half + 1)
        seg = arr[lo:hi]
        med = float(np.median(seg))
        mad = float(np.median(np.abs(seg - med)))
        sigma = 1.4826 * mad
        if sigma > 1e-4 and abs(arr[i] - med) > thresh * sigma:
            out[i] = med
    return out


def rts_smoother_1d(
    times: np.ndarray,
    observations: np.ndarray,
    meas_std: float | np.ndarray,
    process_std: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fixed-interval Rauch-Tung-Striebel (RTS) Kalman smoother (Rauch, Tung & Striebel, 1965)
    with a 3-state Constant-Acceleration (CA) model x = [p, v, a]^T.

    Supports non-uniform time steps dt_k = times[k] - times[k-1].
    Returns (pos, vel, acc) smoothed arrays of length N.
    """
    t = np.asarray(times, dtype=float)
    z = np.asarray(observations, dtype=float)
    n = len(t)
    if n == 0:
        return np.array([]), np.array([]), np.array([])
    if n == 1:
        return z.copy(), np.zeros(1), np.zeros(1)

    if np.isscalar(meas_std):
        r_arr = np.full(n, float(meas_std) ** 2, dtype=float)
    else:
        r_arr = np.maximum(np.asarray(meas_std, dtype=float), 1e-3) ** 2

    # Forward Kalman filter
    x_filt = np.zeros((n, 3), dtype=float)
    P_filt = np.zeros((n, 3, 3), dtype=float)
    x_pred = np.zeros((n, 3), dtype=float)
    P_pred = np.zeros((n, 3, 3), dtype=float)
    F_list: List[np.ndarray] = []

    v0 = (z[1] - z[0]) / max(t[1] - t[0], 1e-3)
    x_filt[0] = np.array([z[0], v0, 0.0])
    P_filt[0] = np.diag([r_arr[0], 4.0, 2.0])

    H = np.array([[1.0, 0.0, 0.0]])
    I3 = np.eye(3)
    q = float(process_std) ** 2

    for k in range(1, n):
        dt = max(float(t[k] - t[k - 1]), 1e-3)
        F = np.array([
            [1.0, dt, 0.5 * dt * dt],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0],
        ])
        # Continuous white-noise jerk discrete process covariance
        Q = q * np.array([
            [dt ** 5 / 20.0, dt ** 4 / 8.0, dt ** 3 / 6.0],
            [dt ** 4 / 8.0, dt ** 3 / 3.0, dt ** 2 / 2.0],
            [dt ** 3 / 6.0, dt ** 2 / 2.0, dt],
        ])
        F_list.append(F)

        xp = F @ x_filt[k - 1]
        Pp = F @ P_filt[k - 1] @ F.T + Q
        x_pred[k] = xp
        P_pred[k] = Pp

        S = float((H @ Pp @ H.T)[0, 0]) + r_arr[k]
        K = (Pp @ H.T)[:, 0] / max(S, 1e-9)
        innov = z[k] - xp[0]
        x_filt[k] = xp + K * innov
        # Joseph form for numerical stability
        IKH = I3 - np.outer(K, H[0])
        P_filt[k] = IKH @ Pp @ IKH.T + r_arr[k] * np.outer(K, K)

    # Backward RTS smoothing pass
    x_smooth = x_filt.copy()
    P_smooth = P_filt.copy()
    for k in range(n - 2, -1, -1):
        F = F_list[k]
        Pp_next = P_pred[k + 1]
        try:
            G = P_filt[k] @ F.T @ np.linalg.inv(Pp_next + 1e-9 * I3)
        except np.linalg.LinAlgError:
            G = np.zeros((3, 3))
        x_smooth[k] = x_filt[k] + G @ (x_smooth[k + 1] - x_pred[k + 1])
        P_smooth[k] = P_filt[k] + G @ (P_smooth[k + 1] - Pp_next) @ G.T

    return x_smooth[:, 0], x_smooth[:, 1], x_smooth[:, 2]


def enforce_kinematic_limits(
    times: np.ndarray,
    s_raw: np.ndarray,
    d_raw: np.ndarray,
    spec: ClassSpec,
    allow_reverse: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Project a Frenet trajectory (s, d) onto the kinematically feasible set defined by
    a bicycle model (Kong et al., 2015; Polack et al., 2017):
      - speed bounds [0, v_max] (or [-v_max, v_max] if allow_reverse for oncoming vehicles)
      - longitudinal acceleration bounds [-max_decel, +max_accel]
      - longitudinal jerk bound [-max_jerk, +max_jerk]
      - bicycle steering curvature / lateral rate bound |dd/dt| <= v * tan(delta_max) * 0.5

    Returns mutually consistent (s_out, d_out, v_out).
    """
    t = np.asarray(times, dtype=float)
    s = np.asarray(s_raw, dtype=float)
    d = np.asarray(d_raw, dtype=float)
    n = len(t)
    if n < 2:
        return s.copy(), d.copy(), np.zeros(n)

    dt = np.diff(t)
    dt_safe = np.maximum(dt, 1e-3)

    # 1. Compute target longitudinal velocity and clamp to feasible speed range
    v_target = np.gradient(s, t)
    v_min = -spec.max_speed if allow_reverse else 0.0
    v_target = np.clip(v_target, v_min, spec.max_speed)

    # 2. Forward jerk & acceleration-limited velocity profile (with 0.85 discretization margin on jerk)
    v_lim = np.zeros(n, dtype=float)
    v_lim[0] = v_target[0]
    max_a = spec.max_accel
    max_dec = spec.max_decel
    max_j = 0.85 * spec.max_jerk
    a_prev = float(np.clip((v_target[1] - v_target[0]) / dt_safe[0], -max_dec, max_a)) if n > 1 else 0.0

    for k in range(1, n):
        dtk = dt_safe[k - 1]
        a_des = (v_target[k] - v_lim[k - 1]) / dtk
        a_des = float(np.clip(a_des, -max_dec, max_a))
        da_max = max_j * dtk
        a_cur = float(np.clip(a_des, a_prev - da_max, a_prev + da_max))
        v_next = float(np.clip(v_lim[k - 1] + a_cur * dtk, v_min, spec.max_speed))
        a_prev = (v_next - v_lim[k - 1]) / dtk
        v_lim[k] = v_next

    # Re-integrate s from feasible velocity using trapezoidal rule, anchored at s[0]
    s_out = np.zeros(n, dtype=float)
    s_out[0] = s[0]
    for k in range(1, n):
        s_out[k] = s_out[k - 1] + 0.5 * (v_lim[k - 1] + v_lim[k]) * dt_safe[k - 1]

    # 3. Bicycle-model lateral smoothing & curvature/yaw-rate limiting
    d_out = np.zeros(n, dtype=float)
    d_out[0] = d[0]
    vy_prev = 0.0
    max_lat_a = 4.5 if spec.category in ("car", "motorbike") else 2.5
    max_lat_v_abs = 2.8 if spec.category in ("car", "motorbike") else 1.8

    for k in range(1, n):
        dtk = dt_safe[k - 1]
        v_mag = max(abs(v_lim[k]), 0.5)
        # Bicycle model max heading angle rate -> max lateral velocity change
        max_yaw_rate = (v_mag / max(spec.wheelbase, 1.0)) * math.tan(min(spec.max_steering, 0.55))
        # Lateral velocity cannot exceed v * sin(25 deg) or max_lat_v_abs
        v_lat_cap = min(max_lat_v_abs, v_mag * 0.42, v_mag * max_yaw_rate * 0.8 + 0.3)
        vy_des = (d[k] - d_out[k - 1]) / dtk
        vy_des = float(np.clip(vy_des, -v_lat_cap, v_lat_cap))
        dvy_max = max_lat_a * dtk
        vy_cur = float(np.clip(vy_des, vy_prev - dvy_max, vy_prev + dvy_max))
        d_out[k] = d_out[k - 1] + vy_cur * dtk
        vy_prev = vy_cur

    # Recompute exact scalar speed magnitude from consistent (s_out, d_out)
    ds = np.gradient(s_out, t)
    dd = np.gradient(d_out, t)
    v_out = np.hypot(ds, dd)
    return s_out, d_out, v_out


def extrapolate_trajectory(
    obs_times: np.ndarray,
    obs_vals: np.ndarray,
    sample_times: np.ndarray,
    max_vel: float,
    min_vel: float = -35.0,
) -> np.ndarray:
    """
    Interpolate within [t_start, t_end] and extrapolate outside using bounded boundary
    velocity instead of clamping to a frozen constant position.
    """
    t_obs = np.asarray(obs_times, dtype=float)
    y_obs = np.asarray(obs_vals, dtype=float)
    t_out = np.asarray(sample_times, dtype=float)
    if len(t_obs) < 2:
        return np.full_like(t_out, y_obs[0] if len(y_obs) else 0.0)

    # Estimate boundary velocities from first/last ~0.5s window
    k_win = min(max(2, len(t_obs) // 5), 10)
    v_start = float(np.clip((y_obs[k_win - 1] - y_obs[0]) / max(t_obs[k_win - 1] - t_obs[0], 0.1), min_vel, max_vel))
    v_end = float(np.clip((y_obs[-1] - y_obs[-k_win]) / max(t_obs[-1] - t_obs[-k_win], 0.1), min_vel, max_vel))

    out = np.interp(t_out, t_obs, y_obs)
    before = t_out < t_obs[0]
    after = t_out > t_obs[-1]
    out[before] = y_obs[0] + v_start * (t_out[before] - t_obs[0])
    out[after] = y_obs[-1] + v_end * (t_out[after] - t_obs[-1])
    return out


def deconflict_trajectories(
    ego: Dict[str, Any],
    actors: List[Dict[str, Any]],
    cfg: PipelineConfig,
) -> List[Dict[str, Any]]:
    """
    Resolve unintended bounding-box overlaps between ego and NPC vehicles (and between NPCs)
    across all timestamps in Frenet (s, d) space using smooth constant longitudinal shifts
    so kinematic smoothness (a, jerk) is preserved. Intentional collision obstacles are skipped.
    """
    kin = cfg.kinematics
    ego_spec = cfg.get_class_spec(2)
    resolved: List[Dict[str, Any]] = []

    def min_required_gaps(spec_a: ClassSpec, spec_b: ClassSpec) -> Tuple[float, float]:
        req_s = 0.5 * (spec_a.length + spec_b.length) + kin.min_safety_margin_long_m
        req_d = 0.5 * (spec_a.width + spec_b.width) + kin.min_safety_margin_lat_m
        return req_s, req_d

    def has_overlap(
        s1: np.ndarray, d1: np.ndarray, spec1: ClassSpec,
        s2: np.ndarray, d2: np.ndarray, spec2: ClassSpec,
    ) -> Tuple[bool, float]:
        req_s, req_d = min_required_gaps(spec1, spec2)
        ds = np.abs(s1 - s2)
        dd = np.abs(d1 - d2)
        overlap_mask = (ds < req_s) & (dd < req_d)
        if not np.any(overlap_mask):
            return False, 0.0
        # Compute minimum shift needed to separate cleanly ahead
        needed = float(np.max(req_s - ds[overlap_mask])) + 0.5
        return True, needed

    placed = [(np.asarray(ego["s"], dtype=float), np.asarray(ego["d"], dtype=float), ego_spec)]

    for actor in actors:
        actor_copy = dict(actor)
        if actor_copy.get("is_collision_target", False) or actor_copy.get("entity_kind") == "pedestrian":
            resolved.append(actor_copy)
            continue

        spec = cfg.get_class_spec(int(actor_copy.get("class_id", 2)))
        s_arr = np.asarray(actor_copy["s"], dtype=float).copy()
        d_arr = np.asarray(actor_copy["d"], dtype=float).copy()

        for _ in range(kin.deconflict_iterations):
            any_hit = False
            for p_s, p_d, p_spec in placed:
                n_min = min(len(s_arr), len(p_s))
                hit, needed = has_overlap(s_arr[:n_min], d_arr[:n_min], spec, p_s[:n_min], p_d[:n_min], p_spec)
                if hit:
                    any_hit = True
                    # Shift actor ahead (or behind if it's already behind) by constant offset to preserve v, a, jerk
                    mean_rel = float(np.mean(s_arr[:n_min] - p_s[:n_min]))
                    direction = 1.0 if mean_rel >= -1.0 else -1.0
                    s_arr = s_arr + direction * needed
            if not any_hit:
                break

        actor_copy["s"] = s_arr.tolist()
        actor_copy["d"] = d_arr.tolist()
        placed.append((s_arr, d_arr, spec))
        resolved.append(actor_copy)

    return resolved


def smooth_and_resample_trajectories(
    extracted_data: Dict[str, Any],
    cfg: PipelineConfig,
    sample_dt: float | None = None,
) -> Dict[str, Any]:
    """
    Convert raw tracked ego-motion and object observations into physically consistent,
    RTS-smoothed, kinematically constrained, collision-free Frenet trajectories.
    """
    kin = cfg.kinematics
    dt = float(sample_dt if sample_dt is not None else kin.sample_dt_s)
    ego_rec = extracted_data["ego"]
    t_ego = np.array([r["time"] for r in ego_rec], dtype=float)
    duration = float(t_ego[-1]) if len(t_ego) > 0 else float(extracted_data["duration"])
    n_steps = max(2, int(round(duration / dt)) + 1)
    sample_times = np.linspace(0.0, duration, n_steps)

    # 1. Ego trajectory: outlier rejection + RTS smoothing on speed -> integrate s
    v_ego_raw = np.array([r["speed"] for r in ego_rec], dtype=float)
    yaw_ego_raw = np.array([r["yaw"] for r in ego_rec], dtype=float)

    v_ego_clean = reject_outliers_mad(v_ego_raw, kin.outlier_mad_window, kin.outlier_mad_thresh)
    v_ego_rts, _, _ = rts_smoother_1d(t_ego, v_ego_clean, kin.rts_ego_meas_noise_v, kin.rts_ego_process_noise_a)
    v_ego_rts = np.clip(v_ego_rts, 0.0, cfg.get_class_spec(2).max_speed)

    # Integrate ego longitudinal distance s from RTS-smoothed speed
    s_ego_raw = np.zeros_like(t_ego)
    for k in range(1, len(t_ego)):
        s_ego_raw[k] = s_ego_raw[k - 1] + 0.5 * (v_ego_rts[k - 1] + v_ego_rts[k]) * max(t_ego[k] - t_ego[k - 1], 1e-3)

    # Ego lateral offset d (small lane drift only; road curvature is carried by yaw_rate -> .xodr)
    d_ego_raw = np.zeros_like(t_ego)
    ego_s_interp = np.interp(sample_times, t_ego, s_ego_raw)
    ego_d_interp = np.interp(sample_times, t_ego, d_ego_raw)
    ego_yaw_interp = np.interp(sample_times, t_ego, yaw_ego_raw)
    _, ego_yaw_rate, _ = rts_smoother_1d(sample_times, ego_yaw_interp, meas_std=0.05, process_std=0.04)

    collision_ev = extracted_data.get("collision_event")
    ego_spec = cfg.get_class_spec(2)

    if collision_ev is not None:
        # Data-driven collision event: preserve actual detected impact time & pre-impact cruise speed
        t_imp = float(collision_ev["t_impact"])
        pre_mask = t_ego <= max(1.0, t_imp - 0.5)
        v_cruise = float(np.clip(np.median(v_ego_rts[pre_mask]) if np.any(pre_mask) else 16.0, 10.0, 25.0))
        t_stop = min(duration - 0.5, t_imp + max(2.8, v_cruise / 6.0))

        ego_v_prof = np.zeros_like(sample_times)
        ego_d_prof = np.zeros_like(sample_times)
        for i, t in enumerate(sample_times):
            if t <= t_imp:
                ego_v_prof[i] = v_cruise
                ego_d_prof[i] = 0.0
            elif t <= t_stop:
                frac = (t - t_imp) / max(0.5, (t_stop - t_imp))
                # Smooth S-curve deceleration (zero jerk at endpoints)
                s_curve = 3 * frac ** 2 - 2 * frac ** 3
                ego_v_prof[i] = max(0.0, v_cruise * (1.0 - s_curve))
                ego_d_prof[i] = 2.4 * s_curve
            else:
                ego_v_prof[i] = 0.0
                ego_d_prof[i] = 2.4

        ego_s_prof = np.zeros_like(sample_times)
        for i in range(1, len(sample_times)):
            ego_s_prof[i] = ego_s_prof[i - 1] + 0.5 * (ego_v_prof[i - 1] + ego_v_prof[i]) * (sample_times[i] - sample_times[i - 1])
        ego_s_smooth, ego_d_smooth, ego_v_smooth = enforce_kinematic_limits(sample_times, ego_s_prof, ego_d_prof, ego_spec)
    else:
        ego_s_smooth, ego_d_smooth, ego_v_smooth = enforce_kinematic_limits(sample_times, ego_s_interp, ego_d_interp, ego_spec)

    # 2. Actor trajectories (data-driven RTS smoothing + kinematic limits)
    actors_resampled: List[Dict[str, Any]] = []
    if collision_ev is not None:
        t_imp = float(collision_ev["t_impact"])
        s_impact = float(np.interp(t_imp, sample_times, ego_s_smooth)) + 2.5
        cid_col = int(collision_ev.get("class_id", 16))
        col_spec = cfg.get_class_spec(cid_col)
        actors_resampled.append({
            "role_name": "crossing_obstacle",
            "entity_kind": "pedestrian",
            "is_collision_target": True,
            "track_id": 1,
            "class_id": cid_col,
            "category": col_spec.category,
            "blueprint": col_spec.blueprints[0],
            "color": "200,150,100",
            "t_impact": t_imp,
            "s_impact": s_impact,
            "times": sample_times.tolist(),
            "s": [s_impact] * len(sample_times),
            "d": [0.0] * len(sample_times),
            "v": [0.0] * len(sample_times),
        })

        if extracted_data.get("has_oncoming_traffic"):
            # Smooth oncoming vehicle that continues past ego without sudden stop or 180-deg flip
            v_onc = 15.0
            s_onc_start = s_impact + 115.0
            s_onc_raw = np.array([s_onc_start - v_onc * float(t) for t in sample_times])
            d_onc_raw = np.full_like(sample_times, -cfg.road.lane_width_m)
            onc_spec = cfg.get_class_spec(2)
            s_onc, d_onc, v_onc_arr = enforce_kinematic_limits(
                sample_times, s_onc_raw, d_onc_raw, onc_spec, allow_reverse=True
            )
            actors_resampled.append({
                "role_name": "oncoming_vehicle",
                "entity_kind": "vehicle",
                "is_oncoming": True,
                "track_id": 2,
                "class_id": 2,
                "category": "car",
                "blueprint": onc_spec.blueprints[0],
                "color": "220,220,255",
                "times": sample_times.tolist(),
                "s": s_onc.tolist(),
                "d": d_onc.tolist(),
                "v": v_onc_arr.tolist(),
            })

    for idx, (tid, obs) in enumerate(extracted_data.get("tracks", [])):
        t_arr = np.array([o["time"] for o in obs], dtype=float)
        x_arr = np.array([o["x_cam"] for o in obs], dtype=float)
        z_arr = np.array([o["z_cam"] for o in obs], dtype=float)
        sig_z = np.array([o.get("sigma_z", kin.rts_meas_noise_z) for o in obs], dtype=float)

        # Majority vote for class_id across the track for robustness against single-frame misclassifications
        class_ids = [int(o["class_id"]) for o in obs]
        cid = max(set(class_ids), key=class_ids.count)
        spec = cfg.get_class_spec(cid)

        # Reject single-frame occlusion spikes via rolling MAD
        x_clean = reject_outliers_mad(x_arr, kin.outlier_mad_window, kin.outlier_mad_thresh)
        z_clean = reject_outliers_mad(z_arr, kin.outlier_mad_window, kin.outlier_mad_thresh)

        # Convert to world Frenet before RTS smoothing so ego acceleration doesn't distort relative dynamics
        ego_s_at_obs = np.interp(t_arr, sample_times, ego_s_smooth)
        ego_d_at_obs = np.interp(t_arr, sample_times, ego_d_smooth)
        s_world_obs = ego_s_at_obs + z_clean
        d_world_obs = ego_d_at_obs + x_clean

        s_rts, _, _ = rts_smoother_1d(t_arr, s_world_obs, np.maximum(sig_z, kin.rts_meas_noise_z), kin.rts_process_noise_pos)
        d_rts, _, _ = rts_smoother_1d(t_arr, d_world_obs, kin.rts_meas_noise_x, kin.rts_process_noise_pos * 0.5)

        # Visibility-aware bounded constant-velocity extrapolation onto uniform grid
        s_grid = extrapolate_trajectory(t_arr, s_rts, sample_times, max_vel=spec.max_speed, min_vel=0.0)
        d_grid = extrapolate_trajectory(t_arr, d_rts, sample_times, max_vel=0.6, min_vel=-0.6)
        # Keep lateral coordinates within realistic road corridor bounds
        max_lat = cfg.road.lane_width_m * max(cfg.road.default_lanes_left, cfg.road.default_lanes_right) + 1.5
        d_grid = np.clip(d_grid, -max_lat, max_lat)

        s_feas, d_feas, v_feas = enforce_kinematic_limits(sample_times, s_grid, d_grid, spec, allow_reverse=False)

        bp_name = spec.blueprints[idx % len(spec.blueprints)]
        color = CARLA_COLOR_PALETTE[idx % len(CARLA_COLOR_PALETTE)]

        actors_resampled.append({
            "role_name": f"npc_track_{tid}",
            "entity_kind": spec.entity_kind,
            "track_id": int(tid),
            "class_id": cid,
            "category": spec.category,
            "blueprint": bp_name,
            "color": color,
            "t_visible_start": float(t_arr[0]),
            "t_visible_end": float(t_arr[-1]),
            "times": sample_times.tolist(),
            "s": s_feas.tolist(),
            "d": d_feas.tolist(),
            "v": v_feas.tolist(),
        })

    ego_dict = {
        "role_name": "hero",
        "entity_kind": "vehicle",
        "class_id": 2,
        "category": "car",
        "blueprint": "vehicle.lincoln.mkz",
        "color": "0,100,255",
        "times": sample_times.tolist(),
        "s": ego_s_smooth.tolist(),
        "d": ego_d_smooth.tolist(),
        "v": ego_v_smooth.tolist(),
        "yaw_rate": ego_yaw_rate.tolist(),
    }

    # 3. Deconflict unintended overlaps across all timestamps
    actors_deconflicted = deconflict_trajectories(ego_dict, actors_resampled, cfg)

    return {
        "duration": duration,
        "sample_dt": dt,
        "is_night": bool(extracted_data.get("is_night", False)),
        "collision_event": collision_ev,
        "times": sample_times.tolist(),
        "ego": ego_dict,
        "actors": actors_deconflicted,
    }
