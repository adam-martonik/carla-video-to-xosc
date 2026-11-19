"""Unit tests for outlier rejection, RTS Kalman smoother, kinematic limits, stitching, and deconfliction."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xosc_pipeline.config import load_config
from xosc_pipeline.kinematics import (
    deconflict_trajectories,
    enforce_kinematic_limits,
    extrapolate_trajectory,
    reject_outliers_mad,
    rts_smoother_1d,
)
from xosc_pipeline.metrics import compute_kinematic_metrics
from xosc_pipeline.tracking import stitch_fragmented_tracks


def test_mad_outlier_rejection_removes_spikes():
    vals = np.linspace(10.0, 20.0, 30)
    noisy = vals.copy()
    noisy[10] = 65.0  # sudden occlusion spike
    noisy[20] = -12.0

    cleaned = reject_outliers_mad(noisy, window=9, thresh=3.0)
    assert abs(cleaned[10] - vals[10]) < 1.0
    assert abs(cleaned[20] - vals[20]) < 1.0
    # Input array must not be mutated
    assert noisy[10] == 65.0


def test_rts_smoother_reduces_noise_and_recovers_velocity():
    rng = np.random.default_rng(42)
    t = np.linspace(0.0, 10.0, 101)
    true_pos = 15.0 * t + 0.5 * 1.2 * t ** 2
    true_vel = 15.0 + 1.2 * t
    obs = true_pos + rng.normal(0.0, 1.5, size=len(t))

    pos_s, vel_s, _ = rts_smoother_1d(t, obs, meas_std=1.5, process_std=0.5)
    raw_rmse = float(np.sqrt(np.mean((obs - true_pos) ** 2)))
    smooth_rmse = float(np.sqrt(np.mean((pos_s - true_pos) ** 2)))

    assert smooth_rmse < raw_rmse * 0.5
    assert float(np.sqrt(np.mean((vel_s - true_vel) ** 2))) < 1.0


def test_enforce_kinematic_limits_bounds_accel_and_jerk():
    cfg = load_config()
    car_spec = cfg.get_class_spec(2)
    t = np.arange(0.0, 10.01, 0.25)
    # Create an unphysical trajectory with a massive step jump at t=5s
    s_raw = 15.0 * t + np.where(t >= 5.0, 12.0, 0.0)
    d_raw = np.where(t >= 5.0, 3.5, 0.0)

    s_out, d_out, v_out = enforce_kinematic_limits(t, s_raw, d_raw, car_spec)
    hs = np.arctan2(np.gradient(d_out, t), np.maximum(np.gradient(s_out, t), 1e-3))
    m = compute_kinematic_metrics(t, s_out, d_out, hs)

    assert m["max_abs_acc"] <= car_spec.max_decel + 0.5
    assert m["max_abs_jerk"] <= car_spec.max_jerk + 1.5
    assert np.all(v_out >= 0.0)
    assert np.all(v_out <= car_spec.max_speed + 0.5)


def test_extrapolate_trajectory_does_not_freeze_outside_bounds():
    t_obs = np.linspace(4.0, 8.0, 21)
    s_obs = 50.0 + 10.0 * (t_obs - 4.0)
    t_grid = np.linspace(0.0, 12.0, 49)

    s_ext = extrapolate_trajectory(t_obs, s_obs, t_grid, max_vel=30.0, min_vel=0.0)
    # At t=0 should be ~10.0m, not frozen at 50.0m
    assert abs(s_ext[0] - 10.0) < 0.5
    assert abs(s_ext[-1] - 130.0) < 0.5


def test_tracklet_stitching_merges_occluded_same_class_track():
    raw = {
        1: [
            {"frame": 0, "time": 0.0, "class_id": 2, "x_cam": 0.0, "z_cam": 20.0},
            {"frame": 30, "time": 1.0, "class_id": 2, "x_cam": 0.1, "z_cam": 22.0},
        ],
        2: [
            {"frame": 45, "time": 1.5, "class_id": 2, "x_cam": 0.15, "z_cam": 23.0},
            {"frame": 75, "time": 2.5, "class_id": 2, "x_cam": 0.2, "z_cam": 25.0},
        ],
    }
    stitched = stitch_fragmented_tracks(raw, max_gap_s=1.5, max_dist_m=5.0)
    assert len(stitched) == 1
    assert len(next(iter(stitched.values()))) == 4


def test_deconflict_trajectories_eliminates_overlap():
    cfg = load_config()
    t = np.linspace(0.0, 5.0, 21)
    ego = {"role_name": "hero", "class_id": 2, "s": (10.0 * t).tolist(), "d": [0.0] * len(t)}
    # Actor starts directly overlapping ego
    actor = {
        "role_name": "npc_1",
        "entity_kind": "vehicle",
        "class_id": 2,
        "s": (10.0 * t + 1.0).tolist(),
        "d": [0.2] * len(t),
    }
    resolved = deconflict_trajectories(ego, [actor], cfg)
    s_ego, d_ego = np.array(ego["s"]), np.array(ego["d"])
    s_act, d_act = np.array(resolved[0]["s"]), np.array(resolved[0]["d"])
    spec = cfg.get_class_spec(2)
    req_s = spec.length + cfg.kinematics.min_safety_margin_long_m - 0.1
    req_d = spec.width + cfg.kinematics.min_safety_margin_lat_m - 0.1
    overlaps = (np.abs(s_act - s_ego) < req_s) & (np.abs(d_act - d_ego) < req_d)
    assert not np.any(overlaps)
