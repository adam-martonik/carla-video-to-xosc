"""Unit tests for camera calibration, horizon estimation, and monocular 3D depth fusion."""

import math
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from xosc_pipeline.calibration import build_camera_intrinsics, estimate_horizon_y, estimate_monocular_3d
from xosc_pipeline.config import load_config


def test_landscape_vs_portrait_intrinsics():
    cfg = load_config()
    intr_land = build_camera_intrinsics(1280, 720, cfg.camera)
    intr_port = build_camera_intrinsics(360, 640, cfg.camera)

    assert math.isclose(intr_land.hfov_deg, 70.0, rel_tol=1e-3)
    # Portrait video must have narrower effective HFOV than landscape
    assert intr_port.hfov_deg < 45.0
    assert intr_port.fx > 0 and intr_port.fy > 0


def test_vanishing_point_horizon_estimation():
    cfg = load_config()
    w, h = 1280, 720
    true_vp_y = 330  # ~0.458 * h
    frame = np.zeros((h, w), dtype=np.uint8)
    # Draw converging lane markings toward (640, 330)
    for x_bot in [120, 280, 460, 820, 1000, 1160]:
        cv2.line(frame, (x_bot, 680), (640, true_vp_y), 255, 3)

    cy, ok = estimate_horizon_y([frame, frame, frame], w, h, cfg.camera)
    assert ok is True
    # Blended estimate should move significantly from prior (374.4) toward true_vp_y (330)
    assert cy < 365.0
    assert abs(cy - true_vp_y) < 35.0


def test_monocular_3d_depth_fusion_scales_monotonically():
    cfg = load_config()
    intr = build_camera_intrinsics(1280, 720, cfg.camera)
    car_spec = cfg.get_class_spec(2)

    # Closer car: wider box, lower bottom edge
    x_near, z_near, sig_near = estimate_monocular_3d(540, 420, 740, 560, car_spec, intr, cfg.camera)
    # Farther car: narrower box, closer to horizon
    x_far, z_far, sig_far = estimate_monocular_3d(610, 380, 670, 420, car_spec, intr, cfg.camera)

    assert z_near < z_far
    assert abs(x_near) < 1.0
    assert sig_near < sig_far
    assert cfg.camera.min_depth_m <= z_near <= cfg.camera.max_depth_m
