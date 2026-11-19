"""Camera intrinsics, vanishing-point horizon estimation, and monocular 3D localization."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Tuple

import cv2
import numpy as np

from .config import CameraConfig, ClassSpec


@dataclass(frozen=True)
class CameraIntrinsics:
    """Pinhole camera model in SI units (meters) and pixel coordinates."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    hfov_deg: float
    cam_height_m: float
    horizon_estimated: bool = False


def estimate_horizon_y(
    sample_frames_gray: Iterable[np.ndarray],
    width: int,
    height: int,
    cfg: CameraConfig,
) -> Tuple[float, bool]:
    """
    Estimate image horizon row c_y from road-edge vanishing points using Canny + Hough lines.
    Returns (cy_px, estimated_ok). Falls back smoothly to cfg.default_horizon_ratio * height.
    """
    prior_cy = height * cfg.default_horizon_ratio
    if not cfg.estimate_horizon:
        return prior_cy, False

    y_min = height * cfg.horizon_min_ratio
    y_max = height * cfg.horizon_max_ratio
    vp_ys: list[float] = []

    for gray in sample_frames_gray:
        if gray is None or gray.size == 0:
            continue
        roi_top = int(height * 0.35)
        roi_bot = int(height * 0.88)
        roi = gray[roi_top:roi_bot, :]
        blurred = cv2.GaussianBlur(roi, (5, 5), 1.0)
        edges = cv2.Canny(blurred, 50, 150)
        lines = cv2.HoughLinesP(
            edges,
            rho=1,
            theta=np.pi / 180.0,
            threshold=40,
            minLineLength=max(25, int(width * 0.04)),
            maxLineGap=20,
        )
        if lines is None or len(lines) < 2:
            continue

        left_lines = []
        right_lines = []
        for seg in np.asarray(lines).reshape(-1, 4):
            x1, y1, x2, y2 = float(seg[0]), float(seg[1] + roi_top), float(seg[2]), float(seg[3] + roi_top)
            dx = x2 - x1
            dy = y2 - y1
            if abs(dx) < 5.0:
                continue
            slope = dy / dx
            # Lane markings converge: left-side lines have negative slope, right-side positive
            if -1.8 < slope < -0.18:
                left_lines.append((x1, y1, x2, y2))
            elif 0.18 < slope < 1.8:
                right_lines.append((x1, y1, x2, y2))

        for (x1, y1, x2, y2) in left_lines[:15]:
            a1 = y2 - y1
            b1 = x1 - x2
            c1 = a1 * x1 + b1 * y1
            for (x3, y3, x4, y4) in right_lines[:15]:
                a2 = y4 - y3
                b2 = x3 - x4
                c2 = a2 * x3 + b2 * y3
                det = a1 * b2 - a2 * b1
                if abs(det) < 1e-3:
                    continue
                vx = (b2 * c1 - b1 * c2) / det
                vy = (a1 * c2 - a2 * c1) / det
                if 0.15 * width <= vx <= 0.85 * width and y_min <= vy <= y_max:
                    vp_ys.append(float(vy))

    if len(vp_ys) < 6:
        return prior_cy, False

    vp_med = float(np.median(vp_ys))
    # Blend estimated vanishing point with prior to guard against noisy night frames
    confidence = min(0.75, len(vp_ys) / 40.0)
    cy = confidence * vp_med + (1.0 - confidence) * prior_cy
    cy = float(np.clip(cy, y_min, y_max))
    return cy, True


def build_camera_intrinsics(
    width: int,
    height: int,
    cfg: CameraConfig,
    sample_frames_gray: Iterable[np.ndarray] = (),
) -> CameraIntrinsics:
    """
    Construct pinhole camera intrinsics adapted to both landscape (16:9, 4:3)
    and portrait (9:16) dashcam videos, with vanishing-point horizon calibration.
    """
    # For portrait videos (e.g. cropped vertical shorts), HFOV is narrower than landscape HFOV
    if width < height:
        # Derive focal length from vertical FOV (~45 deg typical for vertical crop)
        vfov_deg = 50.0
        fy = (height / 2.0) / math.tan(math.radians(vfov_deg / 2.0))
        fx = fy
        hfov_eff = math.degrees(2.0 * math.atan((width / 2.0) / fx))
    else:
        hfov_eff = cfg.hfov_deg
        fx = (width / 2.0) / math.tan(math.radians(hfov_eff / 2.0))
        fy = fx

    cx = width / 2.0
    cy, est_ok = estimate_horizon_y(sample_frames_gray, width, height, cfg)
    return CameraIntrinsics(
        width=width,
        height=height,
        fx=fx,
        fy=fy,
        cx=cx,
        cy=cy,
        hfov_deg=hfov_eff,
        cam_height_m=cfg.cam_height_m,
        horizon_estimated=est_ok,
    )


def estimate_monocular_3d(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    spec: ClassSpec,
    intr: CameraIntrinsics,
    cfg: CameraConfig,
) -> Tuple[float, float, float]:
    """
    Estimate camera-frame metric position (x_cam, z_cam) in meters and depth uncertainty sigma_z
    by uncertainty-weighted fusion of ground-plane homography and class-width pinhole projection.

    Camera frame convention:
      +z_cam: forward along optical axis (m)
      +x_cam: right of optical axis (m)
    """
    bw = max(float(x2 - x1), 12.0)
    uc = 0.5 * (float(x1) + float(x2))
    vb = float(y2)

    # Account for foreshortening / oblique viewing angle of vehicle width+length
    lateral_norm = abs(uc - intr.cx) / max(intr.width * 0.5, 1.0)
    angle_factor = 1.0 + 0.25 * lateral_norm
    z_width = (intr.fx * spec.width * angle_factor) / bw
    z_width = float(np.clip(z_width, cfg.min_depth_m, cfg.max_depth_m))
    # Relative variance of width-based depth (grows quadratically with distance)
    sigma_w = max(1.0, 0.18 * z_width)

    dv = vb - intr.cy
    if dv > 6.0:
        z_gp = (intr.fy * intr.cam_height_m) / dv
        z_gp = float(np.clip(z_gp, cfg.min_depth_m, cfg.max_depth_m))
        # Ground-plane depth sensitivity dz/dv = z^2 / (fy * h_cam); assume ~3 px bottom-edge noise
        sigma_gp = max(0.8, 3.0 * (z_gp ** 2) / max(intr.fy * intr.cam_height_m, 1.0))
        # Inverse-variance weighting (minimum-variance fusion)
        w_gp = 1.0 / (sigma_gp ** 2)
        w_w = 1.0 / (sigma_w ** 2)
        z_cam = (w_gp * z_gp + w_w * z_width) / (w_gp + w_w)
        sigma_z = math.sqrt(1.0 / (w_gp + w_w))
    else:
        z_cam = z_width
        sigma_z = sigma_w

    z_cam = float(np.clip(z_cam, cfg.min_depth_m, cfg.max_depth_m))
    x_cam = float(((uc - intr.cx) * z_cam) / intr.fx)
    return x_cam, z_cam, float(sigma_z)
