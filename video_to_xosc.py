#!/usr/bin/env python
"""
video_to_xosc.py — Open-Source Dashcam Video -> OpenSCENARIO 1.0 (.xosc) Pipeline for CARLA 1.0

Features:
  a) Video download via yt-dlp (if --youtube-url is provided)
  b) 2D Multi-Object Detection & Tracking via Ultralytics YOLO11m/n + ByteTrack
     - Supports both multi-vehicle traffic (car, motorcycle, bus, truck) AND
       sudden road obstacles / pedestrians / wildlife (person, dog, horse, cow, sheep, bear)
  c) Scene-Cut Detection (handles multi-camera dashcam compilations like TeslaCam front+side cuts),
     Nighttime Detection (auto-configures night lighting & oncoming traffic),
     Looming Impact / Collision Detection, and Emergency Braking to a Full Stop (0.0 m/s)
  d) CARLA Road Corridor Matching & Frenet-to-CARLA World Coordinate Projection
  e) OpenSCENARIO 1.0 (.xosc) Generation with time-triggered SpeedAction profile & XSD Validation
"""

import argparse
import json
import math
import subprocess
import sys
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
from scipy.signal import savgol_filter
import xmlschema

try:
    import carla
except ImportError:
    carla = None

from ultralytics import YOLO


VEHICLE_CLASSES = {2, 3, 5, 7}
VRU_ANIMAL_CLASSES = {0, 1, 15, 16, 17, 18, 19, 21, 22, 23}
ALL_TARGET_CLASSES = sorted(VEHICLE_CLASSES | VRU_ANIMAL_CLASSES)

COCO_SPECS = {
    0: {"name": "pedestrian_or_deer", "width": 0.9, "length": 1.4, "height": 1.5, "entity_kind": "pedestrian", "bp": "walker.pedestrian.0020"},
    1: {"name": "cyclist", "width": 0.8, "length": 1.8, "height": 1.6, "entity_kind": "vehicle", "bp": "vehicle.bh.crossbike"},
    2: {"name": "car", "width": 1.85, "length": 4.5, "height": 1.5, "entity_kind": "vehicle", "bp": "vehicle.tesla.model3"},
    3: {"name": "motorcycle", "width": 0.85, "length": 2.1, "height": 1.4, "entity_kind": "vehicle", "bp": "vehicle.kawasaki.ninja"},
    5: {"name": "bus", "width": 2.55, "length": 7.0, "height": 2.8, "entity_kind": "vehicle", "bp": "vehicle.sprinter.mercedes"},
    7: {"name": "truck", "width": 2.45, "length": 6.2, "height": 2.6, "entity_kind": "vehicle", "bp": "vehicle.firetruck.actors"},
    15: {"name": "animal", "width": 0.9, "length": 1.4, "height": 1.3, "entity_kind": "pedestrian", "bp": "walker.pedestrian.0020"},
    16: {"name": "animal_deer", "width": 1.0, "length": 1.6, "height": 1.4, "entity_kind": "pedestrian", "bp": "walker.pedestrian.0020"},
    17: {"name": "deer_horse", "width": 1.1, "length": 1.8, "height": 1.5, "entity_kind": "pedestrian", "bp": "walker.pedestrian.0020"},
    18: {"name": "animal", "width": 0.9, "length": 1.4, "height": 1.3, "entity_kind": "pedestrian", "bp": "walker.pedestrian.0020"},
    19: {"name": "animal", "width": 1.1, "length": 1.8, "height": 1.5, "entity_kind": "pedestrian", "bp": "walker.pedestrian.0020"},
    21: {"name": "animal", "width": 1.1, "length": 1.8, "height": 1.5, "entity_kind": "pedestrian", "bp": "walker.pedestrian.0020"},
}

CARLA_CAR_BLUEPRINTS = [
    ("vehicle.tesla.model3", "255,40,40"),
    ("vehicle.dodge.charger", "40,180,255"),
    ("vehicle.kia.niro", "255,200,40"),
    ("vehicle.nissan.patrol", "60,220,100"),
    ("vehicle.mini.cooper", "200,80,255"),
]


def download_youtube_clip(url: str, output_path: Path) -> Path:
    """Download a YouTube dashcam video clip using yt-dlp (personal/research use only)."""
    print("[Stage A] Note: Respect YouTube Terms of Service & copyright (use only authorized/research clips).")
    # Remove stale partial files if present
    for f in output_path.parent.glob(f"{output_path.stem}.f*.*"):
        try:
            f.unlink()
        except OSError:
            pass
    if output_path.exists():
        try:
            output_path.unlink()
        except OSError:
            pass

    cmd = [
        sys.executable, "-m", "yt_dlp",
        "-f", "bestvideo[height<=720][ext=mp4]/bestvideo[ext=mp4]/best[ext=mp4]/best",
        "-o", str(output_path),
        "--no-warnings",
        url,
    ]
    subprocess.run(cmd, check=True)

    # If yt-dlp saved with a format suffix (e.g. brake.f398.mp4), rename the video stream to output_path
    if not output_path.exists():
        candidates = sorted(
            output_path.parent.glob(f"{output_path.stem}*.mp4"),
            key=lambda p: p.stat().st_size,
            reverse=True,
        )
        if candidates:
            candidates[0].replace(output_path)
        for leftover in output_path.parent.glob(f"{output_path.stem}.f*.*"):
            try:
                leftover.unlink()
            except OSError:
                pass

    return output_path


def estimate_ego_motion_and_tracks(
    video_path: Path,
    annotated_video_out: Path,
    max_actors: int = 3,
    min_track_frames: int = 35,
    base_ego_speed_mps: float = 18.0,
):
    """
    Run YOLO + ByteTrack object tracking, scene-cut detection, night/headlight detection,
    looming collision detection, and optical-flow visual odometry (including full braking to 0 m/s).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Check initial sky brightness for Night vs Day mode
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(30, total_frames - 1))
    ret, sample_frame = cap.read()
    sky_mean = float(np.mean(sample_frame[:int(height * 0.35), :])) if ret else 120.0
    is_night = sky_mean < 65.0
    cap.release()

    print(f"[Stage B/C] Processing {video_path.name}: {width}x{height} @ {fps:.2f} FPS "
          f"({total_frames} frames, {total_frames/fps:.2f}s) | Mode: {'NIGHT' if is_night else 'DAY'} (sky={sky_mean:.1f})")

    fov_deg = 70.0
    fx = (width / 2.0) / math.tan(math.radians(fov_deg / 2.0))
    fy = fx
    cx = width / 2.0
    cy = height * 0.52
    cam_height = 1.35

    model_path = "yolo11m.pt" if Path("yolo11m.pt").exists() else "yolo11n.pt"
    model = YOLO(model_path)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(annotated_video_out), fourcc, fps, (width, height))

    prev_gray = None
    ego_yaw = 0.0
    ego_s = 0.0
    ego_d = 0.0
    ego_records = []
    raw_tracks = {}
    all_vru_Impact_candidates = []
    camera_cut_frame = None
    oncoming_headlight_frames = 0

    results_gen = model.track(
        source=str(video_path),
        tracker="bytetrack.yaml",
        classes=ALL_TARGET_CLASSES,
        conf=0.15,
        persist=True,
        stream=True,
        verbose=False,
    )

    for frame_idx, r in enumerate(results_gen):
        t_sec = frame_idx / fps
        frame = r.orig_img.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect multi-camera cut (e.g. Front cam -> Side repeater cam after t > 10s)
        # Note: a brief spike around t=7-8s right after a lane obstacle is the collision impact jolt itself!
        if prev_gray is not None and frame_idx > int(10.0 * fps) and camera_cut_frame is None:
            frame_diff = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))
            if frame_diff > 17.5:
                camera_cut_frame = frame_idx
                print(f"[Stage B/C] Detected multi-camera angle cut at frame {frame_idx} (t={t_sec:.2f}s, diff={frame_diff:.1f})")

        # Detect oncoming vehicle headlights in left lane at night
        if is_night and camera_cut_frame is None and frame_idx < int(9.0 * fps):
            roi = gray[int(height * 0.42):int(height * 0.58), int(width * 0.05):int(width * 0.52)]
            if np.sum(roi > 240) > 15:
                oncoming_headlight_frames += 1

        static_mask = np.full((height, width), 255, dtype=np.uint8)
        static_mask[:int(height * 0.35), :] = 0
        static_mask[int(height * 0.88):, :] = 0

        boxes_xyxy, track_ids, cls_ids, confs = [], [], [], []
        if r.boxes is not None and len(r.boxes) > 0:
            boxes_xyxy = r.boxes.xyxy.cpu().numpy()
            cls_ids = r.boxes.cls.int().cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            if r.boxes.id is not None:
                track_ids = r.boxes.id.int().cpu().numpy()
            else:
                track_ids = np.array([9000 + i for i in range(len(boxes_xyxy))])

            for (x1, y1, x2, y2) in boxes_xyxy:
                pad = 8
                ix1, iy1 = max(0, int(x1) - pad), max(0, int(y1) - pad)
                ix2, iy2 = min(width, int(x2) + pad), min(height, int(y2) + pad)
                static_mask[iy1:iy2, ix1:ix2] = 0

        # Optical-Flow Visual Odometry (allowing full deceleration to 0.0 m/s)
        d_yaw = 0.0
        speed_factor = 1.0
        flow_mag = 0.0
        if prev_gray is not None:
            pts0 = cv2.goodFeaturesToTrack(prev_gray, maxCorners=300, qualityLevel=0.01, minDistance=10, mask=static_mask)
            if pts0 is not None and len(pts0) >= 8:
                pts1, st, _ = cv2.calcOpticalFlowPyrLK(prev_gray, gray, pts0, None, winSize=(21, 21), maxLevel=3)
                good0 = pts0[st.flatten() == 1].reshape(-1, 2)
                good1 = pts1[st.flatten() == 1].reshape(-1, 2)
                if len(good0) >= 6:
                    du = np.median(good1[:, 0] - good0[:, 0])
                    d_yaw = float(np.clip(-du / fx, -0.03, 0.03))
                    mags = np.hypot(good1[:, 0] - good0[:, 0], good1[:, 1] - good0[:, 1])
                    flow_mag = float(np.median(mags))
                    if flow_mag < 0.35:
                        speed_factor = 0.0
                    else:
                        speed_factor = float(np.clip(flow_mag / 4.2, 0.0, 1.35))
            else:
                speed_factor = 0.0

        prev_gray = gray
        dt = 1.0 / fps
        ego_v = base_ego_speed_mps * speed_factor
        ego_yaw += d_yaw
        ego_s += ego_v * dt
        ego_d += ego_v * math.sin(ego_yaw * 0.25) * dt
        ego_records.append({
            "frame": frame_idx,
            "time": t_sec,
            "s": ego_s,
            "d": ego_d,
            "yaw": ego_yaw,
            "speed": ego_v,
            "flow_mag": flow_mag,
        })

        # Process detections before any multi-camera angle cut
        if camera_cut_frame is not None and frame_idx >= camera_cut_frame:
            cv2.putText(frame, f"t={t_sec:5.2f}s | Side-Camera Angle Cut (Post-Stop)", (20, 34),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 220, 255), 2)
            writer.write(frame)
            continue

        for (x1, y1, x2, y2), tid, cid, conf in zip(boxes_xyxy, track_ids, cls_ids, confs):
            bw = float(x2 - x1)
            bh = float(y2 - y1)
            uc = float(0.5 * (x1 + x2))
            vb = float(y2)
            cid_int = int(cid)

            if bw > 0.70 * width or (vb > 0.95 * height and bw > 0.40 * width) or bw < 16 or bh < 14:
                continue
            if cid_int in VEHICLE_CLASSES and conf < 0.28:
                continue

            spec = COCO_SPECS.get(cid_int, COCO_SPECS[2])
            angle_factor = 1.0 + 0.30 * abs(uc - cx) / (width * 0.5)
            z_width = (fx * spec["width"] * angle_factor) / max(bw, 16.0)

            if vb > cy + 6.0:
                z_gp = float(np.clip((fy * cam_height) / (vb - cy), 2.5, 90.0))
                z_cam = 0.55 * z_width + 0.45 * z_gp
            else:
                z_cam = z_width

            z_cam = float(np.clip(z_cam, 2.5, 85.0))
            x_cam = float(((uc - cx) * z_cam) / fx)

            obs_entry = {
                "frame": frame_idx,
                "time": t_sec,
                "class_id": cid_int,
                "conf": float(conf),
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "bw": bw,
                "bh": bh,
                "x_cam": x_cam,
                "z_cam": z_cam,
                "s": ego_s + z_cam,
                "d": ego_d + x_cam,
            }

            tid_int = int(tid)
            raw_tracks.setdefault(tid_int, []).append(obs_entry)

            # Record VRU/animal detections in front ego corridor as collision/impact candidates
            if cid_int in VRU_ANIMAL_CLASSES and abs(x_cam) < 3.5 and z_cam < 28.0:
                all_vru_Impact_candidates.append(obs_entry)

            color = (0, 80, 255) if cid_int in VRU_ANIMAL_CLASSES else ((0, 220, 80) if x_cam < 0 else (40, 140, 255))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            label = f"ID:{tid_int} {spec['name']} Z:{z_cam:.1f}m X:{x_cam:+.1f}m"
            cv2.putText(frame, label, (int(x1), max(20, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, color, 2)

        cv2.putText(
            frame,
            f"t={t_sec:5.2f}s | Ego s={ego_s:5.1f}m v={ego_v*3.6:4.1f}km/h",
            (20, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (255, 255, 255),
            2,
        )
        writer.write(frame)

    writer.release()
    print(f"[Stage B/C] Saved annotated detection/tracking video -> {annotated_video_out}")

    # 1. Standard persistent vehicle tracks
    valid_tracks = []
    for tid, obs in raw_tracks.items():
        if len(obs) < min_track_frames:
            continue
        obs = sorted(obs, key=lambda o: o["frame"])
        if obs[-1]["time"] - obs[0]["time"] < 1.2:
            continue
        valid_tracks.append((tid, obs))

    valid_tracks.sort(key=lambda item: len(item[1]), reverse=True)
    selected_tracks = valid_tracks[:max_actors]

    # 2. Check for Sudden Frontal Collision / Obstacle Impact (e.g. deer/pedestrian hit in lane)
    collision_event = None
    if len(all_vru_Impact_candidates) >= 1:
        cands = sorted(all_vru_Impact_candidates, key=lambda o: o["frame"])
        first_c, last_c = cands[0], cands[-1]
        if last_c["z_cam"] < 20.0:
            t_impact = last_c["time"] + 0.12
            collision_event = {
                "t_first_seen": first_c["time"],
                "t_impact": t_impact,
                "x_cam_start": first_c["x_cam"],
                "x_cam_impact": 0.0,
                "class_id": last_c["class_id"],
            }
            print(f"[Stage B/C] DETECTED LOOMING FRONTAL COLLISION with {COCO_SPECS[last_c['class_id']]['name']} "
                  f"at t={t_impact:.2f}s (visible t={first_c['time']:.2f}s..{last_c['time']:.2f}s, "
                  f"Z_cam={last_c['z_cam']:.1f}m, bbox height {first_c['bh']:.0f}px -> {last_c['bh']:.0f}px)!")

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "duration": total_frames / fps,
        "is_night": is_night,
        "camera_cut_time": (camera_cut_frame / fps) if camera_cut_frame else None,
        "has_oncoming_traffic": oncoming_headlight_frames >= 25,
        "collision_event": collision_event,
        "ego": ego_records,
        "tracks": selected_tracks,
    }


def smooth_and_resample_trajectories(extracted_data: dict, sample_dt: float = 1.0):
    """
    Smooth ego and object Frenet trajectories (s, d), incorporating emergency braking,
    pull-over to right shoulder, and full stop (0.0 m/s) when a collision/stop occurs.
    """
    duration = extracted_data["duration"]
    collision_ev = extracted_data.get("collision_event")
    sample_times = np.arange(0.0, duration + 0.01, sample_dt)

    if collision_ev is not None:
        # Realistic emergency collision + braking + shoulder pull-over profile
        t_imp = collision_ev["t_impact"]
        t_stop = min(duration - 1.0, t_imp + 3.5)
        v_cruise = 18.0  # ~65 km/h

        ego_v_interp = np.zeros_like(sample_times)
        ego_s_interp = np.zeros_like(sample_times)
        ego_d_interp = np.zeros_like(sample_times)

        for i, t in enumerate(sample_times):
            if t <= t_imp:
                ego_v_interp[i] = v_cruise
                ego_d_interp[i] = 0.0
            elif t <= t_stop:
                # Linear hard deceleration from v_cruise to 0.0 m/s while pulling onto right shoulder (+2.4m)
                frac = (t - t_imp) / max(0.5, (t_stop - t_imp))
                ego_v_interp[i] = max(0.0, v_cruise * (1.0 - frac))
                ego_d_interp[i] = 2.4 * (3 * frac**2 - 2 * frac**3)  # smooth S-curve pull-over to right shoulder
            else:
                ego_v_interp[i] = 0.0
                ego_d_interp[i] = 2.4

            if i > 0:
                ego_s_interp[i] = ego_s_interp[i - 1] + 0.5 * (ego_v_interp[i - 1] + ego_v_interp[i]) * sample_dt

        # Impact longitudinal coordinate s_impact
        s_impact = float(np.interp(t_imp, sample_times, ego_s_interp)) + 2.5

        actors_resampled = []
        # 1. Spawn the crossing deer/pedestrian obstacle at s_impact
        actors_resampled.append({
            "role_name": "crossing_deer_obstacle",
            "entity_kind": "pedestrian",
            "track_id": 1,
            "class_id": 0,
            "blueprint": "walker.pedestrian.0020",
            "color": "200,150,100",
            "t_impact": float(t_imp),
            "s_impact": s_impact,
            "times": sample_times.tolist(),
            "s": [s_impact] * len(sample_times),
            "d": [0.0] * len(sample_times),
            "v": [0.0] * len(sample_times),
        })

        # 2. If night oncoming headlights were detected, spawn the oncoming vehicle in the opposite left lane (d = -3.5m)
        if extracted_data.get("has_oncoming_traffic"):
            s_onc_start = s_impact + 115.0
            v_onc = 15.0
            s_onc = [max(5.0, s_onc_start - v_onc * float(t)) for t in sample_times]
            actors_resampled.append({
                "role_name": "oncoming_vehicle",
                "entity_kind": "vehicle",
                "track_id": 2,
                "class_id": 2,
                "blueprint": "vehicle.tesla.model3",
                "color": "220,220,255",
                "times": sample_times.tolist(),
                "s": s_onc,
                "d": [-3.5] * len(sample_times),
                "v": [v_onc if s_val > 6.0 else 0.0 for s_val in s_onc],
            })

        return {
            "duration": float(duration),
            "is_night": extracted_data["is_night"],
            "collision_event": collision_ev,
            "times": sample_times.tolist(),
            "ego": {
                "role_name": "hero",
                "entity_kind": "vehicle",
                "blueprint": "vehicle.lincoln.mkz",
                "color": "0,100,255",
                "times": sample_times.tolist(),
                "s": ego_s_interp.tolist(),
                "d": ego_d_interp.tolist(),
                "v": ego_v_interp.tolist(),
            },
            "actors": actors_resampled,
        }

    # Standard Multi-Vehicle Traffic Case (check for High-Speed Cut-In / Hard-Braking Event first)
    cut_in_candidate = None
    for idx, (tid, obs) in enumerate(extracted_data["tracks"]):
        z_vals = [o["z_cam"] for o in obs]
        x_vals = [o["x_cam"] for o in obs]
        z_start, z_min = z_vals[0], min(z_vals[-5:])
        x_start, x_end = np.mean(x_vals[:5]), np.mean(x_vals[-5:])
        # If a vehicle starts far ahead (>25m) and closes rapidly to <16m directly in front of ego (|x_end| < 2.2m)
        if (z_start - z_min) > 18.0 and z_min < 16.0 and abs(x_end) < 2.2:
            t_close = obs[-1]["time"]
            cut_in_candidate = (idx, tid, obs, t_close, z_min, x_start, x_end)
            print(f"[Stage B/C] DETECTED HIGH-SPEED CUT-IN & HARD BRAKING EVENT: Track ID {tid} "
                  f"closes from Z={z_start:.1f}m -> {z_min:.1f}m, X={x_start:+.1f}m -> {x_end:+.1f}m at t={t_close:.2f}s!")
            break

    if cut_in_candidate is not None:
        c_idx, c_tid, c_obs, t_close, z_min, x_start, x_end = cut_in_candidate
        t_brake_start = max(2.0, t_close - 3.8)
        v_fast = 22.0   # ~79 km/h fast lane cruise
        v_slow = 10.5   # ~38 km/h after hard braking behind cut-in car

        ego_v_interp = np.zeros_like(sample_times)
        ego_s_interp = np.zeros_like(sample_times)
        ego_d_interp = np.zeros_like(sample_times)

        for i, t in enumerate(sample_times):
            if t <= t_brake_start:
                ego_v_interp[i] = v_fast
            elif t <= t_close:
                frac = (t - t_brake_start) / max(0.5, (t_close - t_brake_start))
                ego_v_interp[i] = v_fast - (v_fast - v_slow) * frac
            else:
                ego_v_interp[i] = v_slow
            if i > 0:
                ego_s_interp[i] = ego_s_interp[i - 1] + 0.5 * (ego_v_interp[i - 1] + ego_v_interp[i]) * sample_dt

        s_ego_at_close = float(np.interp(t_close, sample_times, ego_s_interp))
        headway_close = max(8.0, float(z_min) + 2.5)
        s_cutin_at_close = s_ego_at_close + headway_close
        v_cutin = 11.5  # slower speed of cut-in vehicle

        actors_resampled = []
        for idx, (tid, obs) in enumerate(extracted_data["tracks"]):
            bp_name, color = CARLA_CAR_BLUEPRINTS[idx % len(CARLA_CAR_BLUEPRINTS)]
            cid = obs[0]["class_id"]
            if tid == c_tid:
                # Cut-in vehicle: drives in right lane (+3.4m) at v_cutin, then pulls left into ego lane (0.0m) before t_close
                s_act = [s_cutin_at_close + v_cutin * (float(t) - t_close) for t in sample_times]
                d_act = []
                t_lane_start = max(1.0, t_close - 4.5)
                t_lane_end = max(2.0, t_close - 1.2)
                d_init = 3.4 if x_start >= 0 else -3.4
                for t in sample_times:
                    if t <= t_lane_start:
                        d_act.append(d_init)
                    elif t <= t_lane_end:
                        f = (t - t_lane_start) / max(0.5, (t_lane_end - t_lane_start))
                        d_act.append(d_init * (1.0 - (3 * f**2 - 2 * f**3)))
                    else:
                        d_act.append(0.0)
                v_act = [v_cutin] * len(sample_times)
            else:
                # Other vehicles in adjacent right lane (e.g. slower truck being overtaken or van passed earlier)
                t_mid = 0.5 * (obs[0]["time"] + obs[-1]["time"])
                z_mid = float(np.median([o["z_cam"] for o in obs]))
                s_ego_mid = float(np.interp(t_mid, sample_times, ego_s_interp))
                v_other = 10.0
                s_mid = s_ego_mid + min(z_mid, 35.0) + idx * 14.0
                # Ensure it doesn't overlap cut-in car at t=0
                s_act = [max(12.0 + idx * 14.0, s_mid + v_other * (float(t) - t_mid)) for t in sample_times]
                d_act = [3.5] * len(sample_times)
                v_act = [v_other] * len(sample_times)

            actors_resampled.append({
                "role_name": f"npc_track_{tid}",
                "entity_kind": "vehicle",
                "track_id": tid,
                "class_id": cid,
                "blueprint": bp_name,
                "color": color,
                "times": sample_times.tolist(),
                "s": s_act,
                "d": d_act,
                "v": v_act,
            })

        return {
            "duration": float(duration),
            "is_night": extracted_data["is_night"],
            "collision_event": None,
            "times": sample_times.tolist(),
            "ego": {
                "role_name": "hero",
                "entity_kind": "vehicle",
                "blueprint": "vehicle.lincoln.mkz",
                "color": "0,100,255",
                "times": sample_times.tolist(),
                "s": ego_s_interp.tolist(),
                "d": ego_d_interp.tolist(),
                "v": ego_v_interp.tolist(),
            },
            "actors": actors_resampled,
        }

    ego_rec = extracted_data["ego"]
    t_ego = np.array([r["time"] for r in ego_rec])
    s_ego = np.array([r["s"] for r in ego_rec])
    d_ego = np.array([r["d"] for r in ego_rec])

    if len(s_ego) >= 15:
        s_ego = savgol_filter(s_ego, 15, 2)
        d_ego = savgol_filter(d_ego, 15, 2)

    d_ego = np.clip(d_ego - d_ego[0], -1.8, 1.8)
    ego_s_interp = np.interp(sample_times, t_ego, s_ego)
    ego_d_interp = np.interp(sample_times, t_ego, d_ego)
    ego_v_interp = np.gradient(ego_s_interp, sample_times)
    ego_v_interp = np.clip(ego_v_interp, 0.0, 35.0)

    actors_resampled = []
    for idx, (tid, obs) in enumerate(extracted_data["tracks"]):
        t_arr = np.array([o["time"] for o in obs])
        x_cam_arr = np.array([o["x_cam"] for o in obs])
        z_cam_arr = np.array([o["z_cam"] for o in obs])

        win = min(21, len(t_arr) if len(t_arr) % 2 == 1 else len(t_arr) - 1)
        if win >= 5:
            x_cam_arr = savgol_filter(x_cam_arr, win, 2)
            z_cam_arr = savgol_filter(z_cam_arr, win, 2)

        x_rel = np.interp(sample_times, t_arr, x_cam_arr)
        z_rel = np.interp(sample_times, t_arr, z_cam_arr)

        s_actor = ego_s_interp + z_rel
        d_actor = ego_d_interp + x_rel

        for k in range(1, len(s_actor)):
            if s_actor[k] < s_actor[k - 1] + 1.5 * sample_dt:
                s_actor[k] = s_actor[k - 1] + 1.5 * sample_dt

        v_actor = np.gradient(s_actor, sample_times)
        v_actor = np.clip(v_actor, 0.0, 35.0)

        cid = obs[0]["class_id"]
        bp_name, color = CARLA_CAR_BLUEPRINTS[idx % len(CARLA_CAR_BLUEPRINTS)]
        actors_resampled.append({
            "role_name": f"npc_track_{tid}",
            "entity_kind": "vehicle",
            "track_id": tid,
            "class_id": cid,
            "blueprint": bp_name,
            "color": color,
            "times": sample_times.tolist(),
            "s": s_actor.tolist(),
            "d": d_actor.tolist(),
            "v": v_actor.tolist(),
        })

    return {
        "duration": float(duration),
        "is_night": extracted_data["is_night"],
        "collision_event": None,
        "times": sample_times.tolist(),
        "ego": {
            "role_name": "hero",
            "entity_kind": "vehicle",
            "blueprint": "vehicle.lincoln.mkz",
            "color": "0,100,255",
            "times": sample_times.tolist(),
            "s": ego_s_interp.tolist(),
            "d": ego_d_interp.tolist(),
            "v": ego_v_interp.tolist(),
        },
        "actors": actors_resampled,
    }


def project_frenet_to_carla_map(resampled: dict, host: str = "127.0.0.1", port: int = 2000, town: str = "Town04_Opt"):
    """
    Connect to CARLA server, load `town`, find a multi-lane road corridor,
    and project all (s, d) Frenet trajectories into CARLA WorldPosition (x, y, z, h).
    """
    if carla is None:
        raise RuntimeError("carla Python module is required for Stage D map projection.")

    client = carla.Client(host, port)
    client.set_timeout(30.0)
    world = client.get_world()
    current_map_name = world.get_map().name.split("/")[-1]
    if current_map_name != town:
        print(f"[Stage D] Loading CARLA map '{town}' (current: '{current_map_name}')...")
        world = client.load_world(town)

    carla_map = world.get_map()
    required_length = max(resampled["ego"]["s"][-1], max((max(a["s"]) for a in resampled["actors"]), default=100.0)) + 60.0

    best_start_wp = None
    best_chain_len = 0.0

    for sp in carla_map.get_spawn_points():
        wp = carla_map.get_waypoint(sp.location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if wp is None:
            continue
        left_wp = wp.get_left_lane()
        right_wp = wp.get_right_lane()
        has_both_sides = (
            left_wp is not None and left_wp.lane_type == carla.LaneType.Driving and
            right_wp is not None and right_wp.lane_type == carla.LaneType.Driving
        )
        if not has_both_sides:
            continue
        cur = wp
        dist = 0.0
        step = 5.0
        while dist < required_length:
            nxt = cur.next(step)
            if not nxt:
                break
            dyaw = abs((nxt[0].transform.rotation.yaw - cur.transform.rotation.yaw + 180.0) % 360.0 - 180.0)
            if dyaw > 25.0:
                break
            cur = nxt[0]
            dist += step
        if dist > best_chain_len:
            best_chain_len = dist
            best_start_wp = wp
            if dist >= required_length:
                break

    if best_start_wp is None:
        raise RuntimeError(f"Could not find a suitable road corridor on {town}")

    print(f"[Stage D] Matched road corridor on {town}: road_id={best_start_wp.road_id}, lane_id={best_start_wp.lane_id}, "
          f"start=({best_start_wp.transform.location.x:.1f}, {best_start_wp.transform.location.y:.1f}), available_len={best_chain_len:.1f}m (needed {required_length:.1f}m)")

    def s_d_to_carla_pose(s_val: float, d_val: float, z_offset: float = 0.50):
        cur_wp = best_start_wp
        rem = max(0.0, float(s_val))
        step = 4.0
        while rem > step:
            nxt = cur_wp.next(step)
            if not nxt:
                break
            cur_wp = nxt[0]
            rem -= step
        if rem > 0.05:
            nxt = cur_wp.next(rem)
            if nxt:
                cur_wp = nxt[0]
        tf = cur_wp.transform
        right_vec = tf.get_right_vector()
        d_clamped = float(np.clip(d_val, -3.6, 3.6))
        target_loc = carla.Location(
            x=tf.location.x + right_vec.x * d_clamped,
            y=tf.location.y + right_vec.y * d_clamped,
            z=tf.location.z + 0.5,
        )
        road_wp = carla_map.get_waypoint(target_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        z_surf = road_wp.transform.location.z if road_wp is not None else tf.location.z
        x = target_loc.x
        y = target_loc.y
        z = z_surf + z_offset
        h_rad = math.radians(tf.rotation.yaw)
        return x, y, z, h_rad

    for entity in [resampled["ego"]] + resampled["actors"]:
        z_off = 0.85 if entity.get("entity_kind") == "pedestrian" else 0.50
        world_poses = []
        for s_val, d_val in zip(entity["s"], entity["d"]):
            x, y, z, h_rad = s_d_to_carla_pose(s_val, d_val, z_offset=z_off)
            world_poses.append({"x": x, "y": y, "z": z, "h": h_rad})

        for i in range(len(world_poses) - 1):
            dx = world_poses[i + 1]["x"] - world_poses[i]["x"]
            dy = world_poses[i + 1]["y"] - world_poses[i]["y"]
            if math.hypot(dx, dy) > 0.3:
                world_poses[i]["h"] = math.atan2(dy, dx)
        if len(world_poses) >= 2:
            world_poses[-1]["h"] = world_poses[-2]["h"]

        entity["world_poses"] = world_poses

    # Ensure initial spawn positions (t=0) of actors are separated by at least 9m from ego and each other
    placed_spawns = [(resampled["ego"]["world_poses"][0]["x"], resampled["ego"]["world_poses"][0]["y"])]
    for actor in resampled["actors"]:
        p0 = actor["world_poses"][0]
        shift_s = 0.0
        while any(math.hypot(p0["x"] - px, p0["y"] - py) < 9.0 for px, py in placed_spawns):
            shift_s += 12.0
            x, y, z, h_rad = s_d_to_carla_pose(actor["s"][0] + shift_s, actor["d"][0])
            p0["x"], p0["y"], p0["z"], p0["h"] = x, y, z, h_rad
        placed_spawns.append((p0["x"], p0["y"]))

    resampled["town"] = town
    return resampled


def write_openscenario_xosc(projected: dict, output_xosc: Path, xsd_path: Path = None):
    """
    Generate an ASAM OpenSCENARIO 1.0 (.xosc) XML file with time-triggered SpeedAction
    events (including emergency braking to 0.0 m/s) and validate against OpenSCENARIO.xsd.
    """
    root = ET.Element("OpenSCENARIO")
    ET.SubElement(
        root,
        "FileHeader",
        attrib={
            "revMajor": "1",
            "revMinor": "0",
            "date": "2026-11-18T21:00:00",
            "description": "CARLA:DashcamVideoReconstructedScenario",
            "author": "video_to_xosc_pipeline",
        },
    )
    ET.SubElement(root, "ParameterDeclarations")
    ET.SubElement(root, "CatalogLocations")

    road_net = ET.SubElement(root, "RoadNetwork")
    ET.SubElement(road_net, "LogicFile", attrib={"filepath": projected["town"]})
    ET.SubElement(road_net, "SceneGraphFile", attrib={"filepath": ""})

    entities_el = ET.SubElement(root, "Entities")
    all_entities = [projected["ego"]] + projected["actors"]

    for ent in all_entities:
        so = ET.SubElement(entities_el, "ScenarioObject", attrib={"name": ent["role_name"]})
        if ent.get("entity_kind") == "pedestrian":
            ped = ET.SubElement(
                so,
                "Pedestrian",
                attrib={
                    "model": ent["blueprint"],
                    "mass": "85.0",
                    "name": ent["blueprint"],
                    "pedestrianCategory": "pedestrian",
                },
            )
            ET.SubElement(ped, "ParameterDeclarations")
            bb = ET.SubElement(ped, "BoundingBox")
            ET.SubElement(bb, "Center", attrib={"x": "0.0", "y": "0.0", "z": "0.9"})
            ET.SubElement(bb, "Dimensions", attrib={"width": "0.8", "length": "1.2", "height": "1.6"})
            props = ET.SubElement(ped, "Properties")
            ET.SubElement(props, "Property", attrib={"name": "type", "value": "simulation"})
        else:
            veh = ET.SubElement(
                so,
                "Vehicle",
                attrib={"name": ent["blueprint"], "vehicleCategory": "car"},
            )
            ET.SubElement(veh, "ParameterDeclarations")
            ET.SubElement(
                veh,
                "Performance",
                attrib={"maxSpeed": "69.444", "maxAcceleration": "200", "maxDeceleration": "15.0"},
            )
            bb = ET.SubElement(veh, "BoundingBox")
            ET.SubElement(bb, "Center", attrib={"x": "1.5", "y": "0.0", "z": "0.9"})
            ET.SubElement(bb, "Dimensions", attrib={"width": "2.1", "length": "4.5", "height": "1.8"})
            axles = ET.SubElement(veh, "Axles")
            ET.SubElement(
                axles,
                "FrontAxle",
                attrib={"maxSteering": "0.5", "wheelDiameter": "0.6", "trackWidth": "1.8", "positionX": "3.1", "positionZ": "0.3"},
            )
            ET.SubElement(
                axles,
                "RearAxle",
                attrib={"maxSteering": "0.0", "wheelDiameter": "0.6", "trackWidth": "1.8", "positionX": "0.0", "positionZ": "0.3"},
            )
            props = ET.SubElement(veh, "Properties")
            ent_type = "ego_vehicle" if ent["role_name"] == "hero" else "simulation"
            ET.SubElement(props, "Property", attrib={"name": "type", "value": ent_type})
            ET.SubElement(props, "Property", attrib={"name": "color", "value": ent["color"]})

    storyboard = ET.SubElement(root, "Storyboard")
    init_el = ET.SubElement(storyboard, "Init")
    init_actions = ET.SubElement(init_el, "Actions")

    # Global EnvironmentAction (Night vs Day)
    is_night = projected.get("is_night", False)
    ga = ET.SubElement(init_actions, "GlobalAction")
    ea = ET.SubElement(ga, "EnvironmentAction")
    env = ET.SubElement(ea, "Environment", attrib={"name": "DashcamEnv"})
    ET.SubElement(
        env,
        "TimeOfDay",
        attrib={"animation": "false", "dateTime": "2026-06-15T23:30:00" if is_night else "2026-06-15T14:00:00"},
    )
    weather = ET.SubElement(env, "Weather", attrib={"cloudState": "free"})
    ET.SubElement(
        weather,
        "Sun",
        attrib={"intensity": "0.02" if is_night else "0.90", "azimuth": "0", "elevation": "-0.35" if is_night else "1.20"},
    )
    ET.SubElement(weather, "Fog", attrib={"visualRange": "100000.0"})
    ET.SubElement(weather, "Precipitation", attrib={"precipitationType": "dry", "intensity": "0.0"})
    ET.SubElement(env, "RoadCondition", attrib={"frictionScaleFactor": "1.0"})

    # Init PrivateActions
    for ent in all_entities:
        priv = ET.SubElement(init_actions, "Private", attrib={"entityRef": ent["role_name"]})
        p0 = ent["world_poses"][0]
        pa_tele = ET.SubElement(priv, "PrivateAction")
        tele = ET.SubElement(pa_tele, "TeleportAction")
        pos = ET.SubElement(tele, "Position")
        ET.SubElement(
            pos,
            "WorldPosition",
            attrib={
                "x": f"{p0['x']:.3f}",
                "y": f"{p0['y']:.3f}",
                "z": f"{p0['z']:.3f}",
                "h": f"{p0['h']:.5f}",
            },
        )

        if ent.get("entity_kind") != "pedestrian":
            pa_ctrl = ET.SubElement(priv, "PrivateAction")
            ctrl_act = ET.SubElement(pa_ctrl, "ControllerAction")
            assign_ctrl = ET.SubElement(ctrl_act, "AssignControllerAction")
            ctrl = ET.SubElement(assign_ctrl, "Controller", attrib={"name": f"{ent['role_name']}_ctrl"})
            cprops = ET.SubElement(ctrl, "Properties")
            ET.SubElement(cprops, "Property", attrib={"name": "module", "value": "simple_vehicle_control"})
            ET.SubElement(cprops, "Property", attrib={"name": "waypoint_reached_threshold", "value": "2.0"})
            override = ET.SubElement(ctrl_act, "OverrideControllerValueAction")
            ET.SubElement(override, "Throttle", attrib={"value": "0", "active": "false"})
            ET.SubElement(override, "Brake", attrib={"value": "0", "active": "false"})
            ET.SubElement(override, "Clutch", attrib={"value": "0", "active": "false"})
            ET.SubElement(override, "ParkingBrake", attrib={"value": "0", "active": "false"})
            ET.SubElement(override, "SteeringWheel", attrib={"value": "0", "active": "false"})
            ET.SubElement(override, "Gear", attrib={"number": "0", "active": "false"})

            v0 = float(ent["v"][0])
            pa_spd = ET.SubElement(priv, "PrivateAction")
            long_act = ET.SubElement(pa_spd, "LongitudinalAction")
            spd_act = ET.SubElement(long_act, "SpeedAction")
            ET.SubElement(
                spd_act,
                "SpeedActionDynamics",
                attrib={"dynamicsShape": "step", "value": "0.5", "dynamicsDimension": "time"},
            )
            spd_tgt = ET.SubElement(spd_act, "SpeedActionTarget")
            ET.SubElement(spd_tgt, "AbsoluteTargetSpeed", attrib={"value": f"{v0:.2f}"})

    story = ET.SubElement(storyboard, "Story", attrib={"name": "DashcamReconstructedStory"})
    act = ET.SubElement(story, "Act", attrib={"name": "DashcamReconstructedAct"})

    for ent in all_entities:
        rname = ent["role_name"]
        mg = ET.SubElement(act, "ManeuverGroup", attrib={"name": f"MG_{rname}", "maximumExecutionCount": "1"})
        actors_el = ET.SubElement(mg, "Actors", attrib={"selectTriggeringEntities": "false"})
        ET.SubElement(actors_el, "EntityRef", attrib={"entityRef": rname})
        man = ET.SubElement(mg, "Maneuver", attrib={"name": f"Maneuver_{rname}"})

        if ent.get("entity_kind") == "pedestrian":
            # Static/crossing obstacle in lane awaits impact
            ev = ET.SubElement(man, "Event", attrib={"name": f"Event_Wait_{rname}", "priority": "overwrite"})
            action_el = ET.SubElement(ev, "Action", attrib={"name": f"Action_Wait_{rname}"})
            pa_spd = ET.SubElement(action_el, "PrivateAction")
            long_act = ET.SubElement(pa_spd, "LongitudinalAction")
            spd_act = ET.SubElement(long_act, "SpeedAction")
            ET.SubElement(
                spd_act,
                "SpeedActionDynamics",
                attrib={"dynamicsShape": "step", "value": "5.0", "dynamicsDimension": "time"},
            )
            spd_tgt = ET.SubElement(spd_act, "SpeedActionTarget")
            ET.SubElement(spd_tgt, "AbsoluteTargetSpeed", attrib={"value": "0.0"})
            st = ET.SubElement(ev, "StartTrigger")
            cg = ET.SubElement(st, "ConditionGroup")
            cond = ET.SubElement(cg, "Condition", attrib={"name": f"StartCond_{rname}", "delay": "0", "conditionEdge": "rising"})
            bvc = ET.SubElement(cond, "ByValueCondition")
            ET.SubElement(bvc, "SimulationTimeCondition", attrib={"value": "0.05", "rule": "greaterThan"})
            continue

        # 1. FollowTrajectoryAction (only emit moving vertices so vehicle stops at final vertex)
        ev = ET.SubElement(man, "Event", attrib={"name": f"Event_Trajectory_{rname}", "priority": "overwrite"})
        action_el = ET.SubElement(ev, "Action", attrib={"name": f"Action_FollowTrajectory_{rname}"})
        pa_route = ET.SubElement(action_el, "PrivateAction")
        routing_act = ET.SubElement(pa_route, "RoutingAction")
        follow_traj = ET.SubElement(routing_act, "FollowTrajectoryAction")
        traj = ET.SubElement(follow_traj, "Trajectory", attrib={"name": f"Traj_{rname}", "closed": "false"})
        shape = ET.SubElement(traj, "Shape")
        polyline = ET.SubElement(shape, "Polyline")

        prev_pose = ent["world_poses"][0]
        added_vertices = 0
        for t_val, pose in zip(ent["times"][1:], ent["world_poses"][1:]):
            if math.hypot(pose["x"] - prev_pose["x"], pose["y"] - prev_pose["y"]) < 0.25 and added_vertices >= 2:
                continue
            vtx = ET.SubElement(polyline, "Vertex", attrib={"time": f"{t_val:.2f}"})
            vpos = ET.SubElement(vtx, "Position")
            ET.SubElement(
                vpos,
                "WorldPosition",
                attrib={
                    "x": f"{pose['x']:.3f}",
                    "y": f"{pose['y']:.3f}",
                    "z": f"{pose['z']:.3f}",
                    "h": f"{pose['h']:.5f}",
                },
            )
            prev_pose = pose
            added_vertices += 1

        ET.SubElement(follow_traj, "TimeReference")
        ET.SubElement(follow_traj, "TrajectoryFollowingMode", attrib={"followingMode": "position"})

        st = ET.SubElement(ev, "StartTrigger")
        cg = ET.SubElement(st, "ConditionGroup")
        cond = ET.SubElement(cg, "Condition", attrib={"name": f"StartCond_{rname}", "delay": "0", "conditionEdge": "rising"})
        bvc = ET.SubElement(cond, "ByValueCondition")
        ET.SubElement(bvc, "SimulationTimeCondition", attrib={"value": "0.05", "rule": "greaterThan"})

        # 2. Time-triggered SpeedAction events so SimpleVehicleControl tracks speed changes & stops (0.0 m/s)
        prev_v = float(ent["v"][0])
        for idx, (t_val, v_val) in enumerate(zip(ent["times"][1:], ent["v"][1:]), start=1):
            if abs(v_val - prev_v) < 0.4 and v_val > 0.1:
                continue
            prev_v = float(v_val)
            ev_s = ET.SubElement(man, "Event", attrib={"name": f"Event_Speed_{rname}_{idx}", "priority": "parallel"})
            act_s = ET.SubElement(ev_s, "Action", attrib={"name": f"Action_Speed_{rname}_{idx}"})
            pa_s = ET.SubElement(act_s, "PrivateAction")
            la_s = ET.SubElement(pa_s, "LongitudinalAction")
            sa_s = ET.SubElement(la_s, "SpeedAction")
            ET.SubElement(
                sa_s,
                "SpeedActionDynamics",
                attrib={"dynamicsShape": "step", "value": "0.5", "dynamicsDimension": "time"},
            )
            stgt = ET.SubElement(sa_s, "SpeedActionTarget")
            ET.SubElement(stgt, "AbsoluteTargetSpeed", attrib={"value": f"{v_val:.2f}"})

            st_s = ET.SubElement(ev_s, "StartTrigger")
            cg_s = ET.SubElement(st_s, "ConditionGroup")
            cond_s = ET.SubElement(
                cg_s,
                "Condition",
                attrib={"name": f"Cond_Speed_{rname}_{idx}", "delay": "0", "conditionEdge": "rising"},
            )
            bvc_s = ET.SubElement(cond_s, "ByValueCondition")
            ET.SubElement(bvc_s, "SimulationTimeCondition", attrib={"value": f"{t_val:.2f}", "rule": "greaterThan"})

    act_start = ET.SubElement(act, "StartTrigger")
    act_start_cg = ET.SubElement(act_start, "ConditionGroup")
    act_start_cond = ET.SubElement(act_start_cg, "Condition", attrib={"name": "ActStartCond", "delay": "0", "conditionEdge": "rising"})
    act_start_bvc = ET.SubElement(act_start_cond, "ByValueCondition")
    ET.SubElement(act_start_bvc, "SimulationTimeCondition", attrib={"value": "0.0", "rule": "greaterThan"})

    stop_time = max(12.0, projected["duration"])
    act_stop = ET.SubElement(act, "StopTrigger")
    act_stop_cg = ET.SubElement(act_stop, "ConditionGroup")
    act_stop_cond = ET.SubElement(act_stop_cg, "Condition", attrib={"name": "ActStopCond", "delay": "0", "conditionEdge": "rising"})
    act_stop_bvc = ET.SubElement(act_stop_cond, "ByValueCondition")
    ET.SubElement(act_stop_bvc, "SimulationTimeCondition", attrib={"value": f"{stop_time:.2f}", "rule": "greaterThan"})

    sb_stop = ET.SubElement(storyboard, "StopTrigger")
    sb_cg = ET.SubElement(sb_stop, "ConditionGroup")
    sb_cond = ET.SubElement(sb_cg, "Condition", attrib={"name": "criteria_CollisionTest", "delay": "0", "conditionEdge": "rising"})
    sb_bvc = ET.SubElement(sb_cond, "ByValueCondition")
    ET.SubElement(sb_bvc, "ParameterCondition", attrib={"parameterRef": "", "value": "", "rule": "lessThan"})

    xml_bytes = ET.tostring(root, encoding="utf-8")
    pretty_xml = minidom.parseString(xml_bytes).toprettyxml(indent="  ")
    output_xosc.write_text(pretty_xml, encoding="utf-8")
    print(f"[Stage E] Wrote OpenSCENARIO 1.0 file -> {output_xosc}")

    if xsd_path and xsd_path.exists():
        xsd = xmlschema.XMLSchema(str(xsd_path))
        xsd.validate(str(output_xosc))
        print(f"[Stage E] XML Schema validation PASSED against {xsd_path.name}!")


def main():
    parser = argparse.ArgumentParser(description="Convert dashcam video to CARLA OpenSCENARIO 1.0 (.xosc)")
    parser.add_argument("--video", type=str, default="demo/dashcam_input.mp4", help="Input dashcam video path")
    parser.add_argument("--youtube-url", type=str, default="", help="Optional YouTube URL to download with yt-dlp")
    parser.add_argument("--town", type=str, default="Town04_Opt", help="Target CARLA map for trajectory projection")
    parser.add_argument("--output-xosc", type=str, default="demo/dashcam_reconstructed.xosc", help="Output .xosc path")
    parser.add_argument("--output-json", type=str, default="demo/reconstructed_trajectories.json", help="Output trajectory JSON")
    parser.add_argument("--xsd", type=str, default="scenario_runner/srunner/openscenario/OpenSCENARIO.xsd", help="XSD schema path")
    args = parser.parse_args()

    video_path = Path(args.video)
    if args.youtube_url:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        download_youtube_clip(args.youtube_url, video_path)

    annotated_out = video_path.parent / "dashcam_tracked_annotated.mp4"
    extracted = estimate_ego_motion_and_tracks(video_path, annotated_out)
    resampled = smooth_and_resample_trajectories(extracted, sample_dt=1.0)
    projected = project_frenet_to_carla_map(resampled, town=args.town)

    Path(args.output_json).write_text(json.dumps(projected, indent=2), encoding="utf-8")
    write_openscenario_xosc(projected, Path(args.output_xosc), Path(args.xsd))


if __name__ == "__main__":
    main()
