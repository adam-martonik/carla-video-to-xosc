"""Multi-object tracking (BoT-SORT/ByteTrack), radial-FOE visual odometry, and tracklet stitching."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from .calibration import CameraIntrinsics, build_camera_intrinsics, estimate_monocular_3d
from .config import ALL_TARGET_CLASS_IDS, PipelineConfig, VEHICLE_CLASS_IDS, VRU_ANIMAL_CLASS_IDS


def estimate_frame_ego_motion(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    static_mask: np.ndarray,
    intr: CameraIntrinsics,
    base_speed_mps: float,
) -> Tuple[float, float, float]:
    """
    Estimate frame-to-frame ego yaw increment (rad) and forward speed (m/s) from
    static background optical flow by decoupling horizontal yaw rotation from
    radial expansion around the Focus of Expansion (cx, cy).

    Returns (d_yaw_rad, ego_speed_mps, radial_flow_px).
    """
    pts0 = cv2.goodFeaturesToTrack(
        prev_gray, maxCorners=320, qualityLevel=0.01, minDistance=10, mask=static_mask
    )
    if pts0 is None or len(pts0) < 8:
        return 0.0, 0.0, 0.0

    pts1, st, _ = cv2.calcOpticalFlowPyrLK(
        prev_gray, curr_gray, pts0, None, winSize=(21, 21), maxLevel=3
    )
    if pts1 is None or st is None:
        return 0.0, 0.0, 0.0

    good0 = pts0[st.flatten() == 1].reshape(-1, 2)
    good1 = pts1[st.flatten() == 1].reshape(-1, 2)
    if len(good0) < 8:
        return 0.0, 0.0, 0.0

    # Filter out moving-object outliers via RANSAC homography inliers when enough points exist
    inlier_mask = np.ones(len(good0), dtype=bool)
    if len(good0) >= 12:
        _, mask = cv2.findHomography(good0, good1, cv2.RANSAC, 3.0)
        if mask is not None and int(np.sum(mask)) >= 6:
            inlier_mask = mask.ravel().astype(bool)

    g0 = good0[inlier_mask]
    g1 = good1[inlier_mask]
    flow = g1 - g0

    # Pure yaw rotation produces uniform horizontal shift du_rot; subtract it before radial expansion
    du_med = float(np.median(flow[:, 0]))
    d_yaw = float(np.clip(-du_med / max(intr.fx, 1.0), -0.025, 0.025))

    # Compensate horizontal rotation flow to isolate translational radial expansion from FOE
    flow_trans = flow.copy()
    flow_trans[:, 0] -= du_med

    vec_from_foe = g0 - np.array([[intr.cx, intr.cy]], dtype=float)
    radii = np.hypot(vec_from_foe[:, 0], vec_from_foe[:, 1])
    valid_r = radii > max(25.0, 0.05 * intr.width)
    if int(np.sum(valid_r)) < 4:
        return d_yaw, 0.0, 0.0

    unit_rad = vec_from_foe[valid_r] / radii[valid_r, None]
    radial_proj = np.sum(flow_trans[valid_r] * unit_rad, axis=1)
    # Normalize by normalized image radius so speed is invariant to resolution
    norm_r = radii[valid_r] / max(intr.width * 0.5, 1.0)
    radial_rate = float(np.median(radial_proj / np.maximum(norm_r, 0.2)))

    if radial_rate < 0.35:
        speed_factor = 0.0
    else:
        speed_factor = float(np.clip(radial_rate / 4.5, 0.0, 1.45))

    return d_yaw, base_speed_mps * speed_factor, radial_rate


def stitch_fragmented_tracks(
    raw_tracks: Dict[int, List[Dict[str, Any]]],
    max_gap_s: float = 1.5,
    max_dist_m: float = 5.0,
) -> Dict[int, List[Dict[str, Any]]]:
    """
    Merge fragmented tracklets of the same object class separated by short occlusion gaps
    by linearly extrapolating the earlier tracklet's (x_cam, z_cam) to the start of the
    later tracklet and matching within spatial gate max_dist_m.
    """
    sorted_items = sorted(
        [(tid, sorted(obs, key=lambda o: o["frame"])) for tid, obs in raw_tracks.items() if obs],
        key=lambda item: item[1][0]["time"],
    )
    stitched: List[Tuple[int, List[Dict[str, Any]]]] = []

    for tid, obs in sorted_items:
        t_start = obs[0]["time"]
        x_start = obs[0]["x_cam"]
        z_start = obs[0]["z_cam"]
        cls_new = max({o["class_id"] for o in obs}, key=lambda c: sum(1 for o in obs if o["class_id"] == c))

        best_idx = -1
        best_dist = max_dist_m

        for i, (_, existing_obs) in enumerate(stitched):
            t_end = existing_obs[-1]["time"]
            gap = t_start - t_end
            if gap <= 0.0 or gap > max_gap_s:
                continue
            cls_old = existing_obs[-1]["class_id"]
            if cls_old != cls_new:
                continue

            # Extrapolate end velocity of existing tracklet over last <= 5 samples
            tail = existing_obs[-min(5, len(existing_obs)):]
            dt_tail = max(tail[-1]["time"] - tail[0]["time"], 0.05)
            vx = (tail[-1]["x_cam"] - tail[0]["x_cam"]) / dt_tail
            vz = (tail[-1]["z_cam"] - tail[0]["z_cam"]) / dt_tail
            vx = float(np.clip(vx, -3.0, 3.0))
            vz = float(np.clip(vz, -15.0, 15.0))

            pred_x = existing_obs[-1]["x_cam"] + vx * gap
            pred_z = existing_obs[-1]["z_cam"] + vz * gap
            dist = math.hypot(pred_x - x_start, pred_z - z_start)
            if dist < best_dist:
                best_dist = dist
                best_idx = i

        if best_idx >= 0:
            stitched[best_idx][1].extend(obs)
            stitched[best_idx][1].sort(key=lambda o: o["frame"])
        else:
            stitched.append((tid, list(obs)))

    return {tid: obs for tid, obs in stitched}


def _select_yolo_model(candidates: List[str]) -> str:
    for cand in candidates:
        if Path(cand).exists():
            return cand
    return candidates[-1] if candidates else "yolo11n.pt"


def extract_ego_and_tracks(
    video_path: Path,
    cfg: PipelineConfig,
    annotated_video_out: Path | None = None,
) -> Dict[str, Any]:
    """
    Run YOLO + BoT-SORT (with Camera Motion Compensation), vanishing-point horizon calibration,
    radial-FOE RANSAC visual odometry, night/scene-cut detection, and tracklet stitching.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sample up to 8 evenly spaced frames for horizon estimation and night detection
    sample_grays: List[np.ndarray] = []
    sky_means: List[float] = []
    sample_indices = np.linspace(0, max(0, total_frames - 1), min(8, max(1, total_frames)), dtype=int)
    for idx in sample_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frm = cap.read()
        if ret and frm is not None:
            gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
            sample_grays.append(gray)
            sky_means.append(float(np.mean(gray[: int(height * 0.35), :])))
    cap.release()

    sky_mean = float(np.median(sky_means)) if sky_means else 120.0
    is_night = sky_mean < cfg.tracking.night_sky_threshold
    intr = build_camera_intrinsics(width, height, cfg.camera, sample_grays)

    print(
        f"[Stage B/C] Processing {video_path.name}: {width}x{height} @ {fps:.2f} FPS "
        f"({total_frames} frames, {total_frames / fps:.2f}s) | Mode: {'NIGHT' if is_night else 'DAY'} "
        f"(sky={sky_mean:.1f}) | HFOV={intr.hfov_deg:.1f}deg, cy={intr.cy:.1f}px "
        f"({'VP-cal' if intr.horizon_estimated else 'prior'})"
    )

    model_path = _select_yolo_model(cfg.tracking.model_candidates)
    model = YOLO(model_path)

    writer = None
    if annotated_video_out is not None:
        annotated_video_out.parent.mkdir(parents=True, exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(annotated_video_out), fourcc, fps, (width, height))

    prev_gray = None
    ego_yaw = 0.0
    ego_s = 0.0
    ego_records: List[Dict[str, Any]] = []
    raw_tracks: Dict[int, List[Dict[str, Any]]] = {}
    vru_impact_candidates: List[Dict[str, Any]] = []
    camera_cut_frame = None
    oncoming_headlight_frames = 0

    tracker_yaml = cfg.tracking.tracker
    try:
        results_gen = model.track(
            source=str(video_path),
            tracker=tracker_yaml,
            classes=ALL_TARGET_CLASS_IDS,
            conf=cfg.tracking.conf_threshold,
            persist=True,
            stream=True,
            verbose=False,
        )
        # Touch first result to verify tracker YAML is valid
        results_iter = iter(results_gen)
        first_res = next(results_iter, None)
    except Exception:
        tracker_yaml = cfg.tracking.fallback_tracker
        results_gen = model.track(
            source=str(video_path),
            tracker=tracker_yaml,
            classes=ALL_TARGET_CLASS_IDS,
            conf=cfg.tracking.conf_threshold,
            persist=True,
            stream=True,
            verbose=False,
        )
        results_iter = iter(results_gen)
        first_res = next(results_iter, None)

    def _iter_results():
        if first_res is not None:
            yield first_res
        yield from results_iter

    dt = 1.0 / fps
    for frame_idx, r in enumerate(_iter_results()):
        t_sec = frame_idx / fps
        frame = r.orig_img.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if (
            prev_gray is not None
            and frame_idx > int(cfg.tracking.scene_cut_min_time_s * fps)
            and camera_cut_frame is None
        ):
            frame_diff = float(np.mean(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32))))
            if frame_diff > cfg.tracking.scene_cut_diff_threshold:
                camera_cut_frame = frame_idx
                print(f"[Stage B/C] Detected multi-camera cut at frame {frame_idx} (t={t_sec:.2f}s, diff={frame_diff:.1f})")

        if is_night and camera_cut_frame is None and frame_idx < int(9.0 * fps):
            roi = gray[int(height * 0.42): int(height * 0.58), int(width * 0.05): int(width * 0.52)]
            if int(np.sum(roi > 240)) > 15:
                oncoming_headlight_frames += 1

        static_mask = np.full((height, width), 255, dtype=np.uint8)
        static_mask[: int(height * 0.35), :] = 0
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

        d_yaw, ego_v, radial_flow = (0.0, cfg.tracking.base_ego_speed_mps * 0.5, 0.0)
        if prev_gray is not None:
            d_yaw, ego_v, radial_flow = estimate_frame_ego_motion(
                prev_gray, gray, static_mask, intr, cfg.tracking.base_ego_speed_mps
            )

        prev_gray = gray
        ego_yaw += d_yaw
        ego_s += ego_v * dt
        ego_records.append({
            "frame": frame_idx,
            "time": t_sec,
            "s": ego_s,
            "d": 0.0,
            "yaw": ego_yaw,
            "speed": ego_v,
            "radial_flow": radial_flow,
        })

        if camera_cut_frame is not None and frame_idx >= camera_cut_frame:
            if writer is not None:
                cv2.putText(
                    frame,
                    f"t={t_sec:5.2f}s | Camera Cut (Post-Event)",
                    (20, 34),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (100, 220, 255),
                    2,
                )
                writer.write(frame)
            continue

        for (x1, y1, x2, y2), tid, cid, conf in zip(boxes_xyxy, track_ids, cls_ids, confs):
            bw = float(x2 - x1)
            bh = float(y2 - y1)
            vb = float(y2)
            cid_int = int(cid)

            if bw > 0.72 * width or (vb > 0.95 * height and bw > 0.42 * width) or bw < 14 or bh < 14:
                continue
            if cid_int in VEHICLE_CLASS_IDS and float(conf) < cfg.tracking.vehicle_conf_threshold:
                continue

            spec = cfg.get_class_spec(cid_int)
            x_cam, z_cam, sigma_z = estimate_monocular_3d(x1, y1, x2, y2, spec, intr, cfg.camera)

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
                "sigma_z": sigma_z,
            }
            tid_int = int(tid)
            raw_tracks.setdefault(tid_int, []).append(obs_entry)

            if cid_int in VRU_ANIMAL_CLASS_IDS and abs(x_cam) < 3.5 and z_cam < 28.0:
                vru_impact_candidates.append(obs_entry)

            if writer is not None:
                color = (0, 80, 255) if cid_int in VRU_ANIMAL_CLASS_IDS else ((0, 220, 80) if x_cam < 0 else (40, 140, 255))
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                label = f"ID:{tid_int} {spec.name} Z:{z_cam:.1f}m X:{x_cam:+.1f}m"
                cv2.putText(frame, label, (int(x1), max(20, int(y1) - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, color, 2)

        if writer is not None:
            cv2.line(frame, (0, int(intr.cy)), (width, int(intr.cy)), (255, 255, 0), 1)
            cv2.putText(
                frame,
                f"t={t_sec:5.2f}s | Ego s={ego_s:5.1f}m v={ego_v * 3.6:4.1f}km/h | {tracker_yaml}",
                (20, 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.70,
                (255, 255, 255),
                2,
            )
            writer.write(frame)

    if writer is not None:
        writer.release()

    # Tracklet stitching across brief occlusions
    stitched_tracks = stitch_fragmented_tracks(
        raw_tracks,
        max_gap_s=cfg.tracking.stitch_max_gap_s,
        max_dist_m=cfg.tracking.stitch_max_dist_m,
    )

    valid_tracks = []
    for tid, obs in stitched_tracks.items():
        if len(obs) < cfg.tracking.min_track_frames:
            continue
        if obs[-1]["time"] - obs[0]["time"] < cfg.tracking.min_track_duration_s:
            continue
        valid_tracks.append((tid, obs))

    valid_tracks.sort(key=lambda item: len(item[1]), reverse=True)
    selected_tracks = valid_tracks[: cfg.tracking.max_actors]

    collision_event = None
    if len(vru_impact_candidates) >= 1:
        cands = sorted(vru_impact_candidates, key=lambda o: o["frame"])
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

    return {
        "fps": fps,
        "width": width,
        "height": height,
        "duration": total_frames / fps,
        "tracker_used": tracker_yaml,
        "intrinsics": {
            "fx": intr.fx,
            "fy": intr.fy,
            "cx": intr.cx,
            "cy": intr.cy,
            "hfov_deg": intr.hfov_deg,
            "cam_height_m": intr.cam_height_m,
            "horizon_estimated": intr.horizon_estimated,
        },
        "is_night": is_night,
        "camera_cut_time": (camera_cut_frame / fps) if camera_cut_frame else None,
        "has_oncoming_traffic": oncoming_headlight_frames >= 25,
        "collision_event": collision_event,
        "ego": ego_records,
        "tracks": selected_tracks,
    }
