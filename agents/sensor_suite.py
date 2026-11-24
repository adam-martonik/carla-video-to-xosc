"""
Smart-car perception sensor suite for CARLA: Tesla-style 360-degree multi-camera vision rig,
roof-mounted 360-degree spinning LiDAR (standard & semantic), and optional front radar.

Presets (pass via `--sensors <preset>` or comma-separated sensor names):
  - none           : disabled (default; zero overhead)
  - tesla          : 8 Tesla-style surround RGB cameras (front_main, front_wide, front_narrow,
                     left_pillar, right_pillar, left_repeater, right_repeater, rear)
  - tesla_lidar    : 8 Tesla surround RGB cameras + roof 360-deg spinning LiDAR
  - surround_lidar : 4 surround RGB cameras (front_wide, left_pillar, right_pillar, rear) + roof LiDAR
  - front_lidar    : 2 front RGB cameras (front_main, front_wide) + roof LiDAR (fast RL preset)
  - lidar          : roof 360-deg spinning LiDAR only (fastest spatial RL preset)
  - semantic_lidar : roof 360-deg semantic LiDAR only
  - tesla_full     : 8 Tesla surround RGB cameras + roof LiDAR + front radar

All geometry transforms, point-cloud projections (BEV raster & 360-deg polar obstacle sectors),
and mosaic rendering are pure NumPy/OpenCV so they are fast and unit-testable without a CARLA server.
"""
import json
import math
import os
import queue

import numpy as np

# Per-vehicle-class rough physical envelope: (half_len_m, half_width_m, roof_z_m, windshield_x_m, windshield_z_m)
VEHICLE_DIMENSIONS = {
    "car":        (2.35, 0.95, 1.48, 1.15, 1.35),
    "suv":        (2.45, 1.00, 1.78, 1.30, 1.65),
    "jeep":       (2.45, 1.00, 1.78, 1.30, 1.65),
    "van":        (2.70, 1.02, 2.15, 1.70, 2.00),
    "truck":      (3.75, 1.25, 2.85, 2.60, 2.55),
    "motorcycle": (1.05, 0.40, 1.30, 0.45, 1.25),
    "bike":       (0.90, 0.30, 1.25, 0.35, 1.25),
}

# Canonical smart-car sensor catalog.
# Positions are specified in normalized vehicle-body fractions (fx, fy, fz_mode) so every camera/LiDAR
# adapts cleanly across sedans, SUVs, vans, trucks, and motorcycles without clipping inside the mesh.
SENSOR_CATALOG = {
    # --- Tesla HW3/HW4 8-camera surround layout ---
    "front_main": {
        "kind": "camera", "blueprint": "sensor.camera.rgb", "fov": 65.0,
        "mount": "windshield", "dx": 0.05, "dy": 0.0, "dz": 0.02, "pitch": -2.0, "yaw": 0.0, "roll": 0.0,
        "label": "FRONT MAIN (65deg)",
    },
    "front_wide": {
        "kind": "camera", "blueprint": "sensor.camera.rgb", "fov": 120.0,
        "mount": "windshield", "dx": 0.08, "dy": 0.0, "dz": -0.02, "pitch": -3.0, "yaw": 0.0, "roll": 0.0,
        "label": "FRONT WIDE (120deg)",
    },
    "front_narrow": {
        "kind": "camera", "blueprint": "sensor.camera.rgb", "fov": 35.0,
        "mount": "windshield", "dx": 0.05, "dy": 0.0, "dz": 0.05, "pitch": -1.0, "yaw": 0.0, "roll": 0.0,
        "label": "FRONT NARROW (35deg)",
    },
    "left_pillar": {
        "kind": "camera", "blueprint": "sensor.camera.rgb", "fov": 85.0,
        "mount": "b_pillar_left", "dx": 0.0, "dy": -0.08, "dz": -0.10, "pitch": -4.0, "yaw": -65.0, "roll": 0.0,
        "label": "LEFT B-PILLAR",
    },
    "right_pillar": {
        "kind": "camera", "blueprint": "sensor.camera.rgb", "fov": 85.0,
        "mount": "b_pillar_right", "dx": 0.0, "dy": 0.08, "dz": -0.10, "pitch": -4.0, "yaw": 65.0, "roll": 0.0,
        "label": "RIGHT B-PILLAR",
    },
    "left_repeater": {
        "kind": "camera", "blueprint": "sensor.camera.rgb", "fov": 85.0,
        "mount": "fender_left", "dx": 0.0, "dy": -0.08, "dz": 0.0, "pitch": -2.0, "yaw": -140.0, "roll": 0.0,
        "label": "LEFT REPEATER",
    },
    "right_repeater": {
        "kind": "camera", "blueprint": "sensor.camera.rgb", "fov": 85.0,
        "mount": "fender_right", "dx": 0.0, "dy": 0.08, "dz": 0.0, "pitch": -2.0, "yaw": 140.0, "roll": 0.0,
        "label": "RIGHT REPEATER",
    },
    "rear": {
        "kind": "camera", "blueprint": "sensor.camera.rgb", "fov": 110.0,
        "mount": "rear", "dx": -0.08, "dy": 0.0, "dz": 0.0, "pitch": -6.0, "yaw": 180.0, "roll": 0.0,
        "label": "REAR (110deg)",
    },
    # --- Roof 360-degree spinning LiDAR (Waymo / autonomous research style) ---
    "lidar": {
        "kind": "lidar", "blueprint": "sensor.lidar.ray_cast",
        "mount": "roof", "dx": 0.0, "dy": 0.0, "dz": 0.30, "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
        "channels": 32, "range": 80.0, "points_per_second": 200000,
        "upper_fov": 10.0, "lower_fov": -30.0, "horizontal_fov": 360.0,
        "dropoff_general_rate": 0.0,
        "label": "LIDAR 360",
    },
    "semantic_lidar": {
        "kind": "semantic_lidar", "blueprint": "sensor.lidar.ray_cast_semantic",
        "mount": "roof", "dx": 0.0, "dy": 0.0, "dz": 0.30, "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
        "channels": 32, "range": 80.0, "points_per_second": 200000,
        "upper_fov": 10.0, "lower_fov": -30.0, "horizontal_fov": 360.0,
        "label": "SEMANTIC LIDAR 360",
    },
    # --- Front millimeter-wave radar ---
    "radar_front": {
        "kind": "radar", "blueprint": "sensor.other.radar",
        "mount": "front_bumper", "dx": 0.08, "dy": 0.0, "dz": 0.0, "pitch": 0.0, "yaw": 0.0, "roll": 0.0,
        "horizontal_fov": 40.0, "vertical_fov": 15.0, "range": 100.0, "points_per_second": 2000,
        "label": "RADAR FRONT",
    },
}

TESLA_CAMERAS = (
    "front_main", "front_wide", "front_narrow",
    "left_pillar", "right_pillar",
    "left_repeater", "right_repeater",
    "rear",
)

SENSOR_PRESETS = {
    "none": (),
    "tesla": TESLA_CAMERAS,
    "tesla_lidar": TESLA_CAMERAS + ("lidar",),
    "tesla_full": TESLA_CAMERAS + ("lidar", "radar_front"),
    "surround_lidar": ("front_wide", "left_pillar", "right_pillar", "rear", "lidar"),
    "front_lidar": ("front_main", "front_wide", "lidar"),
    "lidar": ("lidar",),
    "semantic_lidar": ("semantic_lidar",),
}


def resolve_sensor_spec(sensors):
    """Expand a preset name, comma-separated list, or iterable into an ordered deduplicated list of sensor configs.
    Returns [] for None / '' / 'none' / 'off'."""
    if sensors is None:
        return []
    if isinstance(sensors, str):
        raw = sensors.strip()
        if not raw or raw.lower() in ("none", "off", "false", "0"):
            return []
        tokens = [t.strip().lower() for t in raw.split(",") if t.strip()]
    elif isinstance(sensors, (list, tuple)):
        tokens = [str(t).strip().lower() for t in sensors if str(t).strip()]
    else:
        raise ValueError("sensors must be a preset name, comma-separated string, or list, got %r" % (sensors,))

    names = []
    for tok in tokens:
        if tok in ("none", "off"):
            continue
        if tok in SENSOR_PRESETS:
            for nm in SENSOR_PRESETS[tok]:
                if nm not in names:
                    names.append(nm)
        elif tok in SENSOR_CATALOG:
            if tok not in names:
                names.append(tok)
        else:
            avail = sorted(set(SENSOR_PRESETS) | set(SENSOR_CATALOG))
            raise ValueError("unknown sensor or preset %r (available: %s)" % (tok, ", ".join(avail)))
    return [dict(SENSOR_CATALOG[nm], name=nm) for nm in names]


def parse_resolution(res, default=(320, 180)):
    """Parse 'WxH' string or (w, h) tuple into validated positive integer (width, height)."""
    if res is None:
        return int(default[0]), int(default[1])
    if isinstance(res, (list, tuple)) and len(res) == 2:
        w, h = int(res[0]), int(res[1])
    elif isinstance(res, str):
        parts = res.lower().replace("*", "x").split("x")
        if len(parts) != 2:
            raise ValueError("resolution must look like '320x180', got %r" % (res,))
        w, h = int(parts[0].strip()), int(parts[1].strip())
    else:
        raise ValueError("invalid resolution %r" % (res,))
    if w < 32 or h < 32 or w > 3840 or h > 2160:
        raise ValueError("resolution out of bounds (32..3840 x 32..2160): %dx%d" % (w, h))
    return w, h


def _vehicle_envelope(vtype="car", bbox_extent=None):
    key = str(vtype or "car").strip().lower()
    half_len, half_w, roof_z, ws_x, ws_z = VEHICLE_DIMENSIONS.get(key, VEHICLE_DIMENSIONS["car"])
    if bbox_extent is not None:
        ex = float(getattr(bbox_extent, "x", 0.0) or 0.0)
        ey = float(getattr(bbox_extent, "y", 0.0) or 0.0)
        ez = float(getattr(bbox_extent, "z", 0.0) or 0.0)
        if ex > 0.5:
            half_len = max(half_len, ex)
            ws_x = max(0.35, 0.55 * ex)
        if ey > 0.25:
            half_w = max(half_w, ey)
        if ez > 0.35:
            roof_z = max(roof_z, 2.0 * ez)
            ws_z = max(1.10, 2.0 * ez - 0.12)
    return half_len, half_w, roof_z, ws_x, ws_z


def sensor_mount_transform(sensor_cfg, vtype="car", bbox_extent=None):
    """Compute local CARLA mount pose `(x, y, z, pitch, yaw, roll)` on `vtype` so cameras & LiDAR never clip the body.
    CARLA vehicle frame: +X forward, +Y right, +Z up; yaw > 0 turns right."""
    half_len, half_w, roof_z, ws_x, ws_z = _vehicle_envelope(vtype, bbox_extent)
    mount = sensor_cfg.get("mount", "roof")
    dx = float(sensor_cfg.get("dx", 0.0))
    dy = float(sensor_cfg.get("dy", 0.0))
    dz = float(sensor_cfg.get("dz", 0.0))
    if mount == "windshield":
        x, y, z = ws_x + dx, dy, ws_z + dz
    elif mount == "roof":
        x, y, z = 0.05 * half_len + dx, dy, roof_z + dz
    elif mount == "b_pillar_left":
        x, y, z = 0.05 * half_len + dx, -(half_w + abs(dy)), max(1.05, roof_z - 0.22 + dz)
    elif mount == "b_pillar_right":
        x, y, z = 0.05 * half_len + dx, (half_w + abs(dy)), max(1.05, roof_z - 0.22 + dz)
    elif mount == "fender_left":
        x, y, z = 0.42 * half_len + dx, -(half_w + abs(dy)), max(0.75, 0.55 * roof_z + dz)
    elif mount == "fender_right":
        x, y, z = 0.42 * half_len + dx, (half_w + abs(dy)), max(0.75, 0.55 * roof_z + dz)
    elif mount == "rear":
        x, y, z = -(half_len + abs(dx)), dy, max(0.80, 0.68 * roof_z + dz)
    elif mount == "front_bumper":
        x, y, z = half_len + abs(dx), dy, max(0.45, 0.38 * roof_z + dz)
    else:
        x, y, z = dx, dy, roof_z + dz
    return (
        round(x, 3),
        round(y, 3),
        round(z, 3),
        round(float(sensor_cfg.get("pitch", 0.0)), 2),
        round(float(sensor_cfg.get("yaw", 0.0)), 2),
        round(float(sensor_cfg.get("roll", 0.0)), 2),
    )


def wants_tesla_blueprint(sensors):
    """True when the user selected a Tesla preset ('tesla', 'tesla_lidar', 'tesla_full')."""
    if not sensors:
        return False
    if isinstance(sensors, str):
        toks = [t.strip().lower() for t in sensors.split(",") if t.strip()]
    else:
        toks = [str(t).strip().lower() for t in sensors]
    return any(t in ("tesla", "tesla_lidar", "tesla_full") for t in toks)


# ----------------------------------------------------------------------------------------- Raw CARLA buffer decoders
def bgra_to_bgr(raw_bytes, width, height):
    """Convert raw CARLA BGRA camera bytes into contiguous HxWx3 uint8 BGR array."""
    arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((int(height), int(width), 4))
    return arr[:, :, :3].copy()


def bgra_to_rgb(raw_bytes, width, height):
    """Convert raw CARLA BGRA camera bytes into contiguous HxWx3 uint8 RGB array (standard for PyTorch/PIL)."""
    arr = np.frombuffer(raw_bytes, dtype=np.uint8).reshape((int(height), int(width), 4))
    return arr[:, :, 2::-1].copy()


def lidar_bytes_to_points(raw_bytes):
    """Decode raw CARLA `sensor.lidar.ray_cast` bytes into float32 (N, 4) array: [x, y, z, intensity]."""
    if not raw_bytes:
        return np.zeros((0, 4), dtype=np.float32)
    arr = np.frombuffer(raw_bytes, dtype=np.float32)
    n = (arr.size // 4) * 4
    if n == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return arr[:n].reshape((-1, 4)).copy()


def semantic_lidar_bytes_to_array(raw_bytes):
    """Decode raw CARLA `sensor.lidar.ray_cast_semantic` bytes into float32 (N, 6):
    [x, y, z, cos_inc_angle, object_idx, object_tag]."""
    if not raw_bytes:
        return np.zeros((0, 6), dtype=np.float32)
    dt = np.dtype([
        ("x", np.float32), ("y", np.float32), ("z", np.float32),
        ("cos", np.float32), ("obj_idx", np.uint32), ("obj_tag", np.uint32),
    ])
    n = len(raw_bytes) // dt.itemsize
    if n == 0:
        return np.zeros((0, 6), dtype=np.float32)
    rec = np.frombuffer(raw_bytes[: n * dt.itemsize], dtype=dt)
    out = np.empty((n, 6), dtype=np.float32)
    out[:, 0] = rec["x"]
    out[:, 1] = rec["y"]
    out[:, 2] = rec["z"]
    out[:, 3] = rec["cos"]
    out[:, 4] = rec["obj_idx"].astype(np.float32)
    out[:, 5] = rec["obj_tag"].astype(np.float32)
    return out


def radar_bytes_to_array(raw_bytes):
    """Decode raw CARLA `sensor.other.radar` bytes into float32 (N, 4): [velocity, azimuth, altitude, depth]."""
    if not raw_bytes:
        return np.zeros((0, 4), dtype=np.float32)
    arr = np.frombuffer(raw_bytes, dtype=np.float32)
    n = (arr.size // 4) * 4
    if n == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return arr[:n].reshape((-1, 4)).copy()


# ----------------------------------------------------------------------------------------- ML-ready LiDAR encodings
def lidar_to_bev(points, range_m=50.0, size=128, z_min=-2.2, z_max=2.5):
    """Rasterize an (N, >=3) LiDAR point cloud in ego frame (+X forward, +Y right, +Z up) into a CNN-ready
    float32 (3, size, size) Bird's-Eye View tensor:
      channel 0: normalized max height in cell [0, 1]
      channel 1: log-scaled normalized point density [0, 1]
      channel 2: mean intensity in cell [0, 1]
    Row 0 = +range_m (forward), Col 0 = -range_m (left), Ego car sits at center."""
    bev = np.zeros((3, int(size), int(size)), dtype=np.float32)
    if points is None or len(points) == 0:
        return bev
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return bev
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    inten = np.clip(pts[:, 3], 0.0, 1.0) if pts.shape[1] >= 4 else np.ones_like(x)
    rng = float(range_m)
    mask = (np.abs(x) < rng) & (np.abs(y) < rng) & (z >= z_min) & (z <= z_max)
    if not np.any(mask):
        return bev
    x, y, z, inten = x[mask], y[mask], z[mask], inten[mask]
    # Forward (+X) maps to top of image (smaller row index); Right (+Y) maps to right (larger col index)
    res = (2.0 * rng) / float(size)
    rows = np.clip(((rng - x) / res).astype(np.int32), 0, size - 1)
    cols = np.clip(((y + rng) / res).astype(np.int32), 0, size - 1)
    z_norm = np.clip((z - z_min) / max(z_max - z_min, 1e-3), 0.0, 1.0)

    # Max height per cell
    height_grid = np.zeros((size, size), dtype=np.float32)
    np.maximum.at(height_grid, (rows, cols), z_norm)

    # Count & mean intensity per cell
    count_grid = np.zeros((size, size), dtype=np.float32)
    np.add.at(count_grid, (rows, cols), 1.0)
    inten_sum = np.zeros((size, size), dtype=np.float32)
    np.add.at(inten_sum, (rows, cols), inten)

    bev[0] = height_grid
    bev[1] = np.clip(np.log1p(count_grid) / math.log1p(16.0), 0.0, 1.0)
    nz = count_grid > 0.0
    bev[2, nz] = inten_sum[nz] / count_grid[nz]
    return bev


def lidar_to_range_sectors(points, n_sectors=36, max_range=60.0, z_min=-1.35, z_max=2.2, min_ego_radius=2.2):
    """Project a 3D LiDAR cloud into `n_sectors` uniform 360-degree horizontal angular bins around the ego vehicle.
    Sector 0 is centered straight ahead (azimuth 0 rad, +X); indices increase clockwise toward +Y (right),
    then behind (+-pi), then -Y (left).
    Returns float32 (n_sectors,) normalized to [0, 1] where 1.0 = clear up to `max_range`, 0.0 = contact.
    Ground points (z < z_min in sensor frame, where roof sensor is ~1.8 m above road) and self-hits (< min_ego_radius)
    are filtered out so only real obstacles/vehicles/barriers register."""
    n_sec = max(4, int(n_sectors))
    sectors = np.ones(n_sec, dtype=np.float32)
    if points is None or len(points) == 0:
        return sectors
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] < 3:
        return sectors
    x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
    dist = np.hypot(x, y)
    mask = (z >= float(z_min)) & (z <= float(z_max)) & (dist >= float(min_ego_radius)) & (dist <= float(max_range))
    if not np.any(mask):
        return sectors
    x, y, dist = x[mask], y[mask], dist[mask]
    # Azimuth in [0, 2*pi): 0 = straight ahead (+X), pi/2 = right (+Y), pi = rear (-X), 3pi/2 = left (-Y)
    ang = np.mod(np.arctan2(y, x) + math.pi / n_sec, 2.0 * math.pi)
    idx = np.clip((ang / (2.0 * math.pi) * n_sec).astype(np.int32), 0, n_sec - 1)
    norm_d = np.clip(dist / max(float(max_range), 1e-3), 0.0, 1.0)
    np.minimum.at(sectors, idx, norm_d)
    return sectors


# ----------------------------------------------------------------------------------------- Visual rendering (BEV + Mosaic)
def render_lidar_bev_image(points, size=360, range_m=45.0, z_min=-2.2, z_max=2.5):
    """Render a crisp dark-theme 360-deg LiDAR BEV radar BGR image (size x size x 3 uint8) for HUD / video recording."""
    import cv2
    s = int(size)
    img = np.full((s, s, 3), (18, 22, 28), dtype=np.uint8)
    cx, cy = s // 2, s // 2
    scale = (s * 0.5) / float(range_m)

    # Subtle range rings every 10 m and crosshairs
    for r_m in (10, 20, 30, 40):
        if r_m < range_m:
            rad_px = int(round(r_m * scale))
            cv2.circle(img, (cx, cy), rad_px, (42, 50, 62), 1, cv2.LINE_AA)
            cv2.putText(img, "%dm" % r_m, (cx + 3, max(12, cy - rad_px + 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (90, 105, 125), 1, cv2.LINE_AA)
    cv2.line(img, (cx, 0), (cx, s - 1), (34, 40, 50), 1)
    cv2.line(img, (0, cy), (s - 1, cy), (34, 40, 50), 1)

    if points is not None and len(points) > 0:
        pts = np.asarray(points, dtype=np.float32)
        if pts.ndim == 2 and pts.shape[1] >= 3:
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
            mask = (np.abs(x) < range_m) & (np.abs(y) < range_m) & (z >= z_min) & (z <= z_max)
            if np.any(mask):
                x, y, z = x[mask], y[mask], z[mask]
                px = np.clip((cx + y * scale).astype(np.int32), 0, s - 1)
                py = np.clip((cy - x * scale).astype(np.int32), 0, s - 1)
                # Color-code: ground points muted teal, obstacle-height points bright amber/red
                z_norm = np.clip((z - z_min) / max(z_max - z_min, 1e-3), 0.0, 1.0)
                obs = z >= -1.25
                b = np.where(obs, 40, 130).astype(np.uint8)
                g = np.where(obs, np.clip(255 - 120 * z_norm, 90, 255), 115).astype(np.uint8)
                r = np.where(obs, np.clip(120 + 135 * z_norm, 120, 255), 70).astype(np.uint8)
                img[py, px, 0] = b
                img[py, px, 1] = g
                img[py, px, 2] = r

    # Draw ego car footprint at center with forward arrow
    half_l = max(4, int(round(2.35 * scale)))
    half_w = max(2, int(round(0.95 * scale)))
    cv2.rectangle(img, (cx - half_w, cy - half_l), (cx + half_w, cy + half_l), (80, 220, 120), -1)
    cv2.arrowedLine(img, (cx, cy), (cx, max(4, cy - half_l - 10)), (255, 255, 255), 1, cv2.LINE_AA, tipLength=0.35)
    cv2.putText(img, "LIDAR 360 BEV", (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 235, 255), 1, cv2.LINE_AA)
    return img


def compose_sensor_mosaic(cam_frames_bgr, lidar_points=None, telemetry=None, tile_size=(320, 180)):
    """Compose all active camera views + optional LiDAR BEV panel + telemetry HUD bar into a single BGR image.
    `cam_frames_bgr`: ordered dict {cam_name: HxWx3 uint8 BGR}."""
    import cv2
    tw, th = int(tile_size[0]), int(tile_size[1])
    tiles = []

    for name, frame in (cam_frames_bgr or {}).items():
        if frame is None:
            tile = np.full((th, tw, 3), (24, 24, 28), dtype=np.uint8)
        else:
            tile = cv2.resize(frame, (tw, th), interpolation=cv2.INTER_AREA) if (frame.shape[1], frame.shape[0]) != (tw, th) else frame.copy()
        label = (SENSOR_CATALOG.get(name) or {}).get("label", name.upper())
        cv2.rectangle(tile, (0, 0), (min(tw, len(label) * 8 + 16), 22), (16, 18, 22), -1)
        cv2.putText(tile, label, (6, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (235, 245, 255), 1, cv2.LINE_AA)
        tiles.append(tile)

    if lidar_points is not None:
        bev_sq = render_lidar_bev_image(lidar_points, size=th, range_m=45.0)
        # Pad centered inside tw x th tile so grid aligns cleanly
        tile = np.full((th, tw, 3), (18, 22, 28), dtype=np.uint8)
        x0 = max(0, (tw - th) // 2)
        w_put = min(tw - x0, th)
        tile[:, x0 : x0 + w_put] = bev_sq[:, :w_put]
        tiles.append(tile)

    if not tiles:
        tiles.append(np.full((th, tw, 3), (24, 24, 28), dtype=np.uint8))

    n = len(tiles)
    if n == 1:
        cols, rows = 1, 1
    elif n == 2:
        cols, rows = 2, 1
    elif n <= 4:
        cols, rows = 2, 2
    elif n <= 6:
        cols, rows = 3, 2
    else:
        cols = 3
        rows = int(math.ceil(n / 3.0))

    grid = np.full((rows * th, cols * tw, 3), (14, 16, 20), dtype=np.uint8)
    for idx, tile in enumerate(tiles):
        r, c = divmod(idx, cols)
        grid[r * th : (r + 1) * th, c * tw : (c + 1) * tw] = tile

    # Bottom HUD telemetry bar
    hud_h = 32
    hud = np.full((hud_h, grid.shape[1], 3), (20, 24, 30), dtype=np.uint8)
    if isinstance(telemetry, dict):
        spd_kmh = float(telemetry.get("speed_ms", 0.0)) * 3.6
        steer = float(telemetry.get("steer", 0.0))
        thr = float(telemetry.get("throttle", 0.0))
        brk = float(telemetry.get("brake", 0.0))
        t_s = float(telemetry.get("t_sim", 0.0))
        mode = str(telemetry.get("control_mode", "auto")).upper()
        coll = bool(telemetry.get("collided", False))
        min_d = telemetry.get("min_obstacle_dist_m")
        d_txt = (" | MinDist %.1fm" % float(min_d)) if (min_d is not None and math.isfinite(float(min_d))) else ""
        status = "COLLISION!" if coll else "CLEAR"
        color = (70, 85, 255) if coll else (120, 235, 145)
        txt = "t=%.2fs | %s | %.1f km/h | Steer %+.2f Thr %.2f Brk %.2f%s | %s" % (
            t_s, mode, spd_kmh, steer, thr, brk, d_txt, status
        )
        cv2.putText(hud, txt, (10, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.48, color, 1, cv2.LINE_AA)
    return np.vstack([grid, hud])


# ----------------------------------------------------------------------------------------- Live CARLA Sensor Rig
class SmartCarSensorRig:
    """Manages a vehicle's attached CARLA smart-car sensor suite (cameras, 360-deg LiDAR, radar),
    synchronous per-tick readout, optional mosaic video export, and optional synchronized training dataset logging."""

    def __init__(self, world, bp_lib, actor, sensors, vtype="car", dt=0.05, image_size=(320, 180),
                 video_path=None, save_dir=None, lidar_bev_size=128, lidar_sectors=36, spawned_ids=None):
        import carla
        self.world = world
        self.actor = actor
        self.vtype = vtype
        self.dt = float(dt)
        self.image_size = parse_resolution(image_size)
        self.configs = resolve_sensor_spec(sensors)
        self.lidar_bev_size = int(lidar_bev_size)
        self.lidar_sectors = int(lidar_sectors)
        self.video_path = video_path
        self.save_dir = save_dir
        self.entries = []         # [{"name", "kind", "cfg", "mount", "sensor", "q", "last"}]
        self._video_writer = None
        self._frame_log = None
        self._frame_idx = 0
        self._closed = False

        bbox_ext = getattr(getattr(actor, "bounding_box", None), "extent", None)
        rigid_att = getattr(getattr(carla, "AttachmentType", None), "Rigid", None)
        rot_hz = round(1.0 / max(self.dt, 1e-3), 2)

        for cfg in self.configs:
            bp = bp_lib.find(cfg["blueprint"])
            kind = cfg["kind"]
            if kind == "camera":
                bp.set_attribute("image_size_x", str(self.image_size[0]))
                bp.set_attribute("image_size_y", str(self.image_size[1]))
                bp.set_attribute("fov", str(cfg.get("fov", 90.0)))
            elif kind in ("lidar", "semantic_lidar"):
                for attr, val in (
                    ("channels", str(cfg.get("channels", 32))),
                    ("range", str(cfg.get("range", 80.0))),
                    ("points_per_second", str(cfg.get("points_per_second", 200000))),
                    ("rotation_frequency", str(rot_hz)),
                    ("upper_fov", str(cfg.get("upper_fov", 10.0))),
                    ("lower_fov", str(cfg.get("lower_fov", -30.0))),
                    ("horizontal_fov", str(cfg.get("horizontal_fov", 360.0))),
                ):
                    if not hasattr(bp, "has_attribute") or bp.has_attribute(attr):
                        bp.set_attribute(attr, val)
                if kind == "lidar" and (not hasattr(bp, "has_attribute") or bp.has_attribute("dropoff_general_rate")):
                    bp.set_attribute("dropoff_general_rate", str(cfg.get("dropoff_general_rate", 0.0)))
            elif kind == "radar":
                for attr, val in (
                    ("horizontal_fov", str(cfg.get("horizontal_fov", 40.0))),
                    ("vertical_fov", str(cfg.get("vertical_fov", 15.0))),
                    ("range", str(cfg.get("range", 100.0))),
                    ("points_per_second", str(cfg.get("points_per_second", 2000))),
                ):
                    if not hasattr(bp, "has_attribute") or bp.has_attribute(attr):
                        bp.set_attribute(attr, val)

            mx, my, mz, mpitch, myaw, mroll = sensor_mount_transform(cfg, vtype=vtype, bbox_extent=bbox_ext)
            tf = carla.Transform(
                carla.Location(x=mx, y=my, z=mz),
                carla.Rotation(pitch=mpitch, yaw=myaw, roll=mroll),
            )
            if rigid_att is not None:
                sensor = world.try_spawn_actor(bp, tf, attach_to=actor, attachment_type=rigid_att)
            else:
                sensor = world.try_spawn_actor(bp, tf, attach_to=actor)
            if sensor is None:
                continue
            q = queue.Queue()
            sensor.listen(q.put)
            if spawned_ids is not None:
                spawned_ids.append(sensor.id)
            self.entries.append({
                "name": cfg["name"],
                "kind": kind,
                "cfg": cfg,
                "mount": (mx, my, mz, mpitch, myaw, mroll),
                "sensor": sensor,
                "q": q,
                "last": None,
            })

        if self.save_dir and self.entries:
            os.makedirs(self.save_dir, exist_ok=True)
            for e in self.entries:
                os.makedirs(os.path.join(self.save_dir, e["name"]), exist_ok=True)
            manifest = {
                "vehicle_type": self.vtype,
                "dt": self.dt,
                "image_size": list(self.image_size),
                "sensors": [
                    {
                        "name": e["name"],
                        "kind": e["kind"],
                        "blueprint": e["cfg"]["blueprint"],
                        "mount_xyz_pyr": list(e["mount"]),
                        **({k: e["cfg"][k] for k in ("fov",) if k in e["cfg"]}),
                        **({k: e["cfg"][k] for k in ("channels", "range", "points_per_second", "upper_fov", "lower_fov") if k in e["cfg"]}),
                    }
                    for e in self.entries
                ],
            }
            with open(os.path.join(self.save_dir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=1)
            self._frame_log = open(os.path.join(self.save_dir, "frames.jsonl"), "w", encoding="utf-8")

    @property
    def sensor_actors(self):
        return [e["sensor"] for e in self.entries]

    @property
    def active_names(self):
        return [e["name"] for e in self.entries]

    @property
    def has_cameras(self):
        return any(e["kind"] == "camera" for e in self.entries)

    @property
    def has_lidar(self):
        return any(e["kind"] in ("lidar", "semantic_lidar") for e in self.entries)

    def flush_queues(self, timeout=0.25):
        """Drain pending warmup frames from every attached sensor queue."""
        for e in self.entries:
            q = e["q"]
            try:
                q.get(timeout=timeout)
            except queue.Empty:
                pass
            while not q.empty():
                q.get_nowait()

    def read_tick(self, timeout=0.5, telemetry=None, record_frame=True):
        """Read the latest synchronized data from every sensor for the current tick.
        Returns dict with:
          - cameras: {name: HxWx3 uint8 RGB}
          - cameras_bgr: {name: HxWx3 uint8 BGR}
          - lidar: (N, 4) float32 [x, y, z, intensity] (if lidar active)
          - lidar_bev: (3, H, W) float32 (if lidar active)
          - lidar_sectors: (K,) float32 (if lidar active)
          - semantic_lidar: (N, 6) float32 (if semantic_lidar active)
          - radar_front: (N, 4) float32 (if radar active)
        """
        cams_rgb = {}
        cams_bgr = {}
        out = {"cameras": cams_rgb, "cameras_bgr": cams_bgr}

        for e in self.entries:
            q = e["q"]
            raw = None
            while not q.empty():
                raw = q.get_nowait()
            if raw is None:
                try:
                    raw = q.get(timeout=timeout)
                except queue.Empty:
                    raw = None
            if raw is not None:
                e["last"] = raw
            data = e["last"]
            kind = e["kind"]
            if kind == "camera":
                if data is not None:
                    w = int(getattr(data, "width", self.image_size[0]))
                    h = int(getattr(data, "height", self.image_size[1]))
                    bgr = bgra_to_bgr(data.raw_data, w, h)
                    cams_bgr[e["name"]] = bgr
                    cams_rgb[e["name"]] = bgr[:, :, ::-1].copy()
                else:
                    blank = np.zeros((self.image_size[1], self.image_size[0], 3), dtype=np.uint8)
                    cams_bgr[e["name"]] = blank
                    cams_rgb[e["name"]] = blank
            elif kind == "lidar":
                pts = lidar_bytes_to_points(getattr(data, "raw_data", b"")) if data is not None else np.zeros((0, 4), dtype=np.float32)
                out["lidar"] = pts
                out["lidar_bev"] = lidar_to_bev(pts, size=self.lidar_bev_size)
                out["lidar_sectors"] = lidar_to_range_sectors(pts, n_sectors=self.lidar_sectors)
            elif kind == "semantic_lidar":
                arr = semantic_lidar_bytes_to_array(getattr(data, "raw_data", b"")) if data is not None else np.zeros((0, 6), dtype=np.float32)
                out["semantic_lidar"] = arr
                if "lidar" not in out:
                    pts4 = np.hstack([arr[:, :3], np.ones((len(arr), 1), dtype=np.float32)]) if len(arr) else np.zeros((0, 4), dtype=np.float32)
                    out["lidar_bev"] = lidar_to_bev(pts4, size=self.lidar_bev_size)
                    out["lidar_sectors"] = lidar_to_range_sectors(pts4, n_sectors=self.lidar_sectors)
            elif kind == "radar":
                out[e["name"]] = radar_bytes_to_array(getattr(data, "raw_data", b"")) if data is not None else np.zeros((0, 4), dtype=np.float32)

        if record_frame and (self.video_path or self._frame_log is not None):
            self._record(out, telemetry=telemetry)
        return out

    def render_mosaic(self, obs, telemetry=None, tile_size=None):
        """Render the live multi-cam + LiDAR BEV mosaic BGR image from an observation returned by `read_tick`."""
        ts = tile_size or self.image_size
        lidar_pts = obs.get("lidar")
        if lidar_pts is None and "semantic_lidar" in obs:
            lidar_pts = obs["semantic_lidar"][:, :3]
        return compose_sensor_mosaic(obs.get("cameras_bgr") or {}, lidar_points=lidar_pts, telemetry=telemetry, tile_size=ts)

    def _record(self, obs, telemetry=None):
        self._frame_idx += 1
        if self.video_path:
            import cv2
            mosaic = self.render_mosaic(obs, telemetry=telemetry)
            if self._video_writer is None:
                os.makedirs(os.path.dirname(os.path.abspath(self.video_path)), exist_ok=True)
                self._video_writer = cv2.VideoWriter(
                    self.video_path,
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    round(1.0 / max(self.dt, 1e-3), 1),
                    (mosaic.shape[1], mosaic.shape[0]),
                )
            self._video_writer.write(mosaic)
        if self._frame_log is not None:
            import cv2
            idx = self._frame_idx
            files = {}
            for name, bgr in (obs.get("cameras_bgr") or {}).items():
                rel = os.path.join(name, "%06d.jpg" % idx)
                cv2.imwrite(os.path.join(self.save_dir, rel), bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                files[name] = rel.replace("\\", "/")
            if "lidar" in obs:
                rel = os.path.join("lidar", "%06d.npy" % idx)
                np.save(os.path.join(self.save_dir, rel), obs["lidar"].astype(np.float32))
                files["lidar"] = rel.replace("\\", "/")
            if "semantic_lidar" in obs:
                rel = os.path.join("semantic_lidar", "%06d.npy" % idx)
                np.save(os.path.join(self.save_dir, rel), obs["semantic_lidar"].astype(np.float32))
                files["semantic_lidar"] = rel.replace("\\", "/")
            if "radar_front" in obs:
                rel = os.path.join("radar_front", "%06d.npy" % idx)
                np.save(os.path.join(self.save_dir, rel), obs["radar_front"].astype(np.float32))
                files["radar_front"] = rel.replace("\\", "/")
            rec = {"frame": idx, "files": files}
            if isinstance(telemetry, dict):
                rec.update(telemetry)
            if "lidar_sectors" in obs:
                rec["lidar_sectors"] = [round(float(v), 4) for v in obs["lidar_sectors"]]
            self._frame_log.write(json.dumps(rec) + "\n")

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._video_writer is not None:
            try:
                self._video_writer.release()
            except Exception:
                pass
            self._video_writer = None
        if self._frame_log is not None:
            try:
                self._frame_log.close()
            except Exception:
                pass
            self._frame_log = None
        for e in self.entries:
            try:
                e["sensor"].stop()
            except Exception:
                pass
