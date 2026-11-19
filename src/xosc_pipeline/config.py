"""Typed configuration loader for the video-to-OpenSCENARIO pipeline."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import yaml
from pydantic import BaseModel, ConfigDict, Field


DEFAULT_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config" / "default.yaml"

VEHICLE_CLASS_IDS = frozenset({1, 2, 3, 5, 7})
VRU_ANIMAL_CLASS_IDS = frozenset({0, 15, 16, 17, 18, 19, 21, 22, 23})
ALL_TARGET_CLASS_IDS = sorted(VEHICLE_CLASS_IDS | VRU_ANIMAL_CLASS_IDS)

CARLA_COLOR_PALETTE = (
    "255,40,40",
    "40,180,255",
    "255,200,40",
    "60,220,100",
    "200,80,255",
)


class CameraConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    hfov_deg: float = 70.0
    cam_height_m: float = 1.35
    default_horizon_ratio: float = 0.52
    estimate_horizon: bool = True
    horizon_min_ratio: float = 0.38
    horizon_max_ratio: float = 0.62
    min_depth_m: float = 2.5
    max_depth_m: float = 90.0


class TrackingConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    model_candidates: List[str] = Field(default_factory=lambda: ["yolo11m.pt", "yolo11n.pt"])
    tracker: str = "botsort.yaml"
    fallback_tracker: str = "bytetrack.yaml"
    conf_threshold: float = 0.18
    vehicle_conf_threshold: float = 0.28
    max_actors: int = 4
    min_track_frames: int = 25
    min_track_duration_s: float = 1.0
    stitch_max_gap_s: float = 1.5
    stitch_max_dist_m: float = 5.0
    base_ego_speed_mps: float = 18.0
    night_sky_threshold: float = 65.0
    scene_cut_min_time_s: float = 10.0
    scene_cut_diff_threshold: float = 17.5


class KinematicsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    sample_dt_s: float = 0.25
    outlier_mad_window: int = 11
    outlier_mad_thresh: float = 3.0
    rts_process_noise_pos: float = 0.4
    rts_meas_noise_z: float = 1.8
    rts_meas_noise_x: float = 0.45
    rts_ego_meas_noise_v: float = 1.5
    rts_ego_process_noise_a: float = 1.2
    min_safety_margin_long_m: float = 2.0
    min_safety_margin_lat_m: float = 0.45
    deconflict_iterations: int = 8


class RoadConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    lane_width_m: float = 3.5
    shoulder_width_m: float = 2.2
    default_lanes_right: int = 2
    default_lanes_left: int = 1
    max_curvature_inv_m: float = 0.015
    generate_xodr: bool = True


class ScenarioConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    rev_major: int = 1
    rev_minor: int = 0
    author: str = "video_to_xosc_pipeline_v2"
    description: str = "CARLA:RealisticDashcamReconstructedScenario"
    speed_action_threshold_mps: float = 0.5
    speed_action_shape: str = "linear"
    waypoint_reached_threshold_m: float = 2.0
    default_friction: float = 1.0


class ClassSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    entity_kind: str  # "vehicle" | "pedestrian"
    category: str
    length: float
    width: float
    height: float
    mass: float = 1500.0
    wheelbase: float = 2.80
    track_width: float = 1.60
    wheel_diameter: float = 0.66
    max_steering: float = 0.55
    max_speed: float = 40.0
    max_accel: float = 4.5
    max_decel: float = 9.5
    max_jerk: float = 8.0
    blueprints: List[str] = Field(default_factory=lambda: ["vehicle.tesla.model3"])


class PipelineConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    camera: CameraConfig = Field(default_factory=CameraConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    kinematics: KinematicsConfig = Field(default_factory=KinematicsConfig)
    road: RoadConfig = Field(default_factory=RoadConfig)
    scenario: ScenarioConfig = Field(default_factory=ScenarioConfig)
    classes: Dict[int, ClassSpec] = Field(default_factory=dict)

    def get_class_spec(self, class_id: int) -> ClassSpec:
        if class_id in self.classes:
            return self.classes[class_id]
        if class_id in VRU_ANIMAL_CLASS_IDS and 16 in self.classes:
            return self.classes[16]
        if 2 in self.classes:
            return self.classes[2]
        return ClassSpec(
            name="car",
            entity_kind="vehicle",
            category="car",
            length=4.55,
            width=1.88,
            height=1.50,
        )


def load_config(path: Path | str | None = None) -> PipelineConfig:
    """Load pipeline configuration from YAML file (falls back to config/default.yaml)."""
    cfg_path = Path(path) if path is not None else DEFAULT_CONFIG_PATH
    if not cfg_path.exists():
        return PipelineConfig()
    raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    if "classes" in raw and isinstance(raw["classes"], dict):
        raw["classes"] = {int(k): v for k, v in raw["classes"].items()}
    return PipelineConfig.model_validate(raw)
