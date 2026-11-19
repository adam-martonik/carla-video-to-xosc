"""Unit and integration tests for OpenDRIVE (.xodr) and OpenSCENARIO 1.0 (.xosc) generation."""

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from xosc_pipeline.config import load_config
from xosc_pipeline.kinematics import smooth_and_resample_trajectories
from xosc_pipeline.metrics import evaluate_xosc_file
from xosc_pipeline.road_network import project_frenet_offline
from xosc_pipeline.xosc_writer import write_openscenario_xosc


def _make_synthetic_extracted() -> dict:
    fps = 30.0
    n = 150  # 5.0 seconds
    times = np.arange(n) / fps
    ego = [
        {"frame": i, "time": float(t), "s": float(15.0 * t), "d": 0.0, "yaw": float(0.01 * t), "speed": 15.0}
        for i, t in enumerate(times)
    ]
    truck_obs = [
        {
            "frame": i,
            "time": float(t),
            "class_id": 7,  # truck
            "conf": 0.85,
            "x_cam": 3.5,
            "z_cam": 22.0 - 0.5 * t,
            "sigma_z": 1.5,
        }
        for i, t in enumerate(times)
    ]
    car_obs = [
        {
            "frame": i,
            "time": float(t),
            "class_id": 2,  # car
            "conf": 0.90,
            "x_cam": -3.5,
            "z_cam": 15.0 + 1.0 * t,
            "sigma_z": 1.2,
        }
        for i, t in enumerate(times)
    ]
    return {
        "fps": fps,
        "width": 1280,
        "height": 720,
        "duration": 5.0,
        "is_night": False,
        "camera_cut_time": None,
        "has_oncoming_traffic": False,
        "collision_event": None,
        "ego": ego,
        "tracks": [(10, truck_obs), (20, car_obs)],
    }


def test_end_to_end_xosc_and_xodr_generation_validates_against_xsd(tmp_path: Path):
    cfg = load_config()
    extracted = _make_synthetic_extracted()
    resampled = smooth_and_resample_trajectories(extracted, cfg)

    xodr_path = tmp_path / "test_road.xodr"
    xosc_path = tmp_path / "test_scenario.xosc"
    xsd_path = ROOT / "scenario_runner" / "srunner" / "openscenario" / "OpenSCENARIO.xsd"

    projected = project_frenet_offline(resampled, cfg, output_xodr=xodr_path)
    assert xodr_path.exists()
    xodr_root = ET.parse(xodr_path).getroot()
    assert xodr_root.tag == "OpenDRIVE"
    assert xodr_root.find(".//road/lanes/laneSection/right") is not None

    write_openscenario_xosc(projected, xosc_path, cfg, xsd_path=xsd_path if xsd_path.exists() else None)
    assert xosc_path.exists()

    metrics = evaluate_xosc_file(xosc_path, xsd_path=xsd_path if xsd_path.exists() else None)
    if xsd_path.exists():
        assert metrics["xsd_valid"] is True
    assert metrics["param_count"] >= 5
    assert metrics["categories"]["npc_track_10"] == "truck"
    assert metrics["categories"]["npc_track_20"] == "car"
    assert metrics["max_perf_acc_xml"] <= 6.0
    assert metrics["obb_collision_steps"] == 0
    assert metrics["worst_max_abs_acc"] <= 5.0
    assert metrics["worst_max_abs_jerk"] <= 6.0
