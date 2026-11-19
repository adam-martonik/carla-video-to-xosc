#!/usr/bin/env python
"""
video_to_xosc.py — Physically Consistent Dashcam Video -> OpenSCENARIO 1.0 (.xosc) & OpenDRIVE (.xodr) Pipeline.

Maintains backward-compatible CLI flags while delegating to the modular `src/xosc_pipeline` package:
  - Vanishing-point horizon calibration & uncertainty-weighted monocular 3D localization
  - BoT-SORT tracking with Camera Motion Compensation + tracklet stitching
  - RANSAC / radial-FOE visual odometry
  - MAD outlier rejection + Rauch-Tung-Striebel (RTS) Kalman smoothing
  - Bicycle-model kinematic feasibility limits (speed, acceleration, jerk, yaw-rate)
  - Multi-timestamp 2D Frenet/OBB deconfliction
  - ASAM OpenDRIVE (.xodr) road network generation + offline/CARLA projection
  - Parameterized ASAM OpenSCENARIO 1.0 (.xosc) generation & XSD validation
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from xosc_pipeline.config import PipelineConfig, load_config  # noqa: E402
from xosc_pipeline.kinematics import smooth_and_resample_trajectories  # noqa: E402
from xosc_pipeline.road_network import project_frenet_offline, project_frenet_to_carla_map  # noqa: E402
from xosc_pipeline.tracking import extract_ego_and_tracks  # noqa: E402
from xosc_pipeline.xosc_writer import write_openscenario_xosc  # noqa: E402


def download_youtube_clip(url: str, output_path: Path) -> Path:
    """Download a YouTube dashcam video clip using yt-dlp (personal/research use only)."""
    print("[Stage A] Note: Respect YouTube Terms of Service & copyright (use only authorized/research clips).")
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
        sys.executable,
        "-m",
        "yt_dlp",
        "-f",
        "bestvideo[height<=720][ext=mp4]/bestvideo[ext=mp4]/best[ext=mp4]/best",
        "-o",
        str(output_path),
        "--no-warnings",
        url,
    ]
    subprocess.run(cmd, check=True)

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


def run_pipeline(
    video_path: Path,
    output_xosc: Path,
    output_json: Path | None = None,
    output_xodr: Path | None = None,
    xsd_path: Path | None = None,
    town: str = "Town04_Opt",
    offline: bool = False,
    use_xodr_logic_file: bool = False,
    cfg: PipelineConfig | None = None,
    sample_dt: float | None = None,
    save_annotated: bool = True,
) -> dict:
    """Execute full dashcam video -> OpenSCENARIO 1.0 (+ OpenDRIVE) pipeline."""
    config = cfg if cfg is not None else load_config()
    ann_out = (video_path.parent / f"{video_path.stem}_tracked_annotated.mp4") if save_annotated else None

    extracted = extract_ego_and_tracks(video_path, config, annotated_video_out=ann_out)
    resampled = smooth_and_resample_trajectories(extracted, config, sample_dt=sample_dt)

    xodr_target = output_xodr if output_xodr is not None else output_xosc.with_suffix(".xodr")
    if offline:
        projected = project_frenet_offline(resampled, config, output_xodr=xodr_target, town_label=town)
    else:
        projected = project_frenet_to_carla_map(resampled, config, town=town, output_xodr=xodr_target)

    if output_json is not None:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(projected, indent=2), encoding="utf-8")

    write_openscenario_xosc(
        projected,
        output_xosc,
        config,
        xsd_path=xsd_path,
        use_xodr_logic_file=use_xodr_logic_file,
    )
    return projected


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert dashcam video to CARLA OpenSCENARIO 1.0 (.xosc) + OpenDRIVE (.xodr)")
    parser.add_argument("--video", type=str, default="demo/dashcam_input.mp4", help="Input dashcam video path")
    parser.add_argument("--youtube-url", type=str, default="", help="Optional YouTube URL to download with yt-dlp")
    parser.add_argument("--town", type=str, default="Town04_Opt", help="Target CARLA map for trajectory projection")
    parser.add_argument("--output-xosc", type=str, default="demo/dashcam_reconstructed.xosc", help="Output .xosc path")
    parser.add_argument("--output-json", type=str, default="demo/reconstructed_trajectories.json", help="Output trajectory JSON")
    parser.add_argument("--output-xodr", type=str, default="", help="Optional output .xodr path (defaults to <output-xosc>.xodr)")
    parser.add_argument("--xsd", type=str, default="scenario_runner/srunner/openscenario/OpenSCENARIO.xsd", help="XSD schema path")
    parser.add_argument("--config", type=str, default="", help="Optional YAML configuration path")
    parser.add_argument("--sample-dt", type=float, default=None, help="Override trajectory sampling step in seconds")
    parser.add_argument("--offline", action="store_true", help="Force offline OpenDRIVE projection without connecting to CARLA")
    parser.add_argument("--use-xodr-logic", action="store_true", help="Reference generated .xodr in <LogicFile> instead of CARLA town")
    parser.add_argument("--no-annotated-video", action="store_true", help="Skip writing annotated tracking MP4")
    args = parser.parse_args()

    video_path = Path(args.video)
    if args.youtube_url:
        video_path.parent.mkdir(parents=True, exist_ok=True)
        download_youtube_clip(args.youtube_url, video_path)

    cfg = load_config(args.config if args.config else None)
    xodr_path = Path(args.output_xodr) if args.output_xodr else Path(args.output_xosc).with_suffix(".xodr")

    run_pipeline(
        video_path=video_path,
        output_xosc=Path(args.output_xosc),
        output_json=Path(args.output_json) if args.output_json else None,
        output_xodr=xodr_path,
        xsd_path=Path(args.xsd) if args.xsd else None,
        town=args.town,
        offline=args.offline,
        use_xodr_logic_file=args.use_xodr_logic,
        cfg=cfg,
        sample_dt=args.sample_dt,
        save_annotated=not args.no_annotated_video,
    )


if __name__ == "__main__":
    main()
