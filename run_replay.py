#!/usr/bin/env python
"""
run_replay.py — Single-command wrapper that sets PYTHONPATH, starts record_carla_replay.py
in the background (which also moves the CARLA window spectator camera to follow 'hero'),
and executes scenario_runner.py on the given .xosc file.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Replay an OpenSCENARIO (.xosc) file in CARLA 1.0")
    parser.add_argument(
        "xosc",
        nargs="?",
        default="demo/dashcam_reconstructed.xosc",
        help="Path to the .xosc file to replay (default: demo/dashcam_reconstructed.xosc)",
    )
    args = parser.parse_args()

    work_dir = Path(__file__).resolve().parent
    carla_root = work_dir / "CARLA_1.0" / "Carla-1.0-Win64-Shipping"
    sr_root = work_dir / "scenario_runner"
    xosc_path = Path(args.xosc).resolve()

    if not xosc_path.exists():
        print(f"ERROR: OpenSCENARIO file not found: {xosc_path}")
        sys.exit(1)

    env = os.environ.copy()
    env["CARLA_ROOT"] = str(carla_root)
    env["SCENARIO_RUNNER_ROOT"] = str(sr_root)
    py_path = f"{carla_root / 'PythonAPI' / 'carla'};{sr_root}"
    if env.get("PYTHONPATH"):
        py_path = f"{py_path};{env['PYTHONPATH']}"
    env["PYTHONPATH"] = py_path

    print(f"[run_replay] Starting recorder & spectator follow-cam...")
    rec_proc = subprocess.Popen(
        [sys.executable, str(work_dir / "record_carla_replay.py")],
        cwd=str(work_dir),
        env=env,
    )

    try:
        print(f"[run_replay] Launching ScenarioRunner for: {xosc_path}")
        cmd = [
            sys.executable,
            str(sr_root / "scenario_runner.py"),
            "--openscenario",
            str(xosc_path),
            "--output",
        ]
        subprocess.run(cmd, cwd=str(work_dir), env=env, check=False)
    finally:
        try:
            rec_proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            rec_proc.terminate()


if __name__ == "__main__":
    main()
