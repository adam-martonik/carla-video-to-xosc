#!/usr/bin/env bash
# ==============================================================================
# setup.sh — Reproducible Setup Script for CARLA 1.0, ScenarioRunner 1.0,
#            and the Monocular Dashcam Video -> OpenSCENARIO (.xosc) Pipeline
# ==============================================================================
# Verified on:
#   OS:     Windows 11 Pro 64-bit (10.0.26300)
#   GPU:    AMD Radeon RX 9060 XT (16 GB VRAM, Driver 32.0.32015.7029)
#   Python: Python 3.14.10
#   CARLA:  1.0 (Unreal Engine 5.8, git hash 3ea166118d2c3c094b13b15c5b7a66a9a279998d)
#   SR:     ScenarioRunner v1.0 (commit 23c4db61652a2880f09a048b273011ad60628712)
# ==============================================================================

set -euo pipefail

WORKDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$WORKDIR"

echo "=== [1/6] Downloading & Extracting CARLA 1.0 (Windows 11 Shipping Package) ==="
if [ ! -f "$WORKDIR/CARLA_1.0/Carla-1.0-Win64-Shipping/CarlaUnreal.exe" ]; then
    curl.exe -L -C - -o "$WORKDIR/CARLA_1.0.zip" "https://downloads.carlasim.com/Windows/CARLA_1.0.zip"
    mkdir -p "$WORKDIR/CARLA_1.0"
    tar.exe -xf "$WORKDIR/CARLA_1.0.zip" -C "$WORKDIR/CARLA_1.0"
fi

CARLA_ROOT="$WORKDIR/CARLA_1.0/Carla-1.0-Win64-Shipping"

echo "=== [2/6] Downloading & Installing CARLA 1.0 AdditionalMaps Pack (Town01-Town07, Town11, Town13) ==="
if [ ! -d "$CARLA_ROOT/CarlaUnreal/Packs/AdditionalMaps" ]; then
    curl.exe -L -C - -o "$WORKDIR/AdditionalMaps_1.0.tar.gz" "https://downloads.carlasim.com/Windows/AdditionalMaps_1.0.tar.gz"
    python "$CARLA_ROOT/Tools/carla_pack.py" install "$WORKDIR/AdditionalMaps_1.0.tar.gz" --server "$CARLA_ROOT"
fi

echo "=== [3/6] Creating Python Virtual Environment & Installing CARLA 1.0 Wheel ==="
if [ ! -d "$WORKDIR/.venv" ]; then
    python -m venv "$WORKDIR/.venv"
fi

if [ -f "$WORKDIR/.venv/Scripts/python.exe" ]; then
    PY="$WORKDIR/.venv/Scripts/python.exe"
    PIP="$WORKDIR/.venv/Scripts/pip.exe"
else
    PY="$WORKDIR/.venv/bin/python"
    PIP="$WORKDIR/.venv/bin/pip"
fi

"$PY" -m pip install --upgrade pip

# Install matching CARLA 1.0 wheel from PythonAPI/carla/dist/
PY_TAG=$("$PY" -c "import sys; print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
WHEEL_PATH=$(ls "$CARLA_ROOT/PythonAPI/carla/dist/"carla-1.0-"${PY_TAG}"-*.whl | head -n 1)
echo "Installing CARLA wheel: $WHEEL_PATH"
"$PIP" install "$WHEEL_PATH"

echo "=== [4/6] Cloning CARLA ScenarioRunner v1.0 & Installing Dependencies ==="
if [ ! -d "$WORKDIR/scenario_runner" ]; then
    git clone --branch v1.0 https://github.com/carla-simulator/scenario_runner.git "$WORKDIR/scenario_runner"
fi

"$PIP" install -r "$WORKDIR/scenario_runner/requirements.txt" \
    pygame-ce==2.5.8 \
    scenariogeneration \
    yt-dlp \
    ultralytics \
    scipy \
    transformers

echo "=== [5/6] Applying Verified Python 3.13+/py_trees 2.6.0 Compatibility Fixes ==="
"$PY" - << 'EOF'
import pathlib, re, sys

workdir = pathlib.Path.cwd()

# Fix 1: xmlschema 1.0.18 on Python 3.13/3.14 Windows (os.path.isabs('/c:/...') change)
for res_py in workdir.glob(".venv/**/site-packages/xmlschema/resources.py"):
    txt = res_py.read_text(encoding="utf-8")
    if "os.path.isabs(url_parts.path.lstrip('/'))" not in txt:
        txt = txt.replace(
            "elif os.path.isabs(url_parts.path):",
            "elif os.path.isabs(url_parts.path) or os.path.isabs(url_parts.path.lstrip('/')):"
        )
    if "re.sub(r'[a-zA-Z]:/([a-zA-Z]:/)', r'\\1', x)" not in txt:
        txt = txt.replace(
            "while x.startswith('file:////'):\n            x = x.replace('file:////', 'file:///', 1)",
            "while x.startswith('file:////'):\n            x = x.replace('file:////', 'file:///', 1)\n"
            "        x = re.sub(r'[a-zA-Z]:/([a-zA-Z]:/)', r'\\1', x)\n"
            "        x = re.sub(r'^file://([a-zA-Z]:/)', r'file:///\\1', x)"
        )
    res_py.write_text(txt, encoding="utf-8")
    print(f"Patched {res_py}")

# Fix 2: ScenarioRunner v1.0 OpenSCENARIO parser & py_trees 2.6.0 blackboard calls
osc_parser = workdir / "scenario_runner/srunner/tools/openscenario_parser.py"
if osc_parser.exists():
    txt = osc_parser.read_text(encoding="utf-8")
    txt = txt.replace(
        "from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest, OffRoadTest, EndofRoadTest",
        "from srunner.scenariomanager.scenarioatomics.atomic_criteria import (\n"
        "    CollisionTest, OffRoadTest, EndofRoadTest, RunningStopTest, RunningRedLightTest,\n"
        "    WrongLaneTest, OnSidewalkTest, KeepLaneTest, DrivenDistanceTest, MaxVelocityTest,\n"
        "    AverageVelocityTest, ActorBlockedTest, ReachedRegionTest, OutsideRouteLanesTest,\n"
        "    InRadiusRegionTest, InRouteTest, RouteCompletionTest,\n)"
    )
    txt = txt.replace(
        "    blackboard = py_trees.blackboard.Blackboard()\n    # check if the variable_name already exists in the blackboard\n    if blackboard.get(variable_name) is not None:",
        "    if get_blackboard(variable_name) is not None:"
    )
    if "from srunner.tools.blackboard import get_blackboard" not in txt:
        txt = txt.replace(
            "from srunner.tools.py_trees_port import oneshot_behavior",
            "from srunner.tools.py_trees_port import oneshot_behavior\nfrom srunner.tools.blackboard import get_blackboard"
        )
    osc_parser.write_text(txt, encoding="utf-8")

open_scen = workdir / "scenario_runner/srunner/scenarios/open_scenario.py"
if open_scen.exists():
    txt = open_scen.read_text(encoding="utf-8")
    txt = txt.replace(
        "from srunner.tools.blackboard import get_blackboard_keys",
        "from srunner.tools.blackboard import get_blackboard_keys, set_blackboard"
    )
    open_scen.write_text(txt, encoding="utf-8")

trig_cond = workdir / "scenario_runner/srunner/scenariomanager/scenarioatomics/atomic_trigger_conditions.py"
if trig_cond.exists():
    txt = trig_cond.read_text(encoding="utf-8")
    txt = txt.replace(
        "element_start_time = self._blackboard.get(blackboard_variable_name)",
        "element_start_time = get_blackboard(blackboard_variable_name)"
    )
    trig_cond.write_text(txt, encoding="utf-8")
EOF

echo "=== [6/6] Setup Complete! ==="
echo "To start CARLA 1.0 server (offscreen, low quality):"
echo "  \"$CARLA_ROOT/CarlaUnreal/Binaries/Win64/CarlaUnreal-Win64-Shipping.exe\" CarlaUnreal -RenderOffScreen -quality-level=Low &"
echo "To convert a dashcam video to OpenSCENARIO (.xosc):"
echo "  \"$PY\" video_to_xosc.py --video demo/dashcam_input.mp4 --town Town04_Opt --output-xosc demo/dashcam_reconstructed.xosc"
echo "To replay the generated OpenSCENARIO file in CARLA:"
echo "  export PYTHONPATH=\"$CARLA_ROOT/PythonAPI/carla:$WORKDIR/scenario_runner\""
echo "  \"$PY\" scenario_runner/scenario_runner.py --openscenario demo/dashcam_reconstructed.xosc --output"
