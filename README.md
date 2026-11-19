# CARLA 1.0 + ScenarioRunner 1.0 & Monocular Dashcam Video-to-OpenSCENARIO (`.xosc`) Pipeline

This repository contains the complete setup, research survey, and working open-source pipeline for converting a monocular dashcam video clip into an ASAM OpenSCENARIO 1.0 (`.xosc`) scenario and replaying it in **CARLA 1.0 (Unreal Engine 5.8)** via **ScenarioRunner 1.0**.

---

## 1. Environment Check (Step 0)

| Component | Detected Value | CARLA 1.0 Requirement | Status |
|---|---|---|---|
| **OS** | Microsoft Windows 11 Pro 64-bit (`10.0.26300`) | Windows 11 64-bit or Ubuntu 22.04/24.04 | **PASS** |
| **System RAM** | 31.76 GB total (~19.3 GB free) | 16–32 GB recommended | **PASS** |
| **GPU** | AMD Radeon RX 9060 XT (`0x7590`) | Dedicated GPU with 6–8+ GB VRAM | **PASS** |
| **GPU Driver** | `32.0.32015.7029` | Modern DX12 / Vulkan driver | **PASS** |
| **VRAM** | **15.92 GB** (`17,095,983,104` bytes via registry `HardwareInformation.qwMemorySize`; note WMI `Win32_VideoController.AdapterRAM` is a 32-bit uint capped at 4 GB) | 6–8 GB min (12–16 GB recommended for UE5) | **PASS** |
| **Disk Space** | 363.84 GB free on `C:` (prior to install) | ~20–30 GB free | **PASS** |
| **Python** | Python `3.14.10` (`C:\Users\adamk\AppData\Local\Programs\Python\Python314\python.exe`) | Python 3.10–3.15 (CARLA 1.0 ships cp310–cp315 wheels) | **PASS** |

> **Note on AMD GPU & Off-Screen Rendering:** Because this machine uses an AMD Radeon RX 9060 XT (no CUDA/NVIDIA DLSS hardware), the server is run with `-RenderOffScreen -quality-level=Low`. On CARLA 1.0 (UE 5.8), `-quality-level=Low` retains full view distance and simulation geometry while disabling hardware ray tracing, booting to RPC readiness in ~12 seconds. (Avoid `-nullrhi` when spawning RGB cameras, as UE 5.8 requires an active RHI render target for camera sensor readback.)

---

## 2. Installed Components & Paths (Step 1)

* **CARLA Simulator**: **CARLA `1.0`** (Latest official stable release, published 2026-11-17, Unreal Engine 5.8, commit `3ea166118d2c3c094b13b15c5b7a66a9a279998d`)
  * Path: `C:\Users\adamk\.gemini\antigravity\scratch\carla-video-to-xosc\CARLA_1.0\Carla-1.0-Win64-Shipping`
  * Additional Maps Pack: `AdditionalMaps_1.0.tar.gz` installed via `Tools\carla_pack.py` into `CarlaUnreal\Packs\AdditionalMaps` (adds `Town01_Opt`–`Town07_Opt`, `Town11`, `Town13`, `Town_C`, `Eixample`, `Mine_01` alongside base maps `Town10HD_Opt`, `Town12`, `Town15`, `EmptyMap`).
* **Python Virtual Environment**: `C:\Users\adamk\.gemini\antigravity\scratch\carla-video-to-xosc\.venv` (Python `3.14.10`)
  * CARLA Python API Wheel: `carla-1.0-cp314-cp314-win_amd64.whl` (from `PythonAPI\carla\dist\`)
  * Core Packages: `carla==1.0`, `py-trees==2.6.0`, `xmlschema==1.0.18`, `antlr4-python3-runtime==4.9.3`, `pygame-ce==2.5.8`, `scenariogeneration==0.15.1`, `yt-dlp==2026.10.26`, `ultralytics==8.4.215`, `torch==2.15.0`, `opencv-python==5.1.0.95`, `scipy==1.18.1`, `transformers==5.23.0`, `lap==0.5.14`.
* **CARLA ScenarioRunner**: **ScenarioRunner `v1.0`** (`ue58-dev` branch, commit `23c4db61652a2880f09a048b273011ad60628712`, published 2026-11-06)
  * Path: `C:\Users\adamk\.gemini\antigravity\scratch\carla-video-to-xosc\scenario_runner`
  * OpenSCENARIO Support:
    * **OpenSCENARIO 1.0 (`.xosc`)** via `--openscenario` (XML Storyboard/Act/Maneuver/Event/Action format, validated against `srunner/openscenario/OpenSCENARIO.xsd` with `xmlschema==1.0.18`).
    * **OpenSCENARIO 2.0 (`.osc`)** via `--openscenario2` (ANTLR 4.9.3 DSL parser under `srunner/osc2/`).

### Verified Fixes Applied for Windows + Python 3.14 + ScenarioRunner 1.0

During live verification, two upstream bugs were encountered and fixed (automated in `setup.sh`):

1. **`xmlschema==1.0.18` on Python 3.13/3.14 Windows (`URLError: [WinError 123] ... 'C:\\c:\\Users\\...'`)**:
   * In Python 3.13+, `os.path.isabs('/c:/Users/...')` returns `False` on Windows, causing `xmlschema.resources.normalize_url()` to call `os.path.abspath('/c:/Users/...')`, which prepends `C:\` a second time (`C:\c:\Users\...`).
   * **Fix**: Patched `.venv/Lib/site-packages/xmlschema/resources.py` (`normalize_url` and `filter_url`) to recognize `/c:/...` as absolute and strip duplicate drive prefixes.
2. **ScenarioRunner `v1.0` OpenSCENARIO 1.0 (`py_trees==2.6.0` migration omissions)**:
   * `srunner/tools/openscenario_parser.py`: Restored `RunningStopTest`, `RunningRedLightTest`, `WrongLaneTest`, `OnSidewalkTest`, `KeepLaneTest`, `DrivenDistanceTest`, etc. imports accessed dynamically via `globals()` (line 43), and replaced `blackboard.get(variable_name)` with `get_blackboard(variable_name)` (line 73).
   * `srunner/scenarios/open_scenario.py`: Replaced legacy `py_trees 0.8` `self.blackboard.set(name=..., value=..., overwrite=True)` calls with `set_blackboard(key, val)`.
   * `srunner/scenariomanager/scenarioatomics/atomic_trigger_conditions.py`: Replaced `self._blackboard.get(blackboard_variable_name)` in `OSCStartEndCondition.update()` with `get_blackboard(blackboard_variable_name)`.

---

## 3. Survey of Dashcam/Video-to-OpenSCENARIO & CARLA Tools (Step 2)

| Rank | Tool / Paper Name | Repo / Paper Link | Last Commit / Date | License | Code Released? | Output Format | CARLA Support | Input Requirements | Runs w/o Proprietary Data? |
|---|---|---|---|---|---|---|---|---|---|
| **1** | **From Dashcam Videos to Driving Simulations: Stress Testing Automated Vehicles against Rare Events** (Miao et al., ASU / Toyota / UIUC) | [arXiv:2411.16027v2](https://arxiv.org/abs/2411.16027v2) | Nov 2024 (Paper) | N/A (No repo) | **No** (Paper only) | **Scenic** (`.scenic`) scripts via VLM prompting (not `.xosc`) | CARLA 0.9.x (via Scenic) | Plain monocular dashcam crash video | Requires proprietary VLM API (GPT-4o) |
| **2** | **DeepCrashTest: Turning Dashcam Videos into Virtual Crash Tests for Automated Driving Systems** (Bashetty et al., ASU) | [arXiv:2003.11766](https://arxiv.org/abs/2003.11766) / [C-Aniruddh/deepcrashtest_v2](https://github.com/C-Aniruddh/deepcrashtest_v2) | Oct 2022 (`deepcrashtest_v2` placeholder) | Unspecified | **No** (`README.md` only: *"Code coming soon"*) | Custom trajectory waypoints for **Webots / Sim-ATAV** (not `.xosc`) | None (Webots) | Uncalibrated monocular dashcam video | Yes (if re-implemented) |
| **3** | **Dash2Sim: Closed-Loop Driving Simulation from in-the-wild Dashcam Videos** (Ghosh et al., CMU / NEC) | [arXiv:2606.07366](https://arxiv.org/abs/2606.07366) | Jun 2026 (Paper) | N/A (No repo) | **No** (Dataset `ROADWork4D` & paper only) | **nuPlan** 4D driving log + OpenStreetMap alignment (not `.xosc`) | None (nuPlan) | Monocular in-the-wild dashcam video | Yes (uses OSM) |
| **4** | **scenariogeneration (`pyoscx`)** + **Scenic** | [pyoscx/scenariogeneration](https://github.com/pyoscx/scenariogeneration) / [Scenic-Foundation/Scenic](https://github.com/Scenic-Foundation/Scenic) | Nov 2026 (Active) | MPL-2.0 / BSD-3-Clause | **Yes** (PyPI `scenariogeneration`, `scenic`) | **OpenSCENARIO 1.x (`.xosc`)** & OpenDRIVE (`.xodr`) / `.scenic` | CARLA 0.9.x–1.0 (via ScenarioRunner) | Extracted 3D trajectories / programmatic definitions (needs CV front-end for video) | **Yes** |
| **5** | **CARLA 1.0 NVIDIA NuRec Integration** (`PythonAPI/examples/nvidia/nurec`) | Bundled in [CARLA 1.0](https://github.com/carla-simulator/carla/releases/tag/1.0) | Nov 2026 | MIT (client) / NVIDIA NGC | **Yes** (in CARLA 1.0 package) | NuRec `.usdz` neural scene + OpenDRIVE `.xodr` + actor tracks | CARLA 1.0 (UE 5.8) | Multi-sensor calibrated fleet logs reconstructed into `.usdz` (not plain YouTube video); requires NVIDIA GPU + Docker NGC container | Yes (public HF sample dataset `nvidia/PhysicalAI-Autonomous-Vehicles-NuRec`) |
| **6** | **ScenarioNet** (Li et al., MetaDrive / UCLA) | [metadriverse/scenarionet](https://github.com/metadriverse/scenarionet) / [arXiv:2306.12241](https://arxiv.org/abs/2306.12241) | Mar 2026 | Apache-2.0 | **Yes** | MetaDrive `.pkl` scenario description | None (MetaDrive) | Multi-sensor AV logs with 3D boxes & HD maps (Waymo, nuScenes, nuPlan) | Yes |
| **7** | **AC3R: Automatically Reconstructing Car Crashes from Police Reports** (Huynh et al.) | [TriHuynh00/AC3R-Demo](https://github.com/TriHuynh00/AC3R-Demo) | Apr 2023 | Unspecified | **Yes** (Java prototype) | BeamNG.tech scenario JSON | None (BeamNG) | Structured police crash reports (text/NLP, not video) | Yes |

### Conclusion of Tool Survey
No maintained, ready-made open-source tool takes a raw, uncalibrated monocular YouTube dashcam video and directly outputs an ASAM OpenSCENARIO (`.xosc`) file for CARLA. The three closest academic systems (*From Dashcam Videos to Driving Simulations*, *DeepCrashTest*, and *Dash2Sim*) have not released working source code and target Scenic, Webots, or nuPlan respectively. Therefore, **Step 3** implements a clean, modular open-source pipeline (`video_to_xosc.py`) connecting maintained CV components to CARLA ScenarioRunner's `.xosc` execution engine.

---

## 4. Minimal Open-Source Pipeline Design & Glue Code (`video_to_xosc.py`, Step 3)

| Stage | Component / Library Used | Glue Code Written in `video_to_xosc.py` |
|---|---|---|
| **a) Video Download** | `yt-dlp` (`2026.10.26`) | `download_youtube_clip()`: wraps `yt-dlp` CLI to fetch a $\le 720\text{p}$ MP4 stream (with ToS/copyright reminder). |
| **b) 2D Detection & Multi-Object Tracking** | `ultralytics` (`8.4.215`) **YOLO11m/n** + **ByteTrack** (`bytetrack.yaml`) | Detects both vehicles (`car`, `motorcycle`, `bus`, `truck`) and vulnerable road users / wildlife (`person`, `bicycle`, `dog`, `horse`, `cow`, `sheep`, `bear`), detects multi-camera scene cuts (e.g. TeslaCam front-to-side repeater cuts), and detects short-duration looming frontal collision obstacles. |
| **c) Monocular Metric Depth & Ego-Motion** | Pinhole Width + Ground-Plane Depth Fusion + OpenCV Pyramidal Lucas-Kanade Optical Flow VO | Estimates ego yaw rate and forward speed (allowing full deceleration to `0.0 m/s`), auto-detects night scenes (`sky_mean < 65`) and oncoming headlights, and reconstructs emergency braking + shoulder pull-over trajectories after collision impact. |
| **d) Road Layout & CARLA Map Projection** | `carla` Python API (`1.0`) Waypoint Graph | `project_frenet_to_carla_map()`: connects to CARLA, finds a continuous multi-lane highway corridor on `Town04_Opt` where both left and right adjacent lanes are driving lanes, and maps Frenet coordinates $(s(t), d(t))$ onto CARLA world transforms $(x, y, z, \text{yaw})$. |
| **e) OpenSCENARIO 1.0 (`.xosc`) Generation** | `xml.etree.ElementTree` + `xmlschema` (`1.0.18`) | `write_openscenario_xosc()`: emits `<FileHeader description="CARLA:..." revMajor="1" revMinor="0"/>`, spawns `hero`, vehicles, and `<Pedestrian>` collision obstacles via `<TeleportAction>`, writes `<FollowTrajectoryAction>` polylines paired with time-triggered `<SpeedAction>` events (including full stops at `0.0 m/s`), and validates against `srunner/openscenario/OpenSCENARIO.xsd`. |
| **f) Replay in CARLA** | `run_replay.py` (`scenario_runner.py` + `record_carla_replay.py`) | Single-command launcher that executes the `.xosc` in CARLA 1.0, moves the CARLA window spectator camera to follow `hero`, enables vehicle headlights & impact ragdoll physics, and records `demo/carla_replay_dashcam.mp4` + `demo/carla_replay_telemetry.json`. |

---

## 5. Exact Commands to Run Everything

```powershell
$workDir = "C:\Users\adamk\.gemini\antigravity\scratch\carla-video-to-xosc"
$carlaRoot = "$workDir\CARLA_1.0\Carla-1.0-Win64-Shipping"
$srRoot = "$workDir\scenario_runner"
$env:CARLA_ROOT = $carlaRoot
$env:SCENARIO_RUNNER_ROOT = $srRoot
$env:PYTHONPATH = "$carlaRoot\PythonAPI\carla;$srRoot"

# 1. Start CARLA 1.0 Server (Off-Screen, Low Quality)
Start-Process -FilePath "$carlaRoot\CarlaUnreal\Binaries\Win64\CarlaUnreal-Win64-Shipping.exe" `
    -ArgumentList "CarlaUnreal", "-RenderOffScreen", "-quality-level=Low" `
    -WorkingDirectory $carlaRoot

# 2. Run Bundled ScenarioRunner OpenSCENARIO 1.0 Example (FollowLeadingVehicle.xosc)
Start-Process -FilePath "$workDir\.venv\Scripts\python.exe" -ArgumentList "$workDir\drive_hero_headless.py" -NoNewWindow
& "$workDir\.venv\Scripts\python.exe" "$srRoot\scenario_runner.py" `
    --openscenario "$srRoot\srunner\osc_examples\FollowLeadingVehicle.xosc" --output

# 3. Convert Dashcam Video Clip to OpenSCENARIO 1.0 (.xosc)
& "$workDir\.venv\Scripts\python.exe" "$workDir\video_to_xosc.py" `
    --video "$workDir\demo\dashcam_input.mp4" `
    --town "Town04_Opt" `
    --output-xosc "$workDir\demo\dashcam_reconstructed.xosc" `
    --output-json "$workDir\demo\reconstructed_trajectories.json" `
    --xsd "$srRoot\srunner\openscenario\OpenSCENARIO.xsd"

# 4. Replay Reconstructed OpenSCENARIO (.xosc) in CARLA 1.0 & Record Replay Dashcam Video
Start-Process -FilePath "$workDir\.venv\Scripts\python.exe" -ArgumentList "$workDir\record_carla_replay.py" -NoNewWindow
& "$workDir\.venv\Scripts\python.exe" "$srRoot\scenario_runner.py" `
    --openscenario "$workDir\demo\dashcam_reconstructed.xosc" --output
```
