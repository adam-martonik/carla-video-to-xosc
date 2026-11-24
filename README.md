# AutoScenario

## Overview
**AutoScenario** is an automated framework designed to improve the safety and reliability of **autonomous vehicle (AV) systems** through **realistic corner case generation**. Leveraging a **multimodal Large Language Model (LLM)**, it transforms **safety-critical real-world data** into structured scenario representations, enabling **generalization of key risk factors** for robust AV validation.

By integrating with **Simulation of Urban Mobility (SUMO) and CARLA**, AutoScenario automates **scenario generation, execution, and evaluation**, ensuring diverse and challenging test cases for AV systems. The framework supports **multimodal inputs(Text, Image and Video)**, making it an adaptable approach to **simulating rare and high-risk driving conditions**.

---

## Key Features
- 🚗 **Realistic Corner Case Generation**  
  Uses **LLMs** to synthesize **high-risk driving scenarios** from real-world data.  
- 📡 **Multimodal Input Processing**  
  Converts **Multimodal data or scene descriptions** into structured scenario representations.  
- 🏙 **Seamless Integration with SUMO & CARLA**  
  Automates the creation of AV test scenarios in standard simulators.  
- 📊 **Scenario Evaluation & Analysis**  
  Provides  **performance metrics** for generated scenarios.  

---

## Installation
### Prerequisites
- Python 3.8+
- **Recommended**: Use a virtual environment (Conda or virtualenv)

### **Step 1: Create & Activate Virtual Environment (Recommended)**
Using **Conda**:
```sh
conda create --name autoscenario python=3.8
conda activate autoscenario
```
Or using **virtualenv**:
```sh
python -m venv autoscenario
source autoscenario/bin/activate  # On macOS/Linux
autoscenario\Scripts\activate     # On Windows
```

### **Step 2: Install Dependencies**
Once the virtual environment is activated, install the required dependencies:
```sh
pip install -r requirements.txt
```
If you have not installed **CARLA** yet, please follow the official installation guide here:  
👉 [**CARLA Quick Start Guide**](https://carla.readthedocs.io/en/latest/start_quickstart/)

---

## **Usage**
### **Step 1: Set Up API Key and Configuration**
Before running simulations, configure your API key:

1. **Copy the example configuration file:**
   ```sh
   cp config.example .env
   ```

2. **Edit the `.env` file** with your actual API credentials:
   ```
   OPENAI_KEY=your-openai-api-key-here
   OPENAI_URL=https://api.openai.com/v1/chat/completions
   OPENAI_MODEL=gpt-4o
   OPENAI_MAX_TOKENS=2000
   OPENAI_TIMEOUT=30
   ```
3. Save the file in the project root directory.

---

### **Step 2: Run the Main Simulation**
You can run different **scenario generation modes** based on your input type(- Test samples can be found in the **data** folder):

#### **Command-Based Generation**
```sh
python experiments/auto_generate_all_command.py
```
#### **Long-Text-Based Generation**
```sh
python experiments/auto_generate_all_text.py
```
#### **Image-Based Generation**
```sh
python experiments/auto_generate_all_vlm.py
```
#### **Video-Based Generation**
```sh
python experiments/auto_generate_all_video.py
```


### Run Scenario Evaluations
```sh
python evaluation/general_evaluator.py
```



---

## Project Structure
```
/AutoScenario
│── agents/           # Core source code for AV agents
│── experiments/      # Experiment scripts for scenario testing
│── tools/            # Utility and evaluation scripts
│── data/             # Multimodal input datasets
│── requirements.txt  # Python dependencies
│── .gitignore        # Git ignore rules
│── LICENSE           # Open-source license
│── README.md         # Project documentation
```

---

## Contributing
We welcome contributions! 🚀  
To contribute, please follow these steps:

1. **Fork the repository** 📌  
2. **Create a new branch** (`feature-branch`) 🔀  
3. **Commit your changes** (`git commit -m "Add feature X"`) ✅  
4. **Push to your branch** (`git push origin feature-branch`) ⏫  
5. **Create a Pull Request (PR)** 📝  

---

## Citation
If you use **AutoScenario** in your research or projects, please cite us:  
📄 **Qiujing Lu, Meng Ma, Ximiao Dai, Xuanhan Wang, and Shuo Feng.**  
*"Realistic Corner Case Generation for Autonomous Vehicles with Multimodal Large Language Model."*  
**arXiv preprint arXiv:2412.00243 (2024).**  

```bibtex
@article{lu2024realistic,
  title={Realistic Corner Case Generation for Autonomous Vehicles with Multimodal Large Language Model},
  author={Lu, Qiujing and Ma, Meng and Dai, Ximiao and Wang, Xuanhan and Feng, Shuo},
  journal={arXiv preprint arXiv:2412.00243},
  year={2024}
}
```

Your citation **supports our work** and helps improve **future development** of AutoScenario! 🚀

---

## License
This project is **open-source** and licensed under the **MIT License**. See the `LICENSE` file for details.

---

## Contact & Support
💬 **For issues or collaborations, please:**  
- **Open an issue** on GitHub  
- **Reach out to the maintainers**  

🚀 **Let’s build safer AV systems together!** 🚗💨  

## Movement stage (added)
After the object layout and scene generation, `agents/motion_stage.py` adds what the public repo lacks: movement.
1. `agents/motion_planner.py` asks the LLM for a JSON motion plan (mode, speed, path, start trigger per actor) -> `<scene>_motion.json`.
2. `agents/map_builder.py` exports the SUMO net to OpenDRIVE (signals moved off the lanes) and loads it in CARLA.
3. `agents/motion_runner.py` spawns the actors from `<scene>_scene_obj.txt`, drives them along waypoint routes with a pure-pursuit
   controller, walks pedestrians, records collisions -> `<scene>_motion_result.json`.
Every experiment script calls `run_motion_stage(output_folder, scene_id)` at the end. `AS_MOTION=0` skips it, `AS_MOTION_RUN=0` only
writes the plan (no CARLA needed), `AS_MOTION_MAP=0` keeps the currently loaded map. Standalone: `python -m agents.motion_runner <scene_id> --no-map`.

Driving physics of the runner: acceleration limited to 3 m/s^2, braking 3.5 m/s^2 (6 m/s^2 emergency, the measured limit of the CARLA brakes), speed reduced in curves to
keep lateral acceleration below 3 m/s^2, throttle/brake/steering rate limits, physics substepping (0.01 s), pedestrians accelerate
at 1.5 m/s^2, vehicles brake hard 0.4 s after a crash. Vehicles flagged `already_moving` start with their planned speed, all others pull
away from rest. `<scene>_motion_result.json` reports the peak acceleration of each vehicle before the first collision.

## Standard OpenSCENARIO export (added)
`agents/motion_runner.py` also records the pose of every actor at every tick -> `<scene>_trajectories.json` (raw CARLA frame).
`python -m agents.xosc_export <folder> <scene_id> [--dt 0.1] [--hold 0.3] [--xsd tests/fixtures/OpenSCENARIO.xsd]` turns it into
`<scene>.xosc` (OpenSCENARIO 1.0, replayable with CARLA ScenarioRunner, ego named `hero`, road network `<scene>.xodr`):
- `agents/traj_smoothing.py` (ported from carla-video-to-xosc): Hampel outlier filter, RTS Kalman smoothing of x(t) and y(t), then speed, acceleration, braking, jerk and cornering limits per vehicle class. The path shape is kept, only the timing along it changes.
- `agents/xosc_writer.py`: schema-valid writer, one timed `Polyline` trajectory per actor. Validated against `tests/fixtures/OpenSCENARIO.xsd` (ASAM, 1.0.0).
- The recording is cropped 0.3 s after the first contact of the expected pair: motion after an impact is not drivable behaviour.
- Static props (barriers, parked objects) are not exported yet.

## Road shape, traffic and contact checking (added)
`python -m agents.local_pipeline <video> --out <folder>` builds the scene from the video without any hand editing:
- `agents/road_curve.py` measures the road bend from the ego yaw rate (optical flow of the far field + the speed overlay); the road is straight when the estimate is weak or the model's left/right answer disagrees. `agents/road_probes.py` reads vehicles per lane, the road edges (barrier / shoulder / open) and one-way vs two-way from the frames, and searches the crash moment coarse-to-fine.
- `agents/scene_builder.py` places everything in road coordinates (`agents/road_frame.py`), so lanes, traffic, the cutting car and the ego route follow the curve. Measured traffic is queued ahead (never in the crash corridor, at most 12 vehicles).
- The cutting car is steered from live positions (`agents/intercept.py`) instead of a pre-computed path; after contact it stops steering and both cars brake. `agents/map_builder.py` adds a paved shoulder, `agents/motion_runner.py` puts barrier props on the edges.
- `agents/scenario_check.py` accepts a run only when the expected pair touched inside the time window, the contact direction fits (a cut-in must be a side contact) and nothing else was hit. A failed run is repeated with changed controller parameters (4 runs at most) and the report says which rule failed.
- If CARLA rejects a curved map the road is rebuilt straight and the report says so. `python -m tools.smoke_curved_map` tests curved maps against a running CARLA.
- Tests: `python -m unittest discover -s tests -v` (no CARLA needed).

## Colliding-vehicle dashcam & scenario variations (added)
- **Dashcam & first-person view on colliding vehicles (`--dashcam`)**: `python -m agents.local_pipeline <video> --out <folder> --dashcam` (or `--camera dashcam|dashcam_all|both`) mounts windshield dashcam sensors on the actual colliding vehicle(s) (`<scene>_dashcam_<vid>.mp4`) rather than on a third-party bystander camera car, places the live CARLA spectator viewport in first-person perspective on the primary colliding vehicle, and calibrates mount height per vehicle class (`car`, `suv`, `van`, `truck`, `motorcycle`). Use `--dashcam-mode primary` for only the primary colliding vehicle, and `--ego-on-collider` to export that vehicle as OpenSCENARIO `hero`.
- **Timing variations (`--timing-variants`, `--crashing-only`)**: generate multiple scenarios from a single video/spec with shifted/speed-scaled collisions (`earlier_collision`, `later_collision`, `fast_collision`, `slow_collision`, `much_earlier_collision`, `much_later_collision`), timing-shifted non-collisions (`near_miss_early`, `near_miss_late`, `miss_early`, `miss_late`, or custom `dt:+1.5` / `shift:-1.0`), or without the aggressor car (`no_aggressor`). Pass `--crashing-only` (alias `--crash-only`) to generate **only crashing scenarios** when changing timings.
- **Road-design & weather/time variations (`--road-variants`, `--vary-weather`, `--weather-variants`)**: transfer the same corner case onto different road designs (`original`, `straight_2lane_twoway`, `straight_3lane_highway`, `wide_4lane_highway`, `curve_left`, `curve_right`, `s_curve`, `lane_drop`, `highway_fork`, `crossroad_1lane`, `crossroad_2lane`, `crossroad_3lane`). Add `--vary-weather` to automatically assign distinct weather & time-of-day (`day_rain`, `dusk_cloudy`, `night_clear`, `day_fog`, `night_rain`, `dawn_fog`, `day_storm`, ...) per changed map with automatic vehicle headlights, or `--weather-variants` for explicit control.
- **Combined road + timing + weather generation**: pass `--road-variants`, `--timing-variants`, and `--vary-weather`/`--weather-variants` together (`--variant-combine product` or `zip`). Variants are grouped by road design so CARLA reuses the loaded OpenDRIVE map across timing and weather variants on the same road. Standalone from an existing spec: `python -m agents.scenario_variants --spec <spec.json> --out <folder> --road-variants 3 --timing-variants 4 --crashing-only --vary-weather --dashcam --sim`.

## Smart-car sensors (Tesla cameras + 360° LiDAR) & controllable training interface (added)
- **Smart-car perception suite (`--sensors`, `--save-sensors`)** (`agents/sensor_suite.py`): mount a calibrated smart-car sensor rig onto the subject/colliding vehicle (automatically scales mount positions across cars, SUVs, vans, trucks, and motorcycles, and spawns a `vehicle.tesla.model3` when available for Tesla presets):
  - `tesla`: 8 Tesla-style surround RGB cameras (`front_main`, `front_wide`, `front_narrow`, `left_pillar`, `right_pillar`, `left_repeater`, `right_repeater`, `rear`).
  - `tesla_lidar`: all 8 Tesla surround cameras + roof-mounted 360° spinning LiDAR (20 Hz full sweep every tick).
  - `tesla_full`: 8 Tesla surround cameras + 360° roof LiDAR + front radar.
  - `surround_lidar`: 4 surround cameras (`front_wide`, `left_pillar`, `right_pillar`, `rear`) + 360° LiDAR.
  - `front_lidar` / `lidar` / `semantic_lidar`: lightweight presets optimized for fast training loops.
  - With default `--control auto`, the scenario plays normally while recording a multi-camera + 360° LiDAR BEV mosaic (`--record` / `--sensor-video` -> `<scene>_sensors_<vid>.mp4`) and/or synchronized RGB frames, `.npy` point clouds, telemetry, and controls (`--save-sensors` -> `<out>/sensor_data/`).
- **Controllable vehicle option (`--control`, `--action-space`, `--controller`)** (`agents/ego_control.py`, no bundled models):
  - **Standard training action spaces (`--action-space`)**:
    1. `continuous` *(recommended default)*: 2D vector `[steer, accel]` in `[-1, 1] × [-1, 1]` where `accel > 0` drives throttle and `accel < 0` drives brake (preventing simultaneous gas + brake).
    2. `discrete`: 9 driving primitives (`coast`, `accelerate`, `gentle_brake`, `hard_brake`, `steer_left`, `steer_right`, `brake_left`, `brake_right`, `hard_accelerate`).
    3. `target_speed`: `[steer, target_speed_ms]` translated to smooth throttle/brake pedals by an internal longitudinal controller.
  - **Interactive keyboard driving & demo recording**: `python -m agents.local_pipeline --sim-only --out results/smart_1 --sensors tesla_lidar --control keyboard --save-sensors` (WASD / Arrow keys, Space = handbrake, Q = reverse, ESC = quit, with a live Pygame multi-cam + LiDAR BEV HUD).
  - **Plug in your own external controller/model**: `python -m agents.local_pipeline --sim-only --out results/smart_1 --sensors front_lidar --control external --controller path/to/my_agent.py:act --record`. By default, cut-in aggressors execute their natural open-loop lane merge rather than chasing an evading car (`--adversarial-cutter` re-enables homing).
- **Gymnasium-compatible training environment** (`agents/carla_env.py`):
  - `CarlaScenarioEnv` exposes standard `reset()` / `step(action)` / `render()` / `close()` with multi-modal observations (`state`, `cameras`, `lidar`, `lidar_bev`, `lidar_sectors`), crash-avoidance reward shaping, and OpenDRIVE map reuse across resets and `variants_manifest.json` variants (works directly with Gymnasium / Stable-Baselines3 when `gymnasium` is installed, and standalone with any Python/PyTorch/JAX loop otherwise).
