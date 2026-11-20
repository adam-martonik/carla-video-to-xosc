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
