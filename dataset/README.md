# Corner-case dataset generator (CARLA)

Scenario family: V1 turns right at a 3-way junction while a pedestrian crosses the exit road (derived from the
AutoScenario crash-report demo). Each run randomises town/junction, V1 speed, pedestrian timing/speed/model,
extra car and barrier, and weather incl. night.

```
# CARLA editor running and in Play mode
python generate_dataset.py --runs 3 --out ../dataset_test      # test
python generate_dataset.py --runs 200 --out ../dataset         # batch (resumable)
python check_overlay.py ../dataset_test/run_0000               # eyeball the boxes
python export_coco.py ../dataset                               # coco.json + trajectories.csv
```

Per run: `rgb/*.jpg` (960x540, 10 Hz), `lidar/*.npy` (float16 x,y,z,intensity), `labels.json` (per-frame
ego/pedestrian state, ego control, distance, TTC, 2D boxes), `meta.json` (parameters, outcome, camera intrinsics).
Outcome: `collision_pedestrian`, `near_miss` (centre distance < 3 m, no contact) or `safe`.

Caveats
- Ego control labels come from CARLA's Traffic Manager autopilot (scripted), not a human or a learned policy.
- Occlusion flag is a ray-cast heuristic; boxes with part of the 3D box behind the camera are dropped.
- One scenario family, simulator rendering: expect a domain gap to real footage.
- Needs a stable server; the script waits and retries when the connection drops and restores async mode at exit.
