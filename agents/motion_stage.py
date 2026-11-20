"""
Pipeline stage that adds vehicle/pedestrian MOVEMENT after the object layout and scene generation:

  scene description + object layout + road network --(LLM)--> motion plan --(CARLA)--> run + collision result

Called from the experiment scripts as  run_motion_stage(output_folder, scene_id).
Environment switches:  AS_MOTION=0 skips the stage,  AS_MOTION_RUN=0 only writes the plan (no CARLA needed),
AS_MOTION_MAP=0 keeps the map that is currently loaded in CARLA instead of building one from the net.
"""
import json
import os

from agents.motion_planner import MotionPlanner, build_context
from agents.motion_runner import read_objects, run_motion


def run_motion_stage(output_folder, scene_id):
    if os.getenv("AS_MOTION", "1") == "0":
        return None
    obj_fn = os.path.join(output_folder, f"{scene_id}_scene_obj.txt")
    net_fn = os.path.join(output_folder, f"{scene_id}.net.xml")
    txt_fn = os.path.join(output_folder, f"{scene_id}.txt")
    if not (os.path.exists(obj_fn) and os.path.exists(net_fn)):
        print("[motion] skipped: object layout or road network missing for", scene_id)
        return None

    agents, statics = read_objects(obj_fn)
    scene_text = open(txt_fn, encoding="utf-8").read() if os.path.exists(txt_fn) else ""
    context = build_context(scene_text, agents, statics, net_fn)
    plan = MotionPlanner().plan(context, output_folder, scene_id, agents)
    plan_fn = os.path.join(output_folder, f"{scene_id}_motion.json")
    json.dump(plan, open(plan_fn, "w"), indent=1)
    print(f"[motion] plan saved to {plan_fn}")

    if os.getenv("AS_MOTION_RUN", "1") == "0":
        return plan
    try:
        return run_motion(output_folder, scene_id, load_map=os.getenv("AS_MOTION_MAP", "1") != "0")
    except RuntimeError as e:
        print("[motion] not run:", e)
        return plan
