"""
Motion planning stage: asks the LLM how every generated actor should MOVE and returns a validated JSON plan.

The public AutoScenario pipeline only places actors (spawn + brake). This stage adds the missing behaviour:
speed, route, start trigger per actor. The plan is executed by agents/motion_runner.py.

Plan schema (all coordinates in the SUMO net frame, i.e. the frame of the lane table in the prompt):
{
  "ego": "v1",
  "duration_s": 15,
  "agents": {
    "v1": {"mode": "drive", "speed_kmh": 50, "already_moving": true, "path": [[x, y], ...]},
    "v3": {"mode": "drive", "speed_kmh": 50, "path": [[x, y]], "avoid": false,
           "start": {"type": "ttc_to_point", "point": [x, y], "seconds": 2.0}},
    "p1": {"mode": "walk", "speed_ms": 1.4, "path": [[x, y], [x, y]], "start": {"type": "time", "t": 2.0}},
    "v4": {"mode": "static"}
  }
}
"""
import json
import os
import re
import xml.etree.ElementTree as ET

from agents.task_agent import TaskAgent

MODES = ("drive", "walk", "static")
TRIGGERS = ("time", "ttc_to_point")

PROMPT = """
You are a scenario choreographer for autonomous-driving tests in the CARLA simulator.
The road network and the start positions of all actors are already fixed. Your job is to decide how every actor MOVES so that the described event really happens (for example a crossing vehicle that crashes into another one).

Coordinate frame: use the SUMO net frame of the lane table below (x to the east, y to the north, metres). Headings: east = 0 deg, north = 90 deg.

Output rules:
1. Reply with the sections `## Description`, `## Reasoning` and `## Decision`. In `## Decision` give ONE JSON object inside a ```json code block and nothing else.
2. JSON schema:
   {{"ego": "<id of the vehicle under test>", "duration_s": <number, 8 to 25>,
    "agents": {{"<actor id>": {{"mode": "drive" | "walk" | "static", ...}}}}}}
   - mode "drive" (vehicles): "speed_kmh" (number), "path" (list of [x, y] destination points in order; the traffic manager chooses the road between them, so give the point after each turn or the exit of the junction, on the lane the vehicle should end up in), optional "avoid" (true = the vehicle brakes for obstacles, default false so that the described crash can happen), optional "already_moving" (true = the vehicle is already driving at speed_kmh when the scene begins, use it for traffic that is in motion in the description; otherwise the vehicle pulls away from standstill at its start time with a normal acceleration of about 3 m/s^2), optional "start".
   - mode "walk" (pedestrians): "speed_ms" (0.8 to 2.2), "path" (list of [x, y] points, walked in a straight line one after another), optional "start".
   - mode "static": the actor stays braked and does not move.
   - "start" is either {{"type": "time", "t": <seconds after the simulation begins>}} or {{"type": "ttc_to_point", "point": [x, y], "seconds": <s>, "who": "<actor id, default ego>"}}, which starts the actor when the referenced actor would reach the point in about that many seconds. Use ttc_to_point to time a crossing so that two actors meet. Default start is time 0.
3. Every actor of the actor list must appear exactly once. Actors that wait, park or are only background stay "static".
4. Make speeds and timings consistent with the scenario text. Keep all path points on the lanes of the network.

Scenario text:
{scenario}

Actors (SUMO frame, heading in degrees, type):
{actors}

Static objects: {objects}

Lane table (edge id, lane index, first point -> last point, direction of travel is first to last):
{lanes}

Junction centre of the network: {centre}
"""


def _lane_table(net_xml):
    root = ET.parse(net_xml).getroot()
    rows = []
    for e in root.findall("edge"):
        if e.get("function") == "internal":
            continue
        for lane in e.findall("lane"):
            pts = [tuple(float(v) for v in p.split(",")) for p in lane.get("shape").split()]
            rows.append("%s lane %s: (%.1f, %.1f) -> (%.1f, %.1f)" % (
                e.get("id"), lane.get("index"), pts[0][0], pts[0][1], pts[-1][0], pts[-1][1]))
    loc = root.find("location")
    cb = [float(v) for v in loc.get("convBoundary").split(",")]
    centre = "(%.1f, %.1f)" % ((cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2)
    return "\n".join(rows), centre


def build_context(scene_text, obj_agents, obj_static, net_xml):
    lanes, centre = _lane_table(net_xml)
    actors = []
    for aid, a in obj_agents.items():
        x, y, _ = a["location"]
        actors.append("%s: %s at (%.2f, %.2f), heading %.0f deg" % (aid, a["type"], x, -y, -a["rotation"][1]))
    objs = ", ".join("%s: %s at (%.1f, %.1f)" % (k, v["type"], v["location"][0], -v["location"][1])
                     for k, v in obj_static.items()) or "none"
    return {"scenario": scene_text.strip(), "actors": "\n".join(actors), "objects": objs,
            "lanes": lanes, "centre": centre}


def parse_plan(text):
    """Extract and validate the JSON plan from an LLM reply. Returns (plan, error)."""
    m = re.search(r"```json\s*(\{.*?\})\s*```", text, re.DOTALL)
    raw = m.group(1) if m else None
    if raw is None:
        m = re.search(r"(\{.*\})", text, re.DOTALL)
        raw = m.group(1) if m else None
    if raw is None:
        return None, "no JSON object found"
    try:
        plan = json.loads(raw)
    except ValueError as e:
        return None, "invalid JSON: %s" % e
    if not isinstance(plan.get("agents"), dict) or not plan["agents"]:
        return None, "missing 'agents'"
    for aid, a in plan["agents"].items():
        if a.get("mode") not in MODES:
            return None, "agent %s has no valid mode" % aid
        if a["mode"] in ("drive", "walk"):
            path = a.get("path")
            if not path or not all(isinstance(p, list) and len(p) == 2 for p in path):
                return None, "agent %s needs a path of [x, y] points" % aid
        st = a.get("start")
        if st is not None and st.get("type") not in TRIGGERS:
            return None, "agent %s has an unknown start trigger" % aid
        if st is not None and st["type"] == "ttc_to_point" and len(st.get("point", [])) != 2:
            return None, "agent %s: ttc_to_point needs a point" % aid
    plan.setdefault("duration_s", 15)
    return plan, None


def fallback_plan(obj_agents):
    return {"ego": next(iter(obj_agents)), "duration_s": 8, "agents": {k: {"mode": "static"} for k in obj_agents},
            "note": "fallback: the LLM reply could not be parsed, everything stays static"}


class MotionPlanner(TaskAgent):
    def __init__(self):
        super().__init__()
        self.context = None

    def refine_request(self, user_request, add_info=None):
        return PROMPT.format(**self.context)

    def plan(self, context, out_dir, scene_id, obj_agents, retries=3):
        self.context = context
        raw_fn = os.path.join(out_dir, f"{scene_id}_motion.txt")
        for attempt in range(1, retries + 1):
            reply = self.send_request("", {"output_fn": raw_fn})
            plan, err = parse_plan(reply)
            if plan is not None:
                return plan
            print(f"[motion] reply rejected ({err}), attempt {attempt}/{retries}", flush=True)
        print("[motion] using the fallback plan: all actors static", flush=True)
        return fallback_plan(obj_agents)
