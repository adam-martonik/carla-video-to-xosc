"""
Turn the pipeline's SUMO road network into a drivable CARLA map.

  net.xml --netconvert--> .xodr --(signals moved off the lanes)--> client.generate_opendrive_world

Coordinate rule (verified on CARLA 1.0): SUMO net coordinates (x, y) become CARLA (x, -y) with NO netOffset
subtraction, and SUMO headings become CARLA yaw = -heading.
"""
import os
import shutil
import subprocess
import xml.etree.ElementTree as ET


def find_netconvert():
    exe = shutil.which("netconvert")
    if exe:
        return exe
    sumo_home = os.environ.get("SUMO_HOME", r"C:\Program Files (x86)\SUMO")
    for cand in (os.path.join(sumo_home, "bin", "netconvert.exe"), os.path.join(sumo_home, "bin", "netconvert")):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError("netconvert not found: add the SUMO bin folder to PATH or set SUMO_HOME")


def net_to_xodr(net_xml, xodr_path):
    """Export the SUMO net to OpenDRIVE 1.4. Raises RuntimeError when netconvert fails."""
    cmd = [find_netconvert(), "-s", net_xml, "--opendrive-output", xodr_path,
           "--junctions.scurve-stretch", "1.0", "--output.original-names"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or not os.path.exists(xodr_path):
        raise RuntimeError("netconvert failed: " + res.stderr[-400:])
    return xodr_path


def move_signals_off_lanes(xodr_path, margin=3.5):
    """SUMO writes each traffic-light signal at the lane centre, so CARLA puts the pole in the lane.
    Move every signal to just outside the outer lane edge of its road (same side)."""
    tree = ET.parse(xodr_path)
    moved = 0
    for road in tree.getroot().iter("road"):
        signals = road.find("signals")
        if signals is None or not len(signals):
            continue
        right = left = 0.0
        sec = road.find("lanes/laneSection")
        if sec is not None:
            for side, name in (("right", "r"), ("left", "l")):
                node = sec.find(side)
                if node is None:
                    continue
                total = sum(float(w.get("a", 0)) for lane in node.findall("lane") for w in lane.findall("width"))
                if name == "r":
                    right = total
                else:
                    left = total
        for idx, sig in enumerate(signals.findall("signal")):
            t = float(sig.get("t", 0))
            edge = right if t < 0 else left
            sig.set("t", "%.4f" % ((-1 if t < 0 else 1) * (edge + margin + idx * 1.0)))
            moved += 1
    tree.write(xodr_path, encoding="utf-8", xml_declaration=True)
    return moved


def shoulder_sides(plan_road):
    """Sides (of the driving direction) that get a paved shoulder: every edge that is not 'open'."""
    if not plan_road:
        return ()
    return tuple(side for side, key in (("right", "edge_right"), ("left", "edge_left")) if plan_road.get(key, "shoulder") != "open")


def _shoulder_lane(lane_id, width_m):
    lane = ET.Element("lane", {"id": str(lane_id), "type": "shoulder", "level": "true"})
    ET.SubElement(lane, "link")
    ET.SubElement(lane, "width", {"sOffset": "0", "a": "%.2f" % width_m, "b": "0", "c": "0", "d": "0"})
    ET.SubElement(lane, "roadMark", {"sOffset": "0", "type": "none", "weight": "standard", "color": "standard", "width": "0.13"})
    return lane


def add_shoulders(xodr_path, width_m=3.0, sides=("right", "left")):
    """Add a paved shoulder as the outermost lane on the given sides, so that a pushed car stays on pavement.
    SUMO exports every carriageway as its own road with right lanes only: 'right' is the outer edge of each carriageway; 'left'
    is the outer edge only when the road is one-way (with two carriageways the left side is the median, which stays as it is).
    Returns the number of roads changed."""
    if not sides:
        return 0
    tree = ET.parse(xodr_path)
    roads = [r for r in tree.getroot().iter("road") if r.find("lanes/laneSection/right") is not None and r.get("junction") == "-1"]
    changed = 0
    for road in roads:
        for sec in road.findall("lanes/laneSection"):
            done = False
            if "right" in sides:
                right = sec.find("right")
                n = len(right.findall("lane"))
                right.append(_shoulder_lane(-(n + 1), width_m))
                done = True
            if "left" in sides and len(roads) == 1 and sec.find("left") is None:
                left = ET.Element("left")
                left.append(_shoulder_lane(1, width_m))
                sec.insert(list(sec).index(sec.find("center")), left)
                done = True
            changed += 1 if done else 0
    tree.write(xodr_path, encoding="utf-8", xml_declaration=True)
    return changed


def load_xodr(client, xodr_path):
    import carla
    xodr = open(xodr_path, encoding="utf-8").read()
    client.generate_opendrive_world(xodr, carla.OpendriveGenerationParameters(
        vertex_distance=2.0, max_road_length=500.0, wall_height=0.0, additional_width=0.6,
        smooth_junctions=True, enable_mesh_visibility=True, enable_pedestrian_navigation=True))
    return client.get_world()


def build_and_load_map(client, net_xml, out_dir=None, plan_road=None):
    out_dir = out_dir or os.path.dirname(net_xml)
    xodr_path = os.path.join(out_dir, os.path.basename(net_xml).replace(".net.xml", ".xodr"))
    net_to_xodr(net_xml, xodr_path)
    n = move_signals_off_lanes(xodr_path)
    sides = shoulder_sides(plan_road)
    w = add_shoulders(xodr_path, float((plan_road or {}).get("shoulder_m", 3.0)), sides) if sides else 0
    world = load_xodr(client, xodr_path)
    print(f"[map] loaded {world.get_map().name} from {os.path.basename(xodr_path)} ({n} signals moved off the lanes, {w} shoulder(s) added)")
    return world
