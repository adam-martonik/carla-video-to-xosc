"""
Turn the pipeline's SUMO road network into a drivable CARLA map.

  net.xml --netconvert--> .xodr --(signals moved off the lanes)--> client.generate_opendrive_world

Coordinate rule (verified on CARLA 1.0): SUMO net coordinates (x, y) become CARLA (x, -y) with NO netOffset
subtraction, and SUMO headings become CARLA yaw = -heading.
"""
import copy
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


LANE_W = 3.5
TAPER_M = 30.0          # length over which a lane that ends narrows to nothing (and a lane that begins widens)


def _poly(kind, w, T):
    """(a, b, c, d) of a lane width over a piece of length T: 'const', 'shrink' (w -> 0) or 'grow' (0 -> w), smooth ends."""
    if kind == "const":
        return (w, 0.0, 0.0, 0.0)
    if kind == "shrink":
        return (w, 0.0, -3.0 * w / T ** 2, 2.0 * w / T ** 3)
    return (0.0, 0.0, 3.0 * w / T ** 2, -2.0 * w / T ** 3)


def lane_pieces(sections, road_len, frame_len, taper_m=TAPER_M, s0=0.0):
    """OpenDRIVE lane sections for a road whose lane count changes (sections from the scene builder: s0, s1, n, m in the
    builder's road frame). Lanes are numbered j = 0 (left-most of the widest extent) upward; a section holds j in m .. m+n-1.
    Every change gets a taper piece that ends at the change: lanes that end narrow to nothing, lanes that begin widen from
    nothing. Returns [{"s", "len", "lanes": {j: kind}}] in the road's own s (netconvert's length can differ slightly).
    frame_len is the length of the builder's frame that the road covers, from frame distance s0 on."""
    scale = road_len / max(frame_len, 1e-6)
    pieces = []
    start = 0.0
    for i, sc in enumerate(sections):
        js = set(range(sc["m"], sc["m"] + sc["n"]))
        end = road_len if i == len(sections) - 1 else (sc["s1"] - s0) * scale
        if i == len(sections) - 1:
            pieces.append({"s": start, "len": end - start, "lanes": {j: "const" for j in js}})
            break
        nxt = sections[i + 1]
        js2 = set(range(nxt["m"], nxt["m"] + nxt["n"]))
        T = min(taper_m, 0.5 * (end - start))
        if end - T - start > 0.5:
            pieces.append({"s": start, "len": end - T - start, "lanes": {j: "const" for j in js}})
        lanes = {j: ("const" if j in js2 else "shrink") for j in js}
        lanes.update({j: "grow" for j in js2 - js})
        pieces.append({"s": end - T, "len": T, "lanes": lanes})
        start = end
    return pieces


def _offset_poly(piece, w=LANE_W):
    """laneOffset over a piece: the first lane that keeps its width stays where it is (its left edge at -j*w), so lanes that
    end or begin on the left move nothing else."""
    lanes = piece["lanes"]
    fixed = [j for j, k in lanes.items() if k == "const"]
    anchor = min(fixed) if fixed else min(lanes)
    a, b, c, d = -anchor * w, 0.0, 0.0, 0.0
    for j, k in lanes.items():
        if j < anchor:
            pa, pb, pc, pd = _poly(k, w, piece["len"])
            a, b, c, d = a + pa, b + pb, c + pc, d + pd
    return a, b, c, d


def apply_lane_sections(xodr_path, sections, frame_len, road_user_id="eastbound", taper_m=TAPER_M, s0=0.0):
    """Rewrite the lanes of the ego-direction road (netconvert exports it with the widest number of lanes) into lane sections
    that follow the scene builder's sections, with laneOffset keeping every lane that goes on at its lateral position and lane
    links that merge an ending lane into its neighbour (so a waypoint route follows the merge instead of stopping).
    Returns the number of lane sections written, 0 when the road is not found."""
    tree = ET.parse(xodr_path)
    road = None
    for r in tree.getroot().iter("road"):
        ud = r.find("userData")
        if ud is not None and ud.get("value") == road_user_id:
            road = r
    if road is None:
        return 0
    lanes_el = road.find("lanes")
    old = lanes_el.find("laneSection")
    center = old.find("center")
    template = old.find("right/lane")
    pieces = lane_pieces(sections, float(road.get("length")), frame_len, taper_m, s0)
    for el in list(lanes_el):
        lanes_el.remove(el)
    for p in pieces:
        a, b, c, d = _offset_poly(p)
        ET.SubElement(lanes_el, "laneOffset", {"s": "%.4f" % p["s"], "a": "%.4f" % a, "b": "%.6f" % b, "c": "%.8f" % c, "d": "%.10f" % d})

    def ids(p):
        return {j: -(i + 1) for i, j in enumerate(sorted(p["lanes"]))}

    def nearest(j, js):
        return min(js, key=lambda q: (abs(q - j), q))

    for k, p in enumerate(pieces):
        sec = ET.SubElement(lanes_el, "laneSection", {"s": "%.4f" % p["s"]})
        sec.append(copy.deepcopy(center))
        right = ET.SubElement(sec, "right")
        mine = ids(p)
        prev = ids(pieces[k - 1]) if k > 0 else None
        nxt = ids(pieces[k + 1]) if k + 1 < len(pieces) else None
        js = sorted(p["lanes"])
        for j in js:
            lane = ET.SubElement(right, "lane", {"id": str(mine[j]), "type": "driving", "level": "true"})
            link = ET.SubElement(lane, "link")
            if prev is not None:
                ET.SubElement(link, "predecessor", {"id": str(prev[j] if j in prev else prev[nearest(j, list(prev))])})
            if nxt is not None:
                ET.SubElement(link, "successor", {"id": str(nxt[j] if j in nxt else nxt[nearest(j, list(nxt))])})
            wa, wb, wc, wd = _poly(p["lanes"][j], LANE_W, p["len"])
            ET.SubElement(lane, "width", {"sOffset": "0", "a": "%.4f" % wa, "b": "%.6f" % wb, "c": "%.8f" % wc, "d": "%.10f" % wd})
            mark = template.find("roadMark")
            rm = copy.deepcopy(mark) if mark is not None else ET.Element("roadMark", {"sOffset": "0", "weight": "standard",
                                                                                    "color": "standard", "width": "0.13"})
            rm.set("type", "solid" if j == js[-1] else "broken")
            lane.append(rm)
            sp = template.find("speed")
            if sp is not None:
                lane.append(copy.deepcopy(sp))
    if pieces:
        road_id = road.get("id")
        first_ids = ids(pieces[0])
        for other_r in tree.getroot().iter("road"):
            if other_r is road:
                continue
            lk = other_r.find("link/successor")
            if lk is None or lk.get("elementType") != "road" or lk.get("elementId") != road_id:
                continue
            for o_lane in other_r.findall("lanes/laneSection/right/lane"):
                slk = o_lane.find("link/successor")
                if slk is None:
                    continue
                try:
                    old_id = int(slk.get("id"))
                except (TypeError, ValueError):
                    continue
                if old_id < 0:
                    j_old = -old_id - 1
                    new_id = first_ids[j_old] if j_old in first_ids else first_ids[nearest(j_old, list(first_ids))]
                    slk.set("id", str(new_id))
    tree.write(xodr_path, encoding="utf-8", xml_declaration=True)
    return len(pieces)


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


def add_shoulders(xodr_path, width_m=3.0, sides=("right", "left"), one_way=None, skip_roads=()):
    """Add a paved shoulder as the outermost lane on the given sides, so that a pushed car stays on pavement.
    SUMO exports every carriageway as its own road with right lanes only: 'right' is the outer edge of each carriageway; 'left'
    is the outer edge only when the road is one-way (with two carriageways the left side is the median, which stays as it is).
    Returns the number of roads changed."""
    if not sides:
        return 0
    tree = ET.parse(xodr_path)
    def sumo_id(r):
        ud = r.find("userData")
        return ud.get("value") if ud is not None else None
    roads = [r for r in tree.getroot().iter("road") if r.find("lanes/laneSection/right") is not None and r.get("junction") == "-1"
             and sumo_id(r) not in skip_roads]
    changed = 0
    for road in roads:
        for sec in road.findall("lanes/laneSection"):
            done = False
            if "right" in sides:
                right = sec.find("right")
                n = len(right.findall("lane"))
                right.append(_shoulder_lane(-(n + 1), width_m))
                done = True
            # one-way: a road whose lane count changes is several roads in a row, so the flag is passed in when known
            single_direction = (len(roads) == 1) if one_way is None else bool(one_way)
            if "left" in sides and single_direction and sec.find("left") is None:
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
    for pc in (plan_road or {}).get("xodr_pieces") or []:          # road pieces whose lane count changes inside them
        apply_lane_sections(xodr_path, pc["sections"], pc["s1"] - pc["s0"], road_user_id=pc["road"], s0=pc["s0"])
    n = move_signals_off_lanes(xodr_path)
    sides = shoulder_sides(plan_road)
    one_way = bool(plan_road.get("one_way")) if (plan_road or {}).get("sections") else None
    w = add_shoulders(xodr_path, float((plan_road or {}).get("shoulder_m", 3.0)), sides, one_way=one_way,
                      skip_roads=tuple((plan_road or {}).get("no_shoulder") or ())) if sides else 0
    world = load_xodr(client, xodr_path)
    print(f"[map] loaded {world.get_map().name} from {os.path.basename(xodr_path)} ({n} signals moved off the lanes, {w} shoulder(s) added)")
    return world
