"""
Deterministic scene builder: scene spec (agents/scene_spec.py) -> road network, object layout and motion plan
in the file formats the motion runner already uses. No LLM involved, so it cannot fail on a bad reply.

Timing is solved analytically: the actor that is part of the event is placed so that it reaches the conflict
point at the event time (collision: at the same time as the ego car, near miss: about one second later).

Frames: SUMO net frame (x east, y north). Headings: east 0, north 90, west 180, south -90.
CARLA frame = (x, -y), yaw = -heading.
"""
import json
import math
import os
import re
import subprocess

from agents.map_builder import find_netconvert
from agents.road_frame import RoadFrame, lane_offset

W = 3.5                 # lane width
S_START = 15.0          # distance along a straight road where the ego car starts
CAR_LEN = 4.8
HALF_J = None           # set per road
VEHICLE_TYPES = ("car", "suv", "truck", "van", "motorcycle", "bike")
TRAFFIC_TYPES = ("car", "suv", "car", "car", "van")
TRAFFIC_LANES = ("ego_lane_ahead", "left_lane", "right_lane", "oncoming")
MAX_TRAFFIC = 12
SHOULDER_M = 3.0


def _write_text(path, text):
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)


# ----------------------------------------------------------------------------------------- road templates
class Road:
    """Geometry of the generated road in the net frame."""

    def __init__(self, kind, lanes, centerline=None, one_way=False):
        self.kind, self.L = kind, lanes
        self.one_way = bool(one_way)
        if kind == "straight":
            pts = [(float(p[0]), float(p[1])) for p in centerline] if centerline else [(0.0, 0.0), (300.0, 0.0)]
            mx, my = min(p[0] for p in pts), min(p[1] for p in pts)
            # netconvert moves a net so that its centre line starts at x, y >= 0; start there so that no shift is left over
            self.frame = RoadFrame([(x - mx, y - my) for x, y in pts])
            self.length = self.frame.length
            self.end_s = self.length - 10.0
            self.C = None
        else:
            self.end_s = None
            self.frame = None
            self.arm = 100.0
            self.C = (100.0, 100.0)
            self.half_j = lanes * W + 4.0

    def off(self, k):
        """Offset of lane index k (0 = right-most lane of a direction) from the road centre line."""
        return (self.L - 0.5 - k) * W

    # lane centre lines: returns (fixed coordinate, direction) -----------------------------------------
    def eastbound_y(self, k):
        return (-self.off(k)) if self.kind == "straight" else self.C[1] - self.off(k)

    def westbound_y(self, k):
        return (self.off(k)) if self.kind == "straight" else self.C[1] + self.off(k)

    def lane_point(self, s, k):
        """Centre of lane k (0 = right-most) of the main carriageway, at distance s along the road (straight roads)."""
        return self.frame.to_xy(s, lane_offset(self.L, k))

    def lane_heading(self, s):
        return self.frame.heading(s)

    def set_offset(self, ox, oy):
        """Apply the shift netconvert reports (netOffset) to the road frame."""
        self.frame = RoadFrame([(x + ox, y + oy) for x, y in self.frame.pts])

    def opposite_lane_point(self, s, k):
        """Centre of lane k of the opposite carriageway at distance s along the road."""
        return self.frame.to_xy(s, -lane_offset(self.L, k))

    def northbound_x(self, k):
        return self.C[0] + self.off(k)

    def southbound_x(self, k):
        return self.C[0] - self.off(k)


def write_net(road, out_dir, scene_id, signals):
    nod = os.path.join(out_dir, scene_id + ".nod.xml")
    edg = os.path.join(out_dir, scene_id + ".edg.xml")
    net = os.path.join(out_dir, scene_id + ".net.xml")
    L = road.L
    if road.kind == "straight":
        pts = road.frame.pts
        nodes = ['<node id="west" x="%.3f" y="%.3f" type="priority"/>' % pts[0], '<node id="east" x="%.3f" y="%.3f" type="priority"/>' % pts[-1]]
        shape = " ".join("%.3f,%.3f" % p for p in pts)
        rshape = " ".join("%.3f,%.3f" % p for p in reversed(pts))
        # spreadType right: the lanes lie to the right of the shape, so lane k is exactly at lane_offset(L, k) from the centre line
        edges = ['<edge id="eastbound" from="west" to="east" numLanes="%d" speed="30.0" width="%.1f" spreadType="right" shape="%s"/>' % (L, W, shape)]
        if not road.one_way:
            edges.append('<edge id="westbound" from="east" to="west" numLanes="%d" speed="30.0" width="%.1f" spreadType="right" shape="%s"/>' % (L, W, rshape))
    else:
        ctype = "traffic_light" if signals and L > 1 else "priority"
        nodes = ['<node id="center" x="0.0" y="0.0" type="%s"/>' % ctype]
        for nid, (x, y) in {"north": (0, 100), "south": (0, -100), "east": (100, 0), "west": (-100, 0)}.items():
            nodes.append('<node id="%s" x="%.1f" y="%.1f" type="priority"/>' % (nid, x, y))
        edges = []
        for nid in ("north", "south", "east", "west"):
            edges.append('<edge id="%s_in" from="%s" to="center" numLanes="%d" speed="15.0" width="%.1f"/>' % (nid, nid, L, W))
            edges.append('<edge id="%s_out" from="center" to="%s" numLanes="%d" speed="15.0" width="%.1f"/>' % (nid, nid, L, W))
    _write_text(nod, '<?xml version="1.0" encoding="UTF-8"?>\n<nodes>\n  ' + "\n  ".join(nodes) + "\n</nodes>\n")
    _write_text(edg, '<?xml version="1.0" encoding="UTF-8"?>\n<edges>\n  ' + "\n  ".join(edges) + "\n</edges>\n")
    res = subprocess.run([find_netconvert(), "--node-files=" + nod, "--edge-files=" + edg, "--default.spreadtype", "center",
                          "--no-turnarounds", "--output-file=" + net], capture_output=True, text=True)
    if res.returncode != 0 or not os.path.exists(net):
        raise RuntimeError("netconvert failed: " + res.stderr[-300:])
    if road.kind == "straight":
        with open(net, encoding="utf-8") as f:
            m = re.search(r'netOffset="([-\d.]+),([-\d.]+)"', f.read())
        if m and (abs(float(m.group(1))) > 1e-6 or abs(float(m.group(2))) > 1e-6):
            road.set_offset(float(m.group(1)), float(m.group(2)))
    return net


# ----------------------------------------------------------------------------------------- measured traffic
def traffic_actors(traffic, ego_kmh, T, lane_idx, road, existing):
    """Extra, non-event vehicles from the per-lane counts measured in the video. Returns (actors, notes).
    Vehicles in front are placed far enough ahead that neither the ego car nor the cutting car reaches them before the event
    has finished (T + 3 s), even when they are slower than the ego car."""
    notes, out = [], []
    n = {k: int(max(0, traffic.get(k, 0))) for k in TRAFFIC_LANES}
    has_cross = any(a["relation"] in ("cross_left", "cross_right") for a in existing)
    if has_cross:
        n["oncoming"] = 0
    while sum(n.values()) > MAX_TRAFFIC:
        n[max(TRAFFIC_LANES, key=lambda k: n[k])] -= 1
    v_e = ego_kmh / 3.6
    v_t_kmh = max(ego_kmh + float(traffic.get("rel_speed_kmh", 0.0)), 0.0)
    base = max(v_e - v_t_kmh / 3.6, 0.0) * (T + 3.0) + 25.0
    if has_cross:
        base = max(base, v_e * T + 35.0)
    room = road.end_s - S_START
    taken = {}
    for a in existing:
        taken.setdefault((a["relation"], a["lane"]), []).append(a["distance_m"])
        if a["relation"] == "beside" or a["behavior"] == "cutting_in":
            taken.setdefault(("ahead", "right" if a["lane"] == "right" else "left"), []).append(1.0)

    def add(relation, lane, d0, speed_kmh):
        d = d0
        while any(abs(d - x) < 9.0 for x in taken.get((relation, lane), [])):
            d += 9.0
        if relation == "ahead" and d > room:
            notes.append("traffic: a vehicle in the %s lane would stand beyond the end of the road, dropped" % lane)
            return
        taken.setdefault((relation, lane), []).append(d)
        out.append({"id": "t%d" % (len(out) + 1), "type": TRAFFIC_TYPES[len(out) % len(TRAFFIC_TYPES)], "relation": relation, "lane": lane,
                    "distance_m": d, "speed_kmh": speed_kmh, "behavior": "driving", "color": None})

    for i in range(n["ego_lane_ahead"]):
        add("ahead", "same", base + 12.0 * i, v_t_kmh)
    if n["left_lane"]:
        if lane_idx + 1 > road.L - 1:
            notes.append("traffic: the ego car is in the left-most lane, no left lane for %d vehicle(s)" % n["left_lane"])
        else:
            for i in range(n["left_lane"]):
                add("ahead", "left", base + 12.0 * i, v_t_kmh)
    if n["right_lane"]:
        if lane_idx - 1 < 0:
            notes.append("traffic: the ego car is in the right-most lane, no right lane for %d vehicle(s)" % n["right_lane"])
        else:
            for i in range(n["right_lane"]):
                add("ahead", "right", base + 12.0 * i, v_t_kmh)
    for i in range(n["oncoming"]):
        add("oncoming", "left" if (road.L > 1 and i % 2) else "same", 20.0 + 15.0 * i, 40.0)
    return out, notes


def crossroad_traffic_actors(traffic, ego_kmh, lane_idx, road, existing, ego_y0):
    """Extra non-event vehicles on a crossroad: adjacent lanes queue at the stop line; ego-lane traffic drives beyond the junction."""
    notes, out = [], []
    n = {k: int(max(0, traffic.get(k, 0))) for k in TRAFFIC_LANES}
    while sum(n.values()) > MAX_TRAFFIC:
        n[max(TRAFFIC_LANES, key=lambda k: n[k])] -= 1
    taken = {}
    for a in existing:
        if a["relation"] in ("ahead", "oncoming"):
            taken.setdefault((a["relation"], a["lane"]), []).append(ego_y0 + a["distance_m"])

    def add_y(relation, lane, y0, speed_kmh, behavior, max_y):
        y = y0
        while any(abs(y - x) < 8.5 for x in taken.get((relation, lane), [])):
            y += 9.0
        if y > max_y:
            return
        taken.setdefault((relation, lane), []).append(y)
        out.append({"id": "t%d" % (len(out) + 1), "type": TRAFFIC_TYPES[len(out) % len(TRAFFIC_TYPES)],
                    "relation": relation, "lane": lane, "distance_m": round(y - ego_y0, 2),
                    "speed_kmh": speed_kmh, "behavior": behavior, "color": None})

    y_stop = road.C[1] - road.half_j - 5.0
    y_far_base = road.C[1] + road.half_j + 20.0
    for i in range(n["ego_lane_ahead"]):
        add_y("ahead", "same", y_far_base + 12.0 * i, max(ego_kmh, 40.0), "driving", road.C[1] + 90.0)
    for key, adj_lane, ok, edge_name in (
        ("left_lane", "left", lane_idx + 1 <= road.L - 1, "left-most"),
        ("right_lane", "right", lane_idx - 1 >= 0, "right-most"),
    ):
        if not n[key]:
            continue
        if not ok:
            notes.append("traffic: the ego car is in the %s lane, no %s lane for %d vehicle(s)" % (edge_name, adj_lane, n[key]))
            continue
        for i in range(n[key]):
            y_slot = y_stop - 9.0 * i
            if y_slot >= ego_y0 + 3.0 and not any(abs(y_slot - x) < 8.0 for x in taken.get(("ahead", adj_lane), [])):
                taken.setdefault(("ahead", adj_lane), []).append(y_slot)
                out.append({"id": "t%d" % (len(out) + 1), "type": TRAFFIC_TYPES[len(out) % len(TRAFFIC_TYPES)],
                            "relation": "ahead", "lane": adj_lane, "distance_m": round(y_slot - ego_y0, 2),
                            "speed_kmh": 0.0, "behavior": "stopped", "color": None})
            else:
                add_y("ahead", adj_lane, y_far_base + 12.0 * i, max(ego_kmh, 35.0), "driving", road.C[1] + 90.0)
    for i in range(n["oncoming"]):
        lane = "left" if (road.L > 1 and i % 2) else "same"
        add_y("oncoming", lane, road.C[1] + road.half_j + 8.0 + 10.0 * i, 0.0, "stopped", road.C[1] + 95.0)
    return out, notes


# ----------------------------------------------------------------------------------------- the builder
def build(spec, out_dir, scene_id="video_interpreter_0000_split"):
    os.makedirs(out_dir, exist_ok=True)
    notes = []
    road = Road(spec["road"]["type"], spec["road"]["lanes_per_direction"], spec["road"].get("centerline"), spec["road"].get("one_way", False))
    ego = spec["ego"]
    event = spec["event"]
    v_e = ego["speed_kmh"] / 3.6
    lane_idx = 0 if ego["lane"] == "right" or road.L == 1 else road.L - 1
    actors = spec["actors"]
    # several actors given at (almost) the same place would overlap: keep a spacing of 9 m within the same lane and direction
    groups = {}
    for a in actors:                                   # an actor beside/merging occupies its lane at the ego position
        if a["relation"] == "beside" or a["behavior"] == "cutting_in":
            groups[("ahead", "right" if a["lane"] == "right" else "left")] = 1.0
    for a in sorted(actors, key=lambda a: a["distance_m"]):
        key = (a["relation"], a["lane"])
        if a["relation"] in ("ahead", "behind", "oncoming"):
            last = groups.get(key)
            if last is not None and a["distance_m"] < last + 9.0:
                a["distance_m"] = last + 9.0
            groups[key] = a["distance_m"]
    ev_actor = next((a for a in actors if a["id"] == event.get("actor_id")), None) if event["type"] != "none" else None
    ev2 = next((a for a in actors if a["id"] == event.get("actor2_id")), None) if ev_actor is not None else None
    third_party = ev_actor is not None and ev2 is not None and ev2 is not ev_actor
    straight_tp_cross = (
        third_party
        and spec["road"]["type"] == "straight"
        and ev_actor["relation"] in ("cross_left", "cross_right")
        and ev2["relation"] in ("behind", "beside")
    )
    if straight_tp_cross and road.L < 2:
        road = Road("straight", 2, spec["road"].get("centerline"), spec["road"].get("one_way", False))
        lane_idx = 0 if ego["lane"] == "right" else road.L - 1
    if third_party and not straight_tp_cross:
        # the crash is between two other road users: they meet in the junction while the ego car passes by
        if spec["road"]["type"] == "straight":
            notes.append("third-party crash needs a junction: road changed to crossroad")
            road = Road("crossroad", road.L)
        ev_actor.update({"relation": "cross_left", "behavior": "driving", "type": ev_actor["type"] if ev_actor["type"] != "pedestrian" else "car"})
        ev2.update({"relation": "cross_right", "behavior": "turning_left", "type": ev2["type"] if ev2["type"] != "pedestrian" else "car"})
    T = min(max(event["time_s"], 3.0), 8.0) if ev_actor is not None else 6.0       # simulated time of the event
    has_straight_cross = (road.kind == "straight" and (straight_tp_cross or (not third_party and any(
        a["relation"] in ("cross_left", "cross_right") and a["type"] != "pedestrian" for a in actors))))
    shoulder_m = 6.0 if has_straight_cross else SHOULDER_M
    net = write_net(road, out_dir, scene_id, spec["road"]["traffic_signals"])      # first: netconvert may shift the road frame
    tr = spec.get("traffic") or {}
    agents, plan_agents = {}, {}             # id -> {type, x, y, heading}; id -> plan spec
    expect = None

    # ---- ego geometry ---------------------------------------------------------------------------------
    if road.kind == "straight":
        heading = road.lane_heading(S_START)
        end_pt = list(road.lane_point(road.end_s, lane_idx))
    else:
        ego_x = road.northbound_x(lane_idx)
        heading = 90.0
        along = lambda p: p[1]
        ego_xy = None
        end_pt = [ego_x, road.C[1] + 90.0]

    stop_behaviours = ego["behavior"] in ("slow_to_stop", "stopped")
    plan_ego = {"mode": "drive", "speed_kmh": ego["speed_kmh"], "already_moving": ego["speed_kmh"] > 3 and ego["behavior"] != "stopped",
                "avoid": ev_actor is None or event["type"] == "near_miss" or third_party}
    if ev_actor is not None and (ev_actor["relation"] == "beside" or ev_actor["behavior"] == "cutting_in"):
        plan_ego["avoid"] = False

    # conflict point with the event actor (for crossing actors on a crossroad)
    conflict = None
    if ev_actor is not None and road.kind == "crossroad" and ev_actor["relation"] in ("cross_left", "cross_right") \
            and ev_actor["type"] != "pedestrian":
        ay = road.eastbound_y(0) if ev_actor["relation"] == "cross_left" else road.westbound_y(0)
        conflict = (ego_x, ay)
    elif ev_actor is not None and ev_actor["type"] == "pedestrian":
        conflict = None

    # ---- ego start and path ---------------------------------------------------------------------------
    if road.kind == "crossroad":
        if stop_behaviours:
            # the ego car stops short of the conflict point (collision: close enough for contact, near miss: a bit further)
            gap = {"collision": 1.5, "near_miss": 4.0}.get(event["type"], 8.0) if conflict else (road.half_j + 2.0)
            stop_y = (conflict[1] - gap) if conflict else (road.C[1] - road.half_j - 2.0)
            brake_dist = v_e * v_e / (2 * 3.0)
            d_start = brake_dist + 14.0 if v_e > 3 else 14.0
            y0 = stop_y - d_start
            if ego["behavior"] != "stopped" or v_e > 3:
                plan_ego["path"] = [[ego_x, stop_y]]
            else:
                plan_ego["path"] = [[ego_x, stop_y + 6.0]]
                y0 = stop_y
            plan_ego["avoid"] = False
            t_ego_conflict = T
        else:
            if conflict:
                T_eff = min(T, 95.0 / max(v_e, 1.0))          # the ego car reaches the conflict point at the event time
                y0 = conflict[1] - v_e * T_eff
                t_ego_conflict = T_eff
            else:
                y0 = road.C[1] - 45.0
                t_ego_conflict = T
            plan_ego["path"] = [end_pt]
            if ego["behavior"] == "turn_left":
                plan_ego["path"] = [[road.C[0] - 60.0, road.C[1] + road.off(1 if road.L > 1 else 0)]]
            elif ego["behavior"] == "turn_right":
                plan_ego["path"] = [[road.C[0] + 60.0, road.C[1] - road.off(0)]]
        y0 = max(y0, road.C[1] - 95.0)
        if not stop_behaviours and conflict is not None and not third_party:
            t_ego_conflict = (conflict[1] - y0) / max(v_e, 1.0)
        agents["v1"] = {"type": ego["type"], "x": ego_x, "y": y0, "heading": 90.0}
    else:
        s_ego = 65.0 if straight_tp_cross else S_START
        x0, y0 = road.lane_point(s_ego, lane_idx)
        plan_ego["path"] = [end_pt]
        t_ego_conflict = T
        agents["v1"] = {"type": ego["type"], "x": x0, "y": y0, "heading": road.lane_heading(s_ego)}
        if straight_tp_cross and ego["behavior"] == "stopped":
            plan_ego = {"mode": "static"}
        elif stop_behaviours:
            stop_s = s_ego + v_e * v_e / (2 * 3.0) + 6.0
            plan_ego["path"] = [list(road.lane_point(stop_s, lane_idx))]
            plan_ego["avoid"] = False
    if ego.get("color"):
        plan_ego["color"] = ego["color"]
    plan_agents["v1"] = plan_ego
    ego_start = (agents["v1"]["x"], agents["v1"]["y"])
    if any(tr.get(k, 0) for k in TRAFFIC_LANES):
        if road.kind == "straight":
            extra, tnotes = traffic_actors(tr, ego["speed_kmh"], T, lane_idx, road, actors)
        else:
            extra, tnotes = crossroad_traffic_actors(tr, ego["speed_kmh"], lane_idx, road, actors, ego_start[1])
        actors = actors + extra
        notes += tnotes

    # ---- other actors ---------------------------------------------------------------------------------
    idmap = {}
    n_cross = 0
    placed_bg_s = {}
    placed_adj_y = {}
    tp_cross_s = (65.0 + min(max(ev_actor["distance_m"], 6.5), 11.0)) if straight_tp_cross else None
    tp_k2 = (max(lane_idx - 1, 0) if ev2["lane"] == "right" else min(lane_idx + 1, road.L - 1)) if straight_tp_cross else None
    tp_t_cross = None
    if straight_tp_cross:
        d_left_tp = (0.0 if road.one_way else road.L * W) + shoulder_m - 1.5
        d_right_tp = -(road.L * W + shoulder_m - 1.5)
        d_start_tp = d_left_tp if ev_actor["relation"] == "cross_left" else d_right_tp
        d_conf_tp = lane_offset(road.L, tp_k2)
        tp_t_cross = abs(d_start_tp - d_conf_tp) / max(ev_actor["speed_kmh"] / 3.6, 1.0)
    for n, a in enumerate(actors, start=2):
        vid = "v%d" % n
        idmap[a["id"]] = vid
        is_ev = a is ev_actor
        v_a = a["speed_kmh"] / 3.6
        beh = a["behavior"]
        spec_p = None
        lane_k = lane_idx if a["lane"] == "same" else (min(lane_idx + 1, road.L - 1) if a["lane"] == "left" else max(lane_idx - 1, 0))
        if a["type"] == "pedestrian":
            # crosses the ego path ahead of the ego car, from the side given by the relation
            side = 1.0 if a["relation"] != "cross_right" else -1.0
            if road.kind == "straight":
                ps = S_START + a["distance_m"]
                y_edge = road.L * W + 1.5
                start, dest = road.frame.to_xy(ps, -side * y_edge), road.frame.to_xy(ps, side * y_edge)
                agents[vid] = {"type": "pedestrian", "x": start[0], "y": start[1], "heading": road.lane_heading(ps) + 90.0 * side}
                walk = abs(lane_offset(road.L, lane_idx) - (-side * y_edge))
            else:
                py = ego_start[1] + a["distance_m"]
                x_edge = road.L * W + 1.5
                start, dest = (ego_x - side * x_edge, py), (ego_x + side * x_edge, py)
                agents[vid] = {"type": "pedestrian", "x": start[0], "y": start[1], "heading": 0.0 if side > 0 else 180.0}
                walk = abs(ego_x - start[0])
            speed = 1.4
            t_ped_at_lane = walk / speed
            t_arrive = (a["distance_m"] / max(v_e, 1.0)) if ego["behavior"] == "cruise" else T
            if event["type"] == "near_miss" and is_ev:
                t_arrive += 1.0
            spec_p = {"mode": "walk", "speed_ms": speed, "path": [list(dest)],
                      "start": {"type": "time", "t": round(max(t_arrive - t_ped_at_lane, 0.0), 2)}}
        elif a["relation"] == "beside" or beh == "cutting_in":
            # a vehicle in the neighbouring lane that changes into the ego lane (side contact or close pass)
            left = a["lane"] != "right"
            k_src = min(lane_idx + 1, road.L - 1) if left else max(lane_idx - 1, 0)
            if k_src == lane_idx:
                notes.append("%s: the road has no neighbouring lane, placed ahead in the ego lane" % a["id"])
                if road.kind == "straight":
                    sn = S_START + max(a["distance_m"], 15.0)
                    agents[vid] = {"type": a["type"], "x": road.lane_point(sn, lane_idx)[0], "y": road.lane_point(sn, lane_idx)[1],
                                   "heading": road.lane_heading(sn)}
                else:
                    agents[vid] = {"type": a["type"], "x": road.northbound_x(lane_idx), "y": ego_start[1] + max(a["distance_m"], 15.0), "heading": 90.0}
                spec_p = {"mode": "static"}
            else:
                v = v_e if v_e > 2.0 else max(v_a, 3.0)
                if road.kind == "straight":
                    lane_pt = lambda k, s_: road.lane_point(S_START + s_, k)
                    end_s = road.end_s - S_START
                    blend = lambda s_, f: road.frame.to_xy(S_START + s_, lane_offset(road.L, k_src) + (lane_offset(road.L, lane_idx) - lane_offset(road.L, k_src)) * f)
                    head = road.lane_heading(S_START)
                else:
                    lane_pt = lambda k, s_: (road.northbound_x(k), ego_start[1] + s_)
                    end_s = road.C[1] + 90.0 - ego_start[1]
                    blend = lambda s_, f: (road.northbound_x(k_src) + (road.northbound_x(lane_idx) - road.northbound_x(k_src)) * f, ego_start[1] + s_)
                    head = 90.0
                lead = {"collision": 1.4, "near_miss": 0.6}.get(event["type"], 0.0) if is_ev else 0.0
                t_start = max(T - lead, 0.8) if is_ev else 2.0          # the lane change begins this long after the start
                dur = 2.4                                                # and takes this long
                s0 = 1.0                                                 # starts alongside the ego car, 1 m ahead of its centre
                if is_ev and road.kind == "straight":
                    # closed loop: the motion runner steers this car from the live ego position (agents/intercept.py)
                    x_i, y_i = lane_pt(k_src, s0)
                    agents[vid] = {"type": a["type"], "x": x_i, "y": y_i, "heading": head}
                    d_src, d_dst = lane_offset(road.L, k_src), lane_offset(road.L, lane_idx)
                    aim = 0.0 if event["type"] == "collision" else (2.6 if d_src > d_dst else -2.6)     # a near miss keeps a 0.6 m gap
                    spec_p = {"mode": "drive", "speed_kmh": round(v * 3.6, 1), "already_moving": True, "avoid": False,
                              "intercept": {"kind": "cut_in", "target": "v1", "lead_s": lead, "aim_offset_m": aim, "t_event_s": round(T, 2),
                                            "lane_src_d": d_src, "lane_dst_d": d_dst}}
                else:
                    s_ls = s0 + v * t_start
                    pts = [lane_pt(k_src, s_) for s_ in [s0 + 2.0 * j for j in range(int((s_ls - s0) / 2.0) + 1)]]
                    n_lc = 8
                    for j in range(1, n_lc + 1):
                        f = j / n_lc
                        f = f * f * (3 - 2 * f)                              # smooth lateral movement
                        pts.append(blend(s_ls + v * dur * j / n_lc, f))
                    s_c = s_ls + v * dur
                    while s_c < end_s:
                        s_c += 4.0
                        pts.append(lane_pt(lane_idx, min(s_c, end_s)))
                    agents[vid] = {"type": a["type"], "x": pts[0][0], "y": pts[0][1], "heading": head}
                    spec_p = {"mode": "drive", "speed_kmh": round(v * 3.6, 1), "already_moving": True, "avoid": False,
                              "polyline": [[round(x, 2), round(y, 2)] for x, y in pts]}
        elif road.kind == "straight":
            rel = a["relation"]
            if straight_tp_cross and a is ev2:
                s_c2 = max(tp_cross_s - max(v_a, 1.0) * tp_t_cross, 5.0)
                agents[vid] = {"type": a["type"], "x": road.lane_point(s_c2, tp_k2)[0], "y": road.lane_point(s_c2, tp_k2)[1],
                               "heading": road.lane_heading(s_c2)}
                spec_p = {"mode": "drive", "speed_kmh": a["speed_kmh"], "already_moving": True, "avoid": False,
                          "path": [list(road.lane_point(road.end_s, tp_k2))]}
            elif rel in ("cross_left", "cross_right"):
                left = rel == "cross_left"
                d_left = (0.0 if road.one_way else road.L * W) + shoulder_m - 1.5
                d_right = -(road.L * W + shoulder_m - 1.5)
                d_start, d_end = (d_left, d_right) if left else (d_right, d_left)
                if straight_tp_cross and a is ev_actor:
                    ps = tp_cross_s
                elif is_ev and event["type"] != "none" and not stop_behaviours:
                    d_ego = lane_offset(road.L, lane_idx)
                    t_cross = abs(d_start - d_ego) / max(v_a, 1.0)
                    t_ego_conflict = t_cross + (1.0 if event["type"] == "near_miss" else 0.0)
                    ps = min(S_START + v_e * t_ego_conflict, road.end_s - 20.0)
                else:
                    ps = S_START + max(a["distance_m"], 15.0) + 12.0 * n_cross
                n_cross += 1
                n_pts = max(int(abs(d_end - d_start) / 2.0), 4)
                pts = [road.frame.to_xy(ps, d_start + (d_end - d_start) * j / n_pts) for j in range(n_pts + 1)]
                head_a = road.lane_heading(ps) + (-90.0 if left else 90.0)
                agents[vid] = {"type": a["type"], "x": pts[0][0], "y": pts[0][1], "heading": head_a}
                if beh == "stopped" or v_a < 1.0:
                    spec_p = {"mode": "static"}
                else:
                    spec_p = {"mode": "drive", "speed_kmh": a["speed_kmh"], "already_moving": True, "avoid": False,
                              "polyline": [[round(x, 2), round(y, 2)] for x, y in pts]}
            elif rel == "ahead":
                gapd = a["distance_m"]
                if is_ev and event["type"] != "none":
                    close = (v_e - (v_a if beh not in ("stopped",) else 0.0))
                    if beh == "braking":
                        close = v_e
                    gapd = max(close * T + (CAR_LEN if event["type"] == "collision" else 12.0), 8.0)
                    if close <= 0.5:
                        notes.append("%s: event cannot happen, the actor is not slower than the ego car" % a["id"])
                sa = (65.0 if straight_tp_cross else S_START) + gapd
                if straight_tp_cross and not is_ev:
                    if lane_k == lane_idx:
                        sa = max(sa, tp_cross_s + 8.5)
                    elif lane_k == tp_k2:
                        sa = max(sa, tp_cross_s + (25.0 if beh == "stopped" else 15.0))
                elif not straight_tp_cross and not is_ev:
                    if ev_actor is not None and ((a["lane"] == "same" and beh == "driving") or (
                            a["lane"] != "same" and ev_actor["relation"] == "beside" and a["lane"] == ev_actor["lane"])):
                        sa = max(sa, S_START + max(v_e - v_a, 0.0) * (T + 2.0) + 15.0)
                    while any(abs(sa - ps0) < 9.0 for ps0 in placed_bg_s.get(lane_k, [])):
                        sa += 9.0
                    placed_bg_s.setdefault(lane_k, []).append(sa)
                sa = min(sa, road.end_s)
                agents[vid] = {"type": a["type"], "x": road.lane_point(sa, lane_k)[0], "y": road.lane_point(sa, lane_k)[1],
                               "heading": road.lane_heading(sa)}
                if beh in ("stopped",) or v_a < 1.0:
                    spec_p = {"mode": "static"}
                else:
                    spec_p = {"mode": "drive", "speed_kmh": a["speed_kmh"], "already_moving": True, "avoid": False,
                              "path": [list(road.lane_point(road.end_s, lane_k))]}
                    if beh == "braking":
                        spec_p["stop_at_s"] = round(max(T - 1.5, 1.0), 2)
                        if is_ev:           # closed loop: brake when the ego car is about 2 s away, whichever comes first
                            spec_p["brake_trigger"] = {"who": "v1", "ttc_s": 2.0}
            elif rel == "behind":
                sb = S_START - max(a["distance_m"], 10.0)
                if sb < 3.0:
                    notes.append("%s: no room behind the ego car, actor dropped" % a["id"])
                    continue
                agents[vid] = {"type": a["type"], "x": road.lane_point(sb, lane_k)[0], "y": road.lane_point(sb, lane_k)[1],
                               "heading": road.lane_heading(sb)}
                spec_p = {"mode": "drive", "speed_kmh": a["speed_kmh"], "already_moving": True, "avoid": True,
                          "path": [list(road.lane_point(road.end_s, lane_k))]}
            else:       # oncoming
                so = (65.0 if straight_tp_cross else S_START) + a["distance_m"] + v_a * 2
                if straight_tp_cross:
                    so = max(so, tp_cross_s + 55.0)
                so = min(so, road.end_s)
                agents[vid] = {"type": a["type"], "x": road.opposite_lane_point(so, lane_k)[0], "y": road.opposite_lane_point(so, lane_k)[1],
                               "heading": road.lane_heading(so) + 180.0}
                spec_p = {"mode": "drive", "speed_kmh": a["speed_kmh"], "already_moving": True, "avoid": False,
                          "path": [list(road.opposite_lane_point(10.0, lane_k))]}
        else:   # crossroad
            rel = a["relation"]
            if rel == "ahead":
                gapd = a["distance_m"]
                if is_ev and event["type"] != "none":
                    close = v_e - (0.0 if beh == "stopped" else (0.0 if beh == "braking" else v_a))
                    gapd = max(close * T + (CAR_LEN if event["type"] == "collision" else 12.0), 8.0)
                y = ego_start[1] + gapd
                y = min(y, road.C[1] + 90.0)
                if not is_ev and beh == "stopped" and a["lane"] != "same":
                    y = min(y, road.C[1] - road.half_j - 5.0)
                    while any(abs(y - y0) < 8.5 for y0 in placed_adj_y.get(lane_k, [])):
                        y -= 9.0
                    placed_adj_y.setdefault(lane_k, []).append(y)
                elif not is_ev and not third_party and not stop_behaviours and conflict is not None and beh == "driving":
                    y = max(y, road.C[1] + road.half_j + 8.0)
                    while any(abs(y - y0) < 8.5 for y0 in placed_adj_y.get(("far", lane_k), [])):
                        y += 9.0
                    placed_adj_y.setdefault(("far", lane_k), []).append(y)
                    y = min(y, road.C[1] + 90.0)
                agents[vid] = {"type": a["type"], "x": road.northbound_x(lane_k), "y": y, "heading": 90.0}
                if beh == "stopped" or v_a < 1.0:
                    spec_p = {"mode": "static"}
                else:
                    spec_p = {"mode": "drive", "speed_kmh": a["speed_kmh"], "already_moving": True, "avoid": False,
                              "path": [[road.northbound_x(lane_k), road.C[1] + 90.0]]}
                    if beh == "braking":
                        spec_p["stop_at_s"] = round(max(T - 1.5, 1.0), 2)
                        if is_ev:           # closed loop: brake when the ego car is about 2 s away, whichever comes first
                            spec_p["brake_trigger"] = {"who": "v1", "ttc_s": 2.0}
            elif rel == "behind":
                y = ego_start[1] - max(a["distance_m"], 10.0)
                if y < road.C[1] - 97.0:
                    notes.append("%s: no room behind the ego car, actor dropped" % a["id"])
                    continue
                agents[vid] = {"type": a["type"], "x": road.northbound_x(lane_k), "y": y, "heading": 90.0}
                spec_p = {"mode": "drive", "speed_kmh": a["speed_kmh"], "already_moving": True, "avoid": True,
                          "path": [[road.northbound_x(lane_k), road.C[1] + 90.0]]}
            elif rel == "oncoming":
                y = min(ego_start[1] + a["distance_m"] + v_a * 2, road.C[1] + 95.0)
                if not is_ev and (beh == "stopped" or v_a < 1.0):
                    y = min(max(ego_start[1] + a["distance_m"], road.C[1] + road.half_j + 8.0), road.C[1] + 95.0)
                agents[vid] = {"type": a["type"], "x": road.southbound_x(lane_k), "y": y, "heading": -90.0}
                if beh == "stopped" or v_a < 1.0:
                    spec_p = {"mode": "static"}
                else:
                    dest = [road.southbound_x(lane_k), road.C[1] - 90.0]
                    if beh == "turning_left":        # oncoming car turns left across the ego path, toward the east
                        dest = [road.C[0] + 60.0, road.eastbound_y(0) * 0 + road.C[1] - road.off(0)]
                    elif beh == "turning_right":
                        dest = [road.C[0] - 60.0, road.C[1] + road.off(0)]
                    spec_p = {"mode": "drive", "speed_kmh": a["speed_kmh"], "already_moving": True, "avoid": False, "path": [dest]}
            else:       # cross_left (from the west, heading east) / cross_right (from the east, heading west)
                left = rel == "cross_left"
                tp = third_party and (a is ev_actor or a is ev2)
                kc = (road.L - 1) if tp else 0
                ay = road.eastbound_y(kc) if left else road.westbound_y(kc)
                heading_a = 0.0 if left else 180.0
                if not is_ev and beh == "stopped":
                    step = -9.0 if left else 9.0
                    x = (road.C[0] - road.half_j - 6.0) if left else (road.C[0] + road.half_j + 6.0)
                    while any(abs(x - x0) < 8.5 for x0 in placed_adj_y.get(("cross", rel), [])):
                        x += step
                    if abs(x - road.C[0]) > 97.0:
                        continue
                    placed_adj_y.setdefault(("cross", rel), []).append(x)
                    agents[vid] = {"type": a["type"], "x": x, "y": ay, "heading": heading_a}
                    spec_p = {"mode": "static"}
                else:
                    px = (road.C[0] - 1.0) if tp else ego_x
                    t_arr = T if (is_ev and event["type"] != "none") else max(T, 2.0) + 1.6 * n_cross
                    n_cross += 1
                    if is_ev and conflict is not None:
                        if event["type"] == "near_miss":
                            t_arr = t_ego_conflict + 1.0
                        elif event["type"] == "collision":
                            t_arr = t_ego_conflict
                    s = max(v_a * t_arr, 8.0)
                    if third_party and a is ev2:
                        s = max(v_a * t_arr - 6.0, 8.0)            # the left turn arc is about 6 m long
                    s = min(s, 95.0)
                    x = (px - s) if left else (px + s)
                    x = max(min(x, road.C[0] + 97.0), road.C[0] - 97.0)
                    agents[vid] = {"type": a["type"], "x": x, "y": ay, "heading": heading_a}
                    dest = [road.C[0] + (90.0 if left else -90.0), ay]
                    if beh == "turning_right":
                        dest = [road.southbound_x(0) if left else road.northbound_x(0), road.C[1] + (-90.0 if left else 90.0)]
                    elif beh == "turning_left":
                        dest = [road.northbound_x(0) if left else road.southbound_x(0), road.C[1] + (90.0 if left else -90.0)]
                    if beh == "stopped":
                        spec_p = {"mode": "static"}
                    else:
                        spec_p = {"mode": "drive", "speed_kmh": a["speed_kmh"], "already_moving": True, "avoid": False, "path": [dest]}
        if spec_p is not None and a.get("color"):
            spec_p["color"] = a["color"]
        plan_agents[vid] = spec_p
        if is_ev and third_party:
            pass
        elif is_ev:
            pair = ["v1", vid]
            straight_cross = road.kind == "straight" and a["relation"] in ("cross_left", "cross_right")
            use_conflict = not stop_behaviours and not third_party and ((road.kind == "crossroad" and conflict is not None) or straight_cross)
            t_exp = t_ego_conflict if use_conflict else T
            if event["type"] == "collision":
                expect = {"type": "collision", "actors": pair, "time_s": round(t_exp, 2), "window_s": 1.5,
                          "contact": "side" if (spec_p or {}).get("intercept") else "any"}
            elif event["type"] == "near_miss":
                expect = {"type": "near_miss", "actors": pair, "time_s": round(t_exp, 2), "window_s": 1.5, "contact": "any",
                          "max_dist_m": 8.0 if (road.kind == "straight" or a["relation"] in ("ahead", "behind")) else 6.0}
    if third_party:
        t_third = tp_t_cross if straight_tp_cross else T
        expect = {"type": "collision" if event["type"] == "collision" else "near_miss",
                  "actors": [idmap[ev_actor["id"]], idmap[ev2["id"]]], "time_s": round(t_third, 2), "max_dist_m": 6.0,
                  "window_s": 1.5, "contact": "any"}
    if expect is None:
        expect = {"type": "none", "actors": ["v1"], "time_s": round(T, 2)}

    # ---- write files ----------------------------------------------------------------------------------
    ag_dict = {}
    for vid, a in agents.items():
        ag_dict[vid] = {"location": (round(a["x"], 2), round(-a["y"], 2), 2),
                        "rotation": (0, round(-a["heading"], 1), 0), "type": a["type"]}
    _write_text(os.path.join(out_dir, scene_id + "_scene_obj.txt"), str(ag_dict) + "\n{}\n")
    road_plan = {"centerline": [[round(x, 3), round(y, 3)] for x, y in road.frame.pts] if road.frame is not None else None,
                 "lanes": road.L, "one_way": road.one_way,
                 "edge_right": "shoulder" if has_straight_cross else spec["road"].get("edge_right", "shoulder"),
                 "edge_left": "shoulder" if has_straight_cross else spec["road"].get("edge_left", "shoulder"),
                 "shoulder_m": shoulder_m}
    plan = {"ego": "v1", "duration_s": round(T + 5.0, 1), "expect": expect, "agents": plan_agents, "idmap": idmap,
            "notes": notes, "source": "scene_builder", "road": road_plan}
    _write_json(os.path.join(out_dir, scene_id + "_motion.json"), plan)
    _write_json(os.path.join(out_dir, scene_id + "_spec.json"), spec)
    return plan
