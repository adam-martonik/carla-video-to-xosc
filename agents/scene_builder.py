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
def _extend_centerline(pts, min_length):
    """Extend a centerline along its final heading to at least `min_length` metres."""
    pts = [(float(p[0]), float(p[1])) for p in pts]
    if not min_length or len(pts) < 2:
        return pts
    total = sum(math.hypot(b[0] - a[0], b[1] - a[1]) for a, b in zip(pts, pts[1:]))
    if total >= min_length - 1e-3:
        return pts
    k = len(pts) - 2
    while k > 0 and math.hypot(pts[-1][0] - pts[k][0], pts[-1][1] - pts[k][1]) < 1e-3:
        k -= 1
    dx, dy = pts[-1][0] - pts[k][0], pts[-1][1] - pts[k][1]
    norm = math.hypot(dx, dy) or 1.0
    ux, uy = dx / norm, dy / norm
    rem = float(min_length) - total
    if len(pts) == 2:
        pts.append((pts[-1][0] + rem * ux, pts[-1][1] + rem * uy))
        return pts
    x0, y0 = pts[-1]
    d = 2.0
    while d < rem - 1e-3:
        pts.append((x0 + d * ux, y0 + d * uy))
        d += 2.0
    pts.append((x0 + rem * ux, y0 + rem * uy))
    return pts


class Road:
    """Geometry of the generated road in the net frame."""

    def __init__(self, kind, lanes, centerline=None, one_way=False, sections=None, min_length=None):
        """sections (straight roads only): [{"s0", "s1", "n", "m"}] along the road, see agents/lane_profile.py. Without them the
        road has `lanes` lanes everywhere. self.L is the widest extent (max m + n); lanes are numbered k = 0 (right-most of that
        extent) .. L - 1, so a lane keeps its index and lateral position in every section. self.Lw: lanes of the opposite direction."""
        self.kind, self.L, self.Lw = kind, lanes, lanes
        self.one_way = bool(one_way)
        self.sections = None
        if kind == "straight":
            pts = [(float(p[0]), float(p[1])) for p in centerline] if centerline else [(0.0, 0.0), (300.0, 0.0)]
            if min_length is not None:
                pts = _extend_centerline(pts, min_length)
            mx, my = min(p[0] for p in pts), min(p[1] for p in pts)
            # netconvert moves a net so that its centre line starts at x, y >= 0; start there so that no shift is left over
            self.frame = RoadFrame([(x - mx, y - my) for x, y in pts])
            self.length = self.frame.length
            self.end_s = self.length - 10.0
            self.C = None
            self.sections = [{"s0": 0.0, "s1": self.length, "n": lanes, "m": 0}]
            if sections and len(sections) > 1:
                secs = [dict(sc) for sc in sections if sc["s0"] < self.length - 1.0]
                secs[0]["s0"], secs[-1]["s1"] = 0.0, self.length
                if len(secs) > 1:
                    self.sections = secs
                    self.L = max(sc["m"] + sc["n"] for sc in secs)
        else:
            self.end_s = None
            self.frame = None
            self.arm = 100.0
            self.C = (100.0, 100.0)
            self.half_j = lanes * W + 4.0

    # lanes along a straight road ---------------------------------------------------------------------
    @property
    def varying(self):
        return bool(self.sections) and (len(self.sections) > 1 or self.sections[0]["m"] != 0)

    def section(self, s):
        for sc in self.sections:
            if s < sc["s1"]:
                return sc
        return self.sections[-1]

    def lanes_at(self, s):
        """Indices k of the lanes that exist at distance s (all of them on a crossroad)."""
        if not self.sections:
            return list(range(self.L))
        sc = self.section(s)
        return list(range(self.L - sc["m"] - sc["n"], self.L - sc["m"]))

    def has_lane(self, s, k):
        return k in self.lanes_at(s)

    def clamp_lane(self, s, k):
        """The existing lane at s nearest to lane k."""
        return min(self.lanes_at(s), key=lambda q: (abs(q - k), q))

    def right_edge_d(self, s):
        """Lateral offset of the right edge of the ego carriageway at s."""
        if not self.sections:
            return -self.L * W
        sc = self.section(s)
        return -(sc["m"] + sc["n"]) * W

    def end_point(self, k):
        """Route end for a vehicle in lane k: the same lane at the end of the road, or the nearest lane that is left there."""
        return list(self.lane_point(self.end_s, self.clamp_lane(self.end_s, k)))

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
        k = max(0, min(int(k), self.Lw - 1))
        return self.frame.to_xy(s, -lane_offset(self.Lw, k))

    def northbound_x(self, k):
        return self.C[0] + self.off(k)

    def southbound_x(self, k):
        return self.C[0] - self.off(k)


SPLIT_GAP_M = 4.0          # half length of the junction where a branch leaves
DIVERGE_LEN_M = 100.0      # a branch moves this far along the road while it pulls away sideways
DIVERGE_D_M = 10.0         # ... by this much (the strip between the ego road and the branch)
RISE_LEN_M = 100.0         # a branch reaches its full height this far after it starts to climb (6 m: ~6 % average)
RISE_START_M = 55.0        # ... which it does only once it has pulled away (about 5.7 m apart), so the
                           # two roads never meet at different heights


def _required_road_length(spec, s_ego0, v_e, T, plan_ego=None):
    """Minimum straight-road length (m) so a split/branch and replayed vehicles continue well past the simulation end, or None
    for plain specs without a branch or replay (keeping legacy 300 m roads unchanged)."""
    road = spec.get("road") or {}
    if road.get("type") != "straight":
        return None
    prof = road.get("lane_profile") or []
    has_branch = any(isinstance(sc, dict) and sc.get("branch") for sc in prof)
    replay = spec.get("replay") or []
    if not has_branch and not replay:
        return None
    ego = spec.get("ego") or {}
    pe = plan_ego or {"mode": "static" if ego.get("behavior") == "stopped" else "drive",
                      "speed_profile": ego.get("speed_profile")}
    s_ego_t = _ego_travel(pe, v_e)
    s_sim_end = s_ego0 + s_ego_t(T + 8.0)
    need = max(500.0, s_sim_end + 250.0)
    for sc in prof:
        if isinstance(sc, dict) and sc.get("branch") and sc.get("d0") is not None:
            s_b = s_ego0 + float(sc["d0"])
            need = max(need, 550.0, s_b + max(DIVERGE_LEN_M, RISE_START_M + RISE_LEN_M) + 250.0)
    for r in replay:
        max_s = max((p[1] for p in r.get("path") or []), default=0.0)
        if max_s > 0.0:
            need = max(need, min(s_ego0 + max_s + 150.0, 900.0))
    return round(need, 1)


def _smooth(x):
    x = max(0.0, min(1.0, x))
    return x * x * (3.0 - 2.0 * x)


def split_index(road):
    """Index of the first section that carries a branch, or None."""
    for i, sc in enumerate(road.sections or []):
        if i > 0 and sc.get("branch") and SPLIT_GAP_M + 1.0 < sc["s0"] < road.length - 40.0:
            return i
    return None


def _pieces(road):
    """Road pieces of the ego direction: [(edge id, s0, s1, sections)], one piece without a split, before/after it with one.
    Each piece is one edge with the widest lanes of its sections; lane changes inside a piece go into the OpenDRIVE map."""
    b = split_index(road)
    if b is None:
        return [("eastbound", 0.0, road.length, road.sections)]
    s_b = road.sections[b]["s0"]
    return [("eastbound", 0.0, s_b - SPLIT_GAP_M, road.sections[:b]), ("eastbound_after", s_b + SPLIT_GAP_M, road.length, road.sections[b:])]


def _shape(frame, s0, s1, dfun, zfun=None, step=4.0):
    """'x,y[,z] ...' of the line at lateral offset dfun(s) (and height zfun(s)) from s0 to s1, keeping the frame's vertices."""
    ss = sorted(set([s0, s1] + [c for c in frame.cum if s0 < c < s1] + [s0 + step * i for i in range(1, int((s1 - s0) / step))]))
    pts = []
    for s in ss:
        x, y = frame.to_xy(s, dfun(s))
        pts.append("%.3f,%.3f,%.3f" % (x, y, zfun(s)) if zfun else "%.3f,%.3f" % (x, y))
    return " ".join(pts)


def _split_net(road):
    """Nodes, edges and connections of a straight road where some lanes leave as a branch of their own (a fork or a flyover).
    Before the split one edge holds every lane; after it the ego road goes on along the road frame and the branch pulls away
    to its side by DIVERGE_D_M and climbs to rise_m. Every lane connects to exactly one lane after the split (no merging),
    so the junction is clean. With a left branch on a two-way road the opposite carriageway follows the branch (the two are
    the main road), so the strip between them stays."""
    fr, secs = road.frame, road.sections
    b = split_index(road)
    a_sec, b_sec = secs[b - 1], secs[b]
    br = b_sec["branch"]
    s_b = b_sec["s0"]
    left = br["side"] == "left"
    leave = list(range(a_sec["m"], b_sec["m"])) if left else list(range(b_sec["m"] + b_sec["n"], a_sec["m"] + a_sec["n"]))
    rise = float(br.get("rise_m", 0.0))
    f = lambda s: DIVERGE_D_M * _smooth((s - s_b) / DIVERGE_LEN_M) * (1.0 if left else -1.0)
    z = lambda s: rise * _smooth((s - s_b - RISE_START_M) / RISE_LEN_M)
    nodes = ['<node id="west" x="%.3f" y="%.3f" type="priority"/>' % fr.pts[0]]
    x, y = fr.to_xy(s_b, -(a_sec["m"] + 0.5 * a_sec["n"]) * W)
    nodes.append('<node id="split" x="%.3f" y="%.3f" type="priority"/>' % (x, y))
    edges, cons, ext = [], [], {}
    for eid, s0, s1, ps in _pieces(road):
        mP = min(sc["m"] for sc in ps)
        eP = max(sc["m"] + sc["n"] for sc in ps)
        ext[eid] = (mP, eP)
        frm, to = ("west", "split") if eid == "eastbound" else ("split", "east")
        edges.append('<edge id="%s" from="%s" to="%s" numLanes="%d" speed="30.0" width="%.1f" spreadType="right" shape="%s"/>'
                     % (eid, frm, to, eP - mP, W, _shape(fr, s0, s1, lambda s, mP=mP: -mP * W)))
    x, y = fr.to_xy(fr.length, -(ext["eastbound_after"][0]) * W)
    nodes.append('<node id="east" x="%.3f" y="%.3f" type="priority"/>' % (x, y))
    j0 = min(leave)
    bshape = _shape(fr, s_b + SPLIT_GAP_M, fr.length, lambda s: -j0 * W + f(s), z)
    bx, by, bz = map(float, bshape.split()[-1].split(","))
    nodes.append('<node id="branch_end" x="%.3f" y="%.3f" z="%.3f" type="priority"/>' % (bx, by, bz))
    edges.append('<edge id="branch" from="split" to="branch_end" numLanes="%d" speed="30.0" width="%.1f" spreadType="right" shape="%s"/>'
                 % (len(leave), W, bshape))
    local = lambda eid, j: (ext[eid][1] - 1) - j                # SUMO lane 0 is the right-most lane of an edge
    mA, eA = ext["eastbound"]
    for j in range(mA, eA):
        if j in leave:
            cons.append('<connection from="eastbound" to="branch" fromLane="%d" toLane="%d"/>' % (local("eastbound", j), max(leave) - j))
        elif ext["eastbound_after"][0] <= j < ext["eastbound_after"][1]:
            cons.append('<connection from="eastbound" to="eastbound_after" fromLane="%d" toLane="%d"/>'
                        % (local("eastbound", j), local("eastbound_after", j)))
    if not road.one_way:
        if left:                                                # the opposite carriageway follows the branch
            g = lambda s: max(f(s), 0.0)
            hz = lambda s: z(s) if s > s_b else 0.0
            wshape = _shape(fr, 0.0, fr.length, g, hz).split()
            wshape = " ".join(reversed(wshape))
            (ex, ey, ez), (sx, sy, sz) = [tuple(map(float, p.split(","))) for p in (wshape.split()[0], wshape.split()[-1])]
            nodes.append('<node id="wb_from" x="%.3f" y="%.3f" z="%.3f" type="priority"/>' % (ex, ey, ez))
            nodes.append('<node id="wb_to" x="%.3f" y="%.3f" z="%.3f" type="priority"/>' % (sx, sy, sz))
            edges.append('<edge id="westbound" from="wb_from" to="wb_to" numLanes="%d" speed="30.0" width="%.1f" spreadType="right" shape="%s"/>'
                         % (road.Lw, W, wshape))
        else:
            rshape = " ".join("%.3f,%.3f" % p for p in reversed(fr.pts))
            nodes.append('<node id="wb_from" x="%.3f" y="%.3f" type="priority"/>' % fr.pts[-1])
            edges.append('<edge id="westbound" from="wb_from" to="west" numLanes="%d" speed="30.0" width="%.1f" spreadType="right" shape="%s"/>'
                         % (road.Lw, W, rshape))
    return nodes, edges, cons


def xodr_pieces(road):
    """The road pieces whose lanes the OpenDRIVE post-processing rewrites (map_builder.apply_lane_sections): the pieces whose
    lane count changes inside them, with the sections' m relative to the piece's own left-most lane."""
    out = []
    for eid, s0, s1, ps in _pieces(road):
        mP = min(sc["m"] for sc in ps)
        if len(ps) < 2 and ps[0]["m"] == mP:
            continue
        secs = []
        for sc in ps:
            secs.append({"s0": round(max(sc["s0"], s0), 2), "s1": round(min(sc["s1"], s1), 2), "n": sc["n"], "m": sc["m"] - mP})
        out.append({"road": eid, "s0": round(s0, 2), "s1": round(s1, 2), "sections": secs})
    return out


def write_net(road, out_dir, scene_id, signals):
    nod = os.path.join(out_dir, scene_id + ".nod.xml")
    edg = os.path.join(out_dir, scene_id + ".edg.xml")
    net = os.path.join(out_dir, scene_id + ".net.xml")
    L = road.L
    cons = []
    if road.kind == "straight" and split_index(road) is not None:
        nodes, edges, cons = _split_net(road)
    elif road.kind == "straight":
        # a lane count that changes along the road goes into the OpenDRIVE map afterwards (map_builder.apply_lane_sections): the
        # net has the widest number of lanes, so netconvert needs no junction where a lane ends
        pts = road.frame.pts
        nodes = ['<node id="west" x="%.3f" y="%.3f" type="priority"/>' % pts[0], '<node id="east" x="%.3f" y="%.3f" type="priority"/>' % pts[-1]]
        shape = " ".join("%.3f,%.3f" % p for p in pts)
        rshape = " ".join("%.3f,%.3f" % p for p in reversed(pts))
        # spreadType right: the lanes lie to the right of the shape, so lane k is exactly at lane_offset(L, k) from the centre line
        edges = ['<edge id="eastbound" from="west" to="east" numLanes="%d" speed="30.0" width="%.1f" spreadType="right" shape="%s"/>' % (L, W, shape)]
        if not road.one_way:
            edges.append('<edge id="westbound" from="east" to="west" numLanes="%d" speed="30.0" width="%.1f" spreadType="right" shape="%s"/>' % (road.Lw, W, rshape))
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
    cmd = [find_netconvert(), "--node-files=" + nod, "--edge-files=" + edg, "--default.spreadtype", "center",
           "--no-turnarounds", "--output-file=" + net]
    if cons:
        con = os.path.join(out_dir, scene_id + ".con.xml")
        _write_text(con, '<?xml version="1.0" encoding="UTF-8"?>\n<connections>\n  ' + "\n  ".join(cons) + "\n</connections>\n")
        cmd.append("--connection-files=" + con)
    res = subprocess.run(cmd, capture_output=True, text=True)
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
        if not road.has_lane(S_START, lane_idx + 1):
            notes.append("traffic: the ego car is in the left-most lane, no left lane for %d vehicle(s)" % n["left_lane"])
        else:
            for i in range(n["left_lane"]):
                add("ahead", "left", base + 12.0 * i, v_t_kmh)
    if n["right_lane"]:
        if not road.has_lane(S_START, lane_idx - 1):
            notes.append("traffic: the ego car is in the right-most lane, no right lane for %d vehicle(s)" % n["right_lane"])
        else:
            for i in range(n["right_lane"]):
                add("ahead", "right", base + 12.0 * i, v_t_kmh)
    for i in range(n["oncoming"]):
        add("oncoming", "left" if (road.Lw > 1 and i % 2) else "same", 20.0 + 15.0 * i, 40.0)
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


# ----------------------------------------------------------------------------------------- measured vehicles
REPLAY_GAP_M = 5.5           # a replayed vehicle is not spawned closer than this (centre to centre) to one already placed


def _interp(t, ts, vs):
    import numpy as np
    return float(np.interp(t, ts, vs))


CORRIDOR_HALF_D = 2.0        # a replayed vehicle closer than this across the road ...
CORRIDOR_HALF_S = 8.0        # ... and this along it to where a crash vehicle will be is in its way
VEHICLE_HALF_LEN = {"car": 2.4, "suv": 2.5, "van": 2.8, "truck": 4.8, "motorcycle": 1.1, "bike": 0.9}


def _in_corridor(times, ss, ds, corridors):
    """Whether a path (sampled at times) comes within half_d / CORRIDOR_HALF_S of a crash vehicle's planned position."""
    for c in corridors:
        d_c, s_c, t_end = c[0], c[1], c[2]
        t_start = c[3] if len(c) > 3 else 0.0
        half_d = c[4] if len(c) > 4 else CORRIDOR_HALF_D
        for t, s_, d_ in zip(times, ss, ds):
            if t_start <= t <= t_end and abs(d_ - d_c) < half_d and abs(s_ - s_c(t)) < CORRIDOR_HALF_S:
                return True
    return False


def enforce_replay_headway(tracks, static_obstacles=(), gap_margin=3.5):
    """Integrate replayed longitudinal trajectories forward under physical speed and accel limits so no vehicle
    teleports, exceeds plausible speed, or penetrates a same-lane lead or parked obstacle."""
    ACC_MAX, DEC_COMF, DEC_EMERG, V_MAX_ABS = 3.0, 4.5, 7.0, 30.0
    for is_back in (False, True):
        group = [tr for tr in tracks if bool(tr.get("back")) == is_back]
        if not group:
            continue
        dir_sign = -1.0 if is_back else 1.0

        def _first_s(tr):
            for sv in tr["ss"]:
                if sv is not None:
                    return sv if is_back else -sv
            return 0.0

        group.sort(key=_first_s)
        leads = []
        for cur in group:
            ss, ds, ts = cur["ss"], cur["ds"], cur["ts"]
            n_t = len(ts)
            cur_v = [0.0] * n_t
            active = [k for k in range(n_t) if ss[k] is not None and ds[k] is not None]
            if not active:
                leads.append((cur, cur_v))
                continue
            s_meas = [ss[k] for k in range(n_t)]
            step_spds = []
            for idx_p, idx_c in zip(active, active[1:]):
                dt_raw = max(ts[idx_c] - ts[idx_p], 1e-3)
                sp_raw = dir_sign * (s_meas[idx_c] - s_meas[idx_p]) / dt_raw
                if 0.0 <= sp_raw <= 40.0:
                    step_spds.append(sp_raw)
            v_cap = min(max((max(step_spds) * 1.15 if step_spds else 15.0), 6.0), V_MAX_ABS)
            k0 = active[0]
            s_cur = s_meas[k0]

            def _limits_at(k_idx, s_pos, d_pos):
                s_lim = 1e9 if not is_back else -1e9
                v_safe_min = v_cap
                v_obs_at_lim = v_cap
                min_gap = 1e9
                for ld_tr, ld_v in leads:
                    ls, ld_lat = ld_tr["ss"][k_idx], ld_tr["ds"][k_idx]
                    if ls is None or ld_lat is None or abs(ld_lat - d_pos) >= 2.4:
                        continue
                    need = VEHICLE_HALF_LEN.get(cur["type"], 2.4) + VEHICLE_HALF_LEN.get(ld_tr["type"], 2.4) + gap_margin
                    if dir_sign * (ls - s_pos) > -0.5 * need:
                        cand_lim = ls - dir_sign * need
                        if (not is_back and cand_lim < s_lim) or (is_back and cand_lim > s_lim):
                            s_lim = cand_lim
                            v_obs_at_lim = ld_v[k_idx]
                        gap = dir_sign * (ls - s_pos) - need
                        min_gap = min(min_gap, gap)
                        v_s = (ld_v[k_idx] + math.sqrt(2.0 * 3.0 * gap)) if gap > 0.0 else 0.0
                        v_safe_min = min(v_safe_min, max(0.0, v_s))
                for st_s, st_d, st_type in static_obstacles:
                    if abs(st_d - d_pos) >= 2.4:
                        continue
                    need = VEHICLE_HALF_LEN.get(cur["type"], 2.4) + VEHICLE_HALF_LEN.get(st_type, 2.4) + gap_margin
                    if dir_sign * (st_s - s_pos) > -0.5 * need:
                        cand_lim = st_s - dir_sign * need
                        if (not is_back and cand_lim < s_lim) or (is_back and cand_lim > s_lim):
                            s_lim = cand_lim
                            v_obs_at_lim = 0.0
                        gap = dir_sign * (st_s - s_pos) - need
                        min_gap = min(min_gap, gap)
                        v_s = math.sqrt(2.0 * 3.0 * gap) if gap > 0.0 else 0.0
                        v_safe_min = min(v_safe_min, v_s)
                return s_lim, v_safe_min, v_obs_at_lim, min_gap

            s_lim0, v_safe0, v_obs0, _ = _limits_at(k0, s_cur, ds[k0])
            if (not is_back and s_cur >= s_lim0) or (is_back and s_cur <= s_lim0):
                s_cur = s_lim0
                v_cur = min(step_spds[0] if step_spds else 0.0, v_obs0)
            else:
                v_cur = max(0.0, min(v_cap, v_safe0, step_spds[0] if step_spds else 0.0))
            ss[k0] = s_cur
            cur_v[k0] = v_cur
            prev_k = k0
            for k in active[1:]:
                dt = max(ts[k] - ts[prev_k], 1e-3)
                dk = ds[k]
                v_ref = max(0.0, min(v_cap, dir_sign * (s_meas[k] - s_meas[prev_k]) / dt))
                err = dir_sign * (s_meas[k] - s_cur)
                v_des = max(0.0, min(v_cap, v_ref + max(-3.0, min(3.0, 0.5 * err))))
                s_lim, v_safe_min, v_obs_at_lim, min_gap = _limits_at(k, s_cur, dk)
                v_target = min(v_des, v_safe_min)
                dec_lim = DEC_EMERG if min_gap < 4.0 else DEC_COMF
                v_cur = max(0.0, min(v_cap, max(v_cur - dec_lim * dt, min(v_cur + ACC_MAX * dt, v_target))))
                s_new = s_cur + dir_sign * v_cur * dt
                if not is_back:
                    if s_new > s_lim:
                        s_new = s_lim
                        v_cur = min(v_cur, max(0.0, v_obs_at_lim))
                    s_new = max(s_new, s_cur)
                else:
                    if s_new < s_lim:
                        s_new = s_lim
                        v_cur = min(v_cur, max(0.0, v_obs_at_lim))
                    s_new = min(s_new, s_cur)
                ss[k] = s_new
                cur_v[k] = v_cur
                s_cur = s_new
                prev_k = k
            leads.append((cur, cur_v))
    return tracks


def _ego_travel(plan_ego, v_e):
    """s(t): distance the ego car drives in the simulation, from its speed profile or its constant speed."""
    prof = plan_ego.get("speed_profile")
    if not prof or plan_ego.get("mode") != "drive":
        return (lambda t: v_e * t) if plan_ego.get("mode") == "drive" else (lambda t: 0.0)
    ts = [p[0] for p in prof]
    vs = [p[1] / 3.6 for p in prof]

    def s(t, dt=0.1):
        n = int(max(t, 0.0) / dt)
        total = sum(_interp((k + 0.5) * dt, ts, vs) * dt for k in range(n))
        return total + _interp(n * dt, ts, vs) * (max(t, 0.0) - n * dt)
    return s


def lane_centres(road, s, oncoming=False):
    """Lateral offsets (road frame, + left) of the centres of the lanes that exist at distance s: the ego direction's lanes
    there and, after a fork, the branch's lanes pulled away to their side; the opposite carriageway's lanes for oncoming cars."""
    if oncoming:
        return [-lane_offset(road.Lw, k) for k in range(road.Lw)] if not road.one_way else []
    out = [lane_offset(road.L, k) for k in road.lanes_at(s)]
    b = split_index(road)
    if b is not None and s > road.sections[b]["s0"]:
        a_sec, b_sec = road.sections[b - 1], road.sections[b]
        br = b_sec["branch"]
        left = br["side"] == "left"
        leave = range(a_sec["m"], b_sec["m"]) if left else range(b_sec["m"] + b_sec["n"], a_sec["m"] + a_sec["n"])
        f = DIVERGE_D_M * _smooth((s - b_sec["s0"]) / DIVERGE_LEN_M) * (1.0 if left else -1.0)
        out += [-(j + 0.5) * W + f for j in leave]
    return out


def snap_lane(road, s, lat, oncoming=False):
    """A measured lateral position moved to the centre of the nearest lane that exists at s, so that a replayed car neither
    straddles two lanes nor drives where its lane has ended (a measured lane change still happens where the position crosses
    over to the other lane)."""
    cents = lane_centres(road, s, oncoming)
    return min(cents, key=lambda c: abs(c - lat)) if cents else lat


def _on_carriageway(road, s, lat, kind, margin=1.0, shoulder=3.5):
    """Whether a measured vehicle at lateral offset `lat` (road frame, + left) at distance s stands on a road of the scene:
    same direction: on the ego carriageway's lanes (or a diverging branch) there; oncoming: on the opposite carriageway;
    parked: on either or on the outer shoulder beside them. A car on a frontage road or in a yard beyond is not."""
    if kind == "same":
        cents = lane_centres(road, s, oncoming=False)
        return bool(cents) and min(abs(lat - c) for c in cents) <= 0.5 * W + max(margin, 1.75)
    if kind == "oncoming":
        cents = lane_centres(road, s, oncoming=True)
        return bool(cents) and min(abs(lat - c) for c in cents) <= 0.5 * W + margin
    right = road.right_edge_d(s)
    left = -road.section(s)["m"] * W if road.sections else 0.0
    opp = 0.0 if road.one_way else road.Lw * W
    if road.one_way:
        return right - shoulder - margin <= lat <= left + shoulder + margin
    return right - shoulder - margin <= lat <= max(opp, left) + 1.5


def _smooth_lane_offsets(road, s_list, lat_list, back, hyst=0.7, trans_m=25.0):
    """Snap lateral offsets along a replayed path to lane centres with hysteresis, and ramp any lane switch smoothly over
    `trans_m` metres so that measurement noise never causes abrupt zig-zags."""
    if not s_list:
        return []
    snapped = []
    cur_c = snap_lane(road, s_list[0], lat_list[0], back)
    for s_i, lat_i in zip(s_list, lat_list):
        cents = lane_centres(road, s_i, back)
        if not cents:
            snapped.append(lat_i)
            continue
        stay_c = min(cents, key=lambda c: abs(c - cur_c))
        near_c = min(cents, key=lambda c: abs(c - lat_i))
        if near_c != stay_c and abs(lat_i - near_c) + hyst < abs(lat_i - stay_c):
            cur_c = near_c
        else:
            cur_c = stay_c
        snapped.append(cur_c)
    out = list(snapped)
    for i in range(1, len(snapped)):
        if abs(snapped[i] - snapped[i - 1]) > 1.5:
            c_old, c_new = snapped[i - 1], snapped[i]
            s_mid = s_list[i]
            s_lo = s_mid - 0.5 * trans_m if s_mid - s_list[0] >= 0.5 * trans_m else s_mid
            s_hi = s_lo + trans_m
            for j, s_j in enumerate(s_list):
                if s_lo <= s_j <= s_hi:
                    f = _smooth((s_j - s_lo) / max(s_hi - s_lo, 1e-3))
                    out[j] = c_old + (c_new - c_old) * f
    return out


def _replay_agents(replay, road, lane_idx, s_ego0, agents, plan_agents, idmap, corridors=(), duration_s=15.0):
    """Vehicles measured in the video (agents/vehicle_tracks via spec.replay) replayed along their paths: path [t, s, d] in
    simulation time, metres along the road from the ego start and metres left of the ego lane centre. A vehicle first seen after
    the start is put where its first speed says it was at t = 0 (or spawned when it enters the road via spawn_at_s); it drives its
    measured path and speed and then keeps its lane to the end of the road (oncoming: to its start), with guaranteed headway
    clearance against lead and parked vehicles via kinematic_track. Returns notes."""
    from agents.scene_spec import color_rgb
    notes = []
    d_ego = lane_offset(road.L, lane_idx)
    placed_sd = [(*road.frame.to_sd(a["x"], a["y"]), 0.0, a.get("type", "car")) for a in agents.values()]
    static_obs = [(*road.frame.to_sd(a["x"], a["y"]), a.get("type", "car"))
                  for vid_, a in agents.items() if (plan_agents.get(vid_) or {}).get("mode") == "static"]
    n = len(agents) + 1
    kin_candidates = []
    for r in replay:
        path = [p for p in r["path"] if p[0] >= -0.5]
        if len(path) < 2:
            continue
        ts, ss, ds = [p[0] for p in path], [p[1] for p in path], [p[2] for p in path]
        back = r["kind"] == "oncoming" or (r["kind"] != "parked" and ss[-1] < ss[0])
        sp = [p for p in r.get("speed") or [] if p[0] >= 0.0]
        v_first = (ss[1] - ss[0]) / max(ts[1] - ts[0], 1e-3)
        v_ext = ((-1.0 if back else 1.0) * (sp[0][1] / 3.6)) if sp else v_first
        v_ext = max(-28.0, min(28.0, v_ext))
        if r["kind"] == "parked":
            s0, d0 = ss[0], ds[0]
        elif ts[0] > 0.0:
            s0, d0 = ss[0] - v_ext * ts[0], ds[0]
        else:
            s0, d0 = _interp(0.0, ts, ss), _interp(0.0, ts, ds)
        s_road = s_ego0 + s0
        s_obs0 = s_ego0 + ss[0]
        spawn_t = 0.0
        spawn_at_s = None
        vtype_r = r["type"] if r["type"] in VEHICLE_TYPES else "car"
        if corridors:
            v_last = (ss[-1] - ss[-2]) / max(ts[-1] - ts[-2], 1e-3)
            t_obs0 = max(ts[0], 0.0)
            if r["kind"] == "parked":
                obs_t, obs_s, obs_d = [k * 0.5 for k in range(40)], [s0] * 40, [d0] * 40
            elif ts[-1] >= t_obs0:
                n_obs = max(int((ts[-1] - t_obs0) / 0.25) + 1, 2)
                in_t = [t_obs0 + (ts[-1] - t_obs0) * k / (n_obs - 1) for k in range(n_obs)]
                post = [ts[-1] + 0.25 * k for k in range(1, 41)]
                obs_t = in_t + post
                obs_s = [_interp(t_, ts, ss) for t_ in in_t] + [ss[-1] + v_last * (t_ - ts[-1]) for t_ in post]
                obs_d = [_interp(t_, ts, ds) for t_ in in_t] + [ds[-1]] * len(post)
            else:
                post = [ts[-1] + 0.25 * k for k in range(1, 41)]
                obs_t = post
                obs_s = [ss[-1] + v_last * (t_ - ts[-1]) for t_ in post]
                obs_d = [ds[-1]] * len(post)
            if _in_corridor(obs_t, obs_s, obs_d, corridors):
                notes.append("%s: in the way of a crash vehicle before the crash, not replayed" % r["id"])
                continue
        if r["kind"] != "parked" and ts[0] > 0.0 and 3.0 <= s_road <= road.length - 5.0:
            def _t0_slot_ok(s0_try, v_try):
                sr = s_ego0 + s0_try
                if not 3.0 <= sr <= road.length - 5.0:
                    return False
                lat_t = d_ego + d0
                if not _on_carriageway(road, sr, lat_t, r["kind"]):
                    return False
                lat_snap = snap_lane(road, sr, lat_t, back)
                for ps, pd, pt, ptype in placed_sd:
                    need_gap = VEHICLE_HALF_LEN.get(vtype_r, 2.4) + VEHICLE_HALF_LEN.get(ptype, 2.4) + 3.5
                    if abs(pt) < 1.5 and abs(sr - ps) < need_gap and abs(lat_snap - pd) < 2.4:
                        return False
                if corridors:
                    pre_t = [k * 0.25 for k in range(int(ts[0] / 0.25) + 1)]
                    pre_s = [s0_try + v_try * t_ for t_ in pre_t]
                    pre_d = [d0] * len(pre_t)
                    if _in_corridor(pre_t, pre_s, pre_d, corridors):
                        return False
                return True

            if not _t0_slot_ok(s0, v_ext):
                for mag in range(2, 50, 2):
                    found = False
                    for delta in (-mag, mag):
                        s0_try = s0 + delta
                        v_try = (ss[0] - s0_try) / ts[0]
                        v_fwd = -v_try if back else v_try
                        if v_fwd >= 1.5 and abs(v_try) <= 35.0 and _t0_slot_ok(s0_try, v_try):
                            s0, v_ext, s_road = s0_try, v_try, s_ego0 + s0_try
                            found = True
                            break
                    if found:
                        break
        if not 3.0 <= s_road <= road.length - 5.0:
            if r["kind"] != "parked" and ts[0] > 0.0 and 3.0 <= s_obs0 <= road.length - 5.0:
                if s_road < 3.0 and not back and v_ext > 0.5:
                    s_enter = 5.0
                    t_enter = (s_enter - s_road) / v_ext
                    if 0.0 < t_enter < ts[0]:
                        spawn_at_s = round(t_enter, 2)
                        spawn_t = spawn_at_s
                        s_road = s_enter
                        s0 = s_enter - s_ego0
                elif s_road > road.length - 5.0 and back and v_ext < -0.5:
                    s_enter = road.length - 8.0
                    t_enter = (s_road - s_enter) / abs(v_ext)
                    if 0.0 < t_enter < ts[0]:
                        spawn_at_s = round(t_enter, 2)
                        spawn_t = spawn_at_s
                        s_road = s_enter
                        s0 = s_enter - s_ego0
            if spawn_at_s is None and not 3.0 <= s_road <= road.length - 5.0:
                notes.append("%s: measured start outside the road, not replayed" % r["id"])
                continue
        if corridors and r["kind"] != "parked" and ts[0] > spawn_t:
            pre = [spawn_t + k * 0.25 for k in range(int(max(ts[0] - spawn_t, 0.0) / 0.25) + 1)]
            if _in_corridor(pre, [s0 + v_ext * (t_ - spawn_t) for t_ in pre], [d0] * len(pre), corridors):
                notes.append("%s: in the way of a crash vehicle before the crash, not replayed" % r["id"])
                continue
        lat = d_ego + d0
        if not _on_carriageway(road, s_road, lat, r["kind"]):
            notes.append("%s: off the roads of the scene (%.1f m across), not replayed" % (r["id"], lat))
            continue
        if r["kind"] == "parked":
            opp_edge = 0.0 if road.one_way else road.Lw * W
            if road.right_edge_d(s_road) <= lat <= opp_edge:
                on_opp = lat > 0.0 and not road.one_way
                lat0 = snap_lane(road, s_road, lat, oncoming=on_opp)
                heading = road.lane_heading(s_road) + (180.0 if on_opp else 0.0)
            else:
                lat0 = lat
                heading = road.lane_heading(s_road)
        else:
            lat0 = snap_lane(road, s_road, lat, back)
            heading = road.lane_heading(s_road) + (180.0 if back else 0.0)
        too_close = False
        for ps, pd, pt, ptype in placed_sd:
            need_gap = VEHICLE_HALF_LEN.get(vtype_r, 2.4) + VEHICLE_HALF_LEN.get(ptype, 2.4) + 3.5
            if abs(spawn_t - pt) < 1.5 and abs(s_road - ps) < need_gap and abs(lat0 - pd) < 2.4:
                too_close = True
                break
        if too_close:
            notes.append("%s: too close to a vehicle already placed, not replayed" % r["id"])
            continue
        x0, y0 = road.frame.to_xy(s_road, lat0)
        vid = "v%d" % n
        n += 1
        agents[vid] = {"type": vtype_r, "x": x0, "y": y0, "heading": heading}
        placed_sd.append((s_road, lat0, spawn_t, vtype_r))
        idmap[r["id"]] = vid
        if r["kind"] == "parked":
            plan = {"mode": "static"}
            static_obs.append((s_road, lat0, vtype_r))
        else:
            pts = [(s0, d0)] + [(s_, d_) for t_, s_, d_ in zip(ts, ss, ds) if t_ > spawn_t]
            s_end, d_end = pts[-1]
            step = -4.0 if back else 4.0
            s_c = s_end
            while (s_c + step >= 3.0 - s_ego0) if back else (s_c + step <= road.end_s - s_ego0):
                s_c += step
                pts.append((s_c, d_end))
            dense = [pts[0]]                       # a point every 4 m along the road: a long chord would cut a bend
            for (sa, da), (sb, db) in zip(pts, pts[1:]):
                n_in = max(int(math.ceil(abs(sb - sa) / 4.0)), 1)
                dense += [(sa + (sb - sa) * i / n_in, da + (db - da) * i / n_in) for i in range(1, n_in + 1)]
            sm_lats = _smooth_lane_offsets(road, [s_ego0 + s_ for s_, _ in dense], [d_ego + d_ for _, d_ in dense], back)
            poly = [list(road.frame.to_xy(s_ego0 + s_, lat_i)) for (s_, _), lat_i in zip(dense, sm_lats)]
            v0 = abs(v_ext) * 3.6 if ts[0] > spawn_t or not sp else sp[0][1]
            prof = [[spawn_t, round(v0, 1)]] + [p for p in sp if p[0] > spawn_t]
            v0 = float(v0)
            plan = {"mode": "drive", "speed_kmh": round(v0, 1), "already_moving": bool(v0 > 3.0), "avoid": True,
                    "polyline": [[round(x, 2), round(y, 2)] for x, y in poly], "speed_profile": prof}
            if spawn_at_s is not None and spawn_at_s > 0.0:
                plan["spawn_at_s"] = spawn_at_s
            v_last_obs = (ss[-1] - ss[-2]) / max(ts[-1] - ts[-2], 1e-3)
            v_post = ((-1.0 if back else 1.0) * max(sp[-1][1] / 3.6, 2.0)) if sp else v_last_obs
            v_post = max(-28.0, min(28.0, v_post))
            kin_candidates.append({
                "vid": vid, "rid": r["id"], "type": vtype_r, "back": back, "spawn_t": spawn_t,
                "ts_raw": ts, "ss_raw": ss, "ds_raw": ds, "s0": s0, "d0": d0,
                "v_ext": v_ext, "v_post": v_post,
            })
        if r.get("color"):
            plan["color"] = color_rgb(r["color"])
        plan["replay"] = r["id"]
        plan_agents[vid] = plan

    if kin_candidates:
        ts_grid = [round(0.2 * i, 2) for i in range(int(round((duration_s + 3.0) / 0.2)) + 1)]
        b_idx = split_index(road)
        br_sec = road.sections[b_idx] if b_idx is not None else None
        br_prev = road.sections[b_idx - 1] if b_idx is not None else None
        br_left = bool(br_sec and br_sec["branch"]["side"] == "left")
        br_edge_d = (-br_sec["m"] * W) if (br_sec and br_left) else (-(br_sec["m"] + br_sec["n"]) * W if br_sec else None)
        kin_tracks = []
        for kc in kin_candidates:
            ts_r, ss_r, ds_r = kc["ts_raw"], kc["ss_raw"], kc["ds_raw"]
            d_span = (max(ds_r) - min(ds_r)) if ds_r else 0.0
            on_branch = False
            if br_sec is not None and ds_r and ss_r:
                lats_r = [d_ego + dd for dd in ds_r]
                if br_left and any(lt >= br_edge_d - 0.5 * W for lt in lats_r):
                    on_branch = True
                elif (not br_left) and any(lt <= br_edge_d + 0.5 * W for lt in lats_r):
                    on_branch = True
            d_flat = sorted(ds_r)[len(ds_r) // 2] if (ds_r and d_span < 2.8 and not on_branch) else None
            s_arr, d_arr = [None] * len(ts_grid), [None] * len(ts_grid)
            act_idx, act_s, act_d = [], [], []
            for idx_g, tg in enumerate(ts_grid):
                if tg < kc["spawn_t"] - 1e-6:
                    continue
                if tg <= ts_r[0]:
                    sg = s_ego0 + kc["s0"] + kc["v_ext"] * (tg - kc["spawn_t"])
                    dg = d_ego + (d_flat if d_flat is not None else kc["d0"])
                elif tg <= ts_r[-1]:
                    sg = s_ego0 + _interp(tg, ts_r, ss_r)
                    dg = d_ego + (d_flat if d_flat is not None else _interp(tg, ts_r, ds_r))
                else:
                    sg = s_ego0 + ss_r[-1] + kc["v_post"] * (tg - ts_r[-1])
                    dg = d_ego + (d_flat if d_flat is not None else ds_r[-1])
                if 3.0 <= sg <= road.end_s:
                    act_idx.append(idx_g)
                    act_s.append(sg)
                    act_d.append(dg)
            if act_idx:
                sm_d = _smooth_lane_offsets(road, act_s, act_d, kc["back"])
                for idx_g, sg, dg in zip(act_idx, act_s, sm_d):
                    s_arr[idx_g] = sg
                    d_arr[idx_g] = dg
            kin_tracks.append({"id": kc["rid"], "vid": kc["vid"], "type": kc["type"], "back": kc["back"],
                               "ts": ts_grid, "ss": s_arr, "ds": d_arr})
        enforce_replay_headway(kin_tracks, static_obstacles=static_obs)
        for kt_d in kin_tracks:
            ts_k, ss_k, ds_k = kt_d["ts"], kt_d["ss"], kt_d["ds"]
            act_k = [i for i in range(len(ts_k)) if ss_k[i] is not None and ds_k[i] is not None]
            if len(act_k) >= 2:
                prev_t, prev_s, d_cur = ts_k[act_k[0]], ss_k[act_k[0]], ds_k[act_k[0]]
                for idx_k in act_k[1:]:
                    dt_k = max(ts_k[idx_k] - prev_t, 1e-3)
                    step_s = abs(ss_k[idx_k] - prev_s)
                    max_dd = 0.0 if step_s < 0.4 else min(1.8 * dt_k, 0.22 * step_s)
                    d_cur = d_cur + max(-max_dd, min(max_dd, ds_k[idx_k] - d_cur))
                    ds_k[idx_k] = d_cur
                    prev_t, prev_s = ts_k[idx_k], ss_k[idx_k]
            vid, rid = kt_d["vid"], kt_d["id"]
            pts_ok = [(t_g, s_g, d_g) for t_g, s_g, d_g in zip(ts_k, ss_k, ds_k)
                      if s_g is not None and d_g is not None]
            if len(pts_ok) < 2:
                notes.append("%s: blocked outside road after headway adjustment, not replayed" % rid)
                agents.pop(vid, None)
                plan_agents.pop(vid, None)
                idmap.pop(rid, None)
                continue
            if corridors and _in_corridor([p[0] for p in pts_ok], [p[1] - s_ego0 for p in pts_ok],
                                          [p[2] - d_ego for p in pts_ok], corridors):
                notes.append("%s: in the way of a crash vehicle before the crash, not replayed" % rid)
                agents.pop(vid, None)
                plan_agents.pop(vid, None)
                idmap.pop(rid, None)
                continue
            xys = [road.frame.to_xy(p[1], p[2]) for p in pts_ok]
            _wrap = lambda a: ((a + 180.0) % 360.0) - 180.0
            kt_rows = []
            for j, (t_g, s_g, d_g) in enumerate(pts_ok):
                x_j, y_j = xys[j]
                if j + 1 < len(pts_ok):
                    dt_j = max(pts_ok[j + 1][0] - t_g, 1e-3)
                    dx_j, dy_j = xys[j + 1][0] - x_j, xys[j + 1][1] - y_j
                else:
                    dt_j = max(t_g - pts_ok[j - 1][0], 1e-3)
                    dx_j, dy_j = x_j - xys[j - 1][0], y_j - xys[j - 1][1]
                step_dist = math.hypot(dx_j, dy_j)
                spd_j = step_dist / dt_j
                h_road = road.lane_heading(s_g) + (180.0 if kt_d["back"] else 0.0)
                if step_dist > 0.1:
                    h_raw = math.degrees(math.atan2(dy_j, dx_j))
                    dh_road = max(-18.0, min(18.0, _wrap(h_raw - h_road)))
                    blend = 1.0 if step_dist >= 0.4 else (step_dist - 0.1) / 0.3
                    h_des = h_road + blend * dh_road
                else:
                    h_des = h_road
                if kt_rows:
                    dh = max(-30.0 * dt_j, min(30.0 * dt_j, _wrap(h_des - kt_rows[-1][3])))
                    h_j = _wrap(kt_rows[-1][3] + dh)
                else:
                    h_j = _wrap(h_des)
                kt_rows.append([round(t_g, 2), round(x_j, 2), round(y_j, 2), round(h_j, 1), round(spd_j, 2)])
            agents[vid]["x"], agents[vid]["y"], agents[vid]["heading"] = kt_rows[0][1], kt_rows[0][2], kt_rows[0][3]
            plan_agents[vid]["kinematic_track"] = kt_rows
    return notes


# ----------------------------------------------------------------------------------------- the builder
def _road_with_profile(spec, road, s_ego0, d_event, ev_actor, min_length=None):
    """Straight road whose lane count follows spec.road.lane_profile (distances from the ego start, agents/lane_profile.py).
    Returns (road, ego lane index k, note). The ego lane is the measured one (counted from the left) when it exists at the start;
    a car that merges into the ego lane needs its source lane beside the ego car at the start and at the event."""
    prof = spec["road"]["lane_profile"]
    big = 1e9
    secs = [{"s0": s_ego0 + (sc["d0"] if sc["d0"] is not None else -big), "s1": s_ego0 + (sc["d1"] if sc["d1"] is not None else big),
             "n": int(sc["n"]), "m": int(sc.get("m", 0)), **({"branch": dict(sc["branch"])} if sc.get("branch") else {})}
            for sc in prof]
    new = Road("straight", spec["road"]["lanes_per_direction"], spec["road"].get("centerline"), spec["road"].get("one_way", False), min_length=min_length)
    secs = [sc for sc in secs if sc["s1"] > 1.0 and sc["s0"] < new.length - 1.0]
    # Defensive geometric invariant: after a diverging branch, the continuing road cannot regrow lanes into the branch side.
    bi_p = next((i for i in range(1, len(secs)) if secs[i].get("branch")), None)
    if bi_p is not None:
        b_s = secs[bi_p]
        b_sd = b_s["branch"].get("side")
        for sc in secs[bi_p + 1:]:
            if b_sd == "left" and sc["m"] < b_s["m"]:
                cut = b_s["m"] - sc["m"]
                sc["m"] = b_s["m"]
                sc["n"] = max(1, sc["n"] - cut)
            elif b_sd == "right":
                r_lim = b_s["m"] + b_s["n"]
                if sc["m"] >= r_lim:
                    sc["m"] = max(0, r_lim - 1)
                    sc["n"] = 1
                elif sc["m"] + sc["n"] > r_lim:
                    sc["n"] = max(1, r_lim - sc["m"])
        mi = bi_p
        while mi + 1 < len(secs):
            if not secs[mi + 1].get("branch") and secs[mi]["n"] == secs[mi + 1]["n"] and secs[mi]["m"] == secs[mi + 1]["m"]:
                secs[mi]["s1"] = secs[mi + 1]["s1"]
                secs.pop(mi + 1)
            else:
                mi += 1
    if len(secs) < 2:
        return road, (0 if spec["ego"]["lane"] == "right" or road.L == 1 else road.L - 1), \
            "lane profile: the count does not change on the simulated stretch"
    lo = min(sc["m"] for sc in secs)
    for sc in secs:
        sc["m"] -= lo
    new = Road("straight", spec["road"]["lanes_per_direction"], spec["road"].get("centerline"), spec["road"].get("one_way", False), secs, min_length=min_length)
    ks = new.lanes_at(s_ego0)
    idx = spec["ego"].get("lane_index")
    k = new.L - 1 - (idx - lo) if idx is not None else None
    if k not in ks:
        k = min(ks) if spec["ego"]["lane"] == "right" or len(ks) == 1 else max(ks)
    if ev_actor is not None and (ev_actor["relation"] == "beside" or ev_actor["behavior"] == "cutting_in"):
        step = 1 if ev_actor["lane"] != "right" else -1
        ok = [q for q in ks if new.has_lane(s_ego0, q + step) and new.has_lane(s_ego0 + d_event, q + step) and new.has_lane(s_ego0 + d_event, q)]
        if not ok:
            has_branch = any(sc.get("branch") for sc in secs)
            ev_sec = new.section(s_ego0 + d_event)
            # When the profile contains a diverging branch (or has >= 2 lanes at the event and only section 0 lacked the
            # source lane due to a narrow start), repair the section widths so the diverging road is preserved instead of
            # discarding the entire lane_profile.
            if has_branch or ev_sec["n"] >= 2:
                repaired = [dict(sc) for sc in secs]
                ev_idx = next((i for i, sc in enumerate(repaired) if (s_ego0 + d_event) < sc["s1"]), len(repaired) - 1)
                if repaired[ev_idx]["n"] < 2:
                    # Widen post-split ego road to 2 lanes on the non-branch side so the cut-in fits alongside the fork
                    b_i = next((i for i, sc in enumerate(repaired) if sc.get("branch")), None)
                    b_side = repaired[b_i]["branch"]["side"] if b_i is not None else ("right" if step > 0 else "left")
                    for sc in repaired[ev_idx:]:
                        if sc["n"] < 2:
                            if b_side == "left":
                                sc["n"] = 2
                            else:
                                sc["m"] = max(0, sc["m"] - (2 - sc["n"]))
                                sc["n"] = 2
                # Ensure section 0 covers the event section's lanes so both ego and the cutting car have lanes from s_ego0
                e_m, e_end = repaired[ev_idx]["m"], repaired[ev_idx]["m"] + repaired[ev_idx]["n"]
                s0_m = min(repaired[0]["m"], e_m)
                s0_end = max(repaired[0]["m"] + repaired[0]["n"], e_end)
                if s0_end - s0_m <= 6:
                    repaired[0]["m"], repaired[0]["n"] = s0_m, s0_end - s0_m
                lo_r = min(sc["m"] for sc in repaired)
                for sc in repaired:
                    sc["m"] -= lo_r
                if max(sc["m"] + sc["n"] for sc in repaired) <= 6:
                    cand = Road("straight", spec["road"]["lanes_per_direction"], spec["road"].get("centerline"),
                                spec["road"].get("one_way", False), repaired, min_length=min_length)
                    ks_r = cand.lanes_at(s_ego0)
                    ok_r = [q for q in ks_r if cand.has_lane(s_ego0, q + step) and cand.has_lane(s_ego0 + d_event, q + step) and cand.has_lane(s_ego0 + d_event, q)]
                    if ok_r:
                        new, ks, ok = cand, ks_r, ok_r
                        k = new.L - 1 - (idx - lo_r) if idx is not None else None
                        if k not in ks:
                            k = min(ks) if spec["ego"]["lane"] == "right" or len(ks) == 1 else max(ks)
        if not ok:                      # the crash matters more than the lane count: keep the road the crash needs
            return road, (0 if spec["ego"]["lane"] == "right" or road.L == 1 else road.L - 1), \
                "lane profile dropped: no lane on the %s of the ego car for %s's merge at the event" % (
                    "left" if step > 0 else "right", ev_actor["id"])
        lane_off = max(1, int(((ev_actor.get("measured") or {}).get("lane_offset")) or 1))
        if lane_off >= 2:
            ok_multi = [q for q in ok if new.has_lane(s_ego0, q + step * lane_off) and new.has_lane(s_ego0 + d_event, q + step * lane_off)]
            if ok_multi:
                ok = ok_multi
        if k not in ok:
            k = min(ok, key=lambda q: abs(q - k))
    note = "lane profile: %s lanes along the road, ego in lane %d of %d from the left at the start" % (
        " -> ".join("%d" % sc["n"] if i == 0 else "%d (from %.0f m)" % (sc["n"], sc["s0"] - s_ego0) for i, sc in enumerate(new.sections)),
        new.L - k - new.section(s_ego0)["m"], len(ks))
    return new, k, note


def build(spec, out_dir, scene_id="video_interpreter_0000_split"):
    os.makedirs(out_dir, exist_ok=True)
    notes = []
    ego = spec["ego"]
    event = spec["event"]
    v_e = ego["speed_kmh"] / 3.6
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
    t_ev_meas = ((ev_actor.get("measured") or {}).get("t_event_s")) if (ev_actor is not None and not third_party) else None
    if t_ev_meas is not None:
        T = min(max(float(t_ev_meas), 3.0), 12.0)
    else:
        T = min(max(event["time_s"], 3.0), 8.0) if ev_actor is not None else 6.0       # simulated time of the event
    s_ego0_est = 65.0 if straight_tp_cross else S_START
    min_len = _required_road_length(spec, s_ego0_est, v_e, T)
    road = Road(spec["road"]["type"], spec["road"]["lanes_per_direction"], spec["road"].get("centerline"), spec["road"].get("one_way", False), min_length=min_len)
    lane_idx = 0 if ego["lane"] == "right" or road.L == 1 else road.L - 1
    if straight_tp_cross and road.L < 2:
        road = Road("straight", 2, spec["road"].get("centerline"), spec["road"].get("one_way", False), min_length=min_len)
        lane_idx = 0 if ego["lane"] == "right" else road.L - 1
    if road.kind == "straight" and spec["road"].get("lane_profile") and not (third_party and not straight_tp_cross):
        s_ego0 = 65.0 if straight_tp_cross else S_START
        ego_trav_est = _ego_travel({"mode": "static" if ego["behavior"] == "stopped" else "drive",
                                    "speed_profile": ego.get("speed_profile")}, v_e)(T)
        road, lane_idx, pnote = _road_with_profile(spec, road, s_ego0, ego_trav_est, ev_actor, min_length=min_len)
        notes.append(pnote)
    if third_party and not straight_tp_cross:
        # the crash is between two other road users: they meet in the junction while the ego car passes by
        if spec["road"]["type"] == "straight":
            notes.append("third-party crash needs a junction: road changed to crossroad")
            road = Road("crossroad", spec["road"]["lanes_per_direction"])
        ev_actor.update({"relation": "cross_left", "behavior": "driving", "type": ev_actor["type"] if ev_actor["type"] != "pedestrian" else "car"})
        ev2.update({"relation": "cross_right", "behavior": "turning_left", "type": ev2["type"] if ev2["type"] != "pedestrian" else "car"})
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
        end_pt = road.end_point(lane_idx)
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
    if ego.get("speed_profile") and plan_ego.get("mode") == "drive":
        plan_ego["speed_profile"] = ego["speed_profile"]          # the ego speed the video overlay printed, in sim time
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
    if straight_tp_cross:
        tp_k2 = road.clamp_lane(tp_cross_s, tp_k2)
    tp_t_cross = None
    if straight_tp_cross:
        d_left_tp = (0.0 if road.one_way else road.Lw * W) + shoulder_m - 1.5
        d_right_tp = road.right_edge_d(tp_cross_s) - (shoulder_m - 1.5)
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
            # a vehicle in the neighbouring (or outer) lane that changes into the ego lane (side contact or close pass)
            left = a["lane"] != "right"
            meas = a.get("measured") if isinstance(a.get("measured"), dict) else None
            lane_off = max(1, int(meas.get("lane_offset", 1))) if meas else 1
            k_src = lane_idx
            d_check = _ego_travel(plan_ego, v_e)(T)
            for off in range(lane_off, 0, -1):
                kc = min(lane_idx + off, road.L - 1) if left else max(lane_idx - off, 0)
                if kc == lane_idx:
                    continue
                if road.kind == "straight" and not (road.has_lane(S_START, kc) and road.has_lane(S_START + d_check, kc)):
                    continue
                k_src = kc
                break
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
                    lanes_crossed = abs(k_src - lane_idx)
                    cruise_ms = None
                    head_i = head
                    if isinstance(meas, dict) and meas.get("overtake"):
                        v_c = max(float(meas.get("speed_kmh", 20.0)) / 3.6, 5.0)
                        if v_e > 6.0:
                            v_c = min(v_c, 0.85 * v_e)
                        lead = {"collision": 2.0, "near_miss": 1.0}.get(event["type"], lead)
                        s_ev = _ego_travel(plan_ego, v_e)(T)
                        s0_cand = max(1.0, min(s_ev + 1.0 - v_c * T, road.end_s - S_START - 20.0))
                        if road.has_lane(S_START + s0_cand, k_src):
                            s0, v = s0_cand, v_c
                            cruise_ms = round(v_c, 2)
                            head_i = road.lane_heading(S_START + s0)
                    if lanes_crossed >= 2:
                        lead = round(lead + 0.5 * (lanes_crossed - 1), 2)
                    x_i, y_i = lane_pt(k_src, s0)
                    agents[vid] = {"type": a["type"], "x": x_i, "y": y_i, "heading": head_i}
                    d_src, d_dst = lane_offset(road.L, k_src), lane_offset(road.L, lane_idx)
                    aim = 0.0 if event["type"] == "collision" else (2.6 if d_src > d_dst else -2.6)     # a near miss keeps a 0.6 m gap
                    icp_d = {"kind": "cut_in", "target": "v1", "lead_s": lead, "aim_offset_m": aim, "t_event_s": round(T, 2),
                             "lane_src_d": d_src, "lane_dst_d": d_dst}
                    if cruise_ms is not None:
                        icp_d["cruise_speed_ms"] = cruise_ms
                    if lanes_crossed >= 2:
                        icp_d["max_lateral_speed_ms"] = 3.2
                    spec_p = {"mode": "drive", "speed_kmh": round(v * 3.6, 1), "already_moving": True, "avoid": False,
                              "intercept": icp_d}
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
                        k_c = road.clamp_lane(S_START + min(s_c, end_s), lane_idx) if road.kind == "straight" else lane_idx
                        pts.append(lane_pt(k_c, min(s_c, end_s)))
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
                          "path": [road.end_point(tp_k2)]}
            elif rel in ("cross_left", "cross_right"):
                left = rel == "cross_left"
                d_left = (0.0 if road.one_way else road.Lw * W) + shoulder_m - 1.5
                ps = None
                for _ in range(2 if road.varying else 1):      # the right edge (where the car starts or ends) depends on ps
                    d_right = (road.right_edge_d(ps) if ps is not None else -road.L * W) - (shoulder_m - 1.5)
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
                if not road.has_lane(sa, lane_k):
                    lane_k = road.clamp_lane(sa, lane_k)
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
                lane_k = road.clamp_lane(sa, lane_k)
                agents[vid] = {"type": a["type"], "x": road.lane_point(sa, lane_k)[0], "y": road.lane_point(sa, lane_k)[1],
                               "heading": road.lane_heading(sa)}
                if beh in ("stopped",) or v_a < 1.0:
                    spec_p = {"mode": "static"}
                else:
                    spec_p = {"mode": "drive", "speed_kmh": a["speed_kmh"], "already_moving": True, "avoid": False,
                              "path": [road.end_point(lane_k)]}
                    if beh == "braking":
                        spec_p["stop_at_s"] = round(max(T - 1.5, 1.0), 2)
                        if is_ev:           # closed loop: brake when the ego car is about 2 s away, whichever comes first
                            spec_p["brake_trigger"] = {"who": "v1", "ttc_s": 2.0}
            elif rel == "behind":
                sb = S_START - max(a["distance_m"], 10.0)
                if sb < 3.0:
                    notes.append("%s: no room behind the ego car, actor dropped" % a["id"])
                    continue
                lane_k = road.clamp_lane(sb, lane_k)
                agents[vid] = {"type": a["type"], "x": road.lane_point(sb, lane_k)[0], "y": road.lane_point(sb, lane_k)[1],
                               "heading": road.lane_heading(sb)}
                spec_p = {"mode": "drive", "speed_kmh": a["speed_kmh"], "already_moving": True, "avoid": True,
                          "path": [road.end_point(lane_k)]}
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
                cp_raw = event.get("contact_parts")
                if isinstance(cp_raw, dict):
                    mapped = {}
                    for k_cp, v_cp in cp_raw.items():
                        mk = "v1" if k_cp in ("ego", "v1") else idmap.get(k_cp)
                        if mk in pair:
                            mapped[mk] = v_cp
                    if mapped:
                        expect["contact_parts"] = mapped
            elif event["type"] == "near_miss":
                expect = {"type": "near_miss", "actors": pair, "time_s": round(t_exp, 2), "window_s": 1.5, "contact": "any",
                          "max_dist_m": 8.0 if (road.kind == "straight" or a["relation"] in ("ahead", "behind")) else 6.0}
    if third_party:
        t_third = tp_t_cross if straight_tp_cross else T
        tp_pair = [idmap[ev_actor["id"]], idmap[ev2["id"]]]
        expect = {"type": "collision" if event["type"] == "collision" else "near_miss",
                  "actors": tp_pair, "time_s": round(t_third, 2), "max_dist_m": 6.0,
                  "window_s": 1.5, "contact": "any"}
        cp_raw = event.get("contact_parts")
        if event["type"] == "collision" and isinstance(cp_raw, dict):
            mapped = {idmap[k_cp]: v_cp for k_cp, v_cp in cp_raw.items() if k_cp in idmap and idmap[k_cp] in tp_pair}
            if mapped:
                expect["contact_parts"] = mapped
    if expect is None:
        expect = {"type": "none", "actors": ["v1"], "time_s": round(T, 2)}

    if road.kind == "straight" and spec.get("replay"):
        s_ego_t = _ego_travel(plan_agents["v1"], v_e)
        s_ego_rp0 = 65.0 if straight_tp_cross else S_START
        d_ego_off = lane_offset(road.L, lane_idx)
        corridors = []                       # (d left of the ego lane centre, s(t) from the ego start, t_end[, t_start[, half_d]])
        if plan_agents["v1"].get("mode") == "drive":
            corridors.append((0.0, s_ego_t, T + 2.0))
        for vid_c, p in plan_agents.items():
            if vid_c == "v1" or not p:
                continue
            ic = p.get("intercept")
            if ic:                           # the cut-in car drives beside/ahead in its source lane, then crosses to ego lane
                if ic.get("cruise_speed_ms") is not None:
                    s_c0 = road.frame.to_sd(agents[vid_c]["x"], agents[vid_c]["y"])[0] - s_ego_rp0
                    vc = float(ic["cruise_speed_ms"])
                    s_ic_fn = lambda t, s0_=s_c0, vc_=vc: s0_ + vc_ * t
                else:
                    s_ic_fn = lambda t: s_ego_t(t) + 1.0
                if abs(ic["lane_dst_d"] - ic["lane_src_d"]) > 4.0:
                    lead_ic = float(ic.get("lead_s", 1.4))
                    t_lc0 = max(0.0, T - lead_ic)
                    n_lanes = max(2, int(round(abs(ic["lane_dst_d"] - ic["lane_src_d"]) / W)))
                    lat_spd = max(float(ic.get("max_lateral_speed_ms", 1.5)), 2.2 if ic.get("cruise_speed_ms") is not None else 1.5)
                    dt_lane = W / max(lat_spd, 1.0)
                    corridors.append((ic["lane_src_d"] - d_ego_off, s_ic_fn, min(T + 2.0, t_lc0 + 0.8 * dt_lane), 0.0))
                    for m_i in range(1, n_lanes):
                        d_mid = ic["lane_src_d"] + (ic["lane_dst_d"] - ic["lane_src_d"]) * m_i / n_lanes
                        corridors.append((d_mid - d_ego_off, s_ic_fn, min(T + 2.0, t_lc0 + (m_i + 0.8) * dt_lane),
                                          max(0.0, t_lc0 + (m_i - 0.6) * dt_lane)))
                    corridors.append((ic["lane_dst_d"] - d_ego_off, s_ic_fn, T + 2.0,
                                      max(0.0, t_lc0 + (n_lanes - 0.8) * dt_lane)))
                else:
                    t_sweep0 = max(0.0, T - float(ic.get("lead_s", 1.4)) - 0.5)
                    corridors.append((ic["lane_src_d"] - d_ego_off, s_ic_fn, T + 2.0))
                    corridors.append((ic["lane_dst_d"] - d_ego_off, s_ic_fn, T + 2.0, t_sweep0))
            elif vid_c in agents and vid_c in (idmap.get(ev_actor["id"]) if ev_actor else None,
                                               idmap.get(ev2["id"]) if ev2 else None):
                s_ev0, d_ev0 = road.frame.to_sd(agents[vid_c]["x"], agents[vid_c]["y"])
                if p.get("polyline") and abs(agents[vid_c]["heading"] - road.lane_heading(s_ev0)) > 45.0:
                    t_c_end = (tp_t_cross + 1.5) if (straight_tp_cross and tp_t_cross is not None) else (T + 2.0)
                    corridors.append((0.0, lambda t, sc=s_ev0 - s_ego_rp0: sc, t_c_end, 0.0, road.L * W + 6.0))
                elif p.get("mode") == "drive":
                    ve = float(p.get("speed_kmh", 0.0)) / 3.6
                    corridors.append((d_ev0 - d_ego_off, lambda t, s0_=s_ev0 - s_ego_rp0, ve_=ve: s0_ + ve_ * t, T + 2.0))
        notes += _replay_agents(spec["replay"], road, lane_idx, s_ego_rp0, agents, plan_agents, idmap,
                                corridors, duration_s=round(T + 5.0, 1))

    # ---- optional non-conflicting cruisers on an empty diverging branch (only for rich scenes with traffic/replay)
    if road.kind == "straight" and split_index(road) is not None and (spec.get("replay") or any(a["id"].startswith("b") for a in spec.get("actors") or [])):
        b_i = split_index(road)
        a_sec, b_sec = road.sections[b_i - 1], road.sections[b_i]
        s_b = b_sec["s0"]
        left_b = b_sec["branch"]["side"] == "left"
        leave_j = list(range(a_sec["m"], b_sec["m"])) if left_b else list(range(b_sec["m"] + b_sec["n"], a_sec["m"] + a_sec["n"]))
        br_edge = -b_sec["m"] * W if left_b else -(b_sec["m"] + b_sec["n"]) * W
        has_br_car = False
        for vid_k, p_k in plan_agents.items():
            kt = (p_k or {}).get("kinematic_track") or []
            for row in kt:
                sr, dr = road.frame.to_sd(row[1], row[2])
                if sr > s_b + 10.0 and ((left_b and dr > br_edge + 0.5) or (not left_b and dr < br_edge - 0.5)):
                    has_br_car = True
                    break
            if has_br_car:
                break
        if not has_br_car and leave_j:
            dur_br = round(T + 5.0, 1)
            v_br = max(v_e * 1.05, 16.0)
            for idx_bc, (s_off_b, j_b, col_b, vt_b) in enumerate(((25.0, leave_j[0], "175,175,180", "car"),
                                                                  (60.0, leave_j[-1], "235,235,235", "suv"))):
                s_init = s_b + s_off_b
                if s_init >= road.end_s - 40.0:
                    continue
                kt_rows = []
                tg = 0.0
                while tg <= dur_br + 2.5:
                    s_cur = s_init + v_br * tg
                    if s_cur > road.end_s:
                        break
                    f_d = DIVERGE_D_M * _smooth((s_cur - s_b) / DIVERGE_LEN_M) * (1.0 if left_b else -1.0)
                    d_cur = -(j_b + 0.5) * W + f_d
                    x_c, y_c = road.frame.to_xy(s_cur, d_cur)
                    kt_rows.append([round(tg, 2), round(x_c, 2), round(y_c, 2), round(road.lane_heading(s_cur), 1), round(v_br, 2)])
                    tg = round(tg + 0.2, 2)
                if len(kt_rows) >= 2:
                    for k_r in range(len(kt_rows) - 1):
                        dx_r = kt_rows[k_r + 1][1] - kt_rows[k_r][1]
                        dy_r = kt_rows[k_r + 1][2] - kt_rows[k_r][2]
                        if math.hypot(dx_r, dy_r) > 0.1:
                            kt_rows[k_r][3] = round(math.degrees(math.atan2(dy_r, dx_r)), 1)
                    kt_rows[-1][3] = kt_rows[-2][3]
                    vid_b = "v%d" % (len(agents) + 1)
                    bid = "br%d" % (idx_bc + 1)
                    agents[vid_b] = {"type": vt_b, "x": kt_rows[0][1], "y": kt_rows[0][2], "heading": kt_rows[0][3]}
                    idmap[bid] = vid_b
                    plan_agents[vid_b] = {"mode": "drive", "speed_kmh": round(v_br * 3.6, 1), "already_moving": True,
                                          "avoid": False, "color": col_b, "replay": bid, "kinematic_track": kt_rows}

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
    if road.kind == "straight" and road.varying:
        road_plan["sections"] = [{k: (round(v, 2) if isinstance(v, float) else v) for k, v in sc.items()} for sc in road.sections]
        road_plan["lanes_opposite"] = road.Lw
        road_plan["xodr_pieces"] = xodr_pieces(road)
        b_i = split_index(road)
        if b_i is not None:
            a_sec, b_sec = road.sections[b_i - 1], road.sections[b_i]
            br = b_sec["branch"]
            b_side = br["side"]
            leave_j = list(range(a_sec["m"], b_sec["m"])) if b_side == "left" else list(range(b_sec["m"] + b_sec["n"], a_sec["m"] + a_sec["n"]))
            # the roads on the inner side of a fork face each other: skip ONLY their inner facing shoulders so outer shoulders
            # remain intact and inner asphalt never overlaps across the gore
            road_plan["no_shoulder"] = ["branch"] if b_side == "left" else ["eastbound_after"]
            road_plan["no_shoulder_sides"] = ({"branch": ["right"], "eastbound_after": ["left"]} if b_side == "left"
                                              else {"branch": ["left"], "eastbound_after": ["right"]})
            road_plan["fork"] = {
                "s_b": round(b_sec["s0"], 2), "side": b_side, "rise_m": round(float(br.get("rise_m", 0.0)), 2),
                "leave_j": leave_j, "diverge_len_m": DIVERGE_LEN_M, "diverge_d_m": DIVERGE_D_M,
                "rise_start_m": RISE_START_M, "rise_len_m": RISE_LEN_M,
            }
    plan = {"ego": "v1", "duration_s": round(T + 5.0, 1), "expect": expect, "agents": plan_agents, "idmap": idmap,
            "notes": notes, "source": "scene_builder", "road": road_plan}
    if isinstance(spec.get("environment"), dict):
        plan["environment"] = dict(spec["environment"])
    _write_json(os.path.join(out_dir, scene_id + "_motion.json"), plan)
    _write_json(os.path.join(out_dir, scene_id + "_spec.json"), spec)
    return plan
