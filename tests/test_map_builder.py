import os
import shutil
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import map_builder, scene_builder


def make_xodr(tmp, one_way=False):
    spec = {"road": {"type": "straight", "lanes_per_direction": 2, "traffic_signals": False, "one_way": one_way},
            "ego": {"type": "car", "lane": "right", "speed_kmh": 50.0, "behavior": "cruise"}, "actors": [],
            "event": {"type": "none", "actor_id": None, "actor2_id": None, "time_s": 5.0, "description": ""}}
    scene_builder.build(spec, tmp, "s")
    return map_builder.net_to_xodr(os.path.join(tmp, "s.net.xml"), os.path.join(tmp, "s.xodr"))


def lanes(road, side):
    node = road.find("lanes/laneSection/" + side)
    return [] if node is None else node.findall("lane")


class TestShoulders(unittest.TestCase):
    def test_right_shoulder_is_the_outermost_right_lane(self):
        with tempfile.TemporaryDirectory() as tmp:
            x = make_xodr(tmp)
            n = map_builder.add_shoulders(x, 3.0, ("right",))
            self.assertGreaterEqual(n, 1)
            for road in ET.parse(x).getroot().iter("road"):
                right = lanes(road, "right")
                if not right:
                    continue
                last = right[-1]
                self.assertEqual(last.get("type"), "shoulder")
                self.assertEqual(float(last.find("width").get("a")), 3.0)
                self.assertEqual(int(last.get("id")), -(len(right)))
                self.assertEqual([l.get("type") for l in right[:-1]], ["driving", "driving"])

    def test_no_sides_changes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            x = make_xodr(tmp)
            before = Path(x).read_text(encoding="utf-8")
            self.assertEqual(map_builder.add_shoulders(x, 3.0, ()), 0)
            self.assertEqual(Path(x).read_text(encoding="utf-8"), before)

    def test_left_shoulder_on_a_one_way_road_goes_on_the_left_of_the_lanes(self):
        with tempfile.TemporaryDirectory() as tmp:
            x = make_xodr(tmp, one_way=True)
            map_builder.add_shoulders(x, 3.0, ("left",))
            road = next(ET.parse(x).getroot().iter("road"))
            left = lanes(road, "left")
            self.assertEqual(len(left), 1)
            self.assertEqual(left[0].get("type"), "shoulder")
            self.assertEqual(int(left[0].get("id")), 1)

    def test_sides_from_the_plan(self):
        plan = {"edge_right": "barrier", "edge_left": "open", "shoulder_m": 3.0}
        self.assertEqual(map_builder.shoulder_sides(plan), ("right",))
        plan = {"edge_right": "shoulder", "edge_left": "shoulder", "shoulder_m": 3.0}
        self.assertEqual(map_builder.shoulder_sides(plan), ("right", "left"))
        self.assertEqual(map_builder.shoulder_sides(None), ())

    def test_move_signals_off_lanes_uses_wide_margin(self):
        with tempfile.TemporaryDirectory() as tmp:
            x = make_xodr(tmp)
            tree = ET.parse(x)
            road = next(tree.getroot().iter("road"))
            sigs = road.find("signals")
            if sigs is None:
                sigs = ET.SubElement(road, "signals")
            ET.SubElement(sigs, "signal", id="1", s="10.0", t="-1.75")
            tree.write(x, encoding="utf-8", xml_declaration=True)
            moved = map_builder.move_signals_off_lanes(x)
            self.assertEqual(moved, 1)
            sig = next(ET.parse(x).getroot().iter("signal"))
            right_width = sum(float(w.get("a", 0)) for l in road.find("lanes/laneSection/right").findall("lane")
                              for w in l.findall("width"))
            self.assertGreaterEqual(abs(float(sig.get("t"))), right_width + 3.5 - 1e-4)

    def test_apply_lane_sections_remaps_incoming_junction_connector_links(self):
        xml = """<?xml version="1.0" encoding="UTF-8"?>
<OpenDRIVE>
  <road name="" length="8.0" id="64" junction="1">
    <link><predecessor elementType="road" elementId="61" contactPoint="end"/><successor elementType="road" elementId="62" contactPoint="start"/></link>
    <lanes><laneSection s="0"><center><lane id="0" type="none"/></center><right>
      <lane id="-1" type="driving"><link><predecessor id="-3"/><successor id="-2"/></link><width sOffset="0" a="3.5" b="0" c="0" d="0"/></lane>
      <lane id="-2" type="driving"><link><predecessor id="-4"/><successor id="-3"/></link><width sOffset="0" a="3.5" b="0" c="0" d="0"/></lane>
      <lane id="-3" type="driving"><link><predecessor id="-5"/><successor id="-4"/></link><width sOffset="0" a="3.5" b="0" c="0" d="0"/></lane>
    </right></laneSection></lanes>
  </road>
  <road name="" length="450.0" id="62" junction="-1">
    <userData code="sumoId" value="eastbound_after"/>
    <lanes><laneSection s="0"><center><lane id="0" type="none"/></center><right>
      <lane id="-1" type="driving"><link/><width sOffset="0" a="3.5" b="0" c="0" d="0"/></lane>
      <lane id="-2" type="driving"><link/><width sOffset="0" a="3.5" b="0" c="0" d="0"/></lane>
      <lane id="-3" type="driving"><link/><width sOffset="0" a="3.5" b="0" c="0" d="0"/></lane>
      <lane id="-4" type="driving"><link/><width sOffset="0" a="3.5" b="0" c="0" d="0"/></lane>
    </right></laneSection></lanes>
  </road>
</OpenDRIVE>"""
        with tempfile.TemporaryDirectory() as tmp:
            x = os.path.join(tmp, "m.xodr")
            Path(x).write_text(xml, encoding="utf-8")
            secs = [{"s0": 113.0, "s1": 161.0, "n": 3, "m": 1}, {"s0": 161.0, "s1": 563.0, "n": 4, "m": 0}]
            map_builder.apply_lane_sections(x, secs, 450.0, road_user_id="eastbound_after", s0=113.0)
            root = ET.parse(x).getroot()
            r64 = next(r for r in root.iter("road") if r.get("id") == "64")
            succs = [int(l.find("link/successor").get("id")) for l in r64.findall("lanes/laneSection/right/lane")]
            self.assertEqual(succs, [-1, -2, -3])

    def test_fork_shoulders_skip_only_inner_gore_sides_and_keep_outer_shoulders(self):
        prof = [{"d0": None, "d1": 65.0, "n": 4, "m": 0},
                {"d0": 65.0, "d1": None, "n": 2, "m": 2, "branch": {"side": "left", "rise_m": 6.0}}]
        spec = {"road": {"type": "straight", "lanes_per_direction": 2, "traffic_signals": False, "one_way": True,
                         "edge_right": "barrier", "edge_left": "barrier", "lane_profile": prof},
                "ego": {"type": "car", "lane": "right", "lane_index": 3, "speed_kmh": 55.0, "behavior": "cruise"},
                "actors": [], "event": {"type": "none", "actor_id": None, "actor2_id": None, "time_s": 5.0, "description": ""}}
        with tempfile.TemporaryDirectory() as tmp:
            plan = scene_builder.build(spec, tmp, "fk")
            x = map_builder.net_to_xodr(os.path.join(tmp, "fk.net.xml"), os.path.join(tmp, "fk.xodr"))
            map_builder.add_shoulders(x, 3.0, ("right", "left"), one_way=True,
                                      skip_roads=tuple(plan["road"].get("no_shoulder") or ()),
                                      skip_road_sides=plan["road"].get("no_shoulder_sides"))
            by_sumo = {}
            for r in ET.parse(x).getroot().iter("road"):
                ud = r.find("userData")
                if ud is not None:
                    by_sumo[ud.get("value")] = r
            # eastbound_after keeps right outer shoulder, NO inner left shoulder overlapping branch
            self.assertEqual(lanes(by_sumo["eastbound_after"], "right")[-1].get("type"), "shoulder")
            self.assertEqual(lanes(by_sumo["eastbound_after"], "left"), [])
            # branch keeps left outer shoulder, NO inner right shoulder overlapping eastbound_after
            self.assertEqual(lanes(by_sumo["branch"], "left")[0].get("type"), "shoulder")
            self.assertTrue(all(l.get("type") == "driving" for l in lanes(by_sumo["branch"], "right")))


if __name__ == "__main__":
    unittest.main()
