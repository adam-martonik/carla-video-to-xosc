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


if __name__ == "__main__":
    unittest.main()
