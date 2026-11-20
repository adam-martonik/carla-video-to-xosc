import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.road_frame import RoadFrame
from agents.scene_builder import Road


class TestRoadOnFrame(unittest.TestCase):
    def test_straight_road_has_a_frame_and_old_lane_positions(self):
        r = Road("straight", 2)
        self.assertIsInstance(r.frame, RoadFrame)
        self.assertEqual(r.lane_point(15.0, 0), (15.0, r.eastbound_y(0)))
        self.assertEqual(r.lane_point(40.0, 1), (40.0, r.eastbound_y(1)))
        self.assertEqual(r.lane_heading(40.0), 0.0)

    def test_opposite_lanes_mirror_across_the_centre_line(self):
        r = Road("straight", 2)
        self.assertEqual(r.opposite_lane_point(30.0, 0), (30.0, r.westbound_y(0)))

    def test_crossroad_has_no_frame(self):
        self.assertIsNone(Road("crossroad", 2).frame)


if __name__ == "__main__":
    unittest.main()
