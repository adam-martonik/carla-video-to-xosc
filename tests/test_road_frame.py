import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.road_frame import RoadFrame, lane_offset


def left_arc(radius=200.0, angle_deg=90.0, step=2.0):
    n = int(math.radians(angle_deg) * radius / step)
    pts = []
    for i in range(n + 1):
        a = i * step / radius
        pts.append((radius * math.sin(a), radius * (1 - math.cos(a))))
    return pts


class TestRoadFrame(unittest.TestCase):
    def test_straight_matches_old_layout(self):
        f = RoadFrame.straight()
        self.assertEqual(f.to_xy(15, lane_offset(2, 0)), (15.0, -5.25))
        self.assertEqual(lane_offset(2, 1), -1.75)

    def test_arc_roundtrip(self):
        f = RoadFrame(left_arc())
        for s in (0, 50, 150, 300):
            for d in (-5.25, 0.0, 3.5):
                x, y = f.to_xy(s, d)
                s2, d2 = f.to_sd(x, y)
                self.assertAlmostEqual(s2, s, delta=0.05)
                self.assertAlmostEqual(d2, d, delta=0.05)

    def test_heading_on_left_arc(self):
        f = RoadFrame(left_arc())
        self.assertAlmostEqual(f.heading(f.length - 0.5), 90.0, delta=1.0)
        self.assertLess(f.heading(50), f.heading(150))

    def test_left_offset_is_left_of_travel_on_a_left_arc(self):
        f = RoadFrame(left_arc())
        x0, y0 = f.to_xy(100, 0.0)
        xl, yl = f.to_xy(100, 3.5)
        # left of the travel direction is toward the arc centre, i.e. larger y on this arc
        self.assertGreater(yl, y0)


if __name__ == "__main__":
    unittest.main()
