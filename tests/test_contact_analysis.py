import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.contact_analysis import contact_direction


class TestContactDirection(unittest.TestCase):
    def test_lateral_impulse_is_side(self):
        self.assertEqual(contact_direction([0, 800, 0], 0.0), "side")

    def test_longitudinal_impulse_with_other_ahead_is_front(self):
        self.assertEqual(contact_direction([800, 0, 0], 0.0, rel_xy=(2.0, 0.0)), "front")

    def test_longitudinal_impulse_with_other_behind_is_rear(self):
        self.assertEqual(contact_direction([800, 0, 0], 0.0, rel_xy=(-2.0, 0.0)), "rear")

    def test_yaw_rotates_the_ego_frame(self):
        # ego heading +y (CARLA yaw 90): an impulse along x is lateral for it
        self.assertEqual(contact_direction([800, 0, 0], 90.0), "side")

    def test_longitudinal_without_position_defaults_to_front(self):
        self.assertEqual(contact_direction([500, 0, 0], 0.0), "front")


if __name__ == "__main__":
    unittest.main()
