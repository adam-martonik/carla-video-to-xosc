import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.contact_analysis import contact_direction, contact_zone, pair_contact_zones, zones_compatible


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


class TestContactZones(unittest.TestCase):
    def test_all_eight_zones_from_impulse_and_rel(self):
        self.assertEqual(contact_zone([900, 50, 0], 0.0, rel_xy=(3.0, 0.2)), "front")
        self.assertEqual(contact_zone([900, 50, 0], 0.0, rel_xy=(-3.0, -0.2)), "rear")
        self.assertEqual(contact_zone([50, 900, 0], 0.0, rel_xy=(0.2, -2.0)), "left_side")
        self.assertEqual(contact_zone([50, 900, 0], 0.0, rel_xy=(0.2, 2.0)), "right_side")
        self.assertEqual(contact_zone([600, 600, 0], 0.0, rel_xy=(2.0, -1.5)), "front_left")
        self.assertEqual(contact_zone([600, 600, 0], 0.0, rel_xy=(2.0, 1.5)), "front_right")
        self.assertEqual(contact_zone([600, 600, 0], 0.0, rel_xy=(-2.0, -1.5)), "rear_left")
        self.assertEqual(contact_zone([600, 600, 0], 0.0, rel_xy=(-2.0, 1.5)), "rear_right")

    def test_real_video_geometries_and_rotated_yaw(self):
        # Video 1 live CARLA run geometry: v1 ego_yaw=15.9, rel=(2.25, -1.39); v2 ego_yaw=50.6, rel=(-0.30, 1.96)
        self.assertEqual(contact_zone([341.4, -295.5, 0.0], 15.9, rel_xy=(2.25, -1.39)), "front_left")
        self.assertEqual(contact_zone([341.4, -295.5, 0.0], 50.6, rel_xy=(-0.30, 1.96)), "right_side")
        # Video 2 live CARLA run geometry: v2 yaw=90 rel=(2.0, 4.43) vs v3 yaw=0 rel=(4.43, -2.0)
        self.assertEqual(contact_zone([-10257.5, 199.7, 0.0], 90.0, rel_xy=(2.0, 4.43)), "front_right")
        self.assertEqual(contact_zone([-10257.5, 199.7, 0.0], 0.0, rel_xy=(4.43, -2.0)), "front")

    def test_zones_compatible_and_pair_lookup_with_other_rel_fallback(self):
        self.assertTrue(zones_compatible("front_left", "side"))
        self.assertTrue(zones_compatible("front_left", "left_side"))
        self.assertFalse(zones_compatible("front_left", "right_side"))
        self.assertFalse(zones_compatible("right_side", "left_side"))
        events = [{"actor": "v1", "other_actor": "v2", "impulse_vec": [0, 900, 0],
                   "ego_yaw": 0.0, "rel": [2.2, -1.4], "other_yaw": 30.0, "other_rel": [-0.3, 1.9]}]
        self.assertEqual(pair_contact_zones(events, "v1", "v2"), {"v1": "front_left", "v2": "right_side"})


if __name__ == "__main__":
    unittest.main()
