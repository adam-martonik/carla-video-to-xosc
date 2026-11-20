import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import scene_spec


class TestSpecAdditions(unittest.TestCase):
    def test_defaults(self):
        spec, err = scene_spec.normalize({})
        self.assertIsNone(err)
        self.assertEqual(spec["road"]["one_way"], False)
        self.assertEqual(spec["road"]["edge_right"], "shoulder")
        self.assertEqual(spec["road"]["edge_left"], "shoulder")
        self.assertIsNone(spec["road"]["centerline"])
        self.assertIsNone(spec["road"]["curve_note"])
        self.assertEqual(spec["traffic"], {"ego_lane_ahead": 0, "left_lane": 0, "right_lane": 0, "oncoming": 0, "rel_speed_kmh": 0.0})

    def test_bad_values_fall_back(self):
        spec, _ = scene_spec.normalize({"road": {"edge_right": "wall", "one_way": "yes"}})
        self.assertEqual(spec["road"]["edge_right"], "shoulder")
        self.assertIs(spec["road"]["one_way"], True)
        spec, _ = scene_spec.normalize({"road": {"one_way": "no"}})
        self.assertIs(spec["road"]["one_way"], False)

    def test_valid_edges_are_kept(self):
        spec, _ = scene_spec.normalize({"road": {"edge_right": "barrier", "edge_left": "open"}})
        self.assertEqual((spec["road"]["edge_right"], spec["road"]["edge_left"]), ("barrier", "open"))

    def test_traffic_counts_clamp_and_total_is_capped(self):
        spec, _ = scene_spec.normalize({"traffic": {"left_lane": 50, "ego_lane_ahead": 9, "right_lane": -3, "oncoming": "x", "rel_speed_kmh": 500}})
        t = spec["traffic"]
        self.assertEqual(t["right_lane"], 0)
        self.assertEqual(t["oncoming"], 0)
        self.assertLessEqual(t["left_lane"] + t["ego_lane_ahead"] + t["right_lane"] + t["oncoming"], 12)
        self.assertLessEqual(t["rel_speed_kmh"], 60.0)
        self.assertGreater(t["left_lane"], 0)

    def test_centerline_needs_two_finite_points(self):
        spec, _ = scene_spec.normalize({"road": {"centerline": [[0, 0]]}})
        self.assertIsNone(spec["road"]["centerline"])
        spec, _ = scene_spec.normalize({"road": {"centerline": [[0, 0], [float("nan"), 1]]}})
        self.assertIsNone(spec["road"]["centerline"])
        spec, _ = scene_spec.normalize({"road": {"centerline": [[0, 0], [100, 5], [200, 20]]}})
        self.assertEqual(spec["road"]["centerline"], [[0.0, 0.0], [100.0, 5.0], [200.0, 20.0]])

    def test_new_fields_survive_a_second_normalize(self):
        spec, _ = scene_spec.normalize({"road": {"edge_right": "barrier", "centerline": [[0, 0], [100, 5]], "curve_note": "x"},
                                        "traffic": {"left_lane": 4, "rel_speed_kmh": -10}})
        again, _ = scene_spec.normalize(spec)
        self.assertEqual(again["road"], spec["road"])
        self.assertEqual(again["traffic"], spec["traffic"])


if __name__ == "__main__":
    unittest.main()
