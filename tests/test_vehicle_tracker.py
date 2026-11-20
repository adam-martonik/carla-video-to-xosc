import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import vehicle_tracker


def _obs(t, box, x, z, label="black car", vtype="car", color="black"):
    bw = box[2] - box[0]
    bh = box[3] - box[1]
    return {"t": float(t), "box": list(box), "label": label, "type": vtype, "color": color,
            "X": float(x), "Z": float(z), "ar": round(bw / max(bh, 1.0), 2)}


class TestVehicleTracker(unittest.TestCase):
    def test_parse_vehicle_label_maps_colors_and_types(self):
        self.assertEqual(vehicle_tracker.parse_vehicle_label("brown van"), ("brown", "van"))
        self.assertEqual(vehicle_tracker.parse_vehicle_label("white police car"), ("white", "car"))
        self.assertEqual(vehicle_tracker.parse_vehicle_label("silver minivan"), ("silver", "van"))
        self.assertEqual(vehicle_tracker.parse_vehicle_label("gray pickup truck"), ("gray", "truck"))
        self.assertEqual(vehicle_tracker.parse_vehicle_label("dark unknown object"), (None, "car"))

    def test_analyze_tracks_detects_third_party_straight_crossing_and_left_ego_lane(self):
        frame_dets = [
            (12.0, [_obs(12.0, [253, 126, 339, 202], -0.5, 11.0, "white police car", "car", "white"),
                    _obs(12.0, [65, 128, 212, 233], -2.6, 8.0, "black car", "car", "black")]),
            (12.7, [_obs(12.7, [87, 126, 389, 238], -1.1, 7.0, "black sedan", "car", "black"),
                    _obs(12.7, [265, 128, 334, 168], -0.9, 24.0, "white sedan", "car", "white")]),
            (13.4, [_obs(13.4, [194, 125, 583, 241], +0.9, 7.0, "brown sedan", "car", "brown")]),
            (14.1, [_obs(14.1, [303, 125, 640, 247], +1.9, 7.0, "brown sedan", "car", "brown"),
                    _obs(14.1, [255, 133, 333, 199], -0.6, 12.0, "white sedan", "car", "white")]),
            (14.8, [_obs(14.8, [290, 129, 517, 245], +1.1, 7.0, "black sedan", "car", "black"),
                    _obs(14.8, [528, 85, 640, 205], +4.6, 9.0, "brown suv", "suv", "brown"),
                    _obs(14.8, [254, 133, 333, 200], -0.6, 12.0, "white sedan", "car", "white")]),
        ]
        tracks = vehicle_tracker.build_tracks(frame_dets, 640)
        res = vehicle_tracker.analyze_tracks(
            tracks, 640, 14.2,
            roads={"lanes": 2, "intersection": 0.33},
            ego_motion={"pre_edge": 0.14, "imp_edge": 0.11, "imp_bg_max": 0.06, "ego_stopped": True, "camera_steady": True},
            edges={"one_way": False, "edge_right": "shoulder", "edge_left": "shoulder"},
            yellow_left=0.18,
            crop_classifier=lambda tr, obs: ("van", "brown"),
        )
        self.assertFalse(res["ego_involved"])
        self.assertEqual(res["ego_lane"], "left")
        self.assertEqual(res["first_vehicle"]["approach"], "crossing_from_left")
        self.assertEqual(res["first_vehicle"]["type"], "car")
        self.assertEqual(res["second_vehicle"]["approach"], "right_lane_straight")
        self.assertEqual(res["second_vehicle"]["type"], "van")
        self.assertEqual(res["second_vehicle"]["color"], "brown")
        stopped_ahead = [a for a in res["background_actors"] if a["relation"] == "ahead" and a["lane"] == "same"]
        self.assertEqual(len(stopped_ahead), 1)
        self.assertEqual(stopped_ahead[0]["color"], "white")
        self.assertEqual(stopped_ahead[0]["behavior"], "stopped")
        self.assertEqual(stopped_ahead[0]["distance_m"], 12.0)

    def test_analyze_tracks_detects_ego_involved_left_queue_cut_in(self):
        frame_dets = [
            (6.4, [_obs(6.4, [145, 122, 230, 194], -4.0, 17.0, "black suv", "suv", "black"),
                   _obs(6.4, [225, 125, 270, 182], -3.1, 23.0, "white suv", "suv", "white")]),
            (7.1, [_obs(7.1, [141, 117, 241, 189], -4.3, 18.0, "black suv", "suv", "black"),
                   _obs(7.1, [0, 115, 97, 255], -3.7, 7.0, "black suv", "suv", "black")]),
            (7.8, [_obs(7.8, [179, 131, 257, 181], -4.7, 25.0, "white suv", "suv", "white"),
                   _obs(7.8, [0, 85, 126, 232], -3.9, 8.0, "black suv", "suv", "black")]),
            (8.5, [_obs(8.5, [0, 111, 190, 240], -3.4, 8.0, "white suv", "suv", "white"),
                   _obs(8.5, [224, 135, 273, 173], -4.6, 35.0, "silver suv", "suv", "silver")]),
        ]
        tracks = vehicle_tracker.build_tracks(frame_dets, 640)
        res = vehicle_tracker.analyze_tracks(
            tracks, 640, 8.6,
            roads={"lanes": 2, "intersection": 0.1},
            ego_motion={"pre_edge": 1.82, "imp_edge": 0.64, "imp_bg_max": 1.58, "ego_stopped": False, "camera_steady": False},
            edges={"one_way": True, "edge_right": "barrier", "edge_left": "shoulder"},
            yellow_left=0.04,
            ego_speed_kmh=51.0,
        )
        self.assertTrue(res["ego_involved"])
        self.assertEqual(res["ego_lane"], "right")
        self.assertEqual(res["first_vehicle"]["approach"], "merging_from_left")
        left_queue = [a for a in res["background_actors"] if a["relation"] == "ahead" and a["lane"] == "left"]
        self.assertGreaterEqual(len(left_queue), 2)
        self.assertEqual([a["relation"] for a in res["background_actors"]].count("oncoming"), 0)

    def test_analyze_tracks_extracts_crossroad_waiting_cross_street_vehicle(self):
        frame_dets = [
            (3.0, [_obs(3.0, [75, 386, 182, 440], -4.2, 12.0, "silver van", "van", "silver"),
                   _obs(3.0, [513, 391, 627, 442], +5.8, 13.0, "black van", "van", "black")]),
            (3.8, [_obs(3.8, [0, 344, 390, 527], -1.0, 4.0, "silver van", "van", "silver"),
                   _obs(3.8, [513, 391, 627, 442], +5.8, 13.0, "black van", "van", "black")]),
            (4.5, [_obs(4.5, [356, 372, 519, 440], +2.1, 10.0, "black suv", "suv", "black"),
                   _obs(4.5, [566, 291, 640, 487], +2.1, 4.0, "silver suv", "suv", "silver")]),
        ]
        tracks = vehicle_tracker.build_tracks(frame_dets, 640)
        res = vehicle_tracker.analyze_tracks(
            tracks, 640, 4.55,
            roads={"lanes": 2, "intersection": 0.72},
            ego_motion={"pre_edge": 14.67, "imp_edge": 5.73, "imp_bg_max": 0.49, "ego_stopped": False, "camera_steady": False},
            edges={"one_way": False, "edge_right": "open", "edge_left": "open"},
            yellow_left=0.06,
            ego_speed_kmh=50.0,
        )
        self.assertTrue(res["ego_involved"])
        waiting_cross = [a for a in res["background_actors"] if a["relation"] == "cross_right"]
        self.assertEqual(len(waiting_cross), 1)
        self.assertEqual(waiting_cross[0]["behavior"], "stopped")
        self.assertEqual(waiting_cross[0]["type"], "van")

    def test_sideways_right_cross_street_track_does_not_flip_ego_lane_to_left(self):
        frame_dets = [
            (2.35, [_obs(2.35, [471, 403, 569, 444], +5.7, 16.0, "black car", "car", "black")]),
            (3.05, [_obs(3.05, [75, 386, 182, 440], -4.2, 12.0, "silver van", "van", "silver"),
                    _obs(3.05, [471, 403, 569, 444], +5.5, 13.0, "black car", "car", "black")]),
            (3.75, [_obs(3.75, [0, 344, 390, 527], -1.0, 4.2, "silver van", "van", "silver"),
                    _obs(3.75, [440, 390, 570, 444], +5.2, 12.5, "black car", "car", "black")]),
            (4.45, [_obs(4.45, [356, 372, 519, 440], +2.1, 10.0, "black car", "car", "black")]),
        ]
        tracks = vehicle_tracker.build_tracks(frame_dets, 640)
        res = vehicle_tracker.analyze_tracks(
            tracks, 640, 4.5,
            roads={"lanes": 2, "intersection": 0.72},
            ego_motion={"pre_edge": 14.67, "imp_edge": 5.73, "imp_bg_max": 0.49, "ego_stopped": False, "camera_steady": False},
            edges={"one_way": False, "edge_right": "open", "edge_left": "open"},
            yellow_left=0.06,
            ego_speed_kmh=50.0,
        )
        self.assertEqual(res["ego_lane"], "right")
        self.assertEqual([a for a in res["background_actors"] if a["relation"] == "ahead" and a["lane"] == "right"], [])
        self.assertTrue(any(a["relation"] == "cross_right" and a["behavior"] == "stopped" for a in res["background_actors"]))

    def test_single_frame_far_left_ramp_box_filtered_on_straight_road(self):
        frame_dets = [
            (6.4, [_obs(6.4, [0, 117, 62, 197], -8.1, 15.0, "white suv", "suv", "white"),
                   _obs(6.4, [0, 115, 97, 255], -3.7, 7.0, "white suv", "suv", "white")]),
            (7.8, [_obs(7.8, [0, 111, 190, 240], -3.4, 8.0, "white suv", "suv", "white")]),
        ]
        tracks = vehicle_tracker.build_tracks(frame_dets, 640)
        res_straight = vehicle_tracker.analyze_tracks(
            tracks, 640, 8.6,
            roads={"lanes": 2, "intersection": 0.22},
            ego_motion={"pre_edge": 1.82, "imp_edge": 0.64, "imp_bg_max": 1.58, "ego_stopped": False, "camera_steady": False},
            edges={"one_way": False, "edge_right": "shoulder", "edge_left": "open"},
            yellow_left=0.04,
            ego_speed_kmh=51.0,
        )
        self.assertEqual([a for a in res_straight["background_actors"] if a["relation"] == "oncoming"], [])

        frame_cross = [
            (6.9, [_obs(6.9, [205, 114, 257, 144], -6.8, 42.0, "black suv", "suv", "black")]),
            (7.6, [_obs(7.6, [24, 73, 227, 229], -3.0, 8.0, "red suv", "suv", "red")]),
        ]
        res_cross = vehicle_tracker.analyze_tracks(
            vehicle_tracker.build_tracks(frame_cross, 640), 640, 7.7,
            roads={"lanes": 2, "intersection": 0.63},
            ego_motion={"pre_edge": 0.73, "imp_edge": 0.59, "imp_bg_max": 8.09, "ego_stopped": False, "camera_steady": False},
            edges={"one_way": False, "edge_right": "shoulder", "edge_left": "open"},
            yellow_left=0.07,
            ego_speed_kmh=50.0,
        )
        self.assertEqual(len([a for a in res_cross["background_actors"] if a["relation"] == "oncoming"]), 1)


if __name__ == "__main__":
    unittest.main()
