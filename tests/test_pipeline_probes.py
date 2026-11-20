import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import local_pipeline, road_probes, scene_spec


def frames(*a, **k):
    return [b"jpg"]


class TestCrashSearch(unittest.TestCase):
    def test_flat_scores_still_give_spread_out_candidates(self):
        scores = [{"t": 0.5 * (i + 1), "score": 0.1} for i in range(47)]
        c = road_probes.pick_crash_candidates(scores)
        self.assertTrue(1 <= len(c) <= 3)
        self.assertTrue(all(abs(a - b) >= 1.0 for i, a in enumerate(c) for b in c[i + 1:]))

    def test_strongest_peaks_are_picked(self):
        scores = [{"t": 0.5 * (i + 1), "score": 0.1} for i in range(47)]
        scores[13]["score"] = 0.9            # t = 7.0
        scores[23]["score"] = 0.7            # t = 12.0
        c = road_probes.pick_crash_candidates(scores)
        self.assertIn(7.0, c)
        self.assertIn(12.0, c)
        self.assertEqual(c[0], 7.0)

    def test_empty_scores_give_no_candidates(self):
        self.assertEqual(road_probes.pick_crash_candidates([]), [])

    def test_refine_finds_the_peak_between_grid_points(self):
        t, sc = road_probes.refine_crash_time([6.5, 7.0, 9.0], lambda x: 1.0 - abs(x - 7.05))
        self.assertAlmostEqual(t, 7.05, delta=0.15)

    def test_refine_with_equal_scores_keeps_the_first_candidate(self):
        t, sc = road_probes.refine_crash_time([6.5, 7.0], lambda x: 0.3)
        self.assertEqual(t, 6.5)
        self.assertEqual(sc, 0.3)


class TestProbes(unittest.TestCase):
    def setUp(self):
        for target, repl in (("grab_frames", frames),):
            p = mock.patch.object(road_probes.video_events, target, repl)
            p.start()
            self.addCleanup(p.stop)

    def _choice(self, probs):
        p = mock.patch.object(road_probes.llm_backend, "choice_probabilities", lambda prompt, images, letters: dict(probs))
        p.start()
        self.addCleanup(p.stop)

    def test_traffic_bins_use_the_expected_count(self):
        self._choice({"A": 0.0, "B": 0.0, "C": 0.0, "D": 1.0})
        t = road_probes.traffic_bins("v.mp4", 24.0, 7.0)
        self.assertEqual(t["left_lane"], 8)
        self.assertEqual(t["ego_lane_ahead"], 8)
        self.assertEqual(t["rel_speed_kmh"], 0.0)

    def test_traffic_bins_mixed_probabilities_round_to_one(self):
        self._choice({"A": 0.5, "B": 0.5, "C": 0.0, "D": 0.0})
        self.assertEqual(road_probes.traffic_bins("v.mp4", 24.0, 7.0)["left_lane"], 1)

    def test_traffic_bins_empty_road(self):
        self._choice({"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0})
        t = road_probes.traffic_bins("v.mp4", 24.0, None)
        self.assertEqual(sum(t[k] for k in ("ego_lane_ahead", "left_lane", "right_lane", "oncoming")), 0)

    def test_every_probe_returns_none_when_the_model_fails(self):
        def boom(*a, **k):
            raise RuntimeError("server down")
        with mock.patch.object(road_probes.llm_backend, "choice_probabilities", boom), \
                mock.patch.object(road_probes.llm_backend, "yes_probability", boom):
            self.assertIsNone(road_probes.traffic_bins("v.mp4", 24.0, 7.0))
            self.assertIsNone(road_probes.road_edges("v.mp4", 24.0))
            self.assertIsNone(road_probes.curve_direction("v.mp4", 24.0))

    def test_road_edges_barrier_on_the_right_and_two_way(self):
        def choice(prompt, images, letters):
            if "RIGHT edge" in prompt:
                return {"A": 0.9, "B": 0.05, "C": 0.05}
            return {"A": 0.0, "B": 0.1, "C": 0.9, "D": 0.0}
        with mock.patch.object(road_probes.llm_backend, "choice_probabilities", lambda p, i, l: choice(p, i, l)), \
                mock.patch.object(road_probes.llm_backend, "yes_probability", lambda p, i: 0.9):
            e = road_probes.road_edges("v.mp4", 24.0)
        self.assertEqual(e["edge_right"], "barrier")
        self.assertIs(e["one_way"], False)

    def test_road_edges_no_oncoming_traffic_means_one_way(self):
        with mock.patch.object(road_probes.llm_backend, "choice_probabilities", lambda p, i, l: {k: 1.0 / len(l) for k in l}), \
                mock.patch.object(road_probes.llm_backend, "yes_probability", lambda p, i: 0.05):
            e = road_probes.road_edges("v.mp4", 24.0)
        self.assertIs(e["one_way"], True)

    def test_curve_direction_maps_letters(self):
        self._choice({"A": 0.1, "B": 0.8, "C": 0.1})
        self.assertEqual(road_probes.curve_direction("v.mp4", 24.0), "right")
        self._choice({"A": 0.7, "B": 0.1, "C": 0.2})
        self.assertEqual(road_probes.curve_direction("v.mp4", 24.0), "left")
        self._choice({"A": 0.1, "B": 0.1, "C": 0.8})
        self.assertEqual(road_probes.curve_direction("v.mp4", 24.0), "straight")


class TestPipelineHelpers(unittest.TestCase):
    CRASH = {"collision": True, "ego_involved": True, "second_vehicle": None,
             "first_vehicle": {"type": "suv", "color": "black", "approach": "merging_from_left", "speed": "slow"}}

    def _spec(self, traffic):
        spec, _ = scene_spec.normalize({"actors": [{"id": "a1", "type": "car", "relation": "ahead", "lane": "left", "distance_m": 100,
                                                    "speed_kmh": 50, "behavior": "driving"}], "traffic": traffic})
        return spec

    def test_measured_traffic_replaces_the_models_ahead_actors(self):
        spec = local_pipeline.apply_crash(self._spec({"left_lane": 3}), dict(self.CRASH), 7.0)
        self.assertEqual([a["id"] for a in spec["actors"]], ["c1"])

    def test_without_measured_traffic_the_models_actors_are_kept(self):
        spec = local_pipeline.apply_crash(self._spec({}), dict(self.CRASH), 7.0)
        self.assertEqual([a["id"] for a in spec["actors"]], ["c1", "a1"])

    def test_speed_series_converts_overlay_speeds(self):
        caps = [{"t": 1.0, "speed_kmh": 36.0}, {"t": 2.0, "speed_kmh": None}, {"t": 3.0, "speed_kmh": 72.0}]
        s = local_pipeline.speed_series(caps, [1.0, 2.0, 3.0])
        self.assertAlmostEqual(s[0], 10.0, places=1)
        self.assertIsNotNone(s[1])
        self.assertAlmostEqual(s[2], 20.0, places=1)

    def test_speed_series_without_any_overlay_speed_is_unknown(self):
        self.assertEqual(local_pipeline.speed_series([{"t": 1.0, "speed_kmh": None}], [1.0, 2.0]), [None, None])

    def test_report_names_contact_direction_retries_and_failed_rules(self):
        spec = self._spec({})
        ready = {"status": "best_effort", "iterations": 4, "expect": {"type": "collision", "actors": ["v1", "v2"], "time_s": 7.0},
                 "history": [{"detail": {"reasons": ["no contact between v1 and v2"]}},
                             {"detail": {"reasons": ["direction: front contact, expected side"], "direction": "front", "contact_t_sim": 7.3}}],
                 "map_fallback": "CARLA rejected the curved map"}
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            txt = local_pipeline.report(tmp, spec, ready)
        self.assertIn("best_effort after 4 run(s)", txt)
        self.assertIn("direction: front", txt)
        self.assertIn("failed rule: direction: front contact, expected side", txt)
        self.assertIn("CARLA rejected the curved map", txt)

    def test_parse_caption_handles_prose_and_explicit_speeds(self):
        self.assertAlmostEqual(local_pipeline.parse_caption(1.0, "The road has two lanes. The speed is 54 MPH.")["speed_kmh"], 54 * 1.609, places=2)
        self.assertAlmostEqual(local_pipeline.parse_caption(1.0, "Car ahead. The speed is 50MPH.")["speed_kmh"], 50 * 1.609, places=2)
        self.assertAlmostEqual(local_pipeline.parse_caption(1.0, "Highway. speed: 80 km/h")["speed_kmh"], 80.0, places=2)
        self.assertAlmostEqual(local_pipeline.parse_caption(1.0, "Highway SPEED=45")["speed_kmh"], 45 * 1.609, places=2)
        self.assertIsNone(local_pipeline.parse_caption(1.0, "The road has 2 lanes ahead.")["speed_kmh"])

    def test_build_prompt2_forwards_t_focus(self):
        caps = [{"t": float(i), "text": "frame %d" % i} for i in range(100)]
        prompt = local_pipeline.build_prompt2(caps, 100.0, t_focus=33.0)
        self.assertIn("t=33.0s:", prompt)
        self.assertIn("t=32.0s:", prompt)
        self.assertIn("t=34.0s:", prompt)

    def test_reconcile_traffic_zeroes_impossible_lanes(self):
        spec = {"road": {"one_way": True}, "ego": {"lane": "right"},
                "traffic": {"ego_lane_ahead": 2, "left_lane": 1, "right_lane": 2, "oncoming": 3}}
        local_pipeline.reconcile_traffic(spec)
        self.assertEqual(spec["traffic"]["oncoming"], 0)
        self.assertEqual(spec["traffic"]["right_lane"], 0)
        self.assertEqual(spec["traffic"]["left_lane"], 1)

    def test_llm_backend_places_images_before_text(self):
        from agents import llm_backend
        content = llm_backend._build_content("question?", [b"img1", b"img2"])
        self.assertEqual([c["type"] for c in content], ["image_url", "image_url", "text"])
        self.assertEqual(content[-1]["text"], "question?")

    def test_traffic_bins_active_window_single_grab_per_frame_and_tail_bias(self):
        grabbed = []

        def fake_grab(video, times, width=512, **kw):
            grabbed.extend(times)
            return [b"jpg"]

        with mock.patch.object(road_probes.video_events, "grab_frames", fake_grab), \
                mock.patch.object(road_probes.llm_backend, "choice_probabilities",
                                  lambda p, i, l: {"A": 0.55, "B": 0.45, "C": 0.0, "D": 0.0}):
            t = road_probes.traffic_bins("v.mp4", 24.0, 6.5, n=6)
        self.assertEqual(t["left_lane"], 0)
        self.assertTrue(1 <= len(grabbed) <= 6)
        self.assertEqual(len(grabbed), len(set(grabbed)))
        self.assertTrue(all(x <= 8.5 for x in grabbed))

    def test_analyse_crash_window_weights_center_frames_higher(self):
        def fake_choice(prompt, images, letters):
            if letters == "ABC":
                return {"A": 0.1, "B": 0.8, "C": 0.1}
            return getattr(fake_choice, "seq").pop(0)

        # 4 frames with fps=3.0, span=1.2 -> [5.3, 5.68, 6.07, 6.45]; last 2 near t_peak outweigh first 2
        early = [
            {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0},
            {"A": 0.1, "B": 0.9, "C": 0.0, "D": 0.0, "E": 0.0, "F": 0.0, "G": 0.0, "H": 0.0},
            {"A": 0.0, "B": 1.0, "C": 0.0, "D": 0.0, "E": 0.0, "F": 0.0, "G": 0.0, "H": 0.0},
        ]
        near_impact = [
            {"A": 0.0, "B": 1.0, "C": 0.0, "D": 0.0, "E": 0.0},
            {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0, "F": 0.0, "G": 0.0, "H": 0.0},
            {"A": 1.0, "B": 0.0, "C": 0.0, "D": 0.0, "E": 0.0, "F": 0.0, "G": 0.0, "H": 0.0},
        ]
        fake_choice.seq = early + early + near_impact + near_impact
        with mock.patch("agents.video_events.grab_frames", lambda *a, **k: [b"jpg"]), \
                mock.patch("agents.video_events.lateral_entry_flow", return_value=(0.0, 0.0)), \
                mock.patch("agents.llm_backend.choice_probabilities", fake_choice):
            crash, err, _ = local_pipeline.analyse_crash_window("v.mp4", 6.5, 24.0, fps=3.0, span=1.2)
        self.assertIsNone(err)
        self.assertEqual(crash["first_vehicle"]["approach"], "merging_from_left")
        self.assertEqual(crash["first_vehicle"]["type"], "car")
        self.assertEqual(crash["first_vehicle"]["color"], "white")

    def test_video_events_grab_frames_caches_and_crops_bottom(self):
        import cv2
        import numpy as np
        from agents import video_events
        video_events._VIDEO_META.clear()
        video_events._FRAME_CACHE.clear()
        with tempfile.TemporaryDirectory() as tmp:
            vp = str(Path(tmp) / "clip.avi")
            vw = cv2.VideoWriter(vp, cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (160, 120))
            for _ in range(10):
                vw.write(np.full((120, 160, 3), 180, dtype=np.uint8))
            vw.release()
            full = video_events.grab_frames(vp, [0.2], width=80)
            cropped = video_events.grab_frames(vp, [0.2], width=80, crop_bottom=0.25)
            self.assertEqual(len(full), 1)
            self.assertEqual(len(cropped), 1)
            im_full = cv2.imdecode(np.frombuffer(full[0], np.uint8), cv2.IMREAD_COLOR)
            im_crop = cv2.imdecode(np.frombuffer(cropped[0], np.uint8), cv2.IMREAD_COLOR)
            self.assertEqual(im_full.shape[:2], (60, 80))
            self.assertEqual(im_crop.shape[:2], (45, 80))
            with mock.patch.object(cv2, "VideoCapture", side_effect=AssertionError("should use cache")):
                cached = video_events.grab_frames(vp, [0.2], width=80)
            self.assertEqual(cached, full)

    def test_refine_pre_crash_window_ignores_post_crash_stopped(self):
        spec, _ = scene_spec.normalize({"ego": {"behavior": "cruise", "speed_kmh": 0.0}})
        events = [{"t": 6.5, "type": "stopped", "until": 11.8, "score": 5.3},
                  {"t": 5.5, "type": "brake_to_stop", "score": 0.9}]
        out = local_pipeline.refine(spec, [{"text": "intersection ahead"}], events, 12.0, t_crash=4.2)
        self.assertEqual(out["ego"]["behavior"], "cruise")
        self.assertEqual(out["ego"]["speed_kmh"], 50.0)

        spec_stopped, _ = scene_spec.normalize({"ego": {"behavior": "cruise", "speed_kmh": 50.0}})
        pre_events = [{"t": 0.0, "type": "stopped", "until": 5.0, "score": 5.0}]
        out2 = local_pipeline.refine(spec_stopped, [{"text": "intersection ahead"}], pre_events, 12.0, t_crash=4.2)
        self.assertEqual(out2["ego"]["behavior"], "stopped")
        self.assertEqual(out2["ego"]["speed_kmh"], 0.0)

    def test_video_events_detects_v3_replay_and_moving_approach(self):
        import os
        from agents import video_events
        v3 = Path(__file__).resolve().parents[1] / "data" / "user_videos" / "v3" / "v3.mp4"
        if not os.path.exists(v3):
            self.skipTest("v3.mp4 not present")
        ev = video_events.analyze(str(v3))
        self.assertLess(ev["effective_duration_s"], 10.0)
        self.assertTrue(any(e["type"] == "replay" for e in ev["events"]))
        early_stopped = [e for e in ev["events"] if e["type"] == "stopped" and e["t"] < 3.5]
        self.assertEqual(early_stopped, [])

    def test_lateral_entry_flow_detects_v3_left_to_right(self):
        import os
        from agents import video_events
        v3 = Path(__file__).resolve().parents[1] / "data" / "user_videos" / "v3" / "v3.mp4"
        if not os.path.exists(v3):
            self.skipTest("v3.mp4 not present")
        l2r, r2l = video_events.lateral_entry_flow(str(v3), 4.55)
        self.assertGreater(l2r, 5.0)
        self.assertGreater(l2r, 1.5 * r2l)

    def test_analyse_crash_window_uses_lateral_flow_to_override_right_bias(self):
        def right_biased(prompt, images, letters):
            if "Where" in prompt or "where" in prompt:
                return {"A": 0.1, "B": 0.35, "C": 0.45, "D": 0.05, "E": 0.05}
            return {k: (1.0 if k == "B" else 0.0) for k in letters}

        with mock.patch("agents.video_events.grab_frames", lambda *a, **k: [b"jpg"]), \
                mock.patch("agents.video_events.lateral_entry_flow", return_value=(200.0, 20.0)), \
                mock.patch("agents.llm_backend.choice_probabilities", right_biased):
            crash, err, _ = local_pipeline.analyse_crash_window("v.mp4", 4.5, 10.0, roads={"intersection": 0.7})
        self.assertIsNone(err)
        self.assertEqual(crash["first_vehicle"]["approach"], "crossing_from_left")
        self.assertEqual(crash["probabilities"]["entry_flow"], {"left_to_right": 200.0, "right_to_left": 20.0})

    def test_apply_crash_sets_braking_lead_and_realistic_cutin_speeds(self):
        ahead_crash = {"collision": True, "ego_involved": True, "second_vehicle": None,
                       "first_vehicle": {"type": "car", "color": "white", "approach": "ahead", "speed": "slow"}}
        s1 = local_pipeline.apply_crash(self._spec({}), ahead_crash, 5.0, ego_speed=90.0)
        self.assertEqual(s1["actors"][0]["behavior"], "braking")
        self.assertEqual(s1["actors"][0]["speed_kmh"], 54.0)

        cut_crash = {"collision": True, "ego_involved": True, "second_vehicle": None,
                     "first_vehicle": {"type": "suv", "color": "white", "approach": "merging_from_left", "speed": "slow"}}
        s2 = local_pipeline.apply_crash(
            self._spec({"left_lane": 3, "right_lane": 2, "rel_speed_kmh": 0.0}),
            cut_crash, 7.0, ego_speed=80.0)
        self.assertEqual(s2["actors"][0]["speed_kmh"], 80.0)
        self.assertEqual(s2["traffic"]["right_lane"], 0)
        self.assertEqual(s2["traffic"]["rel_speed_kmh"], -15.0)

    def test_analyse_crash_window_resolves_split_lateral_vs_from_behind(self):
        def split_votes(prompt, images, letters):
            if "Where" in prompt or "where" in prompt:
                return {"A": 0.20, "B": 0.26, "C": 0.22, "D": 0.05, "E": 0.27}
            return {k: (1.0 if k == "B" else 0.0) for k in letters}

        with mock.patch("agents.video_events.grab_frames", lambda *a, **k: [b"jpg"]), \
                mock.patch("agents.video_events.lateral_entry_flow", return_value=(279.4, 130.6)), \
                mock.patch("agents.llm_backend.choice_probabilities", split_votes):
            crash, err, _ = local_pipeline.analyse_crash_window("v.mp4", 4.55, 10.0, roads={"intersection": 0.72})
        self.assertIsNone(err)
        self.assertEqual(crash["first_vehicle"]["approach"], "crossing_from_left")

    def test_ego_speed_near_ignores_sparse_hallucinated_hud_speeds(self):
        sparse_caps = [{"t": float(i), "text": "road", "speed_kmh": (27.0 if i < 3 else None)} for i in range(20)]
        self.assertIsNone(local_pipeline.ego_speed_near(sparse_caps, 1.0))

        dense_caps = [{"t": float(i), "text": "road", "speed_kmh": (51.0 if i < 12 else None)} for i in range(20)]
        self.assertEqual(local_pipeline.ego_speed_near(dense_caps, 1.0), 51.0)

    def test_refine_does_not_turn_low_probe_road_into_crossroad_from_caption_echo(self):
        spec, _ = scene_spec.normalize({"road": {"type": "straight", "lanes_per_direction": 1}})
        caps = [{"t": 12.0 + 0.5 * i, "text": "it is an intersection with another road", "speed_kmh": None}
                for i in range(10)]
        roads = {"lanes": 1, "lane_probs": {"A": 0.6, "B": 0.3, "C": 0.1}, "intersection": 0.33}
        out = local_pipeline.refine(spec, caps, [], 20.0, roads=roads, t_crash=14.2)
        self.assertEqual(out["road"]["type"], "straight")
        self.assertFalse(out["road"]["traffic_signals"])

    def test_analyse_crash_window_maneuver_probe_detects_straight_road_crossing_and_color(self):
        calls = {"n": 0}

        def crossing_votes(prompt, images, letters):
            if letters == "ABC":
                idx = calls["n"]
                calls["n"] += 1
                return {"A": 0.75, "B": 0.20, "C": 0.05} if idx < 2 else {"A": 0.03, "B": 0.94, "C": 0.03}
            if letters == "ABCDE":
                return {"A": 0.10, "B": 0.25, "C": 0.45, "D": 0.05, "E": 0.15}
            if "colour" in prompt:
                idx = calls["n"] - 1
                return {"A": 0.05, "B": 0.90, "C": 0.05, "D": 0.0, "E": 0.0, "F": 0.0, "G": 0.0, "H": 0.0} if idx < 2 else \
                    {"A": 0.10, "B": 0.05, "C": 0.85, "D": 0.0, "E": 0.0, "F": 0.0, "G": 0.0, "H": 0.0}
            return {k: (1.0 if k == "A" else 0.0) for k in letters}

        with mock.patch("agents.video_events.grab_frames", lambda *a, **k: [b"jpg"]), \
                mock.patch("agents.video_events.lateral_entry_flow", return_value=(49.8, 6.9)), \
                mock.patch("agents.llm_backend.choice_probabilities", crossing_votes):
            crash, err, _ = local_pipeline.analyse_crash_window("v.mp4", 14.2, 29.0, roads={"intersection": 0.33}, fps=3.0, span=1.2)
        self.assertIsNone(err)
        self.assertEqual(crash["first_vehicle"]["approach"], "crossing_from_left")
        self.assertEqual(crash["first_vehicle"]["color"], "black")
        self.assertIn("maneuver", crash["probabilities"])
        spec, _ = scene_spec.normalize({"road": {"type": "straight"}})
        local_pipeline.apply_crash(spec, crash, 14.2)
        self.assertEqual(spec["road"]["type"], "straight")

    def test_ego_motion_metrics_detects_stopped_ego_on_2_and_moving_on_v1_v3(self):
        import os
        from agents import video_events
        v2 = Path(__file__).resolve().parents[1] / "data" / "user_videos" / "2.mp4"
        v1 = Path(__file__).resolve().parents[1] / "data" / "user_videos" / "v1" / "v1.mp4"
        if not os.path.exists(v2) or not os.path.exists(v1):
            self.skipTest("user videos not present")
        m2 = video_events.ego_motion_metrics(str(v2), 14.2, 29.67)
        m1 = video_events.ego_motion_metrics(str(v1), 8.6, 24.0)
        self.assertTrue(m2["ego_stopped"], m2)
        self.assertTrue(m2["camera_steady"], m2)
        self.assertFalse(m1["ego_stopped"], m1)
        self.assertFalse(m1["camera_steady"], m1)


if __name__ == "__main__":
    unittest.main()
