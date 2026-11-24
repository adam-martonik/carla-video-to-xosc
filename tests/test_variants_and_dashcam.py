"""Tests for colliding-vehicle dashcam mounting, timing variations, road design variations, and combined generation."""
import ast
import copy
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents import (
    intercept,
    local_pipeline,
    map_builder,
    motion_runner,
    scenario_check,
    scenario_variants,
    scene_builder,
    scene_spec,
    xosc_export,
)
from agents.road_frame import RoadFrame, lane_offset

FIX = ROOT / "tests" / "fixtures"
RESULTS = ROOT / "results"


def load_fixture(name):
    with open(FIX / (name + ".json"), encoding="utf-8") as f:
        return json.load(f)


def fake_vehicle(type_id="vehicle.dodge.charger", alive=True):
    return NS(type_id=type_id, is_alive=alive)


# ----------------------------------------------------------------------------------------- Feature 1: Dashcam on colliding vehicles
class TestDashcamSelectionAndMounts(unittest.TestCase):
    def test_third_party_crash_mounts_dashcam_on_colliders_not_bystander(self):
        """When original footage comes from a stopped bystander (v1), dashcam MUST attach to the colliding pair (v3, v2)."""
        spec = load_fixture("crash1")
        with tempfile.TemporaryDirectory() as tmp:
            plan = scene_builder.build(spec, tmp, "s")
            with open(os.path.join(tmp, "s_scene_obj.txt"), encoding="utf-8") as f:
                agents_meta = ast.literal_eval(f.readline())
        actors = {
            "v1": fake_vehicle("vehicle.dodge.charger"),      # stopped bystander camera car
            "v2": fake_vehicle("vehicle.dodge.charger"),      # crossing aggressor
            "v3": fake_vehicle("vehicle.sprinter.mercedes"),  # straight-driving struck van
        }
        chosen = motion_runner.resolve_dashcam_actors(plan, actors, agents_meta, mode="colliders")
        self.assertEqual(chosen, ["v3", "v2"])
        self.assertNotIn("v1", chosen)
        self.assertEqual(motion_runner.resolve_dashcam_actors(plan, actors, agents_meta, mode="primary"), ["v3"])

    def test_ego_involved_crash_orders_ego_first_and_aggressor_second(self):
        spec = load_fixture("v1")
        with tempfile.TemporaryDirectory() as tmp:
            plan = scene_builder.build(spec, tmp, "s")
        actors = {
            "v1": fake_vehicle("vehicle.dodge.charger"),
            "v2": fake_vehicle("vehicle.nissan.patrol"),
            "v3": fake_vehicle("vehicle.dodge.charger"),
        }
        chosen = motion_runner.resolve_dashcam_actors(plan, actors, mode="colliders")
        self.assertEqual(chosen, ["v1", "v2"])

    def test_pedestrian_collision_mounts_dashcam_only_on_vehicle(self):
        plan = {"ego": "v1", "expect": {"type": "collision", "actors": ["v1", "v2"]}, "agents": {"v1": {}, "v2": {}}}
        actors = {"v1": fake_vehicle("vehicle.dodge.charger"), "v2": fake_vehicle("walker.pedestrian.0001")}
        self.assertEqual(motion_runner.resolve_dashcam_actors(plan, actors, mode="colliders"), ["v1"])

    def test_third_party_no_aggressor_keeps_dashcam_on_surviving_collider(self):
        """Even when the aggressor (c1) does not appear at all, third-party dashcam stays on c2 (the van), not bystander v1."""
        spec = load_fixture("crash1")
        vspec = scenario_variants.apply_timing_variant(spec, "no_aggressor")
        with tempfile.TemporaryDirectory() as tmp:
            plan = scene_builder.build(vspec, tmp, "s")
            with open(os.path.join(tmp, "s_scene_obj.txt"), encoding="utf-8") as f:
                agents_meta = ast.literal_eval(f.readline())
        # c1 removed, so c2 became v2!
        self.assertEqual(plan["idmap"]["c2"], "v2")
        self.assertEqual(plan["dashcam_actors"], ["v2"])
        actors = {"v1": fake_vehicle("vehicle.dodge.charger"), "v2": fake_vehicle("vehicle.sprinter.mercedes")}
        chosen = motion_runner.resolve_dashcam_actors(plan, actors, agents_meta, mode="colliders")
        self.assertEqual(chosen, ["v2"])
        self.assertNotIn("v1", chosen)

    def test_dashcam_mount_scales_above_cab_for_suv_van_and_truck(self):
        car_x, _, car_z, _ = motion_runner.dashcam_mount_transform("car")
        suv_x, _, suv_z, _ = motion_runner.dashcam_mount_transform("suv")
        van_x, _, van_z, _ = motion_runner.dashcam_mount_transform("van")
        trk_x, _, trk_z, _ = motion_runner.dashcam_mount_transform("truck")
        self.assertLess(car_z, suv_z)
        self.assertLess(suv_z, van_z)
        self.assertLess(van_z, trk_z)
        self.assertLess(car_x, van_x)
        self.assertLess(van_x, trk_x)

    def test_delayed_already_moving_start_trigger_detected(self):
        """Regression test: already_moving vehicles with start.t > 0 must wait for their start trigger instead of launching at t=0."""
        self.assertFalse(motion_runner._has_delayed_start({"mode": "drive", "already_moving": True}))
        self.assertFalse(motion_runner._has_delayed_start({"mode": "drive", "start": {"type": "time", "t": 0.0}}))
        self.assertTrue(motion_runner._has_delayed_start({"mode": "drive", "already_moving": True, "start": {"type": "time", "t": 1.8}}))
        self.assertTrue(motion_runner._has_delayed_start({"mode": "drive", "start": {"type": "ttc", "seconds": 2.0}}))

    def test_xosc_export_ego_actor_override_assigns_hero_to_collider(self):
        def samples(x0, y, v):
            return [[round(i * 0.05, 3), x0 + v * i * 0.05, y, 0.2, 0.0, v] for i in range(40)]
        rec = {"frame": "carla", "dt": 0.05, "actors": {
            "v1": {"type": "car", "blueprint": "vehicle.dodge.charger", "samples": samples(65.0, 1.75, 0.0)},
            "v3": {"type": "van", "blueprint": "vehicle.sprinter.mercedes", "samples": samples(30.0, 5.25, 13.9)},
        }}
        sc, _ = xosc_export.build_scenario(rec, {"ego": "v1"}, "s.xodr", ego_actor="v3")
        ents = {e["name"]: e for e in sc["entities"]}
        self.assertTrue(ents["v3"]["ego"])
        self.assertFalse(ents["v1"]["ego"])


# ----------------------------------------------------------------------------------------- Feature 2: Timing variations
class TestTimingVariations(unittest.TestCase):
    def _build(self, spec):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        plan = scene_builder.build(spec, tmp.name, "s")
        with open(os.path.join(tmp.name, "s_scene_obj.txt"), encoding="utf-8") as f:
            objs = ast.literal_eval(f.readline())
        return plan, objs

    def test_cut_in_timing_variants_cover_shifted_collision_miss_and_no_aggressor(self):
        base = load_fixture("v1")
        # Earlier / later collision keep side-contact collision expectation with updated t_event_s
        early = scenario_variants.apply_timing_variant(base, "earlier_collision")
        late = scenario_variants.apply_timing_variant(base, "later_collision")
        p_e, _ = self._build(early)
        p_l, _ = self._build(late)
        self.assertEqual(p_e["expect"]["type"], "collision")
        self.assertEqual(p_l["expect"]["type"], "collision")
        self.assertLess(p_e["agents"]["v2"]["intercept"]["t_event_s"], 7.0)
        self.assertGreater(p_l["agents"]["v2"]["intercept"]["t_event_s"], 7.0)

        # Clean misses (early ahead & late behind) expect "miss" (no collision) with non-zero longitudinal offset
        m_early = scenario_variants.apply_timing_variant(base, "miss_early")
        m_late = scenario_variants.apply_timing_variant(base, "miss_late")
        p_me, obj_me = self._build(m_early)
        p_ml, obj_ml = self._build(m_late)
        self.assertEqual(p_me["expect"]["type"], "miss")
        self.assertEqual(p_ml["expect"]["type"], "miss")
        self.assertGreater(p_me["agents"]["v2"]["intercept"]["long_offset_m"], 10.0)
        self.assertLess(p_ml["agents"]["v2"]["intercept"]["long_offset_m"], -10.0)
        self.assertGreaterEqual(obj_ml["v2"]["location"][0], 3.0)      # still safely on the road behind ego

        # Intercept controller honors explicit longitudinal offset
        ic = p_ml["agents"]["v2"]["intercept"]
        ctl = intercept.CutInController(intercept.CutInConfig(
            lane_src_d=ic["lane_src_d"], lane_dst_d=ic["lane_dst_d"], t_event_s=ic["t_event_s"],
            long_offset_m=ic["long_offset_m"],
        ))
        _, v_t = ctl.update(0.0, (25.0, -5.25, 14.0), (11.0, -1.75, 14.0), False)
        self.assertAlmostEqual(v_t, 14.0, delta=0.5)

        # No aggressor: c1 removed completely, background actor kept, event type == none
        no_agg = scenario_variants.apply_timing_variant(base, "no_aggressor")
        p_na, obj_na = self._build(no_agg)
        self.assertEqual(p_na["expect"]["type"], "none")
        self.assertNotIn("c1", p_na["idmap"])
        self.assertIn("a1", p_na["idmap"])
        self.assertIn("v1", obj_na)

    def test_crossroad_crossing_timing_variants_stay_outside_junction_and_shift_cleanly(self):
        base = load_fixture("v3")
        half_j = 2 * 3.5 + 4.0
        for name, exp_type in (
            ("original", "collision"),
            ("earlier_collision", "collision"),
            ("later_collision", "collision"),
            ("near_miss_early", "near_miss"),
            ("near_miss_late", "near_miss"),
            ("miss_early", "miss"),
            ("miss_late", "miss"),
            ("no_aggressor", "none"),
        ):
            vspec = scenario_variants.apply_timing_variant(base, name)
            plan, objs = self._build(vspec)
            self.assertEqual(plan["expect"]["type"], exp_type, name)
            if name != "no_aggressor":
                v2_x = objs["v2"]["location"][0]
                self.assertLessEqual(v2_x, 100.0 - half_j - 1.0, (name, v2_x))
                self.assertGreaterEqual(v2_x, 3.0, (name, v2_x))
            else:
                self.assertEqual(list(objs.keys()), ["v1"])

        # Early miss arrives strictly before ego; late miss arrives strictly after ego
        p_me, obj_me = self._build(scenario_variants.apply_timing_variant(base, "miss_early"))
        p_ml, obj_ml = self._build(scenario_variants.apply_timing_variant(base, "miss_late"))
        t_ego_me = (-obj_me["v2"]["location"][1] - (-obj_me["v1"]["location"][1])) / (50.0 / 3.6)
        t_agg_me = (obj_me["v1"]["location"][0] - obj_me["v2"]["location"][0]) / (p_me["agents"]["v2"]["speed_kmh"] / 3.6)
        self.assertLess(t_agg_me, t_ego_me - 1.5)
        t_ego_ml = (-obj_ml["v2"]["location"][1] - (-obj_ml["v1"]["location"][1])) / (50.0 / 3.6)
        t_agg_ml = (obj_ml["v1"]["location"][0] - obj_ml["v2"]["location"][0]) / (p_ml["agents"]["v2"]["speed_kmh"] / 3.6)
        self.assertGreater(t_agg_ml, t_ego_ml + 1.5)

    def test_third_party_straight_crossing_timing_variants(self):
        base = load_fixture("crash1")
        # Late miss gives c1 a delayed start so c2 clears well ahead
        ml = scenario_variants.apply_timing_variant(base, "miss_late")
        p_ml, _ = self._build(ml)
        self.assertEqual(p_ml["expect"]["type"], "miss")
        self.assertEqual(p_ml["expect"]["actors"], ["v2", "v3"])
        self.assertGreater(p_ml["agents"]["v2"]["start"]["t"], 1.5)

        # Early miss shifts c2 back (and expands s_ego) so c1 crosses before c2 arrives
        me = scenario_variants.apply_timing_variant(base, "miss_early")
        p_me, obj_me = self._build(me)
        self.assertEqual(p_me["expect"]["type"], "miss")
        self.assertGreaterEqual(obj_me["v3"]["location"][0], 5.0)
        self.assertGreater(obj_me["v1"]["location"][0], 65.0)

        # No aggressor preserves c2 driving down the right lane and drops c1
        na = scenario_variants.apply_timing_variant(base, "no_aggressor")
        p_na, obj_na = self._build(na)
        self.assertEqual(p_na["expect"]["type"], "none")
        self.assertNotIn("c1", p_na["idmap"])
        self.assertIn("c2", p_na["idmap"])
        self.assertEqual(p_na["agents"][p_na["idmap"]["c2"]]["mode"], "drive")
        self.assertGreaterEqual(obj_na[p_na["idmap"]["c2"]]["location"][0], 5.0)

    def test_custom_dt_and_shift_timing_strings(self):
        base = load_fixture("v3")
        p1, _ = self._build(scenario_variants.apply_timing_variant(base, "dt:+1.8"))
        self.assertEqual(p1["expect"]["type"], "miss")
        p2, _ = self._build(scenario_variants.apply_timing_variant(base, "dt:-0.8"))
        self.assertEqual(p2["expect"]["type"], "near_miss")
        p3, _ = self._build(scenario_variants.apply_timing_variant(base, "shift:-1.0"))
        self.assertEqual(p3["expect"]["type"], "collision")

    def test_scenario_check_evaluates_miss_and_multi_actor_none(self):
        exp_miss = {"type": "miss", "actors": ["v1", "v2"], "time_s": 5.0, "window_s": 1.5}
        ok, detail = scenario_check.evaluate(exp_miss, {"events": [], "min_pair_dist_m": {"v1-v2": 22.0}})
        self.assertTrue(ok, detail)
        ok_hit, detail_hit = scenario_check.evaluate(exp_miss, {"events": [
            {"t_sim": 5.0, "actor": "v1", "other_actor": "v2", "other": "vehicle.x"}
        ]})
        self.assertFalse(ok_hit)
        self.assertTrue(any("miss" in r for r in detail_hit["reasons"]))

        exp_tp_none = {"type": "none", "actors": ["v2"], "time_s": 5.0}
        ok_clean, _ = scenario_check.evaluate(exp_tp_none, {"events": []})
        self.assertTrue(ok_clean)
        ok_bad, _ = scenario_check.evaluate(exp_tp_none, {"events": [
            {"t_sim": 2.5, "actor": "v2", "other_actor": "v5", "other": "vehicle.x"}
        ]})
        self.assertFalse(ok_bad)


# ----------------------------------------------------------------------------------------- Feature 3: Road design variations
class TestRoadDesignVariations(unittest.TestCase):
    def _build_and_xodr(self, spec):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        plan = scene_builder.build(spec, tmp.name, "s")
        net = os.path.join(tmp.name, "s.net.xml")
        xodr = os.path.join(tmp.name, "s.xodr")
        map_builder.net_to_xodr(net, xodr)
        for pc in (plan.get("road") or {}).get("xodr_pieces") or []:
            map_builder.apply_lane_sections(xodr, pc["sections"], pc["s1"] - pc["s0"], road_user_id=pc["road"], s0=pc["s0"])
        self.assertTrue(os.path.exists(xodr))
        self.assertGreater(os.path.getsize(xodr), 500)
        with open(os.path.join(tmp.name, "s_scene_obj.txt"), encoding="utf-8") as f:
            objs = ast.literal_eval(f.readline())
        return plan, objs

    def test_cut_in_preserves_side_collision_across_diverse_road_designs(self):
        base = load_fixture("v1")
        for preset in ("straight_2lane_twoway", "straight_3lane_highway", "curve_left", "curve_right",
                       "s_curve", "lane_drop", "highway_fork"):
            vspec = scenario_variants.apply_road_design(base, preset)
            plan, objs = self._build_and_xodr(vspec)
            self.assertEqual(plan["expect"]["type"], "collision", preset)
            self.assertEqual(plan["expect"]["contact"], "side", preset)
            self.assertIn("intercept", plan["agents"]["v2"], preset)
            self.assertEqual(plan["dashcam_actors"], ["v1", "v2"], preset)
            if preset in ("curve_left", "curve_right", "s_curve"):
                frame = RoadFrame(plan["road"]["centerline"])
                for aid in ("v1", "v2"):
                    s, d = frame.to_sd(objs[aid]["location"][0], -objs[aid]["location"][1])
                    self.assertAlmostEqual(-objs[aid]["rotation"][1], frame.heading(s), delta=2.0, msg=(preset, aid))
            if preset in ("lane_drop", "highway_fork"):
                self.assertIn("sections", plan["road"], preset)
                self.assertGreaterEqual(len(plan["road"]["sections"]), 2, preset)

    def test_crossroad_crossing_transfers_to_straight_curved_and_other_crossroads(self):
        base = load_fixture("v3")
        for preset in ("crossroad_1lane", "crossroad_3lane", "straight_2lane_twoway", "curve_left", "curve_right", "s_curve"):
            vspec = scenario_variants.apply_road_design(base, preset)
            plan, objs = self._build_and_xodr(vspec)
            self.assertEqual(plan["expect"]["type"], "collision", preset)
            self.assertEqual(plan["expect"]["actors"], ["v1", "v2"], preset)
            if preset.startswith("crossroad"):
                self.assertIsNone(plan["road"]["centerline"], preset)
            else:
                self.assertIsNotNone(plan["road"]["centerline"], preset)
                self.assertIn("polyline", plan["agents"]["v2"], preset)

    def test_third_party_crossing_transfers_cleanly_to_crossroad_and_curves(self):
        base = load_fixture("crash1")
        # Transferring straight third-party T-bone onto a 4-way crossroad preserves straight-vs-crossing T-bone!
        cr = scenario_variants.apply_road_design(base, "crossroad_2lane")
        plan_cr, objs_cr = self._build_and_xodr(cr)
        self.assertEqual(plan_cr["expect"]["type"], "collision")
        self.assertEqual(plan_cr["expect"]["actors"], ["v2", "v3"])
        self.assertEqual(plan_cr["dashcam_actors"], ["v3", "v2"])
        self.assertEqual(plan_cr["agents"]["v1"]["mode"], "static")       # bystander still stopped at stop line
        self.assertAlmostEqual(-objs_cr["v2"]["rotation"][1], 0.0, delta=1.0)    # c1 crosses eastbound
        self.assertAlmostEqual(-objs_cr["v3"]["rotation"][1], 90.0, delta=1.0)   # c2 drives northbound straight!
        # Analytical arrival times match at conflict point
        t_c2 = (-objs_cr["v2"]["location"][1] - (-objs_cr["v3"]["location"][1])) / (50.0 / 3.6)
        t_c1 = (objs_cr["v3"]["location"][0] - objs_cr["v2"]["location"][0]) / (20.0 / 3.6)
        self.assertAlmostEqual(t_c1, t_c2, places=1)

        # Also works on curves
        for preset in ("curve_left", "curve_right", "straight_3lane_highway"):
            vspec = scenario_variants.apply_road_design(base, preset)
            plan, _ = self._build_and_xodr(vspec)
            self.assertEqual(plan["expect"]["type"], "collision", preset)
            self.assertEqual(plan["expect"]["actors"], ["v2", "v3"], preset)

    def test_oncoming_corner_case_never_becomes_one_way(self):
        spec = {
            "road": {"type": "straight", "lanes_per_direction": 2, "one_way": False},
            "ego": {"type": "car", "lane": "right", "speed_kmh": 50.0, "behavior": "cruise"},
            "actors": [{"id": "c1", "type": "car", "relation": "oncoming", "lane": "same", "distance_m": 40.0, "speed_kmh": 50.0, "behavior": "driving"}],
            "event": {"type": "collision", "actor_id": "c1", "time_s": 5.0, "description": "head-on"},
        }
        norm, _ = scene_spec.normalize(spec)
        vspec = scenario_variants.apply_road_design(norm, "straight_3lane_highway")
        self.assertFalse(vspec["road"]["one_way"])

    def test_default_road_presets_never_duplicate_original_road(self):
        v2 = load_fixture("crash1")   # straight 2-lane two-way
        roads2 = scenario_variants.default_road_presets(v2, 5)
        self.assertEqual(roads2[0], "original")
        self.assertNotIn("straight_2lane_twoway", roads2)

        v3 = load_fixture("v3")       # crossroad 2-lane
        roads3 = scenario_variants.default_road_presets(v3, 5)
        self.assertEqual(roads3[0], "original")
        self.assertNotIn("crossroad_2lane", roads3)

    def test_comfort_curve_radius_prevents_unintended_curve_slowdown(self):
        cl, R = scenario_variants.make_curve_centerline(450.0, v_ms=25.0, sign=+1)
        self.assertGreaterEqual(R, 250.0)
        self.assertLess(25.0 * 25.0 / R, 2.5)   # comfortably below RouteFollower.A_LAT = 3.0
        self.assertGreater(cl[-1][1], 50.0)     # clearly bends left (+y)

        cl_s, R_s = scenario_variants.make_s_curve_centerline(450.0, v_ms=20.0, sign=+1)
        self.assertGreaterEqual(R_s, 180.0)
        xs = [p[0] for p in cl_s]
        self.assertTrue(all(b > a for a, b in zip(xs, xs[1:])))     # strictly monotonic +x


# ----------------------------------------------------------------------------------------- Feature 4: Combined Road x Timing Matrix
class TestCombinedRoadAndTimingMatrix(unittest.TestCase):
    def test_product_and_zip_combinations_and_manifest_output(self):
        base = load_fixture("crash1")
        with tempfile.TemporaryDirectory() as tmp:
            man = scenario_variants.build_variants(
                base, tmp,
                road_presets="original,curve_left,crossroad_2lane",
                timing_presets="original,miss_late,no_aggressor",
                combine="product",
                log_fn=None,
            )
            self.assertEqual(len(man["variants"]), 9)
            self.assertEqual(man["roads"], ["original", "curve_left", "crossroad_2lane"])
            self.assertEqual(man["timings"], ["original", "miss_late", "no_aggressor"])
            # Grouped by road for minimal CARLA map reloads
            road_seq = [e["road"] for e in man["variants"]]
            self.assertEqual(road_seq, ["original"] * 3 + ["curve_left"] * 3 + ["crossroad_2lane"] * 3)
            # Every variant subfolder is complete and has valid net + motion + spec
            for e in man["variants"]:
                vdir = e["dir"]
                self.assertTrue(os.path.exists(os.path.join(vdir, "video_interpreter_0000_split.net.xml")))
                self.assertTrue(os.path.exists(os.path.join(vdir, "video_interpreter_0000_split_motion.json")))
                self.assertTrue(os.path.exists(os.path.join(vdir, "video_interpreter_0000_split_spec.json")))
                self.assertNotIn("v1", e["dashcam_actors"])
            # Map signatures match across timing variants on crossroad_2lane (so CARLA reuses the loaded map!)
            cr_sigs = [local_pipeline._road_map_signature(e["dir"]) for e in man["variants"] if e["road"] == "crossroad_2lane"]
            self.assertEqual(len(set(cr_sigs)), 1)
            # Manifest and report written
            self.assertTrue(os.path.exists(os.path.join(tmp, "variants_manifest.json")))
            self.assertTrue(os.path.exists(os.path.join(tmp, "variants_report.txt")))

        # Zip combination pairs 1-to-1
        zipped = scenario_variants.generate_variants(
            base, road_presets="original,curve_left,crossroad_2lane",
            timing_presets="original,miss_late,no_aggressor", combine="zip",
        )
        self.assertEqual(len(zipped), 3)
        self.assertEqual([(v["road"], v["timing"]) for v in zipped],
                         [("original", "original"), ("curve_left", "miss_late"), ("crossroad_2lane", "no_aggressor")])

    def test_crashing_only_flag_produces_only_collision_scenarios(self):
        """--crashing-only must produce only collision variants for counts, 'all', and filtered explicit lists."""
        for fname in ("v1", "crash1", "v3"):
            base = load_fixture(fname)
            vs = scenario_variants.generate_variants(base, timing_presets="5", crashing_only=True)
            self.assertEqual(len(vs), 5, fname)
            self.assertTrue(all(v["expected_outcome"] == "collision" for v in vs), (fname, [v["timing"] for v in vs]))
            # Every variant builds cleanly with collision expectation
            t_exps = []
            for v in vs:
                with tempfile.TemporaryDirectory() as tmp:
                    plan = scene_builder.build(v["spec"], tmp, "s")
                    self.assertEqual(plan["expect"]["type"], "collision", (fname, v["timing"]))
                    t_exps.append(plan["expect"]["time_s"])
            # Must produce genuinely distinct timings (including earlier_collision on straight third-party crash1!)
            self.assertGreaterEqual(len(set(t_exps)), 3, (fname, t_exps))

        # Filtering an explicit mixed list drops misses / near-misses / no_aggressor
        filtered = scenario_variants.generate_variants(
            load_fixture("v3"),
            timing_presets="original,miss_late,no_aggressor,near_miss_early,later_collision,fast_collision",
            crashing_only=True,
        )
        self.assertEqual([v["timing"] for v in filtered], ["original", "later_collision", "fast_collision"])

    def test_first_person_spectator_pose_and_activation(self):
        """First-person spectator activates for --dashcam and --camera dashcam/dashcam_all, and computes world mount pose."""
        self.assertTrue(motion_runner.should_use_first_person_spectator("dashcam", dashcam=False))
        self.assertTrue(motion_runner.should_use_first_person_spectator("dashcam_all", dashcam=False))
        self.assertTrue(motion_runner.should_use_first_person_spectator("chase", dashcam=True))
        self.assertFalse(motion_runner.should_use_first_person_spectator("chase", dashcam=False))
        self.assertFalse(motion_runner.should_use_first_person_spectator("both", dashcam=True))

        # Heading east (yaw=0): local (x=1.2, y=0, z=1.35) adds +1.2 to X, +1.35 to Z, pitch=-3.0
        wx, wy, wz, wp, wyaw, wroll = motion_runner.world_dashcam_pose((10.0, 5.0, 0.2), (0.0, 0.0, 0.0), (1.2, 0.0, 1.35, -3.0))
        self.assertAlmostEqual(wx, 11.2, places=2)
        self.assertAlmostEqual(wy, 5.0, places=2)
        self.assertAlmostEqual(wz, 1.55, places=2)
        self.assertAlmostEqual(wp, -3.0, places=2)
        self.assertAlmostEqual(wyaw, 0.0, places=2)
        # Heading south (CARLA yaw=90): forward offset +1.75 moves along +Y
        wx2, wy2, wz2, _, wyaw2, _ = motion_runner.world_dashcam_pose((10.0, 5.0, 0.2), (0.0, 90.0, 0.0), (1.75, 0.0, 2.0, -4.0))
        self.assertAlmostEqual(wx2, 10.0, places=2)
        self.assertAlmostEqual(wy2, 6.75, places=2)
        self.assertAlmostEqual(wz2, 2.2, places=2)
        self.assertAlmostEqual(wyaw2, 90.0, places=2)

    def test_predict_dashcam_pose_eliminates_sync_tick_lag_and_handles_braking_and_turns(self):
        """Regression test: first-person spectator prediction cancels 1-tick v*DT lag so camera never starts behind
        the rearview mirror at speed nor slides forward out of the car while decelerating/braking."""
        mount = motion_runner.dashcam_mount_transform("car")  # (1.20, 0.0, 1.35, -3.0)
        dt = motion_runner.DT

        # High-speed cruise (20 m/s east): without prediction camera sits 1.0 m behind mount (inside cabin behind mirror);
        # with lead_s=DT it lands on the post-tick vehicle position + mount (0.0 m error).
        x0, v0 = 15.0, 20.0
        wx, wy, wz, wp, wyaw, wroll = motion_runner.predict_dashcam_pose(
            (x0, 0.0, 0.0), (0.0, 0.0, 0.0), mount, vel_xyz=(v0, 0.0, 0.0), lead_s=dt
        )
        x_after_tick = x0 + v0 * dt
        self.assertAlmostEqual(wx - x_after_tick, mount[0], delta=0.005)
        self.assertAlmostEqual(wy, 0.0, delta=0.005)
        self.assertAlmostEqual(wz, mount[2], delta=0.005)

        # Hard emergency braking (-6.0 m/s^2 from 10 m/s): includes 0.5*a*dt^2 to stay locked within 1 mm
        v_brk, a_brk = 10.0, -6.0
        wx_b, _, _, _, _, _ = motion_runner.predict_dashcam_pose(
            (50.0, 0.0, 0.0), (0.0, 0.0, 0.0), mount, vel_xyz=(v_brk, 0.0, 0.0), acc_xyz=(a_brk, 0.0, 0.0), lead_s=dt
        )
        x_brk_after = 50.0 + v_brk * dt + 0.5 * a_brk * dt * dt
        self.assertAlmostEqual(wx_b - x_brk_after, mount[0], delta=0.002)

        # Braking near full stop (v=0.1 m/s, a=-6 m/s^2) must stop cleanly, never predict reversing behind x0
        wx_stop, _, _, _, _, _ = motion_runner.predict_dashcam_pose(
            (60.0, 0.0, 0.0), (0.0, 0.0, 0.0), mount, vel_xyz=(0.1, 0.0, 0.0), acc_xyz=(-6.0, 0.0, 0.0), lead_s=dt
        )
        self.assertGreaterEqual(wx_stop, 60.0 + mount[0])
        self.assertLessEqual(wx_stop, 60.0 + mount[0] + 0.005)

        # Crash impulse acceleration spike (+200 m/s^2) is clamped so spectator never shoots out of the car
        wx_imp, _, _, _, _, _ = motion_runner.predict_dashcam_pose(
            (60.0, 0.0, 0.0), (0.0, 0.0, 0.0), mount, vel_xyz=(5.0, 0.0, 0.0), acc_xyz=(200.0, 0.0, 0.0), lead_s=dt
        )
        self.assertAlmostEqual(wx_imp - (60.0 + 5.0 * dt), mount[0], delta=0.015)

        # Yaw wrap across +/-180 deg computes clean shortest-arc angular rate without 360-deg spin
        rate = motion_runner.angular_rate_pyr((0.0, 179.0, 0.0), (0.0, -179.0, 0.0), dt=dt)
        self.assertAlmostEqual(rate[1], 40.0, places=2)
        _, _, _, _, pyaw, _ = motion_runner.predict_dashcam_pose(
            (0.0, 0.0, 0.0), (0.0, -179.0, 0.0), mount, rot_rate_pyr=rate, lead_s=dt
        )
        self.assertAlmostEqual(pyaw, -177.0, places=2)

        # Zero lead returns exact resting mount pose
        self.assertEqual(
            motion_runner.predict_dashcam_pose((10.0, 5.0, 0.2), (0.0, 0.0, 0.0), mount, vel_xyz=(20.0, 0.0, 0.0), lead_s=0.0),
            motion_runner.world_dashcam_pose((10.0, 5.0, 0.2), (0.0, 0.0, 0.0), mount),
        )

    def test_freeze_actors_for_exit_halts_vehicles_and_walkers_before_async(self):
        """Regression test: leaving synchronous mode must halt every live vehicle (handbrake + zero velocity)
        and walker so no car coasts away from the spectator camera at scenario end."""
        import types

        class _Vec3:
            def __init__(self, x=0.0, y=0.0, z=0.0):
                self.x, self.y, self.z = x, y, z

        class _VCtl:
            def __init__(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False):
                self.throttle, self.steer, self.brake, self.hand_brake = throttle, steer, brake, hand_brake

        class _WCtl:
            def __init__(self, speed=0.0):
                self.speed = speed

        class _Veh:
            def __init__(self, alive=True):
                self.type_id = "vehicle.dodge.charger"
                self.is_alive = alive
                self.autopilot = True
                self.ctrl = None
                self.vel = _Vec3(18.0, 0.0, 0.0)
                self.ang = _Vec3(0.0, 0.0, 10.0)
            def set_autopilot(self, v): self.autopilot = v
            def apply_control(self, c): self.ctrl = c
            def set_target_velocity(self, v): self.vel = v
            def set_target_angular_velocity(self, v): self.ang = v

        class _Walker:
            def __init__(self):
                self.type_id = "walker.pedestrian.0001"
                self.is_alive = True
                self.ctrl = None
            def apply_control(self, c): self.ctrl = c

        v_live, v_dead, w_live = _Veh(True), _Veh(False), _Walker()
        carla_stub = types.SimpleNamespace(VehicleControl=_VCtl, WalkerControl=_WCtl, Vector3D=_Vec3)
        motion_runner.freeze_actors_for_exit({"v1": v_live, "v2": v_dead, "w1": w_live}, carla_stub)

        self.assertFalse(v_live.autopilot)
        self.assertEqual(v_live.ctrl.throttle, 0.0)
        self.assertEqual(v_live.ctrl.brake, 1.0)
        self.assertTrue(v_live.ctrl.hand_brake)
        self.assertEqual((v_live.vel.x, v_live.vel.y, v_live.vel.z), (0.0, 0.0, 0.0))
        self.assertEqual((v_live.ang.x, v_live.ang.y, v_live.ang.z), (0.0, 0.0, 0.0))
        self.assertIsNone(v_dead.ctrl)
        self.assertEqual(w_live.ctrl.speed, 0.0)

    def test_run_motion_first_person_spectator_zero_lag_and_frozen_at_end(self):
        """End-to-end mocked CARLA test reproducing Unreal's synchronous tick order (OnPreTick -> physics -> render):
        verifies the first-person dashcam spectator stays locked to the windshield mount on tick 0, during cruise,
        during hold, and remains stationary with the halted vehicle when asynchronous mode is restored."""
        import types

        class _Vec3:
            def __init__(self, x=0.0, y=0.0, z=0.0):
                self.x, self.y, self.z = float(x), float(y), float(z)
            def length(self):
                return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

        class _Loc(_Vec3):
            def distance(self, other):
                return math.sqrt((self.x - other.x) ** 2 + (self.y - other.y) ** 2 + (self.z - other.z) ** 2)

        class _Rot:
            def __init__(self, pitch=0.0, yaw=0.0, roll=0.0):
                self.pitch, self.yaw, self.roll = float(pitch), float(yaw), float(roll)

        class _Tf:
            def __init__(self, loc=None, rot=None):
                self.location = loc or _Loc()
                self.rotation = rot or _Rot()
            def get_forward_vector(self):
                r = math.radians(self.rotation.yaw)
                return _Vec3(math.cos(r), math.sin(r), 0.0)
            def get_right_vector(self):
                r = math.radians(self.rotation.yaw)
                return _Vec3(-math.sin(r), math.cos(r), 0.0)

        class _VCtl:
            def __init__(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False):
                self.throttle, self.steer, self.brake, self.hand_brake = throttle, steer, brake, hand_brake

        class _WCtl:
            def __init__(self, direction=None, speed=0.0):
                self.direction, self.speed = direction, speed

        class _Settings:
            def __init__(self, synchronous_mode=False, fixed_delta_seconds=0.0, **kw):
                self.synchronous_mode = synchronous_mode
                self.fixed_delta_seconds = fixed_delta_seconds

        class _Actor:
            def __init__(self, aid, tid, tf, parent=None):
                self.id, self.type_id, self._tf, self.parent = aid, tid, tf, parent
                self.is_alive = True
                self._vel = _Vec3()
                self._pending_vel = None
                self._acc = _Vec3()
                self._ctrl = _VCtl()
                self._cb = None
                self.bounding_box = NS(extent=_Vec3(2.4, 0.95, 0.72))
            def get_transform(self): return _Tf(_Loc(self._tf.location.x, self._tf.location.y, self._tf.location.z),
                                                _Rot(self._tf.rotation.pitch, self._tf.rotation.yaw, self._tf.rotation.roll))
            def get_location(self): return _Loc(self._tf.location.x, self._tf.location.y, self._tf.location.z)
            def get_velocity(self): return _Vec3(self._vel.x, self._vel.y, self._vel.z)
            def get_acceleration(self): return _Vec3(self._acc.x, self._acc.y, self._acc.z)
            def set_transform(self, tf): self._tf = tf
            def apply_control(self, c): self._ctrl = c
            def set_autopilot(self, v): pass
            def set_target_velocity(self, v): self._pending_vel = _Vec3(v.x, v.y, v.z)
            def set_target_angular_velocity(self, v): pass
            def get_physics_control(self): return NS(wheels=[NS(max_steer_angle=70.0)])
            def listen(self, cb): self._cb = cb
            def stop(self): self._cb = None

        class _Wp:
            def __init__(self, x, y):
                self.transform = _Tf(_Loc(x, y, 0.0), _Rot(0.0, 0.0, 0.0))
                self.is_junction = False
            def next(self, step):
                return [_Wp(self.transform.location.x + step, self.transform.location.y)]

        class _Map:
            def get_waypoint(self, loc, project_to_road=True, lane_type=None):
                return _Wp(loc.x, loc.y)

        class _Bp:
            def __init__(self, tid): self.id = tid
            def has_attribute(self, k): return False
            def set_attribute(self, k, v): pass

        class _BpLib:
            def find(self, tid): return _Bp(tid)
            def filter(self, pat): return [_Bp(pat)]

        spectator = _Actor(999, "spectator", _Tf())
        rendered_offsets = []      # (sim_t, spec_x - veh_x, veh_vx) observed on each rendered frame while moving
        state_at_async = {}
        spawned_att = []

        class _World:
            def __init__(self):
                self._t = 0.0
                self._next_id = 1
                self._actors = []
                self._sync = False
                self._dt = motion_runner.DT
            def get_map(self): return _Map()
            def get_blueprint_library(self): return _BpLib()
            def get_spectator(self): return spectator
            def get_actors(self): return list(self._actors)
            def get_snapshot(self): return NS(timestamp=NS(elapsed_seconds=self._t))
            def apply_settings(self, s):
                if self._sync and not s.synchronous_mode:
                    veh = next((a for a in self._actors if a.type_id.startswith("vehicle.")), None)
                    if veh is not None:
                        state_at_async["veh_vx"] = veh._vel.x
                        state_at_async["hand_brake"] = veh._ctrl.hand_brake
                        state_at_async["brake"] = veh._ctrl.brake
                        state_at_async["offset_x"] = spectator._tf.location.x - veh._tf.location.x
                self._sync = bool(s.synchronous_mode)
                self._dt = float(s.fixed_delta_seconds or motion_runner.DT)
            def try_spawn_actor(self, bp, tf, attach_to=None, attachment_type=None):
                a = _Actor(self._next_id, bp.id, tf, parent=attach_to)
                self._next_id += 1
                self._actors.append(a)
                if bp.id == "sensor.camera.rgb":
                    spawned_att.append(attachment_type)
                return a
            def spawn_actor(self, bp, tf, attach_to=None, attachment_type=None):
                return self.try_spawn_actor(bp, tf, attach_to=attach_to, attachment_type=attachment_type)
            def tick(self):
                # Unreal sync order: OnPreTick already applied spectator.set_transform & vehicle controls;
                # now TG_DuringPhysics steps vehicles by dt, then RedrawViewports renders vehicle vs spectator!
                self._t += self._dt
                for a in self._actors:
                    if a.type_id.startswith("vehicle.") and a.is_alive:
                        if a._pending_vel is not None:
                            a._vel = a._pending_vel
                            a._pending_vel = None
                        if a._ctrl.hand_brake:
                            a._acc = _Vec3(0.0, 0.0, 0.0)
                        elif a._ctrl.brake > 0.0:
                            ax = -6.0 * a._ctrl.brake if a._vel.x > 0.0 else 0.0
                            vx_new = max(0.0, a._vel.x + ax * self._dt)
                            a._tf.location.x += 0.5 * (a._vel.x + vx_new) * self._dt
                            a._acc = _Vec3((vx_new - a._vel.x) / self._dt, 0.0, 0.0)
                            a._vel.x = vx_new
                        else:
                            a._tf.location.x += a._vel.x * self._dt
                            a._acc = _Vec3(0.0, 0.0, 0.0)
                veh = next((a for a in self._actors if a.type_id.startswith("vehicle.")), None)
                if veh is not None and veh._vel.x > 1.0:
                    rendered_offsets.append((round(self._t, 2), round(spectator._tf.location.x - veh._tf.location.x, 3), round(veh._vel.x, 2)))

        fake_world = _World()
        fake_tm = NS(set_synchronous_mode=lambda v: None, set_random_device_seed=lambda s: None)
        fake_client = NS(set_timeout=lambda t: None, get_world=lambda: fake_world, get_trafficmanager=lambda: fake_tm, apply_batch=lambda cmds: None)
        fake_carla = types.SimpleNamespace(
            Client=lambda h, p: fake_client,
            WorldSettings=_Settings,
            Location=_Loc,
            Rotation=_Rot,
            Transform=_Tf,
            Vector3D=_Vec3,
            VehicleControl=_VCtl,
            WalkerControl=_WCtl,
            LaneType=NS(Driving=1),
            AttachmentType=NS(Rigid="RIGID", SpringArm="SPRING"),
            command=NS(DestroyActor=lambda i: i),
        )

        saved_carla = sys.modules.get("carla")
        sys.modules["carla"] = fake_carla
        try:
            with tempfile.TemporaryDirectory() as tmp:
                # Build a minimal moving-vehicle scene with dashcam first-person spectator and non-zero hold_sec
                with open(os.path.join(tmp, "sc_scene_obj.txt"), "w", encoding="utf-8") as f:
                    f.write("{'v1': {'type': 'car', 'location': [10.0, 0.0, 0.0], 'rotation': [0.0, 0.0, 0.0]}}\n{}\n")
                plan = {
                    "ego": "v1",
                    "duration_s": 0.5,
                    "expect": {"type": "none", "actors": ["v1"]},
                    "dashcam_actors": ["v1"],
                    "agents": {
                        "v1": {"mode": "drive", "speed_kmh": 72.0, "already_moving": True, "path": [[200.0, 0.0]]}
                    },
                }
                with open(os.path.join(tmp, "sc_motion.json"), "w", encoding="utf-8") as f:
                    json.dump(plan, f)
                motion_runner.run_motion(
                    tmp, "sc", load_map=False, hold_sec=0.5, real_time=False,
                    record_video=False, camera="dashcam", dashcam=False,
                )
        finally:
            if saved_carla is None:
                sys.modules.pop("carla", None)
            else:
                sys.modules["carla"] = saved_carla

        # Every rendered tick while moving at 20 m/s MUST stay locked at mount_x=1.20 (NOT lagged to 0.20 behind mirror!)
        self.assertGreater(len(rendered_offsets), 20)
        mount_x = motion_runner.dashcam_mount_transform("car")[0]
        for sim_t, dx, vx in rendered_offsets:
            self.assertAlmostEqual(dx, mount_x, delta=0.05, msg=f"lag/drift at t={sim_t}, vx={vx}: dx={dx}")

        # Before returning to async mode, vehicle MUST be completely stopped with handbrake and spectator aligned
        self.assertEqual(state_at_async.get("veh_vx"), 0.0)
        self.assertTrue(state_at_async.get("hand_brake"))
        self.assertEqual(state_at_async.get("brake"), 1.0)
        self.assertAlmostEqual(state_at_async.get("offset_x"), mount_x, delta=0.01)

    def test_vary_weather_per_road_and_explicit_weather_variants_and_headlights(self):
        """--vary-weather assigns distinct weather/time per changed road without multiplying count, and headlights turn on."""
        base = load_fixture("v1")
        base["environment"] = {"time_of_day": "day", "weather": "clear"}
        with tempfile.TemporaryDirectory() as tmp:
            man = scenario_variants.build_variants(
                base, tmp,
                road_presets="3", timing_presets="3",
                crashing_only=True, vary_weather=True, log_fn=None,
            )
            # 3 roads x 3 crashing timings = 9 (NOT 27!), with 3 distinct weather presets (1 per road)
            self.assertEqual(len(man["variants"]), 9)
            self.assertTrue(all(e["expected_outcome"] == "collision" for e in man["variants"]))
            by_road = {}
            for e in man["variants"]:
                by_road.setdefault(e["road"], set()).add(e["weather"])
            self.assertEqual(by_road["original"], {"original"})
            self.assertEqual(len({next(iter(s)) for s in by_road.values()}), 3)

        # Explicit --weather-variants supports <time>:<weather>, product mode reuses same road map signature!
        with tempfile.TemporaryDirectory() as tmp:
            man_w = scenario_variants.build_variants(
                load_fixture("v3"), tmp,
                road_presets="original", timing_presets="original",
                weather_presets="original,night:rain,dawn:fog,day:storm", log_fn=None,
            )
            self.assertEqual(len(man_w["variants"]), 4)
            sigs = [local_pipeline._road_map_signature(e["dir"]) for e in man_w["variants"]]
            self.assertEqual(len(set(sigs)), 1)        # identical road -> zero extra CARLA map reloads!
            envs = [(e["environment"]["time_of_day"], e["environment"]["weather"]) for e in man_w["variants"]]
            self.assertEqual(envs[1:], [("night", "rain"), ("dawn", "fog"), ("day", "storm")])

        # Weather parameters and automatic headlights
        self.assertEqual(motion_runner.vehicle_lights_for({"time_of_day": "day", "weather": "clear"}), ())
        self.assertIn("low_beam", motion_runner.vehicle_lights_for({"time_of_day": "night", "weather": "clear"}))
        self.assertIn("fog", motion_runner.vehicle_lights_for({"time_of_day": "dawn", "weather": "fog"}))
        storm_w = motion_runner.weather_for({"time_of_day": "day", "weather": "storm"})
        self.assertGreater(storm_w["precipitation"], 80.0)
        self.assertGreater(storm_w["wetness"], 85.0)

        # XSD validation of new dawn/storm environment in OpenSCENARIO export
        xsd = FIX / "OpenSCENARIO.xsd"
        if xsd.exists():
            try:
                import xmlschema  # noqa: F401
                has_xsd = True
            except ImportError:
                has_xsd = False
            if has_xsd:
                def samples(x0, y, v):
                    return [[round(i * 0.05, 3), x0 + v * i * 0.05, y, 0.2, 0.0, v] for i in range(40)]
                rec = {"frame": "carla", "dt": 0.05, "actors": {
                    "v1": {"type": "car", "blueprint": "vehicle.dodge.charger", "samples": samples(0.0, 5.25, 14.0)},
                }}
                for env in ({"time_of_day": "dawn", "weather": "fog"}, {"time_of_day": "night", "weather": "storm"}):
                    with tempfile.TemporaryDirectory() as tmp:
                        Path(tmp, "w_trajectories.json").write_text(json.dumps(rec))
                        Path(tmp, "w_motion.json").write_text(json.dumps({"ego": "v1", "environment": env}))
                        Path(tmp, "w.xodr").write_text("<OpenDRIVE/>")
                        xosc_export.export_scene(tmp, "w", xsd_path=xsd)

    def test_real_result_specs_build_clean_road_and_timing_matrices(self):
        """End-to-end check on real pipeline specs (results/1 highway fork cut-in, results/2 third-party, results/3 crossroad)."""
        for idx in ("1", "2", "3"):
            sp_fn = RESULTS / idx / "video_interpreter_0000_split_spec.json"
            if not sp_fn.exists():
                continue
            with open(sp_fn, encoding="utf-8") as f:
                base = json.load(f)
            with tempfile.TemporaryDirectory() as tmp:
                man = scenario_variants.build_variants(
                    base, tmp, road_presets="3", timing_presets="4", combine="product", log_fn=None
                )
                self.assertEqual(len(man["variants"]), 12, f"results/{idx}")
                # Must include collision, non-collision miss, and aggressor-absent variants
                outcomes = {e["expected_outcome"] for e in man["variants"]}
                self.assertIn("collision", outcomes, f"results/{idx}")
                self.assertIn("miss", outcomes, f"results/{idx}")
                self.assertIn("none", outcomes, f"results/{idx}")
                # Check results/2 (third-party) NEVER puts dashcam on bystander v1 in ANY of the 12 variants
                if idx == "2":
                    for e in man["variants"]:
                        self.assertNotIn("v1", e["dashcam_actors"], e["id"])


if __name__ == "__main__":
    unittest.main()
