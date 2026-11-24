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
