import json
import math
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import xosc_export

XSD = Path(__file__).resolve().parent / "fixtures" / "OpenSCENARIO.xsd"


class FakeActor:
    def __init__(self, x, y, yaw, speed, alive=True):
        self.x, self.y, self.yaw, self.speed, self.is_alive = x, y, yaw, speed, alive

    def get_transform(self):
        return NS(location=NS(x=self.x, y=self.y, z=0.2), rotation=NS(yaw=self.yaw))

    def get_velocity(self):
        return NS(x=self.speed, y=0.0, z=0.0)


def recording(n=100, dt=0.05):
    """v1 (ego, 14 m/s) and v2 (10 m/s) drive east in CARLA coordinates, so y is the negated map y."""
    def samples(x0, y, v):
        return [[round(i * dt, 3), x0 + v * i * dt, y, 0.2, 0.0, v] for i in range(n)]
    return {"frame": "carla", "dt": dt, "actors": {
        "v1": {"type": "car", "blueprint": "vehicle.dodge.charger", "color": "25,25,25", "samples": samples(0.0, 5.25, 14.0)},
        "v2": {"type": "suv", "blueprint": "vehicle.nissan.patrol", "color": None, "samples": samples(20.0, 1.75, 10.0)}}}


class TestFrame(unittest.TestCase):
    def test_carla_to_world(self):
        x, y, h = xosc_export.carla_to_world(10.0, 5.25, 0.0)
        self.assertEqual((x, y, h), (10.0, -5.25, 0.0))
        self.assertAlmostEqual(xosc_export.carla_to_world(0, 0, 90.0)[2], -math.pi / 2)      # CARLA +90 deg (south) -> SUMO -90 deg


class TestRecorder(unittest.TestCase):
    def test_records_alive_actors_only(self):
        rec = xosc_export.TrajectoryRecorder()
        rec.record(0.05, {"a": FakeActor(1, 2, 30, 3.0), "b": FakeActor(0, 0, 0, 0, alive=False)})
        self.assertEqual(list(rec.samples), ["a"])
        self.assertEqual(rec.samples["a"][0], [0.05, 1.0, 2.0, 0.2, 30.0, 3.0])

    def test_save_roundtrip(self):
        rec = xosc_export.TrajectoryRecorder()
        rec.record(0.05, {"a": FakeActor(1, 2, 0, 3.0)})
        with tempfile.TemporaryDirectory() as tmp:
            p = rec.save(Path(tmp) / "t.json", {"a": {"type": "car"}}, 0.05)
            with open(p, encoding="utf-8") as f:
                data = json.load(f)
        self.assertEqual(data["actors"]["a"]["type"], "car")
        self.assertEqual(len(data["actors"]["a"]["samples"]), 1)


class TestCropTime(unittest.TestCase):
    def test_none_without_a_collision(self):
        self.assertIsNone(xosc_export.crop_time({"events": []}, {}))
        self.assertIsNone(xosc_export.crop_time({"events": [{"other": "static.prop.barrier", "t_sim": 2.0}]}, {}))

    def test_first_vehicle_contact_plus_hold(self):
        res = {"events": [{"actor": "v1", "other": "vehicle.x", "other_actor": "v2", "t_sim": 4.0},
                          {"actor": "v1", "other": "vehicle.x", "other_actor": "v2", "t_sim": 4.5}]}
        self.assertAlmostEqual(xosc_export.crop_time(res, {}, hold_s=0.3), 4.3)

    def test_expected_pair_only(self):
        res = {"events": [{"actor": "v1", "other": "vehicle.x", "other_actor": "v3", "t_sim": 2.0},
                          {"actor": "v1", "other": "vehicle.x", "other_actor": "v2", "t_sim": 5.0}]}
        plan = {"expect": {"actors": ["v1", "v2"]}}
        self.assertAlmostEqual(xosc_export.crop_time(res, plan, hold_s=0.0), 5.0)


class TestBuildScenario(unittest.TestCase):
    def test_world_frame_and_ego(self):
        sc, rep = xosc_export.build_scenario(recording(), {"ego": "v1"}, "s.xodr")
        ents = {e["name"]: e for e in sc["entities"]}
        self.assertTrue(ents["v1"]["ego"] and not ents["v2"]["ego"])
        self.assertAlmostEqual(ents["v1"]["y"][0], -5.25, places=2)                    # y negated
        self.assertAlmostEqual(ents["v1"]["h"][5], 0.0, places=2)
        self.assertAlmostEqual(ents["v1"]["v"][10], 14.0, delta=0.5)
        self.assertLess(rep["v1"]["max_deviation_m"], 0.5)
        self.assertEqual(sc["xodr"], "s.xodr")

    def test_recording_is_cropped_at_the_impact(self):
        res = {"events": [{"actor": "v1", "other": "vehicle.x", "other_actor": "v2", "t_sim": 2.0}]}
        sc, _ = xosc_export.build_scenario(recording(), {"ego": "v1"}, None, res, hold_s=0.0)
        self.assertLessEqual(sc["duration"], 2.0 + 1e-6)
        self.assertGreater(sc["duration"], 1.8)

    def test_ego_falls_back_to_first_actor(self):
        sc, _ = xosc_export.build_scenario(recording(), {}, None)
        self.assertEqual(sum(e["ego"] for e in sc["entities"]), 1)

    def test_actor_with_one_sample_is_skipped(self):
        rec = recording()
        rec["actors"]["v2"]["samples"] = rec["actors"]["v2"]["samples"][:1]
        sc, report = xosc_export.build_scenario(rec, {"ego": "v1"}, None)
        self.assertEqual([e["name"] for e in sc["entities"]], ["v1"])
        self.assertIn("skipped", report["v2"])

    def test_empty_recording_is_an_error(self):
        with self.assertRaises(ValueError):
            xosc_export.build_scenario({"actors": {}}, {}, None)


class TestExportScene(unittest.TestCase):
    def test_end_to_end_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "s_trajectories.json").write_text(json.dumps(recording()))
            (tmp / "s_motion.json").write_text(json.dumps({"ego": "v1"}))
            (tmp / "s.xodr").write_text("<OpenDRIVE/>")
            out, report = xosc_export.export_scene(str(tmp), "s", xsd_path=XSD if XSD.exists() else None)
            root = ET.parse(out).getroot()
            self.assertEqual(root.find("RoadNetwork/LogicFile").get("filepath"), "s.xodr")
            self.assertEqual([o.get("name") for o in root.findall("Entities/ScenarioObject")], ["hero", "v2"])
            self.assertEqual(set(report), {"v1", "v2"})

    def test_missing_recording_has_a_helpful_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(FileNotFoundError, "motion_runner"):
                xosc_export.export_scene(tmp, "nope")


if __name__ == "__main__":
    unittest.main()
