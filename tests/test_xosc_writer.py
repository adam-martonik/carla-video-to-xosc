import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents import xosc_writer

XSD = Path(__file__).resolve().parent / "fixtures" / "OpenSCENARIO.xsd"


def entity(name, ego=False, kind="vehicle", vtype="car", x0=0.0, speed=10.0, n=11, color="25,25,25"):
    t = [0.1 * i for i in range(n)]
    return {"name": name, "ego": ego, "kind": kind, "vtype": vtype, "blueprint": "vehicle.dodge.charger", "color": color,
            "times": t, "x": [x0 + speed * ti for ti in t], "y": [-1.75] * n, "h": [0.0] * n, "v": [speed] * n}


def scenario(**kw):
    base = {"scene_id": "s", "duration": 1.0, "xodr": "s.xodr",
            "entities": [entity("v1", ego=True), entity("v2", x0=20.0, vtype="suv"),
                         entity("p1", kind="pedestrian", vtype="pedestrian", speed=1.4, color=None)]}
    base.update(kw)
    return base


class TestBuild(unittest.TestCase):
    def test_structure(self):
        root = xosc_writer.build_xosc(scenario())
        self.assertEqual(root.tag, "OpenSCENARIO")
        self.assertEqual(root.find("RoadNetwork/LogicFile").get("filepath"), "s.xodr")
        names = [o.get("name") for o in root.findall("Entities/ScenarioObject")]
        self.assertEqual(names, ["hero", "v2", "p1"])                        # the ego is called hero
        self.assertEqual(len(root.findall(".//FollowTrajectoryAction")), 3)
        self.assertEqual(len(root.findall("Storyboard/Init/Actions/Private")), 3)

    def test_vehicle_children_follow_schema_order(self):
        veh = xosc_writer.build_xosc(scenario()).find("Entities/ScenarioObject/Vehicle")
        self.assertEqual([c.tag for c in veh], ["ParameterDeclarations", "BoundingBox", "Performance", "Axles", "Properties"])

    def test_vertex_times_start_at_zero_even_for_a_late_recording(self):
        e = entity("v1", ego=True)
        e["times"] = [5.0 + 0.1 * i for i in range(11)]
        poly = xosc_writer.build_xosc(scenario(entities=[e])).find(".//Polyline")
        times = [float(v.get("time")) for v in poly.findall("Vertex")]
        self.assertAlmostEqual(times[0], 0.0)
        self.assertAlmostEqual(times[-1], 1.0)

    def test_stop_trigger_uses_the_duration(self):
        root = xosc_writer.build_xosc(scenario(duration=7.5))
        cond = root.find("Storyboard/StopTrigger//SimulationTimeCondition")
        self.assertEqual(cond.get("value"), "7.500")

    def test_no_xodr_means_no_logic_file(self):
        self.assertIsNone(xosc_writer.build_xosc(scenario(xodr=None)).find("RoadNetwork/LogicFile"))

    def test_bad_entities_are_rejected(self):
        with self.assertRaises(ValueError):
            xosc_writer.build_xosc(scenario(entities=[]))
        bad = entity("v1", ego=True)
        bad["y"] = bad["y"][:-1]
        with self.assertRaises(ValueError):
            xosc_writer.build_xosc(scenario(entities=[bad]))


@unittest.skipUnless(XSD.exists(), "OpenSCENARIO.xsd fixture missing")
class TestSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import xmlschema   # noqa: F401
        except ImportError:
            raise unittest.SkipTest("xmlschema is not installed")

    def test_file_validates_against_openscenario_1_0(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = xosc_writer.write_openscenario_xosc(scenario(), Path(tmp) / "a.xosc", xsd_path=XSD)
            self.assertEqual(ET.parse(out).getroot().tag, "OpenSCENARIO")

    def test_night_scene_without_colors_validates(self):
        sc = scenario(is_night=True, friction=0.6)
        for e in sc["entities"]:
            e["color"] = None
        with tempfile.TemporaryDirectory() as tmp:
            xosc_writer.write_openscenario_xosc(sc, Path(tmp) / "n.xosc", xsd_path=XSD)

    def test_invalid_file_is_detected(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "bad.xosc"
            p.write_text("<OpenSCENARIO><Oops/></OpenSCENARIO>", encoding="utf-8")
            import xmlschema
            with self.assertRaises(xmlschema.XMLSchemaValidationError):
                xosc_writer.validate_xosc(p, XSD)


if __name__ == "__main__":
    unittest.main()
