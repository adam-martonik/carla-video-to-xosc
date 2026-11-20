import json
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import make_golden


class TestBuilderGolden(unittest.TestCase):
    def _check(self, name):
        with open(make_golden.GOLDEN / (name + ".json"), encoding="utf-8") as f:
            want = json.load(f)
        got = json.loads(json.dumps(make_golden.build_fixture(name), sort_keys=True))
        self.assertEqual(got["scene_obj"], want["scene_obj"])
        self.assertEqual(got["motion"], want["motion"])

    def test_straight_cut_in_unchanged(self):
        self._check("v1")

    def test_straight_crossing_unchanged(self):
        self._check("crash1")

    def test_crossroad_unchanged(self):
        self._check("v3")


if __name__ == "__main__":
    unittest.main()
