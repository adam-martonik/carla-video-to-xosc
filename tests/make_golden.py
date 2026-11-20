"""Regenerate tests/golden/*.json from tests/fixtures/*.json with the CURRENT scene_builder.
Run it before a pure refactor of the builder; after a deliberate behaviour change, regenerate only the golden you changed on purpose:
  python tests/make_golden.py [name ...]
"""
import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agents import scene_builder

FIXTURES = ROOT / "tests" / "fixtures"
GOLDEN = ROOT / "tests" / "golden"


def build_fixture(name):
    with open(FIXTURES / (name + ".json"), encoding="utf-8") as f:
        spec = json.load(f)
    with tempfile.TemporaryDirectory() as tmp:
        scene_builder.build(spec, tmp, "s")
        with open(os.path.join(tmp, "s_scene_obj.txt"), encoding="utf-8") as f:
            obj = f.read()
        with open(os.path.join(tmp, "s_motion.json"), encoding="utf-8") as f:
            motion = json.load(f)
    return {"scene_obj": obj, "motion": motion}


if __name__ == "__main__":
    names = sys.argv[1:] or [p.stem for p in sorted(FIXTURES.glob("*.json"))]
    GOLDEN.mkdir(exist_ok=True)
    for n in names:
        with open(GOLDEN / (n + ".json"), "w", encoding="utf-8") as f:
            json.dump(build_fixture(n), f, indent=1, sort_keys=True)
        print("golden written:", n)
