"""
ASAM OpenSCENARIO 1.0 (.xosc) writer for recorded/smoothed actor trajectories (no CARLA needed).

Ported from carla-video-to-xosc (xosc_writer.py) and reworked to be schema-valid:
child order of <Vehicle> is BoundingBox, Performance, Axles, Properties; <FollowTrajectoryAction> carries an
absolute <Timing> reference so the vertex times are honoured; the storyboard stops on simulation time.

Input (plain dict, world frame = the OpenDRIVE/SUMO frame of the map, metres and radians):
{
  "scene_id": "s", "duration": 12.0, "xodr": "s.xodr" | None, "description": "...",
  "is_night": False, "friction": 1.0,
  "entities": [ {"name": "v1", "ego": True, "kind": "vehicle" | "pedestrian", "vtype": "car",
                 "blueprint": "vehicle.dodge.charger", "color": "25,25,25" | None,
                 "times": [...], "x": [...], "y": [...], "h": [...], "v": [...]} ]
}
The ego car is written under the name "hero", which CARLA ScenarioRunner treats as the ego vehicle.
"""
import datetime
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
from pathlib import Path

EGO_NAME = "hero"

# length, width, height, wheelbase, track width, wheel diameter, mass, max steering (rad), OpenSCENARIO category
SPECS = {
    "car":        {"length": 4.7, "width": 1.9, "height": 1.5, "wheelbase": 2.8, "track": 1.6, "wheel": 0.66, "mass": 1500.0, "steer": 0.55, "category": "car"},
    "suv":        {"length": 4.9, "width": 2.0, "height": 1.8, "wheelbase": 2.9, "track": 1.7, "wheel": 0.76, "mass": 2100.0, "steer": 0.55, "category": "car"},
    "van":        {"length": 5.4, "width": 2.0, "height": 2.2, "wheelbase": 3.4, "track": 1.7, "wheel": 0.70, "mass": 2400.0, "steer": 0.50, "category": "van"},
    "truck":      {"length": 7.5, "width": 2.5, "height": 3.2, "wheelbase": 4.5, "track": 1.9, "wheel": 1.00, "mass": 8000.0, "steer": 0.45, "category": "truck"},
    "motorcycle": {"length": 2.1, "width": 0.8, "height": 1.4, "wheelbase": 1.4, "track": 0.2, "wheel": 0.62, "mass": 250.0, "steer": 0.60, "category": "motorbike"},
    "bike":       {"length": 1.8, "width": 0.6, "height": 1.3, "wheelbase": 1.0, "track": 0.2, "wheel": 0.68, "mass": 15.0, "steer": 0.60, "category": "bicycle"},
    "pedestrian": {"length": 0.6, "width": 0.6, "height": 1.75, "wheelbase": 0.0, "track": 0.0, "wheel": 0.0, "mass": 75.0, "steer": 0.0, "category": "pedestrian"},
}


def spec_for(vtype):
    return SPECS.get(vtype, SPECS["car"])


def _sub(parent, tag, **attrs):
    return ET.SubElement(parent, tag, attrib={k: str(v) for k, v in attrs.items()})


def _time_trigger(parent, name, t, edge="rising"):
    trig = _sub(parent, "StartTrigger") if parent.tag in ("Event", "Act") else parent
    cg = _sub(trig, "ConditionGroup")
    cond = _sub(cg, "Condition", name=name, delay="0", conditionEdge=edge)
    by_value = _sub(cond, "ByValueCondition")
    _sub(by_value, "SimulationTimeCondition", value="%.3f" % t, rule="greaterThan")


def _world_position(parent, x, y, h, z=0.0):
    pos = _sub(parent, "Position")
    _sub(pos, "WorldPosition", x="%.3f" % x, y="%.3f" % y, z="%.3f" % z, h="%.5f" % h)


def _add_entity(entities, ent):
    name = EGO_NAME if ent.get("ego") else ent["name"]
    sp = spec_for(ent.get("vtype", "car"))
    obj = _sub(entities, "ScenarioObject", name=name)
    if ent.get("kind") == "pedestrian":
        ped = _sub(obj, "Pedestrian", model=ent.get("blueprint") or "walker.pedestrian.0001", mass="%.1f" % sp["mass"],
                   name=ent.get("blueprint") or "walker.pedestrian.0001", pedestrianCategory="pedestrian")
        _sub(ped, "ParameterDeclarations")
        bb = _sub(ped, "BoundingBox")
        _sub(bb, "Center", x="0.0", y="0.0", z="%.2f" % (sp["height"] / 2))
        _sub(bb, "Dimensions", width="%.2f" % sp["width"], length="%.2f" % sp["length"], height="%.2f" % sp["height"])
        _sub(_sub(ped, "Properties"), "Property", name="type", value="simulation")
        return name
    lim = ent.get("limits") or {}
    veh = _sub(obj, "Vehicle", name=ent.get("blueprint") or "vehicle.tesla.model3", vehicleCategory=sp["category"])
    _sub(veh, "ParameterDeclarations")
    bb = _sub(veh, "BoundingBox")
    _sub(bb, "Center", x="%.2f" % (sp["wheelbase"] / 2), y="0.0", z="%.2f" % (sp["height"] / 2))
    _sub(bb, "Dimensions", width="%.2f" % sp["width"], length="%.2f" % sp["length"], height="%.2f" % sp["height"])
    _sub(veh, "Performance", maxSpeed="%.2f" % lim.get("max_speed", 40.0), maxAcceleration="%.2f" % lim.get("max_accel", 4.5),
         maxDeceleration="%.2f" % lim.get("max_decel", 9.5))
    axles = _sub(veh, "Axles")
    _sub(axles, "FrontAxle", maxSteering="%.2f" % sp["steer"], wheelDiameter="%.2f" % sp["wheel"], trackWidth="%.2f" % sp["track"],
         positionX="%.2f" % sp["wheelbase"], positionZ="%.2f" % (sp["wheel"] / 2))
    _sub(axles, "RearAxle", maxSteering="0.0", wheelDiameter="%.2f" % sp["wheel"], trackWidth="%.2f" % sp["track"],
         positionX="0.0", positionZ="%.2f" % (sp["wheel"] / 2))
    props = _sub(veh, "Properties")
    _sub(props, "Property", name="type", value="ego_vehicle" if ent.get("ego") else "simulation")
    if ent.get("color"):
        _sub(props, "Property", name="color", value=ent["color"])
    return name


def _add_init(init_actions, name, ent):
    priv = _sub(init_actions, "Private", entityRef=name)
    tele = _sub(_sub(priv, "PrivateAction"), "TeleportAction")
    z0 = float(ent["z"][0]) if ent.get("z") else 0.0
    _world_position(tele, ent["x"][0], ent["y"][0], ent["h"][0], z=z0)
    spd = _sub(_sub(_sub(priv, "PrivateAction"), "LongitudinalAction"), "SpeedAction")
    _sub(spd, "SpeedActionDynamics", dynamicsShape="step", value="0.0", dynamicsDimension="time")
    _sub(_sub(spd, "SpeedActionTarget"), "AbsoluteTargetSpeed", value="%.3f" % max(0.0, float(ent["v"][0])))


def _add_trajectory_group(act, name, ent):
    mg = _sub(act, "ManeuverGroup", name="MG_" + name, maximumExecutionCount="1")
    _sub(_sub(mg, "Actors", selectTriggeringEntities="false"), "EntityRef", entityRef=name)
    man = _sub(mg, "Maneuver", name="Maneuver_" + name)
    ev = _sub(man, "Event", name="Event_Trajectory_" + name, priority="overwrite")
    action = _sub(ev, "Action", name="Action_FollowTrajectory_" + name)
    follow = _sub(_sub(_sub(action, "PrivateAction"), "RoutingAction"), "FollowTrajectoryAction")
    traj = _sub(follow, "Trajectory", name="Traj_" + name, closed="false")
    poly = _sub(_sub(traj, "Shape"), "Polyline")
    t0 = float(ent["times"][0])
    zs = ent.get("z") or [0.0] * len(ent["times"])
    for t, x, y, z, h in zip(ent["times"], ent["x"], ent["y"], zs, ent["h"]):
        vtx = _sub(poly, "Vertex", time="%.3f" % (float(t) - t0))
        _world_position(vtx, x, y, h, z=float(z))
    _sub(_sub(follow, "TimeReference"), "Timing", domainAbsoluteRelative="absolute", scale="1.0", offset="0.0")
    _sub(follow, "TrajectoryFollowingMode", followingMode="position")
    _time_trigger(ev, "Start_" + name, 0.0)


def build_xosc(scenario):
    """Return the OpenSCENARIO document as an ElementTree root."""
    ents = scenario["entities"]
    if not ents:
        raise ValueError("a scenario needs at least one entity")
    for e in ents:
        n = len(e["times"])
        if n < 2 or not (len(e["x"]) == len(e["y"]) == len(e["h"]) == len(e["v"]) == n):
            raise ValueError("entity %s: times, x, y, h, v must be the same length (>= 2)" % e.get("name"))
    duration = max(1.0, float(scenario.get("duration") or max(float(e["times"][-1]) - float(e["times"][0]) for e in ents)))

    root = ET.Element("OpenSCENARIO")
    _sub(root, "FileHeader", revMajor="1", revMinor="0", date=datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S"),
         description=scenario.get("description") or "AutoScenario recorded scenario", author="AutoScenario")
    _sub(root, "ParameterDeclarations")
    _sub(root, "CatalogLocations")
    road = _sub(root, "RoadNetwork")
    if scenario.get("xodr"):
        _sub(road, "LogicFile", filepath=scenario["xodr"])
    entities = _sub(root, "Entities")
    names = [_add_entity(entities, e) for e in ents]

    sb = _sub(root, "Storyboard")
    init_actions = _sub(_sub(sb, "Init"), "Actions")
    env_action = _sub(_sub(init_actions, "GlobalAction"), "EnvironmentAction")
    env = _sub(env_action, "Environment", name="RecordedEnv")
    env_cfg = scenario.get("environment") if isinstance(scenario.get("environment"), dict) else {}
    tod = str(env_cfg.get("time_of_day") or ("night" if scenario.get("is_night") else "day")).lower()
    wkind = str(env_cfg.get("weather") or "clear").lower()
    dt_str = {"night": "2026-06-15T23:30:00", "dusk": "2026-06-15T19:45:00", "dawn": "2026-06-15T06:15:00"}.get(tod, "2026-06-15T14:00:00")
    sun_int = {"night": "0.02", "dusk": "0.35", "dawn": "0.30"}.get(tod, "0.90")
    sun_el = {"night": "-0.35", "dusk": "0.15", "dawn": "0.12"}.get(tod, "1.20")
    cloud_state = {"clear": "free", "cloudy": "overcast", "rain": "rainy", "storm": "rainy", "fog": "overcast"}.get(wkind, "free")
    vis_range = {"fog": "150.0", "storm": "4000.0", "rain": "10000.0", "cloudy": "50000.0"}.get(wkind, "100000.0")
    precip_type = "rain" if wkind in ("rain", "storm") else "dry"
    precip_int = "0.9" if wkind == "storm" else ("0.6" if wkind == "rain" else "0.0")
    _sub(env, "TimeOfDay", animation="false", dateTime=dt_str)
    weather = _sub(env, "Weather", cloudState=cloud_state)
    _sub(weather, "Sun", intensity=sun_int, azimuth="0", elevation=sun_el)
    _sub(weather, "Fog", visualRange=vis_range)
    _sub(weather, "Precipitation", precipitationType=precip_type, intensity=precip_int)
    default_fric = 0.6 if wkind == "storm" else (0.7 if wkind == "rain" else 1.0)
    _sub(env, "RoadCondition", frictionScaleFactor="%.2f" % float(scenario.get("friction", default_fric)))
    for name, e in zip(names, ents):
        _add_init(init_actions, name, e)

    story = _sub(sb, "Story", name="RecordedStory")
    act = _sub(story, "Act", name="RecordedAct")
    for name, e in zip(names, ents):
        _add_trajectory_group(act, name, e)
    _time_trigger(_sub(act, "StartTrigger"), "ActStart", 0.0)
    _time_trigger(_sub(act, "StopTrigger"), "ActStop", duration)
    _time_trigger(_sub(sb, "StopTrigger"), "ScenarioEnd", duration)
    return root


def write_openscenario_xosc(scenario, output_path, xsd_path=None):
    """Write the scenario to output_path (pretty printed). When xsd_path is given the file is validated against it."""
    root = build_xosc(scenario)
    pretty = minidom.parseString(ET.tostring(root, encoding="utf-8")).toprettyxml(indent="  ")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(pretty, encoding="utf-8")
    if xsd_path is not None:
        validate_xosc(output_path, xsd_path)
    return output_path


def validate_xosc(xosc_path, xsd_path):
    """Raise xmlschema.XMLSchemaValidationError when the file does not match the OpenSCENARIO schema."""
    import xmlschema
    xmlschema.XMLSchema(str(xsd_path)).validate(str(xosc_path))
