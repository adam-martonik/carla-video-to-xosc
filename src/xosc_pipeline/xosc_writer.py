"""ASAM OpenSCENARIO 1.0 (.xosc) generator with logical parameterization and XSD validation."""

from __future__ import annotations

import datetime as dt_mod
import math
import xml.dom.minidom as minidom
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict

import xmlschema

from .config import PipelineConfig


# Valid ASAM OpenSCENARIO 1.0 VehicleCategory enum values
VALID_VEHICLE_CATEGORIES = frozenset({
    "car", "van", "truck", "trailer", "semitrailer", "bus", "motorbike", "bicycle", "train", "tram"
})
VALID_PEDESTRIAN_CATEGORIES = frozenset({"pedestrian", "wheelchair", "animal"})


def _add_parameter_declarations(root: ET.Element, projected: Dict[str, Any], cfg: PipelineConfig) -> None:
    """Emit logical scenario <ParameterDeclarations> per Menzel et al. (2018) / de Gelder (2024)."""
    pdecl = ET.SubElement(root, "ParameterDeclarations")
    ego_v0 = float(projected["ego"]["v"][0]) if projected["ego"]["v"] else 15.0
    is_night = bool(projected.get("is_night", False))
    duration = max(5.0, float(projected.get("duration", 15.0)))

    params = [
        ("EgoInitialSpeed", "double", f"{ego_v0:.2f}"),
        ("NpcSpeedScale", "double", "1.00"),
        ("RoadFriction", "double", f"{cfg.scenario.default_friction:.2f}"),
        ("SunElevation", "double", "-0.35" if is_night else "1.20"),
        ("SunIntensity", "double", "0.02" if is_night else "0.90"),
        ("ScenarioDuration", "double", f"{duration:.2f}"),
    ]
    for name, ptype, val in params:
        ET.SubElement(
            pdecl,
            "ParameterDeclaration",
            attrib={"name": name, "parameterType": ptype, "value": val},
        )


def _add_entity(entities_el: ET.Element, ent: Dict[str, Any], cfg: PipelineConfig) -> None:
    """Emit class-accurate <ScenarioObject> with realistic dimensions, axles, and performance."""
    cid = int(ent.get("class_id", 2))
    spec = cfg.get_class_spec(cid)
    so = ET.SubElement(entities_el, "ScenarioObject", attrib={"name": ent["role_name"]})

    if ent.get("entity_kind") == "pedestrian":
        ped_cat = spec.category if spec.category in VALID_PEDESTRIAN_CATEGORIES else "pedestrian"
        ped = ET.SubElement(
            so,
            "Pedestrian",
            attrib={
                "model": ent["blueprint"],
                "mass": f"{spec.mass:.1f}",
                "name": ent["blueprint"],
                "pedestrianCategory": ped_cat,
            },
        )
        ET.SubElement(ped, "ParameterDeclarations")
        bb = ET.SubElement(ped, "BoundingBox")
        ET.SubElement(bb, "Center", attrib={"x": "0.0", "y": "0.0", "z": f"{spec.height * 0.5:.2f}"})
        ET.SubElement(
            bb,
            "Dimensions",
            attrib={"width": f"{spec.width:.2f}", "length": f"{spec.length:.2f}", "height": f"{spec.height:.2f}"},
        )
        props = ET.SubElement(ped, "Properties")
        ET.SubElement(props, "Property", attrib={"name": "type", "value": "simulation"})
        return

    veh_cat = spec.category if spec.category in VALID_VEHICLE_CATEGORIES else "car"
    veh = ET.SubElement(
        so,
        "Vehicle",
        attrib={"name": ent["blueprint"], "vehicleCategory": veh_cat},
    )
    ET.SubElement(veh, "ParameterDeclarations")
    ET.SubElement(
        veh,
        "Performance",
        attrib={
            "maxSpeed": f"{spec.max_speed:.2f}",
            "maxAcceleration": f"{spec.max_accel:.2f}",
            "maxDeceleration": f"{spec.max_decel:.2f}",
        },
    )
    bb = ET.SubElement(veh, "BoundingBox")
    # Center relative to rear axle origin
    center_x = spec.wheelbase * 0.5
    ET.SubElement(bb, "Center", attrib={"x": f"{center_x:.2f}", "y": "0.0", "z": f"{spec.height * 0.5:.2f}"})
    ET.SubElement(
        bb,
        "Dimensions",
        attrib={"width": f"{spec.width:.2f}", "length": f"{spec.length:.2f}", "height": f"{spec.height:.2f}"},
    )
    axles = ET.SubElement(veh, "Axles")
    ET.SubElement(
        axles,
        "FrontAxle",
        attrib={
            "maxSteering": f"{spec.max_steering:.2f}",
            "wheelDiameter": f"{spec.wheel_diameter:.2f}",
            "trackWidth": f"{spec.track_width:.2f}",
            "positionX": f"{spec.wheelbase:.2f}",
            "positionZ": f"{spec.wheel_diameter * 0.5:.2f}",
        },
    )
    ET.SubElement(
        axles,
        "RearAxle",
        attrib={
            "maxSteering": "0.0",
            "wheelDiameter": f"{spec.wheel_diameter:.2f}",
            "trackWidth": f"{spec.track_width:.2f}",
            "positionX": "0.0",
            "positionZ": f"{spec.wheel_diameter * 0.5:.2f}",
        },
    )
    props = ET.SubElement(veh, "Properties")
    ent_type = "ego_vehicle" if ent["role_name"] == "hero" else "simulation"
    ET.SubElement(props, "Property", attrib={"name": "type", "value": ent_type})
    ET.SubElement(props, "Property", attrib={"name": "color", "value": ent.get("color", "200,200,200")})


def write_openscenario_xosc(
    projected: Dict[str, Any],
    output_xosc: Path,
    cfg: PipelineConfig,
    xsd_path: Path | None = None,
    use_xodr_logic_file: bool = False,
) -> Path:
    """
    Generate a schema-valid ASAM OpenSCENARIO 1.0 (.xosc) file with:
      - Logical <ParameterDeclarations>
      - Class-accurate vehicle/pedestrian dimensions, axles, and <Performance> limits
      - Fine-grained <Polyline> trajectories starting at t=0.0s
      - Smooth linear <SpeedAction> speed profile updates
      - Valid <Storyboard><StopTrigger>
    """
    root = ET.Element("OpenSCENARIO")
    now_iso = dt_mod.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    ET.SubElement(
        root,
        "FileHeader",
        attrib={
            "revMajor": str(cfg.scenario.rev_major),
            "revMinor": str(cfg.scenario.rev_minor),
            "date": now_iso,
            "description": cfg.scenario.description,
            "author": cfg.scenario.author,
        },
    )
    _add_parameter_declarations(root, projected, cfg)
    ET.SubElement(root, "CatalogLocations")

    road_net = ET.SubElement(root, "RoadNetwork")
    if use_xodr_logic_file and projected.get("xodr_path"):
        logic_fp = Path(projected["xodr_path"]).name
    else:
        logic_fp = projected.get("town", "Town04_Opt")
    ET.SubElement(road_net, "LogicFile", attrib={"filepath": logic_fp})
    ET.SubElement(road_net, "SceneGraphFile", attrib={"filepath": ""})

    entities_el = ET.SubElement(root, "Entities")
    all_entities = [projected["ego"]] + projected["actors"]
    for ent in all_entities:
        _add_entity(entities_el, ent, cfg)

    storyboard = ET.SubElement(root, "Storyboard")
    init_el = ET.SubElement(storyboard, "Init")
    init_actions = ET.SubElement(init_el, "Actions")

    # Global EnvironmentAction parameterized with logical scenario declarations
    is_night = bool(projected.get("is_night", False))
    duration = max(5.0, float(projected.get("duration", 15.0)))
    ga = ET.SubElement(init_actions, "GlobalAction")
    ea = ET.SubElement(ga, "EnvironmentAction")
    env = ET.SubElement(ea, "Environment", attrib={"name": "DashcamEnv"})
    ET.SubElement(
        env,
        "TimeOfDay",
        attrib={"animation": "false", "dateTime": "2026-06-15T23:30:00" if is_night else "2026-06-15T14:00:00"},
    )
    sun_elev = "-0.35" if is_night else "1.20"
    sun_inten = "0.02" if is_night else "0.90"
    fric_val = f"{cfg.scenario.default_friction:.2f}"

    weather = ET.SubElement(env, "Weather", attrib={"cloudState": "free"})
    ET.SubElement(
        weather,
        "Sun",
        attrib={"intensity": sun_inten, "azimuth": "0", "elevation": sun_elev},
    )
    ET.SubElement(weather, "Fog", attrib={"visualRange": "100000.0"})
    ET.SubElement(weather, "Precipitation", attrib={"precipitationType": "dry", "intensity": "0.0"})
    ET.SubElement(env, "RoadCondition", attrib={"frictionScaleFactor": fric_val})

    # Init PrivateActions
    for ent in all_entities:
        priv = ET.SubElement(init_actions, "Private", attrib={"entityRef": ent["role_name"]})
        p0 = ent["world_poses"][0]
        pa_tele = ET.SubElement(priv, "PrivateAction")
        tele = ET.SubElement(pa_tele, "TeleportAction")
        pos = ET.SubElement(tele, "Position")
        ET.SubElement(
            pos,
            "WorldPosition",
            attrib={
                "x": f"{p0['x']:.3f}",
                "y": f"{p0['y']:.3f}",
                "z": f"{p0['z']:.3f}",
                "h": f"{p0['h']:.5f}",
            },
        )

        if ent.get("entity_kind") != "pedestrian":
            pa_ctrl = ET.SubElement(priv, "PrivateAction")
            ctrl_act = ET.SubElement(pa_ctrl, "ControllerAction")
            assign_ctrl = ET.SubElement(ctrl_act, "AssignControllerAction")
            ctrl = ET.SubElement(assign_ctrl, "Controller", attrib={"name": f"{ent['role_name']}_ctrl"})
            cprops = ET.SubElement(ctrl, "Properties")
            ET.SubElement(cprops, "Property", attrib={"name": "module", "value": "simple_vehicle_control"})
            ET.SubElement(
                cprops,
                "Property",
                attrib={"name": "waypoint_reached_threshold", "value": f"{cfg.scenario.waypoint_reached_threshold_m:.1f}"},
            )
            override = ET.SubElement(ctrl_act, "OverrideControllerValueAction")
            ET.SubElement(override, "Throttle", attrib={"value": "0", "active": "false"})
            ET.SubElement(override, "Brake", attrib={"value": "0", "active": "false"})
            ET.SubElement(override, "Clutch", attrib={"value": "0", "active": "false"})
            ET.SubElement(override, "ParkingBrake", attrib={"value": "0", "active": "false"})
            ET.SubElement(override, "SteeringWheel", attrib={"value": "0", "active": "false"})
            ET.SubElement(override, "Gear", attrib={"number": "0", "active": "false"})

            v0_str = "$EgoInitialSpeed" if ent["role_name"] == "hero" else f"{float(ent['v'][0]):.2f}"
            pa_spd = ET.SubElement(priv, "PrivateAction")
            long_act = ET.SubElement(pa_spd, "LongitudinalAction")
            spd_act = ET.SubElement(long_act, "SpeedAction")
            ET.SubElement(
                spd_act,
                "SpeedActionDynamics",
                attrib={"dynamicsShape": "step", "value": "0.0", "dynamicsDimension": "time"},
            )
            spd_tgt = ET.SubElement(spd_act, "SpeedActionTarget")
            ET.SubElement(spd_tgt, "AbsoluteTargetSpeed", attrib={"value": v0_str})

    story = ET.SubElement(storyboard, "Story", attrib={"name": "DashcamReconstructedStory"})
    act = ET.SubElement(story, "Act", attrib={"name": "DashcamReconstructedAct"})

    for ent in all_entities:
        rname = ent["role_name"]
        mg = ET.SubElement(act, "ManeuverGroup", attrib={"name": f"MG_{rname}", "maximumExecutionCount": "1"})
        actors_el = ET.SubElement(mg, "Actors", attrib={"selectTriggeringEntities": "false"})
        ET.SubElement(actors_el, "EntityRef", attrib={"entityRef": rname})
        man = ET.SubElement(mg, "Maneuver", attrib={"name": f"Maneuver_{rname}"})

        if ent.get("entity_kind") == "pedestrian":
            ev = ET.SubElement(man, "Event", attrib={"name": f"Event_Wait_{rname}", "priority": "overwrite"})
            action_el = ET.SubElement(ev, "Action", attrib={"name": f"Action_Wait_{rname}"})
            pa_spd = ET.SubElement(action_el, "PrivateAction")
            long_act = ET.SubElement(pa_spd, "LongitudinalAction")
            spd_act = ET.SubElement(long_act, "SpeedAction")
            ET.SubElement(
                spd_act,
                "SpeedActionDynamics",
                attrib={"dynamicsShape": "linear", "value": "1.0", "dynamicsDimension": "time"},
            )
            spd_tgt = ET.SubElement(spd_act, "SpeedActionTarget")
            ET.SubElement(spd_tgt, "AbsoluteTargetSpeed", attrib={"value": "0.0"})
            st = ET.SubElement(ev, "StartTrigger")
            cg = ET.SubElement(st, "ConditionGroup")
            cond = ET.SubElement(cg, "Condition", attrib={"name": f"StartCond_{rname}", "delay": "0", "conditionEdge": "rising"})
            bvc = ET.SubElement(cond, "ByValueCondition")
            ET.SubElement(bvc, "SimulationTimeCondition", attrib={"value": "0.0", "rule": "greaterThan"})
            continue

        # 1. FollowTrajectoryAction starting cleanly at t=0.0s
        ev = ET.SubElement(man, "Event", attrib={"name": f"Event_Trajectory_{rname}", "priority": "overwrite"})
        action_el = ET.SubElement(ev, "Action", attrib={"name": f"Action_FollowTrajectory_{rname}"})
        pa_route = ET.SubElement(action_el, "PrivateAction")
        routing_act = ET.SubElement(pa_route, "RoutingAction")
        follow_traj = ET.SubElement(routing_act, "FollowTrajectoryAction")
        traj = ET.SubElement(follow_traj, "Trajectory", attrib={"name": f"Traj_{rname}", "closed": "false"})
        shape = ET.SubElement(traj, "Shape")
        polyline = ET.SubElement(shape, "Polyline")

        prev_pose = None
        added_vertices = 0
        for idx, (t_val, pose) in enumerate(zip(ent["times"], ent["world_poses"])):
            if prev_pose is not None and added_vertices >= 2:
                dist = math.hypot(pose["x"] - prev_pose["x"], pose["y"] - prev_pose["y"])
                # Skip only truly stationary duplicates (full stop) so slow motion keeps uniform dt
                if dist < 0.005 and idx < len(ent["times"]) - 1:
                    continue
            vtx = ET.SubElement(polyline, "Vertex", attrib={"time": f"{t_val:.3f}"})
            vpos = ET.SubElement(vtx, "Position")
            ET.SubElement(
                vpos,
                "WorldPosition",
                attrib={
                    "x": f"{pose['x']:.3f}",
                    "y": f"{pose['y']:.3f}",
                    "z": f"{pose['z']:.3f}",
                    "h": f"{pose['h']:.5f}",
                },
            )
            prev_pose = pose
            added_vertices += 1

        ET.SubElement(follow_traj, "TimeReference")
        ET.SubElement(follow_traj, "TrajectoryFollowingMode", attrib={"followingMode": "position"})

        st = ET.SubElement(ev, "StartTrigger")
        cg = ET.SubElement(st, "ConditionGroup")
        cond = ET.SubElement(cg, "Condition", attrib={"name": f"StartCond_{rname}", "delay": "0", "conditionEdge": "rising"})
        bvc = ET.SubElement(cond, "ByValueCondition")
        ET.SubElement(bvc, "SimulationTimeCondition", attrib={"value": "0.0", "rule": "greaterThan"})

        # 2. Smooth linear SpeedAction events for speed transitions
        prev_v = float(ent["v"][0])
        prev_t = float(ent["times"][0])
        for idx, (t_val, v_val) in enumerate(zip(ent["times"][1:], ent["v"][1:]), start=1):
            if abs(v_val - prev_v) < cfg.scenario.speed_action_threshold_mps and v_val > 0.1:
                continue
            trans_dt = max(0.25, float(t_val) - prev_t)
            prev_v = float(v_val)
            prev_t = float(t_val)
            ev_s = ET.SubElement(man, "Event", attrib={"name": f"Event_Speed_{rname}_{idx}", "priority": "parallel"})
            act_s = ET.SubElement(ev_s, "Action", attrib={"name": f"Action_Speed_{rname}_{idx}"})
            pa_s = ET.SubElement(act_s, "PrivateAction")
            la_s = ET.SubElement(pa_s, "LongitudinalAction")
            sa_s = ET.SubElement(la_s, "SpeedAction")
            ET.SubElement(
                sa_s,
                "SpeedActionDynamics",
                attrib={
                    "dynamicsShape": cfg.scenario.speed_action_shape,
                    "value": f"{min(trans_dt, 1.5):.2f}",
                    "dynamicsDimension": "time",
                },
            )
            stgt = ET.SubElement(sa_s, "SpeedActionTarget")
            ET.SubElement(stgt, "AbsoluteTargetSpeed", attrib={"value": f"{v_val:.2f}"})

            st_s = ET.SubElement(ev_s, "StartTrigger")
            cg_s = ET.SubElement(st_s, "ConditionGroup")
            cond_s = ET.SubElement(
                cg_s,
                "Condition",
                attrib={"name": f"Cond_Speed_{rname}_{idx}", "delay": "0", "conditionEdge": "rising"},
            )
            bvc_s = ET.SubElement(cond_s, "ByValueCondition")
            ET.SubElement(bvc_s, "SimulationTimeCondition", attrib={"value": f"{t_val:.3f}", "rule": "greaterThan"})

    act_start = ET.SubElement(act, "StartTrigger")
    act_start_cg = ET.SubElement(act_start, "ConditionGroup")
    act_start_cond = ET.SubElement(act_start_cg, "Condition", attrib={"name": "ActStartCond", "delay": "0", "conditionEdge": "rising"})
    act_start_bvc = ET.SubElement(act_start_cond, "ByValueCondition")
    ET.SubElement(act_start_bvc, "SimulationTimeCondition", attrib={"value": "0.0", "rule": "greaterThan"})

    act_stop = ET.SubElement(act, "StopTrigger")
    act_stop_cg = ET.SubElement(act_stop, "ConditionGroup")
    act_stop_cond = ET.SubElement(act_stop_cg, "Condition", attrib={"name": "ActStopCond", "delay": "0", "conditionEdge": "rising"})
    act_stop_bvc = ET.SubElement(act_stop_cond, "ByValueCondition")
    ET.SubElement(act_stop_bvc, "SimulationTimeCondition", attrib={"value": f"{duration:.2f}", "rule": "greaterThan"})

    # Clean, valid Storyboard StopTrigger with criteria for CARLA ScenarioRunner
    sb_stop = ET.SubElement(storyboard, "StopTrigger")
    sb_cg = ET.SubElement(sb_stop, "ConditionGroup")
    sb_cond = ET.SubElement(sb_cg, "Condition", attrib={"name": "criteria_CollisionTest", "delay": "0", "conditionEdge": "rising"})
    sb_bvc = ET.SubElement(sb_cond, "ByValueCondition")
    ET.SubElement(sb_bvc, "ParameterCondition", attrib={"parameterRef": "", "value": "", "rule": "lessThan"})

    xml_bytes = ET.tostring(root, encoding="utf-8")
    pretty_xml = minidom.parseString(xml_bytes).toprettyxml(indent="  ")
    output_xosc.parent.mkdir(parents=True, exist_ok=True)
    output_xosc.write_text(pretty_xml, encoding="utf-8")
    print(f"[Stage E] Wrote OpenSCENARIO 1.0 file -> {output_xosc}")

    if xsd_path is not None and xsd_path.exists():
        xsd = xmlschema.XMLSchema(str(xsd_path))
        xsd.validate(str(output_xosc))
        print(f"[Stage E] XML Schema validation PASSED against {xsd_path.name}!")

    return output_xosc
