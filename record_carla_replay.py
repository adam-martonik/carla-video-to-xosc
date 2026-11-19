#!/usr/bin/env python
"""
record_carla_replay.py — Attaches a dashcam RGB sensor to 'hero', turns on vehicle
headlights, triggers walker ragdoll physics launch on collision impact, moves the
CARLA window spectator camera to follow 'hero' in real time (chase-cam), and records
demo/carla_replay_dashcam.mp4 + demo/carla_replay_telemetry.json.
"""

import json
import math
import time
from pathlib import Path

import carla
import cv2
import numpy as np

work_dir = Path(r"C:\Users\adamk\.gemini\antigravity\scratch\carla-video-to-xosc")
client = carla.Client("127.0.0.1", 2000)
client.set_timeout(30.0)

print("[CarlaRecorder] Waiting for 'hero' vehicle to spawn (up to 300s)...")
hero = None
for _ in range(1200):
    try:
        world = client.get_world()
        actors = world.get_actors()
        for a in actors:
            if a.attributes.get("role_name") == "hero":
                hero = a
                break
        if hero is not None:
            break
    except Exception:
        pass
    time.sleep(0.25)

if hero is None:
    print("[CarlaRecorder] Hero not found after 300s! Make sure scenario_runner.py is running.")
    raise SystemExit(1)

print(f"[CarlaRecorder] Found 'hero' (id={hero.id}, type={hero.type_id})! Attaching camera & spectator follow-cam...")
world = client.get_world()
spectator = world.get_spectator()

# Turn on headlights (LowBeam + HighBeam + Position) on all spawned vehicles
try:
    lights = carla.VehicleLightState(
        carla.VehicleLightState.Position | carla.VehicleLightState.LowBeam | carla.VehicleLightState.HighBeam
    )
    for v in world.get_actors().filter("vehicle.*"):
        v.set_light_state(lights)
except Exception:
    pass

bp_lib = world.get_blueprint_library()
cam_bp = bp_lib.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x", "1280")
cam_bp.set_attribute("image_size_y", "720")
cam_bp.set_attribute("fov", "70")

cam_tf = carla.Transform(carla.Location(x=1.5, y=0.0, z=1.45))
camera = world.spawn_actor(cam_bp, cam_tf, attach_to=hero)

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
video_out = work_dir / "demo/carla_replay_dashcam.mp4"
video_out.parent.mkdir(parents=True, exist_ok=True)
writer = cv2.VideoWriter(str(video_out), fourcc, 20.0, (1280, 720))
latest_frame = [None]


def on_image(image):
    arr = np.frombuffer(image.raw_data, dtype=np.uint8).reshape((image.height, image.width, 4))
    bgr = arr[:, :, :3].copy()
    latest_frame[0] = bgr


camera.listen(on_image)

telemetry = []
t0 = time.time()
saved_frames = 0
launched_walkers = set()

try:
    while (time.time() - t0) < 60.0:
        all_actors = world.get_actors()
        role_map = {a.attributes.get("role_name"): a for a in all_actors if a.attributes.get("role_name")}
        if "hero" not in role_map:
            break
        hero_actor = role_map["hero"]
        h_tf = hero_actor.get_transform()
        h_loc = h_tf.location
        h_vel = hero_actor.get_velocity()
        h_spd = math.hypot(h_vel.x, h_vel.y)
        fwd = h_tf.get_forward_vector()
        right = h_tf.get_right_vector()

        # Move CARLA window Spectator camera behind and above 'hero' (3rd-person chase view)
        spec_loc = carla.Location(
            x=h_loc.x - fwd.x * 9.0,
            y=h_loc.y - fwd.y * 9.0,
            z=h_loc.z + 4.0,
        )
        spec_rot = carla.Rotation(pitch=-15.0, yaw=h_tf.rotation.yaw, roll=0.0)
        spectator.set_transform(carla.Transform(spec_loc, spec_rot))

        snap = {
            "t": round(time.time() - t0, 3),
            "hero": {"x": h_loc.x, "y": h_loc.y, "z": h_loc.z, "speed_mps": h_spd},
            "npcs": {},
        }
        for rname, act in role_map.items():
            if rname == "hero":
                continue
            aloc = act.get_location()
            avel = act.get_velocity()
            dist = math.hypot(aloc.x - h_loc.x, aloc.y - h_loc.y)

            # If hero impacts a crossing walker/deer obstacle, enable physics & launch it off the hood
            if act.type_id.startswith("walker.") and dist < 4.2 and act.id not in launched_walkers:
                launched_walkers.add(act.id)
                try:
                    act.set_simulate_physics(True)
                    act.set_target_velocity(
                        carla.Vector3D(
                            fwd.x * 12.0 - right.x * 5.0,
                            fwd.y * 12.0 - right.y * 5.0,
                            7.5,
                        )
                    )
                    print(f"[CarlaRecorder] IMPACT with {rname} at t={snap['t']:.2f}s (dist={dist:.2f}m)! Launching obstacle physics.")
                except Exception:
                    pass

            snap["npcs"][rname] = {
                "x": aloc.x,
                "y": aloc.y,
                "z": aloc.z,
                "speed_mps": math.hypot(avel.x, avel.y),
                "dist_to_hero_m": dist,
            }
        telemetry.append(snap)
        if latest_frame[0] is not None:
            frame = latest_frame[0].copy()
            status_str = "STOPPED (0 km/h)" if h_spd < 0.3 and snap["t"] > 8.0 else f"v={h_spd*3.6:4.1f}km/h"
            cv2.putText(
                frame,
                f"CARLA 1.0 Replay | t={snap['t']:4.1f}s | Hero {status_str}",
                (20, 36),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 80, 255) if launched_walkers else (255, 255, 255),
                2,
            )
            y_off = 70
            for rname, nd in snap["npcs"].items():
                cv2.putText(
                    frame,
                    f"{rname}: dist={nd['dist_to_hero_m']:.1f}m v={nd['speed_mps']*3.6:.1f}km/h",
                    (20, y_off),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (80, 230, 80),
                    2,
                )
                y_off += 28
            writer.write(frame)
            saved_frames += 1
        time.sleep(0.05)
finally:
    try:
        camera.stop()
        camera.destroy()
    except Exception:
        pass
    writer.release()
    (work_dir / "demo/carla_replay_telemetry.json").write_text(json.dumps(telemetry, indent=2), encoding="utf-8")
    print(f"[CarlaRecorder] Recorded {len(telemetry)} telemetry samples and {saved_frames} video frames -> {video_out}")
