"""
Corner-case dataset generator for CARLA (right-turn vs. crossing pedestrian, based on the AutoScenario
demo).  Each run randomises town / junction / timing / speed / weather / actor models, records RGB +
LiDAR from the ego vehicle V1 and writes per-frame labels.

  python generate_dataset.py --runs 3 --out ../dataset_test
  python generate_dataset.py --runs 200 --out ../dataset

Needs a running CARLA server (editor in Play mode) on localhost:2000.  Resumable: finished runs are skipped.
"""
import argparse
import csv
import json
import math
import os
import queue
import random
import shutil
import sys
import time

import numpy as np
import cv2
import carla

IMG_W, IMG_H, FOV = 960, 540, 90.0
DT = 0.05            # 20 Hz simulation
SAVE_EVERY = 2       # save every 2nd tick -> 10 Hz
NEAR_MISS_M = 3.0    # centre-to-centre distance below which a non-collision counts as a near miss


# ----------------------------------------------------------------------------- connection helpers
def connect(host, port, wait=600):
    t0 = time.time()
    while True:
        try:
            client = carla.Client(host, port)
            client.set_timeout(120.0)
            client.get_world()
            return client
        except RuntimeError as e:
            if time.time() - t0 > wait:
                raise
            print('waiting for CARLA server...', str(e)[:60], flush=True)
            time.sleep(10)


def list_towns(client, wanted):
    maps = client.get_available_maps()
    if wanted:
        out = []
        for w in wanted:
            out += [m for m in maps if m.endswith(w)]
        return out
    return [m for m in maps if m.split('/')[-1].startswith('Town') and m.endswith('_Opt')]


# ----------------------------------------------------------------------------- junction discovery
def find_right_turns(world):
    """Return (junction_id, entry_wp, exit_wp, n_roads) right-turn pairs at junctions with 3-6 roads."""
    carla_map = world.get_map()
    junctions = {}
    for wp in carla_map.generate_waypoints(4.0):
        if wp.is_junction:
            j = wp.get_junction()
            junctions.setdefault(j.id, j)
    pairs = []
    for j in junctions.values():
        wps = j.get_waypoints(carla.LaneType.Driving)
        n_roads = len({p[0].road_id for p in wps})
        if n_roads < 3 or n_roads > 6:          # skip tiny/odd junctions and huge interchanges
            continue
        for entry_wp, exit_wp in wps:
            d = (exit_wp.transform.rotation.yaw - entry_wp.transform.rotation.yaw + 180) % 360 - 180
            if 60 < d < 120 and entry_wp.previous(20.0):
                pairs.append((j.id, entry_wp, exit_wp, n_roads))
    return pairs


# ----------------------------------------------------------------------------- parameter sampling
def sample_params(rng):
    night = rng.random() < 0.25
    return {
        'approach_m': rng.uniform(6, 14),
        'v1_speed_diff': rng.uniform(0, 55),        # % slower than the speed limit
        'ped_lead_s': rng.uniform(-1.0, 3.0),       # >0: pedestrian starts earlier than needed to meet V1
        'ped_speed_ms': rng.uniform(0.8, 2.2),
        'ped_long_m': rng.uniform(2, 5),            # distance past the junction
        'ped_side_extra_m': rng.uniform(0, 1.5),
        'has_v2': rng.random() < 0.6,
        'has_barrier': rng.random() < 0.5,
        'weather': {
            'cloudiness': rng.uniform(0, 100),
            'precipitation': rng.choice([0, 0, rng.uniform(20, 100)]),
            'precipitation_deposits': rng.uniform(0, 80),
            'wind_intensity': rng.uniform(0, 60),
            'sun_azimuth_angle': rng.uniform(0, 360),
            'sun_altitude_angle': rng.uniform(-30, 5) if night else rng.uniform(10, 80),
            'fog_density': rng.choice([0, 0, rng.uniform(10, 60)]),
            'fog_distance': rng.uniform(0, 50),
            'wetness': rng.uniform(0, 80),
        },
        'night': night,
    }


# ----------------------------------------------------------------------------- geometry / labels
def project_bbox(actor, cam_tf_inv, K):
    """2D box [x1,y1,x2,y2] of the actor's 3D bounding box, or None when behind / outside the image."""
    verts = actor.bounding_box.get_world_vertices(actor.get_transform())
    pts = []
    for v in verts:
        p = cam_tf_inv @ np.array([v.x, v.y, v.z, 1.0])
        if p[0] > 0.1:                                # x is forward in UE camera space
            pts.append((K[0] * p[1] / p[0] + K[2], -K[1] * p[2] / p[0] + K[3]))
    if len(pts) < 8:          # part of the box is behind the camera: projection would be wrong
        return None
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x1, y1, x2, y2 = max(min(xs), 0), max(min(ys), 0), min(max(xs), IMG_W - 1), min(max(ys), IMG_H - 1)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return None
    return [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)]


def occluded(world, cam_loc, actor):
    """Heuristic: True when rays from the camera to head, chest and knee height of the actor are ALL blocked."""
    try:
        t = actor.get_transform()
        ext = actor.bounding_box.extent
        blocked = 0
        for k in (0.85, 0.0, -0.7):
            centre = t.transform(carla.Location(actor.bounding_box.location.x, actor.bounding_box.location.y,
                                                actor.bounding_box.location.z + ext.z * k))
            dist = cam_loc.distance(centre)
            hits = world.cast_ray(cam_loc, centre)
            if any(cam_loc.distance(h.location) < dist - 1.0 for h in hits):
                blocked += 1
        return blocked == 3
    except Exception:
        return None


def state(actor):
    t, v = actor.get_transform(), actor.get_velocity()
    return {'loc': [round(t.location.x, 3), round(t.location.y, 3), round(t.location.z, 3)],
            'yaw': round(t.rotation.yaw, 2),
            'vel': [round(v.x, 3), round(v.y, 3), round(v.z, 3)],
            'speed': round(v.length(), 3)}


def ttc(ego, ped):
    rp = ped.get_location() - ego.get_location()
    rv = ped.get_velocity() - ego.get_velocity()
    d = rp.length()
    if d < 1e-3:
        return 0.0
    closing = -(rp.x * rv.x + rp.y * rv.y + rp.z * rv.z) / d
    return round(d / closing, 2) if closing > 0.1 else None


# ----------------------------------------------------------------------------- one run
def run_one(client, world, tm, pairs, rng, out_dir, args):
    p = sample_params(rng)
    jid, entry_wp, exit_wp, n_roads = pairs[rng.randrange(len(pairs))]
    bp_lib = world.get_blueprint_library()
    actors, sensors = [], []

    def spawn(bp, tf, attach=None):
        a = world.try_spawn_actor(bp, tf, attach_to=attach)
        if a is not None:
            (sensors if attach is not None else actors).append(a)
        return a

    def lift(wp, dz=0.5):
        t = wp.transform
        return carla.Transform(carla.Location(t.location.x, t.location.y, t.location.z + dz),
                               carla.Rotation(yaw=t.rotation.yaw))

    veh_bps = [b for b in bp_lib.filter('vehicle.*') if b.has_attribute('number_of_wheels')
               and int(b.get_attribute('number_of_wheels')) == 4]
    ped_bps = list(bp_lib.filter('walker.pedestrian.*'))
    v1_bp, v2_bp, ped_bp = rng.choice(veh_bps), rng.choice(veh_bps), rng.choice(ped_bps)
    p['v1_blueprint'], p['v2_blueprint'], p['ped_blueprint'] = v1_bp.id, v2_bp.id, ped_bp.id
    p['junction_id'], p['junction_roads'] = jid, n_roads
    p['town'] = world.get_map().name

    world.set_weather(carla.WeatherParameters(**p['weather']))

    v1 = spawn(v1_bp, lift(entry_wp.previous(p['approach_m'])[0]))
    cross = exit_wp.next(p['ped_long_m'])
    cross = cross[0] if cross else exit_wp
    right = cross.transform.get_right_vector()
    ped_loc = cross.transform.location + right * (-(cross.lane_width / 2.0 + p['ped_side_extra_m']))
    ped = spawn(ped_bp, carla.Transform(carla.Location(ped_loc.x, ped_loc.y, ped_loc.z + 1.0),
                                        carla.Rotation(yaw=cross.transform.rotation.yaw + 90)))
    v2 = None
    if p['has_v2']:
        v2 = spawn(v2_bp, lift(entry_wp.previous(p['approach_m'] + 12.0)[0]))
    if p['has_barrier']:
        bps = bp_lib.filter('static.prop.streetbarrier')
        if bps:
            bpos = cross.transform.location + right * (cross.lane_width * 1.2)
            spawn(bps[0], carla.Transform(carla.Location(bpos.x, bpos.y, bpos.z + 0.5),
                                          carla.Rotation(yaw=cross.transform.rotation.yaw)))
    def cleanup():
        for s_ in sensors:
            try:
                s_.stop()
            except Exception:
                pass
        client.apply_batch([carla.command.DestroyActor(a.id) for a in sensors + actors])

    if v1 is None or ped is None:
        cleanup()
        return None, 'spawn_failed'

    # ---- sensors on V1
    ext = v1.bounding_box.extent
    cam_tf = carla.Transform(carla.Location(x=ext.x * 0.6, z=ext.z * 2.0 + 0.2))
    cam_bp = bp_lib.find('sensor.camera.rgb')
    cam_bp.set_attribute('image_size_x', str(IMG_W))
    cam_bp.set_attribute('image_size_y', str(IMG_H))
    cam_bp.set_attribute('fov', str(FOV))
    cam = spawn(cam_bp, cam_tf, attach=v1)
    cam_q = queue.Queue()
    cam.listen(cam_q.put)

    lid, lid_q = None, queue.Queue()
    if not args.no_lidar:
        lb = bp_lib.find('sensor.lidar.ray_cast')
        lb.set_attribute('channels', '32')
        lb.set_attribute('range', '60')
        lb.set_attribute('points_per_second', '200000')
        lb.set_attribute('rotation_frequency', str(1.0 / DT))
        lid = spawn(lb, carla.Transform(carla.Location(z=ext.z * 2.0 + 0.4)), attach=v1)
        lid.listen(lid_q.put)

    col = spawn(bp_lib.find('sensor.other.collision'), carla.Transform(), attach=v1)
    collisions = []
    col.listen(lambda ev: collisions.append({'other': ev.other_actor.type_id, 'frame': ev.frame,
                                             'impulse': round(ev.normal_impulse.length(), 1)}))

    # ---- view of the corner case: editor spectator (and optional overview camera) look at the crossing point
    fwd = cross.transform.get_forward_vector()
    cc = cross.transform.location
    view_tf = carla.Transform(carla.Location(cc.x - fwd.x * 14.0, cc.y - fwd.y * 14.0, cc.z + 9.0),
                              carla.Rotation(pitch=-33.0, yaw=cross.transform.rotation.yaw))
    try:
        world.get_spectator().set_transform(view_tf)
    except Exception:
        pass
    ov, ov_q = None, queue.Queue()
    if args.overview:
        ob = bp_lib.find('sensor.camera.rgb')
        ob.set_attribute('image_size_x', str(IMG_W))
        ob.set_attribute('image_size_y', str(IMG_H))
        ob.set_attribute('fov', '80')
        ov = spawn(ob, view_tf)
        ov.listen(ov_q.put)
        os.makedirs(os.path.join(out_dir, 'overview'), exist_ok=True)

    # ---- behaviour
    if v2 is not None:
        v2.apply_control(carla.VehicleControl(brake=1.0, hand_brake=True))
    goal = exit_wp.next(30.0)
    tm.set_path(v1, [(goal[0] if goal else exit_wp).transform.location])
    tm.ignore_walkers_percentage(v1, 100)
    tm.ignore_lights_percentage(v1, 100)
    tm.auto_lane_change(v1, False)
    tm.vehicle_percentage_speed_difference(v1, p['v1_speed_diff'])
    v1.set_autopilot(True, tm.get_port())

    K = (IMG_W / (2 * math.tan(math.radians(FOV / 2))),) * 2 + (IMG_W / 2, IMG_H / 2)
    os.makedirs(os.path.join(out_dir, 'rgb'), exist_ok=True)
    if lid:
        os.makedirs(os.path.join(out_dir, 'lidar'), exist_ok=True)

    frames, min_dist, impact_speed, end_tick = [], 1e9, None, None
    max_ticks = int(args.max_sec / DT)
    walking = False
    t_ped = (cross.lane_width / 2.0 + p['ped_side_extra_m']) / p['ped_speed_ms']   # time to reach the lane centre
    walk_dir = carla.Vector3D(right.x, right.y, 0.0)
    boxed = [a for a in (ped, v2) if a is not None]
    for tick in range(max_ticks):
        if not walking and (v1.get_location().distance(cross.transform.location) /
                            max(v1.get_velocity().length(), 1.5) <= t_ped + p['ped_lead_s'] or tick * DT > 20):
            p['ped_start_t'] = round(tick * DT, 2)
            ped.apply_control(carla.WalkerControl(direction=walk_dir, speed=p['ped_speed_ms']))
            walking = True
        world.tick()
        cimg = cam_q.get(timeout=30)
        limg = lid_q.get(timeout=30) if lid else None
        oimg = ov_q.get(timeout=30) if ov else None

        d = v1.get_location().distance(ped.get_location())
        min_dist = min(min_dist, d)
        ped_hit = [c for c in collisions if c['other'].startswith('walker')]
        if ped_hit and impact_speed is None:
            impact_speed = round(v1.get_velocity().length() * 3.6, 1)
            end_tick = tick + int(1.0 / DT)

        if tick * DT > 8.0 and not walking and v1.get_velocity().length() < 0.3:
            cleanup()
            return None, 'v1_stuck'

        if tick % SAVE_EVERY == 0:
            name = f'{tick:05d}'
            arr = np.frombuffer(cimg.raw_data, dtype=np.uint8).reshape(IMG_H, IMG_W, 4)[:, :, :3]
            cv2.imwrite(os.path.join(out_dir, 'rgb', name + '.jpg'), arr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if oimg is not None:
                oarr = np.frombuffer(oimg.raw_data, dtype=np.uint8).reshape(IMG_H, IMG_W, 4)[:, :, :3]
                cv2.imwrite(os.path.join(out_dir, 'overview', name + '.jpg'), oarr, [cv2.IMWRITE_JPEG_QUALITY, 92])
            if limg is not None:
                pts = np.frombuffer(limg.raw_data, dtype=np.float32).reshape(-1, 4)
                np.save(os.path.join(out_dir, 'lidar', name + '.npy'), pts.astype(np.float16))
            inv = np.array(cam.get_transform().get_inverse_matrix())
            cam_loc = cam.get_transform().location
            boxes = []
            for a in boxed:
                b = project_bbox(a, inv, K)
                if b is not None:
                    boxes.append({'cls': 'pedestrian' if a is ped else 'vehicle', 'actor_id': a.id,
                                  'bbox': b, 'distance': round(cam_loc.distance(a.get_location()), 2),
                                  'occluded': occluded(world, cam_loc, a)})
            ctl = v1.get_control()
            frames.append({'tick': tick, 't': round(tick * DT, 2), 'image': f'rgb/{name}.jpg',
                           'lidar': f'lidar/{name}.npy' if lid else None,
                           'ego': dict(state(v1), control={'throttle': round(ctl.throttle, 3),
                                                           'steer': round(ctl.steer, 3),
                                                           'brake': round(ctl.brake, 3)}),
                           'ped': state(ped), 'dist_to_ped': round(d, 2), 'ttc': ttc(v1, ped),
                           'boxes': boxes})
        if end_tick is not None and tick >= end_tick:
            break

    outcome = 'collision_pedestrian' if impact_speed is not None else (
        'near_miss' if min_dist < NEAR_MISS_M else 'safe')
    meta = {'params': p, 'outcome': outcome, 'min_dist_m': round(min_dist, 2),
            'impact_speed_kmh': impact_speed, 'collisions': collisions[:10],
            'n_frames': len(frames), 'camera': {'width': IMG_W, 'height': IMG_H, 'fov': FOV,
                                                'K': [round(k, 3) for k in K], 'save_hz': 1.0 / (DT * SAVE_EVERY)},
            'complete': True}
    with open(os.path.join(out_dir, 'labels.json'), 'w') as f:
        json.dump({'frames': frames}, f)
    with open(os.path.join(out_dir, 'meta.json'), 'w') as f:
        json.dump(meta, f, indent=1)

    for _ in range(int(args.hold_sec / DT)):      # keep the final scene on screen
        world.tick()
        for q_ in (cam_q, lid_q, ov_q):
            while not q_.empty():
                q_.get()
    cleanup()
    return meta, outcome


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', type=int, default=3)
    ap.add_argument('--out', default='dataset_test')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--towns', nargs='*', default=None, help='map name suffixes, e.g. Town10HD_Opt Town05_Opt')
    ap.add_argument('--max-sec', type=float, default=14.0)
    ap.add_argument('--no-lidar', action='store_true')
    ap.add_argument('--overview', action='store_true', help='also save a fixed camera looking at the crossing (overview/)')
    ap.add_argument('--hold-sec', type=float, default=0.0, help='keep the final scene visible in the editor for N seconds')
    ap.add_argument('--host', default='localhost')
    ap.add_argument('--port', type=int, default=2000)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    manifest = os.path.join(args.out, 'manifest.csv')
    new_manifest = not os.path.exists(manifest)
    client = connect(args.host, args.port)
    towns = list_towns(client, args.towns)
    if not towns:
        sys.exit('no matching maps: ' + str(client.get_available_maps()))
    print('towns:', towns, flush=True)

    world = client.get_world()
    tm = client.get_trafficmanager()
    pairs_cache = {}
    try:
        for i in range(args.runs):
            run_dir = os.path.join(args.out, f'run_{i:04d}')
            meta_fn = os.path.join(run_dir, 'meta.json')
            if os.path.exists(meta_fn) and json.load(open(meta_fn)).get('complete'):
                continue
            if os.path.exists(run_dir):
                shutil.rmtree(run_dir)
            town = towns[i * len(towns) // args.runs]
            rng = random.Random(args.seed * 100003 + i)
            t0 = time.time()
            try:
                if world.get_map().name != town:
                    world.apply_settings(carla.WorldSettings(synchronous_mode=False))
                    world = client.load_world(town)
                    time.sleep(3)
                    tm = client.get_trafficmanager()
                if town not in pairs_cache:
                    pairs_cache[town] = find_right_turns(world)
                    print(town, 'right-turn candidates:', len(pairs_cache[town]), flush=True)
                if not pairs_cache[town]:
                    print('no junction in', town, '- skipping run', i)
                    continue
                world.apply_settings(carla.WorldSettings(synchronous_mode=True, fixed_delta_seconds=DT))
                tm.set_synchronous_mode(True)
                tm.set_random_device_seed(args.seed * 100003 + i)
                meta, outcome = run_one(client, world, tm, pairs_cache[town], rng, run_dir, args)
            except Exception as e:     # server crash, timeouts, ...
                print(f'run {i} failed: {type(e).__name__}: {str(e)[:120]}', flush=True)
                if os.path.exists(run_dir):
                    shutil.rmtree(run_dir, ignore_errors=True)
                client = connect(args.host, args.port)
                world = client.get_world()
                tm = client.get_trafficmanager()
                pairs_cache.clear()
                continue
            if meta is None:
                shutil.rmtree(run_dir, ignore_errors=True)
                print(f'run {i}: {outcome} (discarded)', flush=True)
                continue
            with open(manifest, 'a', newline='') as f:
                w = csv.writer(f)
                if new_manifest:
                    w.writerow(['run', 'town', 'outcome', 'min_dist_m', 'impact_speed_kmh', 'n_frames', 'seconds'])
                    new_manifest = False
                w.writerow([i, town, outcome, meta['min_dist_m'], meta['impact_speed_kmh'],
                            meta['n_frames'], round(time.time() - t0, 1)])
            print(f'run {i}: {outcome}, min_dist {meta["min_dist_m"]} m, {meta["n_frames"]} frames, '
                  f'{time.time() - t0:.0f}s', flush=True)
    finally:
        try:
            world.apply_settings(carla.WorldSettings(synchronous_mode=False, fixed_delta_seconds=0.0))
            tm.set_synchronous_mode(False)
            print('world restored to asynchronous mode')
        except Exception as e:
            print('could not restore async mode:', e)


if __name__ == '__main__':
    main()
