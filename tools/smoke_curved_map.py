"""
Gate for curved roads: does CARLA accept a SUMO-exported curved road and can a car drive along it?

  python -m tools.smoke_curved_map          (CARLA must be running: editor started and Play pressed)

Prints "CURVE OK" and exits 0, or "CURVE FAILED: <reason>" and exits 1.
"""
import math
import os
import re
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

RADIUS = 250.0
DT = 0.05
RUN_S = 15.0
MIN_TRAVEL_M = 100.0
MAX_DEVIATION_M = 1.5


def build_net(tmp):
    from agents.map_builder import find_netconvert
    pts = [(RADIUS * math.sin(a), RADIUS * (1 - math.cos(a))) for a in [i * (math.pi / 2) / 45 for i in range(46)]]
    nod, edg, net = (os.path.join(tmp, n) for n in ("arc.nod.xml", "arc.edg.xml", "arc.net.xml"))
    with open(nod, "w") as f:
        f.write('<nodes><node id="a" x="%.2f" y="%.2f" type="priority"/><node id="b" x="%.2f" y="%.2f" type="priority"/></nodes>' % (
            pts[0][0], pts[0][1], pts[-1][0], pts[-1][1]))
    shape = " ".join("%.2f,%.2f" % p for p in pts)
    with open(edg, "w") as f:
        f.write('<edges><edge id="arc" from="a" to="b" numLanes="2" speed="15.0" width="3.5" shape="%s"/></edges>' % shape)
    res = subprocess.run([find_netconvert(), "--node-files=" + nod, "--edge-files=" + edg, "--default.spreadtype", "center",
                          "--no-turnarounds", "--output-file=" + net], capture_output=True, text=True)
    if res.returncode != 0 or not os.path.exists(net):
        raise RuntimeError("netconvert failed: " + res.stderr[-300:])
    return net


def main():
    import carla
    from agents import map_builder, motion_runner
    with tempfile.TemporaryDirectory() as tmp:
        try:
            net = build_net(tmp)
            xodr = map_builder.net_to_xodr(net, os.path.join(tmp, "arc.xodr"))
            client = carla.Client("localhost", 2000)
            client.set_timeout(60.0)
            world = map_builder.load_xodr(client, xodr)
        except Exception as e:                      # netconvert, export or map loading
            print("CURVE FAILED: could not build or load the map: %s" % e)
            return 1
        cmap = world.get_map()
        world.apply_settings(carla.WorldSettings(synchronous_mode=True, fixed_delta_seconds=DT))
        actor = None
        try:
            wps = cmap.get_topology()
            start = min((w[0] for w in wps), key=lambda w: w.transform.location.x)
            wp = cmap.get_waypoint(start.transform.location, project_to_road=True, lane_type=carla.LaneType.Driving)
            route = [wp]
            for _ in range(130):                    # 260 m of lane centre line
                nxt = route[-1].next(2.0)
                if not nxt:
                    break
                route.append(nxt[0])
            bp = world.get_blueprint_library().find(motion_runner.VEHICLE_BLUEPRINTS["car"])
            loc = wp.transform.location
            actor = world.try_spawn_actor(bp, carla.Transform(carla.Location(loc.x, loc.y, loc.z + 0.5), wp.transform.rotation))
            if actor is None:
                print("CURVE FAILED: the test car could not be spawned on the curved road")
                return 1
            for _ in range(10):
                world.tick()
            fol = motion_runner.RouteFollower(actor, route, 30.0)
            p0 = actor.get_location()
            worst = 0.0
            for i in range(int(RUN_S / DT)):
                actor.apply_control(fol.control([], DT, world.get_snapshot().timestamp.elapsed_seconds, i * DT))
                world.tick()
                a = actor.get_location()
                worst = max(worst, min(r.transform.location.distance(a) for r in route[max(fol.i - 3, 0):fol.i + 6]))
            travelled = actor.get_location().distance(p0)
        finally:
            if actor is not None:
                actor.destroy()
            world.apply_settings(carla.WorldSettings(synchronous_mode=False, fixed_delta_seconds=0.0))
        if travelled < MIN_TRAVEL_M:
            print("CURVE FAILED: the car travelled only %.0f m (needs %.0f m)" % (travelled, MIN_TRAVEL_M))
            return 1
        if worst > MAX_DEVIATION_M:
            print("CURVE FAILED: the car left the lane centre by %.1f m (limit %.1f m)" % (worst, MAX_DEVIATION_M))
            return 1
        print("CURVE OK (travelled %.0f m, worst deviation %.2f m)" % (travelled, worst))
        return 0


if __name__ == "__main__":
    sys.exit(main())
