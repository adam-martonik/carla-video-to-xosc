"""Integration of RoadFrame + CutInController + RouteFollower steering with a stub 'carla' module and a kinematic bicycle.
It cannot replace a CARLA run, but it catches sign errors between the road frame (y north, d to the left) and CARLA (y south)."""
import math
import sys
import types
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

DT = 0.05


class Loc:
    def __init__(self, x=0.0, y=0.0, z=0.0):
        self.x, self.y, self.z = x, y, z

    def distance(self, o):
        return math.sqrt((self.x - o.x) ** 2 + (self.y - o.y) ** 2 + (self.z - o.z) ** 2)


class Vec:
    def __init__(self, x, y):
        self.x, self.y = x, y

    def length(self):
        return math.hypot(self.x, self.y)


class Ctl:
    def __init__(self, throttle=0.0, steer=0.0, brake=0.0, hand_brake=False):
        self.throttle, self.steer, self.brake = throttle, steer, brake


class FakeCar:
    def __init__(self, x, y, yaw, v):
        self.x, self.y, self.yaw, self.v, self.ctl = x, y, yaw, v, Ctl()
        self.wheelbase = 2.8
        self.max_steer = math.radians(70.0)

    def get_location(self):
        return Loc(self.x, self.y)

    def get_transform(self):
        car = self
        return types.SimpleNamespace(rotation=types.SimpleNamespace(yaw=car.yaw))

    def get_velocity(self):
        return Vec(self.v * math.cos(math.radians(self.yaw)), self.v * math.sin(math.radians(self.yaw)))

    def get_physics_control(self):
        return types.SimpleNamespace(wheels=[types.SimpleNamespace(max_steer_angle=70.0)])

    def apply_control(self, c):
        self.ctl = c

    def step(self):
        a = 5.0 * self.ctl.throttle - 8.0 * self.ctl.brake - 0.1
        self.v = max(self.v + a * DT, 0.0)
        steer = self.ctl.steer * self.max_steer
        self.yaw += math.degrees(self.v / self.wheelbase * math.tan(steer) * DT)
        self.x += self.v * math.cos(math.radians(self.yaw)) * DT
        self.y += self.v * math.sin(math.radians(self.yaw)) * DT


class TestInterceptFollower(unittest.TestCase):
    def setUp(self):
        self._saved = sys.modules.get("carla")
        sys.modules["carla"] = types.SimpleNamespace(Location=Loc, VehicleControl=Ctl)

    def tearDown(self):
        if self._saved is None:
            sys.modules.pop("carla", None)
        else:
            sys.modules["carla"] = self._saved

    def test_cutter_steers_into_the_ego_lane_and_stops_steering_after_contact(self):
        from agents import motion_runner
        from agents.intercept import CutInConfig, CutInController
        from agents.road_frame import RoadFrame, lane_offset

        frame = RoadFrame.straight(300.0)
        d_src, d_dst = lane_offset(2, 1), lane_offset(2, 0)            # net d: left lane -1.75, ego lane -5.25
        ego = FakeCar(15.0, -d_dst, 0.0, 17.4)                         # CARLA y = -net y
        cutter = FakeCar(16.0, -d_src, 0.0, 17.4)
        route = []
        s = 16.0
        while s < 298.0:
            x, y = frame.to_xy(s, d_src)
            route.append(motion_runner._RoutePoint(Loc(x, -y)))
            s += 2.0
        cfg = CutInConfig(lane_src_d=d_src, lane_dst_d=d_dst, t_event_s=7.0)
        fol = motion_runner.InterceptFollower(cutter, route, 62.8, ego, frame, CutInController(cfg))
        t, t_touch, y_at_touch, ys = 0.0, None, None, []
        while t < 12.0:
            cutter.apply_control(fol.control([], DT, t, t))
            ego.step_const = None
            ego.x += ego.v * DT
            cutter.step()
            t += DT
            ys.append(cutter.y)
            if t_touch is None and abs(cutter.y - ego.y) < 2.0 and abs(cutter.x - ego.x) < 4.5:
                t_touch, y_at_touch = t, cutter.y
                fol.crash_time = t
        self.assertIsNotNone(t_touch, "the cutter never reached the ego car")
        self.assertTrue(5.6 <= t_touch <= 8.6, t_touch)
        # moving toward the ego lane means a growing CARLA y (net d goes from -1.75 to -5.25)
        self.assertGreater(y_at_touch, d_src * -1 + 0.5)
        # after contact the cutter does not keep pushing toward the ego lane
        self.assertLess(max(ys[int((t_touch + 1.5) / DT):]) - y_at_touch, 1.0)


if __name__ == "__main__":
    unittest.main()
