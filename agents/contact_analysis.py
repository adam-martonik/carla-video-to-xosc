"""Direction of a contact between two vehicles, seen from the ego car, from CARLA's collision impulse."""
import math


def contact_direction(impulse_xyz, ego_yaw_deg, rel_xy=None):
    """'side' when the impulse is mostly lateral to the ego car, else 'front' or 'rear'.

    impulse_xyz: world-frame normal impulse (only its axis matters, not its sign).
    ego_yaw_deg: CARLA yaw of the ego car. rel_xy: (forward, right) of the other actor in the ego frame, when known.
    """
    yaw = math.radians(ego_yaw_deg)
    ix, iy = float(impulse_xyz[0]), float(impulse_xyz[1])
    lon = abs(ix * math.cos(yaw) + iy * math.sin(yaw))
    lat = abs(-ix * math.sin(yaw) + iy * math.cos(yaw))
    if lat >= lon:
        return "side"
    if rel_xy is not None and float(rel_xy[0]) < 0.0:
        return "rear"
    return "front"
