"""Direction and vehicle part/zone of a contact between two vehicles from CARLA's collision impulse."""
import math

ZONES = ("front", "rear", "left_side", "right_side", "front_left", "front_right", "rear_left", "rear_right")
ALL_ZONES = ZONES + ("side", "any")

_COMPAT = {
    "side": {"side", "left_side", "right_side", "front_left", "front_right", "rear_left", "rear_right"},
    "front": {"front", "front_left", "front_right"},
    "rear": {"rear", "rear_left", "rear_right"},
    "left_side": {"left_side", "front_left", "rear_left"},
    "right_side": {"right_side", "front_right", "rear_right"},
    "front_left": {"front_left", "left_side", "front"},
    "front_right": {"front_right", "right_side", "front"},
    "rear_left": {"rear_left", "left_side", "rear"},
    "rear_right": {"rear_right", "right_side", "rear"},
}


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


def contact_zone(impulse_xyz, yaw_deg, rel_xy=None):
    """Granular impacted zone on an actor: front, rear, left_side, right_side, or a corner (front_left, ...)."""
    yaw = math.radians(yaw_deg)
    ix, iy = float(impulse_xyz[0]), float(impulse_xyz[1])
    lon = abs(ix * math.cos(yaw) + iy * math.sin(yaw))
    lat = abs(-ix * math.sin(yaw) + iy * math.cos(yaw))
    if rel_xy is None:
        return "side" if lat >= lon else "front"
    fwd, right = float(rel_xy[0]), float(rel_xy[1])
    if lon >= 2.5 * lat:
        return "front" if fwd >= 0.0 else "rear"
    side = "right" if right >= 0.0 else "left"
    thresh = 1.2 if lat >= 2.5 * lon else 0.8
    if fwd >= thresh:
        return "front_" + side
    if fwd <= -thresh:
        return "rear_" + side
    return side + "_side"


def zones_compatible(got, want):
    """Whether a simulated contact zone satisfies a coarse or granular expected zone."""
    if not want or want == "any" or got == want:
        return True
    return got in _COMPAT.get(want, {want})


def pair_contact_zones(pair_events, actor_a, actor_b):
    """Determine impacted zones {actor_a: zone_a, actor_b: zone_b} from sorted collision events."""
    out = {}
    for k, p in ((actor_a, actor_b), (actor_b, actor_a)):
        own = next((e for e in pair_events if e.get("actor") == k and e.get("impulse_vec") is not None), None)
        if own is not None:
            out[k] = contact_zone(own["impulse_vec"], own.get("ego_yaw", 0.0), own.get("rel"))
            continue
        other = next((e for e in pair_events
                      if e.get("actor") == p and e.get("impulse_vec") is not None and e.get("other_yaw") is not None), None)
        if other is not None:
            out[k] = contact_zone(other["impulse_vec"], other["other_yaw"], other.get("other_rel"))
    return out
