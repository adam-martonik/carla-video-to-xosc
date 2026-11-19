import carla, time
from agents.navigation.basic_agent import BasicAgent

client = carla.Client("127.0.0.1", 2000)
client.set_timeout(30.0)

print("[EgoDriver] Waiting for 'hero' vehicle to spawn...")
hero = None
for _ in range(120):
    try:
        world = client.get_world()
        for v in world.get_actors().filter("vehicle.*"):
            if v.attributes.get("role_name") == "hero":
                hero = v
                break
        if hero is not None:
            break
    except Exception:
        pass
    time.sleep(0.5)

if hero is None:
    print("[EgoDriver] Could not find hero vehicle!")
else:
    print(f"[EgoDriver] Found hero id={hero.id} ({hero.type_id}) at {hero.get_location()}")
    world = client.get_world()
    wmap = world.get_map()
    agent = BasicAgent(hero, target_speed=35)
    wp = wmap.get_waypoint(hero.get_location())
    # Pick a waypoint 250m ahead along the lane
    for _ in range(50):
        next_wps = wp.next(5.0)
        if next_wps:
            wp = next_wps[0]
    agent.set_destination(wp.transform.location)
    while hero.is_alive:
        try:
            ctrl = agent.run_step()
            hero.apply_control(ctrl)
            time.sleep(0.03)
        except Exception:
            break
    print("[EgoDriver] Hero finished.")
