"""
Checker and retry search: runs a scenario in CARLA, compares the result with the plan's expectation and, when a rule fails,
runs it again with changed controller parameters (never by hand, never by moving the spawn points).

expect = {"type": "collision" | "near_miss" | "none", "actors": ["v1", "vN"], "time_s": T, "max_dist_m": d,
          "contact": "side" | "front" | "rear" | "any", "window_s": 1.5}
Result events (agents/motion_runner.py): t_sim, actor, other_actor, other (type id), impulse_vec, ego_yaw, rel.
"""
import copy
import json
import math
import os
import time

from agents.contact_analysis import contact_direction, pair_contact_zones, zones_compatible

# deltas applied to the event actor's controller; the first run uses the plan as built
# "offset_m" is used only for event actors without a closed-loop controller (crossing, ahead): it moves their spawn point along the lane
VARIANTS = [{}, {"lead_s": 0.5, "offset_m": 3.0}, {"lead_s": -0.5, "offset_m": -3.0}, {"aim_offset_m": 0.4, "offset_m": 6.0}]
NONE_VARIANTS = [{}, {"avoid": True}]
MAX_ITER = 4


def _load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _dump(obj, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1)


def _is_actor_hit(e):
    return str(e.get("other", "")).startswith(("vehicle", "walker"))


def evaluate(expect, result):
    """Returns (ok, detail). detail["reasons"] lists every rule that failed."""
    typ = expect.get("type", "none")
    actors = expect.get("actors", ["v1"])
    events = result.get("events", [])
    window = float(expect.get("window_s", 1.5))
    detail = {"outcome": result.get("outcome"), "reasons": []}
    reasons = detail["reasons"]
    if typ == "none":
        watch = set(actors) | {"v1"}
        hits = [e for e in events if e.get("actor") in watch and _is_actor_hit(e)]
        detail["ego_collisions"] = len(hits)
        if hits:
            reasons.append("a watched vehicle (%s) touched another road user although none was expected" % hits[0].get("actor"))
        for e in events:
            if e.get("actor") in watch and not _is_actor_hit(e):
                if e.get("other") == "static.road":
                    iv = e.get("impulse_vec") or [0.0, 0.0, 0.0]
                    if len(iv) >= 3 and abs(iv[2]) >= math.hypot(iv[0], iv[1]):
                        continue
                reasons.append("scenery: %s hit scenery (%s) at %.1f s" % (e.get("actor"), e.get("other"), e.get("t_sim", 0.0)))
                break
        return not reasons, detail
    a, b = actors[0], actors[1]
    pair = [e for e in events if {e.get("actor"), e.get("other_actor")} == {a, b}]
    d = result.get("min_pair_dist_m", {}).get("-".join(sorted((a, b))))
    detail["pair_collision"] = bool(pair)
    detail["min_dist_m"] = d
    t_first = None
    if pair:
        pair.sort(key=lambda e: e.get("t_sim", 0.0))
        t_first = pair[0].get("t_sim")
    if typ == "collision":
        if not pair:
            reasons.append("no contact between %s and %s" % (a, b))
        else:
            detail["contact_t_sim"] = t_first
            t_exp = expect.get("time_s")
            if t_first is not None and t_exp is not None and abs(t_first - float(t_exp)) > window:
                reasons.append("time: contact at %.1f s, expected %.1f s +-%.1f" % (t_first, float(t_exp), window))
            zones = pair_contact_zones(pair, a, b)
            if zones:
                detail["contact_parts"] = zones
            want = expect.get("contact", "any")
            if want != "any":
                own = next((e for e in pair if e.get("actor") == a), pair[0])
                if own.get("impulse_vec") is None:
                    detail["direction_unchecked"] = True
                else:
                    got = contact_direction(own["impulse_vec"], own.get("ego_yaw", 0.0), own.get("rel"))
                    detail["direction"] = got
                    if got != want:
                        reasons.append("direction: %s contact, expected %s" % (got, want))
            want_parts = expect.get("contact_parts")
            if isinstance(want_parts, dict):
                for aid, want_part in want_parts.items():
                    got_part = zones.get(aid)
                    if got_part is not None and not zones_compatible(got_part, want_part):
                        reasons.append("contact part: %s hit on %s, expected %s" % (aid, got_part, want_part))
    elif typ == "miss":
        if pair:
            reasons.append("the pair touched in a scene that expects a miss (no collision)")
        t_first = float(expect.get("time_s", 0.0))
    else:   # near_miss
        thr = float(expect.get("max_dist_m", 6.0))
        if pair:
            reasons.append("the pair touched in a scene that expects a near miss")
        elif d is None or d > thr:
            reasons.append("no near miss: closest approach %s m, limit %.1f m" % (d, thr))
        t_first = t_first if t_first is not None else float(expect.get("time_s", 0.0))
    # nothing else may be hit by the pair (or by the ego car) until 1.5 s after the event
    watch = set(actors) | {"v1"}
    t_ref = t_first if t_first is not None else float("inf")
    for e in events:
        if e.get("actor") in watch and e.get("other_actor") not in (a, b) and e.get("t_sim", 0.0) <= t_ref + 1.5:
            if e.get("other") == "static.road":
                iv = e.get("impulse_vec") or [0.0, 0.0, 0.0]
                if (t_first is not None and e.get("t_sim", 0.0) >= t_first) or (len(iv) >= 3 and abs(iv[2]) >= math.hypot(iv[0], iv[1])):
                    continue
            kind = "scenery" if not _is_actor_hit(e) else "another road user"
            reasons.append("scenery: %s hit %s (%s) at %.1f s" % (e.get("actor"), kind, e.get("other"), e.get("t_sim", 0.0)))
            break
    return not reasons, detail


def _apply_variant(plan, expect, variant):
    if not variant:
        return
    if "avoid" in variant:
        plan["agents"]["v1"]["avoid"] = True
        return
    target = expect["actors"][1] if len(expect.get("actors", [])) > 1 else None
    spec = plan["agents"].get(target) if target else None
    if not spec:
        return
    icp = spec.get("intercept")
    if icp is not None:
        if "lead_s" in variant:
            icp["lead_s"] = round(float(icp.get("lead_s", 1.4)) + variant["lead_s"], 3)
        if "aim_offset_m" in variant:
            icp["aim_offset_m"] = round(float(icp.get("aim_offset_m", 0.0)) + variant["aim_offset_m"], 3)
    start = spec.get("start")
    if start and start.get("type") == "ttc" and "lead_s" in variant:
        start["seconds"] = round(float(start.get("seconds", 2.0)) + variant["lead_s"], 3)
    if icp is None and "offset_m" in variant:
        if spec.get("mode") == "walk":
            t0 = (spec.get("start") or {}).get("t", 0.0)
            spec["start"] = {"type": "time", "t": round(max(t0 + variant["offset_m"] / 8.0, 0.0), 2)}
        elif spec.get("mode") in ("drive", "static"):
            spec["offset_m"] = float(variant["offset_m"])


MAX_REPLAY_DROPS = 4      # reruns per variant that take out replayed vehicles in the way of the crash


def replay_in_the_way(plan, expect, result):
    """Replayed vehicles (plan agents with "replay") that the ego car or a crash vehicle hit before the expected event: they
    come from the video, but the crash comes first. [vehicle ids]."""
    watch = set(expect.get("actors", [])) | {"v1"}
    t_ref = float(expect.get("time_s", 1e9)) + float(expect.get("window_s", 1.5))
    out = []
    for e in result.get("events", []):
        o = e.get("other_actor")
        if e.get("actor") in watch and o not in watch and (plan["agents"].get(o) or {}).get("replay") and e.get("t_sim", 0.0) <= t_ref:
            if o not in out:
                out.append(o)
    return out


def search(out_dir, scene_id, run_fn, max_iter=MAX_ITER, log=print):
    """run_fn(load_map: bool) runs the scenario currently stored in <scene>_motion.json and returns the result dict."""
    plan_fn = os.path.join(out_dir, scene_id + "_motion.json")
    base = _load(plan_fn)
    _dump(base, os.path.join(out_dir, scene_id + "_motion_base.json"))
    expect = base.get("expect", {"type": "none", "actors": ["v1"]})
    variants = NONE_VARIANTS if expect.get("type") in ("none", "miss") else VARIANTS
    history, best, it = [], None, 0
    dropped = []                                   # replayed vehicles taken out because they were in the way of the crash
    for variant in variants[:max_iter]:
        for _ in range(MAX_REPLAY_DROPS + 1):
            plan = copy.deepcopy(base)
            _apply_variant(plan, expect, variant)
            for vid in dropped:
                plan["agents"][vid] = {"mode": "skip", "replay": plan["agents"][vid].get("replay")}
            _dump(plan, plan_fn)
            t0 = time.time()
            result = run_fn(load_map=(it == 0))
            ok, detail = evaluate(expect, result)
            it += 1
            history.append({"iteration": it, "variant": variant, "ok": ok, "detail": detail, "dropped_replay": list(dropped),
                            "seconds": round(time.time() - t0, 1)})
            log("iteration %d variant=%s ok=%s %s" % (it, variant, ok, detail))
            new = [v for v in replay_in_the_way(plan, expect, result) if v not in dropped] if not ok else []
            if not new:
                break
            dropped += new
            log("replayed vehicles in the way of the crash, taken out: %s" % ", ".join(
                "%s (%s)" % (v, plan["agents"][v].get("replay")) for v in new))
        score = (0 if ok else 1, len(detail["reasons"]), detail.get("min_dist_m") if detail.get("min_dist_m") is not None else 1e9)
        if best is None or score < best[0]:
            best = (score, plan, result, detail)
        if ok:
            break
    status = "ready" if history and history[-1]["ok"] else "best_effort"
    final_plan = best[1] if best else base
    _dump(final_plan, plan_fn)
    out = {"status": status, "expect": expect, "iterations": it, "history": history,
           "best": {"detail": best[3] if best else None}}
    if best and isinstance(best[2], dict):
        for k in ("dashcam_actors", "dashcam_videos", "video_path", "sensor_preset", "sensors_active",
                  "sensor_actor", "sensor_video_path", "sensor_data_dir"):
            if best[2].get(k):
                out[k] = best[2][k]
    _dump(out, os.path.join(out_dir, scene_id + "_ready.json"))
    return out
