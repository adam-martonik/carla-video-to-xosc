"""
Classical (no LLM, no download) event detector for dashcam videos.

  python -m agents.video_events <video.mp4> [--sheet out.jpg]

Uses dense optical flow (OpenCV Farneback) on a downscaled, cropped view. It reports an ego-speed proxy over time, "stopped" and "brake_to_stop" phases and video "cut" moments.
Jolt/crossing detection was removed on purpose.
"""
import argparse
import json
import os

import cv2
import numpy as np

ANALYSIS_FPS = 15.0
WIDTH = 320
_VIDEO_META = {}
_FRAME_CACHE = {}
_FRAME_CACHE_MAX = 96


def _robust_z(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-6
    return (x - med) / (1.4826 * mad)


def _detect_replay(thumbs, thumb_ts):
    """Find the start of a spliced replay segment (a later frame closely matching an earlier frame after major change)."""
    for i, ti in enumerate(thumb_ts):
        if ti < 5.0:
            continue
        for j, tj in enumerate(thumb_ts[:i]):
            if ti - tj < 3.0:
                continue
            if float(np.mean(np.abs(thumbs[i] - thumbs[j]))) < 7.0:
                mid_diff = max(float(np.mean(np.abs(thumbs[k] - thumbs[j]))) for k in range(j + 1, i))
                if mid_diff > 18.0:
                    t_cut = max(ti - 0.5, tj + 2.0)
                    return t_cut, tj
    return None, None


def analyze(path, top=0.15, bottom=0.18):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(int(round(fps / ANALYSIS_FPS)), 1)
    prev, t_list, speed, dy_list, local_frac, local_cx = None, [], [], [], [], []
    cuts, prev_hist = [], None
    thumbs, thumb_ts = [], []
    idx = 0
    sample_idx = 0
    # letterbox / pillarbox: use only the area with picture content
    crop = None
    probe = []
    for k in range(5):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(k * max(n - 1, 1) / 5))
        okp, fp = cap.read()
        if okp:
            probe.append(cv2.cvtColor(fp, cv2.COLOR_BGR2GRAY).astype(np.float32))
    if probe:
        m = np.mean(probe, axis=0)
        rows = np.nonzero(m.mean(axis=1) > 12)[0]
        cols = np.nonzero(m.mean(axis=0) > 12)[0]
        if len(rows) and len(cols):
            crop = (int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % stride == 0:
            ok, frame = cap.retrieve()
            if not ok:
                break
            if crop is not None:
                frame = frame[crop[0]:crop[1], crop[2]:crop[3]]
            h, w = frame.shape[:2]
            frame = frame[int(h * top):int(h * (1 - bottom)), :]            # drop overlay text at the top and the hood
            frame = cv2.resize(frame, (WIDTH, max(int(frame.shape[0] * WIDTH / frame.shape[1]), 1)))
            g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if sample_idx % 3 == 0:
                thumbs.append(cv2.resize(g, (32, 18)).astype(np.float32))
                thumb_ts.append(idx / fps)
            sample_idx += 1
            hist = cv2.calcHist([cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)], [0, 1], None, [16, 8], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist).flatten()
            if prev_hist is not None and cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL) < 0.6:
                cuts.append(idx / fps)                                          # picture changed completely: a video cut
            prev_hist = hist
            if prev is not None:
                flow = cv2.calcOpticalFlowFarneback(prev, g, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                mag = np.hypot(flow[..., 0], flow[..., 1])
                road = mag[mag.shape[0] // 3:, :]
                strips = np.array_split(road, 4, axis=1)
                med = float(np.median([np.percentile(s, 75) for s in strips]))
                speed.append(med)
                dy_list.append(float(np.median(flow[..., 1])))
                strong = mag > (3.0 * med + 2.0)
                local_frac.append(float(strong.mean()))
                local_cx.append(float(np.nonzero(strong)[1].mean() / g.shape[1]) if strong.any() else -1.0)
                t_list.append(idx / fps)
            prev = g
        idx += 1
    cap.release()
    dur_s = round(idx / fps, 2)
    t = np.array(t_list)
    speed = np.array(speed)
    dy = np.array(dy_list)
    local = np.array(local_frac)
    if len(t) < 10:
        return {"video": path, "duration_s": dur_s, "effective_duration_s": dur_s, "events": [], "note": "video too short"}

    # smooth the speed proxy over 0.4 s
    k = max(int(0.4 * ANALYSIS_FPS), 1)
    sp_s = np.convolve(speed, np.ones(k) / k, mode="same")
    events = []

    # (the jolt finder was removed: crashes are found from the model's per-frame labels, not from camera shake)

    # 2) ego-speed phases: brake to stop / stopped
    ref = np.percentile(sp_s, 90) + 1e-6
    stopped = sp_s < 0.12 * ref
    i = 0
    while i < len(t):
        if stopped[i]:
            j = i
            while j < len(t) and stopped[j]:
                j += 1
            if t[j - 1] - t[i] > 1.0:
                events.append({"t": round(float(t[i]), 2), "type": "stopped", "until": round(float(t[j - 1]), 2),
                               "score": round(float((t[j - 1] - t[i])), 1)})
                # look back 2 s for a decrease
                a = max(0, i - int(2 * ANALYSIS_FPS))
                if sp_s[a:i + 1].max() > 0.5 * ref:
                    events.append({"t": round(float(t[a] if a < i else t[i]), 2), "type": "brake_to_stop",
                                   "score": round(float(sp_s[a:i + 1].max() / ref), 2)})
            i = j
        else:
            i += 1

    for c in cuts:
        events.append({"t": round(float(c), 2), "type": "cut", "score": 1.0})
    t_replay_cut, t_replay_src = _detect_replay(thumbs, thumb_ts)
    eff_dur = dur_s
    if t_replay_cut is not None:
        eff_dur = round(float(t_replay_cut), 2)
        events.append({"t": eff_dur, "type": "replay", "of_t": round(float(t_replay_src), 2), "score": 1.0})
    events.sort(key=lambda e: e["t"])
    return {"video": path, "duration_s": dur_s, "effective_duration_s": eff_dur, "fps": fps, "analysis_samples": int(len(t)),
            "speed_proxy": [round(float(v), 2) for v in sp_s[::max(int(ANALYSIS_FPS), 1)]],
            "events": events}


def sheet(path, center_t, out, span=1.2, n=6):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    tiles = []
    for k in range(n):
        tt = max(center_t - span + 2 * span * k / (n - 1), 0)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(tt * fps))
        ok, fr = cap.read()
        if ok:
            h, w = fr.shape[:2]
            fr = cv2.resize(fr, (400, int(400 * h / w)))
            cv2.putText(fr, f"{tt:.1f}s", (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            tiles.append(fr)
    if not tiles:
        cap.release()
        return
    cap.release()
    hh = max(t.shape[0] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, hh - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
    while len(tiles) < 6:
        tiles.append(np.zeros_like(tiles[0]))
    cv2.imwrite(out, np.vstack([np.hstack(tiles[:3]), np.hstack(tiles[3:6])]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("video")
    ap.add_argument("--sheet", help="write a contact sheet of frames around the strongest jolt")
    a = ap.parse_args()
    res = analyze(a.video)
    print(json.dumps(res, indent=1))
    if a.sheet:
        jolts = [e for e in res["events"] if e["type"] == "jolt"]
        if jolts:
            sheet(a.video, max(jolts, key=lambda e: e["score"])["t"], a.sheet)


def _get_video_meta(path, mtime=None, cap=None):
    """Cache (fps, n, letterbox crop) per video path and mtime to avoid re-probing 5 frames per call."""
    if mtime is None:
        mtime = os.path.getmtime(path) if os.path.exists(path) else None
    key = (path, mtime)
    if key in _VIDEO_META:
        return _VIDEO_META[key]
    own_cap = cap is None
    if own_cap:
        cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        probe = []
        for k in range(5):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(k * max(n - 1, 1) / 5))
            ok, fp = cap.read()
            if ok:
                probe.append(cv2.cvtColor(fp, cv2.COLOR_BGR2GRAY).astype(np.float32))
        crop = None
        if probe:
            m = np.mean(probe, axis=0)
            rows = np.nonzero(m.mean(axis=1) > 12)[0]
            cols = np.nonzero(m.mean(axis=0) > 12)[0]
            if len(rows) and len(cols):
                crop = (int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1)
        meta = (fps, n, crop)
        _VIDEO_META[key] = meta
        return meta
    finally:
        if own_cap:
            cap.release()


def grab_frames(path, times, width=512, crop_bottom=0.0):
    """JPEG bytes of the frames at the given times (seconds), black bars cropped, resized to `width`."""
    mtime = os.path.getmtime(path) if os.path.exists(path) else None
    keys = [(path, mtime, round(float(t), 3), int(width), round(float(crop_bottom), 2)) for t in times]
    if all(k in _FRAME_CACHE for k in keys):
        return [_FRAME_CACHE[k] for k in keys]
    cap = cv2.VideoCapture(path)
    try:
        fps, _, crop = _get_video_meta(path, mtime=mtime, cap=cap)
        out = []
        for t, key in zip(times, keys):
            if key in _FRAME_CACHE:
                out.append(_FRAME_CACHE[key])
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(max(t, 0) * fps))
            ok, fr = cap.read()
            if not ok:
                continue
            if crop is not None:
                fr = fr[crop[0]:crop[1], crop[2]:crop[3]]
            if crop_bottom > 0.0:
                fr = fr[:max(int(fr.shape[0] * (1.0 - crop_bottom)), 1), :]
            h, w = fr.shape[:2]
            fr = cv2.resize(fr, (width, max(int(h * width / w), 1)))
            jpg = cv2.imencode(".jpg", fr, [cv2.IMWRITE_JPEG_QUALITY, 88])[1].tobytes()
            if len(_FRAME_CACHE) >= _FRAME_CACHE_MAX:
                _FRAME_CACHE.pop(next(iter(_FRAME_CACHE)))
            _FRAME_CACHE[key] = jpg
            out.append(jpg)
        return out
    finally:
        cap.release()


def lateral_entry_flow(path, t_crash, span=2.0):
    """Measure (left_to_right, right_to_left) relative horizontal optical flow before t_crash."""
    step = 0.15
    n = int(span / step) + 1
    times = sorted({round(max(float(t_crash) - span - step + step * i, 0.0), 2) for i in range(n)})
    raws = grab_frames(path, times, width=320)
    gs = []
    for raw in raws:
        arr = np.frombuffer(raw, np.uint8)
        im = cv2.imdecode(arr, cv2.IMREAD_COLOR) if arr.size else None
        if im is not None:
            gs.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
    if len(gs) < 2:
        return 0.0, 0.0
    left_to_right, right_to_left = 0.0, 0.0
    for g0, g1 in zip(gs, gs[1:]):
        h, w = g0.shape
        r0 = g0[int(h * 0.25):int(h * 0.78), :]
        r1 = g1[int(h * 0.25):int(h * 0.78), :]
        if r0.shape != r1.shape or r0.size == 0:
            continue
        flow = cv2.calcOpticalFlowFarneback(r0, r1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        fx = flow[..., 0]
        rel_fx = fx - float(np.median(fx))
        left_to_right += float(np.mean(np.maximum(rel_fx[:, :w // 2], 0.0) ** 2))
        right_to_left += float(np.mean(np.maximum(-rel_fx[:, w // 2:], 0.0) ** 2))
    return left_to_right, right_to_left


def active_row_bounds(gray):
    """Return (y0, y1) of non-letterbox active rows (handles vertical/portrait videos with black bars)."""
    rs = np.where((gray.mean(axis=1) > 15) & (gray.std(axis=1) > 8))[0]
    return (int(rs[0]), int(rs[-1]) + 1) if rs.size > 20 else (0, gray.shape[0])


def ego_motion_metrics(path, t_crash, duration):
    """Background-stable outer-edge and upper-background optical flow over pre-crash and impact windows."""
    tc = float(t_crash)
    dur = max(float(duration), 0.2)

    def _decode(ts):
        raws = grab_frames(path, ts, width=320)
        gs = []
        for raw in raws:
            arr = np.frombuffer(raw, np.uint8)
            im = cv2.imdecode(arr, cv2.IMREAD_COLOR) if arr.size else None
            if im is not None:
                gs.append(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY))
        return gs

    def _measure(ts):
        gs = _decode(ts)
        if len(gs) < 2:
            return None
        y0, y1 = active_row_bounds(gs[len(gs) // 2])
        edge_min, up_med, up_pan = [], [], []
        for g0, g1 in zip(gs, gs[1:]):
            a0, a1 = g0[y0:y1, :], g1[y0:y1, :]
            h, w = a0.shape
            if a0.shape != a1.shape or h < 16 or w < 16:
                continue
            flow = cv2.calcOpticalFlowFarneback(a0, a1, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.hypot(flow[..., 0], flow[..., 1])
            l_edge = mag[int(h * 0.50):int(h * 0.92), :int(w * 0.22)]
            r_edge = mag[int(h * 0.50):int(h * 0.92), int(w * 0.78):]
            edge_min.append(min(float(np.percentile(l_edge, 75)), float(np.percentile(r_edge, 75))))
            up = flow[int(h * 0.10):int(h * 0.45), :]
            up_med.append(float(np.median(np.hypot(up[..., 0], up[..., 1]))))
            up_pan.append(abs(float(np.median(up[..., 0]))))
        if not edge_min:
            return None
        return float(np.median(edge_min)), max(up_med), max(up_pan)

    pre_ts = sorted({round(max(tc - 3.6 + 0.4 * i, 0.0), 2) for i in range(9)})
    imp_ts = sorted({round(min(max(tc - 0.4 + 0.2 * i, 0.0), dur - 0.05), 2) for i in range(9)})
    pre = _measure(pre_ts)
    imp = _measure(imp_ts)
    if pre is None or imp is None:
        return {"pre_edge": 0.0, "imp_edge": 0.0, "imp_bg_max": 0.0, "imp_pan_max": 0.0,
                "ego_stopped": False, "camera_steady": False}
    pre_e, _, _ = pre
    imp_e, imp_bg, imp_pan = imp
    return {
        "pre_edge": round(pre_e, 3),
        "imp_edge": round(imp_e, 3),
        "imp_bg_max": round(imp_bg, 3),
        "imp_pan_max": round(imp_pan, 3),
        "ego_stopped": bool(pre_e < 0.35),
        "camera_steady": bool(pre_e < 0.35 and imp_e < 0.35 and imp_bg < 0.35),
    }
