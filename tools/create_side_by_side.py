"""
create_side_by_side.py — Creates a synchronized side-by-side video comparing
the real dashcam tracking with the CARLA 1.0 3D OpenSCENARIO replay for 'scenar_jelen'.
"""

import sys
from pathlib import Path
import cv2
import numpy as np

def main():
    root = Path(r"C:\Users\adamk\.gemini\antigravity\scratch\carla-video-to-xosc")
    real_path = root / "demo" / "jelen_tracked_annotated.mp4"
    carla_path = root / "demo" / "carla_replay_dashcam.mp4"
    out_path = root / "demo" / "jelen_reality_vs_carla_sim.mp4"

    if not real_path.exists() or not carla_path.exists():
        print("Missing input videos!")
        sys.exit(1)

    cap_real = cv2.VideoCapture(str(real_path))
    cap_carla = cv2.VideoCapture(str(carla_path))

    fps_real = cap_real.get(cv2.CAP_PROP_FPS) or 30.0
    fps_carla = cap_carla.get(cv2.CAP_PROP_FPS) or 20.0
    out_fps = 20.0

    target_w = 960
    target_h = 540
    header_h = 70
    canvas_w = target_w * 2
    canvas_h = target_h + header_h

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, out_fps, (canvas_w, canvas_h))

    frame_idx = 0
    dt = 1.0 / out_fps

    while True:
        sim_time = frame_idx * dt

        # Set position in real video
        real_frame_no = int(sim_time * fps_real)
        cap_real.set(cv2.CAP_PROP_POS_FRAMES, real_frame_no)
        ret_r, frame_r = cap_real.read()

        # Read carla video frame
        ret_c, frame_c = cap_carla.read()

        if not ret_r and not ret_c:
            break

        if not ret_r:
            frame_r = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        else:
            frame_r = cv2.resize(frame_r, (target_w, target_h))

        if not ret_c:
            frame_c = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        else:
            frame_c = cv2.resize(frame_c, (target_w, target_h))

        # Create combined canvas
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Header background
        canvas[0:header_h, 0:canvas_w] = (25, 25, 30)
        cv2.putText(
            canvas,
            "REAL DASHCAM RECORDING (jelen.mp4)",
            (40, 42),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9,
            (0, 220, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "CARLA 1.0 3D SIMULATION (scenar_jelen.xosc)",
            (target_w + 40, 42),
            cv2.FONT_HERSHEY_DUPLEX,
            0.9,
            (50, 255, 120),
            2,
            cv2.LINE_AA,
        )

        time_str = f"Time: {sim_time:4.1f}s"
        cv2.putText(
            canvas,
            time_str,
            (target_w - 90, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (200, 200, 200),
            2,
            cv2.LINE_AA,
        )

        # Place video frames
        canvas[header_h:canvas_h, 0:target_w] = frame_r
        canvas[header_h:canvas_h, target_w:canvas_w] = frame_c

        # Dividers
        cv2.line(canvas, (target_w, 0), (target_w, canvas_h), (80, 80, 90), 2)
        cv2.line(canvas, (0, header_h), (canvas_w, header_h), (80, 80, 90), 2)

        writer.write(canvas)
        frame_idx += 1

    cap_real.release()
    cap_carla.release()
    writer.release()
    print(f"Generated {frame_idx} side-by-side frames -> {out_path}")

if __name__ == "__main__":
    main()
