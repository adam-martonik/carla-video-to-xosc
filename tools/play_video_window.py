"""
play_video_window.py — Interactive GUI player that opens a desktop window
playing the side-by-side comparison of 'scenar_jelen' in a continuous loop.
Controls:
  [SPACE] : Pause / Resume
  [R]     : Restart from beginning
  [Q / ESC]: Exit player
"""

import sys
import time
from pathlib import Path
import cv2

def main():
    root = Path(r"C:\Users\adamk\.gemini\antigravity\scratch\carla-video-to-xosc")
    vid_path = root / "demo" / "jelen_reality_vs_carla_sim.mp4"
    if not vid_path.exists():
        print(f"Video not found: {vid_path}")
        sys.exit(1)

    win_name = "CARLA 1.0 vs Dashcam Reality - Scenar Jelen Replay"
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win_name, 1600, 508)

    cap = cv2.VideoCapture(str(vid_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    delay_ms = int(1000.0 / fps)

    paused = False
    print(f"Playing {vid_path} in desktop window. Press 'q' to quit, SPACE to pause/resume.")

    while True:
        if not paused:
            ret, frame = cap.read()
            if not ret:
                # Loop back to beginning
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            cv2.imshow(win_name, frame)

        key = cv2.waitKey(delay_ms if not paused else 50) & 0xFF
        if key in (ord('q'), 27):  # 'q' or ESC
            break
        elif key == ord(' '):  # SPACE
            paused = not paused
        elif key == ord('r'):  # 'r' to restart
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
