"""Draw the saved boxes on a few frames of a run:  python check_overlay.py ../dataset_test/run_0000 [n]"""
import json
import os
import sys

import cv2

run = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 4
frames = json.load(open(os.path.join(run, 'labels.json')))['frames']
step = max(len(frames) // n, 1)
os.makedirs(os.path.join(run, 'overlay'), exist_ok=True)
for fr in frames[::step][:n]:
    img = cv2.imread(os.path.join(run, fr['image']))
    for b in fr['boxes']:
        x1, y1, x2, y2 = map(int, b['bbox'])
        col = (0, 0, 255) if b['occluded'] else (0, 255, 0)
        cv2.rectangle(img, (x1, y1), (x2, y2), col, 2)
        cv2.putText(img, f"{b['cls']} {b['distance']}m", (x1, max(y1 - 5, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1)
    out = os.path.join(run, 'overlay', f"{fr['tick']:05d}.jpg")
    cv2.imwrite(out, img)
    print(out, len(fr['boxes']), 'boxes')
