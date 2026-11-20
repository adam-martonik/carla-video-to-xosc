"""
Export a generated dataset.
  python export_coco.py ../dataset_test            -> <dir>/coco.json  and <dir>/trajectories.csv

coco.json: images from rgb/, boxes for pedestrian (id 1) / vehicle (id 2); boxes flagged occluded are
dropped unless --keep-occluded.  trajectories.csv: one row per saved frame with ego/pedestrian state,
ego control, distance, TTC and the run outcome (for prediction / imitation experiments).
"""
import argparse
import csv
import json
import os

CATS = {'pedestrian': 1, 'vehicle': 2}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('--keep-occluded', action='store_true')
    ap.add_argument('--min-area', type=float, default=100.0)
    args = ap.parse_args()

    images, anns = [], []
    rows = []
    for run in sorted(d for d in os.listdir(args.root) if d.startswith('run_')):
        rd = os.path.join(args.root, run)
        if not os.path.exists(os.path.join(rd, 'meta.json')):
            continue
        meta = json.load(open(os.path.join(rd, 'meta.json')))
        if not meta.get('complete'):
            continue
        cam = meta['camera']
        for fr in json.load(open(os.path.join(rd, 'labels.json')))['frames']:
            img_id = len(images) + 1
            images.append({'id': img_id, 'file_name': f'{run}/{fr["image"]}',
                           'width': cam['width'], 'height': cam['height'], 'run': run, 'tick': fr['tick']})
            for b in fr['boxes']:
                if b['occluded'] and not args.keep_occluded:
                    continue
                x1, y1, x2, y2 = b['bbox']
                w, h = x2 - x1, y2 - y1
                if w * h < args.min_area:
                    continue
                anns.append({'id': len(anns) + 1, 'image_id': img_id, 'category_id': CATS[b['cls']],
                             'bbox': [x1, y1, w, h], 'area': w * h, 'iscrowd': 0,
                             'distance': b['distance']})
            e = fr['ego']
            rows.append([run, fr['t'], meta['outcome'], *e['loc'], e['yaw'], e['speed'],
                         e['control']['throttle'], e['control']['steer'], e['control']['brake'],
                         *fr['ped']['loc'], fr['ped']['speed'], fr['dist_to_ped'], fr['ttc']])

    coco = {'images': images, 'annotations': anns,
            'categories': [{'id': v, 'name': k} for k, v in CATS.items()]}
    with open(os.path.join(args.root, 'coco.json'), 'w') as f:
        json.dump(coco, f)
    with open(os.path.join(args.root, 'trajectories.csv'), 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['run', 't', 'outcome', 'ego_x', 'ego_y', 'ego_z', 'ego_yaw', 'ego_speed', 'throttle',
                    'steer', 'brake', 'ped_x', 'ped_y', 'ped_z', 'ped_speed', 'dist_to_ped', 'ttc'])
        w.writerows(rows)
    print(f'{len(images)} images, {len(anns)} boxes, {len(rows)} trajectory rows')


if __name__ == '__main__':
    main()
