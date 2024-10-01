#!/usr/bin/env python3

import os
from glob import glob
from pathlib import Path

data_yaml = 'dataset.yaml'
data_dirs = ['train', 'val', 'test']

def generate_keypoints(data_dir):
    labels = glob(os.path.join(data_dir, "labels", "*.txt"), recursive=True)

    root_dir = Path(labels[0]).parent.parent.parent
    for label in labels:
        label = Path(label)
        split = label.parent.parent.name
        img_path = [(root_dir / split / "images" / label.with_suffix(sfx).name)
                    for sfx in [".png", ".jpg", ".PNG", ".JPG"]]
        img_path = [pth for pth in img_path if pth.exists()][0]
        boxes = []
        points = []
        classes = []
        save_pth = root_dir / split / "labels_kp" / label.name
        save_pth.parent.mkdir(exist_ok=True)
        with open(label) as f:
            lines = f.readlines()
            for line in lines:
                splits = line.rstrip().split(" ")
                cls_id = int(splits[0])
                box = splits[1:]
                if not box:
                    with open(save_pth, "w") as f:
                        pass
                    continue

                box = [float(pt) for pt in box]
                point = (box[0], box[1])
                points.append(point)
                boxes.append(box)
                classes.append(cls_id)

        with open(save_pth, "w") as f:
            for point, box, cls_id in zip(points, boxes, classes):
                f.writelines(f"{cls_id} {box[0]} {box[1]} {box[2]} {box[3]} {point[0]} {point[1]} 1 \n")

def add_to_data_yaml(data_yaml):
    with open(data_yaml, 'a') as f:
        f.write('kpt_shape: [1, 3]')

def main():
    # Generate
    for data_dir in data_dirs:
        generate_keypoints(data_dir)
    # Add to data yaml
    add_to_data_yaml(data_yaml)

if __name__ == '__main__':
    main()