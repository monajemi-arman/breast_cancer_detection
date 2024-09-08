#!/usr/bin/env python
import json
import os
from PIL import Image

# YOLO to JSON Conversion
def yolo_to_coco(yolo_annotations, image_dir, output_file):
    coco_annotations = {
        "images": [],
        "annotations": [],
        "categories": []
    }

    categories = set()
    annotation_id = 1

    for label_file in os.listdir(yolo_annotations):
        if not label_file.endswith(".txt"):
            continue

        image_file = os.path.splitext(label_file)[0] + ".jpg"
        image_path = os.path.join(image_dir, image_file)

        image = Image.open(image_path)
        width, height = image.size

        image_id = len(coco_annotations["images"])
        coco_annotations["images"].append({
            "id": image_id,
            "file_name": image_file,
            "width": width,
            "height": height
        })

        with open(os.path.join(yolo_annotations, label_file)) as f:
            for line in f:
                parts = line.strip().split()
                category_id = int(parts[0])
                categories.add(category_id)
                bbox = [float(x) for x in parts[1:]]
                center_x = bbox[0] * width
                center_y = bbox[1] * height
                bbox_width = bbox[2] * width
                bbox_height = bbox[3] * height

                x_min = center_x - (bbox_width / 2)
                y_min = center_y - (bbox_height / 2)

                # Ensure no negative coordinates
                x_min = max(0, x_min)
                y_min = max(0, y_min)
                bbox_width = min(bbox_width, width - x_min)
                bbox_height = min(bbox_height, height - y_min)

                coco_annotations["annotations"].append({
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [x_min, y_min, bbox_width, bbox_height],
                    "area": bbox_width * bbox_height,
                    "iscrowd": 0
                })
                annotation_id += 1

    categories = list(categories)
    for category_id in range(len(categories)):
        coco_annotations["categories"].append({
            "id": category_id,
            "name": str(category_id)
        })

    with open(output_file, "w") as f:
        json.dump(coco_annotations, f, indent=4)

