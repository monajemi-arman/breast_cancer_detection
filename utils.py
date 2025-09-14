#!/usr/bin/env python
import json
import os
import cv2
import pydicom
import numpy as np
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut


def read_dicom(file_path):
    dicom_data = pydicom.dcmread(file_path, force=True)

    # Decompress if the DICOM is compressed
    if dicom_data.file_meta.TransferSyntaxUID.is_compressed:
        dicom_data.decompress()

    img_array = apply_voi_lut(dicom_data.pixel_array, dicom_data)

    # Invert image if PhotometricInterpretation is MONOCHROME1
    if 'PhotometricInterpretation' in dicom_data and dicom_data.PhotometricInterpretation == "MONOCHROME1":
        img_array = np.max(img_array) - img_array

    # Normalize the image array to 0-255 and convert to uint8 for PIL Image
    min_val = np.min(img_array)
    max_val = np.max(img_array)

    if max_val == min_val:
        normalized_img = np.zeros_like(img_array, dtype=np.uint8)
    else:
        normalized_img = ((img_array - min_val) / (max_val - min_val) * 255).astype(np.uint8)

    gamma = 0.8
    adjusted_img = 255 * (normalized_img / 255.0) ** gamma
    adjusted_img = np.clip(adjusted_img, 0, 255).astype(np.uint8)

    return Image.fromarray(adjusted_img, mode='L')


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


def coco_to_yolo(coco_annotation_file, output_dir, image_dir):
    # Load COCO annotations
    with open(coco_annotation_file) as f:
        coco_data = json.load(f)

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Iterate over images
    for image in coco_data['images']:
        image_id = image['id']
        file_name = image['file_name']
        image_path = os.path.join(image_dir, file_name)

        # Load image
        img = cv2.imread(image_path)

        # Create YOLO annotation file
        yolo_file = os.path.join(output_dir, file_name.replace('.jpg', '.txt'))
        with open(yolo_file, 'w') as f:
            for annotation in coco_data['annotations']:
                if annotation['image_id'] == image_id:
                    x, y, w, h = annotation['bbox']
                    class_id = annotation['category_id']

                    # Convert COCO bounding box format to YOLO format
                    center_x = (x + w / 2) / img.shape[1]
                    center_y = (y + h / 2) / img.shape[0]
                    relative_width = w / img.shape[1]
                    relative_height = h / img.shape[0]

                    # Write YOLO annotation to file
                    f.write(f"{class_id} {center_x} {center_y} {relative_width} {relative_height}\n")
