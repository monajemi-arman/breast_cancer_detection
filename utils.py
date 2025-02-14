#!/usr/bin/env python
import json
import os
import cv2
import pydicom
import numpy as np
from PIL import Image


def read_dicom(file_path):
    dicom_data = pydicom.dcmread(file_path, force=True)

    # Decompress if the DICOM is compressed
    if dicom_data.file_meta.TransferSyntaxUID.is_compressed:
        dicom_data.decompress()

    pixel_array = dicom_data.pixel_array.astype(np.float32)

    # Apply to rescale slope and intercept if they exist
    rescale_slope = float(getattr(dicom_data, "RescaleSlope", 1.0))
    rescale_intercept = float(getattr(dicom_data, "RescaleIntercept", 0.0))
    hu_pixels = (pixel_array * rescale_slope) + rescale_intercept

    # If the photometric interpretation is MONOCHROME1, invert the pixel values
    if dicom_data.PhotometricInterpretation == "MONOCHROME1":
        hu_pixels = np.max(hu_pixels) - hu_pixels

    # Extract window center and window width from the DICOM itself
    if "WindowCenter" in dicom_data and "WindowWidth" in dicom_data:
        window_center = dicom_data.WindowCenter
        window_width = dicom_data.WindowWidth
        if isinstance(window_center, (list, pydicom.multival.MultiValue)):
            window_center = float(window_center[0])
        else:
            window_center = float(window_center)
        if isinstance(window_width, (list, pydicom.multival.MultiValue)):
            window_width = float(window_width[0])
        else:
            window_width = float(window_width)
    else:
        # Fallback default values (ideal for InBreast images)
        window_center = 1500
        window_width = 1000

    normalized = np.clip((hu_pixels - (window_center - window_width / 2)) / window_width, 0, 1)
    image_array = (normalized * 255).astype(np.uint8)

    image_array = normalize_contrast(image_array)

    return Image.fromarray(image_array)


def normalize_contrast(image_array):
    # Apply gamma correction to preserve subtle grayscale differences
    gamma = 0.8  # Adjust gamma value as needed
    adjusted = 255 * (image_array / 255) ** gamma
    return adjusted.astype(np.uint8)


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
