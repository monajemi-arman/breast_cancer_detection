import os
import json
import cv2
import re
import albumentations as A
from tqdm import tqdm
import argparse
from pathlib import Path
from utils import coco_to_yolo

# How many times multiply number of images
n_times = 3
# YOLO labels directory name
labels_dir_name = 'labels'

# Define augmentation pipeline
transform = A.Compose([
    A.Rotate(limit=5, p=1.0),
    A.CLAHE(p=0.5),
    A.HorizontalFlip(p=0.5),
], bbox_params=A.BboxParams(format='coco', label_fields=['category_ids']))

# Argument parser setup
def parse_args():
    parser = argparse.ArgumentParser(
        description="Offline augmentation for COCO-style dataset with optional regex-based image filtering.")
    parser.add_argument('--input_image_folder', type=str, required=True, help='Path to input image folder.')
    parser.add_argument('--input_annotation_file', type=str, required=True,
                        help='Path to input COCO-style annotation JSON.')
    parser.add_argument('--output_image_folder', type=str, required=True, help='Path to save augmented images.')
    parser.add_argument('--output_annotation_file', type=str, required=True,
                        help='Path to save augmented annotations in COCO format.')
    parser.add_argument('--custom_regex', type=str, default=".*",
                        help='Optional custom regex pattern to match image filenames. Defaults to matching all images.')
    parser.add_argument('--only_inbreast', action='store_true',
                        help='If set, only augment images with numerical filenames (e.g., 12345.jpg).')
    parser.add_argument('--append', action='store_true',
                        help='If set, combines augmented and non-augmented images and annotations.')
    return parser.parse_args()

# Main augmentation process
def main():
    args = parse_args()

    # Set regex to numerical only if --only_inbreast is provided
    if args.only_inbreast:
        args.custom_regex = r"^\d+\.(jpg|png|jpeg)$"

    os.makedirs(args.output_image_folder, exist_ok=True)

    with open(args.input_annotation_file, 'r') as f:
        coco_data = json.load(f)

    augmented_annotations = coco_data.copy()
    augmented_annotations['images'] = []
    augmented_annotations['annotations'] = []

    new_image_id = max(image['id'] for image in coco_data['images']) + 1
    new_annotation_id = max(anno['id'] for anno in coco_data['annotations']) + 1

    for image_info in tqdm(coco_data['images']):
        image_filename = image_info['file_name']

        # Match image filenames based on custom or inbreast regex
        if re.match(args.custom_regex, image_filename):
            image_path = os.path.join(args.input_image_folder, image_filename)
            image = cv2.imread(image_path)

            image_id = image_info['id']
            annotations = [anno for anno in coco_data['annotations'] if anno['image_id'] == image_id]

            bboxes = [anno['bbox'] for anno in annotations]
            category_ids = [anno['category_id'] for anno in annotations]

            for i in range(n_times):
                augmented = transform(image=image, bboxes=bboxes, category_ids=category_ids)
                augmented_image = augmented['image']
                augmented_bboxes = augmented['bboxes']
                augmented_category_ids = augmented['category_ids']

                augmented_image_filename = f"{os.path.splitext(image_filename)[0]}_aug_{i}.jpg"
                output_image_path = os.path.join(args.output_image_folder, augmented_image_filename)
                cv2.imwrite(output_image_path, augmented_image)

                new_image_info = {
                    'id': new_image_id,
                    'file_name': augmented_image_filename,
                    'width': image_info['width'],
                    'height': image_info['height'],
                }
                augmented_annotations['images'].append(new_image_info)

                for bbox, category_id in zip(augmented_bboxes, augmented_category_ids):
                    new_annotation_info = {
                        'id': new_annotation_id,
                        'image_id': new_image_id,
                        'category_id': category_id,
                        'bbox': bbox,
                        'area': bbox[2] * bbox[3],
                        'iscrowd': 0
                    }
                    augmented_annotations['annotations'].append(new_annotation_info)
                    new_annotation_id += 1

                new_image_id += 1
        else:
            # If --append is provided, copy the non-augmented images/annotations
            if args.append:
                # Copy the original image to output folder
                image_path = os.path.join(args.input_image_folder, image_filename)
                output_image_path = os.path.join(args.output_image_folder, image_filename)
                cv2.imwrite(output_image_path, cv2.imread(image_path))

                # Append the non-augmented image info and annotations to the output
                augmented_annotations['images'].append(image_info)
                annotations = [anno for anno in coco_data['annotations'] if anno['image_id'] == image_info['id']]
                augmented_annotations['annotations'].extend(annotations)

    with open(args.output_annotation_file, 'w') as f:
        json.dump(augmented_annotations, f, indent=4)

    print(
        f"Augmentation completed. Augmented images saved to {args.output_image_folder} and annotations to {args.output_annotation_file}")

    print("Converting COCO to YOLO...")
    coco_to_yolo(args.output_annotation_file, os.path.join(Path(args.output_image_folder).parent, labels_dir_name),
                 args.output_image_folder)


if __name__ == "__main__":
    main()
