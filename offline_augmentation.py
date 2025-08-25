import os
import json
import cv2
import re
import albumentations as A
from tqdm import tqdm
import argparse
import multiprocessing
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
    parser.add_argument('--convert_to_yolo', action='store_true',
                        help='If set, converts COCO annotations to YOLO format at the end.')
    return parser.parse_args()

def augment_worker(args_tuple):
    image_info, image_id_to_annotations, args = args_tuple
    image_filename = image_info['file_name']
    image_path = os.path.join(args.input_image_folder, image_filename)
    image = cv2.imread(image_path)
    if image is None:
        return None
    image_id = image_info['id']
    annotations = image_id_to_annotations.get(image_id, [])
    bboxes = [anno['bbox'] for anno in annotations]
    category_ids = [anno['category_id'] for anno in annotations]
    results = []
    for i in range(n_times):
        augmented = transform(image=image, bboxes=bboxes, category_ids=category_ids)
        augmented_image = augmented['image']
        augmented_bboxes = augmented['bboxes']
        augmented_category_ids = augmented['category_ids']
        augmented_image_filename = f"{os.path.splitext(image_filename)[0]}_aug_{i}.jpg"
        output_image_path = os.path.join(args.output_image_folder, augmented_image_filename)
        cv2.imwrite(output_image_path, augmented_image)
        results.append((augmented_image_filename, image_info, augmented_bboxes, augmented_category_ids))
    return results

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

    # Precompute annotation lookup for speed
    image_id_to_annotations = {}
    for anno in coco_data['annotations']:
        image_id_to_annotations.setdefault(anno['image_id'], []).append(anno)

    new_image_id = max(image['id'] for image in coco_data['images']) + 1
    new_annotation_id = max(anno['id'] for anno in coco_data['annotations']) + 1

    # Prepare image list for augmentation
    images_to_augment = [img for img in coco_data['images'] if re.match(args.custom_regex, img['file_name'])]

    # Multiprocessing pool for augmentation
    with multiprocessing.Pool() as pool:
        tasks = [(img, image_id_to_annotations, args) for img in images_to_augment]
        all_results = list(tqdm(pool.imap(augment_worker, tasks), total=len(tasks)))

    # Assign new IDs and update annotations
    for result_set in all_results:
        if result_set is None:
            continue
        for augmented_image_filename, image_info, augmented_bboxes, augmented_category_ids in result_set:
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

    # Handle non-augmented images if --append is set
    for image_info in coco_data['images']:
        image_filename = image_info['file_name']
        if not re.match(args.custom_regex, image_filename) and args.append:
            image_path = os.path.join(args.input_image_folder, image_filename)
            image = cv2.imread(image_path)
            if image is not None:
                output_image_path = os.path.join(args.output_image_folder, image_filename)
                cv2.imwrite(output_image_path, image)
            augmented_annotations['images'].append(image_info)
            annotations = image_id_to_annotations.get(image_info['id'], [])
            augmented_annotations['annotations'].extend(annotations)

    with open(args.output_annotation_file, 'w') as f:
        json.dump(augmented_annotations, f, indent=4)

    print(
        f"Augmentation completed. Augmented images saved to {args.output_image_folder} and annotations to {args.output_annotation_file}")

    if args.convert_to_yolo:
        print("Converting COCO to YOLO...")
        coco_to_yolo(args.output_annotation_file, os.path.join(Path(args.output_image_folder).parent, labels_dir_name),
                     args.output_image_folder)


if __name__ == "__main__":
    main()
