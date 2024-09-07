import os
import cv2
import json
import numpy as np
import argparse

# Draw bounding boxes for YOLO format
def draw_yolo_bboxes(image, bbox, color, label=None):
    img_h, img_w, _ = image.shape
    class_id, x_center, y_center, width, height = map(float, bbox)
    x_min = int((x_center - width / 2) * img_w)
    y_min = int((y_center - height / 2) * img_h)
    x_max = int((x_center + width / 2) * img_w)
    y_max = int((y_center + height / 2) * img_h)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    if label:
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# Draw bounding boxes for COCO format
def draw_coco_bboxes(image, bbox, color, label=None):
    x, y, width, height = map(int, bbox)
    cv2.rectangle(image, (x, y), (x + width, y + height), color, 10)
    if label:
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# Resize the image to fit within half the screen dimensions if it's too large
def resize_to_half_screen(image):
    screen_width, screen_height = 1920, 1080
    img_h, img_w = image.shape[:2]
    scaling_factor = min(screen_width / 2 / img_w, screen_height / 2 / img_h, 1.0)  # Only resize if larger than half screen
    new_size = (int(img_w * scaling_factor), int(img_h * scaling_factor))
    return cv2.resize(image, new_size) if scaling_factor < 1 else image

# Visualizer for YOLO dataset
def visualize_yolo(image_path, label_path, class_names):
    image = cv2.imread(image_path)
    with open(label_path, 'r') as file:
        labels = file.readlines()
    for label in labels:
        label = label.strip().split()
        class_id = int(label[0])
        bbox = label[1:]
        color = (0, 255, 0)  # Green for YOLO boxes
        draw_yolo_bboxes(image, [class_id] + bbox, color, class_names[class_id])
    image = resize_to_half_screen(image)
    cv2.imshow('YOLO Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Visualizer for COCO dataset
def visualize_coco(image_path, annotation, class_names):
    image = cv2.imread(image_path)
    for obj in annotation:
        bbox = obj['bbox']
        class_id = obj['category_id']
        color = (0, 0, 255)  # Red for COCO boxes
        if len(class_names) == 1 and class_id > 0:
            class_id -= 1
        draw_coco_bboxes(image, bbox, color, class_names[class_id])
    image = resize_to_half_screen(image)
    cv2.imshow('COCO Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main visualizer
def visualize(image_name, dataset_type, dataset_path, label_path, class_names):
    image_path = os.path.join(dataset_path, image_name)
    if dataset_type == 'yolo':
        label_file = os.path.splitext(image_name)[0] + ".txt"
        label_full_path = os.path.join(label_path, label_file)
        if not os.path.exists(label_full_path):
            print(f"Label file {label_full_path} not found!")
            return
        visualize_yolo(image_path, label_full_path, class_names)
    elif dataset_type == 'coco':
        with open(label_path, 'r') as file:
            annotations = json.load(file)
        image_id = next((img['id'] for img in annotations['images'] if img['file_name'] == image_name), None)
        if image_id is None:
            print(f"Image {image_name} not found in COCO annotations!")
            return
        image_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]
        visualize_coco(image_path, image_annotations, class_names)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize bounding boxes on images for YOLO or COCO datasets.')
    parser.add_argument('-m', '--mode', choices=['yolo', 'coco'], required=True, help="Dataset mode: 'yolo' or 'coco'")
    parser.add_argument('-d', '--dataset-path', type=str, required=True, help='Path to the folder containing images')
    parser.add_argument('-l', '--label-path', type=str, required=True, help='Path to annotations folder or JSON file')
    parser.add_argument('-i', '--image-name', type=str, required=True, help='Name of the image file to visualize')
    args = parser.parse_args()
    class_names = ["mass"]
    visualize(args.image_name, args.mode, args.dataset_path, args.label_path, class_names)
