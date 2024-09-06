import os
import cv2
import json
import numpy as np
import argparse

# Function to draw bounding boxes for YOLO format
def draw_yolo_bboxes(image, bbox, color, label=None):
    img_h, img_w, _ = image.shape
    class_id, x_center, y_center, width, height = bbox
    x_center, y_center, width, height = float(x_center), float(y_center), float(width), float(height)

    # Convert YOLO format to corner coordinates
    x_min = int((x_center - width / 2) * img_w)
    y_min = int((y_center - height / 2) * img_h)
    x_max = int((x_center + width / 2) * img_w)
    y_max = int((y_center + height / 2) * img_h)

    # Draw bounding box and label
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)
    if label:
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# Function to draw bounding boxes for COCO format
def draw_coco_bboxes(image, bbox, color, label=None):
    x, y, width, height = bbox
    x, y, width, height = int(x), int(y), int(width), int(height)

    # Draw bounding box and label
    cv2.rectangle(image, (x, y), (x + width, y + height), color, 2)
    if label:
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

# Function to resize the image to half the screen size
def resize_to_half_screen(image):
    screen_width = 1920  # Replace with your screen width
    screen_height = 1080  # Replace with your screen height

    # Calculate half the screen size
    half_width = screen_width // 2
    half_height = screen_height // 2

    # Resize the image while maintaining the aspect ratio
    img_h, img_w = image.shape[:2]
    scaling_factor = min(half_width / img_w, half_height / img_h)
    new_size = (int(img_w * scaling_factor), int(img_h * scaling_factor))

    return cv2.resize(image, new_size)

# Visualizer for YOLO dataset
def visualize_yolo(image_path, label_path, class_names):
    # Load image
    image = cv2.imread(image_path)

    # Resize image to fit half the screen
    image = resize_to_half_screen(image)

    # Read YOLO label file
    with open(label_path, 'r') as file:
        labels = file.readlines()

    # Draw each bounding box
    for label in labels:
        label = label.strip().split()
        class_id = int(label[0])
        bbox = label[1:]
        color = (0, 255, 0)  # Green for YOLO boxes
        draw_yolo_bboxes(image, [class_id] + bbox, color, class_names[class_id])

    # Show image
    cv2.imshow('YOLO Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Visualizer for COCO dataset
def visualize_coco(image_path, annotation, class_names):
    # Load image
    image = cv2.imread(image_path)

    # Resize image to fit half the screen
    image = resize_to_half_screen(image)

    # Draw each bounding box
    for obj in annotation:
        bbox = obj['bbox']
        class_id = obj['category_id']
        color = (0, 0, 255)  # Red for COCO boxes
        if len(class_names) == 1 and class_id > 0:
            class_id -= 1
        draw_coco_bboxes(image, bbox, color, class_names[class_id])

    # Show image
    cv2.imshow('COCO Bounding Boxes', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Main visualizer
def visualize(image_name, dataset_type, dataset_path, label_path, class_names):
    image_path = os.path.join(dataset_path, image_name)

    if dataset_type == 'yolo':
        label_file = os.path.splitext(image_name)[0] + ".txt"  # Assuming label file has .txt extension
        label_full_path = os.path.join(label_path, label_file)
        if not os.path.exists(label_full_path):
            print(f"Label file {label_full_path} not found!")
            return
        visualize_yolo(image_path, label_full_path, class_names)

    elif dataset_type == 'coco':
        # Load COCO annotations from the JSON file
        with open(label_path, 'r') as file:
            annotations = json.load(file)

        # Find the relevant annotations for the selected image
        image_id = next((img['id'] for img in annotations['images'] if img['file_name'] == image_name), None)
        if image_id is None:
            print(f"Image {image_name} not found in COCO annotations!")
            return

        # Extract annotations corresponding to the selected image
        image_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]
        visualize_coco(image_path, image_annotations, class_names)

if __name__ == '__main__':
    # Set up argument parser
    parser = argparse.ArgumentParser(description='Visualize bounding boxes on images for YOLO or COCO datasets.')

    parser.add_argument('-m', '--mode', choices=['yolo', 'coco'], required=True,
                        help="Dataset mode: 'yolo' for YOLO format or 'coco' for COCO format")
    parser.add_argument('-d', '--dataset-path', type=str, required=True,
                        help='Path to the folder containing images')
    parser.add_argument('-l', '--label-path', type=str, required=True,
                        help='Path to YOLO annotations folder or COCO annotations JSON file')
    parser.add_argument('-i', '--image-name', type=str, required=True,
                        help='Name of the image file to visualize')

    # Parse arguments
    args = parser.parse_args()

    # Set class names to only "mass"
    class_names = ["mass"]

    # Call the visualize function with parsed arguments
    visualize(args.image_name, args.mode, args.dataset_path, args.label_path, class_names)
