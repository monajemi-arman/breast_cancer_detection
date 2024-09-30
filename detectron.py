#!/usr/bin/env python
# Train Faster R-CNN model using Detectron
import os
import sys
from pathlib import Path
from sys import stderr

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer, DefaultPredictor
from detectron2.config import get_cfg
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, LVISEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets.coco import load_coco_json
from sklearn.metrics import precision_recall_curve
from argparse import ArgumentParser
from cloudpickle import pickle
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, Button
import json

# --- Parameters --- #
# Trainer
epochs = 100
checkpoint_period = 10  # Save every 10 epochs
batch_size = 4
num_workers = 4
pretrained = True
# Paths
coco_json = {'train': 'train.json', 'val': 'val.json', 'test': 'test.json'}
coco_image = {'train': 'train/images', 'val': 'val/images', 'test': 'test/images'}
yaml_config = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
pretrained_weights_path = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
cfg_output = "detectron.cfg.pkl"
# --- End of Parameters #

slider = None  # Bug

def load_ground_truth(dataset_path, file_name, format='coco'):
    if dataset_path is None:
        return [], []  # No ground truth if dataset path is not provided

    if format == 'coco':
        with open(dataset_path) as f:
            data = json.load(f)

        # Find the image entry matching the file_name
        image_entry = next((img for img in data['images'] if img['file_name'] == file_name), None)

        if image_entry is None:
            print(f"Image with file name '{file_name}' not found in dataset.", file=sys.stderr)
            return None


        image_id = image_entry['id']

        # Get annotations for this image
        annotations = [ann for ann in data['annotations'] if ann['image_id'] == image_id]
        gt_boxes = [ann['bbox'] for ann in annotations]
        gt_classes = [ann['category_id'] for ann in annotations]
        return gt_boxes, gt_classes

    elif format == 'yolo':
        gt_boxes, gt_classes = []
        label_file = os.path.join(dataset_path, f"{os.path.splitext(file_name)[0]}.txt")

        if not os.path.exists(label_file):
            raise ValueError(f"Label file '{label_file}' not found for image '{file_name}'.")

        with open(label_file, 'r') as f:
            for line in f.readlines():
                cls, x_center, y_center, width, height = map(float, line.split())
                gt_classes.append(int(cls))
                # Convert YOLO format to COCO format
                x1 = x_center - width / 2
                y1 = y_center - height / 2
                gt_boxes.append([x1, y1, width, height])
        return gt_boxes, gt_classes

    else:
        raise ValueError("Unsupported format. Please use 'coco' or 'yolo'.")

def get_dataset_path(image_path, coco_json):
    for split in ('train', 'test', 'val'):
        if f'{split}/' in image_path:
            return coco_json[split]
    return None

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Slider, Button

# Declare global variables
slider, show_labels, show_gt, has_gt = None, True, False, False
gt_boxes, gt_classes = [], []  # Initialize globally to avoid NameError

def visualize_predictions(image, predictions, dataset_path=None, file_name=None, format='coco',
                          confidence_threshold=0.5):
    global slider, show_labels, show_gt, has_gt, gt_boxes, gt_classes  # Declare all global variables
    pred_boxes = predictions['instances'].pred_boxes.tensor.cpu().numpy()
    scores = predictions['instances'].scores.cpu().numpy()
    pred_classes = predictions['instances'].pred_classes.cpu().numpy()

    # Load ground truth data if dataset path is provided
    has_gt = False  # Reset has_gt for each call
    if dataset_path and file_name:
        loaded_gt = load_ground_truth(dataset_path, file_name, format)
        # If loaded successfully
        if loaded_gt:
            gt_boxes, gt_classes = loaded_gt
            has_gt = True

    fig, ax = plt.subplots(1, figsize=(12, 8))
    plt.subplots_adjust(left=0.1, bottom=0.25)
    ax.imshow(image)

    # Function to update plot
    def update(val):
        global gt_boxes, gt_classes  # Access global gt_boxes and gt_classes
        threshold = slider.val
        ax.clear()

        # Filter boxes based on confidence threshold
        keep = scores >= threshold
        filtered_boxes = pred_boxes[keep]
        filtered_scores = scores[keep]
        filtered_classes = pred_classes[keep]

        # Display predictions
        ax.imshow(image)
        for box, score, cls in zip(filtered_boxes, filtered_scores, filtered_classes):
            x1, y1, x2, y2 = box
            width, height = x2 - x1, y2 - y1
            rect = Rectangle((x1, y1), width, height, linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)
            if show_labels:
                ax.text(x1, y1, f'Class: {cls}, Score: {score:.2f}', color='yellow', fontsize=12,
                        verticalalignment='top')

        # Display ground truth boxes if toggle is enabled
        if show_gt and has_gt:
            for gt_box, gt_cls in zip(gt_boxes, gt_classes):
                x1, y1, width, height = gt_box
                rect = Rectangle((x1, y1), width, height, linewidth=2, edgecolor='g', facecolor='none')
                ax.add_patch(rect)
                ax.text(x1, y1, f'GT Class: {gt_cls}', color='green', fontsize=12, verticalalignment='top')

        plt.draw()

    # Initialize the plot with a default threshold
    initial_threshold = confidence_threshold
    ax_slider = plt.axes([0.1, 0.1, 0.8, 0.05], facecolor='lightgray')
    slider = Slider(ax_slider, 'Confidence', 0, 1, valinit=initial_threshold)
    slider.on_changed(update)

    # Toggle button for labels
    ax_toggle_labels = plt.axes([0.81, 0.03, 0.1, 0.05])
    toggle_button_labels = Button(ax_toggle_labels, 'Toggle Labels')

    def toggle_labels(event):
        global show_labels
        show_labels = not show_labels
        update(None)

    toggle_button_labels.on_clicked(toggle_labels)

    # Toggle button for ground truth (only if dataset path is provided)
    if has_gt:
        ax_toggle_gt = plt.axes([0.68, 0.03, 0.1, 0.05])
        toggle_button_gt = Button(ax_toggle_gt, 'Toggle Truth')

        def toggle_gt(event):
            global show_gt
            show_gt = not show_gt
            update(None)

        toggle_button_gt.on_clicked(toggle_gt)

    update(initial_threshold)
    plt.show()

def train(cfg, parsed=None):
    trainer = DefaultTrainer(cfg)
    if cfg.MODEL.WEIGHTS:
        trainer.resume_or_load(resume=False)
    trainer.train()

def predict(cfg, parsed):
    if parsed.image_path:
        image_path = parsed.image_path
    else:
        image_path = input("Enter image path: ")
    if parsed.weights_path:
        weights_path = parsed.weights_path
    else:
        weights_path = input("Enter weights path: ")
    cfg.MODEL.WEIGHTS = weights_path
    predictor = DefaultPredictor(cfg)
    image = cv2.imread(image_path)
    predictions = predictor(image)
    dataset_path = get_dataset_path(image_path, coco_json)
    if os.path.exists(dataset_path):
        file_name = Path(image_path).parts[-1]
        visualize_predictions(image, predictions, dataset_path=dataset_path, file_name=file_name)
    else:
        visualize_predictions(image, predictions)


def evaluate(cfg, parsed=None, dataset_name="test"):
    cfg.DATASETS.TEST = ("test",)
    if parsed.weights_path:
        weights_path = parsed.weights_path
    else:
        weights_path = input("Enter weights path: ")
    cfg.MODEL.WEIGHTS = weights_path

    if dataset_name not in DatasetCatalog.list():
        raise ValueError(f"Dataset '{dataset_name}' is not registered.")

    evaluator = COCOEvaluator(dataset_name, cfg, False, output_dir="./output/")
    test_loader = build_detection_test_loader(cfg, dataset_name)

    predictor = DefaultPredictor(cfg)

    all_ground_truths = []
    all_predictions = []

    small_count, medium_count, large_count = 0, 0, 0  # Initialize counters for small, medium, large objects

    for inputs in test_loader:
        # Get ground truth annotations
        image_id = inputs[0]["image_id"]
        annotations = DatasetCatalog.get(dataset_name)[image_id].get("annotations", [])

        # Prepare ground truth bounding boxes and classes
        gt_boxes = [ann["bbox"] for ann in annotations]
        gt_classes = [ann["category_id"] for ann in annotations]
        all_ground_truths.append({"boxes": gt_boxes, "classes": gt_classes})

        # Count small, medium, and large objects based on bounding box areas
        for bbox in gt_boxes:
            width, height = bbox[2], bbox[3]
            area = width * height
            if area < 1024:
                small_count += 1
            elif 1024 <= area < 9216:
                medium_count += 1
            else:
                large_count += 1

        # Get predictions
        outputs = predictor(
            np.transpose(inputs[0]["image"]
                         .numpy(),
                         (1, 2, 0))
        )
        pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
        scores = outputs["instances"].scores.cpu().numpy()
        pred_classes = outputs["instances"].pred_classes.cpu().numpy()

        predictions = {
            "boxes": pred_boxes,
            "scores": scores,
            "classes": pred_classes
        }
        all_predictions.append(predictions)

    results = inference_on_dataset(predictor.model, test_loader, evaluator)
    print(results)
    print(f"Number of small, medium, and large objects: {small_count}, {medium_count}, {large_count}")

    precision, recall, thresholds = compute_precision_recall(all_ground_truths, all_predictions)

    plt.figure()
    plt.plot(recall, precision, label="Precision-Recall Curve")
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.show()

    return results


def compute_precision_recall(ground_truths, predictions, iou_threshold=0.5):
    all_true_labels = []
    all_scores = []

    for gt, pred in zip(ground_truths, predictions):
        gt_boxes = gt["boxes"]
        pred_boxes = pred["boxes"]
        pred_scores = pred["scores"]

        # Sort predictions by score in descending order
        sorted_indices = np.argsort(pred_scores)[::-1]
        pred_boxes = pred_boxes[sorted_indices]
        pred_scores = pred_scores[sorted_indices]

        true_labels = np.zeros(len(pred_boxes), dtype=bool)

        for i, pred_box in enumerate(pred_boxes):
            for gt_box in gt_boxes:
                iou = calculate_iou(pred_box, gt_box)
                if iou >= iou_threshold:
                    true_labels[i] = True
                    break

        all_true_labels.extend(true_labels)
        all_scores.extend(pred_scores)

    all_true_labels = np.array(all_true_labels)
    all_scores = np.array(all_scores)

    precision, recall, thresholds = precision_recall_curve(all_true_labels, all_scores)

    return precision, recall, thresholds


def calculate_iou(box1, box2):
    # Convert XYWH format to XYXY format
    box1 = [box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]]
    box2 = [box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]]

    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    # Calculate union
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - intersection

    # Calculate IoU
    iou = intersection / union if union > 0 else 0
    return iou

choices_map = {
    'train': train,
    'predict': predict,
    'evaluate': evaluate
}
choices = choices_map.keys()


def main():
    global pretrained_weights_path
    argparser = ArgumentParser()
    argparser.add_argument('-c', '--choice', help="Mode of program: train / predict / evaluate", type=str)
    argparser.add_argument('-i', '--image-path', type=str)
    argparser.add_argument('-w', '--weights-path', type=str)
    argparser.add_argument('-o', '--output-path', type=str, default="output_of_detectron.jpg")
    parsed = argparser.parse_args()

    choice = None
    if parsed.choice:
        choice = parsed.choice.lower()

    if parsed.image_path:
        image_path = parsed.image_path

    if parsed.output_path:
        output_path = parsed.output_path

    while choice not in choices:
        choice = input("Enter mode (train | evaluate | predict): ").lower()

    device = 'cpu'
    if torch.cuda.is_available():
        device = "cuda"

    # Sanity check for input keys
    keys_json = list(coco_json.keys())
    keys_json.sort()
    keys_image = list(coco_image.keys())
    keys_image.sort()
    if keys_json != ['test', 'train', 'val'] or keys_image != keys_json:
        raise Exception("coco_json")

    # Load and register dataset
    for dataset_name in keys_json:
        if dataset_name not in keys_image:
            raise Exception('coco_json and coco_image dictionaries must have same keys!')
        # Load
        loaded_dataset = load_coco_json(json_file=coco_json[dataset_name], image_root=coco_image[dataset_name],
                                        dataset_name=dataset_name)
        if dataset_name == 'train':
            train_size = len(loaded_dataset)
        # Register
        DatasetCatalog.register(dataset_name, lambda: loaded_dataset)

    # Configure before training
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(yaml_config))
    cfg.DATASETS.TRAIN = ("train",)
    cfg.DATASETS.TEST = ("val",)
    cfg.TEST.EVAL_PERIOD = 500
    if pretrained:
        cfg.MODEL.WEIGHTS = pretrained_weights_path

    else:
        cfg.MODEL.WEIGHTS = ""
    cfg.DATALOADER.NUM_WORKERS = num_workers
    cfg.SOLVER.IMS_PER_BATCH = batch_size
    cfg.SOLVER.BASE_LR = 0.0001
    cfg.SOLVER.CHECKPOINT_PERIOD = train_size / batch_size * checkpoint_period
    # (train_size / batch_size) * epochs
    cfg.SOLVER.MAX_ITER = int(train_size / batch_size * epochs)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get("train").thing_classes)
    # Save current config for later use in xai.py
    with open(cfg_output, 'wb') as f:
        pickle.dump(cfg, f)
    # ./output
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # Run choice action
    choices_map[choice](cfg, parsed)


if __name__ == '__main__':
    main()
