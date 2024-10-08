#!/usr/bin/env python
# Explainable AI

import torch
from argparse import ArgumentParser

from Cython.Compiler.Future import annotations
from YOLOv8_Explainer import yolov8_heatmap, display_images
from detectron2.modeling import build_model
from cloudpickle import pickle
from detectron2.checkpoint import DetectionCheckpointer
from detectron import cfg_output
import cv2
from pytorch_grad_cam import AblationCAM, EigenCAM
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels, scale_cam_image
from detectron2.data.datasets.coco import load_coco_json
import numpy as np
from uri_template.expansions import Expansion

cfg_output = 'detectron.cfg.pkl'
json_file = 'test.json'
image_root = 'test/images'
idx = 0  # Index of image in dataset to choose

input_data = load_coco_json(json_file, image_root, dataset_name='test')[idx]

def yolo_heatmap(input_data, weight_path, method='GradCAM', display=False):
    image_path = input_data['file_name']
    model_heatmap = yolov8_heatmap(
        weight=weight_path,
        method=method
    )
    result = model_heatmap(image_path)
    if display:
        display_images(result)
    return result


def detectron_heatmap(input_data, weight_path, method='GradCAM', display=False, cfg_output=cfg_output):
    # Load model
    with open(cfg_output, 'rb') as f:
        cfg = pickle.load(f)
    model = build_model(cfg)
    DetectionCheckpointer(model).load(weight_path)
    # Load data
    image_path = input_data['file_name']
    image_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    labels, boxes = [], []
    for annotation in input_data['annotations']:
        labels.append(annotation['category_id'])
        boxes.append(annotation['bbox'])

function_map = {
    'yolo': yolo_heatmap,
    'detectron': detectron_heatmap
}


def main():
    parser = ArgumentParser()
    parser.add_argument('-t', '--type-model', required=True, help="Model type (yolo, pytorch, detectron)",
                        choices=['yolo', 'pytorch', 'detectron'])
    parser.add_argument('-w', '--weight-path', required=True, help="Path to weights file (.pt or .pth)")
    parser.add_argument('-m', '--method', default="GradCAM", choices=["GradCAM", "HiResCAM",
                                                                      "GradCAMPlusPlus", "XGradCAM", "LayerCAM",
                                                                      "EigenGradCAM", "EigenCAM"])
    parsed = parser.parse_args()

    heatmap = function_map[parsed.type_model]
    heatmap(input_data, weight_path=parsed.weight_path, method=parsed.method, display=True)


if __name__ == '__main__':
    main()
    detectron_heatmap(cfg_output)