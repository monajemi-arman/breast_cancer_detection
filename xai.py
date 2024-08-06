#!/usr/bin/env python
# Explainable AI

from YOLOv8_Explainer import yolov8_heatmap, display_images
from argparse import ArgumentParser


def yolo_heatmap(image_path, weight_path, method='GradCAM', display=False):
    model_heatmap = yolov8_heatmap(
        weight=weight_path,
        method=method
    )
    result = model_heatmap(image_path)
    if display:
        display_images(result)
    return result


def main():
    parser = ArgumentParser()
    parser.add_argument('-i', '--image-path', required=True, help="Path to image")
    parser.add_argument('-w', '--weight-path', required=True, help="Path to weights file (.pt)")
    parser.add_argument('-m', '--method', default="GradCAM", choices=["GradCAM", "HiResCAM",
                                                                      "GradCAMPlusPlus", "XGradCAM", "LayerCAM",
                                                                      "EigenGradCAM", "EigenCAM"])
    parsed = parser.parse_args()

    yolo_heatmap(image_path=parsed.image_path, weight_path=parsed.weight_path, method=parsed.method, display=True)


main()
