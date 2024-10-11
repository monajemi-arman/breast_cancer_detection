#!/usr/bin/environ python3
import argparse
from pathlib import Path
import cloudpickle as pickle
import matplotlib.pyplot as plt
import torch
from detectron2.engine import DefaultPredictor

# Parameters
# Change these
model_path = 'model.pth'
cfg_path = 'detectron.cfg.pkl'
image_target_dims = [512, 512]
threshold = 0.7
# End of parameters

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def show_image(image):
    plt.imshow(image)
    plt.axis('off')
    plt.show()


import cv2


def overlay_predictions(image, predictions, confidence_threshold=0.5):
    height, width = image.shape[:2]
    scale_factor = max(width, height) / 1000

    pred_boxes = predictions['instances'].pred_boxes.tensor.cpu().numpy()
    scores = predictions['instances'].scores.cpu().numpy()
    pred_classes = predictions['instances'].pred_classes.cpu().numpy()

    keep = scores >= confidence_threshold
    filtered_boxes = pred_boxes[keep]
    filtered_scores = scores[keep]
    filtered_classes = pred_classes[keep]

    for box, score, cls in zip(filtered_boxes, filtered_scores, filtered_classes):
        x1, y1, x2, y2 = box
        thickness = max(6, int(6 * scale_factor))
        font_scale = max(0.5, 1 * scale_factor)

        cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), color=(0, 0, 255), thickness=thickness)
        label = f'Class: {cls}, Score: {score:.2f}'
        cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_ITALIC, font_scale, (255, 255, 0),
                    thickness)

    return image

def predict(image):
    # Load model using source from checkpoint
    with open(cfg_path, 'rb') as f:
        cfg = pickle.load(f)
    cfg.MODEL.WEIGHTS = model_path
    predictor = DefaultPredictor(cfg)
    predictions = predictor(image)
    return predictions

def infer(image):
    preds = predict(image)
    image_with_overlay = overlay_predictions(image, preds)
    return image_with_overlay


if __name__ == '__main__':
    # Input arguments
    parser = argparse.ArgumentParser("Inference using model checkpoint")
    parser.add_argument('-i', '--image', required=True, help="Path to input image")
    parser.add_argument('-o', '--output', help="Output image path")
    args = parser.parse_args()
    image_path = args.image
    if args.output:
        output_path = args.output
    else:
        image_path = Path(image_path)
        output_path = image_path.with_stem(image_path.stem + '_out')
        # cv2 doesn't like Path objects as filename apparently
        image_path, output_path = str(image_path), str(output_path)
    # Infer
    image = cv2.imread(image_path)
    output_image = infer(image)
    show_image(output_image)