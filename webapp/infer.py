#!/usr/bin/env python3
import argparse
import hashlib
from pathlib import Path
import cloudpickle as pickle
import matplotlib.pyplot as plt
import torch
from detectron2.engine import DefaultPredictor
import os
import cv2

# Parameters
# Change these
model_path = 'model.pth'
cfg_path = 'detectron.cfg.pkl'
image_target_dims = [512, 512]
threshold = 0.5
# End of parameters

# Update paths to real
script_dir = os.path.dirname(os.path.realpath(__file__))
model_path = os.path.join(script_dir, model_path)
cfg_path = os.path.join(script_dir, cfg_path)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Predictor:
    def __init__(self):
        self.prediction_cache = {}

        # Load model using source from checkpoint
        with open(cfg_path, 'rb') as f:
            cfg = pickle.load(f)
        cfg.MODEL.WEIGHTS = model_path
        cfg.MODEL.DEVICE = device
        self.predictor = DefaultPredictor(cfg)

    @staticmethod
    def show_image(image):
        plt.imshow(image)
        plt.axis('off')
        plt.show()

    @staticmethod
    def array_hash(array):
        return hashlib.md5(array.tobytes()).hexdigest()


    def overlay_predictions(self, image, predictions, confidence_threshold=0.5):
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
            cv2.putText(image, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 0),
                        thickness)

        return image


    def predict(self, image):
        predictions = self.predictor(image)
        return predictions

    def predict_with_cache(self, image):
        image_hash = self.array_hash(image)
        cached_prediction = self.prediction_cache.get(image_hash)
        if cached_prediction:
            return cached_prediction
        else:
            predictions = self.predict(image)
            self.prediction_cache[image_hash] = predictions
            return predictions


    def infer(self, image, details=True):
        predictions = self.predict_with_cache(image)
        image_with_overlay = self.overlay_predictions(image, predictions, confidence_threshold=threshold)
        if details:
            predictions_json = self.convert_predictions_to_json(predictions)
            return image_with_overlay, predictions_json
        else:
            return image_with_overlay


    def convert_predictions_to_json(self, predictions):
        """
        Convert object detection predictions to a JSON-serializable format.

        Args:
            predictions (dict): The predictions dictionary containing `instances`.

        Returns:
            list: A list of JSON-serializable objects.
        """
        instances = predictions.get('instances', None)
        if instances is None:
            return []  # Return an empty list if `instances` is not present

        num_instances = len(instances)
        boxes = (
            instances.pred_boxes.tensor.cpu().numpy()
            if hasattr(instances, 'pred_boxes') and hasattr(instances.pred_boxes, 'tensor')
            else None
        )
        scores = (
            instances.scores.cpu().numpy()
            if hasattr(instances, 'scores')
            else None
        )
        classes = (
            instances.pred_classes.cpu().numpy()
            if hasattr(instances, 'pred_classes')
            else None
        )

        output = []
        for i in range(num_instances):
            prediction = {}
            if boxes is not None and len(boxes) > i:
                prediction["box"] = {
                    "x_min": float(boxes[i][0]),
                    "y_min": float(boxes[i][1]),
                    "x_max": float(boxes[i][2]),
                    "y_max": float(boxes[i][3]),
                }
            if scores is not None and len(scores) > i:
                prediction["score"] = float(scores[i])
            if classes is not None and len(classes) > i:
                prediction["class"] = int(classes[i])  # Convert to plain integer

            if prediction:  # Only add if there's at least one field
                output.append(prediction)

        return output


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
    predictor = Predictor()
    output_image = Predictor.infer(image)
    Predictor.show_image(output_image)
