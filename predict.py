# For prediction
import os
import sys
import argparse
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from net.model import UaNet as Net
from utils.util import pad2factor, normalize, crop_boxes2mask
from config import config

class Predictor:
    def __init__(self, checkpoint_path, config=config, use_rcnn=True, use_mask=True):
        self.config = config
        self.use_rcnn = use_rcnn
        self.use_mask = use_mask
        self.net = self.load_model(checkpoint_path)

    def read_image(self, image_path):
        return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    def image_to_fake_custom_3d(self, image_data):
        # return np.expand_dims(image_data, 0)
        # Apparently, not necessary, for now disabled...
        return image_data

    def image_to_model_input(self, image):
        if isinstance(image, str) or isinstance(image, os.PathLike):
            original_img = self.image_to_fake_custom_3d(
                self.read_image(
                    image
                )
            )
        else:
            original_img = self.image_to_fake_custom_3d(
                    image
                )
        imgs = original_img.copy()
        imgs = imgs[np.newaxis, ...].astype(np.float32)
        imgs = pad2factor(imgs)
        model_input = normalize(imgs)
        return torch.from_numpy(model_input).float()

    def predict(self, image_path):
        if not self.net:
            print("Model not loaded!", file=sys.stderr)
            return False
        model_input = self.image_to_model_input(image_path)
        self.net.set_mode('eval')
        self.net.use_rcnn = self.use_rcnn
        self.net.use_mask = self.use_mask
        with torch.no_grad():
            self.net.forward(model_input, None, None, None, None)
        crop_boxes = self.net.crop_boxes
        segments = [F.sigmoid(m).cpu().numpy() > 0.5 for m in self.net.mask_probs]
        pred_mask = crop_boxes2mask(crop_boxes[:, 1:], segments, model_input.shape[2:])
        pred_mask = pred_mask.astype(np.uint8)
        return pred_mask

    def load_model(self, checkpoint_path):
        net = Net(self.config).cuda()
        checkpoint = torch.load(checkpoint_path)
        # Load weights into model
        state_dict = net.state_dict()
        state_dict.update({k: v for k, v in checkpoint['state_dict'].items() if k in state_dict})
        net.load_state_dict(state_dict)
        return net

def main():
    # Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', required=True, type=str)
    parser.add_argument('-i', '--image', required=True, type=str)
    parsed = parser.parse_args()
    # Predict
    checkpoint_path = parsed.checkpoint
    image_path = parsed.image
    predictor = Predictor(checkpoint_path)
    predictor.predict(image_path)

if __name__ == '__main__':
    main()