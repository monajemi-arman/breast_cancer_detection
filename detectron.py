#!/usr/bin/env python
# Train Faster R-CNN model using Detectron
import os

import cv2
import torch
import torchvision
import torchvision.transforms as transforms
import detectron2
from detectron2.utils.logger import setup_logger
from detectron2 import model_zoo
from detectron2.engine import DefaultTrainer
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.structures import BoxMode
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.evaluation import COCOEvaluator, inference_on_dataset, LVISEvaluator
from detectron2.data import build_detection_test_loader
from detectron2.utils.visualizer import ColorMode
from detectron2.data.datasets.coco import load_coco_json
from argparse import ArgumentParser

# --- Parameters --- #
# Trainer
epochs = 100
checkpoint_period = 10  # Save every 10 epochs
batch_size = 4
num_workers = 4
pretrained = False
# Paths
coco_json = {'train': 'train.json', 'val': 'val.json', 'test': 'test.json'}
coco_image = {'train': 'train/images', 'val': 'val/images', 'test': 'test/images'}
yaml_config = "COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"
weights_path = "detectron2://COCO-Detection/faster_rcnn_R_50_FPN_3x/137849458/model_final_280758.pkl"
# --- End of Parameters #

argparser = ArgumentParser()
argparser.add_argument('-c', '--choice', help="Mode of program: train / predict / evaluate", type=str)
argparser.add_argument('-i', '--image-path', type=str)
argparser.add_argument('-o', '--output-path', type=str, default="output_of_detectron.jpg")
parsed = argparser.parse_args()

choice = None
if hasattr(parsed, "choice"):
    choice = parsed.choice.lower()

if hasattr(parsed, "image_path"):
    image_path = parsed.image_path

if hasattr(parsed, "output_path"):
    output_path = parsed.output_path

while choice not in ['train', 'evaluate', 'predict']:
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
cfg.DATASETS.TRAIN = ("train", )
cfg.DATASETS.TEST = ("test", )
if pretrained:
    cfg.MODEL.WEIGHTS = weights_path
else:
    cfg.MODEL.WEIGHTS = ""
cfg.DATALOADER.NUM_WORKERS = num_workers
cfg.SOLVER.IMS_PER_BATCH = batch_size
cfg.SOLVER.BASE_LR = 0.0001
cfg.SOLVER.CHECKPOINT_PERIOD = train_size / batch_size * checkpoint_period
# (train_size / batch_size) * epochs
cfg.SOLVER.MAX_ITER = int(train_size / batch_size * epochs)
cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(MetadataCatalog.get("train").thing_classes)
# ./output
os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

if choice == 'train':
    # Training
    trainer = DefaultTrainer(cfg)
    if pretrained:
        trainer.resume_or_load(resume=False)
    else:
        trainer.train()

elif choice == 'predict':
    predictor = DefaultPredictor(cfg)
    if not image_path:
        image_path = input("Enter image path: ")
    image = cv2.imread(image_path)
    outputs = predictor(image)
    print(outputs)
