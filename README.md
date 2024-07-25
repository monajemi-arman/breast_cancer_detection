# Breast_Cancer_Detection
Breast cancer detection using mammography images, utilizing deep learning models

Supported datasets:
* InBreast
* CBIS-DDSM (Curated Breast Imaging Subset of DDSM)
* MIAS (Mammography Image Analysis Society)

Supported models:
* Generally supported models
  * YOLO 
  * Any model that supports YOLO / COCO style dataset
* Customized [UaNet](https://github.com/uci-cbcl/UaNet/) for 2D mammography images

# Usage
**1. Clone this repository**
```bash
git clone https://github.com/monajemi-arman/breast_cancer_detection
```
**2. Install prerequisites**
```bash
cd breast_cancer_detection
pip install -r requirements.txt
```
**2. Download the following datasets**  
https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset  
https://wiki.cancerimagingarchive.net/pages/viewpage.action?pageId=22516629  
https://www.kaggle.com/datasets/kmader/mias-mammography  

**3. Move dataset files**  
First create 'datasets' directory:
```bash
mkdir datasets/
```
Then, extract and move the files to this directory so as to have the following inside datasets/:  
* INbreast Release 1.0/
* CBIS-DDSM/
* all-mias/

**4. Convert datasets to YOLO (and COCO) format**
```bash
python convert_dataset.py
```
After completion, images/, labels/, dataset.yaml, annotations.json would be present in the working directory. 

---
# YOLO
## Training
* Install Ultralytics
```bash
pip install ultralytics
```
* Train your desired YOLO model
```bash
yolo train data=dataset.yaml model=yolov8n

```
## Prediction
Example of prediction using YOLO ultralytics framework:
```bash
yolo predict model=runs/detect/train/weights/best.pt source=images/cb_1.jpg conf=0.1 
```
---
# UaNet
## Training
* Clone UaNet repository
```bash
# Make sure you cd to breast_cancer_detection first
# cd breast_cancer_detection
git clone https://github.com/uci-cbcl/UaNet
```
* Apply patch
```bash
# Linux
patch patches/config.diff UaNet/src/config.py
patch patches/train.diff UaNet/src/train.py
# Windows
# patch python module is in requirements.txt
# if not installed yet, do `pip install patch`
python -m patch patches/config.diff
python -m patch patches/train.diff
```
* Prepare dataset
```bash
# Convert datasets to images/ masks/
python convert_dataset.py -m mask
# Convert to 3D NRRD files
python to_3d_nrrd.py
```
* Move dataset to model directory
```bash
# While in breast_cancer_detection directory
mv UaNet-dataset/* UaNet/data/preprocessed/
# Remove old default configs of UaNet
rm UaNet/src/split/*
mv split/* UaNet/src/split/
```
* Start training
```bash
cd UaNet/src
python train.py
```