# Breast_Cancer_Detection
Breast cancer detection using mammography images, utilizing deep learning models  
Using InBreast, CBIS-DDSM, and MIAS mammography image datasets

Status: Work In Progress...

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

# Training
## YOLO
* Install Ultralytics
```bash
pip install ultralytics
```
* Train your desired YOLO model
```bash
yolo train data=dataset.yaml model=yolov8n

```

# Prediction
## YOLO
Example of prediction using YOLO ultralytics framework:
```bash
yolo predict model=runs/detect/train/weights/best.pt source=images/cb_1.jpg conf=0.1 
```