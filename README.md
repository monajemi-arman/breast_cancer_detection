# Breast_Cancer_Detection
Breast cancer detection using mammography images, utilizing deep learning models

# Prerequisites
* Nvidia CUDA drivers
  * Install a PyTorch compatible version of CUDA from:
    * Your Linux repository
    ```commandline
    apt install nvidia-cuda-toolkit
    ```
    * NVIDIA website for Windows and Linux
      * [Link to download page](https://developer.nvidia.com/cuda-downloads)
* Pytorch with CUDA support
  * Visit [PyTorch website](https://pytorch.org/get-started/locally/) for more information

These two must be installed __manually__ or else will break installation of other requirements later on.   

# Datasets

Supported datasets:
* InBreast
* CBIS-DDSM (Curated Breast Imaging Subset of DDSM)
* MIAS (Mammography Image Analysis Society)

Supported models:
* Generally supported models
  * YOLO 
  * Any model that supports YOLO / COCO style dataset
* Customized [UaNet](https://github.com/uci-cbcl/UaNet/) for 2D mammography images

## Download
### Google Colab
* Use download_datasets_colab.ipynb jupyter notebook in Google Colab to download all datasets.
* You will need to upload your _'kaggle.json'_ when the notebook gives you an upload dialog.
* After logging in to kaggle, you can get your kaggle json in API section of https://www.kaggle.com/settings.
* The notebook will clone this repository and download all datasets.
### Manual
Dataset links:
* https://www.kaggle.com/datasets/ramanathansp20/inbreast-dataset
* https://www.kaggle.com/datasets/awsaf49/cbis-ddsm-breast-cancer-image-dataset
* https://www.kaggle.com/datasets/kmader/mias-mammography

Download the above datasets and after cloning this repository, create the following directories:
* breast_cancer_detection/
  * datasets/
    * all-mias/
      * mdb001.pgm
      * ...
    * CBIS-DDSM/
      * csv/
      * jpeg/
    * INbreast Release 1.0/
      * AllDICOMs/
      * ...

Copy datasets to directories accordingly.

## Visualizer
After converting the datasets to COCO / YOLO style in the next section (Usage),
you may visualize the standardized dataset using the following methods.
### COCO Style dataset
```bash
python visualizer.py -m coco -d train/images -l train.json 
```
### YOLO Style dataset
```bash
python visualizer.py -m yolo -d train/images -l train/labels 
```

![](demo/visualizer.png)

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
# Detectron (Faster R-CNN)
* **Install prerequisites** (included in requirements.txt)
```bash
pip install detectron2
```
## Train
The purpose of detectron.py is to train and evaluate a Faster R-CNN model and predict using detectron2 platform.
```bash
python detectron.py -c train
```
## Predict
* Visualize model prediction
* Show ground truth and labels
* Filter predictions by confidence score
``` bash
# After training is complete
python detectron.py -c predict -w output/model_final.pth -i <image path>
# -w: path to model weights
```
![detectron prediction visualizer](demo/detectron_predict_visualize.png)

## Evaluate
### Evaluation using COCOEvaluator
* Calculate mAP
* Uses test dataset by default
```bash
python detectron.py -c evaluate -w output/model_final.pth
```
### Save predictions in COCO style JSON 
* Suitable for later offline metrics calculation
* All predictions of the test dataset will be written to predicions.json
* Follows COCO format
```bash
python detectron.py -c evaluate_test_to_coco -w output/model_final.pth
```

---
# UaNet
## Training
* Clone UaNet repository (patched)
```bash
# Make sure you cd to breast_cancer_detection first
# cd breast_cancer_detection
git clone https://github.com/monajemi-arman/UaNet_2D
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
mv UaNet-dataset/* UaNet_2D/data/preprocessed/
# Remove old default configs of UaNet
mv split/* UaNet_2D/src/split/
```
* Start training
```bash
cd UaNet_2D/src
python train.py
```