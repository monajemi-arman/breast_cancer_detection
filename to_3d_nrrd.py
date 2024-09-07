#!/usr/bin/env python
# Purpose #
# Goal is to convert images/ labels/ into nrrd files supported by UaNet project
# => https://github.com/uci-cbcl/UaNet/
# Why? The model works on 3D images, the input is supposed to be 3D NRRD.
# But breast cancer datasets are 2D, in order to use the mode, we convert them into fake nifti 3d images.
# Usage #
# Use convert_dataset.py on the datasets, but choose 'mask' mode (output_choice = 'mask')
# This script then uses the generated images/ and masks/
# Output #
# By default, this script outputs the following directories:
# - UaNet-Dataset: Containing the images and masks in nifti format
# - split: Containing the csv list of image prefixes
# In order to use UaNet on mass lesions of mammography images, you must change the config.py to have:
# 'roi_names': ['mass']
# Limitations #
# Only works on single class mask images (binary)

import nrrd
import cv2
import os
from pathlib import Path
import numpy as np

# --- Parameters --- #
# Target size
target_size = [512, 512]  # Set to None if image resizing is not required
# File and directory names
images_dir = 'images'
masks_dir = 'masks'
output_dir = 'UaNet-dataset'
mask_suffix = 'mass'
npy_name = {
    'dir': 'split',
    'train': 'dataset1_2_train.csv',
    'val': 'dataset1_2_val.csv',
    'test': 'release_dataset1_test.csv'
}

# --- End of Parameters --- #

# Create directories not existing already
for directory in [output_dir, npy_name['dir']]:
    if not os.path.exists(directory):
        os.mkdir(directory)

mask_names = os.listdir(masks_dir)
mask_prefixes = [Path(x).stem for x in mask_names]

# Convert images to nrrd
for image_name in os.listdir(images_dir):
    image_prefix = Path(image_name).stem
    # Each mask name corresponds to an image name, if not, there is no mask for that image, so skip.
    if image_prefix in mask_prefixes:
        image_path = os.path.join(images_dir, image_name)
        image_data = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        # Resize if necessary
        if target_size and len(target_size) == 2:
            image_data = cv2.resize(image_data, target_size, interpolation=cv2.INTER_AREA)
        # Fake 3D
        image_data = np.expand_dims(image_data, 0)
        image_name_nrrd = Path(image_name).stem + '_clean.nrrd'
        # Write NRRD
        output_path = os.path.join(output_dir, image_name_nrrd)
        if not os.path.exists(output_path):
            nrrd.write(output_path, image_data)

# Create mask nrrd
for mask_name in mask_names:
    mask_path = os.path.join(masks_dir, mask_name)
    mask_data = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    # Resize if necessary
    if target_size and len(target_size) == 2:
        mask_data = cv2.resize(mask_data, target_size, interpolation=cv2.INTER_AREA)
    # Apply threshold to remove random pixels
    _, mask_data = cv2.threshold(mask_data, 127, 255, cv2.THRESH_BINARY)
    mask_name_nrrd = Path(mask_name).stem + '_' + mask_suffix + '.nrrd'
    output_path = os.path.join(output_dir, mask_name_nrrd)
    # Fake 3D
    mask_data = np.expand_dims(mask_data, 0)
    # Write NRRD
    if not os.path.exists(output_path):
        nrrd.write(output_path, mask_data)

# Split into train/val/test
n = len(mask_prefixes)
t, v = int(n * 0.8), int(n * 0.9)
train, val, test = mask_prefixes[:t], mask_prefixes[t:v], mask_prefixes[v:]
# Prepare npy_paths
npy_path = {
    'train': os.path.join(npy_name['dir'], npy_name['train']),
    'val': os.path.join(npy_name['dir'], npy_name['val']),
    'test': os.path.join(npy_name['dir'], npy_name['test'])
}
# Create the required .csv file for train/val/test set
np.savetxt(npy_path['train'], train, delimiter='\n', fmt='%s')
np.savetxt(npy_path['val'], val, delimiter='\n', fmt='%s')
np.savetxt(npy_path['test'], test, delimiter='\n', fmt='%s')
