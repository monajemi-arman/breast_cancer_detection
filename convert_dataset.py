#!/usr/bin/env python
# Convert dataset DICOM and XML to images and masks directory
import os
import random
import sys
import time
import re
from pathlib import Path
import ast
import cv2
from pydicom import dcmread
import xmltodict
import numpy as np
from PIL import Image
import json
import yaml
from csv import DictReader
import shutil
import pandas as pd
import argparse
from utils import yolo_to_coco

# -- How to Use --- #
# The folder in which this script is located in must contain:
# "datasets/INbreast Release 1.0", "datasets/CBIS-DDSM", "datasets/all-mias"
# After running the script, images/, labels/, and dataset.yaml is created for yolo format, json files for coco

# ** Limitations **
# - Mask mode not supported for MIAS dataset, yet
# - Mask mode currently does not support benign/malignant mode (low/high risk)
# Features:
# - YOLO and COCO style dataset output for INBreast, CBIS-DDSM, and MIAS dataset
# - Mass low and high based on Bi-Rads/Malignant or benign

# --- Parameters --- #
# Change as necessary
chosen_datasets = ['inbreast', 'cbis-ddsm', 'mias']  # Available options: 'inbreast', 'cbis-ddsm', 'mias'
# Classes chosen for segmentation
chosen_classes = ['mass']  # Available options: 'mass', 'calcification'
# Recommended: YOLO
# Hacky point: YOLO mode now generates annotations.json for COCO style as well
output_choice = 'yolo'  # available modes: yolo, mask
# Use Bi-Rads or not; When True, adds 'mass_low' and 'mass_high' to class names
low_high_mode = True
# Train / Validation / Test split ratio
split_mode = True
split_ratio = [0.8, 0.1, 0.1]
split_not_in_mask_mode = True  # Must be set to true for to_3d_nrrd.py to work properly

# --- Input paths --- #
# INBreast Dataset
inbreast_path = os.path.join('datasets', 'INbreast Release 1.0')
inbreast_xml_dir = os.path.join(inbreast_path, 'AllXML')
inbreast_dcm_dir = os.path.join(inbreast_path, 'AllDICOMs')
inbreast_csv = os.path.join(inbreast_path, 'INbreast.csv')
# CBIS-DDSM Dataset
cbis_path = os.path.join('datasets', 'CBIS-DDSM')
cbis_jpeg = os.path.join(cbis_path, 'jpeg')
cbis_csv = os.path.join(cbis_path, 'csv')
# MIAS Dataset
mias_path = os.path.join('datasets', 'all-mias')
mias_info = os.path.join(mias_path, 'Info.txt')
mias_chosen = {'mass': ['CIRC', 'SPIC', 'MISC']}
# Output paths
image_out_dir = 'images'  # Images
mask_out_dir = 'masks'  # Mask
yaml_out = 'dataset.yaml'  # YOLO .yaml
txt_out_dir = 'labels'  # YOLO labels .txt
split_dirs = ['train', 'val', 'test']
# --- End of Input paths --- #

# Remove boxes smaller than this amount in length of X or Y
bbox_length_threshold = 0.005

# --- End of Parameters --- #

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('-m', '--mode', type=str, choices=['yolo', 'mask', 'coco'],
                    help="Output dataset style", required=False, default=output_choice)
args = parser.parse_args()
output_choice = args.mode
output_choice = output_choice.lower()

# Skip not implemented features
if output_choice == 'mask':
    # Faulty masks are made due to wrong names
    if os.path.exists(image_out_dir):
        print("Directory 'images' must be removed before running this program in mask mode!")
        choice = input("Should images/ directory be removed? (Y/n)").lower()
        if choice == 'y':
            shutil.rmtree(image_out_dir)
        else:
            raise Exception("Cannot continue with images/ existing!")

    # MIAS dataset does not currently support mask output
    if 'mias' in chosen_datasets:
        print("[!] MIAS dataset mask output is not implemented yet, skipping the dataset...", file=sys.stderr)
        chosen_datasets.remove('mias')
    # Mask mode train/val/test split is not desired, therefore disabled unless chosen otherwise
    if split_not_in_mask_mode:
        split_mode = False

# Lazy code, COCO mode deprecated, YOLO mode also does COCO
if output_choice == 'coco':
    output_choice = 'yolo'

# Overall Counters for ID
image_id = 0

def normalize(pixel_array):
    pixel_range = pixel_array.max() - pixel_array.min()
    return (pixel_array - pixel_array.min()) / pixel_range

def normalize_with_threshold(pixel_array, threshold=(1000, 2000)):
    pixel_array[pixel_array < threshold[0]] = threshold[0]
    pixel_array[pixel_array > threshold[1]] = threshold[1]
    return normalize(pixel_array)

# all_classes is a list of all class names, for later reference and assigning IDs to class names
all_classes = []
if 'mass' in chosen_classes:
    if low_high_mode == True:
        all_classes.extend(['mass_low', 'mass_high'])
    else:
        all_classes.append('mass')
if 'calcification' in chosen_classes:
    all_classes.append('calcification')

# ** Deprecated code **
# json_data is deprecated variable, but code is still not removed. It doesn't interfere with the program.
json_data = None  # Prevent undefined error

# Create output json and/or dirs
if output_choice == 'coco':
    # Prevent unnecessary folders being created
    mask_out_dir = None
    txt_out_dir = None
    json_data = {
        'categories': [],
        'annotations': [],
        'images': []
    }
    # Prepare JSON categories
    for cls in all_classes:
        json_data['categories'].append({
            'id': all_classes.index(cls),
            'name': cls
        })
if output_choice == 'yolo':
    mask_out_dir = None

for directory in [image_out_dir, mask_out_dir, txt_out_dir]:
    if directory and not os.path.isdir(directory):
        os.mkdir(directory)

# Prevent Undefined
txt_lines = []

# --- INBreast --- #

inbreast_classes = set()
inbreast_xmls = [str(x) for x in list(Path(inbreast_xml_dir).glob('*.xml'))]  # Load XML paths

if 'inbreast' in chosen_datasets:
    # Read CSV for malignant/benign
    inbreast_csv_data = pd.read_csv(inbreast_csv, sep=';')
    file_score_pairs = {}
    for filename, score in zip(inbreast_csv_data['File Name'], inbreast_csv_data['Bi-Rads']):
        filename = str(filename)
        score = int(re.sub(r'\D', '', score))  # 4a, 4b, 4c => 4
        file_score_pairs[filename] = score

    for inbreast_xml in inbreast_xmls:
        image_id += 1
        # Read XML to Dict
        with open(inbreast_xml) as f:
            xml_data = f.read()
        xml_dict = xmltodict.parse(xml_data)
        entries = xml_dict['plist']['dict']['array']['dict']['array']['dict']
        if not isinstance(entries, list):
            entries = [entries]
        # Prepare rois dict/list
        rois = {}
        for cls in chosen_classes:
            rois[cls] = []
        # Get every ROI and save in rois[]
        for entry in entries:
            class_name = entry['string'][1]
            if class_name:
                class_name = class_name.lower()
            inbreast_classes.add(class_name)
            if class_name in chosen_classes:
                # Literal eval in order to convert string of list to actual list
                roi = entry['array'][1]['string']
                if isinstance(roi, list):
                    roi_tmp = []
                    for point in roi:
                        roi_tmp.append(ast.literal_eval(point))
                    roi = roi_tmp
                elif isinstance(roi, str):
                    roi = ast.literal_eval(roi)
                if len(roi) > 2:
                    roi = np.array(roi, dtype=np.int32)  # Required for later cv2.fillPoly
                    rois[class_name].append(roi)

        if any(rois.values()):
            # Get corresponding DICOM image prefix in name
            dcm_prefix = Path(inbreast_xml).stem
            patient_dir = None
            for filename in os.listdir(inbreast_dcm_dir):
                if re.match(dcm_prefix + '.*\.dcm', filename):
                    patient_dir = os.path.join(inbreast_dcm_dir, filename)
                    break
            if not patient_dir:
                raise Exception('The following DICOM file was not found in the directory: ' + dcm_prefix)
            # Identify low or high and set the full class name
            cls_suffix = ''
            if low_high_mode:
                if dcm_prefix in file_score_pairs:
                    score = file_score_pairs[dcm_prefix]
                    if score <= 3:
                        cls_suffix = '_low'
                    elif score > 3:
                        cls_suffix = '_high'
            # Extract image from DICOM in dcm_path
            # Read pixels from DICOM, convert tp 0-255 range for JPEG
            pixel_array = dcmread(patient_dir).pixel_array
            pixel_array = normalize_with_threshold(pixel_array) * 255
            pixel_array = np.uint8(pixel_array)
            image = Image.fromarray(pixel_array)
            jpeg_path = os.path.join(image_out_dir, dcm_prefix + '.jpg')
            if not os.path.exists(jpeg_path):
                image.save(jpeg_path, format='JPEG')
            if output_choice == 'mask':
                # Mask mode
                mask = np.zeros(pixel_array.shape, dtype=np.uint8)
                for cls in chosen_classes:
                    mask = cv2.fillPoly(mask, rois[cls], 255 - all_classes.index(cls + cls_suffix))
                mask = Image.fromarray(mask)
                mask.save(os.path.join(mask_out_dir, dcm_prefix + '.jpg'), format='JPEG')
            if output_choice == 'coco' or output_choice == 'yolo':
                if output_choice == 'coco':
                    # Get image creation date
                    image_date = time.gmtime(os.path.getctime(jpeg_path))
                    image_date = '{}/{}/{} {}:{}:{}'.format(image_date.tm_year, image_date.tm_mon, image_date.tm_mday,
                                                            image_date.tm_hour, image_date.tm_min, image_date.tm_sec)
                    # Add image to JSON
                    json_data['images'].append({
                        'id': image_id,
                        'file_name': jpeg_path,
                        'width': image.width,
                        'height': image.height,
                        'date_captured': image_date
                    })
                # Add annotations with YOLO or COCO style
                txt_lines = []  # For YOLO
                for cls in rois.keys():
                    # Identify low or high and set the full class name
                    cls_suffix = ''
                    # Currently only doing low and high for mass segments
                    if low_high_mode == True:
                        if cls == 'mass':
                            if dcm_prefix in file_score_pairs:
                                score = file_score_pairs[dcm_prefix]
                                if score <= 3:
                                    cls_suffix = '_low'
                                elif score > 3:
                                    cls_suffix = '_high'
                    class_id = all_classes.index(cls + cls_suffix)
                    # Go through each ROI in the iamge
                    for roi in rois[cls]:
                        if output_choice == 'coco':
                            # COCO mode
                            # Bug: Currently, we are not taking into account the polygons with disconnected parts.
                            # They may not be present in INBreast dataset, however, it is something that we could not
                            # take into account.
                            # ---
                            # Calculate bounding box
                            roi = roi.astype(int)
                            x_s, y_s = roi[:, 0], roi[:, 1]
                            # Convert all data to simple list of simple integers
                            bbox = np.array([x_s.min(), y_s.min(), x_s.max(), y_s.max()]).tolist()
                            # Relative coord must be lower than 1
                            if any([x > 1 for x in bbox]):
                                raise Exception('Bbox calculated relative coord larger than 1 for ROI: ' + roi)
                            json_data['annotations'].append({
                                'image_id': image_id,
                                'category_id': class_id,
                                'segmentation': [roi.tolist()],
                                'bbox': bbox
                            })
                        elif output_choice == 'yolo':
                            # YOLO mode
                            # Relative Xs and Ys
                            x_s, y_s = roi[:, 0] / image.width, roi[:, 1] / image.height
                            # BBOX in YOLO: X-Center Y-Center Width Height
                            bbox = x_s.mean(), y_s.mean(), x_s.max() - x_s.min(), y_s.max() - y_s.min()
                            # Relative coord must be lower than 1
                            if any([x > 1 for x in bbox]):
                                raise Exception('Bbox calculated relative coord larger than 1 for ROI: ' + roi)
                            bbox = [str(x) for x in bbox]
                            txt_lines.append("{} {} {} {} {}\n".format(str(class_id), *bbox))
        # If YOLO, write TXT labels accumulated for the current image
        if len(txt_lines) > 0:
            txt_path = os.path.join(txt_out_dir, dcm_prefix + '.txt')
            if not os.path.exists(txt_path):
                with open(txt_path, 'w') as f:
                    f.writelines(txt_lines)
        elif output_choice != 'mask':
            raise Exception("Image without mask")

# -- End of INBreast --- #

# --- CBIS-DDSM --- #
if 'cbis-ddsm' in chosen_datasets:
    # dicom_info.csv tells which jpeg corresponds to which dicom originally in DDSM dataset
    # CBIS-DDSM has converted dicom to jpeg and removed the original .dcm files.
    # Creating a dictionary linking previous .dcm to current .jpg files,
    dcm_jpeg_dict = {}
    with open(os.path.join(cbis_csv, 'dicom_info.csv')) as f:
        list_of_dict = list(DictReader(f))
    for item in list_of_dict:
        # Skip cropped images, accept only mammography and ROI
        if 'crop' not in item['SeriesDescription']:
            # patient_dir
            dcm_path = Path(item['file_path'].strip()).parent.parts[-1]
            # patient_dir/jpeg_name.jpg
            jpeg_path = os.path.join(*Path(item['image_path'].strip()).parts[-2:])
            dcm_jpeg_dict[dcm_path] = jpeg_path

    # Load data
    # Paths start with CBIS-DDSM, make sure this is the name of the folder that contains csv and jpeg folders
    cbis_base = str(Path(cbis_path).parent)
    csv_names = [
        'calc_case_description_train_set.csv',
        'calc_case_description_test_set.csv',
        'mass_case_description_test_set.csv',
        'mass_case_description_train_set.csv'
    ]
    # Dictionary with image => mask pairs
    image_mask_pairs = {}
    # Mask path => class id
    mask_class_pairs = {}
    for csv_name in csv_names:
        # Get class_name
        if csv_name[:4].lower() == 'calc':
            class_name = 'calcification'
        elif csv_name[:4].lower() == 'mass':
            class_name = 'mass'
        else:
            raise Exception(
                'Unexpected class name. CSV file prefixes for CBIS-DDSM should only start with calc or mass!')

        # Skip if class is not desired
        if class_name not in chosen_classes:
            continue

        # Load CSV data
        with open(os.path.join(cbis_csv, csv_name)) as f:
            list_of_dict = list(DictReader(f))
        for item in list_of_dict:
            # Separate parent directory for later reference to files (folder structure stuff)
            patient_dir = Path(item['image file path'].strip()).parent.parts[-1]
            # Only support mass for Bi-Rads for now
            cls_suffix = ''
            if low_high_mode:
                score = int(str(item['assessment']).strip())
                if class_name == 'mass':
                    if score <= 3:
                        cls_suffix = '_low'
                    elif score > 3:
                        cls_suffix = '_high'
            # Skip invalid images
            if patient_dir not in dcm_jpeg_dict:
                continue
            # Convert dcm name to actual jpeg name
            jpeg_path = dcm_jpeg_dict[patient_dir]
            if jpeg_path not in image_mask_pairs:
                image_mask_pairs[jpeg_path] = []
            # patient_dir is an arbitrary name pointing to the base folder for jpg files
            patient_dir = Path(item['ROI mask file path'].strip()).parent.parts[-1]
            if patient_dir not in dcm_jpeg_dict:
                continue
            mask_path = dcm_jpeg_dict[patient_dir]
            image_mask_pairs[jpeg_path].append(mask_path)
            # Add mask class
            mask_class_pairs[mask_path] = all_classes.index(class_name + cls_suffix)

    # Mask mode
    # Bug: Multi class not implemented (NOT_IMPLEMENTED)
    image_id = 0
    if output_choice == 'mask':
        for item in image_mask_pairs.items():
            image_id += 1
            image_name = 'cbm_' + str(image_id) + '.jpg'
            shutil.copy(os.path.join(cbis_jpeg, item[0]), os.path.join(image_out_dir, image_name))
            mask_out_path = os.path.join(mask_out_dir, image_name)
            if len(item[1]) == 1:
                mask_path = item[1][0]
                shutil.copy(os.path.join(cbis_jpeg, mask_path), mask_out_path)
            elif len(item[1]) > 1:
                masked = None
                for i in range(len(item[1])):
                    if i < (len(item[1]) - 1):
                        img1_path = os.path.join(cbis_jpeg, item[1][i])
                        img2_path = os.path.join(cbis_jpeg, item[1][i + 1])
                        img1 = Image.open(img1_path)
                        img2 = Image.open(img2_path)
                        img1.paste(img2, (0, 0), mask=img2)
                        masked = img1
                masked.save(mask_out_path)

    # YOLO mode
    image_id = 0
    if output_choice == 'yolo':
        for image_path in image_mask_pairs.keys():
            image_id += 1  # Used only for output name
            output_name = "cb_" + str(image_id)
            txt_lines = []
            for mask_path in image_mask_pairs[image_path]:
                # Read gray mask
                mask = cv2.imread(os.path.join(cbis_jpeg, mask_path), cv2.IMREAD_GRAYSCALE)
                # Apply threshold to mask pixels
                _, mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                # Get contours for label text files
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                # Each element in contours list is a polygon (supposedly)
                for contour in contours:
                    contour = np.array(contour).reshape(-1, 2)  # Fix shape for later use
                    x_s = contour[:, 0]
                    y_s = contour[:, 1]
                    # Relative Xs and Ys
                    x_s, y_s = x_s / mask.shape[1], y_s / mask.shape[0]
                    # BBOX in YOLO: X-Center Y-Center Width Height
                    bbox = x_s.mean(), y_s.mean(), x_s.max() - x_s.min(), y_s.max() - y_s.min()
                    if any([x > 1 for x in bbox]):
                        raise Exception('Bbox calculated relative coord larger than 1 for ROI: ' + str(contour))
                    if bbox[2] >= bbox_length_threshold and bbox[3] >= bbox_length_threshold:
                        bbox = [str(x) for x in bbox]
                        # Get class id
                        class_id = mask_class_pairs[mask_path]
                        txt_lines.append("{} {} {} {} {}\n".format(str(class_id), *bbox))
            # Write label txt file for current image
            # Skip image if txt_lines wasn't filled
            if len(txt_lines) > 0:
                output_path = os.path.join(txt_out_dir, output_name + ".txt")
                if not os.path.exists(output_path):
                    with open(output_path, 'w') as f:
                        f.writelines(txt_lines)
                # Copy image to output dir
                output_path = os.path.join(image_out_dir, output_name + ".jpg")
                if not os.path.exists(output_path):
                    shutil.copy(os.path.join(cbis_jpeg, image_path), output_path)
# --- End of CBIS-DDSM --- #

# --- MIAS --- #
if 'mias' in chosen_datasets:
    with open(mias_info) as f:
        for line in f.readlines():
            line_parts = line.split(' ')
            # Skip irrelevant lines
            if line[:3] != 'mdb' or len(line_parts) < 7:
                continue
            # Image name
            image_name = line_parts[0]
            # Read image
            image = cv2.imread(os.path.join(mias_path, image_name + '.pgm'), cv2.IMREAD_GRAYSCALE)
            # Label txt lines to be filled
            txt_lines = []
            for class_name in chosen_classes:
                if line_parts[2] in mias_chosen[class_name]:
                    # Malignant / benign
                    cls_suffix = ''
                    if low_high_mode:
                        if line_parts[3] == 'M':
                            cls_suffix = '_high'
                        else:
                            cls_suffix = '_low'
                    # Calculate bounding box
                    x_center, y_center, radius = int(line_parts[4]), int(line_parts[5]), int(line_parts[6])
                    # To relative coord
                    height, width = image.shape
                    bbox = [x_center / width, y_center / height, radius / width, radius / height]
                    if any([x > 1 for x in bbox]):
                        raise Exception('Bbox calculated relative coord larger than 1 for ROI: ' + str(contour))
                    # Txt line
                    class_id = all_classes.index(class_name + cls_suffix)
                    bbox = [str(x) for x in bbox]
                    txt_lines.append("{} {} {} {} {}\n".format(str(class_id), *bbox))
            # Skip images with no labels (or with non-chosen labels)
            if len(txt_lines) > 0:
                # Write to labels/
                output_path = os.path.join(txt_out_dir, image_name + ".txt")
                if not os.path.exists(output_path):
                    with open(output_path, 'w') as f:
                        f.writelines(txt_lines)
                # Save to images/
                output_path = os.path.join(image_out_dir, image_name + '.jpg')
                if not os.path.exists(output_path):
                    cv2.imwrite(output_path, image)

# --- End of MIAS --- #

# Train / Val / Test split
if split_mode:
    out_dir = txt_out_dir
    file_suffix = '.txt'
    if output_choice == 'mask':
        out_dir = 'masks'
        file_suffix = '.jpg'
    warning_said = False
    for directory1 in split_dirs:
        if os.path.exists(directory1) and not warning_said:
            print("Train/Val/Test directories exist! You must remove these directories before running the program.\n" + \
                  "Otherwise this may cause random extra images in these directories.", file=sys.stderr)
            warning_said = True
        else:
            os.mkdir(directory1)
        for directory2 in [image_out_dir, out_dir]:
            directory = os.path.join(directory1, directory2)
            if not os.path.exists(directory):
                os.mkdir(directory)
    file_prefixes = [Path(x).stem for x in os.listdir(out_dir)]
    random.shuffle(file_prefixes)
    i = 0
    for file_prefix in file_prefixes:
        i += 1
        # Choose destination
        ratio = i / len(file_prefixes)
        if ratio <= split_ratio[0]:
            dest_id = 0  # train
        elif ratio <= split_ratio[0] + split_ratio[1]:
            dest_id = 1  # val
        elif ratio <= split_ratio[0] + split_ratio[1] + split_ratio[2]:
            dest_id = 2  # test
        else:
            dest = 'train'
        # Move
        dest = os.path.join(split_dirs[dest_id], image_out_dir, file_prefix + '.jpg')
        if not os.path.exists(dest):
            shutil.move(os.path.join(image_out_dir, file_prefix + '.jpg'), dest)
        dest = os.path.join(split_dirs[dest_id], out_dir, file_prefix + file_suffix)
        if not os.path.exists(dest):
            shutil.move(os.path.join(out_dir, file_prefix + file_suffix), dest)

    # Create COCO annotations json regardless of output style choice (yolo/coco)
    if output_choice == 'yolo':
        for directory in split_dirs:
            yolo_to_coco(os.path.join(directory, txt_out_dir), os.path.join(directory, image_out_dir), directory + '.json')
        # Required for YOLO
        yaml_data = {
            'path': os.getcwd(),
            'train': os.path.join(split_dirs[0], image_out_dir),
            'val': os.path.join(split_dirs[1], image_out_dir),
            'test': os.path.join(split_dirs[2], image_out_dir),
            'names': all_classes
        }
        with open(yaml_out, 'w') as f:
            yaml.dump(yaml_data, f)

    # Clean up empty directories
    for directory in [image_out_dir, out_dir]:
        if len(os.listdir(directory)) == 0:
            os.rmdir(directory)
