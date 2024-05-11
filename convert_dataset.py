#!/usr/bin/env python
# Convert dataset DICOM and XML to images and masks directory
import os
import time
import re
from pathlib import Path
import ast
import cv2
import pydicom
import xmltodict
import numpy as np
from PIL import Image
import json

# --- Parameters --- #
# Output paths
image_out_dir = 'images'
mask_out_dir = 'masks' # Mask
json_out = 'annotations.json' # COCO
txt_out_dir = 'labels' # Yolo
output_choice = 'yolo'  # yolo/coco/mask
# Classes chosen for segmentation
chosen_classes = ['Mass']

# Overall Counters for ID
image_id = 0

# INBreast Dataset
# Dir paths
inbreast_path = os.path.join('datasets', 'INbreast Release 1.0/')
inbreast_xml_dir = os.path.join(inbreast_path, 'AllXML')
inbreast_dcm_dir = os.path.join(inbreast_path, 'AllDICOMs')

# --- End of Parameters --- #

# Create output json and/or dirs
output_choice = output_choice.lower()
json_data = None # Prevent undefined error
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
    for cls in chosen_classes:
        json_data['categories'].append({
            'id': chosen_classes.index(cls) + 1,
            'name': cls
        })
if output_choice == 'yolo':
    mask_out_dir = None

for directory in [image_out_dir, mask_out_dir, txt_out_dir]:
    if directory and not os.path.isdir(directory):
        os.mkdir(directory)

inbreast_classes = set()
inbreast_xmls = [str(x) for x in list(Path(inbreast_xml_dir).glob('*.xml'))]  # Load XML paths

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
        for filename in os.listdir(inbreast_dcm_dir):
            if re.match(dcm_prefix + '.*\.dcm', filename):
                dcm_path = os.path.join(inbreast_dcm_dir, filename)
                break
        # Extract image from DICOM in dcm_path
        # Read pixels from DICOM, convert tp 0-255 range for JPEG
        pixel_array = pydicom.read_file(dcm_path).pixel_array.astype(np.uint8)
        image = Image.fromarray(pixel_array)
        jpeg_path = os.path.join(image_out_dir, dcm_prefix + '.jpg')
        if not os.path.exists(jpeg_path):
            image.save(jpeg_path, format='JPEG')
        if output_choice == 'mask':
            # Mask mode
            mask = np.zeros(pixel_array.shape, dtype=np.uint8)
            for cls in chosen_classes:
                mask = cv2.fillPoly(mask, rois[cls], 255 - chosen_classes.index(cls))
            mask = Image.fromarray(mask)
            mask.save(os.path.join(mask_out_dir, dcm_prefix + '.png'), format='PNG')
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
            for cls in rois.keys():
                class_id = chosen_classes.index(cls)
                txt_lines = [] # For YOLO
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
                        bbox = [str(x) for x in bbox]
                        txt_lines.append("{} {} {} {} {}\n".format(str(class_id), *bbox))
    # If YOLO, write TXT labels accumulated for the current image
    if txt_lines and len(txt_lines) > 0:
        with open(os.path.join(txt_out_dir, dcm_prefix + '.txt'), 'w') as f:
            f.writelines(txt_lines)


if json_data:
    with open(json_out, 'w') as f:
        json.dump(json_data, f)
