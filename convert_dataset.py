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
mask_out_dir = 'masks'
json_out = 'annotations.json'
json_or_mask_choice = 'json'.lower()  # json/mask/both
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
if json_or_mask_choice == 'json':
    mask_out_dir = None
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

for directory in [image_out_dir, mask_out_dir]:
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
    # Get every ROI and save in rois[]
    rois = []
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
                rois.append(roi)

    if len(rois) > 0:
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
        if json_or_mask_choice != 'json':
            # Create mask
            mask = np.zeros(pixel_array.shape, dtype=np.uint8)
            mask = cv2.fillPoly(mask, rois, 255)
            mask = Image.fromarray(mask)
            mask.save(os.path.join(mask_out_dir, dcm_prefix + '.png'), format='PNG')
        if json_or_mask_choice != 'mask':
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
            # Add annotations to JSON
            for roi in rois:
                json_data['annotations'].append({
                    'id': annotation_id,
                    'image_id': image_id,
                    'category_id': None,
                    'segmentation': roi,
                    'bbox': None
                })

if json_data:
    with open(json_out, 'w') as f:
        json.dump(json_data, f)
