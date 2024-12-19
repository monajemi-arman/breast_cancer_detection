from radiomics import featureextractor, logging as radiomics_logging
import SimpleITK as sitk
import ast
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import os
import re
import xmltodict
from pydicom import dcmread

# ** DEPRECATION WARNING **
# This script uses PyRadiomics, a python module that as of this date cannot be installed on latest Python.
# If you want to use this script you will either have to use older python version (~3.10), OR
# Use the docker image radiomics/pyradiomics:latest and just install opencv-python, pydicom, and xmltodict on the docker

# Low high mode
low_high_mode = True
chosen_classes = ['mass']

# INBreast Dataset paths
inbreast_path = os.path.join('datasets', 'INbreast Release 1.0')
inbreast_xml_dir = os.path.join(inbreast_path, 'AllXML')
inbreast_dcm_dir = os.path.join(inbreast_path, 'AllDICOMs')
inbreast_csv = os.path.join(inbreast_path, 'INbreast.csv')
output_csv = os.path.join(inbreast_path, 'radiomics_features.csv')

# Initialize PyRadiomics feature extractor
extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.enableAllFeatures()  # Enable all features initially
extractor.enabledFeatures['shape'] = []  # Disable 3D shape features
extractor.enabledFeatures['shape2D'] = []  # Enable 2D shape features

# Reduce verbosity of PyRadiomics logging
radiomics_logging.getLogger('radiomics').setLevel(radiomics_logging.ERROR)

def main():
    # Initialize variables
    all_classes = []
    if 'mass' in chosen_classes:
        if low_high_mode:
            all_classes.extend(['mass_low', 'mass_high'])
        else:
            all_classes.append('mass')
    if 'calcification' in chosen_classes:
        all_classes.append('calcification')
    inbreast_classes = set()
    inbreast_xmls = [str(x) for x in list(Path(inbreast_xml_dir).glob('*.xml'))]  # Load XML paths

    # Read CSV for malignant/benign
    inbreast_csv_data = pd.read_csv(inbreast_csv, sep=';')
    file_score_pairs = {}
    for filename, score in zip(inbreast_csv_data['File Name'], inbreast_csv_data['Bi-Rads']):
        filename = str(filename)
        score = int(re.sub(r'\D', '', score))  # 4a, 4b, 4c => 4
        file_score_pairs[filename] = score

    feature_results = []
    for inbreast_xml in inbreast_xmls:
        # Read XML to Dict
        with open(inbreast_xml) as f:
            xml_data = f.read()
        xml_dict = xmltodict.parse(xml_data)
        entries = xml_dict['plist']['dict']['array']['dict']['array']['dict']
        if not isinstance(entries, list):
            entries = [entries]
        # Prepare rois dict/list
        rois = {cls: [] for cls in chosen_classes}
        for entry in entries:
            class_name = entry['string'][1]
            if class_name:
                class_name = class_name.lower()
            inbreast_classes.add(class_name)
            if class_name in chosen_classes:
                roi = entry['array'][1]['string']
                if isinstance(roi, list):
                    roi = [ast.literal_eval(point) for point in roi]
                elif isinstance(roi, str):
                    roi = ast.literal_eval(roi)
                if len(roi) > 2:
                    roi = np.array(roi, dtype=np.int32)
                    rois[class_name].append(roi)

        if any(rois.values()):
            dcm_prefix = Path(inbreast_xml).stem
            patient_dir = None
            for filename in os.listdir(inbreast_dcm_dir):
                if re.match(dcm_prefix + '.*\.dcm', filename):
                    patient_dir = os.path.join(inbreast_dcm_dir, filename)
                    break
            if not patient_dir:
                raise Exception('The following DICOM file was not found in the directory: ' + dcm_prefix)

            cls_suffix = ''
            if low_high_mode:
                if dcm_prefix in file_score_pairs:
                    score = file_score_pairs[dcm_prefix]
                    cls_suffix = '_low' if score <= 3 else '_high'

            pixel_array = dcmread(patient_dir).pixel_array
            pixel_array = normalize_with_threshold(pixel_array) * 255
            pixel_array = np.uint8(pixel_array)

            for cls in chosen_classes:
                for i, roi in enumerate(rois[cls]):
                    # Create mask for the ROI with label 1
                    roi_mask = np.zeros_like(pixel_array, dtype=np.uint8)
                    roi_mask = cv2.fillPoly(roi_mask, [roi], 1)

                    # Convert to SimpleITK images
                    sitk_image = sitk.GetImageFromArray(pixel_array)
                    sitk_mask = sitk.GetImageFromArray(roi_mask)

                    # Set the same spacing for image and mask (if known, otherwise defaults are used)
                    sitk_image.SetSpacing((1.0, 1.0))
                    sitk_mask.SetSpacing((1.0, 1.0))

                    # Extract radiomics features
                    result = extractor.execute(sitk_image, sitk_mask)
                    result['class'] = cls + cls_suffix
                    result['roi_index'] = i
                    result['dcm_prefix'] = dcm_prefix
                    feature_results.append(result)

    # Save features to CSV
    pd.DataFrame(feature_results).to_csv(output_csv, index=False)

def normalize(pixel_array):
    pixel_range = pixel_array.max() - pixel_array.min()
    return (pixel_array - pixel_array.min()) / pixel_range

def normalize_with_threshold(pixel_array, threshold=(1000, 2000)):
    pixel_array[pixel_array < threshold[0]] = threshold[0]
    pixel_array[pixel_array > threshold[1]] = threshold[1]
    return normalize(pixel_array)

if __name__ == '__main__':
    main()
