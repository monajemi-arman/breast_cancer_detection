#!/usr/bin/env python
import json
import os.path
import zipfile
from pathlib import Path
import pydicom
import numpy as np
from io import BytesIO
from pydicom.pixel_data_handlers.util import apply_voi_lut
from utils import read_dicom
from argparse import ArgumentParser

# --- Parameters ---
input_dataset_path = 'datasets/custom'
output_dataset_path = 'custom-dataset'
selected_tags = {'mass_conc_simpCompCyst': 0, 'mass_conc_cmpxBngCyst': 0, 'mass_conc_compxMlgCyst': 1,
                 'mass_conc_bngSldTumor': 0, 'mass_conc_mlgSldTumor': 1}
image_ext = '.jpg'
input_images_path = os.path.join(input_dataset_path, 'images')
input_labels_path = os.path.join(input_dataset_path, 'labels')
output_images_path = os.path.join(output_dataset_path, 'images')
output_labels_path = os.path.join(output_dataset_path, 'labels.json')
# --- End of parameters --

parser = ArgumentParser()
parser.add_argument('--else-labels', dest="ELSE_LABELS_AS_LAST", default=False, action='store_true',
                    help='Generate all labels even if not selected, put them into else')
parsed = parser.parse_args()
ELSE_LABELS_AS_LAST = parsed.ELSE_LABELS_AS_LAST

# For later use...
selected_tags_keys = list(selected_tags.keys())
for directory in output_dataset_path, output_images_path:
    os.makedirs(directory, exist_ok=True)

if ELSE_LABELS_AS_LAST:
    selected_tags_values = list(set(selected_tags.values()))
    selected_tags_values.sort()
    else_id = selected_tags_values[-1] + 1
    selected_tags['else'] = else_id

counter = 1
annotation_id = 0


def load_image_bytes(data):
    global counter
    try:
        dicom_img = pydicom.dcmread(BytesIO(data), force=True)
        img = apply_voi_lut(dicom_img.pixel_array, dicom_img)
        if dicom_img.PhotometricInterpretation == "MONOCHROME1":
            img = np.max(img) - img
        print(f"[{counter}] Loaded DICOM successfully...")
        counter += 1
        return img.astype(np.float32)
    except Exception as e:
        raise ValueError(f"Failed to decode DICOM image: {e}")


def process_zip_file(zip_path, image_id):
    json_data = {'images': [], 'annotations': []}
    original_id = Path(zip_path).stem
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        for file_info in zip_file.infolist():
            # Process files without extensions (assuming they're DICOM)
            if '.' not in file_info.filename:
                image_name = file_info.filename

                # Process corresponding label
                try:
                    with zip_file.open(file_info.filename) as file:
                        data = file.read()
                        image = read_dicom(BytesIO(data))
                        width, height = image.size

                        annotations = process_label(original_id, image_name, image_id, image.size)
                        if annotations:
                            # Save image
                            image_name_new = original_id + '_' + image_name + image_ext
                            output_path = os.path.join(output_images_path, image_name_new)
                            image.save(output_path)

                            # Save JSON data to array
                            json_data['annotations'].extend(annotations)
                            json_data['images'].append({
                                'id': image_id,
                                'file_name': image_name_new,
                                'width': width,
                                'height': height
                            })

                            image_id += 1

                except Exception as e:
                    print(f"Could not process {file_info.filename} in {zip_path}: {e}")
    return image_id, json_data


def process_label(original_id, image_name, image_id, real_shape):
    global annotation_id
    global ELSE_LABELS_AS_LAST

    annotations = []
    json_path = os.path.join(input_labels_path, original_id, image_name + '.json')
    with open(json_path) as f:
        json_data = json.load(f)
        for rectangle in json_data['rectangles']:
            tags = rectangle['tag']
            found_tag = [item in selected_tags_keys for item in tags]
            if any(found_tag) or ELSE_LABELS_AS_LAST:
                # Determine category_id
                category_id = selected_tags[
                    selected_tags_keys[found_tag.index(True)]] if not ELSE_LABELS_AS_LAST else else_id

                # Bounding box coordination
                bbox = rectangle['x'], rectangle['y'], rectangle['width'], rectangle['height']

                # Bbox convert to real
                bbox = bbox_to_real(bbox, real_shape)

                if bbox:
                    area = bbox[2] * bbox[3]

                    annotation_id += 1

                    annotation = {
                        'id': annotation_id,
                        'image_id': image_id,
                        'category_id': category_id,
                        'bbox': bbox,
                        'area': area,
                        'iscrowd': 0
                    }

                    annotations.append(annotation)

    return annotations


def process_directory(directory_path):
    json_data_final = {'categories': [], 'images': [], 'annotations': []}

    json_data_final['categories'].extend([{'id': 0, 'name': 'mass_low'}, {'id': 1, 'name': 'mass_high'}])

    # Quick hack for else category
    if ELSE_LABELS_AS_LAST:
        json_data_final['categories'].append({'id': else_id, 'name': 'else'})

    image_id = 1

    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        if os.path.isfile(filepath) and filename.lower().endswith('.zip'):
            try:
                image_id, json_data = process_zip_file(filepath, image_id)
                json_data_final = merge_dicts(json_data_final, json_data)
            except Exception as e:
                print(f"Could not process {filename}: {e}")

    # Write final JSON
    with open(output_labels_path, 'w') as f:
        json.dump(json_data_final, f)


def bbox_to_real(bbox, real_shape, canvas_shape=(1100, 636)):
    canvas_width, canvas_height = canvas_shape
    real_width, real_height = real_shape
    x, y, width, height = bbox

    # Which dimension is largest
    max_dim = real_shape.index(max(real_shape))
    # Based on that, calculate the ratio with which the image has got smaller
    ratio = canvas_shape[max_dim] / real_shape[max_dim]

    fake_width = ratio * real_width
    fake_height = ratio * real_height

    if max_dim == 1:
        padding = (canvas_width - fake_width) / 2
        real_x = (x - padding) / ratio
        real_y = y / ratio

    else:
        padding = (canvas_width - fake_height) / 2
        real_x = x / ratio
        real_y = (y - padding) / ratio

    real_width, real_height = width / ratio, height / ratio

    real_bbox = [real_x, real_y, real_width, real_height]

    # Sanity check for bbox values
    if any([i < 0 for i in real_bbox]):
        return False
    else:
        return real_bbox


def merge_dicts(dict1, dict2):
    for key, value in dict2.items():
        dict1[key].extend(value)
    return dict1


def main():
    process_directory(input_images_path)


if __name__ == '__main__':
    main()
