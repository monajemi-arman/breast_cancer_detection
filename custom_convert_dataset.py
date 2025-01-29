#!/usr/bin/env python
import json
import os.path
import zipfile
from pathlib import Path
from PIL import Image, ImageOps
import pydicom
import numpy as np
from io import BytesIO
from PIL import Image
from pydicom.pixel_data_handlers.util import apply_voi_lut

input_dataset_path = 'datasets/custom'
output_dataset_path = 'custom-dataset'
image_ext = '.jpg'
input_images_path = os.path.join(input_dataset_path, 'images')
input_labels_path = os.path.join(input_dataset_path, 'labels')
output_images_path = os.path.join(output_dataset_path, 'images')
output_images_path = os.path.join(output_dataset_path, 'labels')

counter = 1

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
    original_id = Path(zip_path).stem
    with zipfile.ZipFile(zip_path, 'r') as zip_file:
        for file_info in zip_file.infolist():
            # Process files without extensions (assuming they're DICOM)
            if '.' not in file_info.filename:
                image_name = file_info.filename

                # Process corresponding label
                if process_label(original_id, image_name, image_id):
                    # If successful, process and save the image
                    try:
                        with zip_file.open(file_info.filename) as file:
                            data = file.read()
                            image_data = load_image_bytes(data).astype(np.uint8)
                            image = Image.fromarray(image_data)

                            # Save image
                            output_path = os.path.join(output_images_path, image_id + image_ext)
                            image.save(output_path)

                            image_id += 1

                    except Exception as e:
                        print(f"Could not process {file_info.filename} in {zip_path}: {e}")
    return image_id

def process_label(original_id, image_name, image_id):
    json_path = os.path.join(original_id, image_name + '.json')
    with open(json_path) as f:
        json_data = json.load(f)
    pass


def process_directory(directory_path):
    image_id = 1
    for filename in os.listdir(directory_path):
        filepath = os.path.join(directory_path, filename)
        if os.path.isfile(filepath) and filename.lower().endswith('.zip'):
            try:
                image_id = process_zip_file(filepath, image_id)
            except Exception as e:
                print(f"Could not process {filename}: {e}")


def read_custom_dicom(file_path):
    image = read_dicom(file_path)
    image = ImageOps.invert(image)
    return image

def read_dicom(file_path):
    dicom_data = pydicom.dcmread(file_path, force=True)
    pixel_array = dicom_data.pixel_array.astype(np.float32)

    rescale_slope = float(dicom_data.get("RescaleSlope", 1))
    rescale_intercept = float(dicom_data.get("RescaleIntercept", 0))

    # Handle MultiValue for WindowCenter and WindowWidth
    window_center = dicom_data.get("WindowCenter", np.mean(pixel_array))
    if isinstance(window_center, pydicom.multival.MultiValue):
        window_center = float(window_center[0])
    else:
        window_center = float(window_center)

    window_width = dicom_data.get("WindowWidth", np.max(pixel_array) - np.min(pixel_array))
    if isinstance(window_width, pydicom.multival.MultiValue):
        window_width = float(window_width[0])
    else:
        window_width = float(window_width)

    hu_pixels = (pixel_array * rescale_slope) + rescale_intercept
    min_window = window_center - (window_width / 2)
    max_window = window_center + (window_width / 2)

    normalized = np.clip((hu_pixels - min_window) / (max_window - min_window), 0, 1)

    image_array = (normalized * 255).astype(np.uint8)

    image = Image.fromarray(image_array)

    return image


def main():
    pass

if __name__ == '__main__':
    main()