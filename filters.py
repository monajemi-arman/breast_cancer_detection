import os
import cv2
import json
import numpy as np
import argparse
from pathlib import Path
from PIL import Image, ImageFilter

class ImageFilterProcessor:
    def __init__(self, config_file="filter_config.json"):
        self.filters = {
            "canny": self.filter_canny,
            "clahe": self.filter_clahe,
            "gamma": self.filter_gamma,
            "histogram": self.filter_histogram_normalization,
            "unsharp": self.filter_unsharp_mask
        }
        self.config = self.load_config(config_file)

    def load_config(self, config_file):
        try:
            with open(config_file, "r") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"Error loading config file: {config_file}. Using default parameters.")
            return {}

    def get_filter_params(self, filter_name):
        return self.config.get(filter_name, {})

    def filter_canny(self, image, sigma=0.33):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_enhanced = cv2.equalizeHist(image)
        image_blurred = cv2.GaussianBlur(image_enhanced, (5, 5), 0)
        v = np.median(image_blurred)
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        edges = cv2.Canny(image_blurred, lower, upper)
        return edges

    def filter_clahe(self, image, clip_limit=2.0, tile_grid_size=(8, 8)):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        return clahe.apply(image)

    def filter_gamma(self, image, gamma=1.5):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        inv_gamma = 1.0 / gamma
        table = np.array([(i / 255.0) ** inv_gamma * 255 for i in np.arange(0, 256)]).astype("uint8")
        return cv2.LUT(image, table)

    def filter_histogram_normalization(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bgr_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        yuv_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2YUV)
        yuv_image[:, :, 0] = cv2.equalizeHist(yuv_image[:, :, 0])
        return cv2.cvtColor(yuv_image, cv2.COLOR_YUV2BGR)

    def filter_unsharp_mask(self, image, radius=2, percent=150, threshold=3):
        if len(image.shape) == 3:
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = Image.fromarray(image)
        unsharp_image = pil_image.filter(ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold))
        return cv2.cvtColor(np.array(unsharp_image), cv2.COLOR_RGB2BGR)

    def apply_filter(self, filter_name, image):
        if filter_name not in self.filters:
            print("Filter not found")
            return None
        filter_func = self.filters[filter_name]
        params = self.get_filter_params(filter_name)
        return filter_func(image, **params)

    def process_images_in_folder(self, input_folder, output_folder, filter_name):
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        for image_file in input_path.glob("*.[jp][pn]*[g]"):
            image = cv2.imread(str(image_file))
            if image is None:
                print(f"Could not read image {image_file}")
                continue

            filtered_image = self.apply_filter(filter_name, image)
            if filtered_image is not None:
                output_file = output_path / image_file.name
                cv2.imwrite(str(output_file), filtered_image)
                print(f"Saved filtered image to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Apply a filter to all images in a folder")
    parser.add_argument("-i", "--input", required=True, help="Input folder containing images")
    parser.add_argument("-o", "--output", required=True, help="Output folder to save filtered images")
    parser.add_argument("-f", "--filter", required=True, help="Filter name to apply (canny, clahe, gamma, histogram, unsharp)")

    args = parser.parse_args()

    processor = ImageFilterProcessor()
    processor.process_images_in_folder(args.input, args.output, args.filter)

if __name__ == "__main__":
    main()
