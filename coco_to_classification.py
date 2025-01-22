import json
import os
import argparse

# Load the COCO JSON file
def load_coco_json(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# Transform COCO JSON to classification JSON format
def transform_to_classification(coco_data):
    image_map = {img['id']: img['file_name'] for img in coco_data['images']}
    class_data = []

    for annotation in coco_data['annotations']:
        image_id = annotation['image_id']
        category_id = annotation['category_id']
        file_name = image_map.get(image_id)

        if file_name:
            class_data.append({"file_name": file_name, "class_id": category_id})

    return class_data

# Save the new JSON file
def save_classification_json(data, output_path):
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)

# Main function
def main():
    parser = argparse.ArgumentParser(description="Transform COCO JSON to classification JSON format.")
    parser.add_argument("input_json", help="Path to the input COCO JSON file.")
    parser.add_argument("output_json", help="Path to the output classification JSON file.")
    args = parser.parse_args()

    if not os.path.exists(args.input_json):
        print(f"Error: {args.input_json} not found.")
        return

    coco_data = load_coco_json(args.input_json)
    classification_data = transform_to_classification(coco_data)
    save_classification_json(classification_data, args.output_json)

    print(f"Transformed data saved to {args.output_json}")

if __name__ == "__main__":
    main()

