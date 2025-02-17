import os
import cv2
import json
import argparse
import numpy as np
from prompt_toolkit import Application
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout, HSplit, VSplit, Window
from prompt_toolkit.widgets import Frame, TextArea, RadioList, Button
from prompt_toolkit.filters import IsDone
from prompt_toolkit.styles import Style
from prompt_toolkit.layout.controls import FormattedTextControl

def draw_yolo_bboxes(image, bbox, color, label=None):
    img_h, img_w, _ = image.shape
    class_id, x_center, y_center, width, height = map(float, bbox)
    x_min = int((x_center - width / 2) * img_w)
    y_min = int((y_center - height / 2) * img_h)
    x_max = int((x_center + width / 2) * img_w)
    y_max = int((y_center + height / 2) * img_h)
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 8)
    if label:
        cv2.putText(image, label, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def draw_coco_bboxes(image, bbox, color, label=None):
    x, y, width, height = map(int, bbox)
    cv2.rectangle(image, (x, y), (x + width, y + height), color, 8)
    if label:
        cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

def draw_mask(image, mask, color):
    mask_overlay = np.zeros(image.shape, dtype=np.uint8)
    mask_overlay[mask > 0] = color
    cv2.addWeighted(image, 1, mask_overlay, 0.5, 0, image)

def resize_to_half_screen(image):
    screen_width, screen_height = 1920, 1080
    img_h, img_w = image.shape[:2]
    scaling_factor = min(screen_width / 2 / img_w, screen_height / 2 / img_h, 1.0)
    new_size = (int(img_w * scaling_factor), int(img_h * scaling_factor))
    return cv2.resize(image, new_size) if scaling_factor < 1 else image

def visualize_yolo(image_path, label_path, class_names):
    image = cv2.imread(image_path)
    with open(label_path, 'r') as file:
        labels = file.readlines()
    for label in labels:
        label = label.strip().split()
        class_id = int(label[0])
        bbox = label[1:]
        color = (0, 255, 0)  # Green for YOLO boxes
        draw_yolo_bboxes(image, [class_id] + bbox, color, class_names[class_id])
    return resize_to_half_screen(image)

def visualize_coco(image_path, annotation, class_names):
    image = cv2.imread(image_path)
    for obj in annotation:
        bbox = obj['bbox']
        class_id = obj['category_id']
        color = (0, 0, 255)  # Red for COCO boxes
        if len(class_names) == 1 and class_id > 0:
            class_id -= 1
        draw_coco_bboxes(image, bbox, color, class_names[class_id])
    return resize_to_half_screen(image)

def visualize_mask(image_path, mask_path):
    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask.shape[:2] != image.shape[:2]:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
    draw_mask(image, mask, (0, 255, 0))  # Green for mask overlay
    return resize_to_half_screen(image)

def display_image(dataset_path, dataset_type, label_path, class_names, selected_file, mask_suffix='.jpg'):
    if selected_file:
        image_path = os.path.join(dataset_path, selected_file)
        if dataset_type == 'yolo':
            label_file = os.path.splitext(selected_file)[0] + ".txt"
            label_full_path = os.path.join(label_path, label_file)
            if not os.path.exists(label_full_path):
                print(f"Label file {label_full_path} not found!")
                return
            image = visualize_yolo(image_path, label_full_path, class_names)
        elif dataset_type == 'coco':
            with open(label_path, 'r') as file:
                annotations = json.load(file)
            image_id = next((img['id'] for img in annotations['images'] if img['file_name'] == selected_file), None)
            if image_id is None:
                print(f"Image {selected_file} not found in COCO annotations!")
                return
            image_annotations = [ann for ann in annotations['annotations'] if ann['image_id'] == image_id]
            image = visualize_coco(image_path, image_annotations, class_names)
        elif dataset_type == 'mask':
            mask_file = os.path.splitext(selected_file)[0] + mask_suffix
            mask_path = os.path.join(label_path, mask_file)
            if not os.path.exists(mask_path):
                print(f"Mask file {mask_path} not found!")
                return
            image = visualize_mask(image_path, mask_path)

        cv2.imshow('Visualization', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def interactive_file_browser(dataset_path, dataset_type, label_path, class_names):
    image_files = sorted([f for f in os.listdir(dataset_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

    if not image_files:
        print("No image files found in the dataset directory.")
        return None

    search_field = TextArea(height=1, prompt='Search: ', style='class:search-field')
    radio_list = RadioList([(f, f) for f in image_files])

    def show_image():
        selected_file = radio_list.current_value
        if selected_file:
            display_image(dataset_path, dataset_type, label_path, class_names, selected_file)

    def update_radio_list():
        query = search_field.text.strip().lower()
        filtered_files = [f for f in image_files if query in f.lower()]
        radio_list.values = [(f, f) for f in filtered_files]

    show_button = Button(text="Show", handler=show_image)

    kb = KeyBindings()

    @kb.add('tab')
    def _(event):
        if search_field.buffer == event.app.layout.current_buffer:
            event.app.layout.focus(radio_list)
        elif event.app.layout.has_focus(radio_list):
            event.app.layout.focus(show_button)
        else:
            event.app.layout.focus(search_field.buffer)

    @kb.add('enter', filter=~IsDone())
    def _(event):
        if event.app.layout.has_focus(radio_list):
            show_image()
        else:
            update_radio_list()

    @kb.add('c-c')
    def _(event):
        event.app.exit()

    instructions = FormattedTextControl("Press Tab to switch focus, Enter to search/show image, Ctrl-C to exit")

    root_container = HSplit([
        Window(instructions, height=1),
        Frame(radio_list, title="Image Files"),
        VSplit([
            Frame(search_field, width=40),
            Frame(show_button)
        ])
    ])

    layout = Layout(root_container)

    style = Style.from_dict({
        'frame.label': '#ffffff bold',
        'frame.border': '#888888',
    })

    app = Application(
        layout=layout,
        key_bindings=kb,
        style=style,
        full_screen=True,
        mouse_support=True
    )

    app.run()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Visualize bounding boxes or masks on images for YOLO, COCO, or mask datasets.')
    parser.add_argument('-m', '--mode', choices=['yolo', 'coco', 'mask'], required=True, help="Dataset mode: 'yolo', 'coco', or 'mask'")
    parser.add_argument('-d', '--dataset-path', type=str, required=True, help='Path to the folder containing images')
    parser.add_argument('-l', '--label-path', type=str, required=True, help='Path to annotations folder or JSON file')
    args = parser.parse_args()

    class_names = ["low_mass", "high_mass", "else"]
    interactive_file_browser(args.dataset_path, args.mode, args.label_path, class_names)