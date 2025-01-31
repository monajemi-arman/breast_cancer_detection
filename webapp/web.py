import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from infer import infer
from waitress import serve

# Host and port for waitress server
host = '0.0.0.0'
port = 33517

app = Flask(__name__)

# Enable CORS and allow all hosts
CORS(app, resources={r"/*": {"origins": "*"}})

# Allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'nrrd'}


def allowed_file(filename):
    """Validate allowed file types."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def draw_boxes(image, annotations):
    draw = ImageDraw.Draw(image)
    line_width = max(1, int(min(image.width, image.height) * 0.005))

    font_size = int(min(image.width, image.height) * 0.03)  # 3% of smaller dimension
    font = None

    try_fonts = ["arial.ttf", "DejaVuSans.ttf", "LiberationSans.ttf"]
    for font_name in try_fonts:
        try:
            font = ImageFont.truetype(font_name, font_size)
            break
        except IOError:
            continue

    if font is None:
        font = ImageFont.load_default()
        print("Warning: Using non-scalable default font. For better results, install a TTF font.")

    for ann in annotations:
        x, y, width, height = ann['bbox']

        draw.rectangle([x, y, x + width, y + height], outline="red", width=line_width)

        label = str(ann['category_id'])

        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]

        text_x = x
        text_y = y - text_height - 2  # 2px padding
        if text_y < 0:
            text_y = y  # Move inside if at top edge

        # Draw label background
        draw.rectangle(
            [text_x, text_y, text_x + text_width, text_y + text_height],
            fill="red"
        )

        # Draw label text
        draw.text((text_x, text_y), label, fill="white", font=font)

    return image


def process_image(file, gt_file=None, infer_model=True):
    """
    Process the uploaded image, optionally process ground truth,
    and perform inference.
    """
    img = Image.open(file)
    img = img.convert('RGB')

    # Encode original image to base64
    output = io.BytesIO()
    img.save(output, format='JPEG')
    image_data_orig = base64.b64encode(output.getvalue()).decode('utf-8')

    # Process ground truth annotations
    annotations = []
    gt_data = None
    if gt_file and gt_file.filename.endswith('.json'):
        gt_json = json.load(gt_file)
        for image in gt_json['images']:
            if image['file_name'] == file.filename:
                image_id = image['id']

        image_annotations = [
            ann for ann in gt_json['annotations']
            if ann['image_id'] == image_id
        ]
        annotations = image_annotations

    # Draw ground truth if annotations exist
    if annotations:
        img_gt = draw_boxes(img.copy(), annotations)
        gt_output = io.BytesIO()
        img_gt.save(gt_output, format='JPEG')
        gt_data = base64.b64encode(gt_output.getvalue()).decode('utf-8')

    # Perform inference if required
    inferred_data = None
    if infer_model:
        image_array = np.array(img)
        inferred_array, predictions = infer(image_array, details=True)
        img_inferred = Image.fromarray(inferred_array)
        inferred_output = io.BytesIO()
        img_inferred.save(inferred_output, format='JPEG')
        inferred_data = base64.b64encode(inferred_output.getvalue()).decode('utf-8')

    return {
        "original_image": image_data_orig,
        "ground_truth_image": gt_data,
        "inferred_image": inferred_data,
        "predictions": predictions
    }


# Health Check Endpoint
@app.route('/api/v1/health', methods=['GET'])
def health():
    return jsonify({"status": "success", "message": "API is running"}), 200


# Prediction Endpoint
@app.route('/api/v1/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400

    file = request.files['file']
    gt_file = request.files.get('gt_file')

    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    if file and allowed_file(file.filename):
        try:
            result = process_image(file, gt_file)
            response = {
                "status": "success",
                "message": "Inference successful",
                "data": {
                    "original_image": result['original_image'],
                    "ground_truth_image": result['ground_truth_image'],
                    "inferred_image": result['inferred_image'],
                    "predictions": result['predictions']
                }
            }
            return jsonify(response), 200
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify({"status": "error", "message": "Invalid file format"}), 400


# Ground Truth Visualization Endpoint
@app.route('/api/v1/ground-truth', methods=['POST'])
def ground_truth():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400

    file = request.files['file']
    gt_file = request.files.get('gt_file')

    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    if file and allowed_file(file.filename):
        try:
            result = process_image(file, gt_file, infer_model=False)
            response = {
                "status": "success",
                "message": "Ground truth processed successfully",
                "data": {
                    "original_image": result['original_image'],
                    "ground_truth_image": result['ground_truth_image']
                }
            }
            return jsonify(response), 200
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    return jsonify({"status": "error", "message": "Invalid file format"}), 400


# Front-end Endpoint
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']
        gt_file = request.files.get('gt_file')

        if file.filename == '':
            return render_template('index.html', message='No selected file')

        if file and allowed_file(file.filename):
            result = process_image(file, gt_file)
            return render_template(
                'index.html',
                message='File uploaded successfully',
                image_data_orig=result['original_image'],
                gt_data=result['ground_truth_image'],
                image_data=result['inferred_image'],
                predictions=result['predictions']
            )

    return render_template('index.html')


if __name__ == '__main__':
    serve(app.wsgi_app, host=host, port=port)
