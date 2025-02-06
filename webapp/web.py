import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from infer import Predictor
from waitress import serve

# Host and port for waitress server
host = '0.0.0.0'
port = 33517

app = Flask(__name__)

# Enable CORS and allow all hosts
CORS(app, resources={r"/*": {"origins": "*"}})

# Allowed file extensions
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'nrrd'}

predictor = Predictor()


def allowed_file(filename):
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

        label = str(int(ann['category_id']))

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


def process_image(file, gt_file=None, gt_json=None, infer_model=True):
    global predictor

    img = Image.open(file)
    img = img.convert('RGB')

    output = io.BytesIO()
    img.save(output, format='JPEG')
    image_data_orig = base64.b64encode(output.getvalue()).decode('utf-8')

    annotations = []
    gt_data = None
    if gt_json is not None:
        image_id = None
        for image in gt_json['images']:
            if image['file_name'] == file.filename:
                image_id = image['id']
                break
        if image_id is not None:
            annotations = [ann for ann in gt_json['annotations'] if ann['image_id'] == image_id]
    elif gt_file and gt_file.filename.endswith('.json'):
        gt_json_local = json.load(gt_file)
        image_id = None
        for image in gt_json_local['images']:
            if image['file_name'] == file.filename:
                image_id = image['id']
                break
        if image_id is not None:
            annotations = [ann for ann in gt_json_local['annotations'] if ann['image_id'] == image_id]

    if annotations:
        img_gt = draw_boxes(img.copy(), annotations)
        gt_output = io.BytesIO()
        img_gt.save(gt_output, format='JPEG')
        gt_data = base64.b64encode(gt_output.getvalue()).decode('utf-8')

    inferred_data = None
    predictions = []
    if infer_model:
        image_array = np.array(img)
        inferred_array, preds = predictor.infer(image_array, details=True)
        predictions = preds
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


@app.route('/api/v1/health', methods=['GET'])
def health():
    return jsonify({"status": "success", "message": "API is running"}), 200


@app.route('/api/v1/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400

    files = request.files.getlist('file')
    if len(files) == 0:
        return jsonify({"status": "error", "message": "No selected file"}), 400

    for file in files:
        if not allowed_file(file.filename):
            return jsonify({"status": "error", "message": "Invalid file format"}), 400

    if len(files) == 1:
        file = files[0]
        gt_file = request.files.get('gt_file')
        try:
            result = process_image(file, gt_file=gt_file, infer_model=True)
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
    else:
        gt_file = request.files.get('gt_file')
        gt_json = None
        if gt_file:
            if not gt_file.filename.endswith('.json'):
                return jsonify({"status": "error", "message": "GT file must be a JSON"}), 400
            try:
                gt_json = json.load(gt_file)
            except Exception as e:
                return jsonify({"status": "error", "message": f"Error loading GT JSON: {str(e)}"}), 400

        results = []
        for file in files:
            try:
                result = process_image(file, gt_json=gt_json, infer_model=True)
                results.append({
                    "filename": file.filename,
                    "predictions": result['predictions']
                })
            except Exception as e:
                return jsonify({"status": "error", "message": f"Error processing {file.filename}: {str(e)}"}), 500

        response = {
            "status": "success",
            "message": "Inference successful for multiple files",
            "data": results
        }
        return jsonify(response), 200


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
