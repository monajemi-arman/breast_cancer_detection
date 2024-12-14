import json
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from PIL import Image, ImageDraw
import io
import base64
from infer import infer
from waitress import serve
import tempfile

# Host and port for waitress server
host = '0.0.0.0'
port = 33517

app = Flask(__name__)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'jpg', 'jpeg', 'png', 'nrrd'}


def draw_boxes(image, annotations):
    draw = ImageDraw.Draw(image)
    # Determine line width dynamically based on image dimensions
    line_width = max(1, int(min(image.width, image.height) * 0.005))  # 0.5% of the smaller dimension
    for ann in annotations:
        x, y, width, height = ann['bbox']
        draw.rectangle(
            [x, y, x + width, y + height],
            outline="red",
            width=line_width
        )
    return image


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    image_data = None
    gt_data = None
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']
        gt_file = request.files.get('gt_file')  # Ground truth file

        if file.filename == '':
            return render_template('index.html', message='No selected file')
        if file and allowed_file(file.filename):
            img = Image.open(file)
            img = img.convert('RGB')
            output = io.BytesIO()
            img.save(output, format='JPEG')
            image_data = output.getvalue()
            image_data_orig = base64.b64encode(image_data).decode('utf-8')

            # Process ground truth if JSON is provided
            annotations = []
            if gt_file and gt_file.filename.endswith('.json'):
                gt_json = json.load(gt_file)
                image_annotations = [
                    ann for ann in gt_json['annotations']
                    if ann['image_id'] == 0  # Modify based on your dataset's image ID logic
                ]
                annotations = image_annotations

            # Draw ground truth on image
            if annotations:
                img_gt = draw_boxes(img.copy(), annotations)
                gt_output = io.BytesIO()
                img_gt.save(gt_output, format='JPEG')
                gt_data = base64.b64encode(gt_output.getvalue()).decode('utf-8')

            # Infer predictions
            image_data = np.array(img)
            image_data = infer(image_data)
            img = Image.fromarray(image_data)
            output = io.BytesIO()
            img.save(output, format='JPEG')
            image_data = base64.b64encode(output.getvalue()).decode('utf-8')

            return render_template(
                'index.html',
                message='File uploaded successfully',
                image_data_orig=image_data_orig,
                gt_data=gt_data,
                image_data=image_data
            )

    return render_template('index.html')


if __name__ == '__main__':
    serve(app.wsgi_app, host=host, port=port)
