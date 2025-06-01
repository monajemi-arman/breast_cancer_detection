import gzip
import hashlib
import os
from flask import Flask, request, jsonify, send_from_directory, abort
from waitress import serve
from io import BytesIO
import magic
import pydicom
from flask_cors import CORS
from pydicom.errors import InvalidDicomError
from utils import read_dicom

API_PORT = 33521
UPLOAD_FOLDER = 'uploaded_images'

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def calculate_hash(content):
    return hashlib.sha256(content).hexdigest()

def save_as_jpeg(image, filename):
    image.convert('RGB').save(filename, 'JPEG')

def is_gzip(file_content):
    mime = magic.from_buffer(file_content, mime=True)
    return mime == 'application/gzip' or mime == 'application/x-gzip'

def is_dicom(file_content):
    try:
        pydicom.dcmread(BytesIO(file_content), stop_before_pixels=True, force=True)
        return True
    except InvalidDicomError:
        return False

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_content = file.read()
    if is_gzip(file_content):
        try:
            file_content = gzip.decompress(file_content)
        except gzip.BadGzipFile:
            return jsonify({"error": "Invalid GZIP file"}), 400

    if not is_dicom(file_content):
        return jsonify({"error": "Invalid or corrupted DICOM file"}), 400

    file_hash = calculate_hash(file_content)
    jpeg_filename = os.path.join(app.config['UPLOAD_FOLDER'], f"{file_hash}.jpg")

    if os.path.exists(jpeg_filename):
        return jsonify({"path": f"/{UPLOAD_FOLDER}/{file_hash}.jpg", "hash": file_hash}), 200

    dicom_image = read_dicom(BytesIO(file_content))
    save_as_jpeg(dicom_image, jpeg_filename)

    return jsonify({"path": f"/{UPLOAD_FOLDER}/{file_hash}.jpg", "hash": file_hash}), 200

@app.route(f'/{UPLOAD_FOLDER}/<filename>')
def uploaded_file(filename):
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    if not os.path.isfile(file_path):
        abort(404)
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=API_PORT)