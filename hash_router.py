#! /usr/bin/env python
# This is a Hash Router, it sends requests to endpoints and pulls images from hash cache if present
# No need to re-upload the images for each API, once it is cached using dicom_to_jpeg script, it will be cached

from flask import Flask, request, jsonify
from flask_cors import CORS
from waitress import serve
import os
import requests
import logging

API_PORT = 33516
UPLOAD_FOLDER = 'uploaded_images'
DEBUG_MODE = False

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

logging.basicConfig(level=logging.DEBUG if DEBUG_MODE else logging.INFO)
logger = logging.getLogger(__name__)

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)


@app.route('/route', methods=['POST'])
def route_to_endpoint():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No JSON data received"}), 400

        hash_value = data.get('hash')
        endpoint = data.get('endpoint')
        payload = data.get('data')

        if not hash_value or not endpoint or not payload:
            return jsonify({"error": "Missing hash, endpoint, or data in request"}), 400

        file_path = os.path.join(UPLOAD_FOLDER, f"{hash_value}.jpg")
        if not os.path.exists(file_path):
            return jsonify({"error": "File not found"}), 404

        with open(file_path, 'rb') as file:
            files = {'file': file}
            if hash_value in payload:
                payload[hash_value] = file

            response = requests.post(endpoint, files=files, data=payload)

        try:
            response_data = response.json()
        except ValueError:
            response_data = response.content.decode('utf-8')

        return jsonify({
            "status_code": response.status_code,
            "response_data": response_data
        }), response.status_code

    except Exception as e:
        logger.error(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


def main():
    serve(app, host='0.0.0.0', port=API_PORT)


if __name__ == '__main__':
    main()
