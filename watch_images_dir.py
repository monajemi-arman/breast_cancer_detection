import os
import sqlite3
import time
import json
import subprocess
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from flask import Flask, request, jsonify
from waitress import serve
from PIL import Image
from flask_cors import CORS
import pydicom
import hashlib
import collections.abc

BREAST_ONLY = True  # Set to True if only breast images are to be processed
WATCH_FOLDER = "./watch_folder"
UPLOAD_URL = "http://localhost:33521/upload"
DB_FILE = "file_hashes.db"
UPLOADED_IMAGES_DIR = "uploaded_images"
THUMBNAIL_SIZE = (125, 125)
API_PORT = 33522

os.makedirs(WATCH_FOLDER, exist_ok=True)
os.makedirs(UPLOADED_IMAGES_DIR, exist_ok=True)


def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        """CREATE TABLE IF NOT EXISTS file_hashes
                 (hash TEXT PRIMARY KEY, original_filename TEXT, write_time REAL, patient_name TEXT)"""
    )
    c.execute("PRAGMA table_info(file_hashes)")
    columns = [row[1] for row in c.fetchall()]
    if "patient_name" not in columns:
        c.execute("ALTER TABLE file_hashes ADD COLUMN patient_name TEXT")
    if "write_time" not in columns:
        c.execute("ALTER TABLE file_hashes ADD COLUMN write_time REAL")
    if "dicom_metadata" not in columns:
        c.execute("ALTER TABLE file_hashes ADD COLUMN dicom_metadata TEXT")
    conn.commit()
    conn.close()


init_db()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})


def get_dicom_metadata(file_path):
    try:
        ds = pydicom.dcmread(file_path, stop_before_pixels=True)
        metadata = filter_json_serializable_metadata(ds)
        return metadata
    except Exception as e:
        print(f"Failed to read DICOM metadata from {file_path}: {e}")
        return {}


def filter_json_serializable_metadata(ds):
    metadata = {}
    for elem in ds:
        if elem.keyword and not elem.VR == "SQ":
            try:
                # Try to convert value to something JSON serializable
                value = elem.value
                # Convert bytes to string
                if isinstance(value, bytes):
                    value = value.decode(errors="replace")
                # Convert pydicom PersonName to string
                if hasattr(value, "original_string"):
                    value = str(value)
                # Convert multi-value to list of strings
                if isinstance(value, collections.abc.Iterable) and not isinstance(
                    value, str
                ):
                    value = [str(v) for v in value]
                # Only keep JSON serializable types
                json.dumps(value)
                metadata[elem.keyword] = value
            except Exception:
                continue
    return metadata


def get_patient_name(file_path):
    try:
        ds = pydicom.dcmread(file_path, stop_before_pixels=True)
        return str(ds.get("PatientName", "Unknown"))
    except Exception as e:
        print(f"Failed to read PatientName from {file_path}: {e}")
        return "Unknown"


@app.route("/hash_to_original")
def hash_to_original():
    hash_value = request.args.get("hash")
    if not hash_value:
        return jsonify({"error": "Hash required"}), 400

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "SELECT original_filename, dicom_metadata FROM file_hashes WHERE hash=?",
        (hash_value,),
    )
    result = c.fetchone()
    conn.close()

    if not result:
        return jsonify({"error": "Not found"}), 404

    original_filename, metadata_json = result
    try:
        metadata = json.loads(metadata_json) if metadata_json else {}
    except Exception:
        metadata = {}

    return jsonify({"original_filename": original_filename, "dicom_metadata": metadata})


@app.route("/images")
def list_images():
    page = request.args.get("page", default=1, type=int)
    count = request.args.get("count", default=None, type=int)
    after_time = request.args.get("time", default=None, type=float)

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()

    query = "SELECT hash, original_filename, write_time, patient_name FROM file_hashes"
    params = []
    where_clauses = []
    if after_time is not None:
        where_clauses.append("write_time > ?")
        params.append(after_time)
    if where_clauses:
        query += " WHERE " + " AND ".join(where_clauses)
    if count:
        offset = (page - 1) * count
        query += " LIMIT ? OFFSET ?"
        params.extend([count, offset])
    c.execute(query, params)

    images = [
        {
            "hash": row[0],
            "original_filename": row[1],
            "write_time": row[2],
            "patient_name": row[3],
        }
        for row in c.fetchall()
    ]
    conn.close()
    return jsonify(images)


@app.route("/patient")
def get_patient_images():
    patient_name = request.args.get("name")
    if not patient_name:
        return jsonify({"error": "Patient name required"}), 400

    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute(
        "SELECT hash, original_filename, write_time, patient_name FROM file_hashes WHERE patient_name=?",
        (patient_name,),
    )
    images = [
        {
            "hash": row[0],
            "original_filename": row[1],
            "write_time": row[2],
            "patient_name": row[3],
        }
        for row in c.fetchall()
    ]
    conn.close()
    return jsonify(images)


class UploadedImagesHandler(FileSystemEventHandler):
    def on_deleted(self, event):
        if not event.is_directory and event.src_path.endswith(".jpg"):
            filename = os.path.basename(event.src_path)
            if filename.startswith("thumb_"):
                return  # Ignore thumbnails
            file_hash = filename.replace(".jpg", "")
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("DELETE FROM file_hashes WHERE hash=?", (file_hash,))
            conn.commit()
            conn.close()
            print(f"Removed DB entry for deleted image: {file_hash}")


class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            self.process_file(event.src_path)

    def on_modified(self, event):
        if not event.is_directory:
            self.process_file(event.src_path)

    def process_file(self, file_path):
        filename = os.path.basename(file_path)
        # Ignore index and index-wal files
        if filename in ["index", "index-wal"]:
            return

        # If BREAST_ONLY is enabled, check BodyPartExamined
        if BREAST_ONLY:
            try:
                ds = pydicom.dcmread(file_path, stop_before_pixels=True)
                body_part = getattr(ds, "BodyPartExamined", None)
                if not body_part or str(body_part).upper() != "BREAST":
                    print(f"Skipping {file_path}: BodyPartExamined is not BREAST")
                    return
            except Exception as e:
                print(f"Failed to read BodyPartExamined from {file_path}: {e}")
                return

        rel_path = os.path.relpath(file_path, WATCH_FOLDER)

        try:
            with open(file_path, "rb") as f:
                file_hash = hashlib.sha256(f.read()).hexdigest()
        except Exception as e:
            print(f"Failed to hash file {file_path}: {e}")
            return

        # Upload file to server (simulate or implement as needed)
        try:
            result = subprocess.run(
                ["curl", "-F", f"file=@{file_path}", UPLOAD_URL],
                capture_output=True,
                text=True,
            )
        except Exception as e:
            print(f"Failed to upload file {file_path}: {e}")
            return

        if result.returncode != 0:
            print(f"Upload failed: {rel_path}")
            return

        try:
            response = json.loads(result.stdout)
            file_hash = response.get("hash")
        except Exception as e:
            print(f"Failed to parse upload response: {e}")
            return

        if not file_hash:
            return

        write_time = os.path.getmtime(file_path)
        patient_name = get_patient_name(file_path)
        # Get and filter DICOM metadata
        try:
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            filtered_metadata = filter_json_serializable_metadata(ds)
            metadata_json = json.dumps(filtered_metadata)
        except Exception:
            metadata_json = "{}"

        conn = sqlite3.connect(DB_FILE)
        c = conn.cursor()
        c.execute(
            "INSERT OR IGNORE INTO file_hashes (hash, original_filename, write_time, patient_name, dicom_metadata) VALUES (?, ?, ?, ?, ?)",
            (file_hash, rel_path, write_time, patient_name, metadata_json),
        )
        conn.commit()
        conn.close()

        image_path = os.path.join(UPLOADED_IMAGES_DIR, f"{file_hash}.jpg")
        if os.path.exists(image_path):
            with Image.open(image_path) as img:
                img.thumbnail(THUMBNAIL_SIZE)
                img.save(os.path.join(UPLOADED_IMAGES_DIR, f"thumb_{file_hash}.jpg"))


def start_file_watcher():
    observer = Observer()
    observer.schedule(FileHandler(), WATCH_FOLDER, recursive=True)
    observer.schedule(UploadedImagesHandler(), UPLOADED_IMAGES_DIR, recursive=False)
    observer.start()
    return observer


def process_existing_files():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    for root, dirs, files in os.walk(WATCH_FOLDER):
        for file in files:
            if file.lower() in ["index", "index-wal"]:
                continue
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, WATCH_FOLDER)
            c.execute(
                "SELECT hash FROM file_hashes WHERE original_filename=?", (rel_path,)
            )
            row = c.fetchone()
            needs_processing = False
            if not row:
                needs_processing = True
            else:
                file_hash = row[0]
                image_path = os.path.join(UPLOADED_IMAGES_DIR, f"{file_hash}.jpg")
                if not os.path.exists(image_path):
                    needs_processing = True
            if needs_processing:
                FileHandler().process_file(file_path)
    conn.close()


def cleanup_missing_images():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT hash FROM file_hashes")
    hashes = [row[0] for row in c.fetchall()]
    removed = 0
    for file_hash in hashes:
        image_path = os.path.join(UPLOADED_IMAGES_DIR, f"{file_hash}.jpg")
        if not os.path.exists(image_path):
            c.execute("DELETE FROM file_hashes WHERE hash=?", (file_hash,))
            removed += 1
    if removed:
        print(f"Removed {removed} entries from file_hashes for missing images.")
    conn.commit()
    conn.close()


def main():
    cleanup_missing_images()
    process_existing_files()  # Process files already in the watch folder
    observer = start_file_watcher()
    print(f"Watching {WATCH_FOLDER}")
    print(f"API running on port {API_PORT}")
    serve(app, host="0.0.0.0", port=API_PORT)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()


if __name__ == "__main__":
    main()
