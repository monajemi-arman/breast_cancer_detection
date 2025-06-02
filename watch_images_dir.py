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

WATCH_FOLDER = './watch_folder'
UPLOAD_URL = 'http://localhost:33521/upload'
DB_FILE = 'file_hashes.db'
UPLOADED_IMAGES_DIR = 'uploaded_images'
THUMBNAIL_SIZE = (125, 125)
API_PORT = 33522

os.makedirs(WATCH_FOLDER, exist_ok=True)
os.makedirs(UPLOADED_IMAGES_DIR, exist_ok=True)

def init_db():
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    # Add write_time column if not exists
    c.execute('''CREATE TABLE IF NOT EXISTS file_hashes
                 (hash TEXT PRIMARY KEY, original_filename TEXT, write_time REAL)''')
    # Add write_time column if upgrading
    c.execute("PRAGMA table_info(file_hashes)")
    columns = [row[1] for row in c.fetchall()]
    if 'write_time' not in columns:
        c.execute("ALTER TABLE file_hashes ADD COLUMN write_time REAL")
    conn.commit()
    conn.close()

init_db()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/hash_to_original')
def hash_to_original():
    hash_value = request.args.get('hash')
    if not hash_value:
        return jsonify({'error': 'Hash required'}), 400
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    c.execute("SELECT original_filename FROM file_hashes WHERE hash=?", (hash_value,))
    result = c.fetchone()
    conn.close()
    
    return jsonify({'original_filename': result[0]}) if result else (jsonify({'error': 'Not found'}), 404)

@app.route('/images')
def list_images():
    page = request.args.get('page', default=1, type=int)
    count = request.args.get('count', default=None, type=int)
    after_time = request.args.get('time', default=None, type=float)
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    query = "SELECT hash, original_filename, write_time FROM file_hashes"
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
    
    images = [{'hash': row[0], 'original_filename': row[1], 'write_time': row[2]} for row in c.fetchall()]
    conn.close()
    return jsonify(images)

@app.route('/patient')
def get_patient_images():
    patient_id = request.args.get('id')
    if not patient_id:
        return jsonify({'error': 'Patient id required'}), 400

    folder_path = os.path.join(WATCH_FOLDER, patient_id)
    if not os.path.isdir(folder_path):
        return jsonify({'error': 'Folder not found'}), 404

    # List all files and subfolders
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    subfolders = [f for f in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, f))]

    image_rel_paths = []

    # Case 1: ≤4 images directly in folder
    image_files = [f for f in files if f not in ['index', 'index-wal']]
    if 0 < len(image_files) <= 4:
        image_rel_paths = [os.path.join(patient_id, f) for f in image_files]
    # Case 2: ≤4 subfolders, each with one image
    elif 0 < len(subfolders) <= 4:
        valid = True
        subfolder_images = []
        for sub in subfolders:
            sub_path = os.path.join(folder_path, sub)
            imgs = [f for f in os.listdir(sub_path) if os.path.isfile(os.path.join(sub_path, f)) and f not in ['index', 'index-wal']]
            if len(imgs) != 1:
                valid = False
                break
            subfolder_images.append(os.path.join(patient_id, sub, imgs[0]))
        if valid:
            image_rel_paths = subfolder_images

    if not image_rel_paths:
        return jsonify({'error': 'No valid patient images found'}), 404

    # Lookup hashes for these relative paths
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    hashes = []
    for rel_path in image_rel_paths:
        c.execute("SELECT hash FROM file_hashes WHERE original_filename=?", (rel_path,))
        row = c.fetchone()
        if row:
            hashes.append(row[0])
    conn.close()

    return jsonify({'hashes': hashes})

class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            self.process_file(event.src_path)

    def process_file(self, file_path):
        try:
            time.sleep(0.2)
            rel_path = os.path.relpath(file_path, WATCH_FOLDER)
            result = subprocess.run(['curl', '-X', 'POST', '-F', f'file=@{file_path}', UPLOAD_URL], 
                                  capture_output=True, text=True)
            
            if result.returncode != 0:
                print(f"Upload failed: {rel_path}")
                return
            
            response = json.loads(result.stdout)
            file_hash = response.get('hash')
            if not file_hash:
                return

            # Get file write time
            write_time = os.path.getmtime(file_path)
            
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("INSERT OR IGNORE INTO file_hashes (hash, original_filename, write_time) VALUES (?, ?, ?)", 
                      (file_hash, rel_path, write_time))
            conn.commit()
            conn.close()

            image_path = os.path.join(UPLOADED_IMAGES_DIR, f"{file_hash}.jpg")
            if os.path.exists(image_path):
                with Image.open(image_path) as img:
                    img.thumbnail(THUMBNAIL_SIZE)
                    img.save(os.path.join(UPLOADED_IMAGES_DIR, f"thumb_{file_hash}.jpg"))

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

class UploadedImagesHandler(FileSystemEventHandler):
    def on_deleted(self, event):
        if not event.is_directory and event.src_path.endswith('.jpg'):
            filename = os.path.basename(event.src_path)
            if filename.startswith('thumb_'):
                return  # Ignore thumbnails
            file_hash = filename.replace('.jpg', '')
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("DELETE FROM file_hashes WHERE hash=?", (file_hash,))
            conn.commit()
            conn.close()
            print(f"Removed DB entry for deleted image: {file_hash}")

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
            if file.lower() in ['index', 'index-wal']:
                continue
            file_path = os.path.join(root, file)
            rel_path = os.path.relpath(file_path, WATCH_FOLDER)
            c.execute("SELECT hash FROM file_hashes WHERE original_filename=?", (rel_path,))
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
    serve(app, host='0.0.0.0', port=API_PORT)
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == '__main__':
    main()