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
    c.execute('''CREATE TABLE IF NOT EXISTS file_hashes
                 (hash TEXT PRIMARY KEY, original_filename TEXT)''')
    conn.commit()
    conn.close()

init_db()

app = Flask(__name__)
CORS(app)

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
    
    conn = sqlite3.connect(DB_FILE)
    c = conn.cursor()
    
    if count:
        offset = (page - 1) * count
        c.execute("SELECT hash, original_filename FROM file_hashes LIMIT ? OFFSET ?", (count, offset))
    else:
        c.execute("SELECT hash, original_filename FROM file_hashes")
    
    images = [{'hash': row[0], 'original_filename': row[1]} for row in c.fetchall()]
    conn.close()
    return jsonify(images)

class FileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            self.process_file(event.src_path)

    def process_file(self, file_path):
        try:
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
            
            conn = sqlite3.connect(DB_FILE)
            c = conn.cursor()
            c.execute("INSERT OR IGNORE INTO file_hashes VALUES (?, ?)", (file_hash, rel_path))
            conn.commit()
            conn.close()

            image_path = os.path.join(UPLOADED_IMAGES_DIR, f"{file_hash}.jpg")
            if os.path.exists(image_path):
                with Image.open(image_path) as img:
                    img.thumbnail(THUMBNAIL_SIZE)
                    img.save(os.path.join(UPLOADED_IMAGES_DIR, f"thumb_{file_hash}.jpg"))

        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")

def start_file_watcher():
    observer = Observer()
    observer.schedule(FileHandler(), WATCH_FOLDER, recursive=True)
    observer.start()
    return observer

def main():
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