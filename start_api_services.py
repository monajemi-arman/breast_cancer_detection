import os.path
import signal
import subprocess
import sys
import threading

scripts_to_run = [
    ("Image inference server", "webapp/web.py", []),
    ("X-AI (classification) server", "classification_model.py", ["-c", "api"]),
    ("LLM chatbot server", "llm/llm_api_server.py", []),
    ("DICOM to JPEG converter server", "dicom_to_jpeg_api.py", []),
    ("Hash router server", "hash_router.py", []),
    ("Watchdog for DICOM folder", "watch_images_dir.py", [])
]

processes = []

script_dir = os.path.dirname(os.path.realpath(__file__))

def run_script(title, script, args):
    print(f"[OK] Starting {title}...")
    cmd = [sys.executable, script] + args
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    processes.append(process)
    stdout, stderr = process.communicate()
    print(f"Output of {title} ({script}):")
    print(stdout)
    if stderr:
        print(f"Error in {title} ({script}):")
        print(stderr)


def signal_handler(sig, frame):
    print("\n[INFO] Control+C pressed. Stopping all scripts...")
    for process in processes:
        process.terminate()
    sys.exit(0)


def main(scripts_to_run):
    signal.signal(signal.SIGINT, signal_handler)

    threads = []
    for title, script, args in scripts_to_run:
        script = os.path.join(script_dir, script)
        thread = threading.Thread(target=run_script, args=(title, script, args))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main(scripts_to_run)
