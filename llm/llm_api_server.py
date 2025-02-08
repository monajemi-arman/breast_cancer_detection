import os
import sys
import cloudpickle
from uuid import uuid4
from typing import List, Optional
from chardet import UniversalDetector
from flask import Flask, request, jsonify
import waitress
import json
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pathlib import Path
from flask_cors import CORS

# Configuration
language = "Persian"
host = "0.0.0.0"
port = 33518
config_json = "config.json"
config_json_default = "config.json.default"
demo_directory = "demo"
CONVERSATIONS_FOLDER = "conversations"
MAX_CONVERSATIONS = 40
CLEANUP_THRESHOLD = 20

script_dir = os.path.dirname(os.path.realpath(__file__))
config_json = os.path.join(script_dir, config_json)

# Default context
text_context_prepend = ("Role play: You are a radiologist." +
                        "You give expert opinion on mammography images for breast cancer screening." +
                        "We have a deep learning model that predicts suspicious mass and their low / high risk of breast cancer." +
                        "Low risk means BI-RADS <= 3, high risk is BI-RADS > 3." +
                        "You are given the model predictions where class = 0 is low, class = 1 is high risk." +
                        f"No matter the input language, you must ALWAYS speak in {language}.")
text_predictions = "Model predictions on the image is: {}"
text_description = "Description of the image: {}"

# Load configuration
if not os.path.exists(config_json):
    raise Exception(f"{config_json} not found!")

with open(config_json, "r") as config_file:
    config = json.load(config_file)

BASE_URL = config.get("base_url")
API_KEY = config.get("api_key")

app = Flask(__name__)

# Enable CORS and allow all hosts
CORS(app, resources={r"/*": {"origins": "*"}})


# Conversation storage
def init_storage():
    if not os.path.exists(CONVERSATIONS_FOLDER):
        os.makedirs(CONVERSATIONS_FOLDER)


def cleanup_old_conversations():
    """Remove oldest conversations when the number of files exceeds the threshold"""
    conversation_files = []
    for file in Path(CONVERSATIONS_FOLDER).glob("*.pkl"):
        conversation_files.append((file, file.stat().st_mtime))

    if len(conversation_files) >= MAX_CONVERSATIONS:
        # Sort by modification time (oldest first)
        conversation_files.sort(key=lambda x: x[1])

        # Remove the oldest files until we reach MAX_CONVERSATIONS - CLEANUP_THRESHOLD
        files_to_remove = len(conversation_files) - (MAX_CONVERSATIONS - CLEANUP_THRESHOLD)
        for file, _ in conversation_files[:files_to_remove]:
            try:
                file.unlink()
            except Exception as e:
                print(f"Error removing file {file}: {e}")


def save_conversation(conversation_id: str, context: List[dict]):
    file_path = os.path.join(CONVERSATIONS_FOLDER, f"{conversation_id}.pkl")
    with open(file_path, "wb") as file:
        cloudpickle.dump(context, file)

    # Check and cleanup after each save
    cleanup_old_conversations()


def get_conversation(conversation_id: str) -> Optional[List[dict]]:
    file_path = os.path.join(CONVERSATIONS_FOLDER, f"{conversation_id}.pkl")
    if os.path.exists(file_path):
        with open(file_path, "rb") as file:
            return cloudpickle.load(file)
    return None


def convert_to_message(obj: dict):
    role = obj.get("role")
    content = obj.get("content")
    if role == "system":
        return SystemMessage(content=content)
    elif role == "user":
        return HumanMessage(content=content)
    elif role == "assistant":
        return AIMessage(content=content)
    return None


def convert_to_dict(message):
    return {"role": message.role, "content": message.content}


# LangChain API
class LangChainAPI:
    def __init__(self, api_key: str, base_url: str):
        self.llm_chat = ChatOpenAI(model="gpt-3.5-turbo", base_url=base_url, api_key=api_key)

    def generate_response(self, prompt: str, context: List[dict]) -> str:
        messages = [convert_to_message(msg) for msg in context]
        messages.append(HumanMessage(content=prompt))
        response = self.llm_chat.invoke(messages)
        return response.content


langchain_api = LangChainAPI(api_key=API_KEY, base_url=BASE_URL)


@app.route('/generate-response', methods=['POST'])
def generate_response():
    try:
        data = request.get_json()
        conversation_id = data.get('conversation_id')
        prompt = data.get('prompt')
        context = data.get('context')
        predictions = data.get('predictions')
        demo = data.get('demo')

        if not prompt:
            return jsonify({"error": "Prompt is required."}), 400

        if not context:
            context = text_context_prepend
        else:
            context = text_context_prepend + context

        if predictions:
            predictions = str(predictions)
            context += text_predictions.format(predictions)

        if demo:
            demo_text = demo_get_text(demo)
            if demo_text:
                context += text_description.format(demo_text)

        if conversation_id:
            context = get_conversation(conversation_id)
            if not context:
                return jsonify({"error": "Invalid conversation ID."}), 400
        else:
            conversation_id = str(uuid4())
            context = [{"role": "system", "content": context}]

        response_content = langchain_api.generate_response(prompt, context)
        context.append({"role": "user", "content": prompt})
        context.append({"role": "assistant", "content": response_content})

        save_conversation(conversation_id, context)

        return jsonify({"response": response_content, "conversation_id": conversation_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def demo_get_text(demo_file):
    for demo_text_file in os.listdir(demo_directory):
        demo_prefix = Path(demo_text_file).stem
        if demo_prefix in demo_file:
            demo_text_file_path = os.path.join(demo_directory, demo_text_file)
            demo_text = read_demo_text_file(demo_text_file_path)
            return demo_text
    print("Demo requested but demo directory did not contain the required text file!", file=sys.stderr)


def read_demo_text_file(filepath):
    detector = UniversalDetector()

    with open(filepath, 'rb') as file:
        for line in file:
            detector.feed(line)
            if detector.done:
                break
        detector.close()

        encoding = detector.result['encoding']
        if encoding is None:
            raise ValueError("Unable to detect file encoding.")

        file.seek(0)
        return file.read().decode(encoding)


if __name__ == "__main__":
    init_storage()
    waitress.serve(app, host=host, port=port)
