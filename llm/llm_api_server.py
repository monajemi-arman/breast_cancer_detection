import os.path
import sys
from typing import List, Dict, Optional
import waitress
from flask import Flask, request, jsonify
import json
from shutil import copyfile
from langchain_openai import ChatOpenAI
from pathlib import Path
from chardet.universaldetector import UniversalDetector

# Configuration
language = "Persian"
host = "0.0.0.0"
port = 33518
config_json = "config.json"
config_json_default = "config.json.default"
demo_directory = "demo"
# -------------

# Texts
text_context_prepend = ("Role play: You are a radiologist." +
                        "You give expert opinion on mammography images for breast cancer screening." +
                        "We have a deep learning model that predicts suspicious mass and their low / high risk of breast cancer." +
                        "Low risk means BI-RADS <= 3, high risk is BI-RADS > 3." +
                        "You are given the model predictions where class = 0 is low, class = 1 is high risk." +
                        "You may also receive an description on the image in this context." +
                        "The user then asks about the case and you take into account the mentioned details." +
                        f"No matter the input language, you must always speak in {language}.")

text_predictions = "Model predictions on the image is: {}"
text_description = "Description of the image: {}"

if not os.path.exists(config_json):
    copyfile(config_json_default, config_json)
    raise Exception("Config file not found! Creating based on default template...\nPlease edit config.json!")

# Load configuration from llm_config.json
with open(config_json, "r") as config_file:
    config = json.load(config_file)

DEFAULT_CONTEXT = config.get("default_context")
BASE_URL = config.get("base_url")
API_KEY = config.get("api_key")

app = Flask(__name__)


class LangChainAPI:
    def __init__(self, api_key: str, base_url: str, default_context: Optional[List[Dict[str, str]]] = None):
        self.llm_chat = ChatOpenAI(model="gpt-3.5-turbo", base_url=base_url, api_key=api_key)
        self.default_context = default_context or DEFAULT_CONTEXT

    def generate_response(self, prompt: str, context: Optional[List[Dict[str, str]]] = None) -> str:
        messages = (context or self.default_context) + [{"role": "user", "content": prompt}]
        response = self.llm_chat.invoke(messages)
        return response


langchain_api = LangChainAPI(api_key=API_KEY, base_url=BASE_URL)


@app.route('/generate-response', methods=['POST'])
def generate_response():
    try:
        data = request.get_json()
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
            context += text_predictions.format(predictions)

        if demo:
            demo_text = demo_get_text(demo)
            if demo_text:
                context += text_description.format(demo_text)

        response = langchain_api.generate_response(prompt=prompt, context=[{"role": "system", "content": context}])
        return jsonify({"response": response.content})
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
    waitress.serve(app, host=host, port=port)
