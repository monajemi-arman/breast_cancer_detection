import os.path
from typing import List, Dict, Optional
from flask import Flask, request, jsonify
import json
from shutil import copyfile
from langchain_openai import ChatOpenAI

# Config file
config_json = "config.json"
config_json_default = "config.json.default"

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

        if not prompt:
            return jsonify({"error": "Prompt is required."}), 400

        response = langchain_api.generate_response(prompt=prompt, context=context)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
