import os
import cloudpickle
from uuid import uuid4
from typing import List, Optional
from flask import Flask, request, jsonify
import waitress
import json
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

# Configuration
language = "Persian"
host = "0.0.0.0"
port = 33518
config_json = "config.json"
config_json_default = "config.json.default"
CONVERSATIONS_FOLDER = "conversations"

# Default context
text_context_prepend = ("Role play: You are a radiologist." +
                        "You give expert opinion on mammography images for breast cancer screening." +
                        "We have a deep learning model that predicts suspicious mass and their low / high risk of breast cancer." +
                        "Low risk means BI-RADS <= 3, high risk is BI-RADS > 3." +
                        "You are given the model predictions where class = 0 is low, class = 1 is high risk." +
                        f"No matter the input language, you must always speak in {language}.")

# Load configuration
if not os.path.exists(config_json):
    raise Exception(f"{config_json} not found!")

with open(config_json, "r") as config_file:
    config = json.load(config_file)

BASE_URL = config.get("base_url")
API_KEY = config.get("api_key")

app = Flask(__name__)

# Conversation storage
def init_storage():
    if os.path.exists(CONVERSATIONS_FOLDER):
        for file in os.listdir(CONVERSATIONS_FOLDER):
            os.remove(os.path.join(CONVERSATIONS_FOLDER, file))
    else:
        os.makedirs(CONVERSATIONS_FOLDER)

def save_conversation(conversation_id: str, context: List[dict]):
    file_path = os.path.join(CONVERSATIONS_FOLDER, f"{conversation_id}.pkl")
    with open(file_path, "wb") as file:
        cloudpickle.dump(context, file)

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

        if not prompt:
            return jsonify({"error": "Prompt is required."}), 400

        if conversation_id:
            context = get_conversation(conversation_id)
            if not context:
                return jsonify({"error": "Invalid conversation ID."}), 400
        else:
            conversation_id = str(uuid4())
            context = [{"role": "system", "content": text_context_prepend}]

        response_content = langchain_api.generate_response(prompt, context)
        context.append({"role": "user", "content": prompt})
        context.append({"role": "assistant", "content": response_content})

        save_conversation(conversation_id, context)

        return jsonify({"response": response_content, "conversation_id": conversation_id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    init_storage()
    waitress.serve(app, host=host, port=port)
