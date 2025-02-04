import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
from waitress import serve
import os

model_name = 'meditron-7b'
API_PORT = 33520

class LocalLlmServer:
    def __init__(self, model_name="epfl-llm/meditron-7b", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto"
        )

    def chat_completion(self, messages, max_tokens=200, temperature=0.7):
        prompt = " ".join([msg['content'] for msg in messages if msg['role'] == 'user'])

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        outputs = self.model.generate(
            **inputs,
            max_length=max_tokens,
            num_return_sequences=1,
            temperature=temperature,
            do_sample=True
        )

        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        return {
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop",
                "index": 0
            }],
            "usage": {
                "prompt_tokens": len(inputs[0]),
                "completion_tokens": len(outputs[0]),
                "total_tokens": len(inputs[0]) + len(outputs[0])
            }
        }


if __name__ == '__main__':
    print("Starting Local LLM Server...")

    script_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(script_dir, model_name)

    if os.path.exists(model_path):
        local_llm_server = LocalLlmServer(model_name=model_path)
    else:
        print("Model not found in script directory!")
        input("Press Enter to continue and download the large model! (>25G)")
        local_llm_server = LocalLlmServer()

    app = Flask(__name__)

    @app.route('/v1/chat/completions', methods=['POST'])
    def chat_completions():
        data = request.json

        try:
            response = local_llm_server.chat_completion(
                messages=data.get('messages', []),
                max_tokens=data.get('max_tokens', 200),
                temperature=data.get('temperature', 0.7)
            )

            return jsonify(response)

        except Exception as e:
            return jsonify({"error": str(e)}), 500


    serve(app, host='0.0.0.0', port=API_PORT)
