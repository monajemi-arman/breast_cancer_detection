import torch
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from waitress import serve
import os

model_name = 'meditron-7b'
API_PORT = 33520


class LocalLlmServer:
    def __init__(self, model_name="epfl-llm/meditron-7b", device="cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"

        # Configure 4-bit quantization
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quantization_config,
            device_map="auto"
        )

    def chat_completion(self, messages, max_tokens=200, temperature=0.7):
        # Build structured prompt with role tags
        prompt = "\n".join(
            [f"{msg['role'].capitalize()}: {msg['content']}"
             for msg in messages if msg['role'] in ['system', 'user']]
        )
        prompt += "\nAssistant:"

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # Generate response with adjusted parameters
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            temperature=0.5 if temperature < 0.5 else temperature,  # Prevent too low temp
            top_p=0.9,
            repetition_penalty=1.2,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )

        # Decode and clean response
        response_text = self.tokenizer.decode(
            outputs[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        ).strip()

        # Calculate token usage
        prompt_tokens = inputs.input_ids.shape[-1]
        completion_tokens = outputs.shape[-1] - prompt_tokens

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
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }


if __name__ == '__main__':
    print("Starting Local LLM Server...")

    script_dir = os.path.dirname(os.path.realpath(__file__))
    model_path = os.path.join(script_dir, model_name)

    # Verify model path exists
    if os.path.exists(model_path) and os.path.isdir(model_path):
        local_llm_server = LocalLlmServer(model_name=model_path)
    else:
        print(f"Model not found at {model_path}!")
        input("Press Enter to download the model (>25G)...")
        local_llm_server = LocalLlmServer()

    app = Flask(__name__)


    @app.route('/v1/chat/completions', methods=['POST'])
    def chat_completions():
        try:
            data = request.json
            response = local_llm_server.chat_completion(
                messages=data.get('messages', []),
                max_tokens=data.get('max_tokens', 200),
                temperature=data.get('temperature', 0.7)
            )
            return jsonify(response)

        except Exception as e:
            return jsonify({"error": str(e)}), 500


    serve(app, host='0.0.0.0', port=API_PORT)