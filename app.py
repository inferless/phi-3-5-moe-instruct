import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"]='1'
from huggingface_hub import snapshot_download
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import bitsandbytes as bnb

class InferlessPythonModel:
    def initialize(self):
        model_id = "microsoft/Phi-3.5-MoE-instruct"
        snapshot_download(repo_id=model_id,allow_patterns=["*.safetensors"])
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, 
            device_map="cuda",  
            torch_dtype="auto",  
            trust_remote_code=True,
            load_in_4bit=True
        )
        self.pipe = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def infer(self, inputs):
        prompt = inputs["prompt"]
        messages = [
            {"role": "system", "content": "You are a helpful AI assistant."},
            {"role": "user", "content": prompt},
        ]
        generation_args = { 
            "max_new_tokens": 500, 
            "return_full_text": False, 
            "temperature": 0.0, 
            "do_sample": False, 
        }
        out = self.pipe(messages, **generation_args)
        generated_text = out[0]["generated_text"]
        return {'generated_result': generated_text}

    def finalize(self):
        pass
