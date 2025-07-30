import torch
import numpy as np
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from llm_handler import LLM_Handler

class ContrastiveDecoder(LLM_Handler):
    def __init__(self, config):
        # super().__init__(config)  # Loads model_name as "amateur" model
        self.model_exp_name = config.get("model_exp")
        self.device = config.get("device", "cuda")
        self.device_map = {"": 0} if self.device == "cuda" and torch.cuda.is_available() else {"": "cpu"}
        self.head_parameter = config.get("head_parameter", 1.0)
        self.default_params = config.get("default_params", {})

        # Load large expert LLM
        self.model_exp = AutoModelForCausalLM.from_pretrained(self.model_exp_name, device_map=self.device_map)
        self.tokenizer_exp = AutoTokenizer.from_pretrained(self.model_exp_name)
        self.pipe_exp = pipeline("text-generation", model=self.model_exp, tokenizer=self.tokenizer_exp)

    def get_log_probs(self, model, tokenizer, input_ids):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs.cpu().numpy()

    def generate(self, prompt, max_length=None, **override_params):
        if max_length is None:
            max_length = self.default_params.get("max_length", 1000)
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids  # Use amateur tokenizer
        generated_ids = input_ids.tolist()[0]
        for _ in range(max_length):
            ama_log_probs = self.get_log_probs(self.model, self.tokenizer, torch.tensor([generated_ids]))
            exp_log_probs = self.get_log_probs(self.model_exp, self.tokenizer_exp, torch.tensor([generated_ids]))
            contrastive_log_probs = exp_log_probs / (ama_log_probs + 1e-9)
            contrastive_log_probs = contrastive_log_probs * self.head_parameter
            next_token_id = int(np.argmax(contrastive_log_probs[0]))
            generated_ids.append(next_token_id)
            if next_token_id == self.tokenizer.eos_token_id:
                break
        output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        return output

    def clear_space(self):
        print("\nClearing space for contrastive decoding models...")
        if hasattr(self, 'pipe_exp') and self.pipe_exp is not None:
            del self.pipe_exp
            self.pipe_exp = None
        if hasattr(self, 'model_exp') and self.model_exp is not None:
            del self.model_exp
            self.model_exp = None
        if hasattr(self, 'tokenizer_exp') and self.tokenizer_exp is not None:
            del self.tokenizer_exp
            self.tokenizer_exp = None
        super().clear_space()