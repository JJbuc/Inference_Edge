import torch
import numpy as np
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from llm_handler import LLM_Handler

class ContrastiveDecoder(LLM_Handler):
    def __init__(self, config):
        super().__init__(config)
        self.model_ama_name = config.get("model_ama")
        self.model_exp_name = config.get("model_exp")
        self.device = config.get("device", "cuda")
        self.head_parameter = config.get("head_parameter", 1.0)

        # Load small amateur LLM
        self.model_ama = AutoModelForCausalLM.from_pretrained(self.model_ama_name, device_map=self.device_map)
        self.tokenizer_ama = AutoTokenizer.from_pretrained(self.model_ama_name)
        self.pipe_ama = pipeline("text-generation", model=self.model_ama, tokenizer=self.tokenizer_ama)

        # Load large expert LLM
        self.model_exp = AutoModelForCausalLM.from_pretrained(self.model_exp_name, device_map=self.device_map)
        self.tokenizer_exp = AutoTokenizer.from_pretrained(self.model_exp_name)
        self.pipe_exp = pipeline("text-generation", model=self.model_exp, tokenizer=self.tokenizer_exp)

    def get_log_probs(self, model, tokenizer, input_ids):
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits[:, -1, :]  # Last token logits
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs.cpu().numpy()

    def generate(self, prompt, max_length=None, **override_params):
        if max_length is None:
            max_length = self.default_params.get("max_length", 1000)
        input_ids = self.tokenizer_ama(prompt, return_tensors="pt").input_ids
        generated_ids = input_ids.tolist()[0]
        for _ in range(max_length):
            # Get log probs from both models for current sequence
            ama_log_probs = self.get_log_probs(self.model_ama, self.tokenizer_ama, torch.tensor([generated_ids]))
            exp_log_probs = self.get_log_probs(self.model_exp, self.tokenizer_exp, torch.tensor([generated_ids]))
            # Contrastive logic
            contrastive_log_probs = exp_log_probs / (ama_log_probs + 1e-9)
            contrastive_log_probs = contrastive_log_probs * self.head_parameter
            next_token_id = int(np.argmax(contrastive_log_probs[0]))
            generated_ids.append(next_token_id)
            # Optionally, break if EOS token is generated
            if next_token_id == self.tokenizer_ama.eos_token_id:
                break
        output = self.tokenizer_ama.decode(generated_ids, skip_special_tokens=True)
        return output
    
    def clear_space(self):
        print("\nClearing space for contrastive decoding models...")
        # Clear small amateur model
        if hasattr(self, 'pipe_ama') and self.pipe_ama is not None:
            del self.pipe_ama
            self.pipe_ama = None
        if hasattr(self, 'model_ama') and self.model_ama is not None:
            del self.model_ama
            self.model_ama = None
        if hasattr(self, 'tokenizer_ama') and self.tokenizer_ama is not None:
            del self.tokenizer_ama
            self.tokenizer_ama = None
        # Clear large expert model
        if hasattr(self, 'pipe_exp') and self.pipe_exp is not None:
            del self.pipe_exp
            self.pipe_exp = None
        if hasattr(self, 'model_exp') and self.model_exp is not None:
            del self.model_exp
            self.model_exp = None
        if hasattr(self, 'tokenizer_exp') and self.tokenizer_exp is not None:
            del self.tokenizer_exp
            self.tokenizer_exp = None
        # Also call base handler's clear_space for main model
        super().clear_space()