from llm_handler import LLM_Handler
import torch
import numpy as np

class ContrastiveDecoder:
    def __init__(self, config):
        # Load amateur model using LLM_Handler (model_ama)
        ama_config = config.copy()
        ama_config["model_name"] = config.get("model_ama")
        self.amateur_handler = LLM_Handler(ama_config)

        # Load expert model using LLM_Handler (model_exp)
        exp_config = config.copy()
        exp_config["model_name"] = config.get("model_exp")
        self.expert_handler = LLM_Handler(exp_config)

        self.head_parameter = config.get("head_parameter", 1.0)
        self.default_params = config.get("default_params", {})

    def get_log_probs(self, handler, input_ids):
        with torch.no_grad():
            outputs = handler.model(input_ids)
            logits = outputs.logits[:, -1, :]
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs.cpu().numpy()

    def generate(self, prompt, max_length=None, **override_params):
        if max_length is None:
            max_length = self.default_params.get("max_length", 1000)
        device = next(self.amateur_handler.model.parameters()).device  # Get amateur model device
        input_ids = self.amateur_handler.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        generated_ids = input_ids.tolist()[0]
        for _ in range(max_length):
            ids_tensor = torch.tensor([generated_ids], device=device)
            ama_log_probs = self.get_log_probs(self.amateur_handler, ids_tensor)
            # Make sure expert uses its own device
            exp_device = next(self.expert_handler.model.parameters()).device
            ids_tensor_exp = torch.tensor([generated_ids], device=exp_device)
            exp_log_probs = self.get_log_probs(self.expert_handler, ids_tensor_exp)
            contrastive_log_probs = exp_log_probs / (ama_log_probs + 1e-9)
            contrastive_log_probs = contrastive_log_probs * self.head_parameter
            next_token_id = int(np.argmax(contrastive_log_probs[0]))
            generated_ids.append(next_token_id)
            if next_token_id == self.amateur_handler.tokenizer.eos_token_id:
                break
        output = self.amateur_handler.tokenizer.decode(generated_ids, skip_special_tokens=True)
        # print("Here is the output")
        # print(output)
        return output