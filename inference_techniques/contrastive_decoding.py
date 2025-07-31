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
        self.temperature = self.default_params.get("temperature", 1.0)
    def get_log_probs(self, handler, input_ids, temperature):
        with torch.no_grad():
            outputs = handler.model(input_ids)
            logits = outputs.logits[:, -1, :]/ temperature 
            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        return log_probs.cpu().numpy()

    def generate(self, prompt, max_length=None, top_k=10, **override_params):
        if max_length is None:
            max_length = self.default_params.get("max_length", 1000)
        device = next(self.amateur_handler.model.parameters()).device
        input_ids = self.amateur_handler.tokenizer(prompt, return_tensors="pt").input_ids.to(device)
        generated_ids = input_ids.tolist()[0]
        for _ in range(max_length):
            ids_tensor = torch.tensor([generated_ids], device=device)
            ama_log_probs = self.get_log_probs(self.amateur_handler, ids_tensor, self.temperature)
            exp_device = next(self.expert_handler.model.parameters()).device
            ids_tensor_exp = torch.tensor([generated_ids], device=exp_device)
            exp_log_probs = self.get_log_probs(self.expert_handler, ids_tensor_exp, self.temperature)

            # Get top-k candidate tokens from expert
            topk_indices = np.argpartition(exp_log_probs[0], -top_k)[-top_k:]
            # Compute contrastive scores only for top-k candidates
            contrastive_scores = exp_log_probs[0][topk_indices] / (ama_log_probs[0][topk_indices] + 1e-9)
            contrastive_scores = contrastive_scores * self.head_parameter
            # Select the best candidate
            next_token_id = int(topk_indices[np.argmax(contrastive_scores)])
            generated_ids.append(next_token_id)

            # Print the token as soon as it is generated
            print(self.amateur_handler.tokenizer.decode([next_token_id], skip_special_tokens=True), end='', flush=True)

            if next_token_id == self.amateur_handler.tokenizer.eos_token_id:
                break
        # output = self.amateur_handler.tokenizer.decode(generated_ids, skip_special_tokens=True)
        # return output