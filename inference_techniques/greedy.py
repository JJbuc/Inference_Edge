from llm_handler import LLM_Handler

class GreedyDecoder(LLM_Handler):
    def generate(self, prompt, max_length=None, **override_params):
        if max_length is None:
            max_length = self.default_params.get("max_length", 1000)
        params = self.default_params.copy()
        params.update(override_params)
        print(f"\nGreedy decoding for prompt: '{prompt}' with max_length={max_length}")
        outputs = self.pipe(
            prompt,
            max_new_tokens=max_length
        )
        return outputs[0]['generated_text']