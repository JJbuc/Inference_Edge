from llm_handler import LLM_Handler

class TopPDecoder(LLM_Handler):
    def generate(self, prompt, max_length=None, top_p=None, temperature=None, **override_params):
        if max_length is None:
            max_length = self.default_params.get("max_length", 1000)
        if top_p is None:
            top_p = self.default_params.get("top_p", 0.9)
        if temperature is None:
            temperature = self.default_params.get("temperature", 1.0)
        params = self.default_params.copy()
        params.update(override_params)
        print(f"\nTop-p decoding for prompt: '{prompt}' with top_p={top_p}, temperature={temperature}, max_length={max_length}")
        outputs = self.pipe(
            prompt,
            max_new_tokens=max_length,
            top_p=top_p,
            temperature=temperature,
            do_sample=True
        )
        return outputs[0]['generated_text']