from llm_handler import LLM_Handler

class TopKDecoder(LLM_Handler):
    def generate(self, prompt, max_length=None, top_k=None, **override_params):
        if max_length is None:
            max_length = self.default_params.get("max_length", 1000)
        if top_k is None:
            top_k = self.default_params.get("top_k", 50)
        params = self.default_params.copy()
        params.update(override_params)
        outputs = self.pipe(
            prompt,
            max_new_tokens=max_length,
            top_k=top_k,
            do_sample=True
        )
        generated_text = outputs[0]['generated_text']
        print(generated_text)