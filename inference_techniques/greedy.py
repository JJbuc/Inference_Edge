from llm_handler import LLM_Handler

class GreedyDecoder(LLM_Handler):
    def generate(self, prompt, max_length=None, **override_params):
        if max_length is None:
            max_length = self.default_params.get("max_length", 1000)
        params = self.default_params.copy()
        params.update(override_params)
        outputs = self.pipe(
            prompt,
            max_new_tokens=max_length
        )
        generated_text = outputs[0]['generated_text']
        print(generated_text)