from llm_handler import LLM_Handler

class GreedyDecoder:
    def __init__(self, config):
        # model and tokenizer are not used directly, but kept for interface consistency
        self.llm_handler = LLM_Handler(config)
        self.max_length = config.get("default_params", {}).get("max_length", 1000)

    def generate(self, prompt, max_length=None, **kwargs):
        # Override max_length if provided
        if max_length is None:
            max_length = self.max_length
        return self.llm_handler.standard_inference(prompt, max_length=max_length)