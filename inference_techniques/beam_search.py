from llm_handler import LLM_Handler
import time

class BeamSearchDecoder(LLM_Handler):
    def generate(self, prompt, max_length=None, num_beams=None, **override_params):
        if max_length is None:
            max_length = self.default_params.get("max_length", 1000)
        if num_beams is None:
            num_beams = self.default_params.get("num_beams", 3)
        params = self.default_params.copy()
        params.update(override_params)
        print(f"\nBeam search for prompt: '{prompt}' with num_beams={num_beams}, max_length={max_length}")
        start_inference_time = time.time()
        outputs = self.pipe(
            prompt,
            max_new_tokens=max_length,
            num_beams=num_beams,
            do_sample=False
        )
        end_inference_time = time.time()
        print(f"Beam search inference completed in {end_inference_time - start_inference_time:.2f} seconds.")
        return outputs[0]['generated_text']