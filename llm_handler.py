import torch
from transformers import BitsAndBytesConfig, AutoModelForCausalLM, AutoTokenizer, pipeline
import sys
import time
import warnings
warnings.filterwarnings("ignore")
import yaml
import os 

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
      
class LLM_Handler():
    def __init__(self, config):
        self.model_name = config.get('model_name')
        if self.model_name is None:
            print('Error: Please provide a model name to initialize LLM_Handler, make changes in config file.')
            sys.exit(1)
        
        # BitsAndBytes configuration for 4-bit quantization
        quantization = config.get('quantization', '4bit')
        self.use_4bit = quantization == "4bit"
        self.bnb_4bit_compute_dtype = "float16"
        self.bnb_4bit_quant_type = "nf4"
        self.use_nested_quant = False
        
        self.compute_dtype = getattr(torch, self.bnb_4bit_compute_dtype)
        
        self.bnb_config = BitsAndBytesConfig(
            load_in_4bit=self.use_4bit,
            bnb_4bit_quant_type=self.bnb_4bit_quant_type,
            bnb_4bit_compute_dtype=self.compute_dtype,
            bnb_4bit_use_double_quant=self.use_nested_quant,
        )
        device = config.get('device', 'cuda')
        # Device mapping (assuming GPU is available at index 0)
        if device == "cuda" and torch.cuda.is_available():
            self.device_map = {"": 0}
        elif device == "cpu":
            self.device_map = {"": "cpu"}
        else:
            print("Warning: Requested device not available. Using CPU.")
            self.device_map = {"": "cpu"}

        print(f"Loading model: {self.model_name}...")
        start_load_time = time.time()
        
        # Load the model with quantization config
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=self.bnb_config,
            device_map=self.device_map
        )
        self.model.config.use_cache = True
        
        # Load the tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.default_params = config.get("default_params", {})
        
        # Initialize the text-generation pipeline once
        self.pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer
        )
        
        end_load_time = time.time()
        print(f'Model "{self.model_name}" initialized and loaded in {end_load_time - start_load_time:.2f} seconds.')

    def standard_inference(self, prompt: str, **override_params) -> str:
        params = self.default_params.copy()
        params.update(override_params)
        print(f"\nGenerating text for prompt: '{prompt}'")
        start_inference_time = time.time()
        
        outputs = self.pipe(
            prompt,
            max_new_tokens = self.default_params["max_length"]
        )
        end_inference_time = time.time()
        print(f"Inference completed in {end_inference_time - start_inference_time:.2f} seconds.")

        generated_text = outputs[0]['generated_text']
        print(generated_text)
    
# --- Main function to run the example ---
if __name__ == "__main__":
    config_path = os.path.join("config", "config.yaml")
    config = load_config(config_path)

    llm_handler = LLM_Handler(config)

    my_prompt = "Explain why the sky is blue."

    # Use params from config by default
    generated_text = llm_handler.standard_inference(my_prompt)

    print("\n--- Generated Text ---")
    print(generated_text)

    llm_handler.clear_space()