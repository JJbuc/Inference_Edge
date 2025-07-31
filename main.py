import yaml
# from fastapi import FastAPI, Request
from inference_techniques.greedy import GreedyDecoder
from inference_techniques.beam_search import BeamSearchDecoder
from llm_handler import LLM_Handler
from clear_space import clear_space

# Load config
def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Decoder factory
from inference_techniques.beam_search import BeamSearchDecoder
from inference_techniques.greedy import GreedyDecoder
from inference_techniques.top_k import TopKDecoder  
from inference_techniques.top_p import TopPDecoder
from inference_techniques.contrastive_decoding import ContrastiveDecoder

def get_decoder(strategy, config):
    if strategy == "contrastive":
        return ContrastiveDecoder(config)
    elif strategy == "beam_search":
        return BeamSearchDecoder(config)
    elif strategy == "greedy":
        return GreedyDecoder(config)
    elif strategy == "top_k":
        return TopKDecoder(config)
    elif strategy == "top_p":
        return TopPDecoder(config)
    elif strategy == "auto":
        return GreedyDecoder(config)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
def update_config_temperature(config_path, mode):
    config = load_config(config_path)
    if mode == "focused":
        config["default_params"]["temperature"] = 0.7
    elif mode == "exploratory":
        config["default_params"]["temperature"] = 1.5
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config

def main():
    config_path = "config/config.yaml"
    config = load_config(config_path)
    mode = config.get("mode", "exploratory").lower()
    config = update_config_temperature(config_path, mode)
    strategy = config.get("strategy", "auto")
    decoder = get_decoder(strategy, config)
    max_length = config.get("default_params", {}).get("max_length", 1000)

    prompt = input("Enter your prompt: ")

    if decoder:
        decoder.generate(prompt, max_length=max_length)
    else:
        llm_handler = LLM_Handler(config)
        llm_handler.standard_inference(prompt, max_length=max_length)

    clear_space()

if __name__ == "__main__":
    main()