import yaml
# from fastapi import FastAPI, Request
from inference_techniques.greedy import GreedyDecoder
from inference_techniques.beam_search import BeamSearchDecoder
# from inference_techniques.top_k import TopKDecoder
# from inference_techniques.top_p import TopPDecoder
# from inference_techniques.temperature import TemperatureDecoder
from llm_handler import LLM_Handler

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
    if strategy == "beam_search":
        return BeamSearchDecoder(config)
    elif strategy == "greedy":
        return GreedyDecoder(config)
    elif strategy == "top_k":
        return TopKDecoder(config)
    elif strategy == "top_p":
        return TopPDecoder(config)
    elif strategy == "auto":
        return GreedyDecoder(config)
    elif strategy == "contrastive":
        return ContrastiveDecoder(config)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def main():
    config = load_config()
    llm_handler = LLM_Handler(config)
    strategy = config.get("strategy", "auto")
    decoder = get_decoder(strategy, config)
    max_length = config.get("default_params", {}).get("max_length", 1000)

    prompt = input("Enter your prompt: ")

    if decoder:
        output = decoder.generate(prompt, max_length=max_length)
    else:
        output = llm_handler.standard_inference(prompt, max_length=max_length)

    print("\nModel Output:\n", output)

    llm_handler.clear_space()

if __name__ == "__main__":
    main()
