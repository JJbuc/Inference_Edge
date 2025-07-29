import yaml
from fastapi import FastAPI, Request
from inference_techniques.greedy import GreedyDecoder
# from inference_techniques.beam_search import BeamSearchDecoder
# from inference_techniques.top_k import TopKDecoder
# from inference_techniques.top_p import TopPDecoder
# from inference_techniques.temperature import TemperatureDecoder
from llm_handler import LLM_Handler

# Load config
def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

# Decoder factory
def get_decoder(strategy, llm_handler, config):
    if strategy == "greedy":
        return GreedyDecoder(config)
    # elif strategy == "beam_search":
    #     return BeamSearchDecoder(config)
    # Add other strategies here
    elif strategy == "auto":
        return None  # Use llm_handler's default inference
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

# FastAPI setup
app = FastAPI()

from pydantic import BaseModel

class InferRequest(BaseModel):
    prompt: str

@app.post("/infer")
async def infer(request: InferRequest):
    prompt = request.prompt
    config = load_config()
    llm_handler = LLM_Handler(config)
    strategy = config.get("strategy", "auto")

    decoder = get_decoder(strategy, llm_handler, config)
    max_length = config.get("default_params", {}).get("max_length", 1000)
    if decoder:
        output = decoder.generate(prompt, max_length=max_length)
    else:
        output = llm_handler.standard_inference(prompt, max_length=max_length)
    return {"output": output}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)