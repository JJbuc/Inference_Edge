import logging
import argparse
from typing import Dict, Optional
from fastapi import FastAPI, HTTPException
import uvicorn
import streamlit as st
from pydantic import BaseModel
import torch
import yaml
from inference_techniques.greedy import GreedyDecoder
from inference_techniques.beam_search import BeamSearchDecoder
from inference_techniques.top_k import TopKDecoder
from inference_techniques.top_p import TopPDecoder
from inference_techniques.temperature import TemperatureDecoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# FastAPI app
app = FastAPI(
    title="LLM Edge Inference Pipeline",
    description="Optimized edge inference for LLMs with multiple decoding strategies",
    version="1.0.0"
)

# Pydantic model for API input validation
class InferenceRequest(BaseModel):
    prompt: str
    strategy: str = "greedy"
    max_length: int = 50
    temperature: Optional[float] = 1.0
    top_k: Optional[int] = 50
    top_p: Optional[float] = 0.9
    num_beams: Optional[int] = 5

# Load configuration
def load_config(config_path: str = "config/config.yaml") -> Dict:
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        logger.info("Configuration loaded successfully")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {str(e)}")
        raise

# Placeholder for model loading
def load_model(config: Dict):
    # TODO: Add your model loading code here (e.g., Phi-3-mini with 4-bit quantization)
    # Example:
    # from transformers import AutoModelForCausalLM, AutoTokenizer
    # from bitsandbytes import quantize_model
    # model = AutoModelForCausalLM.from_pretrained(config["model_name"], load_in_4bit=True, device_map="auto")
    # tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    # return {"model": model, "tokenizer": tokenizer}
    logger.info("Model loading placeholder - replace with actual model loading")
    return {"model": None, "tokenizer": None}

# Inference strategy factory
def get_decoder(strategy: str, model, tokenizer, config: Dict):
    decoders = {
        "greedy": GreedyDecoder(model, tokenizer, config),
        "beam_search": BeamSearchDecoder(model, tokenizer, config),
        "top_k": TopKDecoder(model, tokenizer, config),
        "top_p": TopPDecoder(model, tokenizer, config),
        "temperature": TemperatureDecoder(model, tokenizer, config)
    }
    if strategy not in decoders:
        raise ValueError(f"Invalid strategy: {strategy}. Choose from {list(decoders.keys())}")
    return decoders[strategy]

# FastAPI endpoint for inference
@app.post("/generate", response_model=Dict[str, str])
async def generate(request: InferenceRequest) -> Dict[str, str]:
    config = load_config()
    model_dict = load_model(config)
    try:
        decoder = get_decoder(request.strategy, model_dict["model"], model_dict["tokenizer"], config)
        result = decoder.generate(
            prompt=request.prompt,
            max_length=request.max_length,
            temperature=request.temperature,
            top_k=request.top_k,
            top_p=request.top_p,
            num_beams=request.num_beams
        )
        logger.info(f"Generated text for prompt: {request.prompt[:50]}... using {request.strategy}")
        return {"generated_text": result}
    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

# Streamlit UI
def run_streamlit():
    st.set_page_config(page_title="LLM Edge Inference", page_icon="🤖")
    st.title("LLM Edge Inference Demo")
    st.markdown("Enter a prompt and select an inference strategy to generate text.")

    # Input form
    prompt = st.text_area("Prompt", placeholder="Enter your prompt here...", height=100)
    strategy = st.selectbox("Inference Strategy", ["greedy", "beam_search", "top_k", "top_p", "temperature"])
    max_length = st.slider("Max Length", min_value=10, max_value=200, value=50, step=10)
    
    # Strategy-specific parameters
    col1, col2 = st.columns(2)
    with col1:
        temperature = st.slider("Temperature", 0.1, 2.0, 1.0, step=0.1) if strategy == "temperature" else 1.0
        top_k = st.slider("Top-K", 10, 100, 50, step=10) if strategy == "top_k" else 50
    with col2:
        top_p = st.slider("Top-P", 0.1, 1.0, 0.9, step=0.1) if strategy == "top_p" else 0.9
        num_beams = st.slider("Num Beams", 2, 10, 5, step=1) if strategy == "beam_search" else 5

    if st.button("Generate", type="primary"):
        if not prompt:
            st.error("Please enter a prompt.")
            return
        try:
            config = load_config()
            model_dict = load_model(config)
            decoder = get_decoder(strategy, model_dict["model"], model_dict["tokenizer"], config)
            with st.spinner("Generating..."):
                result = decoder.generate(
                    prompt=prompt,
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    num_beams=num_beams
                )
            st.success("Generated Text:")
            st.markdown(f"**{result}**")
        except Exception as e:
            st.error(f"Error: {str(e)}")
            logger.error(f"Streamlit inference failed: {str(e)}")

# Main function
def main():
    parser = argparse.ArgumentParser(description="LLM Edge Inference Pipeline")
    parser.add_argument("--mode", choices=["api", "streamlit"], default="api",
                        help="Run as FastAPI server or Streamlit UI")
    args = parser.parse_args()

    if args.mode == "api":
        logger.info("Starting FastAPI server on http://0.0.0.0:8000")
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        logger.info("Starting Streamlit UI")
        run_streamlit()

if __name__ == "__main__":
    main()