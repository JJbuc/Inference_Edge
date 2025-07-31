# Inference Edge

A modular framework for experimenting with decoding strategies for Large Language Models (LLMs), including greedy, beam search, top-k, top-p, and contrastive decoding.  
Supports streaming output, easy configuration, and efficient resource management.

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/JJbuc/Inference_Edge.git
cd Inference_Edge
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Configure Your Models

Edit `config/config.yaml` to set your models, decoding strategy, and parameters:

```yaml
model_name: "Qwen/Qwen2.5-3B-Instruct"
model_ama: "Qwen/Qwen2.5-Coder-0.5B-Instruct"
model_exp: "Qwen/Qwen2.5-1.5B-Instruct"
quantization: "4bit"
device: "cuda"
strategy: "contrastive"  # Options: auto, greedy, beam_search, top_k, top_p, contrastive
mode: "exploratory"      # or "focused"
stream: true             # true for streaming output, false for full output
head_parameter: 1.0      # for contrastive decoding
default_params:
  max_length: 1000
  temperature: 1.5       # Will be set automatically by mode
  top_k: 50
  top_p: 0.9
  num_beams: 3
```

- **mode:**  
  - `"exploratory"` sets a high temperature for creative outputs.
  - `"focused"` sets a low temperature for deterministic outputs.

- **stream:**  
  - `true`: Prints tokens as they are generated.
  - `false`: Prints the full output after generation.

### 4. Run Inference

```bash
python main.py
```

You will be prompted to enter your input. The output will be streamed or printed based on your config.

---

## Decoding Strategies

- **Greedy:** Selects the highest probability token at each step.
- **Beam Search:** Explores multiple hypotheses for more robust completions.
- **Top-K:** Samples from the top K most probable tokens.
- **Top-P (Nucleus):** Samples from the smallest set of tokens whose cumulative probability exceeds P.
- **Contrastive:** Uses an expert and amateur model to filter and re-rank candidate tokens for more informative results.

---

## Contrastive Decoding

Contrastive decoding requires two models:
- **Amateur Model:** Set via `model_ama` in config.
- **Expert Model:** Set via `model_exp` in config.

The framework loads both models and uses the expert's top-k candidates, re-ranking them using the amateur model's probabilities.

---

## Streaming Output

Set `stream: true` in your config to see tokens printed as they are generated.  
Set `stream: false` to print the full output after generation.

---

## Resource Management

After inference, GPU/CPU memory is freed using the `clear_space` utility.  
No need to manually delete models or pipelines.

---

## Extending

- Add new decoding strategies by creating a new class in `inference_techniques/`.
- Register your decoder in `main.py`'s `get_decoder` function.

---

## Troubleshooting

- **CUDA/Device Errors:** Ensure your device is set correctly in config (`device: "cuda"` or `"cpu"`).
- **Model Loading Issues:** Check model names and availability on HuggingFace.
- **Streaming Limitations:** True streaming is only available for contrastive decoding; other decoders simulate streaming by splitting the output.

---

## Example Usage

```bash
python main.py
```
```
Enter your prompt: Why is the sky blue?
```
Output will be streamed or printed based on your config.

---

## License

MIT License

---

## Contact

For questions or contributions, you can reach out to me at jani.36@osu.edu