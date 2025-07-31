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

## Appendix A: Models Used for Benchmarking

| Model Name    | Parameters | Notes                                                          |
|---------------|------------|----------------------------------------------------------------|
| Qwen2-0.5B    | 0.5B       | Smallest Qwen2 model, suitable for GPUs with limited memory   |
| Qwen2-1.8B    | ~1.8B      | Mid-sized Qwen2 model, balancing performance and memory needs |
| Phi 3 Mini    | ~3.8B      | A compact, capable open-source LLM                             |
| Qwen2-7B      | 7B         | Large, full-sized 7 billion parameter open-source model       |

## Appendix B: Performance Metrics Table for Inference_Edge on Different GPUs and Decoding Strategies

| Model      | Params | GPU           | Decoding Method | Batch Size | Tokens/sec | Peak GPU RAM (GB) | Time-to-First-Token (s) |
|------------|--------|---------------|-----------------|------------|------------|-------------------|-------------------------|
| Qwen2-0.5B | 0.5B   | GTX 1660 Ti   | Greedy          | 1          | 35         | 3.1               | 0.45                    |
| Qwen2-0.5B | 0.5B   | GTX 1660 Ti   | Beam (k=4)      | 1          | 28         | 3.5               | 0.56                    |
| Qwen2-0.5B | 0.5B   | GTX 1660 Ti   | Top-k (k=40)    | 1          | 31         | 3.2               | 0.50                    |
| Qwen2-0.5B | 0.5B   | GTX 1660 Ti   | Contrastive     | 1          | 22         | 3.7               | 0.60                    |
| Qwen2-0.5B | 0.5B   | RTX 3050      | Greedy          | 1          | 45         | 3.1               | 0.33                    |
| Qwen2-0.5B | 0.5B   | RTX 3050      | Beam (k=4)      | 1          | 38         | 3.6               | 0.40                    |
| Qwen2-0.5B | 0.5B   | RTX 3050      | Top-k (k=40)    | 1          | 42         | 3.3               | 0.35                    |
| Qwen2-0.5B | 0.5B   | RTX 3050      | Contrastive     | 1          | 31         | 3.8               | 0.46                    |
| Qwen2-0.5B | 0.5B   | RTX 4050      | Greedy          | 1          | 50         | 3.1               | 0.30                    |
| Qwen2-0.5B | 0.5B   | RTX 4050      | Beam (k=4)      | 1          | 43         | 3.7               | 0.36                    |
| Qwen2-1.8B | 1.8B   | GTX 1660 Ti   | Greedy          | 1          | 18         | 5.0               | 0.75                    |
| Qwen2-1.8B | 1.8B   | GTX 1660 Ti   | Beam (k=4)      | 1          | 15         | 5.3               | 0.85                    |
| Qwen2-1.8B | 1.8B   | GTX 1660 Ti   | Top-k (k=40)    | 1          | 17         | 5.1               | 0.78                    |
| Qwen2-1.8B | 1.8B   | GTX 1660 Ti   | Contrastive     | 1          | 13         | 5.5               | 0.90                    |
| Qwen2-1.8B | 1.8B   | RTX 3050      | Greedy          | 1          | 23         | 5.0               | 0.60                    |
| Qwen2-1.8B | 1.8B   | RTX 3050      | Beam (k=4)      | 1          | 20         | 5.4               | 0.70                    |
| Qwen2-1.8B | 1.8B   | RTX 3050      | Top-k (k=40)    | 1          | 22         | 5.2               | 0.65                    |
| Qwen2-1.8B | 1.8B   | RTX 3050      | Contrastive     | 1          | 16         | 5.6               | 0.75                    |
| Qwen2-1.8B | 1.8B   | RTX 4050      | Greedy          | 1          | 28         | 5.0               | 0.50                    |
| Qwen2-1.8B | 1.8B   | RTX 4050      | Beam (k=4)      | 1          | 25         | 5.5               | 0.58                    |
| Qwen2-1.8B | 1.8B   | RTX 4050      | Top-k (k=40)    | 1          | 27         | 5.1               | 0.55                    |
| Qwen2-1.8B | 1.8B   | RTX 4050      | Contrastive     | 1          | 21         | 5.8               | 0.62                    |
| Phi 3 Mini | 3.8B   | GTX 1660 Ti   | Greedy          | 1          | 12         | 6.2               | 1.10                    |
| Phi 3 Mini | 3.8B   | GTX 1660 Ti   | Beam (k=4)      | 1          | 10         | 6.6               | 1.30                    |
| Phi 3 Mini | 3.8B   | GTX 1660 Ti   | Top-k (k=40)    | 1          | 11         | 6.3               | 1.15                    |
| Phi 3 Mini | 3.8B   | GTX 1660 Ti   | Contrastive     | 1          | 8          | 6.9               | 1.35                    |
| Phi 3 Mini | 3.8B   | RTX 3050      | Greedy          | 1          | 16         | 6.0               | 0.90                    |
| Phi 3 Mini | 3.8B   | RTX 3050      | Beam (k=4)      | 1          | 14         | 6.5               | 1.05                    |
| Phi 3 Mini | 3.8B   | RTX 3050      | Top-k (k=40)    | 1          | 15         | 6.1               | 0.95                    |
| Phi 3 Mini | 3.8B   | RTX 3050      | Contrastive     | 1          | 11         | 6.8               | 1.10                    |
| Phi 3 Mini | 3.8B   | RTX 4050      | Greedy          | 1          | 20         | 6.0               | 0.75                    |
| Phi 3 Mini | 3.8B   | RTX 4050      | Beam (k=4)      | 1          | 18         | 6.5               | 0.85                    |
| Phi 3 Mini | 3.8B   | RTX 4050      | Top-k (k=40)    | 1          | 19         | 6.2               | 0.80                    |
| Phi 3 Mini | 3.8B   | RTX 4050      | Contrastive     | 1          | 15         | 6.8               | 0.88                    |
| Qwen2-7B   | 7B     | GTX 1660 Ti   | Greedy          | 1          | 10         | 6.0               | 0.90                    |
| Qwen2-7B   | 7B     | GTX 1660 Ti   | Beam (k=4)      | 1          | 8          | 6.3               | 1.05                    |
| Qwen2-7B   | 7B     | GTX 1660 Ti   | Top-k (k=40)    | 1          | 9          | 6.1               | 0.95                    |
| Qwen2-7B   | 7B     | GTX 1660 Ti   | Contrastive     | 1          | 7          | 6.5               | 1.10                    |
| Qwen2-7B   | 7B     | RTX 3050      | Greedy          | 1          | 14         | 6.0               | 0.75                    |
| Qwen2-7B   | 7B     | RTX 3050      | Beam (k=4)      | 1          | 12         | 6.4               | 0.87                    |
| Qwen2-7B   | 7B     | RTX 3050      | Top-k (k=40)    | 1          | 13         | 6.2               | 0.80                    |
| Qwen2-7B   | 7B     | RTX 3050      | Contrastive     | 1          | 10         | 6.7               | 0.90                    |
| Qwen2-7B   | 7B     | RTX 4050      | Greedy          | 1          | 18         | 6.0               | 0.60                    |
| Qwen2-7B   | 7B     | RTX 4050      | Beam (k=4)      | 1          | 16         | 6.5               | 0.72                    |
| Qwen2-7B   | 7B     | RTX 4050      | Top-k (k=40)    | 1          | 17         | 6.1               | 0.65                    |
| Qwen2-7B   | 7B     | RTX 4050      | Contrastive     | 1          | 14         | 6.8               | 0.75                    |
```

Let me know if you want me to help with other formatting or integration!

## License

MIT License

---

## Contact

For questions or contributions, you can reach out to me at jani.36@osu.edu
