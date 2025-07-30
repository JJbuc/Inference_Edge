import torch
import gc

def clear_space():
    print("\nClearing all models and GPU VRAM...")

    # Run garbage collection to free up memory
    gc.collect()

    # Empty CUDA cache if available
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("PyTorch CUDA cache emptied.")
    else:
        print("No CUDA device found, no cache to empty.")

    print("All models and GPU memory cleared.")