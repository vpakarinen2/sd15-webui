import torch


def get_device_and_dtype():
    """Return (device, dtype) choosing CUDA fp16 when available."""
    if torch.cuda.is_available():
        device = torch.device("cuda")
        dtype = torch.float16
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        except Exception:
            pass
    else:
        device = torch.device("cpu")
        dtype = torch.float32
    return device, dtype


def apply_performance_optimizations(pipe, prefer_sdpa: bool = True):
    """Apply memory/perf optimizations to Diffusers pipeline."""
    try:
        pipe.enable_vae_slicing()
    except Exception:
        pass

    if prefer_sdpa:
        try:
            pipe.enable_attention_slicing("auto")
        except Exception:
            pass
    else:
        try:
            pipe.enable_attention_slicing("auto")
        except Exception:
            pass
    return pipe
