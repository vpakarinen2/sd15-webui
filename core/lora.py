import os

from typing import List

SUPPORTED_EXTS = {".safetensors", ".bin", ".pt"}


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def default_lora_dir() -> str:
    return os.environ.get("LORA_DIR") or \
           ("/workspace/models/lora" if os.path.isdir("/workspace") else os.path.abspath("models/lora"))


def scan_loras(lora_dir: str) -> List[str]:
    """Return a list of LoRA weight filenames found under lora_dir (non-recursive)."""
    if not lora_dir:
        lora_dir = default_lora_dir()
    ensure_dir(lora_dir)
    files = []
    try:
        for name in os.listdir(lora_dir):
            p = os.path.join(lora_dir, name)
            if os.path.isfile(p) and os.path.splitext(name)[1].lower() in SUPPORTED_EXTS:
                files.append(name)
    except FileNotFoundError:
        ensure_dir(lora_dir)
    files.sort()
    return files


def adapter_name_from_filename(filename: str) -> str:
    base = os.path.splitext(os.path.basename(filename))[0]
    return "lora_" + "".join(ch if ch.isalnum() or ch in ("_", "-") else "_" for ch in base)
