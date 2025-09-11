
import datetime as dt
import json
import yaml
import os

from typing import Dict, List, Any
from PIL import PngImagePlugin


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def outputs_base_dir() -> str:
    return os.environ.get("OUTPUTS_DIR", os.path.abspath("outputs"))


def dated_output_dir() -> str:
    base = outputs_base_dir()
    day = dt.datetime.now().strftime("%Y-%m-%d")
    path = os.path.join(base, day)
    ensure_dir(path)
    return path


def save_images_with_metadata(images: List, metadata: Dict[str, Any]) -> List[str]:
    out_dir = dated_output_dir()
    saved_paths = []

    pnginfo = PngImagePlugin.PngInfo()
    try:
        prompt_line = (
            f"Prompt: {metadata.get('prompt','')}\n"
            f"Negative prompt: {metadata.get('negative_prompt','')}\n"
            f"Steps: {metadata.get('steps')}\nCFG scale: {metadata.get('cfg_scale')}\n"
            f"Sampler: {metadata.get('sampler')}\nSeed: {metadata.get('seed')}\nSize: {metadata.get('width')}x{metadata.get('height')}\n"
            f"Model: {metadata.get('model_id','')}\n"
        )
        pnginfo.add_text("parameters", prompt_line)
    except Exception:
        pass

    json_name = dt.datetime.now().strftime("%H%M%S") + "_metadata.json"
    json_path = os.path.join(out_dir, json_name)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)

    for idx, img in enumerate(images):
        file_name = dt.datetime.now().strftime("%H%M%S") + f"_{idx:02d}.png"
        file_path = os.path.join(out_dir, file_name)
        try:
            img.save(file_path, pnginfo=pnginfo)
        except Exception:
            img.save(file_path)
        saved_paths.append(file_path)

    return saved_paths


CONFIG_PATH = os.environ.get("SDWEBUI_CONFIG", os.path.abspath("config.yaml"))

def load_config() -> Dict[str, Any]:
    if not os.path.exists(CONFIG_PATH):
        return {}
    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_config(cfg: Dict[str, Any]):
    ensure_dir(os.path.dirname(CONFIG_PATH) or ".")
    with open(CONFIG_PATH, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
