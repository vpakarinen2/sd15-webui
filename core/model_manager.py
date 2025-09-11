import torch
import os

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline, AutoencoderKL
from .performance import get_device_and_dtype, apply_performance_optimizations
from .persistence import load_config, save_config, outputs_base_dir
from .lora import adapter_name_from_filename
from typing import Optional, Tuple
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except Exception:
    hf_hub_download = None
import requests

CACHE_DIR = os.environ.get("HUGGINGFACE_HUB_CACHE") or os.environ.get("HF_HOME") or None
DEFAULT_MODEL_ID = os.environ.get("SD15_MODEL_ID", "runwayml/stable-diffusion-v1-5")
DEFAULT_VAE_ID = os.environ.get("SD15_VAE_ID", "stabilityai/sd-vae-ft-ema")
DISABLE_SAFETY = os.environ.get("SD_ENABLE_SAFETY", "0") != "1"


class ModelManager:
    def __init__(self, model_id: str = DEFAULT_MODEL_ID, vae_id: str = DEFAULT_VAE_ID):
        self.model_id = model_id
        self.vae_id = vae_id
        self.pipe_txt2img: Optional[StableDiffusionPipeline] = None
        self.pipe_img2img: Optional[StableDiffusionImg2ImgPipeline] = None
        self.device, self.dtype = get_device_and_dtype()
        self._loaded_lora_adapters = set()
        self._custom_vae_file: Optional[str] = None
        self._upscaler_file: Optional[str] = None

    @staticmethod
    def get_hf_token() -> Optional[str]:
        cfg = load_config()
        token = os.environ.get("HUGGINGFACE_TOKEN") or cfg.get("hf_token")
        return token

    @staticmethod
    def set_hf_token(token: str, persist: bool = True):
        os.environ["HUGGINGFACE_TOKEN"] = token
        if persist:
            cfg = load_config()
            cfg["hf_token"] = token
            save_config(cfg)

    def _load_base_pipe(self, token: Optional[str]):
        pipe = StableDiffusionPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            use_safetensors=True,
            token=token,
            cache_dir=CACHE_DIR,
            safety_checker=None if DISABLE_SAFETY else None,
            feature_extractor=None if DISABLE_SAFETY else None,
        )
        if self._custom_vae_file and os.path.exists(self._custom_vae_file):
            try:
                vae = AutoencoderKL.from_single_file(self._custom_vae_file, torch_dtype=self.dtype)
            except Exception:
                vae = AutoencoderKL.from_pretrained(self.vae_id, torch_dtype=self.dtype, token=token, cache_dir=CACHE_DIR)
        else:
            vae = AutoencoderKL.from_pretrained(self.vae_id, torch_dtype=self.dtype, token=token, cache_dir=CACHE_DIR)
        pipe.vae = vae
        if DISABLE_SAFETY:
            try:
                pipe.safety_checker = None
            except Exception:
                pass
        pipe.to(self.device)
        pipe = apply_performance_optimizations(pipe, prefer_sdpa=True)
        return pipe

    def _load_img2img_pipe(self, token: Optional[str]):
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.model_id,
            torch_dtype=self.dtype,
            use_safetensors=True,
            token=token,
            cache_dir=CACHE_DIR,
            safety_checker=None if DISABLE_SAFETY else None,
            feature_extractor=None if DISABLE_SAFETY else None,
        )
        if self._custom_vae_file and os.path.exists(self._custom_vae_file):
            try:
                vae = AutoencoderKL.from_single_file(self._custom_vae_file, torch_dtype=self.dtype)
            except Exception:
                vae = AutoencoderKL.from_pretrained(self.vae_id, torch_dtype=self.dtype, token=token, cache_dir=CACHE_DIR)
        else:
            vae = AutoencoderKL.from_pretrained(self.vae_id, torch_dtype=self.dtype, token=token, cache_dir=CACHE_DIR)
        pipe.vae = vae
        if DISABLE_SAFETY:
            try:
                pipe.safety_checker = None
            except Exception:
                pass
        pipe.to(self.device)
        pipe = apply_performance_optimizations(pipe, prefer_sdpa=True)
        return pipe

    def ensure_loaded(self) -> Tuple[StableDiffusionPipeline, StableDiffusionImg2ImgPipeline]:
        token = self.get_hf_token()
        self._ensure_optional_assets(token)
        if self.pipe_txt2img is None:
            self.pipe_txt2img = self._load_base_pipe(token)
        if self.pipe_img2img is None:
            self.pipe_img2img = self._load_img2img_pipe(token)
        return self.pipe_txt2img, self.pipe_img2img
        
    def _ensure_optional_assets(self, token: Optional[str]):
        """Download user-requested assets if missing."""
        try:
            repo = os.environ.get("SD_EXTRA_REPO", "vpakarinen/sd-vae-upscale-ctrl")
            vae_file = os.environ.get("SD_CUSTOM_VAE_FILE", "vae-ft-mse-840000-ema-pruned.safetensors")
            upscaler_file = os.environ.get("SD_UPSCALER_FILE", "4x-UltraSharp.pth")

            target_dir = Path(os.environ.get("SD_ASSETS_DIR", os.path.join(outputs_base_dir(), "assets")))
            target_dir.mkdir(parents=True, exist_ok=True)

            # Try HF cached download first
            def get_from_hf(filename: str) -> Optional[str]:
                if hf_hub_download is None:
                    return None
                try:
                    return hf_hub_download(repo_id=repo, filename=filename, token=token, cache_dir=CACHE_DIR)
                except Exception:
                    return None

            def get_from_url(filename: str) -> Optional[str]:
                base = f"https://huggingface.co/{repo}/resolve/main/{filename}"
                dst = target_dir / filename
                try:
                    with requests.get(base, stream=True, timeout=60) as r:
                        r.raise_for_status()
                        with open(dst, "wb") as f:
                            for chunk in r.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                    return str(dst)
                except Exception:
                    return None

            vae_path = get_from_hf(vae_file) or get_from_url(vae_file)
            if vae_path and os.path.exists(vae_path):
                self._custom_vae_file = vae_path
                print(f"[Assets] Using custom VAE: {vae_path}")

            up_path = get_from_hf(upscaler_file) or get_from_url(upscaler_file)
            if up_path and os.path.exists(up_path):
                self._upscaler_file = up_path
                print(f"[Assets] Upscaler available: {up_path}")
        except Exception:
            pass

    def unload(self):
        for attr in ("pipe_txt2img", "pipe_img2img"):
            pipe = getattr(self, attr, None)
            if pipe is not None:
                try:
                    pipe.to("cpu")
                    del pipe
                except Exception:
                    pass
                setattr(self, attr, None)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def apply_loras(self, loras: list[tuple[str, float]]):
        """Apply a set of LoRAs to both txt2img and img2img pipelines."""
        txt2img, img2img = self.ensure_loaded()
        active = [(p, float(w)) for (p, w) in loras if p and float(w) > 0.0]
        if not active:
            for pipe in (txt2img, img2img):
                try:
                    pipe.unload_lora_weights()
                except Exception:
                    try:
                        pipe.disable_lora()
                    except Exception:
                        pass
            return

        adapter_names = []
        adapter_weights = []
        for path, weight in active:
            name = adapter_name_from_filename(path)
            adapter_names.append(name)
            adapter_weights.append(float(weight))
            for pipe in (txt2img, img2img):
                if name not in self._loaded_lora_adapters:
                    try:
                        pipe.load_lora_weights(path, adapter_name=name)
                    except Exception:
                        pipe.load_lora_weights(path, adapter_name=name)
        self._loaded_lora_adapters.update(adapter_names)

        for pipe in (txt2img, img2img):
            try:
                pipe.set_adapters(adapter_names, adapter_weights)
            except Exception:
                if adapter_names:
                    pipe.set_adapters(adapter_names[0], adapter_weights[0])
