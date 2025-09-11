import torch

from diffusers import StableDiffusionPipeline, StableDiffusionImg2ImgPipeline
from .schedulers import make_scheduler
from typing import Optional, Tuple
from PIL import Image


class CancelToken:
    def __init__(self):
        self.cancelled = False

    def cancel(self):
        self.cancelled = True

    def reset(self):
        self.cancelled = False


def _make_generators(device, seed: int, batch_size: int):
    if seed is None or seed < 0:
        seed = torch.seed() % (2**31-1)
    gens = [torch.Generator(device=device).manual_seed(int(seed) + i) for i in range(batch_size)]
    return seed, gens


def _resize_to_multiple_of_8(w: int, h: int) -> Tuple[int, int]:
    w = int(w // 8 * 8)
    h = int(h // 8 * 8)
    return max(w, 8), max(h, 8)


def generate_txt2img(
    pipe_txt2img: StableDiffusionPipeline,
    pipe_img2img: StableDiffusionImg2ImgPipeline,
    prompt: str,
    negative_prompt: str,
    steps: int,
    cfg_scale: float,
    sampler_name: str,
    width: int,
    height: int,
    seed: int,
    batch_size: int,
    hires_fix: bool,
    hires_scale: float,
    hires_steps: int,
    hires_denoise: float,
    cancel: Optional[CancelToken] = None,
    progress_callback=None,
):
    device = pipe_txt2img._execution_device
    seed, generators = _make_generators(device, seed, batch_size)
    pipe_txt2img.scheduler = make_scheduler(pipe_txt2img, sampler_name)
    width, height = _resize_to_multiple_of_8(width, height)
    total_steps = steps + (hires_steps if hires_fix else 0)

    cur_step = 0

    def cb_txt2img(step, timestep, latents):
        nonlocal cur_step
        cur_step += 1
        if progress_callback and total_steps > 0:
            progress_callback(cur_step / total_steps)
        if cancel and cancel.cancelled:
            return True  # abort
        return False

    out1 = pipe_txt2img(
        prompt=prompt,
        negative_prompt=negative_prompt or None,
        guidance_scale=cfg_scale,
        num_inference_steps=steps,
        width=width,
        height=height,
        generator=generators,
        num_images_per_prompt=batch_size,
        callback=cb_txt2img,
        callback_steps=1,
    )
    images = out1.images

    if hires_fix:
        if cancel and cancel.cancelled:
            return images, seed
        pipe_img2img.scheduler = make_scheduler(pipe_img2img, sampler_name)

        target_w = int(width * hires_scale)
        target_h = int(height * hires_scale)
        target_w, target_h = _resize_to_multiple_of_8(target_w, target_h)

        upscaled = [img.resize((target_w, target_h), Image.LANCZOS) for img in images]
        img2img_gens = [torch.Generator(device=device).manual_seed(int(seed) + 1000 + i) for i in range(batch_size)]

        def cb_img2img(step, timestep, latents):
            nonlocal cur_step
            cur_step += 1
            if progress_callback and total_steps > 0:
                progress_callback(cur_step / total_steps)
            if cancel and cancel.cancelled:
                return True
            return False

        out2 = pipe_img2img(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            image=upscaled,
            strength=float(hires_denoise),
            guidance_scale=cfg_scale,
            num_inference_steps=hires_steps,
            generator=img2img_gens,
            callback=cb_img2img,
            callback_steps=1,
        )
        images = out2.images

    return images, seed
