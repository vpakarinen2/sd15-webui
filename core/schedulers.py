from typing import List

from diffusers import (
    DPMSolverMultistepScheduler,
    EulerDiscreteScheduler,
    EulerAncestralDiscreteScheduler,
    LMSDiscreteScheduler,
    HeunDiscreteScheduler,
    PNDMScheduler,
    DDIMScheduler,
)

SAMPLER_LABELS = [
    "DPM++ 2M Karras",
    "Euler",
    "Euler a",
    "LMS",
    "Heun",
    "PNDM",
    "DDIM",
]


def available_samplers() -> List[str]:
    return SAMPLER_LABELS


def make_scheduler(pipe, name: str):
    cfg = pipe.scheduler.config
    n = name.strip().lower()
    if n.startswith("dpm++ 2m"):
        return DPMSolverMultistepScheduler.from_config(cfg, use_karras_sigmas=True, algorithm_type="dpmsolver++")
    if n == "euler":
        return EulerDiscreteScheduler.from_config(cfg)
    if n == "euler a":
        return EulerAncestralDiscreteScheduler.from_config(cfg)
    if n == "lms":
        return LMSDiscreteScheduler.from_config(cfg)
    if n == "heun":
        return HeunDiscreteScheduler.from_config(cfg)
    if n == "pndm":
        return PNDMScheduler.from_config(cfg)
    if n == "ddim":
        return DDIMScheduler.from_config(cfg)
    return DPMSolverMultistepScheduler.from_config(cfg, use_karras_sigmas=True, algorithm_type="dpmsolver++")
