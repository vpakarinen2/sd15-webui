import gradio as gr
import time
import os

from sdwebui.core.generation import generate_txt2img, CancelToken
from sdwebui.core.persistence import save_images_with_metadata
from sdwebui.core.lora import scan_loras, default_lora_dir
from sdwebui.core.schedulers import available_samplers
from sdwebui.core.model_manager import ModelManager

RES_PRESETS = ["512x512", "512x768", "768x512"]
DEFAULT_WIDTH, DEFAULT_HEIGHT = 512, 512
DEFAULT_SAMPLER = "DPM++ 2M Karras"

manager = ModelManager()
cancel_token = CancelToken()


def setup_models_if_needed():
    try:
        manager.ensure_loaded()
        return "Models loaded and ready."
    except Exception as e:
        return f"Model load error: {e}. Ensure you accepted the SD 1.5 license and provided a valid HF token."


def set_hf_token(token: str, persist: bool):
    if not token:
        return "Please provide a token."
    try:
        manager.set_hf_token(token, persist=persist)
        msg = setup_models_if_needed()
        return f"Token saved. {msg}"
    except Exception as e:
        return f"Failed to set token: {e}"


def on_preset_change(preset: str):
    try:
        w, h = preset.split("x")
        return int(w), int(h)
    except Exception:
        return DEFAULT_WIDTH, DEFAULT_HEIGHT


def refresh_lora_dropdowns(lora_dir: str):
    choices = scan_loras(lora_dir)
    return (
        gr.update(choices=choices, value=None),
        gr.update(choices=choices, value=None),
        gr.update(choices=choices, value=None),
    )


def upload_lora_files(files, lora_dir: str):
    os.makedirs(lora_dir, exist_ok=True)
    saved = 0
    if files is None:
        files = []
    for f in files:
        try:
            src = f.name if hasattr(f, "name") else str(f)
            dst = os.path.join(lora_dir, os.path.basename(src))
            with open(src, "rb") as rf, open(dst, "wb") as wf:
                wf.write(rf.read())
            saved += 1
        except Exception:
            pass
    updates = refresh_lora_dropdowns(lora_dir)
    status = f"Uploaded {saved} file(s) to {lora_dir}."
    return (status, *updates)


def generate(
    prompt, negative_prompt, steps, cfg_scale, sampler, preset, width, height, seed, batch_size,
    hires_fix, hires_scale, hires_steps, hires_denoise,
    lora_dir, lora1, lora1_w, lora2, lora2_w, lora3, lora3_w,
    progress=gr.Progress(track_tqdm=False)
):
    cancel_token.reset()
    if preset in RES_PRESETS:
        width, height = on_preset_change(preset)

    t0 = time.time()
    try:
        loras = []
        for name, w in ((lora1, lora1_w), (lora2, lora2_w), (lora3, lora3_w)):
            if name and float(w) > 0:
                path = os.path.join(lora_dir, name)
                loras.append((path, float(w)))

        pipe_txt2img, pipe_img2img = manager.ensure_loaded()
        if loras:
            manager.apply_loras(loras)
        else:
            manager.apply_loras([])
        images, used_seed = generate_txt2img(
            pipe_txt2img=pipe_txt2img,
            pipe_img2img=pipe_img2img,
            prompt=prompt,
            negative_prompt=negative_prompt,
            steps=int(steps),
            cfg_scale=float(cfg_scale),
            sampler_name=sampler,
            width=int(width),
            height=int(height),
            seed=int(seed),
            batch_size=int(batch_size),
            hires_fix=bool(hires_fix),
            hires_scale=float(hires_scale),
            hires_steps=int(hires_steps),
            hires_denoise=float(hires_denoise),
            cancel=cancel_token,
            progress_callback=lambda frac: progress(frac),
        )
        t1 = time.time()
        meta = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "steps": int(steps),
            "cfg_scale": float(cfg_scale),
            "sampler": sampler,
            "width": int(width),
            "height": int(height),
            "seed": int(used_seed),
            "batch_size": int(batch_size),
            "hires_fix": bool(hires_fix),
            "hires_scale": float(hires_scale),
            "hires_steps": int(hires_steps),
            "hires_denoise": float(hires_denoise),
            "loras": [
                {"file": os.path.basename(p), "dir": os.path.abspath(os.path.dirname(p)), "weight": w}
                for (p, w) in loras
            ],
            "model_id": manager.model_id,
            "elapsed_sec": round(t1 - t0, 2),
        }
        saved = save_images_with_metadata(images, meta)
        status = f"Generated {len(images)} image(s) in {meta['elapsed_sec']}s. Saved to: {os.path.dirname(saved[0])}"
        return saved, status
    except Exception as e:
        return [], f"Generation failed: {e}"


def cancel_generation():
    cancel_token.cancel()
    return "Cancellation requested. Attempting to stop the current job..."


def build_ui():
    with gr.Blocks(theme=gr.themes.Default()) as grsd:
        gr.Markdown("# Stable Diffusion 1.5 - Txt2Img")

        with gr.Accordion("Setup & Models", open=False):
            token_box = gr.Textbox(label="Hugging Face Token", type="password", placeholder="hf_...")
            persist_token = gr.Checkbox(label="Persist token to config.yaml", value=True)
            save_btn = gr.Button("Save Token & Load Models")
            load_status = gr.Markdown()
            save_btn.click(set_hf_token, inputs=[token_box, persist_token], outputs=load_status)

            refresh_btn = gr.Button("Refresh/Load Models")
            refresh_btn.click(setup_models_if_needed, outputs=load_status)
            
        with gr.Row():
            with gr.Column(scale=2):
                prompt = gr.Textbox(label="Prompt", lines=4)
                negative_prompt = gr.Textbox(label="Negative Prompt", lines=2)
                with gr.Accordion("Generation Settings", open=True):
                    with gr.Row():
                        steps = gr.Slider(1, 100, value=20, step=1, label="Steps")
                        cfg = gr.Slider(1.0, 20.0, value=7.0, step=0.5, label="CFG Scale")
                    with gr.Row():
                        sampler = gr.Dropdown(choices=available_samplers(), value=DEFAULT_SAMPLER, label="Sampler")
                        preset = gr.Dropdown(choices=RES_PRESETS, value="512x512", label="Resolution")
                    with gr.Row():
                        width = gr.Number(value=DEFAULT_WIDTH, precision=0, label="Width")
                        height = gr.Number(value=DEFAULT_HEIGHT, precision=0, label="Height")
                    with gr.Row():
                        seed = gr.Number(value=-1, precision=0, label="Seed (-1 random)")
                        batch_size = gr.Slider(1, 4, value=1, step=1, label="Batch Size")

                with gr.Accordion("LoRA", open=False):
                    lora_dir = gr.Textbox(label="LoRA Directory", value=default_lora_dir())
                    refresh_lora_btn = gr.Button("Refresh LoRAs")
                    lora_choices = scan_loras(default_lora_dir())
                    with gr.Row():
                        lora1 = gr.Dropdown(choices=lora_choices, label="LoRA 1")
                        lora1_w = gr.Slider(0.0, 2.0, value=0.8, step=0.05, label="Weight 1")
                    with gr.Row():
                        lora2 = gr.Dropdown(choices=lora_choices, label="LoRA 2")
                        lora2_w = gr.Slider(0.0, 2.0, value=0.0, step=0.05, label="Weight 2")
                    with gr.Row():
                        lora3 = gr.Dropdown(choices=lora_choices, label="LoRA 3")
                        lora3_w = gr.Slider(0.0, 2.0, value=0.0, step=0.05, label="Weight 3")
                    lora_files = gr.Files(label="Upload LoRA files")
                    lora_status = gr.Markdown()
                    upload_lora_btn = gr.Button("Upload & Refresh")

                with gr.Accordion("Hi-Res Fix", open=False):
                    hires_fix = gr.Checkbox(value=True, label="Enable Hi-Res Fix")
                    with gr.Row():
                        hires_scale = gr.Slider(1.2, 2.0, value=1.5, step=0.05, label="Hi-Res Scale")
                        hires_steps = gr.Slider(1, 60, value=15, step=1, label="Hi-Res Steps")
                    hires_denoise = gr.Slider(0.05, 1.0, value=0.25, step=0.01, label="Hi-Res Denoise Strength")

                with gr.Row():
                    gen_btn = gr.Button("Generate", variant="primary")
                    cancel_btn = gr.Button("Cancel")

            with gr.Column(scale=1):
                gallery = gr.Gallery(label="Results", columns=2, height=600)
                status = gr.Markdown()

        preset.change(on_preset_change, inputs=preset, outputs=[width, height])
        refresh_lora_btn.click(
            refresh_lora_dropdowns,
            inputs=[lora_dir],
            outputs=[lora1, lora2, lora3],
        )
        lora_dir.change(
            refresh_lora_dropdowns,
            inputs=[lora_dir],
            outputs=[lora1, lora2, lora3],
        )
        upload_lora_btn.click(
            upload_lora_files,
            inputs=[lora_files, lora_dir],
            outputs=[lora_status, lora1, lora2, lora3],
        )
        gen_btn.click(
            generate,
            inputs=[
                prompt, negative_prompt, steps, cfg, sampler, preset, width, height, seed, batch_size,
                hires_fix, hires_scale, hires_steps, hires_denoise,
                lora_dir, lora1, lora1_w, lora2, lora2_w, lora3, lora3_w
            ],
            outputs=[gallery, status],
        )
        cancel_btn.click(cancel_generation, outputs=status)
        _ = gr.Markdown(value=setup_models_if_needed())

    return grsd
