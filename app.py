import gradio as gr
import time
import os
import shutil
import requests

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from urllib.parse import urlparse

from core.generation import generate_txt2img, CancelToken
from core.persistence import save_images_with_metadata
from core.lora import scan_loras, default_lora_dir, adapter_name_from_filename
from core.schedulers import available_samplers
from core.model_manager import ModelManager

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


def _ckpt_dir() -> str:
    return os.environ.get("SD_CKPT_DIR", os.path.join(os.getcwd(), "models", "checkpoints"))


def set_custom_ckpt(path: str, persist: bool):
    if not path:
        return "Please provide a checkpoint path."
    try:
        manager.set_custom_checkpoint(path, persist=persist)
        return f"Custom checkpoint set: {path}. Pipelines will reload on next generate."
    except Exception as e:
        return f"Failed to set checkpoint: {e}"


def clear_custom_ckpt(persist: bool):
    try:
        manager.clear_custom_checkpoint(persist=persist)
        return "Cleared custom checkpoint. Reverting to base model on next generate."
    except Exception as e:
        return f"Failed to clear checkpoint: {e}"


def upload_checkpoint_files(files, persist: bool):
    os.makedirs(_ckpt_dir(), exist_ok=True)
    saved = []
    if files is None:
        files = []
    for f in files:
        try:
            src = f.name if hasattr(f, "name") else str(f)
            dst = os.path.join(_ckpt_dir(), os.path.basename(src))
            with open(src, "rb") as rf, open(dst, "wb") as wf:
                wf.write(rf.read())
            saved.append(dst)
        except Exception:
            pass
    status = f"Uploaded {len(saved)} file(s) to {_ckpt_dir()}"
    new_path = saved[0] if saved else None
    if new_path:
        try:
            manager.set_custom_checkpoint(new_path, persist=persist)
            status += f". Using: {new_path}"
        except Exception as e:
            status += f". Failed to set checkpoint: {e}"
    return status, (gr.update(value=new_path) if new_path else gr.update())


def _guess_filename_from_url(url: str) -> str:
    try:
        path = urlparse(url).path
        name = os.path.basename(path)
        if name:
            return name
    except Exception:
        pass
    return "model.safetensors"


def _filename_from_response(resp, fallback: str) -> str:
    try:
        cd = resp.headers.get("content-disposition") or resp.headers.get("Content-Disposition")
        if cd and "filename=" in cd:
            fname = cd.split("filename=")[-1].strip().strip('"')
            if fname:
                return fname
    except Exception:
        pass
    return fallback


def _requests_session() -> requests.Session:
    retry = Retry(
        total=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET", "HEAD"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry, pool_connections=4, pool_maxsize=10)
    s = requests.Session()
    s.headers.update({"User-Agent": "sd-webui/1.0"})
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


def download_ckpt_from_url(url: str, persist: bool):
    if not url or not (url.startswith("http://") or url.startswith("https://")):
        return "Provide a valid http(s) URL.", gr.update()
    os.makedirs(_ckpt_dir(), exist_ok=True)
    filename = _guess_filename_from_url(url)
    dst = os.path.join(_ckpt_dir(), filename)
    tmp = dst + ".part"
    try:
        s = _requests_session()
        try:
            h = s.head(url, allow_redirects=True, timeout=(10, 20))
            if h.ok:
                filename = _filename_from_response(h, filename)
                dst = os.path.join(_ckpt_dir(), filename)
                tmp = dst + ".part"
        except Exception:
            pass

        resume_bytes = 0
        if os.path.exists(tmp):
            try:
                resume_bytes = os.path.getsize(tmp)
            except Exception:
                resume_bytes = 0
        headers = {"Range": f"bytes={resume_bytes}-"} if resume_bytes > 0 else {}

        with s.get(url, stream=True, timeout=(10, 120), headers=headers, allow_redirects=True) as r:
            if r.status_code == 416:
                if os.path.exists(tmp):
                    os.remove(tmp)
                resume_bytes = 0
                headers = {}
                r.close()
                r = s.get(url, stream=True, timeout=(10, 120), headers=headers, allow_redirects=True)
            r.raise_for_status()

            filename2 = _filename_from_response(r, filename)
            if filename2 != filename:
                dst = os.path.join(_ckpt_dir(), filename2)
                tmp = dst + ".part"
            mode = "ab" if resume_bytes > 0 else "wb"
            with open(tmp, mode) as f:
                for chunk in r.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
            os.replace(tmp, dst)
        manager.set_custom_checkpoint(dst, persist=persist)
        return f"Downloaded and set checkpoint: {dst}", gr.update(value=dst)
    except Exception as e:
        try:
            if os.path.exists(tmp) and os.path.getsize(tmp) == 0:
                os.remove(tmp)
        except Exception:
            pass
        return f"Failed to download from URL: {e}", gr.update()


def download_ckpt_from_hf(repo_id: str, filename: str, revision: str, persist: bool):
    if not repo_id or not filename:
        return "Provide repo_id and filename.", gr.update()
    try:
        from huggingface_hub import hf_hub_download
    except Exception as e:
        return f"huggingface_hub not available: {e}", gr.update()
    os.makedirs(_ckpt_dir(), exist_ok=True)
    try:
        token = manager.get_hf_token()
        cached = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            revision=revision or None,
            token=token,
            resume_download=True,
        )
        dst = os.path.join(_ckpt_dir(), os.path.basename(filename))
        shutil.copy2(cached, dst)
        manager.set_custom_checkpoint(dst, persist=persist)
        return f"Downloaded from HF and set: {dst}", gr.update(value=dst)
    except Exception as e:
        return f"Failed to download from HF: {e}", gr.update()


def toggle_download_source(src: str):
    is_url = (src == "URL")
    return (
        gr.update(visible=is_url),
        gr.update(visible=not is_url),
    )


def generate(
    prompt, negative_prompt, steps, cfg_scale, sampler, preset, width, height, seed, batch_size,
    hires_fix, hires_scale, hires_steps, hires_denoise,
    lora_dir, lora1, lora1_w, lora2, lora2_w, lora3, lora3_w,
    add_lora_tags,
    lora_trigger1, lora_trigger2, lora_trigger3, add_lora_triggers,
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
        orig_prompt = prompt
        if add_lora_tags and loras:
            try:
                tags = []
                for (p, w) in loras:
                    name = adapter_name_from_filename(os.path.basename(p))
                    tags.append(f"<lora:{name}:{float(w):.2f}>")
                prompt = (" ".join(tags) + " " + (prompt or "")).strip()
            except Exception:
                pass
        if add_lora_triggers:
            triggers = []
            for tw in (lora_trigger1, lora_trigger2, lora_trigger3):
                if isinstance(tw, str) and tw.strip():
                    triggers.append(tw.strip())
            if triggers:
                prompt = (" ".join(triggers) + " " + (prompt or "")).strip()
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
            "original_prompt": orig_prompt,
            "effective_prompt": prompt,
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
    with gr.Blocks(theme=gr.themes.Default(), css=".btn-full button{width:100%; margin:8px 6px;}") as grsd:
        gr.Markdown("# SD 1.5 - T2I")

        with gr.Accordion("Setup & Models", open=False):
            token_box = gr.Textbox(label="Hugging Face Token", type="password", placeholder="hf_...")
            persist_token = gr.Checkbox(label="Persist token to config.yaml", value=True)
            save_btn = gr.Button("Save Token & Load Models")
            load_status = gr.Markdown()
            save_btn.click(set_hf_token, inputs=[token_box, persist_token], outputs=load_status)

            refresh_btn = gr.Button("Refresh/Load Models")
            refresh_btn.click(setup_models_if_needed, outputs=load_status)
            gr.Markdown("")
            gr.Markdown("### Model & Checkpoint")
            gr.Markdown(value=f"CKPT_DIR: {_ckpt_dir()}")
            ckpt_path = gr.Textbox(label="Custom Checkpoint Path", placeholder="/workspace/models/checkpoints/model.safetensors")
            persist_ckpt = gr.Checkbox(label="Persist path to config.yaml", value=True)
            with gr.Row():
                set_ckpt_btn = gr.Button("Use Checkpoint", variant="secondary")
                clear_ckpt_btn = gr.Button("Clear")
            ckpt_status = gr.Markdown()
            set_ckpt_btn.click(set_custom_ckpt, inputs=[ckpt_path, persist_ckpt], outputs=ckpt_status)
            clear_ckpt_btn.click(clear_custom_ckpt, inputs=[persist_ckpt], outputs=ckpt_status)
            with gr.Accordion("Upload Checkpoint", open=False):
                ckpt_files = gr.Files(label="Upload Checkpoint File(s)")
                upload_ckpt_btn = gr.Button("Upload to CKPT_DIR", variant="secondary", elem_classes=["btn-full"])
            upload_ckpt_btn.click(upload_checkpoint_files, inputs=[ckpt_files, persist_ckpt], outputs=[ckpt_status, ckpt_path])

            with gr.Accordion("Download Checkpoint", open=False):
                src_choice = gr.Radio(["URL", "Hugging Face"], value="URL", label="Source")
                with gr.Group(visible=True) as url_group:
                    url_box = gr.Textbox(label="Direct URL (Civitai/HF)", placeholder="https://.../model.safetensors")
                    dl_url_btn = gr.Button("Download URL to CKPT_DIR", variant="secondary", elem_classes=["btn-full"])
                with gr.Group(visible=False) as hf_group:
                    with gr.Row():
                        repo_id = gr.Textbox(label="HF repo_id", placeholder="author/repo")
                        hf_filename = gr.Textbox(label="Filename", placeholder="model.safetensors")
                        hf_revision = gr.Textbox(label="Revision (optional)")
                    dl_hf_btn = gr.Button("Download HF to CKPT_DIR", variant="secondary", elem_classes=["btn-full"])
                dl_status = gr.Markdown()
                dl_url_btn.click(download_ckpt_from_url, inputs=[url_box, persist_ckpt], outputs=[dl_status, ckpt_path])
                dl_hf_btn.click(download_ckpt_from_hf, inputs=[repo_id, hf_filename, hf_revision, persist_ckpt], outputs=[dl_status, ckpt_path])
                src_choice.change(
                    toggle_download_source,
                    inputs=src_choice,
                    outputs=[url_group, hf_group],
                )
            
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
                    add_lora_tags = gr.Checkbox(value=True, label="Add tags to prompt")
                    with gr.Row():
                        lora_trigger1 = gr.Textbox(label="Trigger 1 (optional)")
                        lora_trigger2 = gr.Textbox(label="Trigger 2 (optional)")
                        lora_trigger3 = gr.Textbox(label="Trigger 3 (optional)")
                    add_lora_triggers = gr.Checkbox(value=False, label="Add trigger word to prompt")
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
                lora_dir, lora1, lora1_w, lora2, lora2_w, lora3, lora3_w, add_lora_tags,
                lora_trigger1, lora_trigger2, lora_trigger3, add_lora_triggers
            ],
            outputs=[gallery, status],
        )
        cancel_btn.click(cancel_generation, outputs=status)
        _ = gr.Markdown(value=setup_models_if_needed())

    return grsd
