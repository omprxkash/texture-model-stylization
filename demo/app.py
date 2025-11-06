"""
Gradio interactive demo — two tabs:

  Tab 1 — Style Transfer
    Upload a content image + style image → choose method (Gatys/AdaIN/Johnson)
    → download stylized result

  Tab 2 — Tactile 3D
    Upload a texture image → generate heightmap → apply to sphere mesh
    → download .glb for viewing in browser / Blender

Usage:
  python demo/app.py
  python demo/app.py --share           # public Gradio link
  python demo/app.py --lora checkpoints/heightmap_lora/final
"""
import argparse
import io
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import gradio as gr
import numpy as np
import torch
from PIL import Image

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────
#  Cached model singletons
# ─────────────────────────────────────────────

_adain_model = None
_diffusion_pipeline = None
LORA_PATH = None


def get_adain():
    global _adain_model
    if _adain_model is None:
        from models.adain import AdaINStyleTransfer
        _adain_model = AdaINStyleTransfer(device=DEVICE)
    return _adain_model


def get_diffusion(lora_path=None):
    global _diffusion_pipeline
    if _diffusion_pipeline is None:
        from models.heightmap_diffusion import HeightmapDiffusionPipeline
        _diffusion_pipeline = HeightmapDiffusionPipeline(
            lora_weights_path=lora_path or LORA_PATH, device=DEVICE
        )
    return _diffusion_pipeline


def _pil_to_tensor(img: Image.Image, size: int = 512) -> torch.Tensor:
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])(img).unsqueeze(0).to(DEVICE)


def _tensor_to_pil(t: torch.Tensor) -> Image.Image:
    from torchvision import transforms
    return transforms.ToPILImage()(t.squeeze(0).clamp(0, 1).cpu())


# ─────────────────────────────────────────────
#  Style Transfer handlers
# ─────────────────────────────────────────────

def run_adain_transfer(content_pil, style_pil, alpha, image_size):
    if content_pil is None or style_pil is None:
        return None, "Upload both a content and style image."
    try:
        model = get_adain()
        c = _pil_to_tensor(content_pil, int(image_size))
        s = _pil_to_tensor(style_pil, int(image_size))
        result = model.transfer(c, s, alpha=float(alpha))
        return _tensor_to_pil(result), "Done."
    except Exception as e:
        return None, f"Error: {e}"


def run_gatys_transfer(content_pil, style_pil, steps, content_w, style_w, image_size, progress=gr.Progress()):
    if content_pil is None or style_pil is None:
        return None, "Upload both a content and style image."
    try:
        from models.vgg_style_transfer import VGGStyleTransfer
        from torchvision import transforms

        nst = VGGStyleTransfer(content_weight=float(content_w), style_weight=float(style_w), device=DEVICE)

        def to_t(img):
            return transforms.Compose([
                transforms.Resize((int(image_size), int(image_size))),
                transforms.ToTensor(),
            ])(img).unsqueeze(0).to(DEVICE)

        progress(0, desc="Optimizing...")
        last_img = [None]

        def cb(step, loss, img):
            last_img[0] = img
            progress(step / int(steps), desc=f"Step {step}/{int(steps)} | loss={loss:.0f}")

        result = nst.transfer(to_t(content_pil), to_t(style_pil), num_steps=int(steps), callback=cb)
        return _tensor_to_pil(result), f"Done in {int(steps)} steps."
    except Exception as e:
        return None, f"Error: {e}"


def run_interpolation(img_a_pil, img_b_pil, alpha, mode, image_size):
    if img_a_pil is None or img_b_pil is None:
        return None, "Upload both images."
    try:
        from inference.interpolate_textures import interpolate_pixel, interpolate_adain
        alphas = [float(alpha)]
        size = int(image_size)
        if mode == "pixel":
            imgs = interpolate_pixel(img_a_pil, img_b_pil, alphas, size=size)
        else:
            imgs = interpolate_adain(img_a_pil, img_b_pil, alphas, size=size, device=DEVICE)
        return imgs[0], f"Interpolated at α={alpha:.2f}"
    except Exception as e:
        return None, f"Error: {e}"


# ─────────────────────────────────────────────
#  Tactile 3D handlers
# ─────────────────────────────────────────────

def generate_tactile_3d(texture_pil, hm_method, scale_factor, mesh_type, uv_method, sigma, progress=gr.Progress()):
    if texture_pil is None:
        return None, None, "Upload a texture image."
    try:
        progress(0.1, desc="Generating heightmap...")

        if hm_method == "diffusion (SD+LoRA)":
            pipeline = get_diffusion()
            hm = pipeline.generate(texture_pil, output_size=512)
        else:
            from data.heightmap_generator import generate_heightmap
            arr = np.array(texture_pil.convert("RGB"))
            hm_arr = generate_heightmap(arr, sigma=float(sigma))
            hm = Image.fromarray(hm_arr, mode="L")

        progress(0.5, desc="Building 3D mesh...")
        from texture3d.mesh_processor import MeshProcessor
        from texture3d.texture_mapper import HeightmapTextureMapper

        if mesh_type == "sphere":
            mp = MeshProcessor.sphere(subdivisions=4)
        elif mesh_type == "box":
            mp = MeshProcessor.box()
        else:
            mp = MeshProcessor.plane()

        mp.generate_uv(method=uv_method)

        progress(0.8, desc="Applying heightmap displacement...")
        mapper = HeightmapTextureMapper(scale_factor=float(scale_factor))
        displaced = mapper.apply(mp, hm)

        progress(0.95, desc="Exporting GLB...")
        with tempfile.NamedTemporaryFile(suffix=".glb", delete=False) as f:
            out_path = f.name
        displaced.export(out_path)

        return hm.convert("RGB"), out_path, "Success! Download your textured 3D model."
    except Exception as e:
        return None, None, f"Error: {e}"


# ─────────────────────────────────────────────
#  Gradio UI
# ─────────────────────────────────────────────

def build_ui():
    with gr.Blocks(title="Texture Model Stylization", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Texture Model Stylization
            **Neural style transfer + tactile heightmap generation + 3D surface application.**
            Upload your images and explore visual & tactile texture synthesis.
            """
        )

        with gr.Tabs():
            # ── Tab 1: Style Transfer ──
            with gr.Tab("Style Transfer"):
                gr.Markdown("### Transfer a visual style from one image onto another.")
                with gr.Row():
                    content_in = gr.Image(label="Content Image", type="pil")
                    style_in   = gr.Image(label="Style Image", type="pil")
                    style_out  = gr.Image(label="Result", type="pil")

                method = gr.Radio(["AdaIN (fast)", "Gatys (quality)"], value="AdaIN (fast)", label="Method")
                with gr.Row():
                    alpha_sl    = gr.Slider(0.0, 1.0, 1.0, step=0.05, label="AdaIN Alpha (style strength)")
                    steps_sl    = gr.Slider(100, 500, 300, step=50,  label="Gatys Steps")
                    cw_sl       = gr.Slider(0.1, 10.0, 1.0,          label="Content Weight")
                    sw_sl       = gr.Number(value=1e6,                label="Style Weight")
                    size_sl     = gr.Slider(128, 512, 256, step=64,   label="Image Size")

                status_st = gr.Textbox(label="Status", interactive=False)
                interp_out = gr.Image(label="Interpolation Preview", type="pil")
                alpha_interp = gr.Slider(0.0, 1.0, 0.5, label="Interpolation Alpha")
                interp_mode  = gr.Radio(["pixel", "adain"], value="adain", label="Interpolation Mode")

                def run_transfer(c, s, meth, alpha, steps, cw, sw, sz):
                    if meth == "AdaIN (fast)":
                        return run_adain_transfer(c, s, alpha, sz)
                    else:
                        return run_gatys_transfer(c, s, steps, cw, sw, sz)

                gr.Button("Transfer Style").click(
                    run_transfer,
                    inputs=[content_in, style_in, method, alpha_sl, steps_sl, cw_sl, sw_sl, size_sl],
                    outputs=[style_out, status_st],
                )
                gr.Button("Interpolate").click(
                    run_interpolation,
                    inputs=[content_in, style_in, alpha_interp, interp_mode, size_sl],
                    outputs=[interp_out, status_st],
                )

            # ── Tab 2: Tactile 3D ──
            with gr.Tab("Tactile 3D"):
                gr.Markdown(
                    "### Generate a tactile heightmap from a texture and apply it to a 3D mesh.\n"
                    "Download the `.glb` file and open it in [gltf.report](https://gltf.report) or Blender."
                )
                with gr.Row():
                    tex_in  = gr.Image(label="Input Texture", type="pil")
                    hm_out  = gr.Image(label="Generated Heightmap", type="pil")
                    glb_out = gr.File(label="Download Textured GLB")

                with gr.Row():
                    hm_method   = gr.Radio(["synthetic (Sobel)", "diffusion (SD+LoRA)"],
                                           value="synthetic (Sobel)", label="Heightmap Method")
                    mesh_type   = gr.Radio(["sphere", "box", "plane"], value="sphere", label="3D Primitive")
                    uv_method   = gr.Radio(["auto", "xatlas", "sphere"], value="auto", label="UV Method")

                with gr.Row():
                    scale_sl = gr.Slider(0.01, 0.3, 0.07, step=0.01, label="Displacement Scale")
                    sigma_sl = gr.Slider(0.5, 8.0, 2.0, step=0.5,    label="Gaussian Sigma (synthetic only)")

                status_3d = gr.Textbox(label="Status", interactive=False)

                gr.Button("Generate Tactile 3D Model", variant="primary").click(
                    generate_tactile_3d,
                    inputs=[tex_in, hm_method, scale_sl, mesh_type, uv_method, sigma_sl],
                    outputs=[hm_out, glb_out, status_3d],
                )

            # ── Tab 3: About ──
            with gr.Tab("About"):
                gr.Markdown(
                    """
                    ## About This Project
                    This system combines:
                    - **Texture Classification** — ResNet-50 fine-tuned on DTD (47 categories)
                    - **Neural Style Transfer** — VGG-19 Gram matrix (Gatys et al.) and AdaIN
                    - **Tactile Heightmap Generation** — Stable Diffusion 1.5 + LoRA fine-tuning
                    - **3D Surface Application** — UV displacement via trimesh
                    - **Texture Interpolation** — Pixel, AdaIN latent, and CLIP embedding blending

                    **Dataset**: DTD — Describable Textures Dataset (5,640 images, 47 classes)

                    **Key references**:
                    - TactStyle: Generating Tactile Textures with Generative AI (CHI 2025)
                    - Gatys et al., "A Neural Algorithm of Artistic Style" (2016)
                    - Huang & Belongie, "Arbitrary Style Transfer in Real-time with AdaIN" (2017)
                    - 3DShape2VecSet (SIGGRAPH 2023)
                    """
                )

    return demo


def main():
    global LORA_PATH
    parser = argparse.ArgumentParser(description="Launch Gradio demo")
    parser.add_argument("--share", action="store_true", help="Create public Gradio link")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--lora", default=None, help="Path to LoRA weights")
    args = parser.parse_args()

    LORA_PATH = args.lora
    demo = build_ui()
    demo.queue()
    demo.launch(share=args.share, server_port=args.port)


if __name__ == "__main__":
    main()
