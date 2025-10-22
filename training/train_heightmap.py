"""
Fine-tune Stable Diffusion 1.5 img2img with LoRA for heightmap generation.

Trains on synthetic (texture → heightmap) pairs from data/heightmap_generator.py.
Saves LoRA adapter weights to checkpoints/heightmap_lora/.

Requirements: ~6 GB VRAM (A100 / 4090). For Colab: use T4 with batch_size=1.

Usage:
  python training/train_heightmap.py --config configs/heightmap.yaml
  python training/train_heightmap.py --data-root data/paired --steps 3000
"""
import argparse
import os
from pathlib import Path
from typing import Optional

import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader

from data.texture_dataset import HeightmapDataset
from models.heightmap_diffusion import build_lora_unet


def train(
    data_root: str,
    output_dir: str = "checkpoints/heightmap_lora",
    base_model: str = "runwayml/stable-diffusion-v1-5",
    lora_rank: int = 4,
    lora_alpha: int = 16,
    train_steps: int = 3000,
    batch_size: int = 1,
    lr: float = 1e-4,
    resolution: int = 512,
    gradient_accumulation_steps: int = 4,
    mixed_precision: str = "fp16",
    save_every: int = 500,
    seed: int = 42,
    device: Optional[str] = None,
):
    try:
        import accelerate
        from accelerate import Accelerator
        from diffusers import (
            AutoencoderKL,
            DDPMScheduler,
            StableDiffusionImg2ImgPipeline,
            UNet2DConditionModel,
        )
        from transformers import CLIPTextModel, CLIPTokenizer
    except ImportError as e:
        raise ImportError(
            "Install required packages: pip install diffusers transformers accelerate peft"
        ) from e

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)

    accelerator = Accelerator(
        mixed_precision=mixed_precision,
        gradient_accumulation_steps=gradient_accumulation_steps,
    )
    device = accelerator.device

    print(f"Loading base model: {base_model}")
    tokenizer = CLIPTokenizer.from_pretrained(base_model, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(base_model, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(base_model, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(base_model, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(base_model, subfolder="scheduler")

    # Freeze everything except LoRA adapters
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    unet = build_lora_unet(unet, rank=lora_rank, alpha=lora_alpha)

    lora_params = [p for p in unet.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=lr)

    dataset = HeightmapDataset(data_root, image_size=resolution, augment=True)
    if len(dataset) == 0:
        raise ValueError(
            f"No paired (texture, heightmap) data found at {data_root}. "
            "Run: python data/heightmap_generator.py --dtd-root data/dtd --output data/dtd_heightmaps"
        )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True)

    prompt = (
        "grayscale surface heightmap, smooth displacement field, "
        "tactile texture relief map, high detail"
    )
    text_inputs = tokenizer(prompt, return_tensors="pt", padding="max_length",
                            max_length=tokenizer.model_max_length, truncation=True)

    unet, optimizer, loader = accelerator.prepare(unet, optimizer, loader)
    vae = vae.to(device)
    text_encoder = text_encoder.to(device)

    with torch.no_grad():
        text_emb = text_encoder(text_inputs.input_ids.to(device))[0]

    global_step = 0
    data_iter = iter(loader)

    print(f"Training for {train_steps} steps on {len(dataset)} pairs ...")
    while global_step < train_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        textures = batch["texture"].to(device)
        heightmaps = batch["heightmap"].to(device)

        # Encode target heightmap to latent space
        hm_rgb = heightmaps.repeat(1, 3, 1, 1)  # L → RGB
        with torch.no_grad():
            latents = vae.encode(hm_rgb * 2 - 1).latent_dist.sample() * 0.18215

        # Add noise
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps,
                                  (latents.shape[0],), device=device).long()
        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

        # Encode texture as conditioning (concatenate along channel dim)
        with torch.no_grad():
            cond_latents = vae.encode(textures).latent_dist.sample() * 0.18215

        # Predict noise
        noise_pred = unet(
            torch.cat([noisy_latents, cond_latents], dim=1) if unet.config.in_channels == 8
            else noisy_latents,
            timesteps,
            encoder_hidden_states=text_emb.expand(latents.shape[0], -1, -1),
        ).sample

        loss = torch.nn.functional.mse_loss(noise_pred, noise)

        accelerator.backward(loss)
        if accelerator.sync_gradients:
            accelerator.clip_grad_norm_(lora_params, 1.0)
        optimizer.step()
        optimizer.zero_grad()

        global_step += 1
        if global_step % 50 == 0:
            print(f"Step {global_step}/{train_steps} | loss={loss.item():.4f}")

        if global_step % save_every == 0:
            ckpt_dir = Path(output_dir) / f"step_{global_step}"
            accelerator.unwrap_model(unet).save_pretrained(ckpt_dir)
            print(f"Saved checkpoint: {ckpt_dir}")

    # Save final LoRA weights
    final_dir = Path(output_dir) / "final"
    accelerator.unwrap_model(unet).save_pretrained(final_dir)
    print(f"Training complete. LoRA weights saved to {final_dir}")


def main():
    parser = argparse.ArgumentParser(description="Fine-tune SD 1.5 with LoRA for heightmap generation")
    parser.add_argument("--config", default=None)
    parser.add_argument("--data-root", default="data/paired",
                        help="Root with textures/ and heightmaps/ subdirs")
    parser.add_argument("--output-dir", default="checkpoints/heightmap_lora")
    parser.add_argument("--base-model", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--lora-rank", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--resolution", type=int, default=512)
    parser.add_argument("--mixed-precision", default="fp16", choices=["no", "fp16", "bf16"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = {}
    if args.config and os.path.exists(args.config):
        with open(args.config) as f:
            cfg = yaml.safe_load(f)

    train(
        data_root=cfg.get("data_root", args.data_root),
        output_dir=cfg.get("output_dir", args.output_dir),
        base_model=cfg.get("base_model", args.base_model),
        lora_rank=cfg.get("lora_rank", args.lora_rank),
        train_steps=cfg.get("train_steps", args.steps),
        batch_size=cfg.get("batch_size", args.batch_size),
        lr=cfg.get("lr", args.lr),
        resolution=cfg.get("resolution", args.resolution),
        mixed_precision=cfg.get("mixed_precision", args.mixed_precision),
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
