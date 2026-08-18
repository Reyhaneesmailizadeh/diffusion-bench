#!/usr/bin/env python
"""Generate images for a fixed list of (breed, caption) pairs from a stage-2 checkpoint.

Single-GPU, non-distributed. Intended for visually comparing a pretrained vs.
fine-tuned checkpoint on the same captions (e.g. pretrain vs. dogs-SFT).

Usage:
    uv run python scripts/generate_dog_comparison.py \
        --config configs/stage2/training/t2i/t2i-ddt-en28d1152hd72-dn2d2048hd128-e2e-invae-vpred-t4-RePA-latents-EMA0.9995.yaml \
        --checkpoint ckpts/vavae-repa-recaptioned-latents-invae-EMA0.9995-200Epochs/checkpoints/ep-0000200.pt \
        --captions-json /path/to/captions.json \
        --output-dir results/dog_comparison/pretrained \
        --seed 0
"""

import argparse
import dataclasses
import json
import math
import os
import sys
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torchvision.utils import save_image

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from configs.stage2 import Stage2Config
from encoders.vision_encoder import load_encoders
from stage2.models import Stage2ModelProtocol
from stage2.transport import create_sampler, create_transport
from stage2.utils import encode_text, setup_text_encoder, validate_stage2_config
from utils.guidance_utils import get_model_forward_fn
from utils.model_utils import instantiate_from_config
from utils.train_utils import get_autocast_kwargs


def main(args):
    if not torch.cuda.is_available():
        raise RuntimeError("Generation requires a GPU.")

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)

    device = torch.device("cuda", 0)
    rank = 0  # single process; main_process_first() calls in setup are no-ops

    autocast_kwargs = get_autocast_kwargs(args)

    config: Stage2Config = OmegaConf.to_object(
        OmegaConf.merge(OmegaConf.structured(Stage2Config), OmegaConf.load(args.config))
    )
    config.post_process()
    validate_stage2_config(config)

    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    latent_size = tuple(config.misc.latent_size)

    rae = instantiate_from_config(config.stage_1).to(device)
    rae.eval()

    if config.repa.use_repa or config.repa.use_reg:
        repa_target_encoder = load_encoders(
            config.repa.target_encoder, device, config.repa.target_encoder_resolution
        )[0]
        repa_target_encoder.eval()
        repa_target_encoder.model.requires_grad_(False)
        config.repa.z_dim = repa_target_encoder.embed_dim

    text_encoder = setup_text_encoder(config, rank, device)
    config.prepare_model_params()

    model: Stage2ModelProtocol = instantiate_from_config(config.stage_2).to(device)
    model.eval()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    ema_weights = ckpt.get("ema", ckpt.get("model"))
    model.load_state_dict(ema_weights)
    print(f"Loaded EMA weights from {args.checkpoint} "
          f"(epoch {ckpt.get('epoch', '?')}, step {ckpt.get('step', '?')})")

    model_fn, sample_model_kwargs = get_model_forward_fn(model, config.guidance)
    use_guidance = config.guidance.any_guidance_active
    print(f"Guidance mode: {config.guidance.get_mode_string()} "
          f"(cfg_scale={config.guidance.cfg.scale if config.guidance.cfg else None})")

    time_dist_shift = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size)) / config.misc.time_dist_shift_base
    )
    time_dist_shift_base_eval = (
        config.misc.time_dist_shift_base
        if config.misc.time_dist_shift_base_eval is None
        else config.misc.time_dist_shift_base_eval
    )
    time_dist_shift_eval = math.sqrt(
        (config.misc.time_dist_shift_dim or math.prod(latent_size)) / time_dist_shift_base_eval
    )
    transport = create_transport(
        config=config.transport,
        time_dist_shift=time_dist_shift,
        time_dist_shift_eval=time_dist_shift_eval,
    )
    transport_sampler = create_sampler(transport, guidance_config=config.guidance)
    sampler_cfg = dataclasses.asdict(config.sampler)
    if args.num_steps is not None:
        sampler_cfg["num_steps"] = args.num_steps
    eval_sampler = transport_sampler.sample_ode(**sampler_cfg)

    with open(args.captions_json) as f:
        items = json.load(f)  # list of {"breed": ..., "stem": ..., "text": ...}

    os.makedirs(args.output_dir, exist_ok=True)

    batch_size = args.batch_size
    for start in range(0, len(items), batch_size):
        batch = items[start:start + batch_size]
        prompts = [it["text"] for it in batch]
        n = len(prompts)

        context, attn_mask = encode_text(text_encoder, prompts)
        zs = torch.randn(n, *latent_size, device=device, dtype=torch.float32)

        if use_guidance:
            zs_in = torch.cat([zs, zs], dim=0)
            context_null, attn_mask_null = encode_text(text_encoder, [""] * n)
            context_in = torch.cat([context, context_null], dim=0)
            attn_mask_in = torch.cat([attn_mask, attn_mask_null], dim=0) if attn_mask is not None else None
        else:
            zs_in, context_in, attn_mask_in = zs, context, attn_mask

        kwargs = dict(sample_model_kwargs)
        kwargs.update(context=context_in, attn_mask=attn_mask_in)

        with torch.autocast(device_type="cuda", **autocast_kwargs):
            samples = eval_sampler(zs_in, model_fn, **kwargs)[-1]
            if use_guidance:
                samples = samples.chunk(2, dim=0)[0]
            images = rae.decode(samples).cpu().float()

        for it, img in zip(batch, images):
            name = f"{it['breed']}_{it['stem']}"
            sample_dir = args.output_dir
            if args.group_by_breed:
                sample_dir = os.path.join(args.output_dir, it["breed"])
                os.makedirs(sample_dir, exist_ok=True)
            save_image(img, os.path.join(sample_dir, f"{name}.png"))
            with open(os.path.join(sample_dir, f"{name}.txt"), "w") as f:
                f.write(it["text"])
            print(f"  saved {name}")

    print(f"Done. {len(items)} images written to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dog images from fixed captions for visual comparison.")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--captions-json", type=str, required=True,
                         help="JSON list of {breed, stem, text} objects.")
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--precision", type=str, choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--num-steps", type=int, default=None, help="Override sampler.num_steps from config.")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--group-by-breed", action="store_true",
                         help="Write outputs to output_dir/{breed}/{breed}_{stem}.png instead of "
                              "flat output_dir/{breed}_{stem}.png -- matches a per-breed caption "
                              "folder layout (e.g. reward_model/breeds/{breed}/...). Off by default "
                              "for backward compatibility with existing flat-output callers.")
    args = parser.parse_args()
    main(args)
