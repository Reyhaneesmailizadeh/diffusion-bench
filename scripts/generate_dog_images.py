"""Render FLUX.1-schnell images for captions produced by scripts/generate_dog_captions.py.

Reads {captions_dir}/{breed}/captions/*.txt and writes one 1024x1024 JPEG per caption to
{output_dir}/{breed}/images/{stem}.jpg — the same breed/images + breed/captions layout
scripts/pack_dogs_wds.py already expects, so the result packs unmodified:

    uv run python scripts/pack_dogs_wds.py --data-dir <output-dir> --output-dir <wds-dir>

Images are generated at Flux's native 1024x1024 deliberately, not at the training image_size
(e.g. 256) — the existing training/eval pipeline (DogsWebDataset / precompute_latents.py)
already resizes+center-crops arbitrary source resolutions down to whatever image_size the
config needs, exactly as it does today for the real photos' arbitrary native resolutions.
Forcing Flux to output tiny images directly would degrade quality since it's trained for ~1024.

Single-GPU usage:
    uv run python scripts/generate_dog_images.py \
        --output-dir /data3/rey/dogs_synthetic \
        --limit 8   # smoke test before committing to the full run

Multi-GPU usage (recommended, splits captions across GPUs — each rank loads its own
full pipeline copy and processes a disjoint slice, no inter-rank communication needed):
    uv run torchrun --nproc_per_node=8 scripts/generate_dog_images.py \
        --output-dir /data3/rey/dogs_synthetic

If you hit CUDA OOM on a 24GB GPU, add --sequential-offload (slower, lower peak VRAM).
"""

import argparse
import os
import zlib
from pathlib import Path

import torch
from diffusers import FluxPipeline


def stable_seed(key: str, base: int) -> int:
    return base + zlib.crc32(key.encode("utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/data3/rey/dogs_synthetic")
    parser.add_argument("--captions-dir", default=None,
                         help="Defaults to --output-dir (where generate_dog_captions.py wrote breed/captions/)")
    parser.add_argument("--model-id", default="black-forest-labs/FLUX.1-schnell")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--num-inference-steps", type=int, default=4)
    parser.add_argument("--guidance-scale", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed-base", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true",
                         help="Overwrite existing images (default: skip already-generated)")
    parser.add_argument("--limit", type=int, default=0,
                         help="Cap the number of images this rank generates (smoke testing)")
    parser.add_argument("--sequential-offload", action="store_true",
                         help="Use enable_sequential_cpu_offload instead of enable_model_cpu_offload "
                              "(slower, lower peak VRAM — use if you hit OOM)")
    args = parser.parse_args()

    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)

    output_dir = Path(args.output_dir)
    captions_dir = Path(args.captions_dir) if args.captions_dir else output_dir

    caption_files = sorted(captions_dir.glob("*/captions/*.txt"))
    if not caption_files:
        raise ValueError(f"No caption .txt files found under {captions_dir}/*/captions/. "
                          "Run scripts/generate_dog_captions.py first.")
    if rank == 0:
        print(f"Total captions: {len(caption_files)} across {world_size} rank(s)")

    my_files = caption_files[rank::world_size]
    if args.limit:
        my_files = my_files[:args.limit]

    if rank == 0:
        print(f"[Rank {rank}] Loading {args.model_id}...")
    pipe = FluxPipeline.from_pretrained(args.model_id, torch_dtype=torch.bfloat16)
    if args.sequential_offload:
        pipe.enable_sequential_cpu_offload(gpu_id=local_rank)
    else:
        pipe.enable_model_cpu_offload(gpu_id=local_rank)

    done = skipped = 0

    def flush(batch):
        nonlocal done
        if not batch:
            return
        prompts = [b[2] for b in batch]
        generators = [torch.Generator(device="cpu").manual_seed(stable_seed(b[0], args.seed_base)) for b in batch]
        images = pipe(
            prompt=prompts,
            width=args.width,
            height=args.height,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=generators,
        ).images
        for (_key, img_path, _caption), image in zip(batch, images):
            img_path.parent.mkdir(parents=True, exist_ok=True)
            image.save(img_path, quality=95)
            done += 1

    batch = []
    for cap_path in my_files:
        breed = cap_path.parent.parent.name
        stem = cap_path.stem
        img_path = output_dir / breed / "images" / f"{stem}.jpg"
        if img_path.exists() and not args.overwrite:
            skipped += 1
            continue
        caption = cap_path.read_text(encoding="utf-8").strip()
        batch.append((f"{breed}/{stem}", img_path, caption))
        if len(batch) >= args.batch_size:
            flush(batch)
            batch = []
            if rank == 0:
                print(f"[Rank {rank}] {done} done, {skipped} skipped / {len(my_files)} assigned")
    flush(batch)

    print(f"[Rank {rank}] Done: {done} generated, {skipped} skipped / {len(my_files)} assigned")


if __name__ == "__main__":
    main()
