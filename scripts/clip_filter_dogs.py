"""
Filter dog images from a WebDataset using CLIP similarity scoring.

Usage:
    python scripts/clip_filter_dogs.py \
        --input_dir /data3/rey/t2i-public-sft-120k-flat \
        --output_dir /data3/rey/t2i-public-sft-120k-dogs-clip \
        --threshold 0.27 \
        --batch_size 128

Writes accepted samples as a new WebDataset (one shard, or multiple if --samples_per_shard set).
"""
import argparse
import io
import tarfile
import time
from pathlib import Path

import torch
import webdataset as wds
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

QUERY = "a photo of a dog"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="/data3/rey/t2i-public-sft-120k-flat")
    p.add_argument("--output_dir", default="/data3/rey/t2i-public-sft-120k-dogs-clip")
    p.add_argument("--threshold", type=float, default=0.27,
                   help="CLIP cosine similarity threshold (0.25–0.30 typical)")
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--samples_per_shard", type=int, default=500)
    p.add_argument("--model_name", default="openai/clip-vit-large-patch14")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dry_run", action="store_true",
                   help="Print stats only, do not write output")
    return p.parse_args()


def load_model(model_name, device):
    print(f"Loading {model_name} on {device} ...")
    model = CLIPModel.from_pretrained(model_name).to(device).eval()
    processor = CLIPProcessor.from_pretrained(model_name)
    return model, processor


@torch.no_grad()
def score_batch(model, processor, images, text_features, device):
    inputs = processor(images=images, return_tensors="pt", padding=True)
    pixel_values = inputs["pixel_values"].to(device)
    image_features = model.get_image_features(pixel_values=pixel_values)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    scores = (image_features @ text_features.T).squeeze(-1)
    return scores.cpu().tolist()


def main():
    args = parse_args()

    shards = sorted(str(p) for p in Path(args.input_dir).glob("*.tar"))
    if not shards:
        raise FileNotFoundError(f"No .tar shards in {args.input_dir}")
    print(f"Input shards: {len(shards)}")

    model, processor = load_model(args.model_name, args.device)

    # Precompute text features once
    with torch.no_grad():
        text_inputs = processor(text=[QUERY], return_tensors="pt", padding=True).to(args.device)
        text_features = model.get_text_features(**text_inputs)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    output_dir = Path(args.output_dir)
    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    total = accepted = shard_idx = samples_in_shard = 0
    current_tar = current_sink = None
    t0 = time.time()

    def open_new_shard():
        nonlocal current_tar, current_sink, shard_idx, samples_in_shard
        if current_sink:
            current_sink.close()
        shard_path = output_dir / f"dogs-clip-{shard_idx:05d}.tar"
        current_sink = tarfile.open(shard_path, "w")
        shard_idx += 1
        samples_in_shard = 0
        print(f"  → writing shard {shard_idx - 1}: {shard_path.name}")

    def write_sample(key, jpg_bytes, txt_bytes):
        nonlocal samples_in_shard
        if samples_in_shard >= args.samples_per_shard:
            open_new_shard()
        for name, data in [(f"{key}.jpg", jpg_bytes), (f"{key}.txt", txt_bytes)]:
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            current_sink.addfile(info, io.BytesIO(data))
        samples_in_shard += 1

    if not args.dry_run:
        open_new_shard()

    # Buffer batches
    batch_keys, batch_imgs, batch_txts, batch_jpgs = [], [], [], []

    def flush_batch():
        nonlocal accepted
        if not batch_imgs:
            return
        scores = score_batch(model, processor, batch_imgs, text_features, args.device)
        for key, score, jpg, txt in zip(batch_keys, scores, batch_jpgs, batch_txts):
            if score >= args.threshold:
                accepted += 1
                if not args.dry_run:
                    write_sample(key, jpg, txt)
        batch_keys.clear(); batch_imgs.clear()
        batch_txts.clear(); batch_jpgs.clear()

    dataset = wds.WebDataset(shards, shardshuffle=False)
    for sample in dataset:
        total += 1
        try:
            img = Image.open(io.BytesIO(sample["jpg"])).convert("RGB")
        except Exception:
            continue
        batch_keys.append(sample["__key__"])
        batch_imgs.append(img)
        batch_jpgs.append(sample["jpg"])
        batch_txts.append(sample["txt"])

        if len(batch_imgs) >= args.batch_size:
            flush_batch()

        if total % 5000 == 0:
            elapsed = time.time() - t0
            print(f"  {total:,} processed | {accepted:,} accepted | "
                  f"{accepted/total*100:.1f}% | {total/elapsed:.0f} img/s")

    flush_batch()

    if not args.dry_run and current_sink:
        current_sink.close()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"Total:    {total:,}")
    print(f"Accepted: {accepted:,} ({accepted/total*100:.1f}%)")
    print(f"Threshold: {args.threshold}")
    if not args.dry_run:
        print(f"Output:   {output_dir}  ({shard_idx} shards)")


if __name__ == "__main__":
    main()
