"""Compute InceptionV3 FID reference statistics for the dogs val split.

Usage:
    python scripts/compute_dogs_val_stats.py \
        --val-dir /data3/rey/dogs_wds/val \
        --output /data3/rey/dogs_wds/dogs_val_stats.npz \
        --batch-size 64 \
        --device cuda
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import webdataset as wds
from torchvision import transforms
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from torch_fidelity.feature_extractor_inceptionv3 import FeatureExtractorInceptionV3


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-dir", default="/data3/rey/dogs_wds/val")
    parser.add_argument("--output", default="/data3/rey/dogs_wds/dogs_val_stats.npz")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=256)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    val_dir = Path(args.val_dir)
    shard_urls = sorted(str(p) for p in val_dir.glob("*.tar"))
    if not shard_urls:
        raise ValueError(f"No tar shards found in {val_dir}")
    print(f"Found {len(shard_urls)} val shards in {val_dir}")

    transform = transforms.Compose([
        transforms.Resize(args.image_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(args.image_size),
        transforms.ToTensor(),
    ])

    def decode_sample(sample):
        image = sample.get("jpg") or sample.get("png") or sample.get("jpeg") or sample.get("webp")
        if image is None:
            return None
        return transform(image)

    dataset = (
        wds.WebDataset(shard_urls)
        .decode("pil", handler=wds.ignore_and_continue)
        .map(decode_sample, handler=wds.ignore_and_continue)
        .select(lambda x: x is not None)
    )
    loader = wds.WebLoader(dataset, batch_size=args.batch_size, num_workers=4)

    # Collect all images as uint8 NCHW
    print("Loading val images...")
    all_images = []
    for batch in tqdm(loader):
        # batch is a list of tensors (float [0,1], NCHW) — collated by WebLoader
        imgs = batch  # (B, C, H, W) float32 in [0, 1]
        imgs_uint8 = (imgs * 255).clamp(0, 255).to(torch.uint8)
        all_images.append(imgs_uint8)

    all_images = torch.cat(all_images, dim=0)  # (N, C, H, W) uint8
    print(f"Loaded {len(all_images)} images, shape {all_images.shape}")

    # Compute InceptionV3 features
    print(f"Computing InceptionV3 features on {args.device}...")
    fe = FeatureExtractorInceptionV3(
        name="inception-v3-compat",
        features_list=["2048"],
    ).to(args.device).eval()

    feats = []
    with torch.no_grad():
        for i in tqdm(range(0, len(all_images), args.batch_size)):
            batch = all_images[i : i + args.batch_size].to(args.device)
            (f,) = fe(batch)
            feats.append(f.cpu())

    feats = torch.cat(feats, dim=0).double().numpy()  # (N, 2048)
    print(f"Features shape: {feats.shape}")

    mu = feats.mean(axis=0)
    sigma = np.cov(feats, rowvar=False)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output, mu=mu, sigma=sigma)
    print(f"Saved reference stats to {output}  (mu={mu.shape}, sigma={sigma.shape})")


if __name__ == "__main__":
    main()
