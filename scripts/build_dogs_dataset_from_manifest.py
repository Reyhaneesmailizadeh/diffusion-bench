"""Reconstruct the 20-breed dog dataset from a captions-only manifest, without redistributing
any image bytes ourselves.

Why this exists: the source images come from arijitghosh/T2I-ImageNet-Normal on the HF Hub,
which is an undocumented, unlicensed third-party repackaging with no dataset card -- its own
rights back to the original images are unclear, so we do not re-host a copy of them. What we
do own is our own captions (generated via a paid vision API, see scripts/recaption_dogs.py)
and the knowledge of which specific rows (by __key__) make up the dog subset. This script
takes only that -- data/dog_captions_manifest.csv, ~12MB, just (key, breed, caption) rows --
and streams the *public* source dataset to pull back the matching images, pairing each with
our caption instead of the source dataset's own generic one.

Usage:
    uv run python scripts/build_dogs_dataset_from_manifest.py \
        --manifest data/dog_captions_manifest.csv \
        --output-dir /path/to/your/data/dogs/dogs_raw

Note: the source dataset is packed as WebDataset shards (images bundled with large embedding
arrays we don't need), so this streams through the full ~1.28M-row dataset once, filtering
down to our ~26k target keys -- expect this to take a while and use meaningful bandwidth on
first run. It's a one-time cost; the result is cached to --output-dir same as if you'd sourced
and captioned the images yourself (Section 8's "from scratch" path).
"""
import argparse
import csv
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--source-dataset", default="arijitghosh/T2I-ImageNet-Normal")
    args = parser.parse_args()

    manifest = {}
    with open(args.manifest, newline="") as f:
        for row in csv.DictReader(f):
            manifest[row["key"]] = (row["breed"], row["caption"])
    print(f"Loaded {len(manifest)} target (key, breed, caption) rows from {args.manifest}")

    out_dir = Path(args.output_dir)
    for breed in {b for b, _ in manifest.values()}:
        (out_dir / breed).mkdir(parents=True, exist_ok=True)

    remaining = set(manifest.keys())
    ds = load_dataset(args.source_dataset, split="train", streaming=True)

    written = 0
    pbar = tqdm(total=len(manifest), desc="Reconstructing dog subset")
    for row in ds:
        if not remaining:
            break
        key = row["__key__"]
        if key not in manifest:
            continue
        breed, caption = manifest[key]
        img_path = out_dir / breed / f"{key}.jpg"
        txt_path = out_dir / breed / f"{key}.txt"
        row["jpg"].save(img_path)
        txt_path.write_text(caption, encoding="utf-8")
        remaining.discard(key)
        written += 1
        pbar.update(1)
    pbar.close()

    print(f"Wrote {written}/{len(manifest)} images to {out_dir}")
    if remaining:
        print(f"WARNING: {len(remaining)} keys from the manifest were not found in "
              f"{args.source_dataset} (source dataset may have changed) -- e.g. {list(remaining)[:5]}")


if __name__ == "__main__":
    main()
