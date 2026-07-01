"""Pack dog breed images + captions into WebDataset tar shards.

Usage:
    uv run python scripts/pack_dogs_wds.py \
        --data-dir /data3/rey/dogs \
        --output-dir /data3/rey/dogs_wds \
        --samples-per-shard 500
"""

import argparse
import io
from pathlib import Path

import webdataset as wds
from PIL import Image

BREEDS = sorted([
    "afghan_hound", "beagle", "boxer", "chihuahua", "chow",
    "corgi", "doberman", "french_bulldog", "german_shepherd",
    "golden_retriever", "great_dane", "labrador_retriever",
    "pomeranian", "pug", "rottweiler", "samoyed", "shih_tzu",
    "siberian_husky", "standard_poodle", "yorkshire_terrier",
])


def pack(data_dir: Path, output_dir: Path, samples_per_shard: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(output_dir / "shard-%05d.tar")

    total = skipped = 0
    with wds.ShardWriter(pattern, maxcount=samples_per_shard) as sink:
        for breed in BREEDS:
            images_dir = data_dir / breed / "images"
            captions_dir = data_dir / breed / "captions"

            for img_path in sorted(images_dir.glob("*.jpg")):
                cap_path = captions_dir / f"{img_path.stem}.txt"
                if not cap_path.exists():
                    skipped += 1
                    continue

                img_bytes = img_path.read_bytes()
                try:
                    Image.open(io.BytesIO(img_bytes)).verify()
                except Exception:
                    skipped += 1
                    continue

                sink.write({
                    "__key__": f"{breed}/{img_path.stem}",
                    "jpg": img_bytes,
                    "txt": cap_path.read_text(encoding="utf-8").strip().encode("utf-8"),
                })
                total += 1

    print(f"Done: {total} samples written, {skipped} skipped → {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data3/rey/dogs")
    parser.add_argument("--output-dir", default="/data3/rey/dogs_wds")
    parser.add_argument("--samples-per-shard", type=int, default=500)
    args = parser.parse_args()
    pack(Path(args.data_dir), Path(args.output_dir), args.samples_per_shard)
