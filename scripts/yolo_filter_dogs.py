"""
Detect dog images from a WebDataset using YOLOv8 (COCO class 16 = dog).

Dry run (count only, no files written):
    python scripts/yolo_filter_dogs.py --dry_run

Full extraction:
    python scripts/yolo_filter_dogs.py \
        --output_dir /data3/rey/t2i-public-sft-120k-dogs-yolo
"""
import argparse
import io
import tarfile
import time
from pathlib import Path

import torch
import webdataset as wds
from PIL import Image
from ultralytics import YOLO

DOG_CLASS = 16  # COCO class index for "dog"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", default="/data3/rey/t2i-public-sft-120k-flat")
    p.add_argument("--output_dir", default="/data3/rey/t2i-public-sft-120k-dogs-yolo")
    p.add_argument("--model", default="yolov8n.pt",
                   help="YOLO model: yolov8n.pt (fast) / yolov8s.pt / yolov8m.pt (accurate)")
    p.add_argument("--conf", type=float, default=0.25,
                   help="Detection confidence threshold (0.25 = YOLO default)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--samples_per_shard", type=int, default=500)
    p.add_argument("--dry_run", action="store_true",
                   help="Count detections only — do not write any files")
    return p.parse_args()


def main():
    args = parse_args()

    shards = sorted(str(p) for p in Path(args.input_dir).glob("*.tar"))
    if not shards:
        raise FileNotFoundError(f"No .tar shards found in {args.input_dir}")
    print(f"Input shards : {len(shards)}")
    print(f"YOLO model   : {args.model}  conf={args.conf}")
    print(f"Mode         : {'DRY RUN (no output written)' if args.dry_run else args.output_dir}")
    print()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = YOLO(args.model)
    model.to(device)

    if not args.dry_run:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)

    total = accepted = shard_idx = samples_in_shard = 0
    current_sink = None
    t0 = time.time()

    def open_new_shard():
        nonlocal current_sink, shard_idx, samples_in_shard
        if current_sink:
            current_sink.close()
        path = Path(args.output_dir) / f"dogs-yolo-{shard_idx:05d}.tar"
        current_sink = tarfile.open(path, "w")
        shard_idx += 1
        samples_in_shard = 0

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

    # Accumulate batches
    batch_keys, batch_imgs, batch_jpgs, batch_txts = [], [], [], []

    def flush():
        nonlocal accepted
        if not batch_imgs:
            return
        results = model(batch_imgs, verbose=False, conf=args.conf)
        for key, res, jpg, txt in zip(batch_keys, results, batch_jpgs, batch_txts):
            classes = [int(b.cls) for b in res.boxes] if res.boxes is not None else []
            if DOG_CLASS in classes:
                accepted += 1
                if not args.dry_run:
                    write_sample(key, jpg, txt)
        batch_keys.clear(); batch_imgs.clear()
        batch_jpgs.clear(); batch_txts.clear()

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
            flush()

        if total % 2000 == 0:
            elapsed = time.time() - t0
            rate = total / elapsed
            eta = (len(shards) * 500 - total) / rate if rate > 0 else 0
            print(f"  {total:>7,} processed | {accepted:>5,} dogs ({accepted/total*100:.1f}%) "
                  f"| {rate:.0f} img/s | ETA {eta/60:.1f} min")

    flush()

    if not args.dry_run and current_sink:
        current_sink.close()

    elapsed = time.time() - t0
    print()
    print(f"{'=' * 50}")
    print(f"Total processed : {total:,}")
    print(f"Dogs detected   : {accepted:,} ({accepted/total*100:.1f}%)")
    print(f"Confidence thr. : {args.conf}")
    print(f"Time            : {elapsed:.0f}s ({total/elapsed:.0f} img/s)")
    if not args.dry_run:
        print(f"Shards written  : {shard_idx}")
        print(f"Output          : {args.output_dir}")


if __name__ == "__main__":
    main()
