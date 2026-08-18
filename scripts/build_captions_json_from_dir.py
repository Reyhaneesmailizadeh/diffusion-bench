"""Convert a folder of per-breed caption .txt files (e.g. reward_model/breeds/{breed}/*.txt,
as written by scripts/generate_captions_RM.py) into the captions.json format expected by
scripts/generate_dog_comparison.py --captions-json (a list of {breed, stem, text} dicts).

Usage:
    uv run python scripts/build_captions_json_from_dir.py \
        --captions-dir reward_model/breeds \
        --output-json reward_model/captions_500.json
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--captions-dir", required=True)
    parser.add_argument("--output-json", required=True)
    args = parser.parse_args()

    captions_dir = Path(args.captions_dir)
    items = []
    for breed_dir in sorted(p for p in captions_dir.iterdir() if p.is_dir()):
        breed = breed_dir.name
        for txt_path in sorted(breed_dir.glob(f"{breed}_*.txt")):
            stem = txt_path.stem[len(breed) + 1:]  # strip "{breed}_" prefix -> just the index
            text = txt_path.read_text(encoding="utf-8").strip()
            items.append({"breed": breed, "stem": stem, "text": text})

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w") as f:
        json.dump(items, f, indent=2)

    breeds = sorted(set(it["breed"] for it in items))
    print(f"wrote {len(items)} items across {len(breeds)} breeds to {args.output_json}")


if __name__ == "__main__":
    main()
