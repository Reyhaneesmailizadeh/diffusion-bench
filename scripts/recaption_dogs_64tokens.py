"""Recaption dog breed images using GPT-5.5 vision, targeting short captions.

Same as scripts/recaption_dogs.py, except the caption-length instruction is tightened from
50-90 words to 25-40 words. Empirically (Qwen3-0.6B tokenizer, tokens/word ratio ~1.32 on the
original 26k-caption set), a 25-40 word caption lands comfortably under a 64-token budget
(dropping conditioning.text_encoder.max_length from 128 to 64 halves attention sequence
length/cost with near-zero truncation risk, vs. ~99.6% truncation if the *existing* long
captions were simply truncated at 64 instead of regenerated shorter).

Writes new captions to --output-dir, leaving the original dataset untouched.
Images are symlinked (not copied) to save disk space.

Usage:
    export OPENAI_API_KEY=sk-...
    uv run python scripts/recaption_dogs_64tokens.py \
        --data-dir /data3/rey/dogs \
        --output-dir /data3/rey/dogs_recaptioned_64tok
    uv run python scripts/recaption_dogs_64tokens.py \
        --data-dir /data3/rey/dogs \
        --output-dir /data3/rey/dogs_recaptioned_64tok \
        --overwrite
"""

import argparse
import base64
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from openai import OpenAI, RateLimitError

BREEDS = sorted([
    "afghan_hound", "beagle", "boxer", "chihuahua", "chow",
    "corgi", "doberman", "french_bulldog", "german_shepherd",
    "golden_retriever", "great_dane", "labrador_retriever",
    "pomeranian", "pug", "rottweiler", "samoyed", "shih_tzu",
    "siberian_husky", "standard_poodle", "yorkshire_terrier",
])

# Human-prose system prompt (1-2 short natural sentences, 25-40 words total).
SYSTEM_PROMPT = (
    "You are a precise image captioner for training a text-to-image model. "
    "Write 1-2 sentences of natural, flowing prose (25-40 words total). "
    "Always identify the primary subject as the named breed, even if other animals are present. "
    "Lead the first sentence with the breed. Vary sentence structure naturally — do not write a comma-separated attribute list. "
    "Write as if describing the image to someone who cannot see it.\n\n"
    "Given the tight word budget, prioritize the most visually salient details first: breed, coat color/texture, "
    "body build, pose or action, and one or two breed-defining or scene-defining details (e.g. distinctive "
    "markings, tail position, facial expression, gaze direction, background/setting, or notable light). "
    "Cover fewer details well rather than listing everything — omit anything not clearly visible or not essential "
    "to picturing the scene. "
    "If multiple dogs are present, state how many and briefly note the interaction. "
    "If people are present, briefly note who and how they're interacting with the dog. "
    "If the dog is wearing anything (collar, leash, bandana, harness, clothes), mention it only if there's room. "
    "No opinions, no filler phrases ('the image shows', 'can be seen', 'appears to be', 'it seems'), no 'I'.\n\n"
    "Examples of good captions:\n"
    "- An adult Golden Retriever with a freshly groomed wavy golden coat rests on a red paw-print pillow indoors, tail still, mouth slightly open. It gazes off-frame in warm light against a soft blurred background.\n"
    "- An adult Siberian Husky with a thick black-and-white coat and piercing blue eyes stands alert on snow-dusted steps, tail curled over its back. It faces the camera in bright daylight, sharp and full-body.\n"
    "- An adult Standard Poodle in a show clip stands calmly on a dirt path in golden grass, tail raised, gaze turned left. Warm late-afternoon light catches its curly white coat in a full-body side profile.\n"
    "- Two adult Beagles with tricolor coats play on a grassy field — one stands wagging its tail, pawing at the other, who lies on its back beneath it. Bright midday sun lights the full-body scene.\n"
    "- An adult Boxer with a fawn coat and black muzzle stands on grass in a harness, panting and looking at the camera. A young woman in a blue t-shirt kneels beside it, gripping the leash.\n"
)


def encode_image(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("utf-8")


def recaption(client: OpenAI, image_path: Path, breed: str, retries: int = 3) -> str:
    breed_label = breed.replace("_", " ")
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model="gpt-5.5",
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": [
                        {"type": "text", "text": f"Breed: {breed_label}\n\nCaption this image."},
                        {"type": "image_url", "image_url": {
                            "url": f"data:image/jpeg;base64,{encode_image(image_path)}",
                            "detail": "low",  # cheaper; enough for captioning
                        }},
                    ]},
                ],
                max_completion_tokens=5000,
            )
            choice = response.choices[0]
            content = choice.message.content
            if not content or not content.strip():
                refusal = getattr(choice.message, "refusal", None)
                reason = refusal or choice.finish_reason or "empty response"
                raise RuntimeError(f"Empty caption: {reason}")
            return content.strip()
        except RateLimitError:
            wait = 2 ** attempt * 5
            print(f"  Rate limited, waiting {wait}s...")
            time.sleep(wait)
    raise RuntimeError(f"Failed after {retries} retries: {image_path}")


def process_image(client, img_path, breed, cap_path, link_path, overwrite):
    if cap_path.exists() and not overwrite:
        return "skipped", img_path, None
    try:
        caption = recaption(client, img_path, breed)
        cap_path.write_text(caption, encoding="utf-8")
        if not link_path.exists():
            link_path.symlink_to(img_path.resolve())
        return "done", img_path, caption
    except Exception as e:
        return "failed", img_path, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data3/rey/dogs")
    parser.add_argument("--output-dir", default="/data3/rey/dogs_recaptioned_64tok")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing captions (default: skip already captioned)")
    parser.add_argument("--dry-run", type=int, metavar="N", default=0,
                        help="Process only N images per breed and print captions, do not write files")
    parser.add_argument("--workers", type=int, default=20,
                        help="Number of parallel API requests (default: 20)")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.output_dir)
    client = OpenAI()  # reads OPENAI_API_KEY from env

    if args.dry_run:
        # Sequential for dry run so output is readable
        for breed in BREEDS:
            src_images_dir = data_dir / breed / "images"
            image_paths = sorted(src_images_dir.glob("*.jpg"))[:args.dry_run]
            print(f"\n[{breed}] {len(image_paths)} images")
            for img_path in image_paths:
                try:
                    caption = recaption(client, img_path, breed)
                    print(f"  [{img_path.name}]\n  {caption}\n")
                except Exception as e:
                    print(f"  ERROR {img_path.name}: {e}")
        return

    # Collect all work items
    work_items = []
    for breed in BREEDS:
        src_images_dir = data_dir / breed / "images"
        out_images_dir = out_dir / breed / "images"
        out_captions_dir = out_dir / breed / "captions"
        out_images_dir.mkdir(parents=True, exist_ok=True)
        out_captions_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(src_images_dir.glob("*.jpg")):
            cap_path = out_captions_dir / f"{img_path.stem}.txt"
            link_path = out_images_dir / img_path.name
            work_items.append((img_path, breed, cap_path, link_path))

    total = len(work_items)
    done = skipped = failed = 0
    lock = threading.Lock()
    start_time = time.time()
    print(f"Processing {total} images with {args.workers} workers...\n")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_image, client, img_path, breed, cap_path, link_path, args.overwrite): img_path
            for img_path, breed, cap_path, link_path in work_items
        }
        for future in as_completed(futures):
            status, img_path, info = future.result()
            with lock:
                if status == "done":
                    done += 1
                    if done % 50 == 0:
                        elapsed = time.time() - start_time
                        rate = (done + skipped) / elapsed * 60
                        print(f"  {done} done, {skipped} skipped, {failed} failed / {total} total — {rate:.1f} img/min")
                elif status == "skipped":
                    skipped += 1
                else:
                    failed += 1
                    print(f"  ERROR {img_path.name}: {info}")

    print(f"\nDone: {done} recaptioned, {skipped} skipped, {failed} failed / {total} total")


if __name__ == "__main__":
    main()
