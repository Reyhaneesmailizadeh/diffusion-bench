"""Recaption dog breed images using GPT-5.5 vision.

Writes new captions to --output-dir, leaving the original dataset untouched.
Images are symlinked (not copied) to save disk space.

Usage:
    export OPENAI_API_KEY=sk-...
    uv run python scripts/recaption_dogs.py \
        --data-dir /data3/rey/dogs \
        --output-dir /data3/rey/dogs_recaptioned
    uv run python scripts/recaption_dogs.py \
        --data-dir /data3/rey/dogs \
        --output-dir /data3/rey/dogs_recaptioned \
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

# Original structured system prompt (comma-separated single-sentence format)
# SYSTEM_PROMPT = (
#     "You are a precise image captioner for training a text-to-image model. "
#     "Write exactly one sentence ending with a single period (max 100 words). Always start with the breed name. "
#     "Always identify the primary subject as the named breed, even if other animals are present. "
#     "Describe: breed, coat color/texture and grooming condition (e.g. freshly groomed, wet coat, muddy, show clip, matted/unkempt), body build and size (e.g. lean, muscular, stocky, puppy, adult, senior), "
#     "breed-defining physical features (e.g. eye color, ear shape, tail shape, distinctive markings), "
#     "tail position and state (e.g. tail wagging, tail tucked between legs, tail raised, tail curled over back), "
#     "pose or action, facial expression and emotional state (e.g. mouth open panting relaxed, alert with mouth closed, ears pinned back anxious, playful), "
#     "gaze direction (e.g. looking directly at camera, gazing off-frame to the left), "
#     "background/setting, specific light source (e.g. golden hour sunlight, window light, studio, artificial indoor), "
#     "depth of field (e.g. soft blurred bokeh background, sharp throughout), "
#     "and camera angle and shot framing (e.g. close-up portrait, full body, overhead). "
#     "If multiple dogs are present, state how many, describe each dog's position relative to the others (e.g. one standing behind, one lying in front, side by side), and describe their interaction (e.g. playing, one sniffing the other, grooming, sleeping together, one chasing the other). "
#     "If one or more people are present, describe each person separately: gender, approximate age, hair color and style, facial expression, "
#     "and clothing in detail (e.g. color, type of top, bottom, footwear, accessories); "
#     "also describe their body posture, position relative to the dog and to each other, any visible limb details relevant to the interaction (e.g. arms wrapped around the dog, kneeling, reaching down), "
#     "and the type of interaction with the dog (e.g. holding the leash, cuddling, training, walking beside, pointing at). "
#     "If the dog is wearing anything (leash, collar, bandana, clothes, etc.), mention it. "
#     "Omit any field that is not visible or not applicable. "
#     "Be specific and visual. No opinions, no 'the image shows', no filler. "
#     "Do not write more than one sentence.\n\n"
#     "Examples of good captions:\n"
#     "- Golden Retriever, adult, lean build, wavy golden coat freshly groomed, floppy ears, and soft brown eyes, tail resting still to the side, lying relaxed on a red paw-print pillow indoors, mouth slightly open with a relaxed expression, gazing off-frame to the right, warm artificial light, soft blurred bokeh background, close-up portrait.\n"
#     "- Siberian Husky, adult, athletic build, thick black and white coat, piercing blue eyes, and erect triangular ears, tail curled over back, standing alert with mouth closed and focused expression, looking directly at camera, on snow-dusted wooden steps, bright natural daylight, sharp throughout, medium full-body shot.\n"
#     "- Standard Poodle, adult, slender build, curly white coat in a show clip, wearing a blue collar with a tag, tail raised, standing calmly with mouth closed, gazing off-frame to the left, on a dirt path surrounded by golden grass, warm late-afternoon sunlight, soft blurred bokeh background, full-body side profile.\n"
#     "- Two Beagles, tricolor coats with short clean coats, one adult standing upright on the left with tail wagging, playfully pawing at the other who lies on its back beneath it with tail tucked, both with mouths open in playful expressions, neither looking at camera, on a grassy field, bright midday sunlight, sharp throughout, full-body overhead shot.\n"
#     "- Boxer, adult, muscular build, fawn coat with a black muzzle, wearing a harness, tail nub wagging, mouth open panting with a relaxed expression, looking directly at camera, standing on grass while a young smiling woman with straight blonde hair wearing a blue fitted t-shirt, white jeans, and white sneakers kneels on the left gripping the leash, and a middle-aged man with short brown hair wearing a gray polo and khaki shorts stands on the right with his hand resting on the dog's back, bright outdoor sunlight, soft blurred bokeh background, full-body shot.\n"
# )

# Human-prose system prompt (2-3 natural sentences)
SYSTEM_PROMPT = (
    "You are a precise image captioner for training a text-to-image model. "
    "Write 2–3 sentences of natural, flowing prose (50–90 words total). "
    "Always identify the primary subject as the named breed, even if other animals are present. "
    "Lead the first sentence with the breed. Vary sentence structure naturally — do not write a comma-separated attribute list. "
    "Write as if describing the image to someone who cannot see it.\n\n"
    "Cover these visual details where visible: breed, coat color/texture and grooming condition (e.g. freshly groomed, wet, muddy, show clip, matted), "
    "body build and size (e.g. lean, muscular, stocky, puppy, adult, senior), "
    "breed-defining features (eye color, ear shape, tail shape, distinctive markings), "
    "tail position (wagging, tucked, raised, curled over back), "
    "pose or action, facial expression and emotional state (e.g. panting relaxed, alert mouth closed, ears pinned back anxious, playful), "
    "gaze direction (looking directly at camera, gazing off-frame to the left), "
    "background and setting, light source (golden hour sunlight, window light, studio, artificial indoor), "
    "depth of field (soft blurred bokeh, sharp throughout), "
    "and camera angle and framing (close-up portrait, full body, overhead). "
    "If multiple dogs are present, state how many, describe each dog's position relative to the others, and describe their interaction. "
    "If people are present, describe each person: gender, approximate age, hair, clothing (color, type of top/bottom/footwear), posture, position relative to the dog, and the nature of the interaction. "
    "If the dog is wearing anything (collar, leash, bandana, harness, clothes), mention it. "
    "Omit any detail that is not visible or not applicable. "
    "No opinions, no filler phrases ('the image shows', 'can be seen', 'appears to be', 'it seems'), no 'I'.\n\n"
    "Examples of good captions:\n"
    "- An adult Golden Retriever with a freshly groomed wavy golden coat and soft brown eyes rests on a red paw-print pillow indoors, its tail lying still to the side. Its mouth is slightly open in a relaxed expression as it gazes off-frame to the right. Warm artificial light falls across the scene against a soft blurred background in a close-up portrait.\n"
    "- An adult Siberian Husky with a thick black-and-white coat, piercing blue eyes, and erect triangular ears stands alert on snow-dusted wooden steps, tail curled over its back. It faces the camera directly with its mouth closed and a focused expression. Bright natural daylight illuminates the scene sharply from background to foreground in a medium full-body shot.\n"
    "- An adult Standard Poodle in a show clip, wearing a blue collar with a tag, stands calmly on a dirt path surrounded by golden grass, its tail raised and gaze turned off-frame to the left. Its slender build and curly white coat catch the warm late-afternoon sunlight against a soft blurred bokeh background in a full-body side profile.\n"
    "- Two adult Beagles with short tricolor coats play on a grassy field, one standing upright on the left with its tail wagging and playfully pawing at the other, who lies on its back beneath it with its tail tucked. Both have their mouths open in playful expressions and neither looks at the camera. Bright midday sunlight illuminates the scene sharply throughout in a full-body overhead shot.\n"
    "- An adult Boxer with a fawn coat, black muzzle, and wagging tail nub stands on grass wearing a harness, panting with a relaxed open-mouthed expression and looking directly at the camera. A young woman with straight blonde hair in a blue fitted t-shirt, white jeans, and white sneakers kneels to the left gripping the leash, while a middle-aged man in a gray polo and khaki shorts stands to the right with his hand on the dog's back. Bright outdoor sunlight and a soft blurred bokeh background frame the full-body shot.\n"
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
    parser.add_argument("--output-dir", default="/data3/rey/dogs_recaptioned")
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
