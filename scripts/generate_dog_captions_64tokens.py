"""Generate synthetic dog-photo captions with GPT-5.5 (text-only, no source image), targeting
short captions.

Same as scripts/generate_dog_captions.py, except the caption-length instruction is tightened
from 50-90 words to 25-40 words — see scripts/recaption_dogs_64tokens.py's docstring for the
tokens/word math behind that number. Everything else (scene knobs, breed/coat-color coverage,
companion-dog/human handling) is unchanged.

Invents plausible photorealistic scenes and captions them in the same structure/style as
scripts/recaption_dogs_64tokens.py's real-photo captions.

Writes one .txt per caption to --output-dir/{breed}/captions/{breed}_{idx:05d}.txt. Pair with
scripts/generate_dog_images.py to render an image per caption, then pack both with
scripts/pack_dogs_wds.py (unmodified — it already expects this exact breed/captions layout).

Usage:
    export OPENAI_API_KEY=sk-...
    uv run python scripts/generate_dog_captions_64tokens.py --dry-run 3
    uv run python scripts/generate_dog_captions_64tokens.py --output-dir /data3/rey/dogs_synthetic_64tok
    uv run python scripts/generate_dog_captions_64tokens.py --output-dir /data3/rey/dogs_synthetic_64tok --overwrite
"""

import argparse
import random
import threading
import time
import zlib
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

# Real-world coat-color ranges per breed. Without this knob the model defaults to a single
# "iconic" color for every sample of a breed (e.g. Afghan Hounds always cream, Great Danes
# always blue-gray) even though the breed varies a lot in reality — this forces coverage.
COAT_COLORS = {
    "afghan_hound": ["cream", "red", "black", "blue", "brindle", "black-and-tan domino", "silver"],
    "beagle": ["tricolor", "red-and-white", "lemon-and-white", "tan-and-white"],
    "boxer": ["fawn", "brindle", "fawn with white markings", "brindle with white markings"],
    "chihuahua": ["fawn", "black", "white", "cream", "chocolate", "brindle", "blue", "black-and-tan"],
    "chow": ["red", "black", "blue", "cinnamon", "cream"],
    "corgi": ["red-and-white", "sable-and-white", "fawn-and-white", "black-and-tan-and-white (tricolor)"],
    "doberman": ["black-and-rust", "red-and-rust", "blue-and-rust", "fawn-and-rust"],
    "french_bulldog": ["brindle", "fawn", "cream", "pied", "black"],
    "german_shepherd": ["black-and-tan", "sable", "solid black", "bicolor black-and-red"],
    "golden_retriever": ["light golden", "golden", "dark golden"],
    "great_dane": ["fawn", "brindle", "black", "harlequin", "mantle (black-and-white)", "blue"],
    "labrador_retriever": ["black", "yellow", "chocolate"],
    "pomeranian": ["orange", "black", "cream", "white", "blue merle", "black-and-tan", "sable"],
    "pug": ["fawn", "black"],
    "rottweiler": ["black with rust/mahogany markings"],
    "samoyed": ["white", "cream", "biscuit"],
    "shih_tzu": ["gold-and-white", "black", "black-and-white", "red-and-white", "brindle", "solid black", "liver"],
    "siberian_husky": ["black-and-white", "gray-and-white", "red-and-white", "agouti wolf-gray", "all-white", "black"],
    "standard_poodle": ["black", "white", "apricot", "red", "brown", "silver", "cream", "blue"],
    "yorkshire_terrier": ["steel-blue-and-tan"],
}

# Adapted from the human-prose SYSTEM_PROMPT in recaption_dogs_64tokens.py: same tightened
# 25-40 word budget, but inventing a scene instead of describing a supplied image.
SYSTEM_PROMPT = (
    "You are a precise scene writer inventing training captions for a text-to-image model. "
    "You are NOT looking at a real photo — you are imagining a plausible, photorealistic snapshot "
    "of a dog and describing it as if to someone who cannot see it. "
    "Write 1-2 sentences of natural, flowing prose (25-40 words total). "
    "Always make the primary subject the named breed. "
    "Lead the first sentence with the breed. Vary sentence structure naturally — do not write a comma-separated attribute list.\n\n"
    "The scene sketch below gives you a setting, light, and action as loose inspiration, plus a coat color to use "
    "exactly. Never copy the setting/light wording verbatim into the caption — re-express it in your own fresh "
    "phrasing every time, as a real photographer or captioner would. Given the tight word budget, pick the one or "
    "two most visually salient details to close the caption with (e.g. a distinctive marking, the light, or the "
    "framing) rather than trying to list everything — don't default to the same closing template every time.\n\n"
    "Given the tight word budget, prioritize: coat color/texture, body build, pose or action, and one or two "
    "breed-defining or scene-defining details (distinctive markings, tail position, facial expression, gaze "
    "direction, background/setting, or notable light). Cover fewer details well rather than listing everything. "
    "If the scene sketch names a second or third dog, its breed, color, and interaction are given directly — use "
    "that exact breed and color and briefly state the interaction, in your own words. Never substitute a "
    "different companion breed than the one specified. "
    "Always write every dog breed name — the primary breed and any companion dogs — as a properly capitalized "
    "proper noun (e.g. 'German Shepherd', 'Jack Russell Terrier'), never lowercase and never with underscores, "
    "regardless of how it's cased in the scene sketch. "
    "If the scene sketch gives a human interaction, briefly stage that specific interaction naturally instead of "
    "a generic 'watches nearby' — note who's present and what they're doing, in as few words as the budget allows. "
    "If the dog is wearing anything (collar, leash, bandana, harness, clothes), mention it only if there's room. "
    "Only include people, a second/third dog, or an accessory if the scene sketch below asks for one — do not "
    "invent extra subjects or extra dogs beyond what's specified. "
    "No opinions, no filler phrases ('the image shows', 'can be seen', 'appears to be', 'it seems'), no 'I'.\n\n"
    "Examples of good captions:\n"
    "- An adult Golden Retriever with a freshly groomed wavy golden coat rests on a red paw-print pillow indoors, tail still, mouth slightly open. It gazes off-frame in warm light against a soft blurred background.\n"
    "- An adult Siberian Husky with a thick black-and-white coat and piercing blue eyes stands alert on snow-dusted steps, tail curled over its back. It faces the camera in bright daylight, sharp and full-body.\n"
    "- An adult Standard Poodle in a show clip stands calmly on a dirt path in golden grass, tail raised, gaze turned left. Warm late-afternoon light catches its curly white coat in a full-body side profile.\n"
    "- Two adult Beagles with tricolor coats play on a grassy field — one stands wagging its tail, pawing at the other, who lies on its back beneath it. Bright midday sun lights the full-body scene.\n"
    "- An adult Boxer with a fawn coat and black muzzle stands on grass in a harness, panting and looking at the camera. A young woman in a blue t-shirt kneels beside it, gripping the leash.\n"
)

# Lightweight scene knobs randomized per-request so 500 captions/breed don't collapse
# toward the same "golden hour, panting, grass" scene. Passed as loose hints, not slot-filling.
SETTINGS = [
    "a backyard lawn", "a dog show ring", "a sandy beach", "a snow-covered yard",
    "a plain indoor studio backdrop", "a public park path", "a living room on a couch",
    "a forest hiking trail", "the back seat of a car", "an agility course with tunnels and poles",
    "a city sidewalk", "an open farm field", "a veterinary exam table", "a wooden boat dock",
    "a covered porch", "a kitchen floor", "a suburban driveway", "a lake shoreline",
    "a hallway indoors", "a mountain overlook",
]

LIGHTS = [
    "golden hour sunlight", "overcast diffuse daylight", "harsh midday sun",
    "warm artificial indoor light", "soft window light", "studio strobe lighting",
    "dusk light with long shadows", "bright natural daylight", "cool fluorescent indoor light",
]

ACTIONS = [
    "sitting upright", "standing alert", "running", "playing fetch with a ball",
    "sleeping curled up", "swimming", "digging in dirt", "being brushed and groomed",
    "jumping mid-air", "tugging on a rope toy", "walking on a leash", "rolling on its back",
    "eating from a bowl", "shaking off water", "howling with head raised",
]

ACCESSORIES = [
    None, None, None, "a plain collar", "a bandana", "a harness and leash",
    "a raincoat", "a show lead", "a knitted sweater", "a cone collar",
]

# Breeds a second/third dog can be drawn from. Left unconstrained, GPT reliably defaults to
# a tiny "generically cute" pool (Jack Russell Terrier, Cavalier King Charles Spaniel) — this
# forces real spread. Deliberately includes non-purebred entries too, since real photo sets do.
SECOND_DOG_BREEDS = [
    "jack russell terrier", "cavalier king charles spaniel", "border collie", "australian shepherd",
    "bernese mountain dog", "cocker spaniel", "dachshund", "shiba inu", "bull terrier", "weimaraner",
    "vizsla", "brittany spaniel", "english springer spaniel", "miniature schnauzer",
    "west highland white terrier", "cairn terrier", "basset hound", "bloodhound", "newfoundland",
    "st. bernard", "greyhound", "whippet", "australian cattle dog", "staffordshire bull terrier",
    "american bulldog", "toy poodle", "miniature poodle", "papillon", "maltese", "bichon frise",
    "collie", "shetland sheepdog", "portuguese water dog", "irish setter", "english setter",
    "pointer", "akita", "shar pei", "cane corso", "dalmatian", "old english sheepdog",
    "mixed-breed rescue dog", "tan mixed-breed dog", "black mixed-breed dog",
] + [b.replace("_", " ") for b in BREEDS]  # occasionally the same species pool too (e.g. two Goldens together)

GENERIC_COAT_COLORS = [
    "black", "white", "brown", "tan", "cream", "brindle", "gray", "red",
    "black-and-white", "black-and-tan", "tricolor", "golden", "fawn", "liver-and-white",
]

# Concrete dog-dog interactions to hand to the model, instead of leaving "interacting" to its
# default (which collapses to a generic nose-to-nose sniff almost every time).
DOG_INTERACTIONS = [
    "play-bowing to invite a chase", "locked in a tug-of-war over a toy",
    "wrestling playfully on the ground", "sniffing noses in a greeting",
    "racing alongside it stride for stride", "standing back-to-back, alert to different directions",
    "curled up sleeping against its side", "grooming the other's ear",
    "barking playfully inches from its face", "trotting single-file just behind it",
    "leaning its head over the other's shoulder", "circling it warily, unsure whether to play",
    "chasing it in a tight circle", "resting its chin across the other's back",
]

# Same idea for the human(s) in frame — otherwise captions default to "watches nearby".
HUMAN_INTERACTIONS = [
    "crouching to offer a treat from an open palm", "kneeling with a hand resting on its back",
    "holding the leash taut mid-stride", "laughing while pointing at something off-frame",
    "brushing its coat with slow, even strokes", "kneeling nose-to-nose with the dog",
    "reaching down to buckle its collar", "clapping to call it over",
    "sitting on the ground with the dog leaning into their lap", "wiping mud from its paws with a towel",
    "holding up a toy just out of reach, teasing", "scratching behind its ears while it leans in",
]


def pick_companion_breed(primary_breed_label: str, rng: random.Random, exclude: set) -> str:
    for _ in range(10):
        candidate = rng.choice(SECOND_DOG_BREEDS)
        if candidate != primary_breed_label and candidate not in exclude:
            return candidate
    return candidate


def build_scene_hint(breed: str, rng: random.Random) -> str:
    breed_label = breed.replace("_", " ")
    setting = rng.choice(SETTINGS)
    light = rng.choice(LIGHTS)
    action = rng.choice(ACTIONS)
    accessory = rng.choice(ACCESSORIES)
    coat_color = rng.choice(COAT_COLORS[breed])
    num_dogs = rng.choices([1, 2, 3], weights=[85, 12, 3])[0]
    people = rng.choices(
        ["no people", "one adult", "two adults", "an adult and a child"],
        weights=[6, 2, 1, 1],
    )[0]

    parts = [
        f"Coat color: {coat_color}.",
        f"Setting: {setting}.", f"Light: {light}.", f"Action/pose: {action}.",
    ]

    chosen_breeds = {breed_label}
    for n, ordinal in zip(range(num_dogs - 1), ["Second", "Third"]):
        companion_breed = pick_companion_breed(breed_label, rng, chosen_breeds)
        chosen_breeds.add(companion_breed)
        companion_color = rng.choice(GENERIC_COAT_COLORS)
        interaction = rng.choice(DOG_INTERACTIONS)
        parts.append(f"{ordinal} dog: a {companion_color} {companion_breed}, {interaction}.")

    parts.append(f"People in scene: {people}.")
    if people != "no people":
        parts.append(f"Human interaction: {rng.choice(HUMAN_INTERACTIONS)}.")
    if accessory:
        parts.append(f"The dog is wearing/using: {accessory}.")
    return " ".join(parts)


def stable_seed(key: str, base: int) -> int:
    return base + zlib.crc32(key.encode("utf-8"))


def generate_caption(client: OpenAI, breed: str, rng: random.Random, model: str, retries: int = 5) -> str:
    breed_label = breed.replace("_", " ")
    scene_hint = build_scene_hint(breed, rng)
    user_content = (
        f"Breed: {breed_label}\n\n"
        f"Scene sketch (loose inspiration — write natural prose, don't just restate it):\n{scene_hint}\n\n"
        "Invent and caption this scene."
    )
    for attempt in range(retries):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
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
    raise RuntimeError(f"Failed after {retries} retries for breed {breed}")


def process_item(client, breed, idx, cap_path, seed, model, overwrite):
    if cap_path.exists() and not overwrite:
        return "skipped", breed, idx, None
    try:
        rng = random.Random(seed)
        caption = generate_caption(client, breed, rng, model)
        cap_path.write_text(caption, encoding="utf-8")
        return "done", breed, idx, caption
    except Exception as e:
        return "failed", breed, idx, str(e)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="/data3/rey/dogs_synthetic_64tok")
    parser.add_argument("--count-per-breed", type=int, default=500)
    parser.add_argument("--model", default="gpt-5.5")
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true",
                         help="Overwrite existing captions (default: skip already-generated)")
    parser.add_argument("--dry-run", type=int, metavar="N", default=0,
                         help="Print N sample captions per breed and exit, do not write files")
    args = parser.parse_args()

    client = OpenAI()  # reads OPENAI_API_KEY from env

    if args.dry_run:
        rng = random.Random(args.seed)
        for breed in BREEDS:
            print(f"\n[{breed}]")
            for _ in range(args.dry_run):
                try:
                    caption = generate_caption(client, breed, rng, args.model)
                    print(f"  {caption}\n")
                except Exception as e:
                    print(f"  ERROR: {e}")
        return

    out_dir = Path(args.output_dir)
    work_items = []
    for breed in BREEDS:
        cap_dir = out_dir / breed / "captions"
        cap_dir.mkdir(parents=True, exist_ok=True)
        for idx in range(args.count_per_breed):
            cap_path = cap_dir / f"{breed}_{idx:05d}.txt"
            item_seed = stable_seed(f"{breed}_{idx:05d}", args.seed)
            work_items.append((breed, idx, cap_path, item_seed))

    total = len(work_items)
    done = skipped = failed = 0
    lock = threading.Lock()
    start_time = time.time()
    print(f"Generating {total} captions with {args.workers} workers...\n")

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(process_item, client, breed, idx, cap_path, seed, args.model, args.overwrite): (breed, idx)
            for breed, idx, cap_path, seed in work_items
        }
        for future in as_completed(futures):
            status, breed, idx, info = future.result()
            with lock:
                if status == "done":
                    done += 1
                    if done % 200 == 0:
                        elapsed = time.time() - start_time
                        rate = (done + skipped) / elapsed * 60
                        print(f"  {done} done, {skipped} skipped, {failed} failed / {total} total — {rate:.1f} captions/min")
                elif status == "skipped":
                    skipped += 1
                else:
                    failed += 1
                    print(f"  ERROR {breed}_{idx:05d}: {info}")

    print(f"\nDone: {done} generated, {skipped} skipped, {failed} failed / {total} total")


if __name__ == "__main__":
    main()
