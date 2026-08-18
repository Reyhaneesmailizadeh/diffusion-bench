"""
Score images in reward_model/images/{pretrained,sft}/<breed>/ with HPSv2,
using each image's paired .txt file as its prompt, and report per-breed
and overall mean scores for both variants.

Loads the HPSv2 model/checkpoint once and reuses it for every image
(hpsv2.score() reloads the checkpoint on every call, which is far too
slow for a few thousand images).
"""
import os
import glob
import json

import torch
import huggingface_hub
from PIL import Image

from hpsv2.src.open_clip import create_model_and_transforms, get_tokenizer
from hpsv2.utils import root_path, hps_version_map

IMAGES_ROOT = os.path.join(os.path.dirname(__file__), 'images')
VARIANTS = ['pretrained', 'sft']
HPS_VERSION = 'v2.0'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_model():
    model, _, preprocess_val = create_model_and_transforms(
        'ViT-H-14',
        'laion2B-s32B-b79K',
        precision='amp',
        device=DEVICE,
        jit=False,
        force_quick_gelu=False,
        force_custom_text=False,
        force_patch_dropout=False,
        force_image_size=None,
        pretrained_image=False,
        image_mean=None,
        image_std=None,
        light_augmentation=True,
        aug_cfg={},
        output_dict=True,
        with_score_predictor=False,
        with_region_predictor=False,
    )

    os.makedirs(root_path, exist_ok=True)
    cp = huggingface_hub.hf_hub_download('xswu/HPSv2', hps_version_map[HPS_VERSION])
    checkpoint = torch.load(cp, map_location=DEVICE)
    model.load_state_dict(checkpoint['state_dict'])
    model = model.to(DEVICE)
    model.eval()

    tokenizer = get_tokenizer('ViT-H-14')
    return model, preprocess_val, tokenizer


@torch.no_grad()
def score_one(model, preprocess_val, tokenizer, img_path: str, prompt: str) -> float:
    image = preprocess_val(Image.open(img_path)).unsqueeze(0).to(DEVICE, non_blocking=True)
    text = tokenizer([prompt]).to(DEVICE, non_blocking=True)
    with torch.cuda.amp.autocast():
        outputs = model(image, text)
        image_features, text_features = outputs['image_features'], outputs['text_features']
        logits_per_image = image_features @ text_features.T
        hps_score = torch.diagonal(logits_per_image).cpu().numpy()
    return float(hps_score[0])


def score_variant(model, preprocess_val, tokenizer, variant: str) -> dict:
    variant_dir = os.path.join(IMAGES_ROOT, variant)
    breed_scores = {}

    for breed in sorted(os.listdir(variant_dir)):
        breed_dir = os.path.join(variant_dir, breed)
        if not os.path.isdir(breed_dir):
            continue

        img_paths = sorted(glob.glob(os.path.join(breed_dir, '*.png')))
        scores = []
        for img_path in img_paths:
            prompt_path = os.path.splitext(img_path)[0] + '.txt'
            with open(prompt_path, 'r', encoding='utf-8') as f:
                prompt = f.read().strip()
            scores.append(score_one(model, preprocess_val, tokenizer, img_path, prompt))

        breed_scores[breed] = scores
        mean = sum(scores) / len(scores) if scores else float('nan')
        print(f'[{variant}] {breed}: {len(scores)} images, mean HPS = {mean:.4f}')

    return breed_scores


def summarize(all_scores: dict) -> None:
    print('\n=== Summary (mean HPSv2 score) ===')
    header = f'{"breed":22s} {"pretrained":>12s} {"sft":>12s} {"delta":>10s}'
    print(header)
    print('-' * len(header))

    breeds = sorted(all_scores['pretrained'].keys())
    for breed in breeds:
        pre = all_scores['pretrained'][breed]
        sft = all_scores['sft'][breed]
        pre_mean = sum(pre) / len(pre)
        sft_mean = sum(sft) / len(sft)
        print(f'{breed:22s} {pre_mean:12.4f} {sft_mean:12.4f} {sft_mean - pre_mean:10.4f}')

    for variant in VARIANTS:
        all_vals = [s for scores in all_scores[variant].values() for s in scores]
        overall_mean = sum(all_vals) / len(all_vals)
        print(f'\nOverall {variant} mean HPS ({len(all_vals)} images): {overall_mean:.4f}')


def main():
    model, preprocess_val, tokenizer = load_model()

    all_scores = {}
    for variant in VARIANTS:
        all_scores[variant] = score_variant(model, preprocess_val, tokenizer, variant)

    summarize(all_scores)

    out_path = os.path.join(os.path.dirname(__file__), 'hps_scores.json')
    with open(out_path, 'w') as f:
        json.dump(all_scores, f, indent=2)
    print(f'\nRaw per-image scores written to {out_path}')


if __name__ == '__main__':
    main()
