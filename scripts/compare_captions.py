"""Compare caption statistics between original and recaptioned dog datasets.

Usage:
    uv run python scripts/compare_captions.py \
        --old-dir /data3/rey/dogs \
        --new-dir /data3/rey/dogs_recaptioned
"""

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np

BREEDS = sorted([
    "afghan_hound", "beagle", "boxer", "chihuahua", "chow",
    "corgi", "doberman", "french_bulldog", "german_shepherd",
    "golden_retriever", "great_dane", "labrador_retriever",
    "pomeranian", "pug", "rottweiler", "samoyed", "shih_tzu",
    "siberian_husky", "standard_poodle", "yorkshire_terrier",
])

FILLER_PHRASES = [
    "the image", "the photo", "appears to be", "in the background",
    "it appears", "suggesting", "this is a", "can be seen",
    "is visible", "we can see", "there is a", "there are",
]

COLOR_WORDS = [
    "black", "white", "brown", "golden", "tan", "gray", "grey", "red",
    "cream", "fawn", "blue", "silver", "brindle", "tricolor", "bicolor",
    "sable", "apricot", "liver", "merle", "chocolate",
]

ATTRIBUTE_PATTERNS = {
    "coat":       r"\bcoat\b",
    "ears":       r"\bears?\b",
    "tail":       r"\btail\b",
    "eyes":       r"\beyes?\b",
    "expression": r"\bexpression\b",
    "background": r"\bbackground\b",
    "lighting":   r"\b(light|sunlight|daylight|lighting|studio|shadow)\b",
    "camera angle": r"\b(close.up|full.body|overhead|portrait|profile|angle|shot|frame)\b",
    "breed start": None,  # handled separately
}


def load_captions(data_dir: Path) -> list[str]:
    captions = []
    for breed in BREEDS:
        cap_dir = data_dir / breed / "captions"
        if not cap_dir.exists():
            continue
        for p in sorted(cap_dir.glob("*.txt")):
            text = p.read_text(encoding="utf-8").strip()
            if text:
                captions.append(text)
    return captions


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-zA-Z']+", text.lower())


def sentence_count(text: str) -> int:
    return len([s for s in re.split(r"[.!?]+", text) if s.strip()])


def compute_stats(captions: list[str], label: str) -> dict:
    words_per = [len(tokenize(c)) for c in captions]
    chars_per = [len(c) for c in captions]
    sents_per = [sentence_count(c) for c in captions]

    all_words = [w for c in captions for w in tokenize(c)]
    total_words = len(all_words)
    unique_words = len(set(all_words))
    ttr = unique_words / total_words if total_words > 0 else 0

    top_words = Counter(all_words).most_common(20)

    breed_names = {b.replace("_", " ") for b in BREEDS} | {b.split("_")[0] for b in BREEDS}
    breed_start_rate = sum(
        1 for c in captions
        if any(c.lower().startswith(b) for b in breed_names)
    ) / len(captions)

    filler_counts = {
        phrase: sum(1 for c in captions if phrase in c.lower())
        for phrase in FILLER_PHRASES
    }
    filler_rate = sum(filler_counts.values()) / len(captions)

    color_counts_per = [
        sum(1 for cw in COLOR_WORDS if re.search(rf"\b{cw}\b", c.lower()))
        for c in captions
    ]

    attr_rates = {}
    for attr, pattern in ATTRIBUTE_PATTERNS.items():
        if attr == "breed start":
            attr_rates[attr] = breed_start_rate
        else:
            attr_rates[attr] = sum(
                1 for c in captions if re.search(pattern, c.lower())
            ) / len(captions)

    return {
        "label": label,
        "n": len(captions),
        "words_mean": np.mean(words_per),
        "words_median": np.median(words_per),
        "words_std": np.std(words_per),
        "words_min": np.min(words_per),
        "words_max": np.max(words_per),
        "chars_mean": np.mean(chars_per),
        "sents_mean": np.mean(sents_per),
        "sents_median": np.median(sents_per),
        "unique_vocab": unique_words,
        "total_words": total_words,
        "ttr": ttr,
        "top_words": top_words,
        "filler_rate": filler_rate,
        "filler_counts": filler_counts,
        "color_density_mean": np.mean(color_counts_per),
        "attr_rates": attr_rates,
    }


def print_comparison(old: dict, new: dict):
    W = 28

    def row(label, old_val, new_val, fmt=".1f"):
        print(f"  {label:<{W}} {old_val:{fmt}}   {new_val:{fmt}}")

    def pct_row(label, old_val, new_val):
        print(f"  {label:<{W}} {old_val*100:5.1f}%      {new_val*100:5.1f}%")

    print("=" * 65)
    print(f"  {'':>{W}} {'OLD':>10}   {'NEW':>10}")
    print(f"  {'':>{W}} {old['label']:>10}   {new['label']:>10}")
    print("=" * 65)

    print("\n── Caption count ──")
    print(f"  {'total captions':<{W}} {old['n']:>10}   {new['n']:>10}")

    print("\n── Word count ──")
    row("mean", old["words_mean"], new["words_mean"])
    row("median", old["words_median"], new["words_median"])
    row("std", old["words_std"], new["words_std"])
    row("min", old["words_min"], new["words_min"], fmt=".0f")
    row("max", old["words_max"], new["words_max"], fmt=".0f")

    print("\n── Character count ──")
    row("mean chars", old["chars_mean"], new["chars_mean"])

    print("\n── Sentence count ──")
    row("mean sentences", old["sents_mean"], new["sents_mean"])
    row("median sentences", old["sents_median"], new["sents_median"])

    print("\n── Lexical richness ──")
    print(f"  {'unique vocab':<{W}} {old['unique_vocab']:>10}   {new['unique_vocab']:>10}")
    print(f"  {'total words':<{W}} {old['total_words']:>10}   {new['total_words']:>10}")
    row("type-token ratio", old["ttr"], new["ttr"], fmt=".4f")

    print("\n── Content specificity ──")
    row("color words / caption", old["color_density_mean"], new["color_density_mean"])
    pct_row("filler phrase rate", old["filler_rate"] / 20, new["filler_rate"] / 20)  # normalize

    print("\n── Attribute coverage (% captions mentioning) ──")
    for attr in ATTRIBUTE_PATTERNS:
        pct_row(attr, old["attr_rates"][attr], new["attr_rates"][attr])

    print("\n── Filler phrases (caption count) ──")
    all_phrases = sorted(
        set(old["filler_counts"]) | set(new["filler_counts"])
    )
    for phrase in all_phrases:
        o = old["filler_counts"].get(phrase, 0)
        n = new["filler_counts"].get(phrase, 0)
        print(f"  '{phrase}'{'':>{max(1,30-len(phrase))}} {o:>8}   {n:>8}")

    print("\n── Top 20 words (old) ──")
    print("  " + "  ".join(f"{w}({c})" for w, c in old["top_words"]))

    print("\n── Top 20 words (new) ──")
    print("  " + "  ".join(f"{w}({c})" for w, c in new["top_words"]))

    print("\n" + "=" * 65)


def save_csv(old: dict, new: dict, path: Path):
    rows = []

    def add(section, metric, old_val, new_val, unit=""):
        rows.append({
            "section": section,
            "metric": metric,
            "original": f"{old_val:.4g}",
            "recaptioned": f"{new_val:.4g}",
            "unit": unit,
        })

    add("word count",      "mean",              old["words_mean"],        new["words_mean"],        "words")
    add("word count",      "median",            old["words_median"],      new["words_median"],      "words")
    add("word count",      "std",               old["words_std"],         new["words_std"],         "words")
    add("word count",      "min",               old["words_min"],         new["words_min"],         "words")
    add("word count",      "max",               old["words_max"],         new["words_max"],         "words")
    add("character count", "mean",              old["chars_mean"],        new["chars_mean"],        "chars")
    add("sentence count",  "mean",              old["sents_mean"],        new["sents_mean"],        "sentences")
    add("sentence count",  "median",            old["sents_median"],      new["sents_median"],      "sentences")
    add("lexical richness","unique vocab",      old["unique_vocab"],      new["unique_vocab"],      "words")
    add("lexical richness","total words",       old["total_words"],       new["total_words"],       "words")
    add("lexical richness","type-token ratio",  old["ttr"],               new["ttr"],               "")
    add("specificity",     "color words/caption",old["color_density_mean"],new["color_density_mean"],"words")
    add("specificity",     "filler phrase rate",old["filler_rate"]/20,   new["filler_rate"]/20,    "%")

    for attr in ATTRIBUTE_PATTERNS:
        add("attribute coverage", attr, old["attr_rates"][attr] * 100, new["attr_rates"][attr] * 100, "%")

    for phrase in sorted(old["filler_counts"]):
        add("filler phrases", f'"{phrase}"', old["filler_counts"][phrase], new["filler_counts"][phrase], "captions")

    for rank, ((ow, oc), (nw, nc)) in enumerate(zip(old["top_words"], new["top_words"]), 1):
        rows.append({
            "section": "top 20 words",
            "metric": f"rank {rank}",
            "original": f"{ow} ({oc})",
            "recaptioned": f"{nw} ({nc})",
            "unit": "",
        })

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["section", "metric", "original", "recaptioned", "unit"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"CSV saved to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--old-dir", default="/data3/rey/dogs")
    parser.add_argument("--new-dir", default="/data3/rey/dogs_recaptioned")
    parser.add_argument("--csv", default="results/caption_comparison.csv")
    args = parser.parse_args()

    print("Loading old captions...")
    old_caps = load_captions(Path(args.old_dir))
    print("Loading new captions...")
    new_caps = load_captions(Path(args.new_dir))

    old_stats = compute_stats(old_caps, "original")
    new_stats = compute_stats(new_caps, "recaptioned")

    print_comparison(old_stats, new_stats)
    save_csv(old_stats, new_stats, Path(args.csv))


if __name__ == "__main__":
    main()
