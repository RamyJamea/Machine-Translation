import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import MarianTokenizer
from tqdm import tqdm


def plot_token_distribution():
    dataset = load_dataset(
        "ramyibrahim/cniomt-ar-en-v1", split="train", cache_dir="./.cache"
    )
    tokenizer = MarianTokenizer.from_pretrained(
        "Helsinki-NLP/opus-mt-ar-en", cache_dir="./.cache"
    )

    def get_token_lengths(texts, desc="Processing"):
        lengths = []

        for text in tqdm(texts, desc=desc):
            tokens = tokenizer(text, truncation=False, add_special_tokens=True)[
                "input_ids"
            ]

            lengths.append(len(tokens))

        return lengths

    print("Calculating token lengths...")

    ar_lengths = get_token_lengths(dataset["ar"], desc="Train Arabic")
    en_lengths = get_token_lengths(dataset["en"], desc="Train English")

    def print_stats(name, lengths):
        print(f"\n{name}")
        print("-" * 50)
        print(f"Samples : {len(lengths)}")
        print(f"Mean    : {np.mean(lengths):.2f}")
        print(f"Median  : {np.median(lengths):.2f}")
        print(f"Min     : {np.min(lengths)}")
        print(f"Max     : {np.max(lengths)}")
        print(f"95%tile : {np.percentile(lengths, 95):.2f}")

    print_stats("Arabic", ar_lengths)
    print_stats("English", en_lengths)

    bins = 100
    plt.figure(figsize=(12, 6))
    plt.hist(ar_lengths, bins=bins, alpha=0.6, label="Arabic")
    plt.hist(en_lengths, bins=bins, alpha=0.6, label="English")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.title("Token Length Distribution")
    plt.legend()

    # output_path = "./token_distribution.png"
    # plt.savefig(output_path, dpi=300, bbox_inches="tight")
    # print(f"\nSaved plot to: {output_path}")
    plt.show()


if __name__ == "__main__":
    plot_token_distribution()
