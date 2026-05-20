import os
import yaml
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import MarianTokenizer
from tqdm import tqdm


def plot_token_distribution():
    config_path = r"C:\Users\ramyu\OneDrive\Desktop\MachineT\lab\config.yml"

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]

    tokenizer = MarianTokenizer.from_pretrained(
        cfg["model"]["tokenizer_name"], cache_dir="./tokenizers/opus-mt-ar-en"
    )

    dataset = load_dataset(data_cfg["data_name"], split="train", cache_dir="./data")
    dataset = dataset.train_test_split(test_size=data_cfg["test_ratio"], seed=123)
    train_data = dataset["train"]
    test_data = dataset["test"]

    def get_token_lengths(texts, desc="Processing"):
        lengths = []

        for text in tqdm(texts, desc=desc):
            tokens = tokenizer(text, truncation=False, add_special_tokens=True)[
                "input_ids"
            ]

            lengths.append(len(tokens))

        return lengths

    print("Calculating token lengths...")

    train_ar_lengths = get_token_lengths(train_data["ar"], desc="Train Arabic")
    train_en_lengths = get_token_lengths(train_data["en"], desc="Train English")
    test_ar_lengths = get_token_lengths(test_data["ar"], desc="Test Arabic")
    test_en_lengths = get_token_lengths(test_data["en"], desc="Test English")

    def print_stats(name, lengths):
        print(f"\n{name}")
        print("-" * 50)
        print(f"Samples : {len(lengths)}")
        print(f"Mean    : {np.mean(lengths):.2f}")
        print(f"Median  : {np.median(lengths):.2f}")
        print(f"Min     : {np.min(lengths)}")
        print(f"Max     : {np.max(lengths)}")
        print(f"95%tile : {np.percentile(lengths, 95):.2f}")

    print_stats("Train Arabic", train_ar_lengths)
    print_stats("Train English", train_en_lengths)
    print_stats("Test Arabic", test_ar_lengths)
    print_stats("Test English", test_en_lengths)

    os.makedirs("./results", exist_ok=True)

    bins = 100
    plt.figure(figsize=(12, 6))
    plt.hist(train_ar_lengths, bins=bins, alpha=0.6, label="Train Arabic")
    plt.hist(train_en_lengths, bins=bins, alpha=0.6, label="Train English")
    plt.xlabel("Number of Tokens")
    plt.ylabel("Frequency")
    plt.title("Token Length Distribution")
    plt.legend()

    output_path = "./results/token_distribution.png"
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved plot to: {output_path}")
    plt.show()


if __name__ == "__main__":
    plot_token_distribution()
