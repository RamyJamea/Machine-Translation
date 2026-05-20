import os
import json
import yaml
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm


def eval():
    config_path = r"C:\Users\ramyu\OneDrive\Desktop\MachineT\lab\config.yml"
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    data_cfg = cfg["data"]

    model_path = r"C:\Users\ramyu\OneDrive\Desktop\MachineT\lab\outputs\holol-ar-en-gamma-synth-v1\checkpoint-9795"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = MarianTokenizer.from_pretrained(
        cfg["model"]["tokenizer_name"], cache_dir="./tokenizers/opus-mt-ar-en"
    )
    model = MarianMTModel.from_pretrained(model_path).to(device)

    model.eval()

    # dataset = load_dataset(data_cfg["data_name"], split="train", cache_dir="./data")
    # dataset = dataset.train_test_split(test_size=data_cfg["test_ratio"], seed=123)

    ####################################
    dataset_a = load_dataset(data_cfg["data_name"], split="train", cache_dir="./data")
    dataset_b = load_dataset(
        "ramyibrahim/holol-ar-en-synth-v1", split="train", cache_dir="./data"
    )
    merged_dataset = concatenate_datasets([dataset_a, dataset_b])
    merged_dataset = merged_dataset.shuffle(seed=123)
    dataset = merged_dataset.train_test_split(test_size=data_cfg["test_ratio"], seed=123)
    ####################################


    eval_data = dataset["test"]

    results = []
    batch_size = 16

    print(f"Starting inference on {len(eval_data)} examples...")

    for i in tqdm(range(0, len(eval_data), batch_size), desc="Generating Translations"):
        batch = eval_data[i : i + batch_size]
        source_texts = batch["ar"]
        target_texts = batch["en"]
        inputs = tokenizer(
            source_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg["model"]["max_length"],
        ).to(device)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_length=cfg["model"]["max_length"],
                num_beams=4,
                early_stopping=True,
            )

        generated_texts = tokenizer.batch_decode(
            generated_ids, skip_special_tokens=True
        )

        for src, tgt, gen in zip(source_texts, target_texts, generated_texts):
            results.append({"source_ar": src, "reference_en": tgt, "generated_en": gen})

    output_file = f"./outputs/results/{cfg['train']['run_name']}.json"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)

    print(f"Successfully saved {len(results)} results to {output_file}")


if __name__ == "__main__":
    eval()

