import torch
import evaluate
from tqdm import tqdm


def evaluate_on_test_set(trainer, test_split, tokenizer, max_new_tokens=256):
    chrf_metric = evaluate.load("chrf")
    trainer.model.eval()
    predictions = []
    references = []
    print("\n> Starting generation and evaluation on the test set...")
    for example in tqdm(test_split, desc="Evaluating"):
        messages = [
            {
                "role": "system",
                "content": "You are a professional Arabic to English translator.",
            },
            {"role": "user", "content": example["ar"]},
        ]

        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = tokenizer(prompt_text, return_tensors="pt").to(trainer.model.device)

        with torch.no_grad():
            generated_ids = trainer.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = generated_ids[0][input_length:]
        pred_text = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
        predictions.append(pred_text)
        references.append([example["en"].strip()])

    results = chrf_metric.compute(predictions=predictions, references=references)
    print(f"\n> Final Test chrF Score: {results['score']:.4f}")

    return results
