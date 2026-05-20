import torch
from transformers import MarianMTModel, MarianTokenizer

def translate_sentence(text, model, tokenizer, device, max_length=128):
    inputs = tokenizer(
        [text], 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=max_length
    ).to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_length=max_length,
            num_beams=4,
            early_stopping=True
        )

    translated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    
    return translated_text


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = r"./lab/results/opus-mt-ar-en/final"
    tokenizer_name = "Helsinki-NLP/opus-mt-ar-en"

    print("Loading model...")
    tokenizer = MarianTokenizer.from_pretrained(tokenizer_name)
    model = MarianMTModel.from_pretrained(model_path).to(device)
    model.eval()

    user_input = input("Enter an Arabic sentence: ")
    result = translate_sentence(user_input, model, tokenizer, device)
    
    print("-" * 30)
    print(f"English: {result}")