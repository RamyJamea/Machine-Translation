from transformers import MarianMTModel, MarianTokenizer
from huggingface_hub import login
from .config import get_settings

SETTINGS = get_settings()
login(SETTINGS.HUGGINGFACE_TOKEN)

LOCAL_CHECKPOINT_PATH = r".\.outputs\cniomt-ar-en-70B\checkpoint-9927"
ORIGINAL_MODEL_BASE = "Helsinki-NLP/opus-mt-ar-en"
HF_REPO_ID = "ramyibrahim/cniomt-ar-en-70B"

print("Loading fine-tuned model checkpoint...")
model = MarianMTModel.from_pretrained(LOCAL_CHECKPOINT_PATH)

print("Loading original tokenizer/processor...")
tokenizer = MarianTokenizer.from_pretrained(ORIGINAL_MODEL_BASE, cache_dir="./.cache")

print("Pushing model and tokenizer to {HF_REPO_ID}...")
model.push_to_hub(HF_REPO_ID)
tokenizer.push_to_hub(HF_REPO_ID)

print("Upload complete!")
# https://wandb.ai/ramy-jamea/huggingface/runs/drghu9kh?nw=nwuserramyjamea