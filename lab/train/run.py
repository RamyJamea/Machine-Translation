import yaml
import torch
import wandb
import evaluate
import numpy as np
from huggingface_hub import login
from datasets import load_dataset, concatenate_datasets
from transformers import MarianMTModel, MarianTokenizer, DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback

wandb.login("")
login("")

with open(r"C:\Users\ramyu\OneDrive\Desktop\MachineT\lab\config.yml") as f:
    cfg = yaml.safe_load(f)

model_cfg = cfg["model"]
data_cfg = cfg["data"]
train_cfg = cfg["train"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

# dataset = load_dataset(data_cfg["data_name"], split="train", cache_dir="./data")
# dataset = dataset.train_test_split(test_size=data_cfg["test_ratio"], seed=123)

###########################################
dataset_a = load_dataset(data_cfg["data_name"], split="train", cache_dir="./data")
dataset_b = load_dataset(
    "ramyibrahim/holol-ar-en-synth-v1", split="train", cache_dir="./data"
)
dataset_c = load_dataset(
    "ramyibrahim/holol-ar-en-synth-v2", split="train", cache_dir="./data"
)
merged_dataset = concatenate_datasets([dataset_a, dataset_b, dataset_c])
merged_dataset = merged_dataset.shuffle(seed=123)
dataset = merged_dataset.train_test_split(test_size=data_cfg["test_ratio"], seed=123)
print("+ Train Dataset: ", len(dataset["train"]))
print("+ Test Dataset: ", len(dataset["test"]))
#########################################


tokenizer = MarianTokenizer.from_pretrained(
    model_cfg["tokenizer_name"], cache_dir="./tokenizers/opus-mt-ar-en"
)
tokenizer.padding_side = "right"


def tokenize_function(examples):
    model_inputs = tokenizer(
        examples["ar"],
        max_length=model_cfg["max_length"],
        padding=False,
        truncation=False,
    )

    labels = tokenizer(
        examples["en"],
        max_length=model_cfg["max_length"],
        padding=False,
        truncation=False,
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs


train_dataset = dataset["train"].map(
    tokenize_function, batched=True, remove_columns=dataset["train"].column_names
)
test_dataset = dataset["test"].map(
    tokenize_function, batched=True, remove_columns=dataset["test"].column_names
)

train_dataset = train_dataset.filter(
    lambda example: len(example["input_ids"]) <= model_cfg["max_length"]
    and len(example["labels"]) <= model_cfg["max_length"]
)
test_dataset = test_dataset.filter(
    lambda example: len(example["input_ids"]) <= model_cfg["max_length"]
    and len(example["labels"]) <= model_cfg["max_length"]
)

model = MarianMTModel.from_pretrained(
    model_cfg["model_name"], cache_dir="./models/opus-mt-ar-en"
)

if tokenizer.bos_token_id is None:
    tokenizer.bos_token = tokenizer.eos_token

model.config.bos_token_id = tokenizer.bos_token_id
model.config.eos_token_id = tokenizer.eos_token_id
model.config.pad_token_id = tokenizer.pad_token_id

model.generation_config.bos_token_id = tokenizer.bos_token_id
model.generation_config.eos_token_id = tokenizer.eos_token_id
model.generation_config.pad_token_id = tokenizer.pad_token_id

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer, model=model, padding="longest", label_pad_token_id=-100
)

metric = evaluate.load("chrf")


def compute_metrics(eval_preds):
    preds, labels = eval_preds

    if isinstance(preds, tuple):
        preds = preds[0]

    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"chrf": result["score"]}


training_args = Seq2SeqTrainingArguments(
    output_dir=f"outputs/{train_cfg['run_name']}",
    num_train_epochs=train_cfg["num_train_epochs"],
    per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
    per_device_eval_batch_size=train_cfg["per_device_eval_batch_size"],
    logging_strategy=train_cfg["logging_strategy"],
    eval_strategy=train_cfg["eval_strategy"],
    save_strategy=train_cfg["save_strategy"],
    save_total_limit=train_cfg["save_total_limit"],
    load_best_model_at_end=train_cfg["load_best_model_at_end"],
    metric_for_best_model=train_cfg["metric_for_best_model"],
    greater_is_better=train_cfg["greater_is_better"],
    bf16=train_cfg["bf16"],
    optim=train_cfg["optim"],
    gradient_checkpointing=train_cfg["gradient_checkpointing"],
    learning_rate=float(train_cfg["learning_rate"]),
    max_grad_norm=train_cfg["max_grad_norm"],
    warmup_steps=(
        len(train_dataset)
        * train_cfg["warmup_ratio"]
        * train_cfg["num_train_epochs"]
        // (train_cfg["per_device_train_batch_size"])
    ),
    lr_scheduler_type=train_cfg["lr_scheduler_type"],
    predict_with_generate=train_cfg["predict_with_generate"],
    run_name=train_cfg["run_name"],
    report_to=train_cfg["report_to"],
    hub_model_id=train_cfg["hub_model_id"] + train_cfg["run_name"],
    push_to_hub=train_cfg["push_to_hub"],
    hub_strategy=train_cfg["hub_strategy"],
    hub_private_repo=train_cfg["hub_private_repo"],
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=train_cfg["early_stopping"],
            early_stopping_threshold=0.0,
        )
    ],
)

trainer.train()
