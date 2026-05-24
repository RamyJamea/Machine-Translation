import yaml, wandb
from huggingface_hub import login
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer, EarlyStoppingCallback
from ..config import get_settings
from .model import MT
from .prepare import DataPreprocess
from .metrics import compute_metrics

CONFIG_PATH = r"C:\Users\ramyu\code\MachineT\lab\train\config_opus_v1.yml"
SETTINGS = get_settings()
wandb.login(SETTINGS.WANDB_API_KEY)
login(SETTINGS.HUGGINGFACE_TOKEN)

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)
    model_config = config["model"]
    data_config = config["data"]
    train_config = config["train"]


preprocessor = DataPreprocess(data_config)
dataset = preprocessor.init_dataset()
splits = preprocessor.split_dataset(dataset)
train_dataset = preprocessor.tokenize_split(
    MT.tokenizer,
    splits["train"],
    model_config["max_length"],
)
test_dataset = preprocessor.tokenize_split(
    MT.tokenizer,
    splits["test"],
    model_config["max_length"],
)

print(f"> Num Parameters: {MT.num_parameters()}")

collator = DataCollatorForSeq2Seq(
    tokenizer=MT.tokenizer,
    model=MT.model,
    padding="longest",
    label_pad_token_id=-100,
)

training_steps = (
    len(train_dataset)
    * train_config["num_train_epochs"]
    // (train_config["per_device_train_batch_size"])
)

training_args = Seq2SeqTrainingArguments(
    output_dir=f".outputs/{train_config['run_name']}",
    num_train_epochs=train_config["num_train_epochs"],
    per_device_train_batch_size=train_config["per_device_train_batch_size"],
    per_device_eval_batch_size=train_config["per_device_eval_batch_size"],
    logging_strategy=train_config["logging_strategy"],
    eval_strategy=train_config["eval_strategy"],
    save_strategy=train_config["save_strategy"],
    save_total_limit=train_config["save_total_limit"],
    load_best_model_at_end=train_config["load_best_model_at_end"],
    metric_for_best_model=train_config["metric_for_best_model"],
    greater_is_better=train_config["greater_is_better"],
    bf16=train_config["bf16"],
    optim=train_config["optim"],
    gradient_checkpointing=train_config["gradient_checkpointing"],
    learning_rate=float(train_config["learning_rate"]),
    max_grad_norm=train_config["max_grad_norm"],
    warmup_steps=training_steps * train_config["warmup_ratio"],
    lr_scheduler_type=train_config["lr_scheduler_type"],
    predict_with_generate=train_config["predict_with_generate"],
    run_name=train_config["run_name"],
    report_to=train_config["report_to"],
)

trainer = Seq2SeqTrainer(
    model=MT.model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    processing_class=MT.tokenizer,
    data_collator=collator,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=train_config["early_stopping"],
            early_stopping_threshold=0.0,
        )
    ],
)

trainer.train()
