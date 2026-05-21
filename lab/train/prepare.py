from huggingface_hub import login
from datasets import load_dataset, concatenate_datasets, Dataset
from ..config import get_settings

SETTINGS = get_settings()


class DataPreprocess:
    def __init__(self, data_config, cache_dir: str = "./.cache"):
        login(SETTINGS.HUGGINGFACE_TOKEN)
        self.data_config = data_config
        self.cache_dir = cache_dir

    def init_dataset(self):
        datasets = self.data_config["datasets"]
        merged = load_dataset(datasets[0], split="train", cache_dir=self.cache_dir)

        for i in range(1, len(datasets)):
            tmp = load_dataset(datasets[i], split="train", cache_dir=self.cache_dir)
            merged = concatenate_datasets([merged, tmp])

        df = merged.to_pandas()
        df_unique = df.drop_duplicates(subset=["ar", "en"])

        merged = Dataset.from_pandas(df_unique, preserve_index=False)
        shuffled = merged.shuffle(seed=123)
        return shuffled

    def split_dataset(self, dataset):
        splits = dataset.train_test_split(
            test_size=self.data_config["split_ratio"],
            seed=123,
        )
        print(f"> Train Dataset: {len(splits['train'])}")
        print(f"> Test  Dataset: {len(splits['test'])}")
        return splits

    def tokenize_split(self, tokenizer, split, max_length: int):
        def __tokenize(examples):
            model_inputs = tokenizer(
                examples["ar"],
                max_length=max_length,
                padding=False,
                truncation=False,
            )

            labels = tokenizer(
                examples["en"],
                max_length=max_length,
                padding=False,
                truncation=False,
            )

            model_inputs["labels"] = labels["input_ids"]
            return model_inputs

        tokenized = split.map(
            __tokenize,
            batched=True,
            remove_columns=split.column_names,
        )

        filtered = tokenized.filter(
            lambda example: len(example["input_ids"]) <= max_length
            and len(example["labels"]) <= max_length
        )
        return filtered
