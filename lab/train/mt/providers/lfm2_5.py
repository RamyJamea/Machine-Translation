import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from ..interface import BaseMT


class LFM2_5MT(BaseMT):
    def _init_model(self):
        model = AutoModelForCausalLM.from_pretrained(
            self.model_config["model"],
        #     quantization_config=BitsAndBytesConfig(
        #     load_in_4bit=True,
        #     bnb_4bit_compute_dtype=torch.bfloat16,
        #     bnb_4bit_quant_type="nf4",
        #     bnb_4bit_use_double_quant=True,
        # ),
            device_map="auto",
            cache_dir=self.cache_dir,
        )
        return model

    def _init_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_config["tokenizer"],
            cache_dir=self.cache_dir,
        )
        tokenizer.padding_side = "right"
        return tokenizer
