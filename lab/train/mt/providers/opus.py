from transformers import MarianMTModel, MarianTokenizer
from ..interface import BaseMT


class OpusMT(BaseMT):
    def _init_model(self):
        model = MarianMTModel.from_pretrained(
            self.model_config["model"],
            cache_dir=self.cache_dir,
        )
        return model

    def _init_tokenizer(self):
        tokenizer = MarianTokenizer.from_pretrained(
            self.model_config["tokenizer"],
            cache_dir=self.cache_dir,
        )
        tokenizer.padding_side = "right"
        return tokenizer
