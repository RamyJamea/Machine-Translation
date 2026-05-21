from abc import ABC, abstractmethod


class BaseMT(ABC):
    def __init__(self, model_config, cache_dir: str = "./.cache"):
        self.model_config = model_config
        self.cache_dir = cache_dir

        self.model = self._init_model()
        self.tokenizer = self._init_tokenizer()

        self._suit_model_tokenizer()

    @abstractmethod
    def _init_model(self): ...

    @abstractmethod
    def _init_tokenizer(self): ...

    def _suit_model_tokenizer(self):
        if self.tokenizer == None:
            raise RuntimeError("Tokenizer is None")
        if self.model == None:
            raise RuntimeError("Model is None")
        if self.tokenizer.bos_token_id is None:
            self.tokenizer.bos_token = self.tokenizer.eos_token

        self.model.config.bos_token_id = self.tokenizer.bos_token_id
        self.model.config.eos_token_id = self.tokenizer.eos_token_id
        self.model.config.pad_token_id = self.tokenizer.pad_token_id
        self.model.generation_config.bos_token_id = self.tokenizer.bos_token_id
        self.model.generation_config.eos_token_id = self.tokenizer.eos_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id

    def num_parameters(self):
        return self.model.num_parameters()
