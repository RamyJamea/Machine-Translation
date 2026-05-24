from typing import Literal
from .interface import BaseMT
from .providers import OpusMT, LFM2_5MT


class FactoryMT:
    def __init__(self, model_config):
        self.model_config = model_config

    def create(self, provider: Literal["opus", "lmf2.5"]) -> BaseMT:
        if provider == "opus":
            return OpusMT(model_config=self.model_config)
        elif provider == "lfm2.5":
            return LFM2_5MT(model_config=self.model_config)
