from typing import Literal
from .providers import OpusMT
from .interface import BaseMT


class FactoryMT:
    def __init__(self, model_config):
        self.model_config = model_config

    def create(self, provider: Literal["opus"]) -> BaseMT:
        if provider == "opus":
            return OpusMT(model_config=self.model_config)
