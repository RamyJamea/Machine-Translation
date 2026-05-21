import yaml
from huggingface_hub import login
from ..config import get_settings
from .mt import FactoryMT

CONFIG_PATH = r"C:\Users\ramyu\code\MachineT\lab\train\config_opus_v1.yml"
CACHE_DIR = r"./cache"
SETTINGS = get_settings()
login(SETTINGS.HUGGINGFACE_TOKEN)

with open(CONFIG_PATH) as f:
    config = yaml.safe_load(f)
    model_config = config["model"]

factory = FactoryMT(model_config=model_config)
MT = factory.create(provider=model_config["provider"])
