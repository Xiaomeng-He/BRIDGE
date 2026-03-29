import yaml
import numpy as np
import random
import torch

def read_config(config_path):
    with open(config_path, "r", encoding="utf-8") as stream:
        return yaml.safe_load(stream)

def set_seed(seed: int):
    random.seed(seed)
    
    np.random.seed(seed)
    
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False