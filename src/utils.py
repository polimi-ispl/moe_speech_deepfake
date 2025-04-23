import os
import random

import numpy as np
import torch
import torch.nn as nn
import yaml


def init_weights(module):
    if isinstance(module, nn.Linear):
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.01)


def read_yaml(config_path):
    """
    Read YAML file.

    :param config_path: path to the YAML config file.
    :type config_path: str
    :return: dictionary correspondent to YAML content
    :rtype dict
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def seed_everything(seed: int):
    """
    Set seed for everything.
    :param seed: seed value
    :type seed: int
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
