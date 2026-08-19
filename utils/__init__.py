import os
import pickle
import logging
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import random
from utils.parser import *
from utils.masking import *
from utils.logger import set_logger
from utils.my_lr_scheduler import ChainedScheduler
from utils.unfreeze import UnfreezeDecoderCallback
from utils.metric import log_metrics_as_bar_chart, calculate_table_metrics
from utils.test import run_inference, calc_ic, RankIC, Cal_IC_IR
from utils.corr import corr_cluster_order


def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False  
    torch.use_deterministic_algorithms(True)
    
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_root_dir():
    return Path(__file__).parent.parent


def save_model(models_dict: dict, dirname='res', id: str = ''):
    """
    :param models_dict: {'model_name': model, ...}
    """

    if not os.path.isdir(get_root_dir().joinpath(dirname)):
        os.mkdir(get_root_dir().joinpath(dirname))

    id_ = id[:]
    if id != '':
        id_ = '-' + id_
    for model_name, model in models_dict.items():
        torch.save(model.state_dict(), get_root_dir().joinpath(dirname, model_name + id_ + '.ckpt'))