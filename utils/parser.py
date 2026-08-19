# yaml
from argparse import ArgumentParser
import os
from pathlib import Path
import yaml

def get_root_dir():
    return Path(__file__).parent.parent

def load_args():
    parser = ArgumentParser()
    parser.add_argument('--config', type=str, help="Path to the config data  file.",
                        default=get_root_dir().joinpath('configs', 'config.yaml'))
    parser.add_argument('--data_handler_config', type=str, help="Path to the data handler config file.",
                        default=get_root_dir().joinpath('configs', 'data_handler_config.yaml'))
    parser.add_argument('--seed', type=int, help="Seed for reproducibility.",
                        default=3)
    return parser.parse_args()


def load_yaml_param_settings(yaml_fname: str, seed: int=None):
    """
    :param yaml_fname: .yaml file that consists of hyper-parameter settings.
    """
    stream = open(yaml_fname, 'r')
    config = yaml.load(stream, Loader=yaml.FullLoader)
    if seed is not None:
        config['train']['seed'] = seed
    return config