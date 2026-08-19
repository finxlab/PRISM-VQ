import logging
import os
import json
import re

def set_logger(run_name, save_dir):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(formatter)
    if not os.path.exists(f'{save_dir}/log'):
        os.makedirs(f'{save_dir}/log')
    fh = logging.FileHandler(filename=f'{save_dir}/log/{run_name}.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(ch)
    logger.addHandler(fh)
    return logger