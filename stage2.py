import os
import shutil
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

# Set env vars before importing PyTorch.
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

import copy
import hydra
import pandas as pd
import pickle
import pytorch_lightning as pl
import torch
import wandb
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from dataset.dataset import init_data_loader
from trainer.train_ypred import GenerateReturn
from utils import (get_root_dir, log_metrics_as_bar_chart, run_inference,
                   seed_everything)
from utils.wandb import make_wandb_config

torch.set_float32_matmul_precision('high')

# OmegaConf resolver for computing half of n_expert
OmegaConf.register_new_resolver("half", lambda x: int(x) // 2)

_SNAPSHOT_ENV_KEY = "STAGE2_CONFIG_SNAPSHOT_DIR"
_ORIGINAL_CWD: Optional[Path] = None


def _ensure_config_snapshot() -> Path:
    """Copy the configs directory once and reuse it during multi-run sweeps."""
    snapshot_dir = os.environ.get(_SNAPSHOT_ENV_KEY)
    if snapshot_dir and Path(snapshot_dir).exists():
        return Path(snapshot_dir)

    configs_dir = Path(__file__).resolve().parent / 'configs'
    if not configs_dir.exists():
        raise FileNotFoundError(f"Config directory not found: {configs_dir}")

    runtime = HydraConfig.get()
    project_root = Path(runtime.runtime.cwd)
    snapshot_root = project_root / '.hydra_config_snapshots'
    snapshot_root.mkdir(parents=True, exist_ok=True)

    tmp_dir = Path(tempfile.mkdtemp(prefix='stage2_', dir=str(snapshot_root)))
    frozen_configs = tmp_dir / 'configs'
    shutil.copytree(configs_dir, frozen_configs)

    os.environ[_SNAPSHOT_ENV_KEY] = str(frozen_configs)
    return frozen_configs


def _compose_frozen_config(overrides: List[str]) -> DictConfig:
    snapshot_dir = _ensure_config_snapshot()

    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(snapshot_dir), job_name='stage2_frozen'):
        frozen_cfg = compose(config_name='config', overrides=overrides)
    GlobalHydra.instance().clear()

    return frozen_cfg


def _to_absolute_path(path: str) -> str:
    if os.path.isabs(path):
        return path
    if _ORIGINAL_CWD is None:
        raise RuntimeError('Original working directory is not set.')
    return str((_ORIGINAL_CWD / path).resolve())



def _build_run_name(cfg: DictConfig) -> str:
    n_expert = cfg.predictor.n_expert
    k = cfg.predictor.k
    dim = cfg.predictor.transformer.d_model
    n_heads = cfg.predictor.transformer.num_heads
    n_layer = cfg.predictor.transformer.num_layers
    dropout = cfg.predictor.transformer.dropout
    aux_weight = cfg.predictor.aux_weight
    kernel_size = cfg.predictor.kernel_size
    moe_hidden = cfg.predictor.moe_hidden
    moe_drop = cfg.predictor.dropout
    horizon = cfg.predictor.pred_len
    aux_imp = cfg.predictor.aux_imp

    saved_model = Path(cfg.predictor.saved_model).name
    model_name_part = saved_model.split('-')[0]

    tokens = model_name_part.split('_')
    param_dict = {}
    for token in tokens:
        if token.startswith('VQ'):
            param_dict['num_embed'] = token[2:]
        elif token.startswith('n'):
            param_dict['enc_heads'] = token[1:]
        elif token.startswith('e'):
            param_dict['vq_embed_dim'] = token[1:]
        elif token.startswith('d'):
            param_dict['dropout_pred'] = token[1:]
        elif token in ['l2', 'cos']:
            param_dict['distance'] = token

    num_embed = param_dict.get('num_embed', cfg.vqvae.num_embed)
    enc_heads = param_dict.get('enc_heads', cfg.vqvae.encoder.num_heads)
    vq_embed_dim = param_dict.get('vq_embed_dim', cfg.vqvae.vq_embed_dim)
    dropout_pred = param_dict.get('dropout_pred', cfg.vqvae.predictor.dropout)
    distance = param_dict.get('distance', cfg.vqvae.quantizer.distance)

    return (
        f"{cfg.train.seed}_VQ{num_embed}_{cfg.data.universe}_mo{n_expert}_"
        f"k{k}_mh{moe_hidden}_md{moe_drop}_dm{dim}_nh{n_heads}_l{n_layer}_"
        f"d{dropout}_au{aux_weight}_1h{enc_heads}_1e{vq_embed_dim}_"
        f"1d{dropout_pred}_1{distance}_p{horizon}_ai{aux_imp}_ks{kernel_size}"
    )

def _build_wandb_logger(cfg: DictConfig, run_name: str) -> WandbLogger:
    project_name = f"{cfg.train.project_name}_{cfg.data.universe}_stage2"

    logger_cfg = {
        "_target_": "pytorch_lightning.loggers.wandb.WandbLogger",
        "project": project_name,
        "name": run_name,
        "group": cfg.data.universe,
        "entity": cfg.train.get("wandb_entity"),
        "save_dir": str(Path(get_root_dir()) / cfg.train.save_dir),
        "log_model": cfg.train.get("wandb_log_model", False),
        "reinit": True,
    }
    logger_cfg = {k: v for k, v in logger_cfg.items() if v is not None}
    logger = instantiate(logger_cfg)

    try:
        clean_config = make_wandb_config(cfg)
        if hasattr(logger, 'experiment') and clean_config:
            logger.experiment.config.update(clean_config)
    except Exception:
        pass

    return logger


def _build_callbacks(cfg: DictConfig, run_name: str) -> Tuple[List[Callback], 'ModelCheckpoint', Optional['ModelCheckpoint']]:

    checkpoint_dir = Path(get_root_dir()) / cfg.train.save_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    early_cfg = cfg.train.early_stopping
    lr_monitor_cfg = {
        "_target_": "pytorch_lightning.callbacks.LearningRateMonitor",
        "logging_interval": "step",
    }
    checkpoint_cfg = {
        "_target_": "pytorch_lightning.callbacks.ModelCheckpoint",
        "save_top_k": 1,
        "monitor": early_cfg.monitor,
        "mode": early_cfg.mode,
        "dirpath": str(checkpoint_dir),
        "filename": f"{run_name}" + "-{epoch}-{val_loss:.4f}",
    }
    # ric_checkpoint_cfg = {
    #     "_target_": "pytorch_lightning.callbacks.ModelCheckpoint",
    #     "save_top_k": 1,
    #     "monitor": "Val_RIC",
    #     "mode": "max",
    #     "dirpath": str(checkpoint_dir),
    #     "filename": f"{run_name}" + "-{epoch}-{Val_RIC:.4f}",
    # }
    early_stop_cfg = {
        "_target_": "pytorch_lightning.callbacks.EarlyStopping",
        "monitor": early_cfg.monitor,
        "min_delta": early_cfg.min_delta,
        "patience": early_cfg.patience,
        "verbose": early_cfg.verbose,
        "mode": early_cfg.mode,
    }

    lr_monitor = instantiate(lr_monitor_cfg)
    checkpoint_callback: ModelCheckpoint = instantiate(checkpoint_cfg)
    #ric_callback: ModelCheckpoint = instantiate(ric_checkpoint_cfg)
    early_stop = instantiate(early_stop_cfg)

    callbacks: List[Callback] = [lr_monitor, checkpoint_callback, early_stop]
    # if cfg.train.get('enable_ric_checkpoint', True):
    #     callbacks.append(ric_callback)
    # else:
    #     ric_callback = None

    return callbacks, checkpoint_callback, #ric_callback


def _resolve_data_path(path: Optional[str]) -> Path:
    if not path:
        raise FileNotFoundError("`data.data_path` must be provided for stage2 training.")
    abs_path = Path(_to_absolute_path(path))
    if not abs_path.exists():
        raise FileNotFoundError(f"Data path '{abs_path}' does not exist.")
    return abs_path


def _resolve_universe(universe: str) -> Tuple[str, str]:
    mapping = {
        'csi300': ('CN', 'csi300'),
        'csi500': ('CN', 'csi500'),
        'sp500': ('US', 'sp500'),
        'nasdaq': ('US', 'nasdaq'),
    }
    if universe not in mapping:
        raise ValueError(f"Invalid universe: {universe}")
    return mapping[universe]


def _prepare_dataloaders(cfg: DictConfig,
                         region_code: str,
                         universe_prefix: str):
    data_path = _resolve_data_path(cfg.data.get('data_path'))
    horizon = cfg.vqvae.predictor.pred_len
    window_size = cfg.data.window_size
    base = f"{universe_prefix}_{window_size}_h{horizon}_dl2"

    file_map = {
        'train': data_path / region_code / f"{base}_train.pkl",
        'valid': data_path / region_code / f"{base}_valid.pkl",
        'test': data_path / region_code / f"{base}_test.pkl",
    }

    missing = [str(path) for path in file_map.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required pickle files: " + ", ".join(missing) +
            ". Please ensure data is pre-processed."
        )

    train_prepare = pickle.load(open(file_map['train'], 'rb'))
    valid_prepare = pickle.load(open(file_map['valid'], 'rb'))
    test_prepare = pickle.load(open(file_map['test'], 'rb'))

    num_workers = cfg.train.num_workers
    train_loader, num_batches_per_epoch_train = init_data_loader(train_prepare, shuffle=True, num_workers=num_workers)
    valid_loader, _ = init_data_loader(valid_prepare, shuffle=False, num_workers=num_workers)
    test_loader, _ = init_data_loader(test_prepare, shuffle=False, num_workers=num_workers)

    return train_loader, valid_loader, test_loader, num_batches_per_epoch_train


def _make_absolute_saved_model_path(config_dict: dict) -> dict:
    model_config = copy.deepcopy(config_dict)
    saved_model = model_config['predictor']['saved_model']
    saved_model_path = Path(saved_model)
    if not saved_model_path.is_absolute():
        saved_model_path = Path(get_root_dir()) / 'checkpoints' / saved_model_path
    model_config['predictor']['saved_model'] = str(saved_model_path)
    return model_config


def train(cfg: DictConfig,
          config_dict: dict,
          train_loader,
          valid_loader,
          num_batches_per_epoch_train: int,
          test_loader):
    run_name = _build_run_name(cfg)
    model_config = _make_absolute_saved_model_path(config_dict)

    T_max = num_batches_per_epoch_train * cfg.train.num_epochs
    model = GenerateReturn(model_config, T_max=T_max)
    wandb_logger = _build_wandb_logger(cfg, run_name)
    wandb_logger.watch(model, log='all')
    callbacks, checkpoint_callback = _build_callbacks(cfg, run_name)

    trainer = pl.Trainer(
        logger=wandb_logger,
        enable_checkpointing=True,
        callbacks=callbacks,
        max_epochs=cfg.train.num_epochs,
        accelerator=cfg.train.get('accelerator', 'gpu'),
        devices=cfg.train.get('gpu_counts', 1),
        precision=cfg.train.precision,
        gradient_clip_val=cfg.train.get('gradient_clip_val', 2),
        deterministic=True,
        log_every_n_steps=cfg.train.get('log_every_n_steps', 50),
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=valid_loader)

    best_path = checkpoint_callback.best_model_path
    if not best_path:
        raise RuntimeError("No checkpoint was saved during training.")
    print(f"========== Best checkpoint path: {best_path} ==========")
    print(f"========== Best validation loss: {checkpoint_callback.best_model_score} ==========")

    best_model = GenerateReturn.load_from_checkpoint(best_path, config=model_config, T_max=T_max)
    best_model.freeze_vqvae()
    best_model.eval()

    with torch.no_grad():
        print("========== Validation evaluation ==========")
        _, _, val_metric = run_inference(best_model, valid_loader, model_config)
        print(f"Validation RIC: {val_metric['RankIC']:.4f}")

        print("========== Test evaluation ==========")
        pred_df, _, metric = run_inference(best_model, test_loader, model_config)
        print(f"Test RIC: {metric['RankIC']:.4f}")

    save_run_name = '_'.join(run_name.split('_')[1:])
    run_dir = Path(get_root_dir()) / cfg.train.save_res / save_run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    seed = cfg.train.seed
    output_path = run_dir / f"{seed}_best.pkl"
    pred_df.to_pickle(output_path)

    output_csv_path = run_dir / f"{seed}_metric.csv"
    pd.DataFrame([metric], index=['values']).transpose().to_csv(output_csv_path)
    print(f"Results saved to {output_path} and {output_csv_path}")

    log_metrics_as_bar_chart(metric, model_name=run_name)
    wandb.log({'metrics_best': metric})

    experiment = getattr(wandb_logger, 'experiment', None)
    if experiment is not None:
        experiment.finish()

    return metric


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig) -> float:
    hydra_cfg = HydraConfig.get()
    overrides = list(hydra_cfg.overrides.task or [])

    global _ORIGINAL_CWD
    if _ORIGINAL_CWD is None:
        _ORIGINAL_CWD = Path(hydra_cfg.runtime.cwd)

    if hydra_cfg.mode == RunMode.MULTIRUN:
        base_cfg = _compose_frozen_config(overrides)
    else:
        base_cfg = cfg

    frozen_cfg = OmegaConf.create(OmegaConf.to_container(base_cfg, resolve=True))
    OmegaConf.set_readonly(frozen_cfg, True)

    config_dict = OmegaConf.to_container(frozen_cfg, resolve=True)

    seed_everything(frozen_cfg.train.seed)
    pl.seed_everything(frozen_cfg.train.seed, workers=True)

    region_code, universe_prefix = _resolve_universe(frozen_cfg.data.universe)

    print(f"Seed value: {frozen_cfg.train.seed}")
    print(f"********** Region: {region_code} **********")
    print(f"********** Universe: {frozen_cfg.data.universe} **********")

    train_loader, valid_loader, test_loader, num_batches_per_epoch_train = _prepare_dataloaders(
        frozen_cfg, region_code, universe_prefix
    )

    ric_score = train(frozen_cfg, config_dict, train_loader, valid_loader, num_batches_per_epoch_train, test_loader)
    return ric_score


if __name__ == "__main__":
    score = main()
    print(score)
