from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

try:
    import numpy as _np
except ImportError:
    _np = None

try:
    import torch as _torch
except ImportError:
    _torch = None

from omegaconf import DictConfig, ListConfig, OmegaConf


def _as_base_types(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if _np is not None and isinstance(obj, _np.ndarray):
        return obj.tolist()
    if _torch is not None and isinstance(obj, _torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, (DictConfig, ListConfig)):
        return _as_base_types(OmegaConf.to_container(obj, resolve=True, enum_to_str=True))
    if isinstance(obj, dict):
        return {str(k): _as_base_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_as_base_types(v) for v in obj]
    return str(obj)


def _flatten(prefix: str, value: Any, out: Dict[str, Any]) -> None:
    if isinstance(value, dict):
        if not value:
            out[prefix] = {}
            return
        for key, val in value.items():
            key = str(key)
            new_prefix = f"{prefix}.{key}" if prefix else key
            _flatten(new_prefix, val, out)
        return

    if isinstance(value, list):
        # wandb config values must be JSON-serializable; store lists as JSON strings.
        out[prefix] = json.dumps(value)
        return

    if isinstance(value, (str, int, float, bool)) or value is None:
        out[prefix] = value
        return

    out[prefix] = str(value)


def make_wandb_config(cfg: DictConfig) -> Dict[str, Any]:
    """Flatten a Hydra DictConfig into a wandb-friendly dict."""
    container = OmegaConf.to_container(cfg, resolve=True, enum_to_str=True)
    if not isinstance(container, dict):
        return {}

    # Hydra metadata is bulky; keep only the bits we want surfaced.
    hydra_meta = container.pop("hydra", {})
    if isinstance(hydra_meta, dict):
        job = hydra_meta.get("job", {})
        if isinstance(job, dict):
            container["_hydra_job_name"] = job.get("name")
            container["_hydra_overrides"] = job.get("override_dirname")

    base = _as_base_types(container)
    if not isinstance(base, dict):
        return {}

    flat: Dict[str, Any] = {}
    for key, val in base.items():
        _flatten(str(key), val, flat)

    try:
        json.dumps(flat)
        return flat
    except (TypeError, ValueError):
        return {}