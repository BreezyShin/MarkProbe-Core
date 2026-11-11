"""Reusable helpers for weight-only MarkProbe fine-tuning."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Optional

import torch

from .engine import train_model


DEFAULT_FINETUNE_SETTINGS = {
    "epochs": 10,
    "batch_size": 4,
    "lr": 1e-5,
    "val_fraction": 0.1,
    "seed": 2027,
    "no_im_loss_steps": 0,
    "image_weight": 2.0,
    "l2_loss_ramp_steps": 5000,
    "lpips_loss_ramp_steps": 5000,
    "geometry_ramp_steps": 5000,
    "jpeg_ramp_steps": 1000,
    "lpips_weight": 1.5,
    "yuv_scales": (1.0, 100.0, 100.0),
    "edge_gain": 10.0,
    "edge_delay_steps": 0,
    "edge_ramp_steps": 5000,
    "secret_loss_scale": 1.5,
    "secret_loss_ramp_steps": 1,
    "gan_weight": 0.5,
    "gan_ramp_steps": 5000,
    "use_gan": True,
    "use_jpeg": True,
    "border": "white",
    "dual_branch": True,
    "crop_scale_min": 0.80,
    "crop_scale_max": 0.95,
    "adaptive_image_loss": False,
    "adaptive_msg_threshold": 0.30,
    "adaptive_ber_threshold": 0.25,
    "adaptive_patience": 2,
    "adaptive_ramp_steps": 5000,
    "validation_samples": 256,
}

_ARCHITECTURE_KEYS = {"image_size", "message_bits", "width", "use_stn", "residual_scale"}


def merge_finetune_settings(settings: Optional[Mapping] = None, **overrides) -> dict:
    """Merge defaults with user settings and reject architecture changes."""
    merged = dict(DEFAULT_FINETUNE_SETTINGS)
    merged.update(settings or {})
    merged.update({key: value for key, value in overrides.items() if value is not None})
    changed_architecture = _ARCHITECTURE_KEYS.intersection(merged)
    if changed_architecture:
        names = ", ".join(sorted(changed_architecture))
        raise ValueError(
            "Fine-tuning keeps the checkpoint architecture fixed; architecture fields "
            "cannot be passed as training settings: {}".format(names)
        )
    return merged


def _checkpoint_architecture(checkpoint: Path) -> dict:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(state, dict):
        raise ValueError("Fine-tuning checkpoint must be a MarkProbe checkpoint dictionary")
    if state.get("format") != "markprobe-core-v6-robust-training":
        raise ValueError("Fine-tuning requires a v6 robust-training checkpoint")
    required = ("image_size", "message_bits", "width", "use_stn", "residual_scale")
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError("Checkpoint is missing architecture metadata: {}".format(", ".join(missing)))
    return {key: state[key] for key in required}


def run_finetune(data_dir, checkpoint, output, *, settings=None, device=None, limit=None):
    """Fine-tune from model weights while resetting all training state.

    ``settings`` accepts keyword arguments supported by :func:`train_model`, making
    new loss or curriculum parameters available to the script and notebook without
    duplicating the training loop.
    """
    source = Path(checkpoint).expanduser().resolve()
    destination = Path(output).expanduser().resolve()
    latest = destination.with_name(destination.stem + ".last" + destination.suffix)
    if not source.is_file():
        raise FileNotFoundError("Fine-tune checkpoint not found: {}".format(source))
    if source in {destination, latest}:
        raise ValueError("Fine-tune output or latest-checkpoint path would overwrite its source")

    architecture = _checkpoint_architecture(source)
    train_settings = merge_finetune_settings(settings)
    return train_model(
        data_dir,
        destination,
        finetune_from=source,
        device=device,
        limit=limit,
        **architecture,
        **train_settings
    )
