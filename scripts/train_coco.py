"""Run quickstart notebook dataset preparation and training (9 epochs by default)."""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))
from markprobe_core import train_model  # noqa: E402

ZIP_PATH = Path("/public/data/image/COCO2014/train2014.zip")
DATA_DIR = Path("/workspace/dataset/coco2014")
OUTPUT = PROJECT / "runs/markprobe_core.pt"
METRICS = PROJECT / "runs/markprobe_core.metrics.jsonl"
CURVES = PROJECT / "runs/markprobe_core.curves.json"
EPOCH_EXPORT_DIR = PROJECT / "runs/markprobe_core.epochs"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Full-mode quickstart settings; only the epoch target is set to 9.
IMAGE_SIZE = 400
MESSAGE_BITS = 96
WIDTH = 32
EPOCHS = 9
MAX_STEPS = None
BATCH_SIZE = 4
VAL_FRACTION = 0.1
DUAL_BRANCH = True
CROP_SCALE_MIN = 0.65
CROP_SCALE_MAX = 0.90
EPOCH_EXPORT_SAMPLES = 4
ADAPTIVE_IMAGE_LOSS = False
ADAPTIVE_MSG_THRESHOLD = 0.30
ADAPTIVE_BER_THRESHOLD = 0.25
ADAPTIVE_PATIENCE = 2
ADAPTIVE_RAMP_STEPS = 15000
VALIDATION_SAMPLES = 256
NO_IM_LOSS_STEPS = 1500
LOSS_RAMP_STEPS = 15000
GEOMETRY_RAMP_STEPS = 10000
JPEG_RAMP_STEPS = 1000
EDGE_DELAY_STEPS = 80000
EDGE_RAMP_STEPS = 10000
YUV_SCALES = (1.0, 100.0, 100.0)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zip", type=Path, default=ZIP_PATH, help="COCO train2014.zip path")
    parser.add_argument("--data", type=Path, default=DATA_DIR, help="Extraction and training directory")
    parser.add_argument("--output", type=Path, default=OUTPUT, help="Best checkpoint output path")
    parser.add_argument("--metrics", type=Path, default=METRICS, help="JSONL metrics output path")
    parser.add_argument("--curves", type=Path, default=CURVES,
                        help="Column-oriented JSON curve data output path")
    parser.add_argument("--epoch-export-dir", type=Path, default=EPOCH_EXPORT_DIR,
                        help="Directory for per-epoch cover/encoded/decode artifacts")
    parser.add_argument("--epoch-export-samples", type=int, default=EPOCH_EXPORT_SAMPLES,
                        help="Validation samples exported after each epoch; 0 disables export")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Total target epochs (default: 9)")
    max_steps = parser.add_mutually_exclusive_group()
    max_steps.add_argument(
        "--max-steps", type=int, default=MAX_STEPS,
        help="Optional optimizer-step ceiling; omitted by default so all epochs run",
    )
    max_steps.add_argument(
        "--no-max-steps", action="store_true",
        help="Disable the optimizer-step ceiling and complete all requested epochs",
    )
    parser.add_argument("--device", default=None, help="cuda:0, cpu; defaults to CUDA when available")
    parser.add_argument("--limit", type=int, default=None, help="Optional image cap for smoke tests")
    parser.add_argument("--crop-scale-min", type=float, default=None,
                        help="Minimum retained side fraction for the crop branch")
    parser.add_argument("--crop-scale-max", type=float, default=None,
                        help="Maximum retained side fraction for the crop branch")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--resume", type=Path, default=None,
                                  help="Resume an exact training state; settings must match")
    checkpoint_group.add_argument("--finetune-from", type=Path, default=None,
                                  help="Load model weights only, reset optimizers/ramp, and allow changed training settings")
    parser.add_argument("--no-resume", action="store_true", help="Start fresh even if .last.pt exists")
    return parser.parse_args()


def prepare_data(zip_path: Path, data_dir: Path):
    marker = data_dir / ".train2014_extracted"
    data_dir.mkdir(parents=True, exist_ok=True)
    print(f"Dataset archive: {zip_path}")
    print(f"Dataset extraction/training directory: {data_dir.resolve()}")
    if not marker.is_file():
        if not zip_path.is_file():
            raise FileNotFoundError(f"COCO archive not found: {zip_path}; pass --zip")
        print("Stage 1/2: extracting train2014.zip")
        with zipfile.ZipFile(zip_path) as archive:
            files = [entry for entry in archive.infolist() if not entry.is_dir()]
            with tqdm(files, desc="Extract train2014.zip", unit="file", mininterval=1.0) as bar:
                for entry in bar:
                    archive.extract(entry, data_dir)
        marker.touch()
        print(f"Extraction complete: {len(files):,} files")
    else:
        print(f"Extraction marker found; reusing {data_dir.resolve()}")

    images = sorted(p for p in data_dir.rglob("*")
                    if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise RuntimeError(f"No supported images found under {data_dir.resolve()}")
    val_count = max(1, int(len(images) * VAL_FRACTION)) if len(images) > 1 else 0
    print("Dataset loaded successfully")
    print(f"  Resolved path: {data_dir.resolve()}")
    print(f"  Total images: {len(images):,}")
    print(f"  Training images: {len(images) - val_count:,} | Validation images: {val_count:,}")
    return images


def expected_train_config(crop_scale_min=CROP_SCALE_MIN, crop_scale_max=CROP_SCALE_MAX):
    return {
        "batch_size": BATCH_SIZE, "lr": 1e-4, "seed": 2026,
        "val_fraction": VAL_FRACTION, "no_im_loss_steps": NO_IM_LOSS_STEPS,
        "l2_loss_scale": 2.0, "l2_loss_ramp_steps": LOSS_RAMP_STEPS,
        "lpips_loss_ramp_steps": LOSS_RAMP_STEPS,
        "geometry_ramp_steps": GEOMETRY_RAMP_STEPS,
        "jpeg_ramp_steps": JPEG_RAMP_STEPS, "lpips_weight": 1.5,
        "secret_loss_scale": 1.5, "secret_loss_ramp_steps": 1,
        "yuv_scales": YUV_SCALES, "edge_gain": 10.0,
        "edge_delay_steps": EDGE_DELAY_STEPS, "edge_ramp_steps": EDGE_RAMP_STEPS,
        "gan_weight": 0.5, "gan_ramp_steps": LOSS_RAMP_STEPS,
        "use_gan": True, "use_jpeg": True, "border": "white",
        "dual_branch": DUAL_BRANCH, "crop_scale_min": crop_scale_min,
        "crop_scale_max": crop_scale_max, "adaptive_image_loss": ADAPTIVE_IMAGE_LOSS,
        "adaptive_msg_threshold": ADAPTIVE_MSG_THRESHOLD,
        "adaptive_ber_threshold": ADAPTIVE_BER_THRESHOLD,
        "adaptive_patience": ADAPTIVE_PATIENCE,
        "adaptive_ramp_steps": ADAPTIVE_RAMP_STEPS,
        "validation_samples": VALIDATION_SAMPLES,
    }


def select_resume(output: Path, explicit: Optional[Path], no_resume: bool):
    if no_resume:
        print("Resume checkpoint: disabled")
        return None, None
    checkpoint = explicit or output.with_name(output.stem + ".last" + output.suffix)
    if not checkpoint.is_file():
        print(f"Resume checkpoint: none found at {checkpoint}")
        return None, None
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    arch = (MESSAGE_BITS, WIDTH, True, 0.1, IMAGE_SIZE)
    saved_arch = (state.get("message_bits"), state.get("width"), state.get("use_stn"),
                  state.get("residual_scale"), state.get("image_size"))
    saved_config = state.get("training_state", {}).get("train_config", {})
    if state.get("format") != "markprobe-core-v6-robust-training":
        reason = "checkpoint format is not resumable"
    elif saved_arch != arch:
        reason = f"architecture mismatch: expected {arch}, found {saved_arch}"
    elif not saved_config:
        reason = "checkpoint has no saved training configuration"
    else:
        print(f"Compatible resume checkpoint: {checkpoint.resolve()}")
        print("Reusing saved training configuration from checkpoint")
        return checkpoint, saved_config
    if explicit:
        raise ValueError(f"Cannot resume {checkpoint}: {reason}. Use matching settings or --no-resume.")
    print(f"Found checkpoint but {reason}; starting a fresh run (checkpoint is left untouched).")
    return None, None


def training_config_from_checkpoint(saved_config, crop_scale_min=None, crop_scale_max=None):
    """Reuse checkpoint settings, rejecting explicit crop changes during resume."""
    config = expected_train_config()
    if saved_config is not None:
        missing = sorted(set(config) - set(saved_config))
        if missing:
            raise ValueError(
                "Resume checkpoint has an incomplete training configuration; missing: {}. "
                "Use --finetune-from to load its weights with a new configuration."
                .format(", ".join(missing))
            )
        config.update(saved_config)
    for name, requested in (("crop_scale_min", crop_scale_min),
                            ("crop_scale_max", crop_scale_max)):
        if requested is not None and saved_config is not None and requested != config[name]:
            raise ValueError(
                "--{}={} differs from the resumed checkpoint value {}. "
                "Resume must keep its saved configuration; use --finetune-from "
                "to intentionally change training settings."
                .format(name.replace("_", "-"), requested, config[name])
            )
        if requested is not None and saved_config is None:
            config[name] = requested
    config["yuv_scales"] = tuple(config["yuv_scales"])
    return config


def select_finetune(checkpoint: Optional[Path]):
    if checkpoint is None:
        return None
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Fine-tune checkpoint not found: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    arch = (MESSAGE_BITS, WIDTH, True, 0.1, IMAGE_SIZE)
    saved_arch = (state.get("message_bits"), state.get("width"), state.get("use_stn"),
                  state.get("residual_scale"), state.get("image_size"))
    if state.get("format") != "markprobe-core-v6-robust-training":
        raise ValueError(f"Cannot fine-tune from {checkpoint}: checkpoint format is not supported")
    if saved_arch != arch:
        raise ValueError(f"Cannot fine-tune from {checkpoint}: architecture mismatch; expected {arch}, found {saved_arch}")
    print(f"Fine-tune source: {checkpoint.resolve()} (weights only; optimizer will be reset)")
    return checkpoint


def main():
    args = parse_args()
    max_steps = None if args.no_max_steps else args.max_steps
    if max_steps is not None and max_steps < 1:
        raise ValueError("--max-steps must be positive; use --no-max-steps to disable it")
    images = prepare_data(args.zip, args.data)
    print(f"Training image limit: {args.limit:,}" if args.limit else "Training image limit: all images")

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"Requested {device}, but CUDA is unavailable")
    if device.type == "cuda":
        print(f"GPU training device: {torch.cuda.get_device_name(device)} ({device})")
    else:
        print(f"Training device: {device} (full COCO training will be slow)")

    if args.no_resume and args.finetune_from is not None:
        raise ValueError("--no-resume cannot be combined with --finetune-from")
    resume, saved_config = select_resume(
        args.output, args.resume,
        args.no_resume or args.finetune_from is not None,
    )
    training_config = training_config_from_checkpoint(
        saved_config if resume is not None else None,
        args.crop_scale_min,
        args.crop_scale_max,
    )
    finetune_from = select_finetune(args.finetune_from)
    if finetune_from is not None:
        source = finetune_from.resolve()
        output = args.output.resolve()
        latest = args.output.with_name(args.output.stem + ".last" + args.output.suffix).resolve()
        if source in {output, latest}:
            raise ValueError(
                "Fine-tuning would overwrite its source checkpoint. Pass a new --output path, "
                "for example runs/markprobe_core_finetune.pt."
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    print("Stage 2/2: training MarkProbe-Core")
    print(f"Training config: image_size={IMAGE_SIZE}, width={WIDTH}, message_bits={MESSAGE_BITS}, "
        f"STN=True, dual_branch={training_config['dual_branch']}, "
        f"crop={training_config['crop_scale_min']:.2f}-{training_config['crop_scale_max']:.2f}, epochs={args.epochs}, "
          f"max_steps={max_steps}, batch_size={training_config['batch_size']}")
    print(f"Best checkpoint: {args.output.resolve()}")
    print(f"Metrics log: {args.metrics.resolve()}")
    print(f"Curve data: {args.curves.resolve()}")
    print(f"Epoch inference artifacts: {args.epoch_export_dir.resolve()} (samples={args.epoch_export_samples})")

    train_model(
        args.data, args.output, epochs=args.epochs, max_steps=max_steps,
        limit=args.limit, device=device, resume=resume,
        finetune_from=finetune_from,
        image_size=IMAGE_SIZE, message_bits=MESSAGE_BITS, width=WIDTH,
        use_stn=True, residual_scale=0.1,
        batch_size=training_config["batch_size"],
        lr=training_config["lr"], seed=training_config["seed"],
        val_fraction=training_config["val_fraction"],
        no_im_loss_steps=training_config["no_im_loss_steps"],
        image_weight=training_config["l2_loss_scale"],
        l2_loss_ramp_steps=training_config["l2_loss_ramp_steps"],
        lpips_loss_ramp_steps=training_config["lpips_loss_ramp_steps"],
        geometry_ramp_steps=training_config["geometry_ramp_steps"],
        jpeg_ramp_steps=training_config["jpeg_ramp_steps"],
        edge_gain=training_config["edge_gain"],
        edge_delay_steps=training_config["edge_delay_steps"],
        edge_ramp_steps=training_config["edge_ramp_steps"],
        lpips_weight=training_config["lpips_weight"],
        yuv_scales=training_config["yuv_scales"],
        secret_loss_scale=training_config["secret_loss_scale"],
        secret_loss_ramp_steps=training_config["secret_loss_ramp_steps"],
        gan_weight=training_config["gan_weight"],
        gan_ramp_steps=training_config["gan_ramp_steps"],
        use_gan=training_config["use_gan"],
        use_jpeg=training_config["use_jpeg"],
        border=training_config["border"],
        dual_branch=training_config["dual_branch"],
        crop_scale_min=training_config["crop_scale_min"],
        crop_scale_max=training_config["crop_scale_max"],
        adaptive_image_loss=training_config["adaptive_image_loss"],
        adaptive_msg_threshold=training_config["adaptive_msg_threshold"],
        adaptive_ber_threshold=training_config["adaptive_ber_threshold"],
        adaptive_patience=training_config["adaptive_patience"],
        adaptive_ramp_steps=training_config["adaptive_ramp_steps"],
        validation_samples=training_config["validation_samples"], gradient_log_interval=100,
        metrics_path=args.metrics, curves_path=args.curves,
        epoch_export_dir=args.epoch_export_dir,
        epoch_export_samples=args.epoch_export_samples,
    )
    latest = args.output.with_name(args.output.stem + ".last" + args.output.suffix)
    print(f"Training complete; best checkpoint: {args.output.resolve()}")
    print(f"Latest resumable checkpoint: {latest.resolve()}")


if __name__ == "__main__":
    main()
