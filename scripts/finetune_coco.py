"""Fine-tune a MarkProbe checkpoint on an already prepared image dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT))

from markprobe_core import DEFAULT_FINETUNE_SETTINGS, run_finetune  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True,
                        help="Source v6 checkpoint; loaded weights-only")
    parser.add_argument("--data", type=Path, required=True,
                        help="Prepared image directory used for fine-tuning")
    parser.add_argument("--output", type=Path, required=True,
                        help="New best-checkpoint path; must not overwrite the source")
    parser.add_argument("--config", type=Path,
                        help="Optional JSON object of train_model keyword overrides")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None,
                        help="Optional image cap for a short diagnostic run")
    parser.add_argument("--device", default=None,
                        help="Training device such as cuda:0 or cpu")
    parser.add_argument("--crop-scale-min", type=float, default=None)
    parser.add_argument("--crop-scale-max", type=float, default=None)
    parser.add_argument("--no-dual-branch", action="store_true")
    parser.add_argument("--no-gan", action="store_true")
    return parser.parse_args()


def load_settings(config_path: Path | None) -> dict:
    settings = dict(DEFAULT_FINETUNE_SETTINGS)
    if config_path is not None:
        with config_path.open("r", encoding="utf-8") as config_file:
            overrides = json.load(config_file)
        if not isinstance(overrides, dict):
            raise ValueError("Fine-tune config must be a JSON object")
        settings.update(overrides)
    return settings


def main():
    args = parse_args()
    settings = load_settings(args.config)
    cli_overrides = {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "max_steps": args.max_steps,
        "crop_scale_min": args.crop_scale_min,
        "crop_scale_max": args.crop_scale_max,
    }
    settings.update({key: value for key, value in cli_overrides.items() if value is not None})
    if args.no_dual_branch:
        settings["dual_branch"] = False
    if args.no_gan:
        settings["use_gan"] = False

    print("Fine-tuning configuration:")
    print(json.dumps(settings, indent=2, default=str))
    print("Source checkpoint: {}".format(args.checkpoint.resolve()))
    print("Output checkpoint: {}".format(args.output.resolve()))
    run_finetune(
        args.data,
        args.checkpoint,
        args.output,
        settings=settings,
        device=args.device,
        limit=args.limit,
    )
    latest = args.output.with_name(args.output.stem + ".last" + args.output.suffix)
    print("Fine-tuning complete. Best checkpoint: {}".format(args.output.resolve()))
    print("Latest resumable checkpoint: {}".format(latest.resolve()))


if __name__ == "__main__":
    main()
