import argparse
import json

from PIL import Image

from .engine import decode_image, encode_image, load_checkpoint, train_model
from .payload import text_to_bits, bits_to_text


def main():
    parser = argparse.ArgumentParser(prog="markprobe-core")
    commands = parser.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train")
    train.add_argument("--data", required=True)
    train.add_argument("--output", default="runs/markprobe.pt")
    train.add_argument("--epochs", type=int, default=10)
    train.add_argument("--max-steps", type=int, default=None,
                       help="Optional total optimizer-step target, useful to match a reference step-based run")
    train.add_argument("--image-size", type=int, default=400)
    train.add_argument("--batch-size", type=int, default=4)
    train.add_argument("--val-fraction", type=float, default=0.1, help="Held-out dataset fraction (required by adaptive loss gating)")
    train.add_argument("--message-bits", type=int, default=96)
    train.add_argument("--width", type=int, default=32)
    train.add_argument("--no-stn", action="store_true", help="Disable the decoder spatial transformer")
    train.add_argument("--residual-scale", type=float, default=0.1)
    train.add_argument("--device", default=None)
    train.add_argument("--resume", default=None,
                       help="Resume training from a checkpoint; --epochs is the total target epoch count")
    train.add_argument("--limit", type=int, default=None, help="Optional image cap for a quick pipeline smoke test")
    train.add_argument("--no-lpips", action="store_true", help="Disable the StegaStamp LPIPS image objective")
    train.add_argument("--no-gan", action="store_true", help="Disable the training-only discriminator objective")
    train.add_argument("--no-jpeg", action="store_true", help="Disable differentiable JPEG in the simulated camera pipeline")
    dual_branch = train.add_mutually_exclusive_group()
    dual_branch.add_argument("--dual-branch", dest="dual_branch", action="store_true",
                             help="Enable the independent random crop+camera message BCE branch (default)")
    dual_branch.add_argument("--no-dual-branch", dest="dual_branch", action="store_false",
                             help="Disable the crop+camera message branch")
    train.set_defaults(dual_branch=True)
    train.add_argument("--crop-scale-min", type=float, default=0.65)
    train.add_argument("--crop-scale-max", type=float, default=0.90)
    train.add_argument("--adaptive-image-loss", action="store_true", help="Gate/ramp YUV, LPIPS and GAN losses on held-out BER/BCE")
    train.add_argument("--adaptive-msg-threshold", type=float, default=0.30)
    train.add_argument("--adaptive-ber-threshold", type=float, default=0.25)
    train.add_argument("--adaptive-patience", type=int, default=2)
    train.add_argument("--adaptive-ramp-steps", type=int, default=15000)
    train.add_argument("--validation-samples", type=int, default=256)
    train.add_argument("--metrics", default=None, help="JSONL epoch diagnostics output (defaults beside checkpoint)")
    train.add_argument("--curves", default=None, help="Column-oriented JSON curve data output")
    train.add_argument("--epoch-export-dir", default=None,
                       help="Directory for per-epoch cover/encoded/decode artifacts")
    train.add_argument("--epoch-export-samples", type=int, default=4,
                       help="Validation samples exported after each epoch; 0 disables export")
    train.add_argument("--no-image-loss-steps", type=int, default=1500,
                       help="Secret-only warm-up steps before image/perceptual/adversarial losses")
    enc = commands.add_parser("encode")
    enc.add_argument("--checkpoint", required=True)
    enc.add_argument("--image", required=True)
    payload = enc.add_mutually_exclusive_group(required=True)
    payload.add_argument("--bits", help="JSON list of 0/1 values")
    payload.add_argument("--text", help="UTF-8 text payload (simple length prefix, no ECC)")
    enc.add_argument("--output", required=True)
    enc.add_argument("--device", default=None)
    dec = commands.add_parser("decode")
    dec.add_argument("--checkpoint", required=True)
    dec.add_argument("--image", required=True)
    dec.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.command == "train":
        train_model(args.data, args.output, epochs=args.epochs, image_size=args.image_size,
                    batch_size=args.batch_size, val_fraction=args.val_fraction, message_bits=args.message_bits,
                    width=args.width, device=args.device, resume=args.resume, limit=args.limit,
                    max_steps=args.max_steps,
                    use_stn=not args.no_stn, residual_scale=args.residual_scale,
                    no_im_loss_steps=args.no_image_loss_steps,
                    lpips_weight=0.0 if args.no_lpips else 1.0,
                    use_gan=not args.no_gan, use_jpeg=not args.no_jpeg,
                    dual_branch=args.dual_branch,
                    crop_scale_min=args.crop_scale_min, crop_scale_max=args.crop_scale_max,
                    adaptive_image_loss=args.adaptive_image_loss,
                    adaptive_msg_threshold=args.adaptive_msg_threshold,
                    adaptive_ber_threshold=args.adaptive_ber_threshold,
                    adaptive_patience=args.adaptive_patience,
                    adaptive_ramp_steps=args.adaptive_ramp_steps,
                    validation_samples=args.validation_samples,
                    metrics_path=args.metrics, curves_path=args.curves,
                    epoch_export_dir=args.epoch_export_dir,
                    epoch_export_samples=args.epoch_export_samples)
    elif args.command == "encode":
        model, device = load_checkpoint(args.checkpoint, args.device)
        bits = json.loads(args.bits) if args.bits else text_to_bits(args.text, model.message_bits)
        encoded = encode_image(model, Image.open(args.image), bits, device)
        encoded.save(args.output)
    else:
        model, device = load_checkpoint(args.checkpoint, args.device)
        bits, probabilities = decode_image(model, args.image, device)
        result = {"bits": bits, "probabilities": probabilities}
        try:
            result["text"] = bits_to_text(bits)
        except (ValueError, UnicodeDecodeError):
            pass
        print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
