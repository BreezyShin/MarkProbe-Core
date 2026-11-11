# MarkProbe Core

An isolated PyTorch implementation of Mark-Probe's image/message encoder and robust bit decoder, ported to PyTorch with StegaStamp's released encoder/decoder topology: fixed 32/64/128/256 U-Net channels, dense 50x50x3 secret map, four encoder skip connections, affine STN, and seven-layer convolutional decoder. The training and inference pipeline is an independent PyTorch port aligned to StegaStamp's released training flow. The workflow remains `cover + random bits -> subtle residual -> transformed image -> bit logits`.

## Environment

Python 3.9+ and PyTorch 2.x are recommended. Install a PyTorch wheel suitable for the target machine first (CPU or the appropriate CUDA build), then:

```bash
python -m pip install -r requirements.txt
```

## Train

Put RGB/JPEG/PNG images under a directory (nested directories are searched). Images are resized to a square and each training step samples independent payload bits.

```bash
python -m markprobe_core.cli train --data /path/to/images --output runs/markprobe.pt --epochs 20 --image-size 400 --batch-size 4
```

With validation enabled, training writes the best validation checkpoint to the requested output path and the latest resumable state beside it with `.last` before the extension. With `--val-fraction 0`, it writes the latest checkpoint to both paths. The reference-aligned defaults follow `StegaStamp/scripts/base.sh`: a 1,500-step secret-only warm-up; 10,000-step perspective ramp; 15,000-step YUV/LPIPS/adversarial ramps; weighted YUV `(1, 100, 100)`; loss weights `secret=1.5`, `YUV=2`, `LPIPS=1.5`, `GAN=0.5`; JPEG quality 50; and a delayed/ramped edge penalty. Each reference-style epoch performs `floor(N / batch_size) + 1` full batches sampled with replacement. `--epochs` is the total target epoch count, so a run stopped after epoch 5 can be continued to epoch 20 with:

```bash
python -m markprobe_core.cli train --data /path/to/images \
  --output runs/markprobe.pt --epochs 20 --resume runs/markprobe.last.pt
```

The resumable checkpoint includes model weights, architecture metadata, both generator Adam states, the training-only discriminator and RMSProp state, global step, best validation loss, and Python/NumPy/PyTorch RNG states. Keep architecture and training objectives unchanged when resuming. Validation reports clean-PNG and transformed-image bit accuracy. Training uses the reference cover-warp/embed/unwarp geometry, reference border modes, and camera transform (blur/noise/brightness/hue/contrast/saturation/differentiable JPEG). TensorFlow's sampler/convolution kernels are ported to PyTorch equivalents, so floating-point outputs are not bit-identical.

When changing the crop range, loss weights, ramps, edge penalty, or other training settings after a completed run, use weight-only fine-tuning instead of `--resume`. It loads the model at the selected checkpoint, resets optimizer/discriminator state and step-based ramps, and writes to a new output path:

```bash
python -m markprobe_core.cli train --data /path/to/images \
  --output runs/markprobe_finetune.pt --epochs 10 \
  --finetune-from runs/markprobe.last.pt \
  --crop-scale-min 0.80 --crop-scale-max 0.95
```

The architecture must still match (`message_bits`, width, image size, STN, and residual scale). Keep the original checkpoint as a baseline; the fine-tuned run starts its log at epoch 1 even though its initial weights came from a later checkpoint.

The `notebooks/quickstart.ipynb` notebook extracts the configured COCO `train2014.zip`, trains, displays progress, and evaluates PNG plus JPEG quality-75 inference. Adjust dataset paths for your Linux machine. Full mode uses all data, 96 message bits, batch 4, and a 140,000-step target. Its default training objective follows `StegaStamp/scripts/base.sh`: one message branch, secret-only warm-up, then fixed step-based ramps for YUV, LPIPS, GAN, and perspective strength. A 10% held-out split is used for diagnostics and checkpoint selection, not to gate training losses. `FAST_SMOKE_TEST=True` uses a short schedule and 512-image limit; its metrics are not quality training. The original repo's default data path points at MIRFLICKR; this notebook uses COCO2014, so the image distribution differs.

Run the equivalent dataset-preparation and training stages without Jupyter using `python scripts/train_coco.py`. It uses the notebook's full-mode settings (including a 140,000-step ceiling) with a 9-epoch target and automatically resumes a compatible `.last.pt` checkpoint. Override paths with `--zip`, `--data`, `--output`, and `--metrics`; use `--no-resume` to start fresh. To run all 9 complete epochs without the notebook's step ceiling, pass `--no-max-steps`.

## Encode and decode

Use exactly `message_bits` binary values (96 by default). New training presets consistently use 96 bits, an 8-bit-aligned size. `text_to_bits` and `bits_to_text` provide a simple UTF-8 convenience format with a one-byte length prefix. With the default 96-bit message, up to 11 UTF-8 bytes fit. This format has no BCH/error-correction coding. Non-default bit lengths remain supported for compatibility, but checkpoints can only be resumed with the exact bit length used to create them.

```bash
python -m markprobe_core.cli encode --checkpoint runs/markprobe.pt --image cover.jpg --text "hello" --output encoded.png
python -m markprobe_core.cli decode --checkpoint runs/markprobe.pt --image encoded.png
```

Enable the optional crop branch and adaptive image-loss gate with `--dual-branch --adaptive-image-loss`. The gate needs a held-out split; adjust its size with `--val-fraction 0.1`. Use `--metrics runs/metrics.jsonl` to choose a diagnostics path.

PNG is the intended deployment format. JPEG robustness must be evaluated at the intended quality and processing pipeline; quality-75 is only a notebook diagnostic sample.

## Design and compatibility

- Encoder: the released StegaStamp dense 7500-value secret projection (50x50x3), nearest-neighbour x8 expansion, 32/32/64/128/256 down path, and four skip-connected up stages are retained. The released source computes `conv10` but taps `conv9` for the residual; this port preserves that graph. The residual head is linear and unscaled (no `tanh`). The canonical input size is 400x400, matching the released model and preserving the fixed 50x50 secret map followed by exact 8x nearest-neighbour expansion. Smaller sizes are useful only for smoke tests and resize that map to the cover size.
- Decoder: StegaStamp’s 32/64/128 localization CNN, identity-initialized affine STN, 32/32/64/64/64/128/128 decoding CNN, and 512-unit dense head are retained. `use_stn=False` remains available only for backwards-compatible experiments; the reference/default path keeps STN enabled.
- Training baseline: single-branch StegaStamp message BCE on transformed marked images; secret-only warm-up; then weighted YUV and LPIPS losses plus the reference training-only discriminator. The generator uses Adam, the discriminator uses RMSProp with weight clipping, and the YUV edge penalty is delayed/ramped. Optional Mark-Probe extensions add a second differentiable crop+camera message BCE branch and validation-gated ramping of image/GAN losses; model layers and activations remain unchanged. The notebook and COCO script default to the StegaStamp-style single branch and fixed schedule; the API/CLI extensions remain opt-in.
- Inference encodes to 8-bit RGB, matching the PNG representation used in the notebook and preventing float-only evaluation from hiding quantization failures.
- Training checkpoint format is `markprobe-core-v6-robust-training`. v3-v5 weights remain loadable for inference because the encoder/decoder architecture is unchanged, but old optimizer states are not resumed with the updated training objective. Start a fresh v6 run; older weights remain usable for inference.
- Remaining differences from the released training implementation: (1) TensorFlow geometry/ops, convolutions, and floating-point kernels are PyTorch ports, not bit-identical; (2) PyTorch `lpips` is used instead of the repository's TensorFlow LPIPS submodule; (3) this project standardizes new training defaults on 96 payload bits rather than StegaStamp's 100; (4) COCO2014 replaces MIRFLICKR; (5) held-out evaluation/checkpoint engineering is added; and (6) notebook dual-branch/adaptive-loss options intentionally extend the reference objective. Exact parity is not claimed.
- Validation logs include BER/BCE, confidence, and accuracy for clean PNG, camera/photo transform, crop-only, crop+photo, and JPEG75, plus clean/geometry image MSE, PSNR, SSIM, LPIPS, MAE, and 95th-percentile absolute pixel delta. PSNR is pixel-MSE based and should be read together with SSIM and LPIPS; lower MAE, p95 delta, and LPIPS are better. Training diagnostics also record raw YUV/LPIPS/GAN losses and their effective weighted contributions. Encoder/decoder gradient norms are sampled periodically; non-finite gradients stop training immediately. JSONL metrics default beside the checkpoint as `<name>.metrics.jsonl`. The adaptive gate requires both message BCE and BER criteria to pass for a configured number of validation epochs before it begins the image-loss ramp. This is an experimental curriculum, not a validated convergence guarantee. RGB/HSV/DISTS ablation losses and BCH/ECC remain unimplemented.

## Checks

```bash
python -m unittest discover -s tests -v
```

